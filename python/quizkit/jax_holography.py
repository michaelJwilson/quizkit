import logging
import jax

jax.config.update("jax_enable_x64", True)

import atexit

import aim.ext.cleanup
import jax.numpy as jnp
import numpy as np
import optax
from quizkit.configs import SolverConfig
from quizkit.hologram_experiment import HologramExperiment

# TODO HACK
atexit.unregister(aim.ext.cleanup.AutoClean.cleanup)

logger = logging.getLogger(__name__)


def get_gaussian_blur_otf(shape, sigma):
    H, W = shape
    x = jnp.arange(-W // 2, W - W // 2)
    y = jnp.arange(-H // 2, H - H // 2)
    X, Y = jnp.meshgrid(x, y)
    kernel = jnp.exp(-(X**2 + Y**2) / (2 * sigma**2))
    kernel = kernel / jnp.sum(kernel)
    return jnp.fft.fft2(jnp.fft.ifftshift(kernel))


def propagate_ff_native(complex_near):
    return jnp.fft.fft2(complex_near, norm="ortho")


def propagate_nf_native(complex_far):
    return jnp.fft.ifft2(complex_far, norm="ortho")


def smooth_phase_regularization(phase):
    diff_x = jnp.diff(phase, axis=1)
    diff_y = jnp.diff(phase, axis=0)
    penalty_x = 1.0 - jnp.cos(diff_x)
    penalty_y = 1.0 - jnp.cos(diff_y)
    return jnp.mean(penalty_x) + jnp.mean(penalty_y)


# TODO must mirror compute_performance_metrics in run_phase_retrieval.py
#      for consistency
def compute_step_metrics_jax(
    inferred_intensity, target_intensity, trap_labels, num_traps
):
    flat_intensity = inferred_intensity.ravel()
    flat_labels = trap_labels.ravel()

    trap_powers = jnp.bincount(
        flat_labels, weights=flat_intensity, length=num_traps + 1
    )[1:]

    trap_min = jnp.min(trap_powers)
    trap_max = jnp.max(trap_powers)
    trap_med = jnp.median(trap_powers)
    trap_mean = jnp.mean(trap_powers)
    trap_std = jnp.std(trap_powers)

    total_power = jnp.sum(flat_intensity)
    sig_power = jnp.sum(trap_powers)
    bg_power = total_power - sig_power

    bg_mask = flat_labels == 0
    bg_intensities = jnp.where(bg_mask, flat_intensity, 0.0)
    max_bg = jnp.max(bg_intensities)

    ff_centered = flat_intensity - jnp.mean(flat_intensity)
    target_centered = target_intensity.ravel() - jnp.mean(target_intensity)
    pearson = jnp.sum(ff_centered * target_centered) / (
        jnp.sqrt(jnp.sum(ff_centered**2) * jnp.sum(target_centered**2)) + 1e-12
    )

    uniformity = 1.0 - ((trap_max - trap_min) / (trap_max + trap_min + 1e-12))

    trap_ps = trap_powers / (sig_power + 1e-12)
    entropy = -jnp.sum(trap_ps * jnp.log(trap_ps + 1e-12)) / jnp.log(num_traps + 1e-12)

    ghost_to_trap_med_ratio = max_bg / (trap_med + 1e-12)

    return {
        "uniformity": uniformity,
        "entropy": entropy,
        "efficiency": sig_power / total_power,
        "stray_light_fraction": bg_power / total_power,
        "pearson": pearson,
        "trap_med": trap_med,
        "trap_mean": trap_mean,
        "trap_std": trap_std,
        "trap_min": trap_min,
        "trap_max": trap_max,
        "ghost_to_trap_med_ratio": ghost_to_trap_med_ratio,
    }


class JaxHologramBackend:
    def __init__(self, experiment: HologramExperiment, config: SolverConfig):
        self.exp = experiment
        self.config = config

        self.source_amp = jnp.array(self.exp.slm_illumination)

        raw_target_amp = jnp.sqrt(jnp.array(self.exp.target_intensity))

        source_power = jnp.sum(self.source_amp**2)
        target_power = jnp.sum(raw_target_amp**2)

        scale_factor = jnp.sqrt(source_power / (target_power + 1e-12))
        self.target_amp = raw_target_amp * scale_factor

        self.trap_labels = jnp.array(self.exp.trap_labels)
        self.num_traps = self.exp.num_traps

        rng = np.random.default_rng(self.config.random_seed)
        self.initial_phase = jnp.array(
            rng.uniform(-np.pi, np.pi, size=self.exp.run_config.slm_shape)
        )

        self.final_phase = None
        self.final_intensity = None
        self.history = {}

    def optimize(self):
        method = self.config.method.upper()
        if method == "GS":
            self.final_phase, self.final_intensity, self.history = self.__run_gs()
        elif method == "GD":
            if getattr(self.config, "downsample_factor", 1) > 1:
                self.final_phase, self.final_intensity, self.history = (
                    self.__run_gd_bp_limited(
                        downsample_factor=self.config.downsample_factor
                    )
                )
            else:
                self.final_phase, self.final_intensity, self.history = self.__run_gd()
        elif method == "AA":
            self.final_phase, self.final_intensity, self.history = self.__run_aa()
        elif method == "HIO":
            self.final_phase, self.final_intensity, self.history = self.__run_hio()
        else:
            raise ValueError(f"JAX backend does not support method: {method}")

    def __run_gs(self):
        logger.info(
            f"Solving for Gerchberg-Saxton with {self.config.maxiter} iterations."
        )

        source_amp_native = jnp.fft.ifftshift(self.source_amp)
        target_amp_native = jnp.fft.ifftshift(self.target_amp)
        initial_phase_native = jnp.fft.ifftshift(self.initial_phase)
        blur_otf = get_gaussian_blur_otf(
            self.source_amp.shape, self.config.smooth_sigma
        )

        def gs_step(phase, _):
            complex_nf = source_amp_native * jnp.exp(1j * phase)
            complex_ff = propagate_ff_native(complex_nf)

            ff_phase = jnp.angle(complex_ff)
            constrained_ff = target_amp_native * jnp.exp(1j * ff_phase)

            complex_nf_new = propagate_nf_native(constrained_ff)
            new_phase = jnp.angle(complex_nf_new)

            if self.config.smooth_phase:
                complex_phase = jnp.exp(1j * new_phase)
                blurred_complex = jnp.fft.ifft2(blur_otf * jnp.fft.fft2(complex_phase))
                new_phase = jnp.angle(blurred_complex)

            inferred_intensity = jnp.abs(complex_ff) ** 2
            metrics = compute_step_metrics_jax(
                inferred_intensity,
                target_amp_native**2,
                self.trap_labels,
                self.num_traps,
            )
            return new_phase, metrics

        final_phase_native, history = jax.lax.scan(
            gs_step, initial_phase_native, jnp.arange(self.config.maxiter)
        )

        final_complex_ff_native = propagate_ff_native(
            source_amp_native * jnp.exp(1j * final_phase_native)
        )
        final_phase = (
            jnp.mod(jnp.fft.fftshift(final_phase_native) + jnp.pi, 2 * jnp.pi) - jnp.pi
        )
        final_intensity = jnp.fft.fftshift(jnp.abs(final_complex_ff_native) ** 2)

        return np.asarray(final_phase), np.asarray(final_intensity), history

    def __run_aa(self):
        logger.info(
            f"Solving for Adaptive-Additive (AA) with {self.config.maxiter} iterations."
        )

        source_amp_native = jnp.fft.ifftshift(self.source_amp)
        target_amp_native = jnp.fft.ifftshift(self.target_amp)
        initial_phase_native = jnp.fft.ifftshift(self.initial_phase)
        blur_otf = get_gaussian_blur_otf(
            self.source_amp.shape, self.config.smooth_sigma
        )

        # Hyperparameter fallback if not present in SolverConfig
        alpha = getattr(self.config, "aa_alpha", 0.5)

        def aa_step(carry, _):
            phase, a_comp = carry
            complex_nf = source_amp_native * jnp.exp(1j * phase)
            complex_ff = propagate_ff_native(complex_nf)

            ff_amp = jnp.abs(complex_ff)
            ff_phase = jnp.angle(complex_ff)

            # AA updates the target strictly in the signal regions.
            # Background is suppressed to 0.0.
            a_comp_new = jnp.where(
                target_amp_native > 0,
                a_comp + alpha * (target_amp_native - ff_amp),
                0.0,
            )
            a_comp_new = jnp.maximum(a_comp_new, 0.0)

            constrained_ff = a_comp_new * jnp.exp(1j * ff_phase)
            complex_nf_new = propagate_nf_native(constrained_ff)
            new_phase = jnp.angle(complex_nf_new)

            if self.config.smooth_phase:
                complex_phase = jnp.exp(1j * new_phase)
                blurred_complex = jnp.fft.ifft2(blur_otf * jnp.fft.fft2(complex_phase))
                new_phase = jnp.angle(blurred_complex)

            inferred_intensity = ff_amp**2
            metrics = compute_step_metrics_jax(
                inferred_intensity,
                target_amp_native**2,
                self.trap_labels,
                self.num_traps,
            )
            return (new_phase, a_comp_new), metrics

        # State carry includes both the phase and the running compensated amplitude map
        init_state = (
            initial_phase_native.astype(jnp.float64),
            target_amp_native.astype(jnp.float64),
        )

        (final_phase_native, _), history = jax.lax.scan(
            aa_step, init_state, jnp.arange(self.config.maxiter)
        )

        final_complex_ff_native = propagate_ff_native(
            source_amp_native * jnp.exp(1j * final_phase_native)
        )
        final_phase = (
            jnp.mod(jnp.fft.fftshift(final_phase_native) + jnp.pi, 2 * jnp.pi) - jnp.pi
        )
        final_intensity = jnp.fft.fftshift(jnp.abs(final_complex_ff_native) ** 2)

        return np.asarray(final_phase), np.asarray(final_intensity), history

    def __run_hio(self):
        logger.info(f"Solving for Fienup HIO with {self.config.maxiter} iterations.")

        source_amp_native = jnp.fft.ifftshift(self.source_amp)
        target_amp_native = jnp.fft.ifftshift(self.target_amp)
        initial_phase_native = jnp.fft.ifftshift(self.initial_phase)
        blur_otf = get_gaussian_blur_otf(
            self.source_amp.shape, self.config.smooth_sigma
        )

        # Hyperparameter fallback if not present in SolverConfig
        beta = getattr(self.config, "hio_beta", 0.8)

        def hio_step(carry, _):
            phase, g_prev = carry
            complex_nf = source_amp_native * jnp.exp(1j * phase)
            complex_ff = propagate_ff_native(complex_nf)

            signal_mask = target_amp_native > 0

            # HIO Far-field update rule
            g_new = jnp.where(
                signal_mask,
                target_amp_native
                * jnp.exp(1j * jnp.angle(complex_ff)),  # Project signal constraint
                g_prev - beta * complex_ff,  # Accumulate negative background error
            )

            complex_nf_new = propagate_nf_native(g_new)
            new_phase = jnp.angle(complex_nf_new)

            if self.config.smooth_phase:
                complex_phase = jnp.exp(1j * new_phase)
                blurred_complex = jnp.fft.ifft2(blur_otf * jnp.fft.fft2(complex_phase))
                new_phase = jnp.angle(blurred_complex)

            inferred_intensity = jnp.abs(complex_ff) ** 2
            metrics = compute_step_metrics_jax(
                inferred_intensity,
                target_amp_native**2,
                self.trap_labels,
                self.num_traps,
            )
            return (new_phase, g_new), metrics

        # State carry includes both the phase and the memory of the previous constrained far-field
        init_g_prev = jnp.zeros_like(target_amp_native, dtype=jnp.complex128)
        init_state = (initial_phase_native, init_g_prev)

        (final_phase_native, _), history = jax.lax.scan(
            hio_step, init_state, jnp.arange(self.config.maxiter)
        )

        final_complex_ff_native = propagate_ff_native(
            source_amp_native * jnp.exp(1j * final_phase_native)
        )
        final_phase = (
            jnp.mod(jnp.fft.fftshift(final_phase_native) + jnp.pi, 2 * jnp.pi) - jnp.pi
        )
        final_intensity = jnp.fft.fftshift(jnp.abs(final_complex_ff_native) ** 2)

        return np.asarray(final_phase), np.asarray(final_intensity), history

    def __run_gd(self, smooth_lambda=0.0):
        logger.info(f"Solving for Gradient Descent at native resolution.")
        source_amp_native = jnp.fft.ifftshift(self.source_amp)
        target_amp_native = jnp.fft.ifftshift(self.target_amp)
        initial_phase_native = jnp.fft.ifftshift(self.initial_phase)

        target_intensity_native = target_amp_native**2
        optimizer = optax.adam(learning_rate=self.config.learning_rate)
        blur_otf = get_gaussian_blur_otf(
            self.source_amp.shape, self.config.smooth_sigma
        )

        def loss_fn(phase, normed=False):
            complex_phasor = jnp.exp(1j * phase)

            # IN-LOOP BLUR: Apply physics before propagating
            if self.config.smooth_phase:
                blurred_complex = jnp.fft.ifft2(blur_otf * jnp.fft.fft2(complex_phasor))
                # Re-normalize to physical SLM amplitude
                complex_phasor = blurred_complex / (jnp.abs(blurred_complex) + 1e-12)

            complex_nf = source_amp_native * complex_phasor
            complex_ff = propagate_ff_native(complex_nf)
            forward_intensity = jnp.abs(complex_ff) ** 2

            if normed:
                norm_inferred = forward_intensity / (
                    jnp.mean(forward_intensity) + 1e-12
                )
                norm_target = target_intensity_native / (
                    jnp.mean(target_intensity_native) + 1e-12
                )
                diff = norm_inferred - norm_target
            else:
                diff = forward_intensity - target_intensity_native

            if self.config.loss_norm.upper() == "L1":
                loss_val = jnp.mean(jnp.abs(diff))
            elif self.config.loss_norm.upper() == "L2":
                loss_val = jnp.mean(diff**2)
            else:
                raise ValueError(f"Unsupported loss norm: {self.config.loss_norm}")

            loss_val += smooth_lambda * smooth_phase_regularization(phase)

            # Return the applied phasor so we know the true physical phase
            return loss_val, (forward_intensity, complex_phasor)

        loss_and_grad = jax.value_and_grad(loss_fn, has_aux=True)

        @jax.jit
        def gd_step(carry, step_idx):
            # Pass physical_phase through the carry state to avoid VRAM explosion
            phase, opt_state, key, _ = carry
            key, subkey = jax.random.split(key)

            (loss_val, (inferred_intensity, complex_phasor)), grads = loss_and_grad(
                phase
            )
            updates, opt_state = optimizer.update(grads, opt_state, phase)

            new_phase = optax.apply_updates(phase, updates)

            epsilon = self.config.initial_epsilon * jnp.exp(
                -self.config.anneal_rate * step_idx
            )
            key_phase, key_mask = jax.random.split(subkey, 2)
            random_phases = jax.random.uniform(
                key_phase, phase.shape, minval=-jnp.pi, maxval=jnp.pi
            )
            explore_mask = jax.random.uniform(key_mask, phase.shape) < epsilon
            new_phase = jnp.where(explore_mask, random_phases, new_phase)

            new_phase = jnp.mod(new_phase + jnp.pi, 2 * jnp.pi) - jnp.pi

            metrics = compute_step_metrics_jax(
                inferred_intensity,
                target_intensity_native,
                self.trap_labels,
                self.num_traps,
            )
            metrics["loss"] = loss_val

            return (new_phase, opt_state, key, complex_phasor), metrics

        opt_state = optimizer.init(initial_phase_native)
        step_key = jax.random.PRNGKey(self.config.random_seed)

        # Initialize a dummy phasor for the carry state
        init_physical_phasor = jnp.zeros_like(
            initial_phase_native, dtype=jnp.complex128
        )

        # The scan now only accumulates the metrics in history
        (final_latent_phase, _, _, final_complex_phasor_native), history = jax.lax.scan(
            gd_step,
            (initial_phase_native, opt_state, step_key, init_physical_phasor),
            jnp.arange(self.config.maxiter),
        )

        final_phase_native = jnp.angle(final_complex_phasor_native)
        # TODO
        if self.config.smooth_phase:
             complex_phase = jnp.exp(1j * final_phase_native)
             blurred_complex = jnp.fft.ifft2(blur_otf * jnp.fft.fft2(complex_phase))
        
             final_phase_native = jnp.angle(blurred_complex)

        final_complex_ff_native = propagate_ff_native(
            source_amp_native * jnp.exp(1j * final_phase_native)
        )
        final_intensity_native = jnp.abs(final_complex_ff_native) ** 2

        final_phase = jnp.fft.fftshift(final_phase_native)
        final_intensity = jnp.fft.fftshift(final_intensity_native)
        final_phase = jnp.mod(final_phase + jnp.pi, 2 * jnp.pi) - jnp.pi

        return np.asarray(final_phase), np.asarray(final_intensity), history

    def __run_gd_bp_limited(self, downsample_factor=4, interp_method="lanczos3"):
        logger.info(
            f"Assuming a band-limited slm phase space with ds={downsample_factor}."
        )

        source_amp_native = jnp.fft.ifftshift(self.source_amp)
        target_amp_native = jnp.fft.ifftshift(self.target_amp)
        initial_phase_native = jnp.fft.ifftshift(self.initial_phase)

        target_intensity_native = target_amp_native**2
        full_shape = self.source_amp.shape
        sub_shape = (
            full_shape[0] // downsample_factor,
            full_shape[1] // downsample_factor,
        )

        optimizer = optax.adam(learning_rate=self.config.learning_rate)

        blur_otf = get_gaussian_blur_otf(
            self.source_amp.shape, self.config.smooth_sigma
        )

        initial_complex = jnp.exp(1j * initial_phase_native)
        # Upgrade to lanczos3 for a cleaner frequency cutoff
        real_sub_init = jax.image.resize(
            jnp.real(initial_complex), sub_shape, method=interp_method
        )
        imag_sub_init = jax.image.resize(
            jnp.imag(initial_complex), sub_shape, method=interp_method
        )
        initial_sub_phase = jnp.angle(real_sub_init + 1j * imag_sub_init)

        def loss_fn(sub_phase, normed=False):
            # 1. Band-limited reconstruction (upsampling) via Lanczos
            complex_sub = jnp.exp(1j * sub_phase)
            real_full = jax.image.resize(
                jnp.real(complex_sub), full_shape, method=interp_method
            )
            imag_full = jax.image.resize(
                jnp.imag(complex_sub), full_shape, method=interp_method
            )

            complex_phasor_full = real_full + 1j * imag_full

            # Apply exact Gaussian physics inside the loss gradient calculation
            blurred_complex = jnp.fft.ifft2(
                blur_otf * jnp.fft.fft2(complex_phasor_full)
            )

            complex_phasor_full = blurred_complex / (jnp.abs(blurred_complex) + 1e-12)

            complex_nf = source_amp_native * complex_phasor_full
            complex_ff = propagate_ff_native(complex_nf)
            forward_intensity = jnp.abs(complex_ff) ** 2

            if normed:
                norm_inferred = forward_intensity / (
                    jnp.mean(forward_intensity) + 1e-12
                )
                norm_target = target_intensity_native / (
                    jnp.mean(target_intensity_native) + 1e-12
                )
                diff = norm_inferred - norm_target
            else:
                diff = forward_intensity - target_intensity_native

            if self.config.loss_norm.upper() == "L1":
                loss_val = jnp.mean(jnp.abs(diff))
            elif self.config.loss_norm.upper() == "L2":
                loss_val = jnp.mean(diff**2)
            else:
                raise ValueError(f"Unsupported loss norm: {self.config.loss_norm}")

            return loss_val, (forward_intensity, complex_phasor_full)

        loss_and_grad = jax.value_and_grad(loss_fn, has_aux=True)

        @jax.jit
        def gd_step(carry, step_idx):
            # Pass physical_phase through the carry state to avoid VRAM explosion
            sub_phase, opt_state, key, _ = carry
            key, subkey = jax.random.split(key)

            (loss_val, (inferred_intensity, complex_phasor_full)), grads = (
                loss_and_grad(sub_phase)
            )

            updates, opt_state = optimizer.update(grads, opt_state, sub_phase)
            new_sub_phase = optax.apply_updates(sub_phase, updates)

            epsilon = self.config.initial_epsilon * jnp.exp(
                -self.config.anneal_rate * step_idx
            )

            key_phase, key_mask = jax.random.split(subkey, 2)

            random_phases = jax.random.uniform(
                key_phase, new_sub_phase.shape, minval=-jnp.pi, maxval=jnp.pi
            )

            explore_mask = jax.random.uniform(key_mask, new_sub_phase.shape) < epsilon
            new_sub_phase = jnp.where(explore_mask, random_phases, new_sub_phase)

            new_sub_phase = jnp.mod(new_sub_phase + jnp.pi, 2 * jnp.pi) - jnp.pi

            metrics = compute_step_metrics_jax(
                inferred_intensity,
                target_intensity_native,
                self.trap_labels,
                self.num_traps,
            )
            metrics["loss"] = loss_val

            return (new_sub_phase, opt_state, key, complex_phasor_full), metrics

        opt_state = optimizer.init(initial_sub_phase)
        step_key = jax.random.PRNGKey(self.config.random_seed)

        # Initialize a dummy phasor for the carry state
        init_physical_phasor = jnp.zeros(full_shape, dtype=jnp.complex128)

        # The scan now only accumulates the metrics in history
        (final_sub_phase, _, _, final_complex_phasor_native), history = jax.lax.scan(
            gd_step,
            (initial_sub_phase, opt_state, step_key, init_physical_phasor),
            jnp.arange(self.config.maxiter),
        )

        # Extract the final physically validated phase directly from the last scan carry state
        final_phase_native = jnp.angle(final_complex_phasor_native)

        final_complex_ff_native = propagate_ff_native(
            source_amp_native * jnp.exp(1j * final_phase_native)
        )
        final_intensity_native = jnp.abs(final_complex_ff_native) ** 2

        
        if self.config.smooth_phase:
             complex_phase = jnp.exp(1j * final_phase_native)
             blurred_complex = jnp.fft.ifft2(blur_otf * jnp.fft.fft2(complex_phase))
        
             final_phase_native = jnp.angle(blurred_complex)

        final_phase = jnp.fft.fftshift(final_phase_native)
        final_intensity = jnp.fft.fftshift(final_intensity_native)
        final_phase = jnp.mod(final_phase + jnp.pi, 2 * jnp.pi) - jnp.pi

        return np.asarray(final_phase), np.asarray(final_intensity), history
