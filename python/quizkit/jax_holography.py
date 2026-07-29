import logging
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import optax
# from aim import Run

# from rich.pretty import pprint
# from quizkit.readers import read_hdf5
from quizkit.configs import SolverConfig
from quizkit.takehome import (
    # plot_phase_retrieval_results,
    # get_trap_zoom,
    HologramExperiment,
)

import atexit
import aim.ext.cleanup

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


# TODO
def compute_step_metrics_jax(inferred_intensity, target_intensity, trap_labels, num_traps):
    flat_intensity = inferred_intensity.ravel()
    flat_labels = trap_labels.ravel()

    trap_powers = jnp.bincount(flat_labels, weights=flat_intensity, length=num_traps + 1)[1:]
    trap_mean = jnp.mean(trap_powers)
    trap_min = jnp.min(trap_powers)
    trap_max = jnp.max(trap_powers)
    trap_cv = jnp.std(trap_powers) / (trap_mean + 1e-12)

    total_power = jnp.sum(flat_intensity)
    sig_power = jnp.sum(trap_powers)
    bg_power = total_power - sig_power

    bg_mask = (flat_labels == 0)
    bg_intensities = jnp.where(bg_mask, flat_intensity, 0.0)
    max_bg = jnp.max(bg_intensities)

    ff_centered = flat_intensity - jnp.mean(flat_intensity)
    target_centered = target_intensity.ravel() - jnp.mean(target_intensity)
    pearson = jnp.sum(ff_centered * target_centered) / (jnp.sqrt(jnp.sum(ff_centered**2) * jnp.sum(target_centered**2)) + 1e-12)

    return {
        "efficiency": sig_power / total_power,
        "stray_light_fraction": bg_power / total_power,
        "pearson": pearson,
        "trap_cv": trap_cv,
        "trap_uniformity_minmax": 1.0 - ((trap_max - trap_min) / (trap_max + trap_min + 1e-12)),
        "ghost_to_dimmest_ratio": max_bg / (trap_min + 1e-12),
    }

class JaxHologramBackend:
    def __init__(self, experiment: HologramExperiment, config: SolverConfig):
        self.exp = experiment
        self.config = config

        self.source_amp = jnp.array(self.exp.slm_illumination)
        self.target_amp = jnp.sqrt(jnp.array(self.exp.target_intensity))
        self.trap_labels = jnp.array(self.exp.trap_labels)
        self.num_traps = self.exp.num_traps

        rng = np.random.default_rng(self.config.random_seed)
        self.initial_phase = jnp.array(rng.uniform(-np.pi, np.pi, size=self.exp.run_config.slm_shape))

        self.final_phase = None
        self.final_intensity = None
        self.history = {}

    def optimize(self):
        if self.config.method.upper() == "GS":
            self.final_phase, self.final_intensity, self.history = self.__run_gs()
        elif self.config.method.upper() == "GD":
            self.final_phase, self.final_intensity, self.history = self.__run_gd()
        else:
            raise ValueError(f"JAX backend does not support method: {self.config.method}")

    # TODO stop grad tracking; in-place updates; FFT(W) plan; for GS.
    def __run_gs(self):
        logger.info(f"Solving for Gerchberg-Saxton with {self.config.maxiter} iterations.")

        source_amp_native = jnp.fft.ifftshift(self.source_amp)
        target_amp_native = jnp.fft.ifftshift(self.target_amp)
        initial_phase_native = jnp.fft.ifftshift(self.initial_phase)
        blur_otf = get_gaussian_blur_otf(self.source_amp.shape, self.config.smooth_sigma)

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
            metrics = compute_step_metrics_jax(inferred_intensity, target_amp_native**2, self.trap_labels, self.num_traps)
            
            return new_phase, metrics

        final_phase_native, history = jax.lax.scan(gs_step, initial_phase_native, jnp.arange(self.config.maxiter))

        final_complex_ff_native = propagate_ff_native(source_amp_native * jnp.exp(1j * final_phase_native))
        
        final_phase = jnp.mod(jnp.fft.fftshift(final_phase_native) + jnp.pi, 2 * jnp.pi) - jnp.pi
        final_intensity = jnp.fft.fftshift(jnp.abs(final_complex_ff_native) ** 2)

        return np.asarray(final_phase), np.asarray(final_intensity), history

    def __run_gd(self, smooth_lambda=0.0):
        source_amp_native = jnp.fft.ifftshift(self.source_amp)
        target_amp_native = jnp.fft.ifftshift(self.target_amp)
        initial_phase_native = jnp.fft.ifftshift(self.initial_phase)

        target_intensity_native = target_amp_native**2

        # TODO bail out on loss or parameter convergence
        optimizer = optax.adam(learning_rate=self.config.learning_rate)

        blur_otf = get_gaussian_blur_otf(self.source_amp.shape, self.config.smooth_sigma)

        def loss_fn(phase):
            complex_phasor = jnp.exp(1j * phase)
            complex_nf = source_amp_native * complex_phasor
            complex_ff = propagate_ff_native(complex_nf)
            inferred_intensity = jnp.abs(complex_ff) ** 2

            norm_inferred = inferred_intensity / (jnp.mean(inferred_intensity) + 1e-12)
            norm_target = target_intensity_native / (
                jnp.mean(target_intensity_native) + 1e-12
            )

            diff = norm_inferred - norm_target

            if self.config.loss_norm.upper() == "L1":
                loss_val = jnp.mean(jnp.abs(diff))
            elif self.config.loss_norm.upper() == "L2":
                loss_val = jnp.mean(diff**2)
            else:
                raise ValueError(f"Unsupported loss norm: {self.config.loss_norm}")

            loss_val += smooth_lambda * smooth_phase_regularization(phase)

            return loss_val, inferred_intensity

        loss_and_grad = jax.value_and_grad(loss_fn, has_aux=True)

        @jax.jit
        def gd_step(carry, step_idx):
            phase, opt_state, key = carry

            key, subkey = jax.random.split(key)

            (loss_val, inferred_intensity), grads = loss_and_grad(phase)
            updates, opt_state = optimizer.update(grads, opt_state, phase)

            new_phase = optax.apply_updates(phase, updates)

            epsilon = self.config.initial_epsilon * jnp.exp(-self.config.anneal_rate * step_idx)

            key_phase, key_mask = jax.random.split(subkey, 2)

            random_phases = jax.random.uniform(
                key_phase, phase.shape, minval=-jnp.pi, maxval=jnp.pi
            )

            explore_mask = jax.random.uniform(key_mask, phase.shape) < epsilon
            new_phase = jnp.where(explore_mask, random_phases, new_phase)

            if self.config.smooth_phase:
                complex_phase = jnp.exp(1j * new_phase)
                blurred_complex = jnp.fft.ifft2(blur_otf * jnp.fft.fft2(complex_phase))
                new_phase = jnp.angle(blurred_complex)

            new_phase = jnp.mod(new_phase + jnp.pi, 2 * jnp.pi) - jnp.pi
            metrics = compute_step_metrics_jax(
                inferred_intensity, 
                target_intensity_native, 
                self.trap_labels, 
                self.num_traps
            )
            metrics["loss"] = loss_val

            return (new_phase, opt_state, key), metrics

        opt_state = optimizer.init(initial_phase_native)
        step_key = jax.random.PRNGKey(self.config.random_seed)

        (final_phase_native, _, _), history = jax.lax.scan(
            gd_step, (initial_phase_native, opt_state, step_key), jnp.arange(self.config.maxiter)
        )

        final_complex_ff_native = propagate_ff_native(
            source_amp_native * jnp.exp(1j * final_phase_native)
        )
        final_intensity_native = jnp.abs(final_complex_ff_native) ** 2

        final_phase = jnp.fft.fftshift(final_phase_native)
        final_intensity = jnp.fft.fftshift(final_intensity_native)
        final_phase = jnp.mod(final_phase + jnp.pi, 2 * jnp.pi) - jnp.pi

        return np.asarray(final_phase), np.asarray(final_intensity), history

# launch GUI with: aim up
if __name__ == "__main__":
    """
    logging.basicConfig(level=logging.INFO)

    # TODO HARDCODE
    hdf5_path = "./results/data/exercise_gs_20260727_184334.h5"

    slm_illumination, slm_meta = read_hdf5(
        hdf5_path, group_name="slm", dataset_name="slm_illumination"
    )

    target_intensity, target_meta = read_hdf5(
        hdf5_path, group_name="target", dataset_name="target_intensity"
    )

    WAVELENGTH = target_meta["wavelength"]  # m
    PIXEL_PITCH = target_meta["pixel_pitch"]  # m
    SLM_SHAPE = tuple(target_meta["slm_shape"])  # (height, width) in pixels
    ARRAY_SHAPE = tuple(target_meta["array_shape"])
    ARRAY_PITCH = tuple(target_meta["array_pitch"])
    """
    # assert slm_illumination.max() > 0.0
    # assert target_intensity.max() > 0.0

    """
    slm_illumination = jnp.array(slm_illumination, dtype=jnp.float64)
    target_intensity = jnp.array(target_intensity, dtype=jnp.float64)
    target_amp = jnp.sqrt(target_intensity)

    key = jax.random.PRNGKey(42)
    initial_phase = jax.random.uniform(
        key, SLM_SHAPE, minval=-jnp.pi, maxval=jnp.pi, dtype=jnp.float64
    )

    config = SolverConfig(method="GD", maxiter=200, smooth_phase=True, smooth_sigma=3)
    logger.info(
        f"Starting {config.method} optimization over {config.maxiter} iterations..."
    )

    final_phase, inferred_intensity, history = solve_hologram(
        slm_illumination, target_amp, initial_phase, config
    )
    """
    """
    # TODO aim logging; track metrics and plots.
    run = Run(experiment=f"{config.method}_optimization")
    run["hparams"] = config.__dict__

    for step_idx in range(config.maxiter):
        if "loss" in history:
            run.track(
                history["loss"][step_idx].item(),
                name="Total",
                step=step_idx,
                context={"subset": "Loss"},
            )

        run.track(
            history["efficiency"][step_idx].item(),
            name="Efficiency",
            step=step_idx,
            context={"subset": "Metrics"},
        )
        run.track(
            history["uniformity"][step_idx].item(),
            name="Uniformity",
            step=step_idx,
            context={"subset": "Metrics"},
        )
        run.track(
            history["ghost_trap_ratio"][step_idx].item(),
            name="Ghost_Trap_Ratio",
            step=step_idx,
            context={"subset": "Metrics"},
        )
        run.track(
            history["stray_light_fraction"][step_idx].item(),
            name="Stray_Light",
            step=step_idx,
            context={"subset": "Metrics"},
        )
        run.track(
            history["pearson"][step_idx].item(),
            name="Pearson",
            step=step_idx,
            context={"subset": "Metrics"},
        )

    run.close()
    """
    """
    performance_metrics = compute_performance_metrics(
        inferred_intensity, target_intensity
    )

    performance_metrics = {k: float(v) for k, v in performance_metrics.items()}
    pprint(performance_metrics)

    extent = get_trap_zoom(target_intensity)
    x_min, x_max, y_min, y_max = extent

    plot_phase_retrieval_results(
        "./results/plots/phase_retrieval_results.pdf",
        final_phase,
        inferred_intensity[y_min:y_max, x_min:x_max],
        extent,
    )

    logger.info("Done.")
    """
