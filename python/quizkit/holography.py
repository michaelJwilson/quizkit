import datetime
import logging
from dataclasses import dataclass

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax

from quizkit.readers import read_hdf5

logger = logging.getLogger(__name__)


@dataclass
class SolverConfig:
    method: str = "GS"  # "GS" or "GD"
    maxiter: int = 30

    # GS-specific parameters
    smooth_phase: bool = False  # Apply Gaussian smoothing to phase at each step
    smooth_sigma: float = 2.0  # Sigma in pixels

    # GD-specific parameters
    learning_rate: float = 0.1
    lambda_uniformity: float = 0.0


def get_gaussian_blur_otf(N, sigma):
    """Generates an Optical Transfer Function for periodic Gaussian smoothing."""
    x = jnp.arange(-N // 2, N // 2)
    X, Y = jnp.meshgrid(x, x)
    kernel = jnp.exp(-(X**2 + Y**2) / (2 * sigma**2))
    kernel = kernel / jnp.sum(kernel)
    return jnp.fft.fft2(jnp.fft.ifftshift(kernel))


def propagate_ff(complex_near):
    """Fraunhofer propagation (Near-field to Far-field)."""
    return jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.ifftshift(complex_near), norm="ortho"))


def propagate_nf(complex_far):
    """Inverse Fraunhofer propagation (Far-field to Near-field)."""
    return jnp.fft.fftshift(jnp.fft.ifft2(jnp.fft.ifftshift(complex_far), norm="ortho"))


def run_gs(source_amp, target_amp, initial_phase, config: SolverConfig):
    """JAX-compiled Gerchberg-Saxton with optional complex-plane phase smoothing."""
    N = source_amp.shape[0]
    blur_otf = get_gaussian_blur_otf(N, config.smooth_sigma)

    def gs_step(phase, _):
        # 1. Forward propagate to Far-field
        complex_nf = source_amp * jnp.exp(1j * phase)
        complex_ff = propagate_ff(complex_nf)

        # 2. Replace amplitude with Target constraint, keep phase
        ff_phase = jnp.angle(complex_ff)
        constrained_ff = target_amp * jnp.exp(1j * ff_phase)

        # 3. Backward propagate to Near-field
        complex_nf_new = propagate_nf(constrained_ff)
        new_phase = jnp.angle(complex_nf_new)

        # 4. Optional Periodic Phase Smoothing
        if config.smooth_phase:
            complex_phase = jnp.exp(1j * new_phase)
            blurred_complex = jnp.fft.ifft2(jnp.fft.fft2(complex_phase) * blur_otf)
            new_phase = jnp.angle(blurred_complex)

        return new_phase, None

    # Use JAX scan for lightning-fast GPU execution without unrolling loops in Python
    final_phase, _ = jax.lax.scan(gs_step, initial_phase, jnp.arange(config.maxiter))

    # Calculate final physical intensity for HDF5 output
    final_complex_ff = propagate_ff(source_amp * jnp.exp(1j * final_phase))
    final_intensity = jnp.abs(final_complex_ff) ** 2

    return final_phase, final_intensity


# ==========================================
# 3. Gradient Descent (Optax) Implementation
# ==========================================


def run_gd(source_amp, target_amp, initial_phase, config: SolverConfig):
    """Optax-based optimizer targeting the far-field intensity."""
    target_intensity = target_amp**2
    optimizer = optax.adam(learning_rate=config.learning_rate)

    def loss_fn(phase):
        complex_nf = source_amp * jnp.exp(1j * phase)
        complex_ff = propagate_ff(complex_nf)
        inferred_intensity = jnp.abs(complex_ff) ** 2

        # Mean Squared Error against target intensity
        loss_mse = jnp.mean((inferred_intensity - target_intensity) ** 2)

        # Optional uniformity regularization
        is_active = (target_intensity > 0.01 * jnp.max(target_intensity)).astype(
            jnp.float32
        )
        n_active = jnp.sum(is_active) + 1e-8
        mean_active = jnp.sum(inferred_intensity * is_active) / n_active
        variance_active = (
            jnp.sum(is_active * (inferred_intensity - mean_active) ** 2) / n_active
        )

        return loss_mse + (config.lambda_uniformity * variance_active)

    loss_and_grad = jax.value_and_grad(loss_fn)

    @jax.jit
    def gd_step(carry, _):
        phase, opt_state = carry
        loss, grads = loss_and_grad(phase)
        updates, opt_state = optimizer.update(grads, opt_state, phase)
        new_phase = optax.apply_updates(phase, updates)

        # Keep phase bounded [-pi, pi]
        new_phase = jnp.mod(new_phase + jnp.pi, 2 * jnp.pi) - jnp.pi
        return (new_phase, opt_state), loss

    opt_state = optimizer.init(initial_phase)

    (final_phase, _), losses = jax.lax.scan(
        gd_step, (initial_phase, opt_state), jnp.arange(config.maxiter)
    )

    final_complex_ff = propagate_ff(source_amp * jnp.exp(1j * final_phase))
    final_intensity = jnp.abs(final_complex_ff) ** 2

    return final_phase, final_intensity


# ==========================================
# 4. Dispatcher & HDF5 I/O
# ==========================================


def solve_hologram(source_amp, target_amp, initial_phase, config: SolverConfig):
    """Routes the optimization to the chosen backend strategy."""
    if config.method.upper() == "GS":
        return run_gs(source_amp, target_amp, initial_phase, config)
    elif config.method.upper() == "GD":
        return run_gd(source_amp, target_amp, initial_phase, config)
    else:
        raise ValueError(f"Unknown solver method: {config.method}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # TODO HARDCODE
    hdf5_path = "./results/data/exercise_gs_20260727_145258.h5"

    slm_illumination = read_hdf5(
        hdf5_path, group_name="slm", dataset_name="slm_illumination"
    )

    target_intensity, target_meta = read_hdf5(
        hdf5_path, group_name="target", dataset_name="target_intensity"
    )
    target_amp = jnp.sqrt(target_intensity)

    # TODO
    WAVELENGTH = target_meta["wavelength"]

    PIXEL_PITCH = target_meta["pixel_pitch"]
    SLM_SHAPE = target_meta["slm_shape"]

    ARRAY_SHAPE = target_meta["array_shape"]
    ARRAY_PITCH = target_meta["array_pitch"]

    key = jax.random.PRNGKey(42)

    initial_phase = jax.random.uniform(
        key, (SLM_SHAPE[0], SLM_SHAPE[1]), minval=-jnp.pi, maxval=jnp.pi
    )

    config = SolverConfig(
        method="GS",  # {"GS", "GD"}
        maxiter=30,
        smooth_phase=False,  # Toggles Gaussian blur inside the GS loop
        smooth_sigma=1.5,
    )

    logger.info(
        f"Starting {config.method} optimization over {config.maxiter} iterations..."
    )

    final_phase, inferred_intensity = solve_hologram(
        slm_illumination, target_amp, initial_phase, config
    )

    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    hdf5_path = f"./hologram_{config.method}_{timestamp}.h5"

    write_hdf5(
        filepath=hdf5_path,
        data=final_phase,
        group_name="slm",
        dataset_name="slm_phase",
        wavelength=WAVELENGTH,
        pixel_pitch=PIXEL_PITCH,
        maxiter=config.maxiter,
        method=config.method
    )

    write_hdf5(
        filepath=hdf5_path,
        data=slm_illumination,
        group_name="slm",
        dataset_name="slm_illumination",
    )

    write_hdf5(
        filepath=hdf5_path,
        data=target_intensity,
        group_name="target",
        dataset_name="target_intensity",
    )

    write_hdf5(
        filepath=hdf5_path,
        data=inferred_intensity,
        group_name="target",
        dataset_name="inferred_farfield_intensity",
        **stats
    )
    """
