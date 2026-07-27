import logging
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

from rich.pretty import pprint
from quizkit.readers import read_hdf5
from quizkit.exercise import compute_performance_metrics

logger = logging.getLogger(__name__)


@dataclass
class SolverConfig:
    method: str = "GS"  # {"GS", "GD"}
    maxiter: int = 30

    smooth_phase: bool = False  # Apply Gaussian smoothing to phase at each step
    smooth_sigma: float = 2.0  # Sigma in pixels

    learning_rate: float = 0.1
    lambda_uniformity: float = 0.0


def get_gaussian_blur_otf(shape, sigma):
    H, W = shape

    x = jnp.arange(-W // 2, W - W // 2)
    y = jnp.arange(-H // 2, H - H // 2)

    X, Y = jnp.meshgrid(x, y)

    kernel = jnp.exp(-(X**2 + Y**2) / (2 * sigma**2))
    kernel = kernel / jnp.sum(kernel)

    # NB (inverse) fftshift rearranges quadrants so the "center pixel" is at [0, 0] for fft, with wrapping.
    #    fft2 computes the 2D Fourier transform.
    return jnp.fft.fft2(jnp.fft.ifftshift(kernel))


def propagate_ff(complex_near):
    # NB norm accounts for the 1/sqrt(N) normalization factor, ensuring energy conservation.
    return jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.ifftshift(complex_near), norm="ortho"))


def propagate_nf(complex_far):
    return jnp.fft.fftshift(jnp.fft.ifft2(jnp.fft.ifftshift(complex_far), norm="ortho"))


def run_gs(source_amp, target_amp, initial_phase, config: SolverConfig):
    slm_shape = source_amp.shape
    blur_otf = get_gaussian_blur_otf(slm_shape, config.smooth_sigma)

    def gs_step(phase, _):
        complex_nf = source_amp * jnp.exp(1j * phase)
        complex_ff = propagate_ff(complex_nf)

        # NB replace amplitude with target constraint, keep current phase.
        ff_phase = jnp.angle(complex_ff)
        constrained_ff = target_amp * jnp.exp(1j * ff_phase)

        complex_nf_new = propagate_nf(constrained_ff)
        new_phase = jnp.angle(complex_nf_new)

        if config.smooth_phase:
            complex_phase = jnp.exp(1j * new_phase)

            # TODO BUG phase must be wrapped for smoothing to work correctly.
            blurred_complex = jnp.fft.ifft2(blur_otf * jnp.fft.fft2(complex_phase))
            new_phase = jnp.angle(blurred_complex)

        return new_phase, None

    # NB aid XLA for efficient compilation.
    final_phase, _ = jax.lax.scan(gs_step, initial_phase, jnp.arange(config.maxiter))

    final_complex_ff = propagate_ff(source_amp * jnp.exp(1j * final_phase))
    final_intensity = jnp.abs(final_complex_ff) ** 2

    return final_phase, final_intensity


def run_gd(source_amp, target_amp, initial_phase, config: SolverConfig):
    target_intensity = target_amp**2
    optimizer = optax.adam(learning_rate=config.learning_rate)

    def loss(phase):
        complex_nf = source_amp * jnp.exp(1j * phase)
        complex_ff = propagate_ff(complex_nf)
        inferred_intensity = jnp.abs(complex_ff) ** 2

        loss_mse = jnp.mean((inferred_intensity - target_intensity) ** 2)

        return loss_mse

    loss_and_grad = jax.value_and_grad(loss)

    @jax.jit
    def gd_step(carry, _):
        phase, opt_state = carry
        loss, grads = loss_and_grad(phase)
        updates, opt_state = optimizer.update(grads, opt_state, phase)
        new_phase = optax.apply_updates(phase, updates)

        # NB boudn phase
        new_phase = jnp.mod(new_phase + jnp.pi, 2 * jnp.pi) - jnp.pi
        return (new_phase, opt_state), loss

    opt_state = optimizer.init(initial_phase)

    (final_phase, _), _ = jax.lax.scan(
        gd_step, (initial_phase, opt_state), jnp.arange(config.maxiter)
    )

    final_complex_ff = propagate_ff(source_amp * jnp.exp(1j * final_phase))
    final_intensity = jnp.abs(final_complex_ff) ** 2

    return final_phase, final_intensity


def solve_hologram(source_amp, target_amp, initial_phase, config: SolverConfig):
    if config.method.upper() == "GS":
        return run_gs(source_amp, target_amp, initial_phase, config)
    elif config.method.upper() == "GD":
        return run_gd(source_amp, target_amp, initial_phase, config)
    else:
        raise ValueError(f"Unknown solver method: {config.method}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # TODO HARDCODE
    hdf5_path = "./results/data/exercise_gs_20260727_145747.h5"

    slm_illumination, slm_meta = read_hdf5(
        hdf5_path, group_name="slm", dataset_name="slm_illumination"
    )

    target_intensity, target_meta = read_hdf5(
        hdf5_path, group_name="target", dataset_name="target_intensity"
    )

    slm_illumination = jnp.array(slm_illumination)
    target_intensity = jnp.array(target_intensity)
    target_amp = jnp.sqrt(target_intensity)

    WAVELENGTH = target_meta["wavelength"]
    PIXEL_PITCH = target_meta["pixel_pitch"]

    SLM_SHAPE = tuple(target_meta["slm_shape"])
    ARRAY_SHAPE = tuple(target_meta["array_shape"])
    ARRAY_PITCH = tuple(target_meta["array_pitch"])

    key = jax.random.PRNGKey(42)

    initial_phase = jax.random.uniform(key, SLM_SHAPE, minval=-jnp.pi, maxval=jnp.pi)

    config = SolverConfig(
        method="GS", maxiter=30, smooth_phase=False, smooth_sigma=1.5  # {"GS", "GD"}
    )

    logger.info(
        f"Starting {config.method} optimization over {config.maxiter} iterations..."
    )

    final_phase, inferred_intensity = solve_hologram(
        slm_illumination, target_amp, initial_phase, config
    )

    performance_metrics = compute_performance_metrics(inferred_intensity, target_intensity)
    
    pprint(performance_metrics)

    logger.info("Optimization complete.")

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
