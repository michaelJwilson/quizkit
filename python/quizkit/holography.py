import logging
from dataclasses import dataclass

import jax
# Enable 64-bit precision globally (float64 / complex128)
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import optax

from rich.pretty import pprint
from quizkit.readers import read_hdf5
from quizkit.exercise import (
    compute_performance_metrics,
    plot_phase_retrieval_results,
    get_trap_zoom,
)

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
    #    fft2 computes the 2D Fourier transform. The returned OTF is naturally in the native convention.
    return jnp.fft.fft2(jnp.fft.ifftshift(kernel))


def propagate_ff_native(complex_near):
    return jnp.fft.fft2(complex_near, norm="ortho")


def propagate_nf_native(complex_far):
    return jnp.fft.ifft2(complex_far, norm="ortho")


def run_gs(source_amp, target_amp, initial_phase, config: SolverConfig):
    source_amp_native = jnp.fft.ifftshift(source_amp)
    target_amp_native = jnp.fft.ifftshift(target_amp)
    initial_phase_native = jnp.fft.ifftshift(initial_phase)

    slm_shape = source_amp.shape
    blur_otf = get_gaussian_blur_otf(slm_shape, config.smooth_sigma)

    def gs_step(phase, _):
        complex_nf = source_amp_native * jnp.exp(1j * phase)
        complex_ff = propagate_ff_native(complex_nf)

        # NB replace amplitude with target constraint, keep current phase.
        ff_phase = jnp.angle(complex_ff)
        constrained_ff = target_amp_native * jnp.exp(1j * ff_phase)

        complex_nf_new = propagate_nf_native(constrained_ff)
        new_phase = jnp.angle(complex_nf_new)

        if config.smooth_phase:
            complex_phase = jnp.exp(1j * new_phase)
            blurred_complex = jnp.fft.ifft2(blur_otf * jnp.fft.fft2(complex_phase))
            new_phase = jnp.angle(blurred_complex)

        return new_phase, None

    # NB aid XLA for efficient compilation.
    final_phase_native, _ = jax.lax.scan(
        gs_step, initial_phase_native, jnp.arange(config.maxiter)
    )

    final_complex_ff_native = propagate_ff_native(
        source_amp_native * jnp.exp(1j * final_phase_native)
    )
    final_intensity_native = jnp.abs(final_complex_ff_native) ** 2

    # NB shift the optimized outputs back to the centered physical convention
    final_phase = jnp.fft.fftshift(final_phase_native)
    final_intensity = jnp.fft.fftshift(final_intensity_native)

    final_phase = jnp.mod(final_phase + jnp.pi, 2 * jnp.pi) - jnp.pi

    return final_phase, final_intensity


def run_gd(source_amp, target_amp, initial_phase, config: SolverConfig):
    source_amp_native = jnp.fft.ifftshift(source_amp)
    target_amp_native = jnp.fft.ifftshift(target_amp)
    initial_phase_native = jnp.fft.ifftshift(initial_phase)

    target_intensity_native = target_amp_native ** 2
    optimizer = optax.adam(learning_rate=config.learning_rate)

    def loss(phase):
        complex_nf = source_amp_native * jnp.exp(1j * phase)
        complex_ff = propagate_ff_native(complex_nf)
        inferred_intensity = jnp.abs(complex_ff) ** 2

        loss_mse = jnp.mean((inferred_intensity - target_intensity_native) ** 2)

        return loss_mse

    loss_and_grad = jax.value_and_grad(loss)

    @jax.jit
    def gd_step(carry, _):
        phase, opt_state = carry
        loss_val, grads = loss_and_grad(phase)
        updates, opt_state = optimizer.update(grads, opt_state, phase)
        new_phase = optax.apply_updates(phase, updates)

        # NB bound phase
        new_phase = jnp.mod(new_phase + jnp.pi, 2 * jnp.pi) - jnp.pi
        return (new_phase, opt_state), loss_val

    opt_state = optimizer.init(initial_phase_native)

    (final_phase_native, _), _ = jax.lax.scan(
        gd_step, (initial_phase_native, opt_state), jnp.arange(config.maxiter)
    )

    final_complex_ff_native = propagate_ff_native(
        source_amp_native * jnp.exp(1j * final_phase_native)
    )
    final_intensity_native = jnp.abs(final_complex_ff_native) ** 2

    # NB shift the optimized outputs back to the centered physical convention
    final_phase = jnp.fft.fftshift(final_phase_native)
    final_intensity = jnp.fft.fftshift(final_intensity_native)

    final_phase = jnp.mod(final_phase + jnp.pi, 2 * jnp.pi) - jnp.pi

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
    hdf5_path = "./results/data/exercise_gs_20260727_184334.h5"

    slm_illumination, slm_meta = read_hdf5(
        hdf5_path, group_name="slm", dataset_name="slm_illumination"
    )

    target_intensity, target_meta = read_hdf5(
        hdf5_path, group_name="target", dataset_name="target_intensity"
    )

    WAVELENGTH = target_meta["wavelength"] # m
    PIXEL_PITCH = target_meta["pixel_pitch"] # m

    SLM_SHAPE = tuple(target_meta["slm_shape"]) # (height, width) in pixels

    # NB e.g. 10x10 optical tweezer array sampling a 200x200 image.
    ARRAY_SHAPE = tuple(target_meta["array_shape"])
    ARRAY_PITCH = tuple(target_meta["array_pitch"])

    assert slm_illumination.max() > 0.0
    assert target_intensity.max() > 0.0

    slm_illumination = jnp.array(slm_illumination, dtype=jnp.float64)

    target_intensity = jnp.array(target_intensity, dtype=jnp.float64)
    target_amp = jnp.sqrt(target_intensity)

    key = jax.random.PRNGKey(42)

    initial_phase = jax.random.uniform(key, SLM_SHAPE, minval=-jnp.pi, maxval=jnp.pi, dtype=jnp.float64)

    config = SolverConfig(method="GD", maxiter=1_000, smooth_phase=False, smooth_sigma=5)

    logger.info(
        f"Starting {config.method} optimization over {config.maxiter} iterations..."
    )

    final_phase, inferred_intensity = solve_hologram(
        slm_illumination, target_amp, initial_phase, config
    )

    performance_metrics = compute_performance_metrics(
        inferred_intensity, target_intensity
    )

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
