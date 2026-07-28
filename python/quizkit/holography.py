import logging
import datetime
import os
from dataclasses import dataclass

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import optax
from torch.utils.tensorboard import SummaryWriter

from rich.pretty import pprint
from quizkit.readers import read_hdf5
from quizkit.exercise import (
    plot_phase_retrieval_results,
    get_trap_zoom,
)

logger = logging.getLogger(__name__)


@dataclass
class SolverConfig:
    method: str = "GD"  # {"GS", "GD"}
    maxiter: int = 200

    smooth_phase: bool = False  
    smooth_sigma: float = 2.0  

    loss_norm : str = "L2"  # {"L1", "L2"}

    learning_rate: float = 0.1

    # TODO HACK
    initial_epsilon: float = 0.0
    anneal_rate: float = 0.05


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


def compute_performance_metrics(ff_int, target_int):
    ff_int = jnp.asarray(ff_int)
    target_int = jnp.asarray(target_int)

    signal_mask = target_int > 0.0
    bg_mask = ~signal_mask

    signal_vals = jnp.where(signal_mask, ff_int, 0.0)
    bg_vals = jnp.where(bg_mask, ff_int, 0.0)

    total_power = jnp.sum(ff_int)
    signal_power = jnp.sum(signal_vals)
    bg_power = jnp.sum(bg_vals)

    efficiency = signal_power / (total_power + 1e-12)
    stray_light_fraction = bg_power / (total_power + 1e-12)

    signal_vals_nan = jnp.where(signal_mask, ff_int, jnp.nan)
    
    sig_min = jnp.nanmin(signal_vals_nan)
    sig_max = jnp.nanmax(signal_vals_nan)

    sig_constrast = sig_max / (sig_min + 1e-12)
    uniformity = 1.0 - ((sig_max - sig_min) / (sig_max + sig_min + 1e-12))

    max_bg_intensity = jnp.max(bg_vals)
    ghost_trap_ratio = max_bg_intensity / (sig_max + 1e-12)

    ff_centered = ff_int - jnp.mean(ff_int)
    target_centered = target_int - jnp.mean(target_int)

    numerator = jnp.sum(ff_centered * target_centered)
    denominator = jnp.sqrt(jnp.sum(ff_centered**2) * jnp.sum(target_centered**2))

    pearson = numerator / (denominator + 1e-12)

    return {
        "efficiency": efficiency,
        "stray_light_fraction": stray_light_fraction,
        "sig_constrast": sig_constrast,
        "uniformity": uniformity,
        "ghost_trap_ratio": ghost_trap_ratio,
        "pearson": pearson,
    }


def run_gs(source_amp, target_amp, initial_phase, config: SolverConfig):
    source_amp_native = jnp.fft.ifftshift(source_amp)
    target_amp_native = jnp.fft.ifftshift(target_amp)
    initial_phase_native = jnp.fft.ifftshift(initial_phase)

    slm_shape = source_amp.shape
    blur_otf = get_gaussian_blur_otf(slm_shape, config.smooth_sigma)

    def gs_step(phase, _):
        complex_nf = source_amp_native * jnp.exp(1j * phase)
        complex_ff = propagate_ff_native(complex_nf)

        ff_phase = jnp.angle(complex_ff)
        constrained_ff = target_amp_native * jnp.exp(1j * ff_phase)

        complex_nf_new = propagate_nf_native(constrained_ff)
        new_phase = jnp.angle(complex_nf_new)

        if config.smooth_phase:
            complex_phase = jnp.exp(1j * new_phase)
            blurred_complex = jnp.fft.ifft2(blur_otf * jnp.fft.fft2(complex_phase))
            new_phase = jnp.angle(blurred_complex)

        inferred_intensity = jnp.abs(complex_ff) ** 2
        metrics = compute_performance_metrics(inferred_intensity, target_amp_native**2)

        return new_phase, metrics

    final_phase_native, history = jax.lax.scan(
        gs_step, initial_phase_native, jnp.arange(config.maxiter)
    )

    final_complex_ff_native = propagate_ff_native(
        source_amp_native * jnp.exp(1j * final_phase_native)
    )
    final_intensity_native = jnp.abs(final_complex_ff_native) ** 2

    final_phase = jnp.fft.fftshift(final_phase_native)
    final_intensity = jnp.fft.fftshift(final_intensity_native)
    final_phase = jnp.mod(final_phase + jnp.pi, 2 * jnp.pi) - jnp.pi

    return final_phase, final_intensity, history


def run_gd(source_amp, target_amp, initial_phase, config: SolverConfig, smooth_lambda=0.0):
    source_amp_native = jnp.fft.ifftshift(source_amp)
    target_amp_native = jnp.fft.ifftshift(target_amp)
    initial_phase_native = jnp.fft.ifftshift(initial_phase)

    target_intensity_native = target_amp_native**2
    optimizer = optax.adam(learning_rate=config.learning_rate)

    slm_shape = source_amp.shape
    blur_otf = get_gaussian_blur_otf(slm_shape, config.smooth_sigma)

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

        if config.loss_norm.upper() == "L1":
            loss_val = jnp.mean(jnp.abs(diff))
        elif config.loss_norm.upper() == "L2":
            loss_val = jnp.mean(diff ** 2)
        else:
            raise ValueError(f"Unsupported loss norm: {config.loss_norm}")

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
        
        epsilon = config.initial_epsilon * jnp.exp(-config.anneal_rate * step_idx)
        
        key_phase, key_mask = jax.random.split(subkey, 2)
        
        random_phases = jax.random.uniform(
            key_phase, phase.shape, minval=-jnp.pi, maxval=jnp.pi
        )
        
        explore_mask = jax.random.uniform(key_mask, phase.shape) < epsilon
        
        new_phase = jnp.where(explore_mask, random_phases, new_phase)

        if config.smooth_phase:
            complex_phase = jnp.exp(1j * new_phase)
            blurred_complex = jnp.fft.ifft2(blur_otf * jnp.fft.fft2(complex_phase))
            new_phase = jnp.angle(blurred_complex)

        new_phase = jnp.mod(new_phase + jnp.pi, 2 * jnp.pi) - jnp.pi

        metrics = compute_performance_metrics(inferred_intensity, target_intensity_native)
        metrics["loss"] = loss_val

        return (new_phase, opt_state, key), metrics

    opt_state = optimizer.init(initial_phase_native)
    step_key = jax.random.PRNGKey(42)

    (final_phase_native, _, _), history = jax.lax.scan(
        gd_step, 
        (initial_phase_native, opt_state, step_key), 
        jnp.arange(config.maxiter)
    )

    final_complex_ff_native = propagate_ff_native(
        source_amp_native * jnp.exp(1j * final_phase_native)
    )
    final_intensity_native = jnp.abs(final_complex_ff_native) ** 2

    final_phase = jnp.fft.fftshift(final_phase_native)
    final_intensity = jnp.fft.fftshift(final_intensity_native)
    final_phase = jnp.mod(final_phase + jnp.pi, 2 * jnp.pi) - jnp.pi

    return final_phase, final_intensity, history


def solve_hologram(source_amp, target_amp, initial_phase, config: SolverConfig):
    if config.method.upper() == "GS":
        return run_gs(source_amp, target_amp, initial_phase, config)
    elif config.method.upper() == "GD":
        return run_gd(source_amp, target_amp, initial_phase, config)
    else:
        raise ValueError(f"Unknown solver method: {config.method}")

# tensorboard --logdir=./runs
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

    WAVELENGTH = target_meta["wavelength"]  # m
    PIXEL_PITCH = target_meta["pixel_pitch"]  # m
    SLM_SHAPE = tuple(target_meta["slm_shape"])  # (height, width) in pixels
    ARRAY_SHAPE = tuple(target_meta["array_shape"])
    ARRAY_PITCH = tuple(target_meta["array_pitch"])

    assert slm_illumination.max() > 0.0
    assert target_intensity.max() > 0.0

    slm_illumination = jnp.array(slm_illumination, dtype=jnp.float64)
    target_intensity = jnp.array(target_intensity, dtype=jnp.float64)
    target_amp = jnp.sqrt(target_intensity)

    key = jax.random.PRNGKey(42)
    initial_phase = jax.random.uniform(
        key, SLM_SHAPE, minval=-jnp.pi, maxval=jnp.pi, dtype=jnp.float64
    )

    config = SolverConfig(method="GD", maxiter=200, smooth_phase=False, smooth_sigma=5)
    logger.info(f"Starting {config.method} optimization over {config.maxiter} iterations...")

    final_phase, inferred_intensity, history = solve_hologram(
        slm_illumination, target_amp, initial_phase, config
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./runs/{config.method}_optimization_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    for step_idx in range(config.maxiter):
        if "loss" in history:
            writer.add_scalar("Loss/Total", history["loss"][step_idx].item(), step_idx)
            
        writer.add_scalar("Metrics/Efficiency", history["efficiency"][step_idx].item(), step_idx)
        writer.add_scalar("Metrics/Uniformity", history["uniformity"][step_idx].item(), step_idx)
        writer.add_scalar("Metrics/Ghost_Trap_Ratio", history["ghost_trap_ratio"][step_idx].item(), step_idx)
        writer.add_scalar("Metrics/Stray_Light", history["stray_light_fraction"][step_idx].item(), step_idx)
        writer.add_scalar("Metrics/Pearson", history["pearson"][step_idx].item(), step_idx)

    writer.close()

    performance_metrics = compute_performance_metrics(
        inferred_intensity, target_intensity
    )
    
    # Cast final scalars to float for rich.pprint
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
