import ast
import h5py
import datetime
import random
import pickle

import json
from dataclasses import dataclass, asdict, field
from typing import Tuple, Optional, Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rich.pretty import pprint
from scipy.ndimage import find_objects, label
from slmsuite.holography.algorithms import SpotHologram
import matplotlib.gridspec as gridspec
from path import pathlib
from functools import cached_property

from quizkit.writers import write_hdf5

"""
GS algorithm application via slm suite, see

https://slmsuite.readthedocs.io/en/latest/_examples/computational_holography.html#Basic-Image-Formation
"""

# NB slmsuite seeds on random, not numpy (but potentially cupy/cuda).
random.seed(42)
np.random.seed(42)


def get_gaussian_slm_illumination(slm_shape):
    # NB construct source amplitude profile
    x = np.arange(slm_shape[1]) - (slm_shape[1] - 1) / 2
    y = np.arange(slm_shape[0]) - (slm_shape[0] - 1) / 2

    xx, yy = np.meshgrid(x, y)

    # TODO HARDCODE
    beam_waist_px = 0.35 * min(slm_shape)  # 1/e^2 amplitude radius, in pixels

    return np.exp(-(xx**2 + yy**2) / beam_waist_px**2).astype(np.float32)

def _add_colorbar(ax, im, label=None):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.figure.colorbar(im, cax=cax, label=label)

def plot_scalar_field(plot_path, field, cmap="inferno", title=None, extent=None, 
                      cbar_label=None, hide_ticks=False, figsize=(5, 3.2),
                      xlabel=None, ylabel=None, **imshow_kwargs):
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(field, cmap=cmap, extent=extent, **imshow_kwargs)
    ax.set_aspect("equal")
    
    if title:
        ax.set_title(title, fontsize=14)  # Match your phase retrieval titlesize
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if hide_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        
    _add_colorbar(ax, im, cbar_label)
    
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

"""
def plot_target_intensity(plot_path, target_intensity, target_extent=None):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    im = ax.imshow(target_intensity, cmap="inferno", extent=target_extent)
    ax.set_aspect("equal")  # show that the beam is symmetric

    # fig.colorbar(im, ax=ax, label="slm illumination")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(im, cax=cax, label="slm illumination")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)
"""
"""
def plot_trap_stack_mean(plot_path, trap_stack_mean):
    fig, ax = plt.subplots(figsize=(4, 4))

    im = ax.imshow(trap_stack_mean, cmap="inferno")
    ax.set_aspect("equal")
    ax.set_title("Mean Trap Profile", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, label="Forward Intensity")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
"""

def plot_phase_retrieval_results(plot_path, phase, intensity, intensity_extent=None):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.titlesize": 14,
            "figure.titlesize": 16,
        }
    )

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))

    im0 = axs[0].imshow(phase, cmap="twilight", interpolation="nearest")
    axs[0].set_title("slm")
    axs[0].set_xlabel(r"$x [\Delta]$")
    axs[0].set_ylabel(r"$y [\Delta]$")

    # adjustable="box"
    # axs[0].set_aspect("equal")

    div0 = make_axes_locatable(axs[0])
    cax0 = div0.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im0, cax=cax0, label="phase [rad]")

    im1 = axs[1].imshow(
        intensity / intensity.max(),
        cmap="inferno",
        origin="lower",
        extent=intensity_extent,
    )
    axs[1].set_title("far-field")
    axs[1].set_xlabel(r"$k_n$ [knm]")
    axs[1].set_ylabel(r"$k_m$ [knm]")

    # adjustable="box"
    axs[1].set_aspect("equal")

    div1 = make_axes_locatable(axs[1])
    cax1 = div1.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im1, cax=cax1, label="intensity [a.u.]")

    fig.tight_layout(pad=2.0)
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def compute_structural_metrics(wavelength, pixel_pitch, slm_shape):
    # TODO https://slmsuite.readthedocs.io/en/latest/_autosummary/slmsuite.holography.algorithms.Hologram.html

    # NB maximum direction we can redirect the input beam,
    max_steering_angle = wavelength / pixel_pitch

    # NB O(1) degrees
    max_steering_angle_deg = np.degrees(max_steering_angle)

    # NB nyquist wavenumber on the image place,
    #    requires a period of exactly 2 pixels (a phase map of [0,π,0,π])
    farfield_extent = max_steering_angle / 2.0  # radians
    farfield_resolution = (
        farfield_extent / slm_shape
    )  # radians, assumes square slm pixels.

    return {
        "max_steering_angle_rad": float(max_steering_angle),
        "max_steering_angle_deg": float(max_steering_angle_deg),
        "farfield_extent_rad": float(farfield_extent),
        "farfield_resolution_rad": float(farfield_resolution),
    }


def compute_performance_metrics(forward_intensity, target_intensity):
    # NB given far-field intensity and target intensity ...
    ff_int = np.asarray(forward_intensity, dtype=np.float32)
    target_int = np.asarray(target_intensity, dtype=np.float32)

    # NB max target intensity;
    # target_max = np.max(target_int)

    # TODO HARDCODE
    # signal_mask = target_int > (0.01 * target_max)
    signal_mask = target_int > 0.0
    bg_mask = ~signal_mask

    signal_intensities = forward_intensity[signal_mask]
    bg_intensities = forward_intensity[bg_mask]

    total_power = np.sum(forward_intensity)

    signal_power = np.sum(signal_intensities)
    bg_power = np.sum(bg_intensities)

    # NB fraction of realized power in the target/signal region
    efficiency = signal_power / total_power

    # TODO
    stray_light_fraction = bg_power / total_power

    sig_min = np.min(signal_intensities)
    sig_max = np.max(signal_intensities)

    sig_constrast = sig_max / (sig_min + 1e-12)

    # NB michelson uniformity = (I_max - I_min) / (I_max + I_min) = 0.5 * (max - min) / mean
    uniformity = 1.0 - ((sig_max - sig_min) / (sig_max + sig_min + 1e-12))

    # NB frac. standard deviation wrt the med. target intensity (no background).
    # cv = sig_std / (sig_med + 1e-12)

    max_bg_intensity = np.max(bg_intensities)
    ghost_trap_ratio = max_bg_intensity / (sig_max + 1e-12)

    # ff_norm = ff_int / total_power
    # target_norm = target_int / np.sum(target_int)

    # NB root mean square error (RMSE) between normalized far-field and target intensities
    # rmse = np.sqrt(np.mean((ff_norm - target_norm) ** 2))

    ff_centered = ff_int - np.mean(ff_int)
    target_centered = target_int - np.mean(target_int)

    numerator = np.sum(ff_centered * target_centered)
    denominator = np.sqrt(np.sum(ff_centered**2) * np.sum(target_centered**2))

    pearson = numerator / (denominator + 1e-12)

    return {
        "efficiency": float(efficiency),
        "stray_light_fraction": float(stray_light_fraction),
        "sig_constrast": float(sig_constrast),
        "uniformity": float(uniformity),
        "ghost_trap_ratio": float(ghost_trap_ratio),
        "pearson": float(pearson),
    }

def compute_trap_metrics(inferred_intensity, trap_mask, num_traps):
    """
    Computes inter/intra contrast metrics using a single 2D integer mask.
    Args:
        inferred_intensity: (H, W) array of forward intensity.
        trap_mask: (H, W) static integer array.
        num_traps: Static integer.
    """
    flat_intensity = inferred_intensity.ravel()
    flat_labels = trap_mask.ravel()

    trap_powers = jnp.bincount(
        flat_labels, weights=flat_intensity, length=num_traps + 1
    )[1:]

    mean_power = jnp.mean(trap_powers)
    inter_uniformity = (jnp.max(trap_powers) - jnp.min(trap_powers)) / (
        2 * mean_power + 1e-12
    )

    return {
        "inter_uniformity": inter_uniformity,
        "trap_powers": trap_powers,
    }


def write_performance_metrics_tex(
    filepath: str | pathlib.Path,
    metrics_by_run: dict[str, dict], 
    caption: str = "Computed performance metrics for the optimized SLM phase.", 
    label: str = "tab:hologram_metrics"
) -> None:
    """
    Generates and writes a LaTeX table comparing performance metrics across solver runs.
    
    Args:
        filepath: The destination path for the .tex file.
        metrics_by_run: Dictionary mapping run names to their metric dictionaries.
                        e.g., {"gs": dict, "wgs": dict}.
        caption: Table caption string.
        label: Table LaTeX label.
    """
    # Descriptions for the PerformanceMetrics properties
    descriptions = {
        "efficiency": "Fraction of total power within target trap regions",
        "stray_light_fraction": "Fraction of total power outside target trap regions (1 - efficiency)",
        "pearson": "Pearson correlation of forward intensity and target intensity",
        "trap_cv": "Coefficient of variation of integrated trap powers",
        "trap_mean": "Mean integrated trap power",
        "trap_min": "Minimum integrated trap power",
        "trap_max": "Maximum integrated trap power",
        "trap_uniformity_minmax": "Min-max (Michelson) uniformity of integrated trap powers",
        "ghost_to_mean_ratio": "Ratio of max background intensity to mean trap power",
        "ghost_to_dimmest_ratio": "Ratio of max background intensity to minimum trap power",
        "signal_to_background_floor": "Ratio of mean trap power to mean background intensity"
    }

    run_keys = list(metrics_by_run.keys())
    n_runs = len(run_keys)

    # 1. Determine Column Headers and Tabular Alignment
    # Always use the provided key(s), capitalized
    headers = [f"\\textbf{{{key.capitalize()}}}" for key in run_keys]
        
    c_cols = "c" * n_runs
    tabular_def = f"\\begin{{tabular}}{{l{c_cols}p{{11.5cm}}}}"

    # 2. Build Header Row
    header_row = " & ".join(["\\textbf{Metric Key}"] + headers + ["\\textbf{Description}"]) + " \\\\"

    # 3. Build Data Rows
    # Grab the metric names from the first run (excluding the raw array)
    metric_names = [k for k in metrics_by_run[run_keys[0]].keys() if k != "trap_powers"]

    rows = []
    for m_name in metric_names:
        escaped_m_name = m_name.replace("_", "\\_")
        row_parts = [f"\\texttt{{{escaped_m_name}}}"]

        for key in run_keys:
            val = metrics_by_run[key].get(m_name, np.nan)
            
            if isinstance(val, float) and not np.isnan(val):
                if val != 0 and (abs(val) < 1e-3 or abs(val) > 1e4):
                    row_parts.append(f"{val:.2e}")
                else:
                    row_parts.append(f"{val:.4f}")
            else:
                row_parts.append(str(val))

        desc = descriptions.get(m_name, "")
        row_parts.append(desc)
        rows.append(" & ".join(row_parts) + " \\\\")

    latex = f"""\\begin{{table}}[htbp]
\\centering
\\small
{tabular_def}
\\toprule
{header_row}
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\caption{{{caption}}}
\\label{{{label}}}
\\end{{table}}"""

    out_path = pathlib.Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w") as f:
        f.write(latex)


def get_trap_zoom(target_intensity, pad=25):
    trap_coords = np.argwhere(target_intensity > 0)

    assert trap_coords.shape[0] > 0, "No traps found in target intensity."

    y_min, y_max = np.min(trap_coords[:, 0]), np.max(trap_coords[:, 0])
    x_min, x_max = np.min(trap_coords[:, 1]), np.max(trap_coords[:, 1])

    y_min = max(0, y_min - pad)
    y_max = min(target_intensity.shape[0] - 1, y_max + pad)
    x_min = max(0, x_min - pad)
    x_max = min(target_intensity.shape[1] - 1, x_max + pad)

    return (x_min, x_max, y_min, y_max)


def encode_target_traps(target_intensity, threshold_frac=0.0):
    threshold = threshold_frac * target_intensity.max()
    binary_target = target_intensity > threshold

    labeled_mask, num_traps = label(binary_target)
    slices = find_objects(labeled_mask)

    coords = []
    trap_h, trap_w = 0, 0

    for s in slices:
        trap_h = max(trap_h, s[0].stop - s[0].start)
        trap_w = max(trap_w, s[1].stop - s[1].start)

    for s in slices:
        center_y = (s[0].start + s[0].stop) // 2
        center_x = (s[1].start + s[1].stop) // 2
        coords.append((center_y, center_x))

    return labeled_mask, num_traps, np.array(coords), trap_h, trap_w


def reduce_stack_similar_traps(
    forward_intensity,
    center_coords,
    stack_h,
    stack_w,
    weights=None,
    reducer=jnp.mean,
):
    def crop_single(coord):
        start_y = coord[0] - (stack_h // 2)
        start_x = coord[1] - (stack_w // 2)

        return jax.lax.dynamic_slice(
            forward_intensity, (start_y, start_x), (stack_h, stack_w)
        )

    # Shape: (N, stack_h, stack_w)
    trap_stack = jax.vmap(crop_single)(center_coords)

    if weights is not None:
        w = weights[:, None, None]

        if reducer in (jnp.mean, jnp.average):
            reduced_profile = jnp.sum(trap_stack * w, axis=0) / (jnp.sum(w) + 1e-12)
        else:
            raise NotImplementedError(
                "Weighted reduction is only implemented for mean/average."
            )
    else:
        reduced_profile = reducer(trap_stack, axis=0)

    return reduced_profile


def extract_background_artifacts(
    forward_intensity, target_intensity, threshold_frac=0.25, max_artifacts=9
):
    bg_mask = target_intensity == 0.0
    residual_int = forward_intensity * bg_mask

    artifact_threshold = residual_int.max() * threshold_frac
    binary_artifacts = residual_int > artifact_threshold

    labeled_artifacts, _ = label(binary_artifacts)
    slices = find_objects(labeled_artifacts)

    artifacts = []
    for i, s in enumerate(slices):
        cy = (s[0].start + s[0].stop) // 2
        cx = (s[1].start + s[1].stop) // 2

        comp_mask = labeled_artifacts == (i + 1)
        comp_power = np.sum(residual_int[comp_mask])

        artifacts.append({"id": i, "cy": cy, "cx": cx, "power": comp_power})

    artifacts.sort(key=lambda x: x["power"], reverse=True)
    artifacts = artifacts[:max_artifacts]

    return artifacts, residual_int


def crop_artifact_stacks(ff_int, artifacts, stack_h, stack_w):
    stacks = []
    H, W = ff_int.shape
    for art in artifacts:
        cy, cx = art["cy"], art["cx"]

        y0 = cy - stack_h // 2
        y1 = y0 + stack_h
        x0 = cx - stack_w // 2
        x1 = x0 + stack_w

        crop = np.zeros((stack_h, stack_w), dtype=ff_int.dtype)

        valid_y0, valid_y1 = max(0, y0), min(H, y1)
        valid_x0, valid_x1 = max(0, x0), min(W, x1)

        dest_y0 = valid_y0 - y0
        dest_y1 = dest_y0 + (valid_y1 - valid_y0)
        dest_x0 = valid_x0 - x0
        dest_x1 = dest_x0 + (valid_x1 - valid_x0)

        if valid_y1 > valid_y0 and valid_x1 > valid_x0:
            crop[dest_y0:dest_y1, dest_x0:dest_x1] = ff_int[
                valid_y0:valid_y1, valid_x0:valid_x1
            ]

        stacks.append(crop)

    return np.array(stacks)

"""
def label_slm_fuzz(artifacts, array_shape, array_pitch, array_center, slm_shape, pad=25):
    center_y, center_x = slm_shape[0] / 2.0, slm_shape[1] / 2.0

    if array_center is not None:
        center_x += array_center[0]
        center_y += array_center[1]

    Ny, Nx = array_shape
    dy, dx = array_pitch

    x_half_width = (Nx - 1) / 2.0 * dx
    y_half_height = (Ny - 1) / 2.0 * dy

    x_min = center_x - x_half_width - pad
    x_max = center_x + x_half_width + pad
    y_min = center_y - y_half_height - pad
    y_max = center_y + y_half_height + pad

    for art in artifacts:
        cx, cy = art["cx"], art["cy"]

        if (x_min <= cx <= x_max) and (y_min <= cy <= y_max):
            art["is_fuzz"] = True
        else:
            art["is_fuzz"] = False
            
    return artifacts


def export_artifact_data_for_streamlit(
    filepath, residual_int, artifacts, artifact_stacks, slm_shape, array_shape, array_pitch, array_center, 
):
    data_bundle = {
        'residual_int': residual_int,
        'artifacts': artifacts,
        'artifact_stacks': artifact_stacks,
        'slm_shape': slm_shape,
        'array_shape': array_shape,
        'array_pitch': array_pitch,
        'array_center': array_center,
    }
    with open(filepath, 'wb') as f:
        pickle.dump(data_bundle, f)
    print(f"Artifact data exported to {filepath}")
"""
class ConfigMixin:
    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 4) -> str:
        return json.dumps(self.to_dict(), indent=indent)

@dataclass
class TrapConfig(ConfigMixin):
    trap_config_id: int 
    array_shape: Tuple[int, int]
    array_pitch: Tuple[int, int]
    array_center: Optional[Tuple[float, float]] = None


@dataclass
class RunConfig(ConfigMixin):
    wavelength: float
    pixel_pitch: float
    slm_shape: Tuple[int, int]
    trap_config: TrapConfig
    comment: Optional[str] = None

    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    
    def __getattr__(self, name):
        try:
            return getattr(self.trap_config, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

@dataclass
class SolverConfig(ConfigMixin):
    method: str  # {"GS", "GS", "GD"}
    maxiter: int = 200

    solver_backend : str | None = None

    smooth_phase: bool = False
    smooth_sigma: int = 3 # pixels

    loss_norm: str = "L2"  # {"L1", "L2"}
    learning_rate: float = 0.1

    # TODO HACK
    initial_epsilon: float = 0.0
    anneal_rate: float = 0.05

    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )

class HologramExperiment:
    _is_frozen = False

    def __init__(self, run_config: RunConfig):
        self.run_config = run_config
        self.slm_illumination = get_gaussian_slm_illumination(self.run_config.slm_shape)        
        self.slm_illumination.flags.writeable = False
        
        hologram = SpotHologram.make_rectangular_array(
            self.run_config.slm_shape,
            array_shape=self.run_config.array_shape,
            array_pitch=self.run_config.array_pitch,
            basis="knm",
            amp=self.slm_illumination,
            array_center=self.run_config.array_center,
            phase=np.random.uniform(-np.pi, np.pi, self.run_config.slm_shape),
        )

        self.target = hologram.target.copy()
        self.target.flags.writeable = False
        
        (
            trap_labels_np, 
            self.num_traps, 
            coords_np, 
            self.trap_h, 
            self.trap_w
        ) = encode_target_traps(self.target, threshold_frac=0.0)
        
        self.trap_labels = jnp.array(trap_labels_np)
        self.trap_coords = jnp.array(coords_np)

        # TODO DEPRECATE
        self.crop_coords = self.trap_coords
        
        self._is_frozen = True

    @classmethod
    def from_run_h5(cls, run_dir: str | pathlib.Path):
        run_dir = pathlib.Path(run_dir)
        
        with open(run_dir / "run_config.json", "r") as f:
            data = json.load(f)
            trap_data = data.pop("trap_config")
            data["trap_config"] = TrapConfig(**trap_data)
            run_config = RunConfig(**data)
            
        obj = cls.__new__(cls)
        obj.run_config = run_config
        
        h5_path = run_dir / "experiment.h5"

        with h5py.File(h5_path, "r") as f:
            obj.slm_illumination = f["slm/slm_illumination"][:]
            obj.target = f["target/target"][:]
            
        obj.slm_illumination.flags.writeable = False
        obj.target.flags.writeable = False

        (
            trap_labels_np, 
            obj.num_traps, 
            coords_np, 
            obj.trap_h, 
            obj.trap_w
        ) = encode_target_traps(obj.target, threshold_frac=0.0)
        
        obj.trap_labels = jnp.array(trap_labels_np)
        obj.trap_coords = jnp.array(coords_np)
        obj.crop_coords = obj.trap_coords
        
        obj._is_frozen = True
        return obj

    def __setattr__(self, key, value):
        if getattr(self, "_is_frozen", False):
            cls_attr = getattr(type(self), key, None)
            if isinstance(cls_attr, cached_property):
                super().__setattr__(key, value)
                return
            raise AttributeError(
                f"'{type(self).__name__}' is immutable. Cannot modify attribute '{key}'."
            )
        super().__setattr__(key, value)

    @cached_property
    def target_intensity(self):
        intensity = np.abs(self.target) ** 2
        intensity.flags.writeable = False
        return intensity

    @cached_property
    def target_extent(self):
        return get_trap_zoom(self.target)

    @cached_property
    def geometric_metrics(self):
        wave = self.run_config.wavelength
        pitch = self.run_config.pixel_pitch
        h, w = self.run_config.slm_shape

        max_steering_angle = wave / pitch
        farfield_extent = max_steering_angle / 2.0
        
        farfield_res_y = farfield_extent / h
        farfield_res_x = farfield_extent / w

        return {
            "max_steering_angle_rad": float(max_steering_angle),
            "max_steering_angle_deg": float(np.degrees(max_steering_angle)),
            "farfield_extent_rad": float(farfield_extent),
            "farfield_res_y_rad": float(farfield_res_y),
            "farfield_res_x_rad": float(farfield_res_x),
        }

    def _get_run_dir(self, base_dir: str) -> pathlib.Path:
        return pathlib.Path(base_dir) / f"run_{self.run_config.timestamp}"

    def plot(self, base_dir: str = "./results"):
        run_dir = self._get_run_dir(base_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "plots").mkdir(parents=True, exist_ok=True)
        
        plot_scalar_field(
            run_dir / "plots" / "gaussian_slm_illumination.pdf",
            np.log(self.slm_illumination + 1e-12),
            cbar_label="ln. slm illumination"
        )

        x_min, x_max, y_min, y_max = self.target_extent
        plot_scalar_field(
            run_dir / "plots" / "target_intensity.pdf",
            np.log(self.target_intensity[y_min:y_max, x_min:x_max] + 1e-12),
            extent=self.target_extent,
            cbar_label="ln. target intensity"
        )

    def write_h5(self, base_dir: str = "./results"):
        run_dir = self._get_run_dir(base_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        
        with open(run_dir / "run_config.json", "w") as f:
            f.write(self.run_config.to_json())
            
        hdf5_path = run_dir / "experiment.h5"
        
        write_hdf5(
            filepath=hdf5_path,
            data=self.slm_illumination,
            group_name="slm",
            dataset_name="slm_illumination"
        )
    
        write_hdf5(
            filepath=hdf5_path,
            data=self.target,
            group_name="target",
            dataset_name="target"
        )

@dataclass
class PerformanceMetrics(ConfigMixin):
    efficiency: float
    stray_light_fraction: float
    pearson: float
    trap_cv: float
    trap_mean: float
    trap_min: float
    trap_max: float
    trap_uniformity_minmax: float
    ghost_to_mean_ratio: float
    ghost_to_dimmest_ratio: float
    signal_to_background_floor: float
    trap_powers: np.ndarray

    @classmethod
    def compute(
        cls,
        forward_intensity: np.ndarray,
        target_intensity: np.ndarray,
        trap_labels: jnp.ndarray | np.ndarray,
        num_traps: int,
    ) -> "PerformanceMetrics":
        ff_int = np.asarray(forward_intensity, dtype=np.float64)
        target_int = np.asarray(target_intensity, dtype=np.float64)

        flat_forward_intensity = ff_int.ravel()
        flat_trap_labels = np.asarray(trap_labels).ravel()

        # Integrated trap power reduction via JAX bincount
        trap_powers_jax = jnp.bincount(
            flat_trap_labels,
            weights=flat_forward_intensity,
            length=num_traps + 1,
        )[1:]
        trap_powers = np.asarray(trap_powers_jax, dtype=np.float64)

        # Trap Power Statistics
        trap_min = float(np.min(trap_powers))
        trap_max = float(np.max(trap_powers))
        trap_mean = float(np.mean(trap_powers))
        trap_std = float(np.std(trap_powers))

        trap_cv = float(trap_std / (trap_mean + 1e-12))

        # Global Power Distributions
        total_power = float(np.sum(flat_forward_intensity))
        sig_power = float(np.sum(trap_powers))
        bg_power = total_power - sig_power

        # Background Floor & Ghost Trap Analysis
        bg_mask = flat_trap_labels == 0
        bg_intensities = flat_forward_intensity[bg_mask]
        max_bg = float(np.max(bg_intensities))
        mean_bg = float(np.mean(bg_intensities))

        # Global Field Pearson Correlation
        ff_centered = ff_int - np.mean(flat_forward_intensity)
        target_centered = target_int - np.mean(target_int)
        numerator = np.sum(ff_centered * target_centered)
        denominator = np.sqrt(np.sum(ff_centered**2) * np.sum(target_centered**2))
        pearson = float(numerator / (denominator + 1e-12))

        return cls(
            efficiency=sig_power / total_power,
            stray_light_fraction=bg_power / total_power,
            pearson=pearson,
            trap_cv=trap_cv,
            trap_mean=trap_mean,
            trap_min=trap_min,
            trap_max=trap_max,
            trap_uniformity_minmax=1.0 - ((trap_max - trap_min) / (trap_max + trap_min + 1e-12)),
            ghost_to_mean_ratio=max_bg / (trap_mean + 1e-12),
            ghost_to_dimmest_ratio=max_bg / (trap_min + 1e-12),
            signal_to_background_floor=trap_mean / (mean_bg + 1e-12),
            trap_powers=trap_powers,
        )

    @classmethod
    def from_solver(cls, solver: Any) -> "PerformanceMetrics":
        return cls.compute(
            forward_intensity=solver.forward_intensity,
            target_intensity=solver.target_intensity,
            trap_labels=solver.exp.trap_labels,
            num_traps=solver.exp.num_traps,
        )

class HologramExperimentSolver:
    def __init__(self, experiment: HologramExperiment, config: SolverConfig):
        self.exp = experiment
        self.config = config

        self.__hologram = SpotHologram.make_rectangular_array(
            self.exp.run_config.slm_shape,
            array_shape=self.exp.run_config.array_shape,
            array_pitch=self.exp.run_config.array_pitch,
            basis="knm",
            amp=self.exp.slm_illumination,
            array_center=self.exp.run_config.array_center,
            phase=np.random.uniform(-np.pi, np.pi, self.exp.run_config.slm_shape),
        )

        assert np.allclose(self.exp.target, self.__hologram.target)

    def optimize(self):
        self.__hologram.optimize(
            method=self.config.method,
            maxiter=self.config.maxiter,
            stat_groups=["computational_spot"],
            verbose=False,
        )

    @property
    def target_intensity(self):
        return self.exp.target_intensity
    
    @property
    def target_extent(self):
        return self.exp.target_extent

    @property
    def slm_phase(self):
        return self.__hologram.get_phase()

    @property
    def forward_intensity(self):
        return np.abs(self.__hologram.get_farfield()) ** 2

    def plot(self, base_dir: str = "./results"):
        plot_scalar_field(
            plot_path="./results/plots/slm_phase.pdf",
            field=slm_phase,
            cmap="twilight",
            title="slm",
            cbar_label="phase [rad]",
            xlabel=r"$x [\Delta]$",
            ylabel=r"$y [\Delta]$",
            interpolation="nearest"
            )

        # TODO HARDCODE
        stack_h, stack_w = 50, 50
        stack_mean_similar_traps = reduce_stack_similar_traps(
            self.forward_intensity, self.exp.trap_coords, stack_h, stack_w
        )
        
        plot_scalar_field(
            f"{base_dir}/plots/trap_stack_forward_ln_intensity.pdf",
            np.log(stack_mean_similar_traps + 1e-12),
        )

        plot_scalar_field(
            plot_path="./results/plots/farfield_intensity.pdf",
            field=ff_int[y_min:y_max, x_min:x_max] / ff_int[y_min:y_max, x_min:x_max].max(),
            cmap="inferno",
            title="far-field",
            extent=target_extent,
            cbar_label="intensity [a.u.]",
            xlabel=r"$k_n$ [knm]",
            ylabel=r"$k_m$ [knm]",
            origin="lower"  # Crucial: prevents the array from rendering upside down
        )

    def write_h5(self, base_dir: str = "./results"):
        out_dir = self.exp._get_run_dir(base_dir) / f"phase_retrieval/{self.config.timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / "solver_config.json", "w") as f:
            f.write(self.config.to_json())
            
        hdf5_path = out_dir / "phase_solution_{self.config.timestamp}.h5"

        write_hdf5(
            filepath=hdf5_path,
            data=self.slm_phase,
            group_name="slm",
            dataset_name=f"slm_phase_{self.config.method.lower()}",
        )

        write_hdf5(
            filepath=hdf5_path,
            data=self.forward_intensity,
            group_name=self.config.method.lower(),
            dataset_name="inferred_farfield_intensity",
        )


if __name__ == "__main__":
    slm_shape=(1200, 1920) # (height, width) in pixels,

    # NB (float, float) or None; shift from zeroth order in the far-field basis. If None, defaults to the zeroth order position.
    #    see https://github.com/holodyne/slmsuite/blob/39243f081de020ad3ba74e672d126694b80778d2/slmsuite/holography/algorithms/_spots.py#L1423
    #
    # `"knm"``, this is ``(shape[1], shape[0])/2``.
    # ``"kxy"``, this is ``(0,0)``.
    # ``"ij"``, this is the pixel position of the zeroth order on the camera (via Fourier calibration).
    trap_config = TrapConfig(
       trap_config_id=0,
       array_shape=(10, 10),
       array_pitch=(20, 20), # spot separation in far-field grid samples
       array_center=None
    )

    trap_config_off_center = TrapConfig(
        trap_config_id=1,
        array_shape=(10, 10),
        array_pitch=(20, 20), # spot separation in far-field grid samples
        array_center=(3. * slm_shape[1], 2. * slm_shape[0])/4, 
    )

    pprint(trap_config, expand_all=True)
    pprint(trap_config_off_center, expand_all=True)

    # NB 10x10 optical tweezer array sampling a 200x200 image.
    run_config = RunConfig(
       wavelength=780e-9, #m
       pixel_pitch=8.0e-6, #m
       slm_shape=slm_shape,
       trap_config=trap_config,
       comment="default slm suite run"
    )

    pprint(run_config, expand_all=True)

    solver_config = SolverConfig(
        method="GS", # {GS, WGS}
        maxiter=200,
        solver_backend="slm_suite",
    )

    pprint(solver_config, expand_all=True)

    slm_illumination = get_gaussian_slm_illumination(run_config.slm_shape)

    # NB construct tweezer array hologram and optimize it with GS algorithm
    #    see https://github.com/holodyne/slmsuite/blob/39243f081de020ad3ba74e672d126694b80778d2/slmsuite/holography/algorithms/_hologram.py#L26
    hologram = SpotHologram.make_rectangular_array(
        run_config.slm_shape,
        array_shape=run_config.array_shape,
        array_pitch=run_config.array_pitch,
        basis="knm",  # pixel coordinates in the far-field image plane
        amp=slm_illumination,  # fixed Gaussian illumination
        array_center=run_config.array_center,  # shift from zeroth order
        phase=np.random.uniform(-np.pi, np.pi, run_config.slm_shape),  # reproducibility required.
    )

    trap_labels_np, num_traps, coords_np, trap_h, trap_w = encode_target_traps(
        hologram.target, threshold_frac=0.0
    )
    
    trap_labels_jax = jnp.array(trap_labels_np)
    crop_coords_jax = jnp.array(coords_np)

    # NB desired farfield amplitude in the "knm" basis
    target_intensity = np.abs(hologram.target) ** 2

    # TODO
    target_extent = get_trap_zoom(hologram.target)
    x_min, x_max, y_min, y_max = target_extent

    plot_scalar_field(
        "./results/plots/gaussian_slm_illumination.pdf", 
        slm_illumination, 
        cbar_label="slm illumination"
    )

    plot_scalar_field(
        "./results/plots/target_intensity.pdf",
        target_intensity[y_min:y_max, x_min:x_max],
        extent=target_extent,
    )

    # NB callback definition,
    #    https://github.com/holodyne/slmsuite/blob/39243f081de020ad3ba74e672d126694b80778d2/slmsuite/holography/algorithms/_hologram.py#L1473
    hologram.optimize(
        method=solver_config.method,
        maxiter=solver_config.maxiter,
        stat_groups=["computational_spot"],
        verbose=False,
    )

    # NB get optimized slm phase and far-field intensity,
    #    crop to show only the central region.
    #
    # NB current nearfield phase from the GPU shifted to [0, 2*pi].
    slm_phase = hologram.get_phase()
    ff_int = np.abs(hologram.get_farfield()) ** 2

    stack_h, stack_w = 50, 50
    stack_mean_similar_traps = reduce_stack_similar_traps(
        ff_int, crop_coords_jax, stack_h, stack_w
    )
    # NB inter_uniformity=0.84553164
    trap_metrics = compute_trap_metrics(ff_int, trap_labels_jax, num_traps)

    # plot_trap_stack_mean(
    #     "./results/plots/trap_stack_mean.pdf",
    #     np.log(stack_mean_similar_traps + 1e-12),
    # )

    plot_scalar_field(
        "./results/plots/trap_stack_mean.pdf",
        np.log(stack_mean_similar_traps + 1e-12),
    )

    """
    artifacts, residual_int = extract_background_artifacts(
        ff_int, target_intensity, max_artifacts=9
    )

    artifact_stacks = crop_artifact_stacks(
        ff_int,
        artifacts,
        stack_h=stack_h,
        stack_w=stack_w,
    )

    artifacts = label_slm_fuzz(artifacts, target_extent)

    export_artifact_data_for_streamlit(
        "./results/data/artifact_data.pkl", 
        residual_int, 
        artifacts, 
        artifact_stacks, 
        slm_shape=run_config.slm_shape,
        array_shape=run_config.array_shape,
        array_pitch=run_config.array_pitch,
        array_center=run_config.array_center
    )
    """

    performance_metrics = compute_performance_metrics(ff_int, target_intensity)
    write_metrics_table("./results/tables/performance_metrics.tex", performance_metrics)

    pprint(performance_metrics)

    # cy, cx = ff_int.shape[0] // 2, ff_int.shape[1] // 2
    # half = 130
    # ff_crop = ff_int[cy - half : cy + half, cx - half : cx + half]

    # extent = [cx - half, cx + half, cy - half, cy + half],

    plot_phase_retrieval_results(
        "./results/plots/phase_retrieval_results.pdf",
        slm_phase,
        ff_int[y_min:y_max, x_min:x_max],
        target_extent,
    )

    # TODO stats etc.
    header = run_config.copy()

    # NB h5diff -d 1e-3 results/data/exercise_reference_gs_20260727_120804.h5 results/data/exercise_gs_20260727_120932.h5
    hdf5_path = f"./results/data/exercise_{solver_config.method.lower()}_{timestamp}.h5"

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
        **header,
    )

    # TODO better write of config.
    write_hdf5(
        filepath=hdf5_path,
        data=slm_phase,
        group_name="slm",
        dataset_name="slm_phase",
        wavelength=run_config.wavelength,
        pixel_pitch=run_config.pixel_pitch,
        maxiter=solver_config.maxiter,
    )

    write_hdf5(
        filepath=hdf5_path,
        data=ff_int,
        group_name=solver_config.method.lower(),
        dataset_name="inferred_farfield_intensity",
    )


# def main():
#     run_slmsuit_phase_retrieval()


# if __name__ == "__main__":
#     main()
