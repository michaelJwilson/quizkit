import atexit
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from aim import Run
from rich.pretty import pprint
from scipy.ndimage import binary_dilation, find_objects, label
from slmsuite.holography.algorithms import SpotHologram

from quizkit.configs import (ConfigMixin, RunConfig, SolverConfig,
                             TrapConfigs)
from quizkit.hologram_experiment import HologramExperiment
from quizkit.jax_holography import JaxHologramBackend
from quizkit.plotting import plot_scalar_field
from quizkit.writers import write_hdf5

import aim.ext.cleanup

# TODO HACK aim thread issue for python 3.12 (TBC)
atexit.unregister(aim.ext.cleanup.AutoClean.cleanup)

"""
GS algorithm application via slm suite, see

https://slmsuite.readthedocs.io/en/latest/_examples/computational_holography.html#Basic-Image-Formation
"""

# NB slmsuite seeds on random, not numpy (but potentially cupy/cuda).
random.seed(42)
np.random.seed(42)

start_time = time.time()


class RuntimeFormatter(logging.Formatter):
    def format(self, record):
        runtime_minutes = (time.time() - start_time) / 60.0
        record.runtime = f"{runtime_minutes:.2f}m"
        return super().format(record)


formatter = RuntimeFormatter(
    fmt="%(asctime)s - %(runtime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

for handler in logger.handlers[:]:
    logger.removeHandler(handler)

file_handler = logging.FileHandler("quizkit.log")
stream_handler = logging.StreamHandler()

file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

logger = logging.getLogger(__name__)

"""
def _add_colorbar(ax, im, label=None):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.figure.colorbar(im, cax=cax, label=label)


def plot_scalar_field(
    plot_path,
    field,
    cmap="inferno",
    title=None,
    extent=None,
    cbar_label=None,
    hide_ticks=False,
    figsize=(5, 3.2),
    xlabel=None,
    ylabel=None,
    **imshow_kwargs,
):
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

    logger.info(f"Writing {plot_path}.")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
"""


def write_performance_metrics_tex(
    filepath: str | Path,
    metrics_by_run: dict[str, dict],
    caption: str = "Computed performance metrics for the optimized SLM phase.",
    label: str = "tab:hologram_metrics",
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
        "signal_to_background_floor": "Ratio of mean trap power to mean background intensity",
    }

    run_keys = list(metrics_by_run.keys())
    n_runs = len(run_keys)

    # 1. Determine Column Headers and Tabular Alignment
    # Always use the provided key(s), capitalized
    headers = [f"\\textbf{{{key.capitalize()}}}" for key in run_keys]

    c_cols = "c" * n_runs
    tabular_def = f"\\begin{{tabular}}{{l{c_cols}p{{11.5cm}}}}"

    # 2. Build Header Row
    header_row = (
        " & ".join(["\\textbf{Metric Key}"] + headers + ["\\textbf{Description}"])
        + " \\\\"
    )

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

    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing {out_path}.")

    with open(out_path, "w") as f:
        f.write(latex)


"""
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
"""
"""
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
"""


def reduce_stack_similar_crops(
    forward_intensity,
    center_coords,
    stack_h,
    stack_w,
    weights=None,
    reducer=None,
):
    """Crops and optionally reduces a stack of sub-regions. Supports boundary-safe zero-padding."""
    pad_y, pad_x = stack_h // 2, stack_w // 2

    # Pad intensity with zeros so boundary crops don't clamp-shift in JAX
    padded_int = jnp.pad(
        forward_intensity, ((pad_y, pad_y), (pad_x, pad_x)), mode="constant"
    )

    def crop_single(coord):
        # Because we padded by the half-widths, the original center coordinate
        # maps perfectly to the starting index of the slice in the padded array.
        return jax.lax.dynamic_slice(
            padded_int, (coord[0], coord[1]), (stack_h, stack_w)
        )

    # Shape: (N, stack_h, stack_w)
    crop_stack = jax.vmap(crop_single)(center_coords)

    if reducer is None:
        return crop_stack

    if weights is not None:
        w = weights[:, None, None]
        if reducer in (jnp.mean, jnp.average):
            reduced_profile = jnp.sum(crop_stack * w, axis=0) / (jnp.sum(w) + 1e-12)
        else:
            raise NotImplementedError(
                "Weighted reduction is only implemented for mean/average."
            )
    else:
        reduced_profile = reducer(crop_stack, axis=0)

    return reduced_profile


def extract_background_artifacts(
    forward_intensity,
    trap_labels,
    exclusion_mask=None,
    exclusion_pad=15,
    percentile_q=99.9,
    max_artifacts=100,
):
    """
    Isolates background speckle/ghost traps using percentile thresholding,
    excluding regions around intended traps.
    """
    if exclusion_pad > 0:
        struct = np.ones((exclusion_pad * 2 + 1, exclusion_pad * 2 + 1), dtype=bool)
        trap_mask = binary_dilation(trap_labels > 0, structure=struct)
    else:
        trap_mask = trap_labels > 0

    if exclusion_mask is not None:
        trap_mask = np.logical_or(trap_mask, exclusion_mask)

    bg_mask = ~trap_mask
    residual_int = forward_intensity * bg_mask

    bg_pixels = residual_int[bg_mask]
    if len(bg_pixels) == 0:
        return [], residual_int

    artifact_threshold = np.percentile(bg_pixels, percentile_q)
    binary_artifacts = residual_int > artifact_threshold

    labeled_artifacts, _ = label(binary_artifacts)
    slices = find_objects(labeled_artifacts)

    artifacts = []
    for i, s in enumerate(slices):
        cy = (s[0].start + s[0].stop) // 2
        cx = (s[1].start + s[1].stop) // 2

        comp_mask = labeled_artifacts[s] == (i + 1)
        comp_power = np.sum(residual_int[s][comp_mask])

        artifacts.append({"id": i, "cy": cy, "cx": cx, "power": comp_power})

    artifacts.sort(key=lambda x: x["power"], reverse=True)
    artifacts = artifacts[:max_artifacts]

    return artifacts, residual_int


"""
def get_trap_array_mask(slm_shape, array_shape, array_pitch, array_center, pad=25):
    if array_center is not None:
        center_x, center_y = array_center[0], array_center[1]
    else:
        center_y, center_x = slm_shape[0] / 2.0, slm_shape[1] / 2.0

    array_extent = (np.array(array_shape) - 1) * np.array(array_pitch)

    x_min = int(np.floor(center_x - array_extent[1] / 2.0 - pad))
    x_max = int(np.ceil(center_x + array_extent[1] / 2.0 + pad))
    y_min = int(np.floor(center_y - array_extent[0] / 2.0 - pad))
    y_max = int(np.ceil(center_y + array_extent[0] / 2.0 + pad))

    x_min_clamped = max(0, x_min)
    x_max_clamped = min(slm_shape[1], x_max)
    y_min_clamped = max(0, y_min)
    y_max_clamped = min(slm_shape[0], y_max)

    trap_array_mask = np.zeros(slm_shape, dtype=bool)
    trap_array_mask[y_min_clamped:y_max_clamped, x_min_clamped:x_max_clamped] = True

    return trap_array_mask
"""
"""
class ConfigMixin:
    def to_dict(self) -> dict:
        data = asdict(self)
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                data[key] = value.tolist()
        return data

    def to_json(self, indent: int = 4) -> str:
        return json.dumps(self.to_dict(), indent=indent)


@dataclass
class TrapConfig(ConfigMixin):
    trap_id: int
    trap_type: str
    array_shape: Tuple[int, int]
    array_pitch: Tuple[int, int]
    array_center: Optional[Tuple[float, float]] = None


# TODO
class TrapConfigs(Enum):
    # NB format=(trap_type, array_shape, array_pitch, array_center)
    ON_AXIS = (0, "on_axis", (10, 10), (20, 20), None)
    OFF_AXIS = (1, "off_axis", (10, 10), (20, 20), (3.0 * 1920 / 4, 2.0 * 1200 / 4))

    @property
    def id(self) -> int:
        return self.value[0]

    def to_config(self) -> TrapConfig:
        config_id, trap_type, array_shape, array_pitch, array_center = self.value
        return TrapConfig(
            trap_id=self.id,
            trap_type=trap_type,
            array_shape=array_shape,
            array_pitch=array_pitch,
            array_center=array_center,
        )


@dataclass
class RunConfig(ConfigMixin):
    wavelength: float
    pixel_pitch: float
    slm_shape: Tuple[int, int]
    trap_config: TrapConfig
    comment: Optional[str] = None

    hash: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    def __getattr__(self, name):
        try:
            return getattr(self.trap_config, name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )


@dataclass
class SolverConfig(ConfigMixin):
    method: str  # {"GS", "WGS", "GD"}
    maxiter: int = 200

    solver_backend: str | None = None
    solver_runtime: float | None = None

    smooth_phase: bool = False
    smooth_sigma: int = 3  # pixels

    loss_norm: str = "L2"  # {"L1", "L2"}
    learning_rate: float = 0.1

    random_seed: int = 42

    # TODO HACK
    initial_epsilon: float = 0.0
    anneal_rate: float = 0.05

    hash: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
"""
"""
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
            amp=self.slm_illumination.copy(),
            array_center=self.run_config.array_center,
            phase=np.random.uniform(-np.pi, np.pi, self.run_config.slm_shape),
        )

        self.target = hologram.target.copy()
        self.target.flags.writeable = False

        self.trap_array_mask = get_trap_array_mask(
            self.run_config.slm_shape,
            self.run_config.array_shape,
            self.run_config.array_pitch,
            self.run_config.array_center,
            pad=25,
        )
        self.trap_array_mask.flags.writeable = False

        trap_labels_np, self.num_traps, coords_np, self.trap_h, self.trap_w = (
            encode_target_traps(self.target, threshold_frac=0.0)
        )

        self.trap_labels = jnp.array(trap_labels_np)
        self.trap_coords = jnp.array(coords_np)

        # TODO DEPRECATE
        self.crop_coords = self.trap_coords

        self._is_frozen = True

    @classmethod
    def from_run_h5(cls, run_dir: str | Path):
        run_dir = Path(run_dir)

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
            obj.trap_array_mask = f["trap_array_mask/trap_array_mask"][:]

        obj.slm_illumination.flags.writeable = False
        obj.target.flags.writeable = False

        trap_labels_np, obj.num_traps, coords_np, obj.trap_h, obj.trap_w = (
            encode_target_traps(obj.target, threshold_frac=0.0)
        )

        obj.trap_labels = jnp.array(trap_labels_np)
        obj.trap_coords = jnp.array(coords_np)
        obj.trap_array_mask = jnp.array(obj.trap_array_mask)
        obj.crop_coords = obj.trap_coords

        obj._is_frozen = True

        logger.info(f"Loaded HologramExperiment from {run_dir}.")

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

    def _get_run_dir(self, base_dir: str) -> Path:
        return Path(base_dir) / f"run_{self.run_config.hash}"

    def plot(self, base_dir: str = "./results"):
        run_dir = self._get_run_dir(base_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "plots").mkdir(parents=True, exist_ok=True)

        plot_scalar_field(
            run_dir / "plots" / "gaussian_slm_illumination.pdf",
            np.log(self.slm_illumination + 1e-12),
            cbar_label="ln. slm illumination",
        )

        x_min, x_max, y_min, y_max = self.target_extent
        plot_scalar_field(
            run_dir / "plots" / "target_intensity.pdf",
            np.log(self.target_intensity[y_min:y_max, x_min:x_max] + 1e-12),
            extent=self.target_extent,
            cbar_label="ln. target intensity",
        )

    def write_h5(self, base_dir: str = "./results"):
        run_dir = self._get_run_dir(base_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Writing run configuration to {run_dir / 'run_config.json'}.")

        with open(run_dir / "run_config.json", "w") as f:
            f.write(self.run_config.to_json())

        hdf5_path = run_dir / "experiment.h5"

        logger.info(f"Writing hologram experiment to {hdf5_path}.")

        write_hdf5(
            filepath=hdf5_path,
            data=self.slm_illumination,
            group_name="slm",
            dataset_name="slm_illumination",
        )

        write_hdf5(
            filepath=hdf5_path,
            data=self.target,
            group_name="target",
            dataset_name="target",
        )

        write_hdf5(
            filepath=hdf5_path,
            data=self.trap_array_mask,
            group_name="trap_array_mask",
            dataset_name="trap_array_mask",
        )
"""


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
    trap_powers: np.ndarray = field(repr=False)

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

        trap_powers_jax = jnp.bincount(
            flat_trap_labels,
            weights=flat_forward_intensity,
            length=num_traps + 1,
        )[1:]
        trap_powers = np.asarray(trap_powers_jax, dtype=np.float64)

        trap_min = float(np.min(trap_powers))
        trap_max = float(np.max(trap_powers))
        trap_mean = float(np.mean(trap_powers))
        trap_std = float(np.std(trap_powers))

        trap_cv = float(trap_std / (trap_mean + 1e-12))

        total_power = float(np.sum(flat_forward_intensity))
        sig_power = float(np.sum(trap_powers))
        bg_power = total_power - sig_power

        bg_mask = flat_trap_labels == 0
        bg_intensities = flat_forward_intensity[bg_mask]
        max_bg = float(np.max(bg_intensities))
        mean_bg = float(np.mean(bg_intensities))

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
            trap_uniformity_minmax=1.0
            - ((trap_max - trap_min) / (trap_max + trap_min + 1e-12)),
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
        self.backend_type = (self.config.solver_backend or "slm_suite").lower()

        # TODO
        self.stack_h = 50
        self.stack_w = 50

        rng = np.random.default_rng(seed=self.config.random_seed)
        self.init_phases = rng.uniform(
            -np.pi, np.pi, size=self.exp.run_config.slm_shape
        )

        if self.backend_type == "slm_suite":
            self.backend = SpotHologram.make_rectangular_array(
                self.exp.run_config.slm_shape,
                array_shape=self.exp.run_config.array_shape,
                array_pitch=self.exp.run_config.array_pitch,
                basis="knm",
                amp=self.exp.slm_illumination.copy(),
                array_center=self.exp.run_config.array_center,
                phase=self.init_phases,
            )

            # TODO
            # assert np.allclose(self.exp.target, self.__hologram.target)

        elif self.backend_type == "jax":
            self.backend = JaxHologramBackend(self.exp, self.config)
        else:
            raise ValueError(f"Unknown backend: {self.backend_type}")

    def optimize(self):
        logger.info(
            f"Solving the phase retrieval problemm with {self.config.method} & {self.config.maxiter} iterations."
        )
        start_time = time.time()

        if self.backend_type == "slm_suite":
            self.backend.optimize(
                method=self.config.method,
                maxiter=self.config.maxiter,
                stat_groups=["computational_spot"],
                verbose=False,
            )
        elif self.backend_type == "jax":
            self.backend.optimize()

        self.config.solver_runtime = time.time() - start_time

        logger.info(
            f"Solved phase retrieval problem in {self.config.solver_runtime:.2f}s"
        )

    def log_to_aim(self, experiment_name: str = "phase_retrieval_sweep"):
        if self.backend_type != "jax":
            logger.warning(
                "Aim logging currently only supports the JAX backend history."
            )
            return

        run = Run(experiment=experiment_name)

        # Flatten configs for Aim hparams
        run["hparams"] = {
            "run_config": self.exp.run_config.to_dict(),
            "solver_config": self.config.to_dict(),
        }

        history = self.backend.history

        for step_idx in range(self.config.maxiter):
            for metric_name, metric_array in history.items():
                run.track(
                    metric_array[step_idx].item(),
                    name=metric_name,
                    step=step_idx,
                    context={"subset": "Metrics"},
                )

        run.close()
        logger.info(f"Aim run closed for {self.config.hash}")

    @property
    def target_intensity(self):
        return self.exp.target_intensity

    @property
    def target_extent(self):
        return self.exp.target_extent

    @property
    def slm_phase(self):
        if self.backend_type == "slm_suite":
            return self.backend.get_phase()
        return self.backend.final_phase

    @property
    def forward_intensity(self):
        if self.backend_type == "slm_suite":
            return np.abs(self.backend.get_farfield()) ** 2
        return self.backend.final_intensity

    def __get_run_dir(self, base_dir: str) -> Path:
        return self.exp._get_run_dir(base_dir) / f"phase_retrieval/{self.config.hash}"

    def plot(self, base_dir: str = "./results"):
        out_dir = self.__get_run_dir(base_dir)
        plot_dir = out_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        plot_scalar_field(
            plot_path=plot_dir / "slm_phase.pdf",
            field=self.slm_phase,
            cmap="twilight",
            title="slm",
            cbar_label="phase [rad]",
            xlabel=r"$x [\Delta]$",
            ylabel=r"$y [\Delta]$",
            interpolation="nearest",
        )

        stack_mean_similar_traps = reduce_stack_similar_crops(
            self.forward_intensity,
            self.exp.trap_coords,
            self.stack_h,
            self.stack_w,
            reducer=jnp.mean,
        )

        plot_scalar_field(
            plot_path=plot_dir / "trap_stack_forward_intensity.pdf",
            field=np.log(stack_mean_similar_traps + 1e-12),
            cmap="inferno",
            title="trap stack: mean",
            cbar_label="ln. intensity [a.u.]",
            xlabel=r"$k_n$ [knm]",
            ylabel=r"$k_m$ [knm]",
        )

        x_min, x_max, y_min, y_max = self.target_extent
        ff_int = self.forward_intensity

        plot_scalar_field(
            plot_path=plot_dir / "farfield_intensity.pdf",
            field=ff_int[y_min:y_max, x_min:x_max]
            / ff_int[y_min:y_max, x_min:x_max].max(),
            cmap="inferno",
            title="far field",
            extent=self.target_extent,
            cbar_label="ln. intensity [a.u.]",
            xlabel=r"$k_n$ [knm]",
            ylabel=r"$k_m$ [knm]",
            origin="lower",
        )

    def extract_off_target_intensity(self):
        # TODO
        # exclusion_mask=~self.exp.trap_array_mask,
        exclusion_mask = None

        artifacts, _ = extract_background_artifacts(
            self.forward_intensity,
            self.exp.trap_labels,
            exclusion_pad=15,
            exclusion_mask=exclusion_mask,
            percentile_q=99.9,
        )

        artifact_coords = jnp.array([[art["cy"], art["cx"]] for art in artifacts])
        artifact_stacks = reduce_stack_similar_crops(
            self.forward_intensity,
            artifact_coords,
            stack_h=self.stack_h,
            stack_w=self.stack_w,
            reducer=None,  # Skips reduction, returns (N, H, W)
        )

        return artifact_stacks

    def write_h5(self, base_dir: str = "./results"):
        out_dir = self.__get_run_dir(base_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Writing solver configuration to {out_dir / 'solver_config.json'}."
        )

        with open(out_dir / "solver_config.json", "w") as f:
            f.write(self.config.to_json())

        logger.info(
            f"Writing performance metrics to {out_dir / 'performance_metrics.json'}."
        )

        metrics = PerformanceMetrics.from_solver(self)

        with open(out_dir / "performance_metrics.json", "w") as f:
            f.write(metrics.to_json())

        hdf5_path = out_dir / f"phase_solution_{self.config.hash}.h5"
        logger.info(f"Writing HologramExperimentSovler results to {hdf5_path}.")

        write_hdf5(
            filepath=hdf5_path,
            data=self.slm_phase,
            group_name="slm",
            dataset_name=f"slm_phase_{self.config.method.lower()}",
        )

        write_hdf5(
            filepath=hdf5_path,
            data=self.forward_intensity,
            group_name="target",
            dataset_name="inferred_farfield_intensity",
        )

        artefact_stacks = self.extract_off_target_intensity()

        write_hdf5(
            filepath=hdf5_path,
            data=artefact_stacks,
            group_name="background",
            dataset_name="off_target_intensity_stacks",
        )


def run_phase_retrieval():
    slm_shape = (1200, 1920)  # (height, width) in pixels,

    # NB (float, float) or None; shift from zeroth order in the far-field basis. If None, defaults to the zeroth order position.
    #    see https://github.com/holodyne/slmsuite/blob/39243f081de020ad3ba74e672d126694b80778d2/slmsuite/holography/algorithms/_spots.py#L1423
    #
    # `"knm"``, this is ``(shape[1], shape[0])/2``.
    # ``"kxy"``, this is ``(0,0)``.
    # ``"ij"``, this is the pixel position of the zeroth order on the camera (via Fourier calibration).
    trap_config = TrapConfigs.ON_AXIS.to_config()
    trap_config_off_center = TrapConfigs.OFF_AXIS.to_config()

    # pprint(trap_config, expand_all=True)
    # pprint(trap_config_off_center, expand_all=True)

    run_config = RunConfig(
        wavelength=780e-9,
        pixel_pitch=8.0e-6,
        slm_shape=slm_shape,
        trap_config=trap_config,
        comment="default slm suite run",
    )

    pprint(run_config, expand_all=True)

    # NB h5diff -d 1e-3 results/data/exercise_reference_gs_20260727_120804.h5 results/data/exercise_gs_20260727_120932.h5
    exp = HologramExperiment(run_config)
    exp.plot(base_dir="./results")
    exp.write_h5(base_dir="./results")

    for random_seed in (42,):
        solver_config = SolverConfig(
            method="GD",
            maxiter=200,
            random_seed=random_seed,
            solver_backend="jax",  # {"slm_suite", "jax"}
        )

        pprint(solver_config, expand_all=True)

        solver = HologramExperimentSolver(exp, solver_config)
        solver.optimize()
        solver.plot(base_dir="./results")
        solver.write_h5(base_dir="./results")

        # TODO
        # solver.log_to_aim(experiment_name="dummy")

        metrics = PerformanceMetrics.from_solver(solver)

        pprint(metrics, expand_all=True)

        write_performance_metrics_tex(
            filepath=exp._get_run_dir("./results")
            / f"phase_retrieval/{solver.config.timestamp}"
            / "performance_metrics.tex",
            metrics_by_run={solver.config.method: metrics.to_dict()},
            caption=f"Computed performance metrics for the {solver.config.method}-optimized SLM phase.",
        )

    logger.info(f"Done.")


def main():
    run_phase_retrieval()


if __name__ == "__main__":
    main()
