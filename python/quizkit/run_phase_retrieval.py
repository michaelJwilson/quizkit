import atexit
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from aim import Run, Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rich.pretty import pprint
from scipy.ndimage import binary_dilation, find_objects, label
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve
from slmsuite.holography.algorithms import SpotHologram

from quizkit.configs import ConfigMixin, RunConfig, SolverConfig, TrapConfigs
from quizkit.hologram_experiment import HologramExperiment
from quizkit.jax_holography import JaxHologramBackend
from quizkit.plotting import plot_scalar_field, plot_stack_with_marginals
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

def sinc2_2d(coords, I0, x0, y0, wx, wy, bg):
    x_val, y_val = coords
    wx, wy = np.maximum(abs(wx), 1e-9), np.maximum(abs(wy), 1e-9)
    sinc2_x = np.sinc((x_val - x0) / wx) ** 2
    sinc2_y = np.sinc((y_val - y0) / wy) ** 2
    return I0 * sinc2_x * sinc2_y + bg

def gaussian_2d(coords, I0, x0, y0, wx, wy, bg):
    x_val, y_val = coords
    wx, wy = np.maximum(abs(wx), 1e-9), np.maximum(abs(wy), 1e-9)
    r2 = ((x_val - x0) / wx) ** 2 + ((y_val - y0) / wy) ** 2
    return I0 * np.exp(-0.5 * r2) + bg

def lorentzian_2d(coords, I0, x0, y0, wx, wy, bg):
    # Separable 2D Lorentzian (better matches rectangular aperture cross-wings)
    x_val, y_val = coords
    wx, wy = np.maximum(abs(wx), 1e-9), np.maximum(abs(wy), 1e-9)
    lx = 1.0 / (1.0 + ((x_val - x0) / wx) ** 2)
    ly = 1.0 / (1.0 + ((y_val - y0) / wy) ** 2)
    return I0 * lx * ly + bg

def moffat_2d(coords, I0, x0, y0, wx, wy, bg):
    # Elliptical Moffat profile (beta=2.5 is standard for optical turbulence/scattering)
    x_val, y_val = coords
    wx, wy = np.maximum(abs(wx), 1e-9), np.maximum(abs(wy), 1e-9)
    r2 = ((x_val - x0) / wx) ** 2 + ((y_val - y0) / wy) ** 2
    return I0 * (1.0 + r2) ** (-2.5) + bg

psf_models = {
    "sinc2": sinc2_2d,
    "gaussian": gaussian_2d,
    "lorentzian": lorentzian_2d,
    "moffat": moffat_2d,
}

def fit_forward_psf(z_data, model="gaussian", fit_background=True):
    """
    Fits a 2D intensity stack to a chosen PSF model, returning parameters and 1D marginals.
    Models support independent (asymmetric) wx and wy.

    Supported models: 'sinc2', 'gaussian', 'lorentzian', 'moffat'
    """
    h, w = z_data.shape
    cy, cx = h // 2, w // 2

    if model not in psf_models:
        raise ValueError(
            f"Unknown PSF model '{model}'. Available: {list(psf_models.keys())}"
        )

    fit_func = psf_models[model]

    core_radius = 10

    y_slice = slice(max(0, cy - core_radius), min(h, cy + core_radius + 1))
    x_slice = slice(max(0, cx - core_radius), min(w, cx + core_radius + 1))

    z_fit = z_data[y_slice, x_slice]

    y_coords = np.arange(h)[y_slice]
    x_coords = np.arange(w)[x_slice]
    X_fit, Y_fit = np.meshgrid(x_coords, y_coords)

    bg_guess = 0.0
    I0_guess = max(0.0, np.max(z_fit) - bg_guess)
    x0_guess, y0_guess = float(cx), float(cy)

    # NB slm aperture expected to produce a spot with a first-null width of ~1 pixel
    wx_guess, wy_guess = 1.0, 1.0

    coords_fit = (X_fit.ravel(), Y_fit.ravel())

    if fit_background:
        p0 = [I0_guess, x0_guess, y0_guess, wx_guess, wy_guess, bg_guess]
        popt, _ = curve_fit(fit_func, coords_fit, z_fit.ravel(), p0=p0)
        I0, x0, y0, wx, wy, bg = popt
    else:
        # Wrap the function to hide the 'bg' parameter from the optimizer
        def fit_func_no_bg(coords, _I0, _x0, _y0, _wx, _wy):
            return fit_func(coords, _I0, _x0, _y0, _wx, _wy, 0.0)

        p0 = [I0_guess, x0_guess, y0_guess, wx_guess, wy_guess]
        popt_5, _ = curve_fit(fit_func_no_bg, coords_fit, z_fit.ravel(), p0=p0)

        I0, x0, y0, wx, wy = popt_5
        bg = 0.0
        # Reconstruct standard 6-parameter tuple for return signature
        popt = (I0, x0, y0, wx, wy, bg)

    x_line = np.arange(w)
    y_line = np.arange(h)

    # Always evaluate marginals cleanly without the background component
    fit_x_no_bg = fit_func((x_line, np.full_like(x_line, y0)), I0, x0, y0, wx, wy, 0.0)
    fit_y_no_bg = fit_func((np.full_like(y_line, x0), y_line), I0, x0, y0, wx, wy, 0.0)

    return popt, fit_x_no_bg, fit_y_no_bg


def calculate_diffuse_efficiencies(
    forward_intensity: np.ndarray,
    target: np.ndarray,
    psf_model: str,
    psf_params: tuple,
    array_center: tuple[int, int] | None = None,
) -> tuple[float, float]:
    if psf_model not in psf_models:
        raise ValueError(f"Unknown PSF model '{psf_model}'. Available: {list(psf_models.keys())}")
        
    fit_func = psf_models[psf_model]
    
    _, _, _, wx, wy, _ = psf_params
    
    h, w = forward_intensity.shape
    total_power = float(np.sum(forward_intensity))

    # TODO
    stack_h, stack_w = 50, 50
    ky, kx = np.arange(-stack_h // 2, stack_h // 2), np.arange(-stack_w // 2, stack_w // 2)
    X, Y = np.meshgrid(kx, ky)
    
    psf_kernel = fit_func((X, Y), 1.0, 0.0, 0.0, wx, wy, 0.0)
    
    target_sum = np.sum(target)
    normalized_target = target * (total_power / target_sum) if target_sum > 0 else target

    diffuse_target = fftconvolve(normalized_target, psf_kernel, mode='same')
    
    # TODO
    diffuse_target = np.clip(diffuse_target, 0.0, 1.0)

    efficiency = float(np.sum(forward_intensity * normalized_target) / (total_power + 1e-12))
    efficiency_diffuse = float(np.sum(forward_intensity * diffuse_target) / (total_power + 1e-12))
    
    # -------------------------------------------------------------------------
    # 2. 0th Order PSF Efficiency
    # -------------------------------------------------------------------------
    # Generate a peak-normalized envelope perfectly centered in the far field
    if array_center is not None:
        cy, cx = array_center
    else:
        cy, cx = h // 2, w // 2

    Y_full, X_full = np.ogrid[:h, :w]

    # TODO
    zeroth_mask = fit_func((X_full, Y_full), 1.0, cx, cy, wx, wy, 0.0)    
    efficiency_zeroth = float(np.sum(forward_intensity * zeroth_mask) / (total_power + 1e-12))
    
    return efficiency, efficiency_diffuse, efficiency_zeroth


@dataclass
class PerformanceMetrics(ConfigMixin):
    _DESCRIPTIONS: ClassVar[dict[str, str]] = {
        "uniformity": "Michelson uniformity of integrated trap powers",
        "efficiency": "Fraction of forward power within the target trap regions",
        "efficiency_diffuse": "Fraction of forward power within the (best-fit) forward psf target trap regions (TBD)",
        "stray_light_fraction": "Fraction of forward power outside the target trap regions (1 - efficiency)",
        "efficiency_perimeter": "Fraction of forward power within the trap array perimeter",
        "efficiency_dual": "Fraction of forward power within the dual array",
        "efficiency_zeroth": "Fraction of forward power located at the diffuse zeroth order",
        "pearson": "Pearson correlation of forward intensity and target intensity",
        "trap_med": "Median integrated trap power",
        "trap_mean": "Mean integrated trap power",
        "trap_std": "Standard deviation of integrated trap power",
        "trap_min": "Minimum integrated trap power",
        "trap_max": "Maximum integrated trap power",
        "ghost_to_trap_med_ratio": "Ratio of max background intensity to median trap power",
        "runtime": "Time taken to solve the phase retrieval problem [seconds]"
    }

    uniformity: float
    efficiency: float
    efficiency_diffuse: float
    stray_light_fraction: float
    efficiency_perimeter: float
    efficiency_dual: float
    efficiency_zeroth: float
    pearson: float
    trap_med: float
    trap_mean: float
    trap_std: float
    trap_min: float
    trap_max: float
    ghost_to_trap_med_ratio: float
    runtime: float | None
    trap_powers: np.ndarray = field(repr=False)

    @classmethod
    def compute(
        cls,
        forward_intensity: np.ndarray,
        target_intensity: np.ndarray,
        trap_labels: jnp.ndarray | np.ndarray,
        trap_coords: np.ndarray,
        trap_array_perimeter_mask: jnp.ndarray | np.ndarray,
        num_traps: int,
        dual_mask: np.ndarray,
        psf_model: str | None = None,
        psf_params: tuple | None = None,
        array_center: tuple[int, int] | None = None,
        runtime: float | None = None
    ) -> "PerformanceMetrics":
        ff_int = np.asarray(forward_intensity, dtype=np.float64)
        target_int = np.asarray(target_intensity, dtype=np.float64)
        trap_array_perimeter_mask = np.asarray(trap_array_perimeter_mask, dtype=bool)

        flat_forward_intensity = ff_int.ravel()
        flat_trap_labels = np.asarray(trap_labels).ravel()
        total_power = float(np.sum(flat_forward_intensity))

        trap_powers_jax = jnp.bincount(
            flat_trap_labels,
            weights=flat_forward_intensity,
            length=num_traps + 1,
        )[1:]
        trap_powers = np.asarray(trap_powers_jax, dtype=np.float64)

        trap_min = float(np.min(trap_powers))
        trap_max = float(np.max(trap_powers))
        trap_med = float(np.median(trap_powers))
        trap_mean = float(np.mean(trap_powers))
        trap_std = float(np.std(trap_powers))

        total_trap_perimeter_power = float(np.sum(ff_int[trap_array_perimeter_mask]))
        # total_trap_dual_power = float(np.sum(ff_int[dual_mask]))

        sig_power = float(np.sum(trap_powers))
        bg_power = total_power - sig_power

        efficiency_perimeter = total_trap_perimeter_power / (total_power + 1e-12)

        dual_power = float(np.sum(ff_int[dual_mask]))
        efficiency_dual = dual_power / (total_power + 1e-12)

        bg_mask = flat_trap_labels == 0
        bg_intensities = flat_forward_intensity[bg_mask]
        max_bg = float(np.max(bg_intensities))

        ff_centered = ff_int - np.mean(flat_forward_intensity)
        target_centered = target_int - np.mean(target_int)
        numerator = np.sum(ff_centered * target_centered)
        denominator = np.sqrt(np.sum(ff_centered**2) * np.sum(target_centered**2))
        pearson = float(numerator / (denominator + 1e-12))

        # TODO should be peak in a trap; rather than integrated.
        uniformity = 1.0 - ((trap_max - trap_min) / (trap_max + trap_min + 1e-12))
        ghost_to_trap_med_ratio = max_bg / (trap_med + 1e-12)

        efficiency = sig_power / total_power

        # TODO HACK? _, ...
        _, efficiency_diffuse, efficiency_zeroth = calculate_diffuse_efficiencies(
            forward_intensity=ff_int,
            target=target_int,
            psf_model=psf_model,
            psf_params=psf_params,
            array_center=array_center,
        )

        return cls(
            uniformity=uniformity,
            efficiency=efficiency,
            efficiency_diffuse=efficiency_diffuse,
            efficiency_zeroth=efficiency_zeroth,
            stray_light_fraction=bg_power / total_power,
            efficiency_perimeter=efficiency_perimeter,
            efficiency_dual=efficiency_dual,
            pearson=pearson,
            trap_med=trap_med,
            trap_mean=trap_mean,
            trap_std=trap_std,
            trap_min=trap_min,
            trap_max=trap_max,
            ghost_to_trap_med_ratio=ghost_to_trap_med_ratio,
            runtime=runtime,
            trap_powers=trap_powers,
        )

    @classmethod
    def from_solver(cls, solver: Any) -> "PerformanceMetrics":
        return cls.compute(
            forward_intensity=solver.forward_intensity,
            target_intensity=solver.target_intensity,
            trap_labels=solver.exp.trap_labels,
            trap_coords=np.array(solver.exp.trap_coords),
            trap_array_perimeter_mask=solver.exp.trap_array_perimeter_mask,
            num_traps=solver.exp.num_traps,
            dual_mask=solver.exp.dual_mask,
            psf_model=solver.forward_psf_model,
            psf_params=solver.forward_psf_model_params,
            array_center=solver.exp.run_config.array_center,
            runtime=solver.config.runtime,
        )
    
    @classmethod
    def write_tex_table(
        cls,
        filepath: str | Path,
        metrics_by_run: dict[str, dict],
        caption: str = "Computed performance metrics for the optimized SLM phase.",
        label: str = "tab:hologram_metrics",
    ) -> None:
        run_keys = list(metrics_by_run.keys())
        if not run_keys:
            logger.warning("No metrics provided to write_tex_table.")
            return

        n_runs = len(run_keys)

        headers = [f"\\textbf{{{key.capitalize()}}}" for key in run_keys]
        c_cols = "c" * n_runs
        tabular_def = f"\\begin{{tabular}}{{l{c_cols}p{{11.5cm}}}}"

        header_row = (
            " & ".join(["\\textbf{Metric Key}"] + headers + ["\\textbf{Description}"])
            + " \\\\"
        )

        metric_names = [
            k for k in metrics_by_run[run_keys[0]].keys() if k != "trap_powers"
        ]

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

            desc = cls._DESCRIPTIONS.get(m_name, "")
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
            f"Solving the phase retrieval problem with {self.config.method} & {self.config.maxiter} iterations."
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

        self.config.runtime = time.time() - start_time

        logger.info(
            f"Solved phase retrieval problem in {self.config.runtime:.2f}s"
        )

        forward_stack_mean = np.asarray(
            reduce_stack_similar_crops(
                self.forward_intensity,
                self.exp.trap_coords,
                self.stack_h,
                self.stack_w,
                reducer=jnp.mean,
            )
        )

        # TODO
        self.forward_psf_model = "gaussian"
        forward_psf_model_params, forward_psf_xprofile, forward_psf_yprofile = fit_forward_psf(
            forward_stack_mean, model=self.forward_psf_model, fit_background=False
        )

        self.forward_psf_model_params = forward_psf_model_params
        self.forward_psf_xprofile = forward_psf_xprofile
        self.forward_psf_yprofile = forward_psf_yprofile

    # TODO BUG
    def update_aim(self, experiment_name: str):
        if not hasattr(self.backend, "history"):
            logger.warning(
                "Backend does not support history attribute. Skipping aim metric tracking."
            )

        run = Run(experiment=experiment_name)

        # TODO Run Params
        run["hparams"] = {
            "trap": self.exp.run_config.trap_config.to_dict(),
            "run": self.exp.run_config.to_dict(),
            "solver": self.config.to_dict(),
        }

        if hasattr(self.backend, "history"):
            for step_idx in range(self.config.maxiter):
                for metric_name, metric_array in self.backend.history.items():
                    run.track(
                        float(metric_array[step_idx]),
                        name=metric_name,
                        step=step_idx,
                        context={"subset": "Metrics"},
                    )
        else:
            logger.warning(
                "Backend does not support history attribute. Skipping aim metric tracking."
            )

        # TODO
        # fig = plot_scalar_field(..., return_fig=True)
        # run.track(Figure(fig), name="slm_phase", context={"type": "final_state"})
        # plt.close(fig)

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

        # NB target intensity trap stack
        target_stack_mean = reduce_stack_similar_crops(
            self.target_intensity,
            self.exp.trap_coords,
            self.stack_h,
            self.stack_w,
            reducer=jnp.mean,
        )

        plot_stack_with_marginals(
            plot_path=plot_dir / "trap_stack_target_intensity.pdf",
            field=target_stack_mean,  # Linear scale usually better for pure target
            cmap="viridis",
            title="target trap stack: mean",
            cbar_label="Intensity [a.u.]",
            xlabel=r"$k_n$ [knm]",
            ylabel=r"$k_m$ [knm]",
        )

        # NB forward intensity trap stack
        forward_stack_mean = np.asarray(
            reduce_stack_similar_crops(
                self.forward_intensity,
                self.exp.trap_coords,
                self.stack_h,
                self.stack_w,
                reducer=jnp.mean,
            )
        )

        # NB
        # LATEST_RUN=$(ls -td results/run_* | head -n 1) && LATEST_SOLVER=$(ls -td "$LATEST_RUN"/phase_retrieval/*/ | head -n 1) && echo "${LATEST_SOLVER}plots/trap_stack_forward_intensity.pdf"

        plot_stack_with_marginals(
            plot_path=plot_dir / "trap_stack_forward_intensity.pdf",
            field=np.log(forward_stack_mean + 1e-12),
            cmap="inferno",
            title="forward trap stack: mean",
            cbar_label="ln. intensity [a.u.]",
            xlabel=r"$k_n$ [knm]",
            ylabel=r"$k_m$ [knm]",
            fit_x=np.log(self.forward_psf_xprofile + 1e-12),
            fit_y=np.log(self.forward_psf_yprofile + 1e-12),
        )

        # NB forward intensity trap stack std. dev.
        forward_stack_std = reduce_stack_similar_crops(
            self.forward_intensity,
            self.exp.trap_coords,
            self.stack_h,
            self.stack_w,
            reducer=jnp.std,
        )

        plot_stack_with_marginals(
            plot_path=plot_dir / "trap_stack_forward_intensity_std.pdf",
            field=np.log(forward_stack_std + 1e-12),
            cmap="inferno",
            title="forward trap stack: std. dev.",
            cbar_label="ln. std dev [a.u.]",
            xlabel=r"$k_n$ [knm]",
            ylabel=r"$k_m$ [knm]",
        )

        # NB forward intensity dual trap stack
        dual_stack_mean = reduce_stack_similar_crops(
            self.forward_intensity,
            self.exp.dual_coords,
            self.stack_h,
            self.stack_w,
            reducer=jnp.mean,
        )

        plot_stack_with_marginals(
            plot_path=plot_dir / "dual_trap_stack_forward.pdf",
            field=np.log(dual_stack_mean + 1e-12),
            cmap="inferno",
            title="forward dual trap stack: mean",
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
        # exclusion_mask=~self.exp.trap_array_perimeter_mask,
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
    method = "GS"
    solver_backend = "jax"  # {"slm_suite", "jax"}
    num_random_seeds = 1
    slm_shape = (1200, 1920)  # (height, width) in pixels,

    # NB (float, float) or None; shift from zeroth order in the far-field basis. If None, defaults to the zeroth order position.
    #    see https://github.com/holodyne/slmsuite/blob/39243f081de020ad3ba74e672d126694b80778d2/slmsuite/holography/algorithms/_spots.py#L1423
    #
    # `"knm"``, this is ``(shape[1], shape[0])/2``.
    # ``"kxy"``, this is ``(0,0)``.
    # ``"ij"``, this is the pixel position of the zeroth order on the camera (via Fourier calibration).
    trap_config = TrapConfigs.ON_AXIS.to_config()
    trap_config_off_center = TrapConfigs.OFF_AXIS.to_config()

    trap_configs = (trap_config,)

    for trap_config in trap_configs:
        run_config = RunConfig(
            wavelength=780e-9,
            pixel_pitch=8.0e-6,
            slm_shape=slm_shape,
            trap_config=trap_config,
        )

        pprint(run_config, expand_all=True)

        # NB h5diff -d 1e-3 results/data/exercise_reference_gs_20260727_120804.h5 results/data/exercise_gs_20260727_120932.h5
        exp = HologramExperiment(run_config)
        exp.plot(base_dir="./results")
        exp.write_h5(base_dir="./results")

        for random_seed in np.arange(num_random_seeds):
            random_seed = int(42 + random_seed)

            for smooth_phase in (False,):
                solver_config = SolverConfig(
                    method=method,  # {"GS", "GD", "AA", "HIO"}
                    maxiter=200,
                    random_seed=int(random_seed),
                    smooth_phase=smooth_phase,
                    solver_backend=solver_backend,  # {"slm_suite", "jax"}
                )

                pprint(solver_config, expand_all=True)

                solver = HologramExperimentSolver(exp, solver_config)
                solver.optimize()
                solver.plot(base_dir="./results")
                solver.write_h5(base_dir="./results")

                metrics = PerformanceMetrics.from_solver(solver)

                pprint(metrics, expand_all=True)

                PerformanceMetrics.write_tex_table(
                    filepath=exp._get_run_dir("./results")
                    / f"phase_retrieval/{solver.config.hash}"
                    / "performance_metrics.tex",
                    metrics_by_run={solver.config.method: metrics.to_dict()},
                    caption=f"Computed performance metrics for the {solver.config.method}-optimized SLM phase.",
                )

                # solver.update_aim(experiment_name=solver_config.hash)

    logger.info(f"Done.")


def main():
    run_phase_retrieval()


if __name__ == "__main__":
    main()
