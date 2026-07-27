import datetime
import random
import matplotlib.pyplot as plt
import numpy as np
from rich.pretty import pprint
from mpl_toolkits.axes_grid1 import make_axes_locatable
from quizkit.writers import write_hdf5
from slmsuite.holography.algorithms import Hologram, SpotHologram
from scipy.ndimage import gaussian_filter

"""
GS algorithm application via slm suite, see

https://slmsuite.readthedocs.io/en/latest/_examples/computational_holography.html#Basic-Image-Formation
"""

random.seed(42)
np.random.seed(42)


def get_uniform_slm_illumination(slm_shape):
    return np.ones(slm_shape, dtype=np.float32)


def get_gaussian_slm_illumination(slm_shape):
    # NB construct source amplitude profile
    x = np.arange(slm_shape[1]) - (slm_shape[1] - 1) / 2
    y = np.arange(slm_shape[0]) - (slm_shape[0] - 1) / 2

    xx, yy = np.meshgrid(x, y)

    # TODO HARDCODE
    beam_waist_px = 0.35 * min(slm_shape)  # 1/e^2 amplitude radius, in pixels

    return np.exp(-(xx**2 + yy**2) / beam_waist_px**2).astype(np.float32)


def plot_slm_illumination(plot_path, slm_illumination):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    im = ax.imshow(slm_illumination, cmap="inferno")
    ax.set_aspect("equal")  # show that the beam is symmetric

    # fig.colorbar(im, ax=ax, label="slm illumination")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(im, cax=cax, label="slm illumination")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)


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
        intensity,
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
    fig.colorbar(im1, cax=cax1, label="intensity")

    fig.tight_layout(pad=2.0)
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def compute_metrics(wavelength, pixel_pitch, slm_shape):
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


def compute_performance_metrics(ff_int, target_int):
    # NB given far-field intensity and target intensity ...
    ff_int = np.asarray(ff_int, dtype=np.float32)
    target_int = np.asarray(target_int, dtype=np.float32)

    # NB max target intensity;
    target_max = np.max(target_int)

    # TODO HARDCODE
    signal_mask = target_int > (0.01 * target_max)
    signal_mask = target_int > 0.0

    bg_mask = ~signal_mask

    signal_intensities = ff_int[signal_mask]
    bg_intensities = ff_int[bg_mask]

    total_power = np.sum(ff_int)

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


def write_metrics_table(filepath: str, metrics: dict) -> str:
    num_cols = len(metrics)
    col_alignment = "l" * num_cols

    headers = [f"\\textbf{{{k.replace('_', '\\_')}}}" for k in metrics.keys()]
    header_row = " & ".join(headers) + " \\\\"

    type_row = " & ".join(["\\texttt{float}"] * num_cols) + " \\\\"

    values = [f"{v:.4f}" for v in metrics.values()]
    value_row = " & ".join(values) + " \\\\"

    latex_string = f"""\\begin{{table}}[htbp]
\\centering
\\small
\\begin{{tabular}}{{{col_alignment}}}
\\toprule
{header_row}
{type_row}
\\midrule
{value_row}
\\bottomrule
\\end{{tabular}}
\\caption{{Computed performance metrics for the optimized SLM phase mask.}}
\\label{{tab:hologram_metrics}}
\\end{{table}}"""

    with open(filepath, "w") as f:
        f.write(latex_string)

    return latex_string


def smooth_slm_array(array, sigma=200):
    # TODO cupy support.
    device_array = np.asarray(array)

    # NB sigma [pixels]
    #    see https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
    #    see https://shimat.github.io/opencvsharp_docs/html/7b0301d7-322d-a554-8d3f-32fd8ca0ee50.htm
    # return gaussian_filter(device_array, sigma=sigma)
    #
    # for bordertypes, see
    #     https://shimat.github.io/opencvsharp_docs/html/040d5c3f-bd31-f5ff-76c8-106304d8135c.htm
    # return cv2.GaussianBlur(
    #     device_array,
    #     ksize=(0, 0),
    #     sigmaX=sigma,
    #     sigmaY=sigma
    # )
    device_array = np.asarray(array)
    complex_field = np.exp(1j * device_array)

    return gaussian_filter(complex_field, sigma=sigma, mode="wrap")


def smooth_phase_callback(hologram, sigma=200):
    # TODO see https://github.com/holodyne/slmsuite/blob/39243f081de020ad3ba74e672d126694b80778d2/slmsuite/holography/algorithms/_hologram.py#L1550
    print(f"Smoothing iteration {hologram.iter}")

    # TODO BUG?  device transfer needs to be accounted for in terms of updates.
    # phase = hologram.get_phase()
    phase = hologram.phase
    phase = smooth_slm_array(phase, sigma=sigma)

    # print(phase.shape, phase.sum())
    # (1200, 1920) 7310169.0
    # (1200, 1920) 7303059.5
    # (1200, 1920) 7302162.5
    # (1200, 1920) 7304882.5
    # (1200, 1920) 7285615.0
    # (1200, 1920) 7286407.5
    # (1200, 1920) 7283061.0
    # (1200, 1920) 7274069.5


def get_trap_zoom(target_intensity, pad=100):
    trap_coords = np.argwhere(target_intensity > 0)

    assert trap_coords.shape[0] > 0, "No traps found in target intensity."

    y_min, y_max = np.min(trap_coords[:, 0]), np.max(trap_coords[:, 0])
    x_min, x_max = np.min(trap_coords[:, 1]), np.max(trap_coords[:, 1])

    y_min = max(0, y_min - pad)
    y_max = min(target_intensity.shape[0] - 1, y_max + pad)
    x_min = max(0, x_min - pad)
    x_max = min(target_intensity.shape[1] - 1, x_max + pad)

    return (x_min, x_max, y_min, y_max)


if __name__ == "__main__":
    WAVELENGTH = 780e-9  # m
    PIXEL_PITCH = 8.0e-6  # m
    SLM_SHAPE = (1200, 1920)  # (height, width) in pixels

    # NB 10x10 optical tweezer array sampling a 200x200 image.
    ARRAY_SHAPE = (10, 10)  # 10 x 10 = 100 spots
    ARRAY_PITCH = (20, 20)  # spot separation in far-field grid samples

    METHOD = "GS"  # {GS, WGS}
    MAXITER = 30

    config = {
        "wavelength": WAVELENGTH,
        "pixel_pitch": PIXEL_PITCH,
        "slm_shape": SLM_SHAPE,
        "array_shape": ARRAY_SHAPE,
        "array_pitch": ARRAY_PITCH,
    }

    slm_illumination = get_gaussian_slm_illumination(SLM_SHAPE)

    # compute_metrics(WAVELENGTH, PIXEL_PITCH, SLM_SHAPE)

    # NB plot Gaussian slm amplitude profile
    plot_slm_illumination(
        "./results/plots/gaussian_slm_illumination.pdf", slm_illumination
    )

    # NB construct tweezer array hologram and optimize it with GS algorithm
    #    see https://github.com/holodyne/slmsuite/blob/39243f081de020ad3ba74e672d126694b80778d2/slmsuite/holography/algorithms/_hologram.py#L26
    hologram = SpotHologram.make_rectangular_array(
        SLM_SHAPE,
        array_shape=ARRAY_SHAPE,
        array_pitch=ARRAY_PITCH,
        basis="knm",  # pixel coordinates in the far-field image plane
        amp=slm_illumination,  # fixed Gaussian illumination
        phase=np.random.uniform(-np.pi, np.pi, SLM_SHAPE), # reproducibility required.
    )

    # NB callback definition,
    #    https://github.com/holodyne/slmsuite/blob/39243f081de020ad3ba74e672d126694b80778d2/slmsuite/holography/algorithms/_hologram.py#L1473
    hologram.optimize(
        method=METHOD,
        maxiter=MAXITER,
        # callback=smooth_phase_callback,
        stat_groups=["computational_spot"],
        verbose=False,
    )

    # hologram.plot_nearfield(cbar=True)

    # NB see https://github.com/holodyne/slmsuite/blob/main/slmsuite/holography/algorithms/_stats.py
    # hologram.stats.keys() == ['method', 'flags', 'stats']
    # hologram.stats["stats"].keys() == ['computational_spot']
    # hologram.stats["stats"]["computational_spot"].keys() == ['pkpk_err', 'std_err', 'uniformity', 'efficiency']
    stats = hologram.stats["stats"]["computational_spot"]

    # limits=zoombox
    # hologram.plot_farfield(cbar=True, title='FF Amp');

    # NB see https://github.com/holodyne/slmsuite/blob/39243f081de020ad3ba74e672d126694b80778d2/slmsuite/holography/algorithms/_stats.py#L7
    #    see https://github.com/holodyne/slmsuite/blob/39243f081de020ad3ba74e672d126694b80778d2/slmsuite/holography/algorithms/_stats.py#L729
    # hologram.plot_stats(show=True)

    # NB get optimized slm phase and far-field intensity,
    #    crop to show only the central region.
    #
    # NB current nearfield phase from the GPU shifted to [0, 2*pi].
    slm_phase = hologram.get_phase()

    ff_int = np.abs(hologram.get_farfield()) ** 2
    target_intensity = np.abs(hologram.target) ** 2

    performance_metrics = compute_performance_metrics(ff_int, target_intensity)
    write_metrics_table("./results/tables/performance_metrics.tex", performance_metrics)

    pprint(performance_metrics)

    # TODO
    extent = get_trap_zoom(target_intensity)
    x_min, x_max, y_min, y_max = extent

    # cy, cx = ff_int.shape[0] // 2, ff_int.shape[1] // 2
    # half = 130
    # ff_crop = ff_int[cy - half : cy + half, cx - half : cx + half]

    ff_crop = ff_int[y_min:y_max, x_min:x_max]

    # extent = [cx - half, cx + half, cy - half, cy + half],

    plot_phase_retrieval_results(
        "./results/plots/phase_retrieval_results.pdf",
        slm_phase,
        ff_crop,
        extent,
    )

    # TODO stats etc.
    header = config.copy()

    # NB h5diff -d 1e-3 results/data/exercise_reference_gs_20260727_120804.h5 results/data/exercise_gs_20260727_120932.h5
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    hdf5_path = f"./results/data/exercise_{METHOD.lower()}_{timestamp}.h5"

    # TODO better write of config.
    write_hdf5(
        filepath=hdf5_path,
        data=slm_phase,
        group_name="slm",
        dataset_name="slm_phase",
        wavelength=WAVELENGTH,
        pixel_pitch=PIXEL_PITCH,
        maxiter=MAXITER,
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
        **header,
    )

    write_hdf5(
        filepath=hdf5_path,
        data=ff_int,
        group_name=METHOD.lower(),
        dataset_name="inferred_farfield_intensity",
    )


# def main():
#     run_slmsuit_phase_retrieval()


# if __name__ == "__main__":
#     main()
