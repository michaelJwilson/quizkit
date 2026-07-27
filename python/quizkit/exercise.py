import datetime
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from quizkit.writers import write_hdf5
from slmsuite.holography.algorithms import Hologram, SpotHologram


# import cupy as xp
# from cupyx.scipy.ndimage import gaussian_filter

from scipy.ndimage import gaussian_filter

"""
GS algorithm application via slm suite, see

https://slmsuite.readthedocs.io/en/latest/_examples/computational_holography.html#Basic-Image-Formation
"""

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


def plot_phase_retrieval_results(plot_path, phase, intensity, intensity_extent):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.titlesize": 14,
            "figure.titlesize": 16,
        }
    )

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))

    im0 = axs[0].imshow(phase, cmap="twilight", interpolation="nearest")
    axs[0].set_title("slm phase")
    axs[0].set_xlabel("pixel x")
    axs[0].set_ylabel("pixel y")

    axs[0].set_aspect("equal", adjustable="box")

    div0 = make_axes_locatable(axs[0])
    cax0 = div0.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im0, cax=cax0, label="phase [rad]")

    im1 = axs[1].imshow(
        intensity,
        cmap="inferno",
        origin="lower",
        extent=intensity_extent,
    )
    axs[1].set_title("Far-field intensity")
    axs[1].set_xlabel(r"$k_n$ [knm]")
    axs[1].set_ylabel(r"$k_m$ [knm]")

    axs[1].set_aspect("equal", adjustable="box")

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
    farfield_extent = wavelength / pixel_pitch / 2. # radians
    farfield_resolution = farfield_extent / slm_shape# radians, assumes square slm pixels.

    # print(max_steering_angle_deg)
    # print(slm_fundamental_modes)
    # print(nyquist_max)

def smooth_slm_array(array, sigma=2.0):
    # TODO cupy support.
    device_array = np.asarray(array)

    # TODO cv2.GaussianBlur
    # NB sigma [pixels] - see https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
    return gaussian_filter(device_array, sigma=sigma)

def smooth_phase_callback(hologram, sigma=4):
    if hologram.iter % 1 == 0:
        # TODO BUG?  device transfer needs to be accounted for in terms of updates. 
        # phase = hologram.get_phase()
        phase = hologram.phase
        # phase = smooth_slm_array(phase, sigma=sigma)

        print(phase.shape, phase.sum())
        # (1200, 1920) 7310169.0
        # (1200, 1920) 7303059.5
        # (1200, 1920) 7302162.5
        # (1200, 1920) 7304882.5
        # (1200, 1920) 7285615.0
        # (1200, 1920) 7286407.5
        # (1200, 1920) 7283061.0
        # (1200, 1920) 7274069.5


if __name__ == "__main__":
    WAVELENGTH = 780e-9  # m
    PIXEL_PITCH = 8.0e-6  # m
    SLM_SHAPE = (1200, 1920)  # (height, width) in pixels

    # NB 10x10 optical tweezer array sampling a 200x200 image.
    ARRAY_SHAPE = (10, 10)  # 10 x 10 = 100 spots
    ARRAY_PITCH = (20, 20)  # spot separation in far-field grid samples

    METHOD = "GS"  # {GS, WGS}
    MAXITER = 30

    # NB construct source amplitude profile
    # x = np.arange(SLM_SHAPE[1]) - (SLM_SHAPE[1] - 1) / 2
    # y = np.arange(SLM_SHAPE[0]) - (SLM_SHAPE[0] - 1) / 2
    # xx, yy = np.meshgrid(x, y)
    # beam_waist_px = 0.35 * min(SLM_SHAPE)  # 1/e^2 amplitude radius, in pixels
    # slm_illumination = np.exp(-(xx**2 + yy**2) / beam_waist_px**2).astype(np.float32)

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
    )

    hologram.optimize(
        method=METHOD,
        maxiter=MAXITER,
        callback=smooth_phase,
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

    # TODO
    cy, cx = ff_int.shape[0] // 2, ff_int.shape[1] // 2
    half = 130
    ff_crop = ff_int[cy - half : cy + half, cx - half : cx + half]

    plot_phase_retrieval_results(
        "./results/plots/phase_retrieval_results.pdf",
        slm_phase,
        ff_crop,
        [cx - half, cx + half, cy - half, cy + half],
    )

    # NB h5diff -d 1e-3 results/data/exercise_reference_gs_20260727_120804.h5 results/data/exercise_gs_20260727_120932.h5
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    hdf5_path = f"./results/data/exercise_{METHOD.lower()}_{timestamp}.h5"
    
    write_hdf5(
        filepath=hdf5_path,
        data=slm_phase,
        group_name="slm",
        dataset_name="slm_phase",
        wavelength=WAVELENGTH,
        pixel_pitch=PIXEL_PITCH,
        maxiter=MAXITER
    )

    write_hdf5(
        filepath=hdf5_path,
        data=slm_illumination,
        group_name="slm",
        dataset_name="slm_illumination"
    )

    write_hdf5(
        filepath=hdf5_path,
        data=np.abs(hologram.target) ** 2,
        group_name="target",
        dataset_name="target_intensity"
    )

    write_hdf5(
        filepath=hdf5_path,
        data=target_intensity,
        group_name="target",
        dataset_name="inferred_farfield_intensity",
        # **stats
    )


# def main():
#     run_slmsuit_phase_retrieval()

 
# if __name__ == "__main__":
#     main()
