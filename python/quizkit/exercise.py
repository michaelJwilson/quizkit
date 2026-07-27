import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from slmsuite.holography.algorithms import SpotHologram


"""
GS algorithm application via slm suite, see

https://slmsuite.readthedocs.io/en/latest/_examples/computational_holography.html#Basic-Image-Formation
"""


def plot_source_amplitude(plot_path, source_amp):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    im = ax.imshow(source_amp, cmap="inferno")
    ax.set_aspect("equal")  # show that the beam is symmetric

    # fig.colorbar(im, ax=ax, label="slm illumination")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    fig.colorbar(im, cax=cax, label="slm illumination")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)


def plot_phase_retrieval_results(plot_path, phase, intensity, intensity_extent):
    plt.rcParams.update({
        "font.family": "serif",
        "axes.titlesize": 14,
        "figure.titlesize": 16,
    })

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))

    im0 = axs[0].imshow(phase, cmap="twilight", interpolation="nearest")
    axs[0].set_title("slm phase")
    axs[0].set_xlabel("pixel x")
    axs[0].set_ylabel("pixel y")
    
    axs[0].set_aspect('equal', adjustable='box')
    
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
    
    axs[1].set_aspect('equal', adjustable='box')
    
    div1 = make_axes_locatable(axs[1])
    cax1 = div1.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im1, cax=cax1, label="intensity")

    fig.tight_layout(pad=2.0) 
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def compute_metrics(wavelength, pixel_pitch, slm_shape):
    max_steering_angle = wavelength / pixel_pitch

    # NB O(1) degrees
    max_steering_angle_deg = np.degrees(max_steering_angle)

    # NB dk_x = 1/L is goverened by the diffraction limit
    slm_aperture_extent = np.array(slm_shape) * pixel_pitch
    slm_fundamental_modes = 2. * np.pi / slm_aperture_extent

    # NB position in the image plane is k_x * eff. focal length (of a microscope).
    nyquist_max = wavelength / pixel_pitch / 2.

    # print(max_steering_angle_deg)
    # print(slm_fundamental_modes)
    print(nyquist_max)


def run_slmsuit_phase_retrieval():
    WAVELENGTH = 780e-9  # m
    PIXEL_PITCH = 8.0e-6  # m
    SLM_SHAPE = (1200, 1920)  # (height, width) in pixels

    # NB optical tweezer array configuration
    ARRAY_SHAPE = (10, 10)  # 10 x 10 = 100 spots
    ARRAY_PITCH = (20, 20)  # spot separation in far-field grid samples

    # NB gs max. iterations.
    METHOD = "GS"
    MAXITER = 30

    # NB construct source amplitude profile
    x = np.arange(SLM_SHAPE[1]) - (SLM_SHAPE[1] - 1) / 2
    y = np.arange(SLM_SHAPE[0]) - (SLM_SHAPE[0] - 1) / 2
    xx, yy = np.meshgrid(x, y)
    beam_waist_px = 0.35 * min(SLM_SHAPE)  # 1/e^2 amplitude radius, in pixels
    source_amp = np.exp(-(xx**2 + yy**2) / beam_waist_px**2).astype(np.float32)

    # compute_metrics(WAVELENGTH, PIXEL_PITCH, SLM_SHAPE)

    # NB plot Gaussian slm amplitude profile
    # plot_source_amplitude("./results/plots/gaussian_source_amplitude.pdf", source_amp)

    # NB construct tweezer array hologram and optimize it with GS algorithm
    hologram = SpotHologram.make_rectangular_array(
        SLM_SHAPE,
        array_shape=ARRAY_SHAPE,
        array_pitch=ARRAY_PITCH,
        basis="knm",  # pixel coordinates in the far-field image plane
        amp=source_amp,  # fixed Gaussian illumination
    )

    hologram.optimize(
        method=METHOD,
        maxiter=MAXITER,
        stat_groups=["computational_spot"],
        verbose=False,
    )

    # NB get optimized slm phase and far-field intensity,
    #    crop to show only the central region.
    phase = hologram.get_phase()  # in [0, 2*pi]
    ff_int = np.abs(hologram.get_farfield()) ** 2
    cy, cx = ff_int.shape[0] // 2, ff_int.shape[1] // 2
    half = 130
    ff_crop = ff_int[cy - half : cy + half, cx - half : cx + half]

    """
    plot_phase_retrieval_results(
        "./results/plots/phase_retrieval_results.pdf",
        phase,
        ff_crop,
        [cx - half, cx + half, cy - half, cy + half],
    )
    """

def main():
    run_slmsuit_phase_retrieval()


if __name__ == "__main__":
    main()
