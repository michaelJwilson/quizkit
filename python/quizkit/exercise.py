import matplotlib.pyplot as plt
import numpy as np
from slmsuite.holography.algorithms import SpotHologram

"""
GS algorithm application via slm suite, see

https://slmsuite.readthedocs.io/en/latest/_examples/computational_holography.html#Basic-Image-Formation
"""

WAVELENGTH = 780e-9  # m
PIXEL_PITCH = 8.0e-6  # m
SLM_SHAPE = (1200, 1920)  # (height, width) in pixels

# NB optical tweezer array configuration
ARRAY_SHAPE = (10, 10)  # 10 x 10 = 100 spots
ARRAY_PITCH = (20, 20)  # spot separation in far-field grid samples

MAXITER = 30  # GS max. iterations.

x = np.arange(SLM_SHAPE[1]) - (SLM_SHAPE[1] - 1) / 2
y = np.arange(SLM_SHAPE[0]) - (SLM_SHAPE[0] - 1) / 2
xx, yy = np.meshgrid(x, y)
beam_waist_px = 0.35 * min(SLM_SHAPE)  # 1/e^2 amplitude radius, in pixels
source_amp = np.exp(-(xx**2 + yy**2) / beam_waist_px**2).astype(np.float32)

fig, ax = plt.subplots(figsize=(5, 3.2))
im = ax.imshow(source_amp, cmap="inferno")
ax.set_aspect("equal")  # show that the beam is symmetric
ax.set_title("Source amplitude")
fig.colorbar(im, ax=ax, label="amplitude")
fig.tight_layout()
plt.show()

hologram = SpotHologram.make_rectangular_array(
    SLM_SHAPE,
    array_shape=ARRAY_SHAPE,
    array_pitch=ARRAY_PITCH,
    basis="knm",
    amp=source_amp,  # fixed Gaussian illumination
)

hologram.optimize(
    method="GS",
    maxiter=MAXITER,
    stat_groups=["computational_spot"],
    verbose=False,
)

phase = hologram.get_phase()  # in [0, 2*pi]
ff_int = np.abs(hologram.get_farfield()) ** 2
cy, cx = ff_int.shape[0] // 2, ff_int.shape[1] // 2
half = 130
ff_crop = ff_int[cy - half : cy + half, cx - half : cx + half]

fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))

im0 = axs[0].imshow(phase, cmap="twilight", interpolation="nearest")
axs[0].set_title("SLM phase")
axs[0].set_xlabel("pixel x")
axs[0].set_ylabel("pixel y")
fig.colorbar(im0, ax=axs[0], label="phase [rad]")

# Far field on the computational knm grid, centered on the zeroth order.
im1 = axs[1].imshow(
    ff_crop,
    cmap="inferno",
    origin="lower",
    extent=[cx - half, cx + half, cy - half, cy + half],
)
axs[1].set_title("Reconstructed far field")
axs[1].set_xlabel(r"$k_n$ [knm]")
axs[1].set_ylabel(r"$k_m$ [knm]")
fig.colorbar(im1, ax=axs[1], label="intensity")

fig.tight_layout()
plt.show()
