import logging
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

logger = logging.getLogger(__name__)

"""
def plot_image(filepath, data):
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.imshow(data, cmap="inferno", origin="lower")

    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Counts")

    ax.set_xlabel("Pixels")
    ax.set_ylabel("Pixels")

    fig.savefig(filepath, dpi=300, bbox_inches="tight")
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