import logging
import numpy as np
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
    return_fig=False,
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

    logger.debug(f"Writing {plot_path}.")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")

    if return_fig:
        return fig
    else:
        plt.close(fig)

def plot_stack_with_marginals(
    plot_path,
    field,
    cmap="inferno",
    title=None,
    extent=None,
    cbar_label=None,
    figsize=(6, 5.5),
    xlabel=None,
    ylabel=None,
    return_fig=False,
    **imshow_kwargs,
):
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(field, cmap=cmap, extent=extent, **imshow_kwargs)
    
    cy, cx = field.shape[0] // 2, field.shape[1] // 2

    divider = make_axes_locatable(ax)
    
    ax_top = divider.append_axes("top", size="25%", pad=0.1, sharex=ax)
    ax_right = divider.append_axes("right", size="25%", pad=0.1, sharey=ax)
    
    cax = divider.append_axes("right", size="5%", pad=0.2)

    x_axis = np.arange(field.shape[1])
    if extent:
        x_axis = np.linspace(extent[0], extent[1], field.shape[1])
    ax_top.plot(x_axis, field[cy, :], color="black", linewidth=1.5)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.set_ylabel("Profile")

    y_axis = np.arange(field.shape[0])
    if extent:
        y_axis = np.linspace(extent[2], extent[3], field.shape[0])
    ax_right.plot(field[:, cx], y_axis, color="black", linewidth=1.5)
    ax_right.tick_params(axis="y", labelleft=False)
    ax_right.set_xlabel("Profile")

    if title:
        ax_top.set_title(title, fontsize=14)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    fig.colorbar(im, cax=cax, label=cbar_label)

    logger.debug(f"Writing {plot_path}.")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")

    if return_fig:
        return fig
    else:
        plt.close(fig)
