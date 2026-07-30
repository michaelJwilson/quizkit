import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
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
    fit_x=None,
    fit_y=None,
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

    # Extract data slices
    data_x = field[cy, :]
    data_y = field[:, cx]

    # Calculate normalization bounds (prevent div by zero)
    min_x, ptp_x = np.min(data_x), np.ptp(data_x)
    ptp_x = ptp_x if ptp_x > 0 else 1.0

    min_y, ptp_y = np.min(data_y), np.ptp(data_y)
    ptp_y = ptp_y if ptp_y > 0 else 1.0

    # Normalize X
    norm_data_x = (data_x - min_x) / ptp_x
    ax_top.plot(x_axis, norm_data_x, color="black", linewidth=1.5, label="Mean data")

    if fit_x is not None:
        norm_fit_x = (fit_x - min_x) / ptp_x
        ax_top.plot(
            x_axis, norm_fit_x, color="cyan", alpha=0.5, linestyle="-", linewidth=1.2
        )

    ax_top.tick_params(axis="x", labelbottom=False)
    # ax_top.set_ylabel(r"Norm. $y$ profile")
    ax_top.set_ylim(-0.05, 1.05)  # Lock limits to data range, ignoring fit extremes

    y_axis = np.arange(field.shape[0])
    if extent:
        y_axis = np.linspace(extent[2], extent[3], field.shape[0])

    norm_data_y = (data_y - min_y) / ptp_y
    ax_right.plot(norm_data_y, y_axis, color="black", linewidth=1.5)

    if fit_y is not None:
        norm_fit_y = (fit_y - min_y) / ptp_y
        ax_right.plot(norm_fit_y, y_axis, color="red", linestyle="--", linewidth=1.2)

    ax_right.tick_params(axis="y", labelleft=False)
    # ax_right.set_xlabel(r"Norm. $x$ profile")
    ax_right.set_xlim(-0.05, 1.05)  # Lock limits to data range, ignoring fit extremes

    if title:
        ax_top.set_title(title, fontsize=12)
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

def plot_unraveled_trap_profiles(
    forward_intensity: np.ndarray,
    trap_coords: np.ndarray,
    wx: float,
    wy: float,
    plot_path: Path | str
):
    """
    Creates a horizontal waterfall plot of 1D X and Y trap profiles.
    Traps are sorted by peak intensity and staggered horizontally.
    Normalized globally to preserve relative peak intensities.
    """
    x_radius = max(2, int(np.ceil(2 * wx)))
    y_radius = max(2, int(np.ceil(2 * wy)))
    
    x_span = np.arange(-x_radius, x_radius + 1)
    y_span = np.arange(-y_radius, y_radius + 1)
    
    ff_int = np.asarray(forward_intensity)
    h, w = ff_int.shape
    
    # 1. Pre-extract profiles and calculate local peaks
    profiles_data = []
    
    for i, (cy, cx) in enumerate(trap_coords):
        x_min, x_max = cx - x_radius, cx + x_radius + 1
        y_min, y_max = cy - y_radius, cy + y_radius + 1
        
        # Only keep traps that don't clip the camera boundary
        if 0 <= x_min and x_max <= w and 0 <= y_min and y_max <= h:
            prof_x = ff_int[cy, x_min:x_max]
            prof_y = ff_int[y_min:y_max, cx]
            
            # Find the max intensity of this specific trap
            local_max = max(np.max(prof_x), np.max(prof_y))
            
            profiles_data.append({
                "orig_idx": i,
                "prof_x": prof_x,
                "prof_y": prof_y,
                "max_val": local_max
            })
            
    if not profiles_data:
        logger.warning("No valid profiles found to unravel. Skipping plot.")
        return
        
    # 2. Sort traps by max height (ascending, so the "mountains" grow left-to-right)
    profiles_data.sort(key=lambda item: item["max_val"], reverse=True)
    
    # Global max is now the last item in the sorted list
    global_max = profiles_data[-1]["max_val"]
    
    # 3. Render the staggered plot
    fig_prof, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # TODO
    x_stagger_step = wx
    y_stagger_step = wy
    
    for rank, item in enumerate(profiles_data):
        # Top Row: X Profiles
        prof_x_norm = item["prof_x"] / (global_max + 1e-12)
        
        # Stagger on the X-axis
        x_shifted_x = x_span + (rank * x_stagger_step)
        
        ax1.plot(x_shifted_x, prof_x_norm, color="cyan", alpha=0.8, linewidth=1.0)
        ax1.fill_between(x_shifted_x, 0, prof_x_norm, color="cyan", alpha=0.05)
        
        # Bottom Row: Y Profiles
        prof_y_norm = item["prof_y"] / (global_max + 1e-12)
        
        # Stagger on the X-axis (even though it's a Y-profile, we spread them horizontally for the viewer)
        x_shifted_y = y_span + (rank * y_stagger_step)
        
        ax2.plot(x_shifted_y, prof_y_norm, color="magenta", alpha=0.8, linewidth=1.0)
        ax2.fill_between(x_shifted_y, 0, prof_y_norm, color="magenta", alpha=0.05)

    ax1.set_title("Trap x-profile")
    ax1.set_xlabel(rf"Local X Distance $+ (Rank \times {x_stagger_step:.2f})$")
    ax1.set_ylabel("Intensity [a.u.]")
    ax1.set_ylim(bottom=0)
    
    ax2.set_title("Trap y-profile")
    ax2.set_xlabel(rf"Local Y Distance $+ (Rank \times {y_stagger_step:.2f})$")
    ax2.set_ylabel("Intensity [a.u.]")
    ax2.set_ylim(bottom=0)
    
    fig_prof.tight_layout()
    fig_prof.savefig(plot_path)
    plt.close(fig_prof)
