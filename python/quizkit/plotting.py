import matplotlib.pyplot as plt


def plot_image(filepath, data):
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.imshow(data, cmap="inferno", origin="lower")

    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Counts")

    ax.set_xlabel("Pixels")
    ax.set_ylabel("Pixels")

    fig.savefig(filepath, dpi=300, bbox_inches="tight")
