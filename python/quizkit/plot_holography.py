import argparse
import matplotlib.pyplot as plt
import h5py
import numpy as np


def plot_holography(hdf5_path):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.titlesize": 14,
            "figure.titlesize": 16,
        }
    )

    with h5py.File(hdf5_path, "r") as f:
        # Safely extract data only if the key exists in the file
        source = f["source_image"][:] if "source_image" in f else None
        sampled = f["target_image"][:] if "target_image" in f else None
        phi = f["slm_phases"][:] if "slm_phases" in f else None
        inferred = f["model_intensity"][:] if "model_intensity" in f else None

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    panels = [
        (axs[0, 0], source, "source", "magma", None, None),
        (axs[0, 1], sampled, "sampled", "magma", None, None),
        (axs[1, 0], phi, r"inferred phase", "twilight", 0, 2 * np.pi),
        (axs[1, 1], inferred, "inferred intensity", "magma", None, None),
    ]

    for ax, data, title, cmap, vmin, vmax in panels:
        if data is not None:
            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.axis("off")

    plt.tight_layout()
    return fig, axs


def main():
    parser = argparse.ArgumentParser(description="Generate holography plots from saved HDF5 data.")
    parser.add_argument("--traps_path", type=str, required=True, help="Path to the input trap data HDF5")
    parser.add_argument("--results_path", type=str, required=True, help="Path to the optimized results HDF5")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output plot (e.g., holography.pdf)")
    
    args = parser.parse_args()

    fig, _ = plot_holography(args.results_path)
    
    fig.savefig(args.output_path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()