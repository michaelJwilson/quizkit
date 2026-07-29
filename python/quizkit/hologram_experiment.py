import datetime
import json
import logging
import random
import time
import uuid  # TODO
from dataclasses import asdict, dataclass, field
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Any, Optional, Tuple

import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rich.pretty import pprint
from scipy.ndimage import binary_dilation, find_objects, label
from slmsuite.holography.algorithms import SpotHologram

from quizkit.configs import RunConfig, TrapConfig
from quizkit.plotting import plot_scalar_field
from quizkit.writers import write_hdf5

logger = logging.getLogger(__name__)


def get_gaussian_slm_illumination(slm_shape):
    # NB construct source amplitude profile
    x = np.arange(slm_shape[1]) - (slm_shape[1] - 1) / 2
    y = np.arange(slm_shape[0]) - (slm_shape[0] - 1) / 2

    xx, yy = np.meshgrid(x, y)

    # TODO HARDCODE
    beam_waist_px = 0.35 * min(slm_shape)  # 1/e^2 amplitude radius, in pixels

    return np.exp(-(xx**2 + yy**2) / beam_waist_px**2).astype(np.float32)


def get_trap_array_mask(slm_shape, array_shape, array_pitch, array_center, pad=25):
    if array_center is not None:
        center_x, center_y = array_center[0], array_center[1]
    else:
        center_y, center_x = slm_shape[0] / 2.0, slm_shape[1] / 2.0

    array_extent = (np.array(array_shape) - 1) * np.array(array_pitch)

    x_min = int(np.floor(center_x - array_extent[1] / 2.0 - pad))
    x_max = int(np.ceil(center_x + array_extent[1] / 2.0 + pad))
    y_min = int(np.floor(center_y - array_extent[0] / 2.0 - pad))
    y_max = int(np.ceil(center_y + array_extent[0] / 2.0 + pad))

    x_min_clamped = max(0, x_min)
    x_max_clamped = min(slm_shape[1], x_max)
    y_min_clamped = max(0, y_min)
    y_max_clamped = min(slm_shape[0], y_max)

    trap_array_mask = np.zeros(slm_shape, dtype=bool)
    trap_array_mask[y_min_clamped:y_max_clamped, x_min_clamped:x_max_clamped] = True

    return trap_array_mask


def encode_target_traps(target_intensity, threshold_frac=0.0):
    threshold = threshold_frac * target_intensity.max()
    binary_target = target_intensity > threshold

    labeled_mask, num_traps = label(binary_target)
    slices = find_objects(labeled_mask)

    coords = []
    trap_h, trap_w = 0, 0

    for s in slices:
        trap_h = max(trap_h, s[0].stop - s[0].start)
        trap_w = max(trap_w, s[1].stop - s[1].start)

    for s in slices:
        center_y = (s[0].start + s[0].stop) // 2
        center_x = (s[1].start + s[1].stop) // 2
        coords.append((center_y, center_x))

    return labeled_mask, num_traps, np.array(coords), trap_h, trap_w


def get_trap_zoom(target_intensity, pad=25):
    trap_coords = np.argwhere(target_intensity > 0)

    assert trap_coords.shape[0] > 0, "No traps found in target intensity."

    y_min, y_max = np.min(trap_coords[:, 0]), np.max(trap_coords[:, 0])
    x_min, x_max = np.min(trap_coords[:, 1]), np.max(trap_coords[:, 1])

    y_min = max(0, y_min - pad)
    y_max = min(target_intensity.shape[0] - 1, y_max + pad)
    x_min = max(0, x_min - pad)
    x_max = min(target_intensity.shape[1] - 1, x_max + pad)

    return (x_min, x_max, y_min, y_max)


# TODO
def compute_structural_metrics(wavelength, pixel_pitch, slm_shape):
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

    return {
        "max_steering_angle_rad": float(max_steering_angle),
        "max_steering_angle_deg": float(max_steering_angle_deg),
        "farfield_extent_rad": float(farfield_extent),
        "farfield_resolution_rad": float(farfield_resolution),
    }


class HologramExperiment:
    _is_frozen = False

    def __init__(self, run_config: RunConfig):
        self.run_config = run_config
        self.slm_illumination = get_gaussian_slm_illumination(self.run_config.slm_shape)
        self.slm_illumination.flags.writeable = False

        hologram = SpotHologram.make_rectangular_array(
            self.run_config.slm_shape,
            array_shape=self.run_config.array_shape,
            array_pitch=self.run_config.array_pitch,
            basis="knm",
            amp=self.slm_illumination.copy(),
            array_center=self.run_config.array_center,
            phase=np.random.uniform(-np.pi, np.pi, self.run_config.slm_shape),
        )

        self.target = hologram.target.copy()
        self.target.flags.writeable = False

        self.trap_array_mask = get_trap_array_mask(
            self.run_config.slm_shape,
            self.run_config.array_shape,
            self.run_config.array_pitch,
            self.run_config.array_center,
            pad=25,
        )
        self.trap_array_mask.flags.writeable = False

        trap_labels_np, self.num_traps, coords_np, self.trap_h, self.trap_w = (
            encode_target_traps(self.target, threshold_frac=0.0)
        )

        self.trap_labels = jnp.array(trap_labels_np)
        self.trap_coords = jnp.array(coords_np)

        # TODO DEPRECATE
        self.crop_coords = self.trap_coords

        self._is_frozen = True

    @classmethod
    def from_run_h5(cls, run_dir: str | Path):
        run_dir = Path(run_dir)

        with open(run_dir / "run_config.json", "r") as f:
            data = json.load(f)
            trap_data = data.pop("trap_config")
            data["trap_config"] = TrapConfig(**trap_data)
            run_config = RunConfig(**data)

        obj = cls.__new__(cls)
        obj.run_config = run_config

        h5_path = run_dir / "experiment.h5"

        with h5py.File(h5_path, "r") as f:
            obj.slm_illumination = f["slm/slm_illumination"][:]
            obj.target = f["target/target"][:]
            obj.trap_array_mask = f["trap_array_mask/trap_array_mask"][:]

        obj.slm_illumination.flags.writeable = False
        obj.target.flags.writeable = False

        trap_labels_np, obj.num_traps, coords_np, obj.trap_h, obj.trap_w = (
            encode_target_traps(obj.target, threshold_frac=0.0)
        )

        obj.trap_labels = jnp.array(trap_labels_np)
        obj.trap_coords = jnp.array(coords_np)
        obj.trap_array_mask = jnp.array(obj.trap_array_mask)
        obj.crop_coords = obj.trap_coords

        obj._is_frozen = True

        logger.info(f"Loaded HologramExperiment from {run_dir}.")

        return obj

    def __setattr__(self, key, value):
        if getattr(self, "_is_frozen", False):
            cls_attr = getattr(type(self), key, None)
            if isinstance(cls_attr, cached_property):
                super().__setattr__(key, value)
                return
            raise AttributeError(
                f"'{type(self).__name__}' is immutable. Cannot modify attribute '{key}'."
            )
        super().__setattr__(key, value)

    @cached_property
    def target_intensity(self):
        intensity = np.abs(self.target) ** 2
        intensity.flags.writeable = False
        return intensity

    @cached_property
    def target_extent(self):
        return get_trap_zoom(self.target)

    @cached_property
    def geometric_metrics(self):
        wave = self.run_config.wavelength
        pitch = self.run_config.pixel_pitch
        h, w = self.run_config.slm_shape

        max_steering_angle = wave / pitch
        farfield_extent = max_steering_angle / 2.0

        farfield_res_y = farfield_extent / h
        farfield_res_x = farfield_extent / w

        return {
            "max_steering_angle_rad": float(max_steering_angle),
            "max_steering_angle_deg": float(np.degrees(max_steering_angle)),
            "farfield_extent_rad": float(farfield_extent),
            "farfield_res_y_rad": float(farfield_res_y),
            "farfield_res_x_rad": float(farfield_res_x),
        }

    def _get_run_dir(self, base_dir: str) -> Path:
        return Path(base_dir) / f"run_{self.run_config.hash}"

    def plot(self, base_dir: str = "./results"):
        run_dir = self._get_run_dir(base_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "plots").mkdir(parents=True, exist_ok=True)

        plot_scalar_field(
            run_dir / "plots" / "gaussian_slm_illumination.pdf",
            np.log(self.slm_illumination + 1e-12),
            cbar_label="ln. slm illumination",
        )

        x_min, x_max, y_min, y_max = self.target_extent
        plot_scalar_field(
            run_dir / "plots" / "target_intensity.pdf",
            np.log(self.target_intensity[y_min:y_max, x_min:x_max] + 1e-12),
            extent=self.target_extent,
            cbar_label="ln. target intensity",
        )

    def write_h5(self, base_dir: str = "./results"):
        run_dir = self._get_run_dir(base_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Writing run configuration to {run_dir / 'run_config.json'}.")

        with open(run_dir / "run_config.json", "w") as f:
            f.write(self.run_config.to_json())

        hdf5_path = run_dir / "experiment.h5"

        logger.info(f"Writing hologram experiment to {hdf5_path}.")

        write_hdf5(
            filepath=hdf5_path,
            data=self.slm_illumination,
            group_name="slm",
            dataset_name="slm_illumination",
        )

        write_hdf5(
            filepath=hdf5_path,
            data=self.target,
            group_name="target",
            dataset_name="target",
        )

        write_hdf5(
            filepath=hdf5_path,
            data=self.trap_array_mask,
            group_name="trap_array_mask",
            dataset_name="trap_array_mask",
        )
