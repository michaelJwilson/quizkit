import datetime
import hashlib
import json
import uuid  # TODO
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional, Tuple

import numpy as np


class ConfigMixin:
    def to_dict(self) -> dict:
        data = asdict(self)
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                data[key] = value.tolist()
        return data

    def to_json(self, indent: int = 4) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def _generate_param_hash(self) -> str:
        data = self.to_dict()

        # Remove volatile or output fields that do not define the parameter set
        for key in ["hash", "timestamp", "solver_runtime"]:
            data.pop(key, None)

        # Recursive pop for nested configs (like trap_config inside RunConfig)
        if "trap_config" in data and isinstance(data["trap_config"], dict):
            data["trap_config"].pop("hash", None)
            data["trap_config"].pop("timestamp", None)

        # sort_keys=True guarantees identical parameter sets produce identical strings
        param_string = json.dumps(data, sort_keys=True)

        # Use MD5 for a fast, short, deterministic hex ID
        return hashlib.md5(param_string.encode("utf-8")).hexdigest()[:7]


@dataclass
class TrapConfig(ConfigMixin):
    trap_id: int
    trap_type: str
    array_shape: Tuple[int, int]
    array_pitch: Tuple[int, int]
    array_center: Optional[Tuple[float, float]] = None


# TODO
class TrapConfigs(Enum):
    # NB format=(trap_type, array_shape, array_pitch, array_center)
    ON_AXIS = (0, "on_axis", (10, 10), (20, 20), None)
    OFF_AXIS = (1, "off_axis", (10, 10), (20, 20), (3.0 * 1920 / 4, 2.0 * 1200 / 4))

    @property
    def id(self) -> int:
        return self.value[0]

    def to_config(self) -> TrapConfig:
        config_id, trap_type, array_shape, array_pitch, array_center = self.value
        return TrapConfig(
            trap_id=self.id,
            trap_type=trap_type,
            array_shape=array_shape,
            array_pitch=array_pitch,
            array_center=array_center,
        )


@dataclass
class RunConfig(ConfigMixin):
    wavelength: float
    pixel_pitch: float
    slm_shape: Tuple[int, int]
    trap_config: TrapConfig
    comment: Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    hash: Optional[str] = None

    def __post_init__(self):
        # TODO !!! remove before flight
        if not self.hash:
            self.hash = self._generate_param_hash()

    def __getattr__(self, name):
        try:
            return getattr(self.trap_config, name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )


@dataclass
class SolverConfig(ConfigMixin):
    method: str  # {"GS", "WGS", "GD", "AA", "HIO"}
    maxiter: int = 200

    solver_backend: str | None = None
    solver_runtime: float | None = None

    smooth_phase: bool = True
    smooth_sigma: int = 1  # pixels

    # NB
    downsample_factor: int = 4

    loss_norm: str = "L2"  # {"L1", "L2"}
    learning_rate: float = 0.1

    # --- New Algorithm Hyperparameters ---
    aa_alpha: float = 0.5  # Feedback parameter for Adaptive-Additive
    hio_beta: float = 0.8  # Feedback parameter for Fienup HIO

    random_seed: int = 42

    initial_epsilon: float = 0.0
    anneal_rate: float = 0.05

    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    hash: Optional[str] = None

    def __post_init__(self):
        # TODO !!! remove before flight
        if not self.hash:
            self.hash = self._generate_param_hash()
