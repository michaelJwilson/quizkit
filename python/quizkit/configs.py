import datetime
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

    hash: str = field(default_factory=lambda: uuid.uuid4().hex[:7])
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    def __getattr__(self, name):
        try:
            return getattr(self.trap_config, name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )


@dataclass
class SolverConfig(ConfigMixin):
    method: str  # {"GS", "WGS", "GD"}
    maxiter: int = 200

    solver_backend: str | None = None
    solver_runtime: float | None = None

    smooth_phase: bool = False
    smooth_sigma: int = 3  # pixels

    loss_norm: str = "L2"  # {"L1", "L2"}
    learning_rate: float = 0.1

    random_seed: int = 42

    # TODO HACK
    initial_epsilon: float = 0.0
    anneal_rate: float = 0.05

    hash: str = field(default_factory=lambda: uuid.uuid4().hex[:7])
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
