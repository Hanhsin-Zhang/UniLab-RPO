from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from unilab.envs.locomotion.common.base import (
    BaseNoiseConfig,
    ControlConfigBase,
    LocomotionBaseCfg,
    LocomotionBaseEnv,
)
from unilab.envs.locomotion.common.base import (
    Sensor as LocomotionSensor,
)


@dataclass
class NoiseConfig(BaseNoiseConfig):
    scale_joint_angle: float = 0.02
    scale_joint_vel: float = 0.3
    scale_gyro: float = 0.1


@dataclass
class ControlConfig(ControlConfigBase):
    action_scale: float | np.ndarray = 0.25  # type: ignore[assignment]


@dataclass
class Sensor(LocomotionSensor):
    local_linvel: str = "linear-velocity"
    gyro: str = "angular-velocity"
    upvector: str = "upvector"


@dataclass
class Asset:
    base_name = "base_link"
    foot_name = "ankle_roll_link"
    ground = "floor"


@dataclass
class G1RPOBaseCfg(LocomotionBaseCfg):
    noise_config: NoiseConfig = field(default_factory=NoiseConfig)  # type: ignore[assignment]
    control_config: ControlConfig = field(default_factory=ControlConfig)  # type: ignore[assignment]
    sensor: Sensor = field(default_factory=Sensor)
    asset: Asset = field(default_factory=Asset)
    sim_dt: float = 0.02 / 3.0
    ctrl_dt: float = 0.02


class G1RPOBaseEnv(LocomotionBaseEnv):
    _cfg: G1RPOBaseCfg
    _keyframe_name = "stand"
