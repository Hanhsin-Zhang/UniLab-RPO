from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from unilab.envs.locomotion.common.base import (
    BaseNoiseConfig,
    ControlConfigBase,
    LocomotionBaseCfg,
    LocomotionBaseEnv,
)
from unilab.envs.locomotion.common.base import Sensor as LocomotionSensor


@dataclass
class NoiseConfig(BaseNoiseConfig):
    level: float = 1.0
    scale_joint_angle: float = 0.03
    scale_joint_vel: float = 1.75
    scale_gyro: float = 0.2
    scale_gravity: float = 0.05
    scale_linvel: float = 0.1


@dataclass
class ControlConfig(ControlConfigBase):
    # Keep XML actuator kp/kv authoritative in stage 2; Python only exposes action scaling.
    action_scale: float = 0.25


@dataclass
class Sensor(LocomotionSensor):
    local_linvel: str = "linear-velocity"
    gyro: str = "angular-velocity"
    upvector: str = "upvector"
    orientation: str = "orientation"
    position: str = "position"
    linear_acc: str = "linear-acceleration"
    magnetometer: str = "magnetometer"
    foot_pos: tuple[str, str] = ("left_foot_pos", "right_foot_pos")
    foot_quat: tuple[str, str] = ("left_foot_quat", "right_foot_quat")
    foot_linvel: tuple[str, str] = ("left_foot_linvel", "right_foot_linvel")
    knee_pos: tuple[str, str] = ("left_knee_pos", "right_knee_pos")
    foot_contact_force: tuple[str, str] = ("left_foot_contact", "right_foot_contact")
    # thigh_yaw / thigh_roll have no collision in URDF (commented out).
    # undesired contacts covers all non-foot bodies that have collision geometry,
    # matching roboparty's regex body_names="(?!.*ankle_roll.*).*".
    undesired_contact_force: tuple[str, ...] = (
        "torso_contact",
        # "left_thigh_yaw_contact",
        # "left_thigh_roll_contact",
        # "right_thigh_yaw_contact",
        # "right_thigh_roll_contact",
        "base_contact",
        "left_thigh_pitch_contact",
        "right_thigh_pitch_contact",
        "left_knee_contact",
        "right_knee_contact",
        "left_arm_pitch_contact",
        "left_arm_roll_contact",
        "left_arm_yaw_contact",
        "left_elbow_pitch_contact",
        "right_arm_pitch_contact",
        "right_arm_roll_contact",
        "right_arm_yaw_contact",
        "right_elbow_pitch_contact",
    )
    termination_contact_force: tuple[str, ...] = (
        "torso_contact",
        # "left_thigh_yaw_contact",
        # "left_thigh_roll_contact",
        # "right_thigh_yaw_contact",
        # "right_thigh_roll_contact",
        "base_contact",
    )
    actuator_frc: tuple[str, ...] = (
        "left_thigh_yaw_f",
        "left_thigh_roll_f",
        "left_thigh_pitch_f",
        "left_knee_f",
        "left_ankle_pitch_f",
        "left_ankle_roll_f",
        "right_thigh_yaw_f",
        "right_thigh_roll_f",
        "right_thigh_pitch_f",
        "right_knee_f",
        "right_ankle_pitch_f",
        "right_ankle_roll_f",
        "torso_yaw_f",
        "left_arm_pitch_f",
        "left_arm_roll_f",
        "left_arm_yaw_f",
        "left_elbow_pitch_f",
        "left_elbow_yaw_f",
        "right_arm_pitch_f",
        "right_arm_roll_f",
        "right_arm_yaw_f",
        "right_elbow_pitch_f",
        "right_elbow_yaw_f",
    )


@dataclass
class Asset:
    base_name: str = "base_link"
    ground: str = "floor"
    torso_name: str = "torso_link"
    imu_site: str = "imu"
    foot_site_names: tuple[str, str] = ("left_foot", "right_foot")
    floating_base_joint: str = "floating_base_joint"
    foot_name: str = "ankle_roll_link"
    foot_body_names: tuple[str, ...] = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    )
    knee_body_names: tuple[str, ...] = (
        "left_knee_link",
        "right_knee_link",
    )
    # thigh_yaw / thigh_roll have no collision in URDF; termination on torso + base_link.
    termination_body_names: tuple[str, ...] = (
        "torso_link",
        # "left_thigh_yaw_link",
        # "left_thigh_roll_link",
        # "right_thigh_yaw_link",
        # "right_thigh_roll_link",
        "base_link",
    )


@dataclass
class RPOBaseCfg(LocomotionBaseCfg):
    noise_config: NoiseConfig = field(default_factory=NoiseConfig)  # type: ignore[assignment]
    control_config: ControlConfig = field(default_factory=ControlConfig)  # type: ignore[assignment]
    sensor: Sensor = field(default_factory=Sensor)
    asset: Asset = field(default_factory=Asset)
    sim_dt: float = 0.005
    ctrl_dt: float = 0.02


class RPOBaseEnv(LocomotionBaseEnv):
    _cfg: RPOBaseCfg
    _keyframe_name = "stand"

    def get_upvector(self) -> np.ndarray:
        upvector: np.ndarray = self._backend.get_sensor_data(self._cfg.sensor.upvector)
        return upvector

    def get_orientation(self) -> np.ndarray:
        orientation: np.ndarray = self._backend.get_sensor_data(self._cfg.sensor.orientation)
        return orientation

    def get_position(self) -> np.ndarray:
        position: np.ndarray = self._backend.get_sensor_data(self._cfg.sensor.position)
        return position

    def get_foot_pos(self) -> np.ndarray:
        foot_pos = [self._backend.get_sensor_data(name) for name in self._cfg.sensor.foot_pos]
        return np.stack(foot_pos, axis=1)

    def get_foot_quat(self) -> np.ndarray:
        foot_quat = [self._backend.get_sensor_data(name) for name in self._cfg.sensor.foot_quat]
        return np.stack(foot_quat, axis=1)

    def get_foot_linvel(self) -> np.ndarray:
        foot_linvel = [self._backend.get_sensor_data(name) for name in self._cfg.sensor.foot_linvel]
        return np.stack(foot_linvel, axis=1)

    def get_knee_pos(self) -> np.ndarray:
        knee_pos = [self._backend.get_sensor_data(name) for name in self._cfg.sensor.knee_pos]
        return np.stack(knee_pos, axis=1)
