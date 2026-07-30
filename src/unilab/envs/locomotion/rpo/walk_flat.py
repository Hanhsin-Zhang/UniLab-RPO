"""RPO flat walking environment with the validated G1-style SAC profile."""

from __future__ import annotations

from copy import deepcopy
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.augmentation import SymmetryObsLayout
from unilab.base.backend import create_backend, env_backend_kwargs
from unilab.base.curriculum import EpisodeLengthTracker, PenaltyCurriculum
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dr import ResetPlan
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.commands import (
    Commands,
    sample_heading_commands,
    zero_small_xy_commands,
)
from unilab.envs.locomotion.common.domain_rand import DomainRandConfig
from unilab.envs.locomotion.common.dr_provider import LocomotionDRProvider
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.rpo.base import RPOBaseCfg, RPOBaseEnv
from unilab.utils.rotation import np_quat_apply_inverse


@dataclass
class RPOWalkDomainRandConfig(DomainRandConfig):
    com_offset_y: list[float] = field(default_factory=lambda: [-0.025, 0.025])
    com_offset_z: list[float] = field(default_factory=lambda: [-0.05, 0.05])

    randomize_kp: bool = True
    kp_multiplier_range: list[float] = field(default_factory=lambda: [0.9, 1.1])

    randomize_kd: bool = True
    kd_multiplier_range: list[float] = field(default_factory=lambda: [0.9, 1.1])

    randomize_reset_joint_qpos: bool = False
    reset_joint_qpos_range: list[float] = field(default_factory=lambda: [-0.05, 0.05])

    randomize_reset_base_qvel: bool = False
    reset_base_qvel_range: list[list[float]] = field(
        default_factory=lambda: [
            [-0.5, -0.5, -0.2, -0.52, -0.52, -0.78],
            [0.5, 0.5, 0.2, 0.52, 0.52, 0.78],
        ]
    )


@dataclass
class InitState:
    pos = [0.0, 0.0, 0.78]


def sample_gait_phase_pairs(rng, num_samples: int, mode: str) -> np.ndarray:
    if mode == "independent":
        return np.asarray(
            np.column_stack(
                [
                    rng.uniform(0.0, 2.0 * np.pi, size=(num_samples,)),
                    rng.uniform(0.0, 2.0 * np.pi, size=(num_samples,)),
                ]
            ),
            dtype=get_global_dtype(),
        )

    phase = rng.uniform(0.0, 2.0 * np.pi, size=(num_samples,))
    return np.asarray(np.column_stack([phase, phase + np.pi]), dtype=get_global_dtype())


def build_upper_body_pose_weights(pose_weights: list[float]) -> np.ndarray:
    weights = np.asarray(pose_weights, dtype=get_global_dtype()).copy()
    weights[:12] = 0.0
    return np.asarray(weights, dtype=get_global_dtype())


def compute_feet_phase_height_targets(
    gait_phase: np.ndarray, swing_height: float
) -> tuple[np.ndarray, np.ndarray]:
    def cubic_bezier_height(phi: np.ndarray, swing_height: float) -> np.ndarray:
        phi_normalized = np.fmod(phi + np.pi, 2 * np.pi) - np.pi
        x = (phi_normalized + np.pi) / (2 * np.pi)

        def cubic_bezier_interpolation(
            y_start: np.ndarray, y_end: np.ndarray, t: np.ndarray
        ) -> np.ndarray:
            y_diff = y_end - y_start
            bezier = t**3 + 3 * (t**2 * (1 - t))
            return np.asarray(y_start + y_diff * bezier, dtype=get_global_dtype())

        stance = cubic_bezier_interpolation(np.zeros_like(x), np.full_like(x, swing_height), 2 * x)
        swing = cubic_bezier_interpolation(
            np.full_like(x, swing_height), np.zeros_like(x), 2 * x - 1
        )
        return np.where(x <= 0.5, stance, swing)

    left_target = cubic_bezier_height(gait_phase[:, 0], swing_height)
    right_target = cubic_bezier_height(gait_phase[:, 1], swing_height)
    return left_target, right_target


LEFT_FOOT_CONTACT_SENSORS = [f"left_foot_contact_{i}" for i in range(5)]
RIGHT_FOOT_CONTACT_SENSORS = [f"right_foot_contact_{i}" for i in range(5)]


def _scalarize_sensor_values(sensor_values: np.ndarray) -> np.ndarray:
    sensor_array = np.asarray(sensor_values, dtype=get_global_dtype())
    if sensor_array.ndim == 1:
        return sensor_array
    if sensor_array.ndim == 2 and sensor_array.shape[1] == 1:
        return sensor_array[:, 0]
    raise ValueError(f"Expected scalar sensor values, got shape {sensor_array.shape}")


def compute_aggregated_foot_contact(backend: Any, sensor_names: list[str]) -> np.ndarray:
    contacts = [_scalarize_sensor_values(backend.get_sensor_data(name)) for name in sensor_names]
    return np.asarray(np.any(np.stack(contacts, axis=1) > 0.5, axis=1), dtype=np.bool_)


def compute_feet_phase_contact_targets(
    gait_phase: np.ndarray, swing_height: float
) -> tuple[np.ndarray, np.ndarray]:
    left_target, right_target = compute_feet_phase_height_targets(gait_phase, swing_height)
    contact_height_threshold = swing_height * 0.5
    return left_target <= contact_height_threshold, right_target <= contact_height_threshold


def compute_forward_speed_gate(linvel: np.ndarray, min_forward_speed: float) -> np.ndarray:
    forward_speed = np.maximum(linvel[:, 0], 0.0)
    return np.asarray(forward_speed >= min_forward_speed, dtype=get_global_dtype())


def compute_forward_command_mask(commands: np.ndarray) -> np.ndarray:
    return np.asarray(np.maximum(commands[:, 0], 0.0) > 1.0e-6, dtype=get_global_dtype())


def _scale_symmetric_range(values: list[float], scale: float) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    return np.asarray(arr * scale, dtype=np.float64).tolist()


def _scale_multiplier_range(values: list[float], scale: float) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    center = 1.0
    return np.asarray(center + (arr - center) * scale, dtype=np.float64).tolist()


def _scale_matrix_range(values: list[list[float]], scale: float) -> list[list[float]]:
    arr = np.asarray(values, dtype=np.float64)
    center = np.mean(arr, axis=0, keepdims=True)
    return np.asarray(center + (arr - center) * scale, dtype=np.float64).tolist()


@dataclass
class RPOWalkRewardConfig:
    scales: dict[str, float]
    tracking_sigma: float
    gait_frequency: float
    feet_phase_swing_height: float
    feet_phase_tracking_sigma: float
    base_height_target: float
    min_base_height: float
    max_tilt_deg: float
    undesired_contact_threshold: float = 1.0
    feet_distance_min: float = 0.16
    feet_distance_max: float = 0.5
    knee_distance_min: float = 0.18
    knee_distance_max: float = 0.35
    min_forward_speed_for_gait_reward: float = 0.0
    close_feet_threshold: float = 0.15
    feet_height_threshold: float = 0.02
    feet_motion_command_threshold: float = 0.01
    feet_motion_command_full_scale: float = 0.01
    feet_height_probe_reduction: str = "min"
    stand_still_command_threshold: float = 0.01
    stand_still_body_vel_threshold: float = 0.5
    stand_still_pos_weight: float = 1.0
    stand_still_vel_weight: float = 0.04
    pose_weights: list[float] = field(default_factory=lambda: [0.01] * 12 + [50.0] * 11)


@dataclass
class CurriculumConfig:
    enabled: bool = False
    initial_scale: float = 0.5
    min_scale: float = 0.5
    max_scale: float = 1.0
    level_down_threshold: float = 150.0
    level_up_threshold: float = 750.0
    degree: float = 0.001


@dataclass
class RPOWalkEnvCfg(RPOBaseCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "rpo" / "scene_flat.xml")
        )
    )
    max_episode_seconds: float = 20.0
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    reward_config: RPOWalkRewardConfig | None = None
    domain_rand: RPOWalkDomainRandConfig = field(default_factory=RPOWalkDomainRandConfig)
    gait_phase_init_mode: str = "offset_phase"
    reset_base_qvel_limit: float = 0.5
    actor_obs_history_length: int = 1
    critic_obs_history_length: int = 1
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)


class RPOWalkDomainRandomizationProvider(LocomotionDRProvider):
    def __init__(
        self,
        *,
        base_kp: np.ndarray | None = None,
        base_kd: np.ndarray | None = None,
        base_body_mass: np.ndarray | None = None,
        base_geom_friction: np.ndarray | None = None,
        ground_geom_id: int | None = None,
        base_dof_armature: np.ndarray | None = None,
    ):
        self._base_kp = base_kp
        self._base_kd = base_kd
        self._base_body_mass = base_body_mass
        self._base_geom_friction = base_geom_friction
        self._ground_geom_id = ground_geom_id
        self._base_dof_armature = base_dof_armature

    def _get_base_actuator_gains(self, env: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
        return self._base_kp, self._base_kd

    def _get_reset_randomization_baselines(
        self, env: Any
    ) -> tuple[np.ndarray | None, np.ndarray | None, int | None, np.ndarray | None]:
        return (
            self._base_body_mass,
            self._base_geom_friction,
            self._ground_geom_id,
            self._base_dof_armature,
        )

    def _get_qvel_limit(self, env: Any) -> float:
        if getattr(env.cfg.domain_rand, "randomize_reset_base_qvel", False):
            qvel_range = np.asarray(env.cfg.domain_rand.reset_base_qvel_range, dtype=np.float64)
            if qvel_range.shape != (2, 6):
                raise ValueError(
                    "domain_rand.reset_base_qvel_range must have shape (2, 6), "
                    f"got {qvel_range.shape}"
                )
            return float(np.max(np.abs(qvel_range)))
        return float(env.cfg.reset_base_qvel_limit)

    def _build_extra_info_updates(self, env: Any, num_reset: int) -> dict[str, np.ndarray]:
        updates = {"gait_phase": self._sample_gait_phase(env, num_reset)}
        if getattr(env.cfg.commands, "heading_command", False):
            updates["heading_commands"] = sample_heading_commands(env, num_reset)
        return updates

    def _sample_commands(self, env: Any, num_reset: int) -> np.ndarray:
        commands = super()._sample_commands(env, num_reset)
        zero_small_xy_commands(commands)
        standing_prob = float(getattr(env.cfg.commands, "rel_standing_envs", 0.0))
        if standing_prob > 0.0:
            standing = np.random.uniform(size=(num_reset,)) < min(standing_prob, 1.0)
            commands[standing] = 0.0
        if getattr(env.cfg.commands, "heading_command", False):
            commands[:, 2] = 0.0
        return commands

    def _sample_gait_phase(self, env: Any, num_reset: int) -> np.ndarray:
        mode = env.cfg.gait_phase_init_mode
        if mode == "independent":
            left = np.random.uniform(0.0, 2.0 * np.pi, size=(num_reset,))
            right = np.random.uniform(0.0, 2.0 * np.pi, size=(num_reset,))
            return np.asarray(np.column_stack([left, right]), dtype=get_global_dtype())

        phase = np.random.uniform(0.0, 2.0 * np.pi, size=(num_reset,))
        return np.asarray(np.column_stack([phase, phase + np.pi]), dtype=get_global_dtype())

    def _compute_reset_obs(
        self,
        env: Any,
        env_ids: Any,
        info_updates: Any,
        linvel: Any,
        gyro: Any,
        gravity: Any,
        dof_pos: Any,
        dof_vel: Any,
    ) -> dict[str, np.ndarray]:
        return env._compute_obs(info_updates, linvel, gyro, gravity, dof_pos, dof_vel)  # type: ignore[no-any-return]

    def build_reset_observation(
        self, env: Any, env_ids: np.ndarray, info_updates: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        env._current_feet_air_time[env_ids] = 0.0
        env._current_feet_contact_time[env_ids] = 0.0
        feet_contact = env._get_feet_contact()[env_ids]
        dof_vel = env.get_dof_vel()[env_ids]
        info_updates["feet_air_time"] = env._current_feet_air_time[env_ids].copy()
        info_updates["feet_contact_time"] = env._current_feet_contact_time[env_ids].copy()
        info_updates["feet_contact"] = feet_contact.copy()
        info_updates["feet_height"] = env._get_foot_height_from_probes()[env_ids].copy()
        info_updates["torques"] = env._get_joint_torque()[env_ids].copy()
        info_updates["qacc"] = np.zeros((len(env_ids), env._num_action), dtype=get_global_dtype())
        info_updates["prev_dof_vel"] = dof_vel.copy()
        linvel = env.get_local_linvel()[env_ids]
        gyro = env.get_gyro()[env_ids]
        gravity = env._backend.get_sensor_data(env._cfg.sensor.upvector)[env_ids]
        dof_pos = env.get_dof_pos()[env_ids]
        current_obs = env._compute_obs(info_updates, linvel, gyro, gravity, dof_pos, dof_vel)
        env._fill_histories(env_ids, current_obs["obs"], current_obs["critic"])
        return env._snapshot_histories(env_ids)

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        plan = super().build_reset_plan(env, env_ids)
        num_reset = len(env_ids)
        qpos = np.asarray(plan.qpos, dtype=get_global_dtype()).copy()
        qvel = np.asarray(plan.qvel, dtype=get_global_dtype()).copy()
        info_updates = dict(plan.info_updates)
        domain_rand = env.cfg.domain_rand

        if getattr(domain_rand, "randomize_reset_base_qvel", False) and num_reset > 0:
            qvel_range = np.asarray(domain_rand.reset_base_qvel_range, dtype=np.float64)
            if qvel_range.shape != (2, 6):
                raise ValueError(
                    "domain_rand.reset_base_qvel_range must have shape (2, 6), "
                    f"got {qvel_range.shape}"
                )
            low = np.minimum(qvel_range[0], qvel_range[1])
            high = np.maximum(qvel_range[0], qvel_range[1])
            qvel[:, 0:6] = np.asarray(
                np.random.uniform(low=low, high=high, size=(num_reset, 6)),
                dtype=qvel.dtype,
            )

        if getattr(domain_rand, "randomize_reset_joint_qpos", False) and num_reset > 0:
            low, high = domain_rand.reset_joint_qpos_range
            low_f = float(min(low, high))
            high_f = float(max(low, high))
            joint_qpos = qpos[:, -env._num_action :]
            joint_qpos += np.asarray(
                np.random.uniform(low_f, high_f, size=(num_reset, env._num_action)),
                dtype=joint_qpos.dtype,
            )
            if env._joint_range is not None:
                lower = env._joint_range[:, 0]
                upper = env._joint_range[:, 1]
                np.clip(joint_qpos, lower, upper, out=joint_qpos)

        return ResetPlan(
            env_ids=plan.env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=plan.randomization,
        )


class RPOWalkEnv(RPOBaseEnv):
    _cfg: RPOWalkEnvCfg
    _reward_cfg: Any

    def __init__(self, cfg: RPOWalkEnvCfg, num_envs=1, backend_type="mujoco"):
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        backend = create_backend(
            backend_type,
            cfg.scene,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.asset.base_name,
            push_body_name=cfg.domain_rand.push_body_name,
            **env_backend_kwargs(cfg),
        )
        super().__init__(cfg, backend, num_envs)
        self._enable_reward_log = True
        self._reward_cfg = cfg.reward_config
        dtype = get_global_dtype()
        self._actor_hist_len = max(1, int(cfg.actor_obs_history_length))
        self._critic_hist_len = max(1, int(cfg.critic_obs_history_length))
        self._actor_obs_dim = 3 + 3 + self._num_action + self._num_action + self._num_action + 3 + 2
        self._critic_obs_dim = self._actor_obs_dim + 3 + 2 + 2 + 2 + 2 + self._num_action + self._num_action
        self._actor_hist = np.zeros(
            (num_envs, self._actor_hist_len, self._actor_obs_dim), dtype=dtype
        )
        self._critic_hist = np.zeros(
            (num_envs, self._critic_hist_len, self._critic_obs_dim), dtype=dtype
        )
        self._feet_pos_b = np.zeros((num_envs, 2, 3), dtype=dtype)
        self._knee_pos_b = np.zeros((num_envs, 2, 3), dtype=dtype)
        self._current_feet_air_time = np.zeros((num_envs, 2), dtype=dtype)
        self._current_feet_contact_time = np.zeros((num_envs, 2), dtype=dtype)
        joint_range = self._backend.get_joint_range()
        self._joint_range = (
            np.asarray(joint_range, dtype=dtype) if joint_range is not None else None
        )

        self._gait_phase_delta = float(
            2.0 * math.pi * self._reward_cfg.gait_frequency * cfg.ctrl_dt
        )
        self._pose_weights = np.array(self._reward_cfg.pose_weights, dtype=get_global_dtype())
        if self._pose_weights.shape[0] != self._num_action:
            raise ValueError("pose_weights length mismatch")
        self._upper_body_pose_weights = build_upper_body_pose_weights(self._reward_cfg.pose_weights)
        self._episode_tracker: EpisodeLengthTracker | None = None
        self._penalty_curriculum: PenaltyCurriculum | None = None
        if cfg.curriculum.enabled:
            self._episode_tracker = EpisodeLengthTracker(num_envs)
            self._penalty_curriculum = PenaltyCurriculum(
                self,
                enabled=True,
                initial_scale=cfg.curriculum.initial_scale,
                min_scale=cfg.curriculum.min_scale,
                max_scale=cfg.curriculum.max_scale,
                level_down_threshold=cfg.curriculum.level_down_threshold,
                level_up_threshold=cfg.curriculum.level_up_threshold,
                degree=cfg.curriculum.degree,
            )
        self._domain_rand_curriculum_base: RPOWalkDomainRandConfig | None = None
        if cfg.curriculum.enabled:
            self._domain_rand_curriculum_base = deepcopy(cfg.domain_rand)
            self._apply_domain_rand_curriculum_scale(cfg.curriculum.initial_scale)

        self._init_reward_functions()
        base_kp, base_kd = backend.get_actuator_gains()
        dr_provider = RPOWalkDomainRandomizationProvider(
            base_kp=np.asarray(base_kp, dtype=np.float64),
            base_kd=np.asarray(base_kd, dtype=np.float64),
            base_body_mass=np.asarray(backend.get_body_mass(), dtype=np.float64),
            base_geom_friction=np.asarray(backend.get_geom_friction(), dtype=np.float64),
            ground_geom_id=int(backend.get_geom_id(cfg.asset.ground)),
            base_dof_armature=np.asarray(backend.get_dof_armature(), dtype=np.float64),
        )
        self._init_domain_randomization(dr_provider)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {
            "obs": self._actor_obs_dim * self._actor_hist_len,
            "critic": self._critic_obs_dim * self._critic_hist_len,
        }

    def _init_reward_functions(self):
        self._reward_fns: dict[str, Any] = {
            "tracking_lin_vel": rewards.tracking_lin_vel,
            "tracking_ang_vel": rewards.tracking_ang_vel,
            "forward_progress": rewards.forward_progress,
            "under_speed": rewards.under_speed,
            "lin_vel_z": rewards.lin_vel_z,
            "orientation": rewards.orientation,
            "penalty_orientation": rewards.orientation,
            "ang_vel_xy": rewards.ang_vel_xy,
            "penalty_ang_vel_xy": rewards.ang_vel_xy,
            "action_rate": rewards.action_rate,
            "penalty_action_rate": rewards.action_rate,
            "base_height": rewards.base_height,
            "pose": rewards.weighted_pose,
            "upper_body_pose": self._reward_upper_body_pose,
            "penalty_close_feet_xy": self._reward_close_feet_xy,
            "penalty_feet_ori": self._reward_feet_ori,
            "feet_distance": self._reward_feet_distance,
            "knee_distance": self._reward_knee_distance,
            "undesired_contacts": self._reward_undesired_contacts,
            "feet_phase": self._reward_feet_phase,
            "feet_phase_contrast": self._reward_feet_phase_contrast,
            "feet_phase_contact": self._reward_feet_phase_contact,
            "feet_double_stance": self._reward_feet_double_stance,
            "feet_air_time": self._reward_feet_air_time,
            "feet_height": self._reward_feet_height,
            "feet_contact_without_cmd": self._reward_feet_contact_without_cmd,
            "alive": rewards.alive,
            "stand_still": self._reward_stand_still,
        }

    def _terrain_relative_base_height(self) -> np.ndarray:
        return np.asarray(self._backend.get_base_pos()[:, 2], dtype=get_global_dtype())

    def update_state(self, state: NpEnvState) -> NpEnvState:
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data(self._cfg.sensor.upvector)
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        feet_contact = self._get_feet_contact()
        self._update_feet_timing(feet_contact)
        joint_torque = self._get_joint_torque()
        prev_dof_vel = state.info.get("prev_dof_vel")
        if prev_dof_vel is None or not isinstance(prev_dof_vel, np.ndarray) or prev_dof_vel.shape != dof_vel.shape:
            prev_dof_vel = None
        joint_acc = self._compute_joint_acc(dof_vel, prev_dof_vel)
        base_pos = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())
        base_quat = np.asarray(self._backend.get_base_quat(), dtype=get_global_dtype())
        foot_pos = np.asarray(self.get_foot_pos(), dtype=get_global_dtype())
        knee_pos = np.asarray(self.get_knee_pos(), dtype=get_global_dtype())
        self._feet_pos_b = self._body_pos_b_from_world(
            foot_pos, base_pos=base_pos, base_quat=base_quat
        )
        self._knee_pos_b = self._body_pos_b_from_world(
            knee_pos, base_pos=base_pos, base_quat=base_quat
        )
        state.info["feet_height"] = self._get_foot_height_from_probes()
        state.info["feet_air_time"] = self._current_feet_air_time.copy()
        state.info["feet_contact_time"] = self._current_feet_contact_time.copy()
        state.info["feet_contact"] = feet_contact.copy()
        state.info["torques"] = joint_torque.copy()
        state.info["qacc"] = joint_acc.copy()
        state.info["prev_dof_vel"] = dof_vel.copy()

        max_tilt_rad = np.deg2rad(self._reward_cfg.max_tilt_deg)
        tilt = np.arccos(np.clip(gravity[:, 2], -1, 1))
        terminated = np.logical_or(
            tilt > max_tilt_rad,
            self._terrain_relative_base_height() < self._reward_cfg.min_base_height,
        )
        reward = self._compute_reward(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        current_obs = self._compute_obs(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        self._push_histories(None, current_obs["obs"], current_obs["critic"])
        obs = self._snapshot_histories()

        state = state.replace(obs=obs, reward=reward, terminated=terminated)

        done = state.terminated | state.truncated
        if self._episode_tracker is None or self._penalty_curriculum is None or not np.any(done):
            return state

        done_indices = np.where(done)[0]
        episode_lengths = state.info["steps"][done_indices] + 1
        self._episode_tracker.update(episode_lengths)
        self._penalty_curriculum.update(self._episode_tracker.average_length)
        self._apply_domain_rand_curriculum_scale(self._penalty_curriculum.current_scale)

        if "log" not in state.info:
            state.info["log"] = {}
        state.info["log"]["curriculum/average_episode_length"] = float(
            self._episode_tracker.average_length
        )
        state.info["log"]["curriculum/penalty_scale"] = float(
            self._penalty_curriculum.current_scale
        )
        state.info["log"]["curriculum/domain_rand_scale"] = float(
            self._penalty_curriculum.current_scale
        )
        return state

    def _apply_domain_rand_curriculum_scale(self, scale: float) -> None:
        base = self._domain_rand_curriculum_base
        if base is None:
            return
        curr = self._cfg.domain_rand
        curr.added_mass_range = _scale_symmetric_range(base.added_mass_range, scale)
        curr.body_mass_multiplier_range = _scale_multiplier_range(
            base.body_mass_multiplier_range, scale
        )
        curr.com_offset_x = _scale_symmetric_range(base.com_offset_x, scale)
        curr.com_offset_y = _scale_symmetric_range(base.com_offset_y, scale)
        curr.com_offset_z = _scale_symmetric_range(base.com_offset_z, scale)
        curr.gravity_range = _scale_matrix_range(base.gravity_range, scale)
        curr.ground_friction_multiplier_range = _scale_multiplier_range(
            base.ground_friction_multiplier_range, scale
        )
        curr.dof_armature_multiplier_range = _scale_multiplier_range(
            base.dof_armature_multiplier_range, scale
        )
        curr.kp_multiplier_range = _scale_multiplier_range(base.kp_multiplier_range, scale)
        curr.kd_multiplier_range = _scale_multiplier_range(base.kd_multiplier_range, scale)
        curr.max_force = _scale_symmetric_range(base.max_force, scale)
        curr.reset_joint_qpos_range = _scale_symmetric_range(base.reset_joint_qpos_range, scale)
        curr.reset_base_qvel_range = _scale_matrix_range(base.reset_base_qvel_range, scale)

    def _compute_obs(
        self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel
    ) -> dict[str, np.ndarray]:
        batch_size = dof_pos.shape[0]
        noise_cfg = self._cfg.noise_config
        diff = dof_pos - self.default_angles
        command = info["commands"]
        last_actions = info.get("current_actions", np.zeros_like(diff))
        gait_phase = info.get("gait_phase", np.zeros((batch_size, 2), dtype=get_global_dtype()))
        feet_contact = np.asarray(
            info.get("feet_contact", np.zeros((batch_size, 2), dtype=get_global_dtype())),
            dtype=get_global_dtype(),
        )
        feet_air_time = np.asarray(
            info.get("feet_air_time", np.zeros((batch_size, 2), dtype=get_global_dtype())),
            dtype=get_global_dtype(),
        )
        feet_contact_time = np.asarray(
            info.get("feet_contact_time", np.zeros((batch_size, 2), dtype=get_global_dtype())),
            dtype=get_global_dtype(),
        )
        feet_height = np.asarray(
            info.get("feet_height", np.zeros((batch_size, 2), dtype=get_global_dtype())),
            dtype=get_global_dtype(),
        )
        joint_acc = np.asarray(
            info.get("qacc", np.zeros_like(dof_vel)),
            dtype=get_global_dtype(),
        )
        joint_torque = np.asarray(
            info.get("torques", np.zeros_like(dof_vel)),
            dtype=get_global_dtype(),
        )
        walk_profile = self._uses_walk_observation_profile()

        noisy_gyro = self._obs_noise(gyro, noise_cfg.scale_gyro)
        noisy_gravity = self._obs_noise(gravity, noise_cfg.scale_gravity)
        noisy_diff = self._obs_noise(diff, noise_cfg.scale_joint_angle)
        noisy_dof_vel = self._obs_noise(dof_vel, noise_cfg.scale_joint_vel)
        actor_gyro_scale = 0.25 if walk_profile else 1.0
        actor_dof_vel_scale = 0.05 if walk_profile else 1.0

        actor = np.concatenate(
            [
                noisy_gyro * actor_gyro_scale,
                -noisy_gravity,
                noisy_diff,
                noisy_dof_vel * actor_dof_vel_scale,
                last_actions,
                command,
                gait_phase,
            ],
            axis=1,
            dtype=get_global_dtype(),
        )

        critic_gyro_scale = 0.25 if walk_profile else 1.0
        critic_dof_vel_scale = 0.05 if walk_profile else 1.0
        critic_linvel_scale = 2.0 if walk_profile else 1.0
        critic_time_scale = 5.0
        critic_feet_height_scale = 20.0
        critic_joint_acc_scale = 1.0 / 400.0
        critic_joint_torque_scale = 1.0 / 40.0
        critic_base = np.concatenate(
            [
                gyro * critic_gyro_scale,
                -gravity,
                diff,
                dof_vel * critic_dof_vel_scale,
                last_actions,
                command,
                gait_phase,
            ],
            axis=1,
            dtype=get_global_dtype(),
        )
        critic = np.concatenate(
            [
                critic_base,
                np.asarray(linvel * critic_linvel_scale, dtype=get_global_dtype()),
                feet_contact,
                feet_air_time * critic_time_scale,
                feet_contact_time * critic_time_scale,
                feet_height * critic_feet_height_scale,
                joint_acc * critic_joint_acc_scale,
                joint_torque * critic_joint_torque_scale,
            ],
            axis=1,
            dtype=get_global_dtype(),
        )

        return {"obs": actor, "critic": critic}

    def _uses_walk_observation_profile(self) -> bool:
        scales = getattr(getattr(self, "_reward_cfg", None), "scales", None)
        if scales is None:
            reward_cfg = getattr(self._cfg, "reward_config", None)
            scales = getattr(reward_cfg, "scales", None)

        if scales is not None:
            if any(
                key in scales
                for key in (
                    "penalty_orientation",
                    "penalty_ang_vel_xy",
                    "penalty_action_rate",
                    "alive",
                )
            ):
                return True
            if any(key in scales for key in ("orientation", "ang_vel_xy", "action_rate")):
                return False

        curriculum = getattr(self._cfg, "curriculum", None)
        return bool(curriculum is not None and curriculum.enabled)

    def _actor_symmetry_obs_layout(self) -> SymmetryObsLayout:
        return (
            ("gyro", 3),
            ("gravity", 3),
            ("dof_pos", self._num_action),
            ("dof_vel", self._num_action),
            ("actions", self._num_action),
            ("command", 3),
            ("gait_phase", 2),
        )

    @staticmethod
    def _repeat_symmetry_obs_layout(
        layout: SymmetryObsLayout, history_length: int
    ) -> SymmetryObsLayout:
        return tuple(entry for _ in range(max(1, int(history_length))) for entry in layout)

    def get_symmetry_obs_layouts(self) -> dict[str, SymmetryObsLayout]:
        actor_layout_single = self._actor_symmetry_obs_layout()
        critic_layout_single = (
            *actor_layout_single,
            ("linvel", 3),
            ("feet_contact", 2),
            ("feet_air_time", 2),
            ("feet_contact_time", 2),
            ("feet_height", 2),
            ("joint_acc", self._num_action),
            ("joint_torque", self._num_action),
        )
        return {
            "obs": self._repeat_symmetry_obs_layout(actor_layout_single, self._actor_hist_len),
            "critic": self._repeat_symmetry_obs_layout(
                critic_layout_single, self._critic_hist_len
            ),
        }

    def build_symmetry_augmentation(self, *, device: str):
        if self._backend.backend_type != "mujoco":
            return None
        from unilab.envs.locomotion.rpo.symmetry import RPOSymmetryAugmentation

        return RPOSymmetryAugmentation(
            self._backend.model,
            self.get_symmetry_obs_layouts(),
            device=device,
        )

    def _build_reward_context(
        self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel
    ) -> RewardContext:
        return RewardContext(
            info=info,
            linvel=linvel,
            gyro=gyro,
            dof_pos=dof_pos,
            num_envs=self._num_envs,
            default_angles=self.default_angles,
            tracking_sigma=self._reward_cfg.tracking_sigma,
            base_height_target=self._reward_cfg.base_height_target,
            base_height=self._backend.get_base_pos()[:, 2],
            gravity=gravity,
            dof_vel=dof_vel,
            pose_weights=self._pose_weights,
        )

    def _compute_reward(self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel) -> np.ndarray:
        cfg = self._reward_cfg
        ctx = self._build_reward_context(info, linvel, gyro, gravity, dof_pos, dof_vel)
        return rewards.run_reward_dispatch(
            scales=cfg.scales,
            fns=self._reward_fns,
            ctx=ctx,
            info=info,
            enable_log=self._enable_reward_log,
            ctrl_dt=self._cfg.ctrl_dt,
        )

    def _reward_feet_phase(self, ctx: RewardContext):
        """Reward gait phase tracking by encouraging the expected swing-foot height."""
        left_foot = self._backend.get_sensor_data("left_foot_pos")
        right_foot = self._backend.get_sensor_data("right_foot_pos")
        gait_phase = ctx.info.get(
            "gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        )
        swing_height = self._reward_cfg.feet_phase_swing_height
        left_target, right_target = compute_feet_phase_height_targets(gait_phase, swing_height)
        left_error = np.square(left_foot[:, 2] - left_target)
        right_error = np.square(right_foot[:, 2] - right_target)
        reward = np.exp(-(left_error + right_error) / self._reward_cfg.feet_phase_tracking_sigma)
        commands = np.asarray(
            ctx.info.get("commands", np.zeros((self._num_envs, 3), dtype=get_global_dtype())),
            dtype=get_global_dtype(),
        )
        return np.asarray(
            reward * self._gait_reward_gate(ctx.linvel) * self._feet_motion_command_scale(commands),
            dtype=get_global_dtype(),
        )

    def _gait_reward_gate(self, linvel: np.ndarray) -> np.ndarray:
        min_forward_speed = getattr(self._reward_cfg, "min_forward_speed_for_gait_reward", 0.0)
        return compute_forward_speed_gate(linvel, min_forward_speed)

    def _feet_motion_command_scale(self, commands: np.ndarray) -> np.ndarray:
        command_arr = np.asarray(commands, dtype=get_global_dtype())
        command_norm = np.linalg.norm(command_arr[:, :2], axis=1) + np.abs(command_arr[:, 2])
        lower = float(self._reward_cfg.feet_motion_command_threshold)
        upper = float(getattr(self._reward_cfg, "feet_motion_command_full_scale", lower))
        if upper <= lower:
            return np.asarray(command_norm > lower, dtype=get_global_dtype())
        return np.asarray(
            np.clip((command_norm - lower) / (upper - lower), 0.0, 1.0),
            dtype=get_global_dtype(),
        )

    def _reward_feet_phase_contrast(self, ctx: RewardContext):
        left_foot = self._backend.get_sensor_data("left_foot_pos")
        right_foot = self._backend.get_sensor_data("right_foot_pos")
        gait_phase = ctx.info.get(
            "gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        )
        swing_height = self._reward_cfg.feet_phase_swing_height
        left_target, right_target = compute_feet_phase_height_targets(gait_phase, swing_height)
        actual_delta = left_foot[:, 2] - right_foot[:, 2]
        target_delta = left_target - right_target
        error = np.square(actual_delta - target_delta)
        reward = np.exp(-error / self._reward_cfg.feet_phase_tracking_sigma)
        return np.asarray(reward * self._gait_reward_gate(ctx.linvel), dtype=get_global_dtype())

    def _reward_feet_phase_contact(self, ctx: RewardContext):
        gait_phase = ctx.info.get(
            "gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        )
        swing_height = self._reward_cfg.feet_phase_swing_height
        left_target_contact, right_target_contact = compute_feet_phase_contact_targets(
            gait_phase, swing_height
        )
        left_contact = compute_aggregated_foot_contact(self._backend, LEFT_FOOT_CONTACT_SENSORS)
        right_contact = compute_aggregated_foot_contact(self._backend, RIGHT_FOOT_CONTACT_SENSORS)
        left_match = np.asarray(left_contact == left_target_contact, dtype=get_global_dtype())
        right_match = np.asarray(right_contact == right_target_contact, dtype=get_global_dtype())
        reward = np.asarray(0.5 * (left_match + right_match), dtype=get_global_dtype())
        return np.asarray(reward * self._gait_reward_gate(ctx.linvel), dtype=get_global_dtype())

    def _reward_feet_double_stance(self, ctx: RewardContext):
        commands = ctx.info.get("commands", np.zeros((self._num_envs, 3), dtype=get_global_dtype()))
        left_contact = compute_aggregated_foot_contact(self._backend, LEFT_FOOT_CONTACT_SENSORS)
        right_contact = compute_aggregated_foot_contact(self._backend, RIGHT_FOOT_CONTACT_SENSORS)
        double_stance = np.asarray(
            np.logical_and(left_contact, right_contact), dtype=get_global_dtype()
        )
        return np.asarray(
            double_stance * compute_forward_command_mask(commands), dtype=get_global_dtype()
        )

    def _reward_feet_ori(self, ctx: RewardContext):
        left_foot_quat = self._backend.get_sensor_data("left_foot_quat")
        right_foot_quat = self._backend.get_sensor_data("right_foot_quat")
        return (
            np.square(left_foot_quat[:, 1])
            + np.square(left_foot_quat[:, 2])
            + np.square(right_foot_quat[:, 1])
            + np.square(right_foot_quat[:, 2])
        )

    def _reward_close_feet_xy(self, ctx: RewardContext):
        left_foot = self._backend.get_sensor_data("left_foot_pos")
        right_foot = self._backend.get_sensor_data("right_foot_pos")
        feet_dist = np.linalg.norm(left_foot[:, :2] - right_foot[:, :2], axis=1)
        return np.where(
            feet_dist < self._reward_cfg.close_feet_threshold,
            np.square(feet_dist - self._reward_cfg.close_feet_threshold),
            0.0,
        )

    def _reward_feet_air_time(self, ctx: RewardContext):
        air_time = ctx.info.get(
            "feet_air_time", np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        )
        in_range = (air_time > 0.05) & (air_time < 0.35)
        reward = np.sum(in_range.astype(float), axis=1)
        commands = np.asarray(
            ctx.info.get("commands", np.zeros((self._num_envs, 3), dtype=get_global_dtype())),
            dtype=get_global_dtype(),
        )
        return np.asarray(
            reward * self._feet_motion_command_scale(commands),
            dtype=get_global_dtype(),
        )

    def _get_feet_contact(self) -> np.ndarray:
        left_contact = compute_aggregated_foot_contact(self._backend, LEFT_FOOT_CONTACT_SENSORS)
        right_contact = compute_aggregated_foot_contact(self._backend, RIGHT_FOOT_CONTACT_SENSORS)
        return np.stack([left_contact, right_contact], axis=1)

    def _update_feet_timing(self, feet_contact: np.ndarray) -> None:
        self._current_feet_air_time[~feet_contact] += float(self._cfg.ctrl_dt)
        self._current_feet_air_time[feet_contact] = 0.0
        self._current_feet_contact_time[feet_contact] += float(self._cfg.ctrl_dt)
        self._current_feet_contact_time[~feet_contact] = 0.0

    def _push_histories(
        self,
        env_ids: np.ndarray | None,
        actor_obs: np.ndarray,
        critic_obs: np.ndarray,
    ) -> None:
        sel = slice(None) if env_ids is None else env_ids
        self._actor_hist[sel, :-1] = self._actor_hist[sel, 1:]
        self._actor_hist[sel, -1] = actor_obs
        self._critic_hist[sel, :-1] = self._critic_hist[sel, 1:]
        self._critic_hist[sel, -1] = critic_obs

    def _fill_histories(
        self,
        env_ids: np.ndarray,
        actor_obs: np.ndarray,
        critic_obs: np.ndarray,
    ) -> None:
        self._actor_hist[env_ids, :] = actor_obs[:, None, :]
        self._critic_hist[env_ids, :] = critic_obs[:, None, :]

    def _snapshot_histories(
        self,
        env_ids: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        sel = slice(None) if env_ids is None else env_ids
        # reshape(len(...), -1) flattens (batch, history, dim) -> (batch, history * dim).
        # copy() is required here because off-policy workers keep previous obs across
        # env.step() calls; returning a view into the mutable history buffers would let
        # later steps overwrite earlier replay transitions in-place.
        return {
            "obs": self._actor_hist[sel].reshape(len(self._actor_hist[sel]), -1).copy(),
            "critic": self._critic_hist[sel].reshape(len(self._critic_hist[sel]), -1).copy(),
        }

    def _get_foot_height_from_probes(self) -> np.ndarray:
        probe_pos = self.get_foot_probe_pos()
        probe_height = np.asarray(probe_pos[:, :, :, 2], dtype=get_global_dtype())
        reduction = str(self._reward_cfg.feet_height_probe_reduction).strip().lower()
        if reduction == "mean":
            return np.asarray(np.mean(probe_height, axis=2), dtype=get_global_dtype())
        if reduction == "min":
            return np.asarray(np.min(probe_height, axis=2), dtype=get_global_dtype())
        raise ValueError(
            "reward.feet_height_probe_reduction must be either 'min' or 'mean', "
            f"got {self._reward_cfg.feet_height_probe_reduction!r}"
        )

    def _get_joint_torque(self) -> np.ndarray:
        return np.asarray(
            self._backend.get_sensor_data_batch(self._cfg.sensor.actuator_frc),
            dtype=get_global_dtype(),
        )

    def _compute_joint_acc(
        self, dof_vel: np.ndarray, prev_dof_vel: np.ndarray | None
    ) -> np.ndarray:
        if prev_dof_vel is None:
            return np.zeros_like(dof_vel)
        return np.asarray((dof_vel - prev_dof_vel) / float(self._cfg.ctrl_dt), dtype=get_global_dtype())

    def _reward_feet_height(self, ctx: RewardContext):
        left_contact = compute_aggregated_foot_contact(self._backend, LEFT_FOOT_CONTACT_SENSORS)
        right_contact = compute_aggregated_foot_contact(self._backend, RIGHT_FOOT_CONTACT_SENSORS)
        contacts = np.stack([left_contact, right_contact], axis=1)
        single_stance = np.sum(contacts.astype(np.int32), axis=1) == 1
        foot_height = np.clip(self._get_foot_height_from_probes(), 0.0, 1.0)
        threshold = float(self._reward_cfg.feet_height_threshold)
        reward_per_foot = np.clip(foot_height / max(threshold, 1.0e-6), 0.0, 1.0).astype(
            get_global_dtype()
        )
        reward = np.where((~contacts) & single_stance[:, None], reward_per_foot, 0.0).sum(axis=1)
        commands = np.asarray(
            ctx.info.get("commands", np.zeros((self._num_envs, 3), dtype=get_global_dtype())),
            dtype=get_global_dtype(),
        )
        upright = rewards.upright_scale(ctx.gravity, ctx.num_envs)
        return np.asarray(
            reward * self._feet_motion_command_scale(commands) * upright,
            dtype=get_global_dtype(),
        )

    def _reward_feet_contact_without_cmd(self, ctx: RewardContext):
        commands = np.asarray(
            ctx.info.get("commands", np.zeros((self._num_envs, 3), dtype=get_global_dtype())),
            dtype=get_global_dtype(),
        )
        command_norm = np.linalg.norm(commands[:, :2], axis=1) + np.abs(commands[:, 2])
        still = command_norm < float(self._reward_cfg.stand_still_command_threshold)
        contacts = np.asarray(
            ctx.info.get("feet_contact", np.zeros((self._num_envs, 2), dtype=np.bool_)), dtype=np.bool_
        )
        both_contact = np.sum(contacts.astype(np.int32), axis=1) == 2
        upright = rewards.upright_scale(ctx.gravity, ctx.num_envs)
        return np.asarray(still * both_contact * upright, dtype=get_global_dtype())

    def _reward_stand_still(self, ctx: RewardContext):
        commands = np.asarray(
            ctx.info.get("commands", np.zeros((self._num_envs, 3), dtype=get_global_dtype())),
            dtype=get_global_dtype(),
        )
        command_norm = np.linalg.norm(commands[:, :2], axis=1) + np.abs(commands[:, 2])
        body_lin_vel = np.linalg.norm(ctx.linvel[:, :2], axis=1)
        body_ang_vel = np.abs(ctx.gyro[:, 2])
        body_vel = body_lin_vel + body_ang_vel
        pos_reward = float(self._reward_cfg.stand_still_pos_weight) * np.sum(
            np.abs(ctx.dof_pos - ctx.default_angles), axis=1
        )
        assert ctx.dof_vel is not None
        vel_reward = float(self._reward_cfg.stand_still_vel_weight) * np.sum(np.abs(ctx.dof_vel), axis=1)
        penalty = np.where(
            (command_norm > float(self._reward_cfg.stand_still_command_threshold))
            | (body_vel > float(self._reward_cfg.stand_still_body_vel_threshold)),
            0.0,
            pos_reward + vel_reward,
        )
        upright = rewards.upright_scale(ctx.gravity, ctx.num_envs)
        return np.asarray(penalty * upright, dtype=get_global_dtype())

    def _reward_upper_body_pose(self, ctx: RewardContext):
        diff = ctx.dof_pos - self.default_angles
        return np.asarray(
            np.sum(self._upper_body_pose_weights * np.square(diff), axis=1),
            dtype=get_global_dtype(),
        )

    def _body_pos_b_from_world(
        self,
        body_pos_w: np.ndarray,
        *,
        base_pos: np.ndarray,
        base_quat: np.ndarray,
    ) -> np.ndarray:
        rel = body_pos_w - base_pos[:, None, :]
        flat_rel = rel.reshape(-1, 3)
        q_rep = np.repeat(base_quat, rel.shape[1], axis=0)
        flat_b = np_quat_apply_inverse(q_rep, flat_rel)
        return np.asarray(flat_b.reshape(rel.shape), dtype=get_global_dtype())

    def _body_distance_y_exp(self, pos_b: np.ndarray, *, min_dist: float, max_dist: float) -> np.ndarray:
        distance = np.abs(pos_b[:, 0, 1] - pos_b[:, 1, 1])
        d_min = np.clip(distance - float(min_dist), -0.5, 0.0)
        d_max = np.clip(distance - float(max_dist), 0.0, 0.5)
        return np.asarray(
            (np.exp(-np.abs(d_min) * 100.0) + np.exp(-np.abs(d_max) * 100.0)) / 2.0,
            dtype=get_global_dtype(),
        )

    def _reward_feet_distance(self, ctx: RewardContext):
        del ctx
        return self._body_distance_y_exp(
            self._feet_pos_b,
            min_dist=float(self._reward_cfg.feet_distance_min),
            max_dist=float(self._reward_cfg.feet_distance_max),
        )

    def _reward_knee_distance(self, ctx: RewardContext):
        del ctx
        return self._body_distance_y_exp(
            self._knee_pos_b,
            min_dist=float(self._reward_cfg.knee_distance_min),
            max_dist=float(self._reward_cfg.knee_distance_max),
        )

    def _contact_count_from_sensors(
        self,
        sensor_names: tuple[str, ...],
        *,
        threshold: float,
    ) -> np.ndarray:
        if not sensor_names:
            return np.zeros((self._num_envs,), dtype=get_global_dtype())
        forces = [np.asarray(self._backend.get_sensor_data(name), dtype=get_global_dtype()) for name in sensor_names]
        stacked = np.stack(forces, axis=1)
        exceeded = np.linalg.norm(stacked, axis=2) > float(threshold)
        return np.asarray(np.sum(exceeded, axis=1), dtype=get_global_dtype())

    def _reward_undesired_contacts(self, ctx: RewardContext):
        del ctx
        threshold = float(getattr(self._reward_cfg, "undesired_contact_threshold", 1.0))
        return self._contact_count_from_sensors(
            self._cfg.sensor.undesired_contact_force,
            threshold=threshold,
        )

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        state.info["last_actions"] = state.info.get("current_actions", np.zeros_like(actions))
        state.info["current_actions"] = actions

        gait_phase = state.info.get(
            "gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        )
        gait_phase[:, 0] = (gait_phase[:, 0] + self._gait_phase_delta) % (2 * np.pi)
        gait_phase[:, 1] = (gait_phase[:, 1] + self._gait_phase_delta) % (2 * np.pi)
        state.info["gait_phase"] = gait_phase

        ctrl: np.ndarray = actions * self._cfg.control_config.action_scale + self.default_angles
        return ctrl


def _walk_curriculum() -> CurriculumConfig:
    return CurriculumConfig(
        enabled=True,
        initial_scale=0.5,
        min_scale=0.5,
        max_scale=1.0,
        level_down_threshold=150.0,
        level_up_threshold=750.0,
        degree=0.001,
    )


@dataclass
class RPOWalkControlConfig:
    action_scale: float = 1.0
    simulate_action_latency: bool = False


@dataclass
class RPOWalkProfileRewardConfig(RPOWalkRewardConfig):
    """Reward profile validated for the RPO SAC walking migration."""


@registry.envcfg("RPOWalkFlat")
@dataclass
class RPOWalkFlatCfg(RPOWalkEnvCfg):
    reward_config: RPOWalkProfileRewardConfig | None = None
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "rpo" / "scene_flat.xml")
        )
    )
    control_config: RPOWalkControlConfig = field(default_factory=RPOWalkControlConfig)  # type: ignore[assignment]
    curriculum: CurriculumConfig = field(default_factory=_walk_curriculum)


registry.register_env("RPOWalkFlat", RPOWalkEnv, sim_backend="mujoco")
