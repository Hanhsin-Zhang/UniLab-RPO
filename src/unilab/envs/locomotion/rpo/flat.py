from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend, env_backend_kwargs
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.commands import Commands
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.rpo.base import RPOBaseCfg, RPOBaseEnv
from unilab.utils.rotation import np_quat_apply, np_quat_apply_inverse, np_yaw_quat


@dataclass
class RPOFlatRewardConfig:
    scales: dict[str, float]
    tracking_sigma: float = 0.25
    min_base_height: float = 0.22
    max_tilt_deg: float = 70.0
    undesired_contact_threshold: float = 1.0
    feet_air_time_threshold: float = 0.4
    feet_air_time_command_threshold: float = 0.01
    feet_height_ankle_height: float = 0.04
    feet_height_threshold: float = 0.02
    feet_height_command_threshold: float = 0.01
    feet_distance_min: float = 0.16
    feet_distance_max: float = 0.50
    knee_distance_min: float = 0.18
    knee_distance_max: float = 0.35
    stand_still_command_threshold: float = 0.01
    stand_still_body_vel_threshold: float = 0.5
    stand_still_pos_weight: float = 1.0
    stand_still_vel_weight: float = 0.04


def _default_reward_config() -> RPOFlatRewardConfig:
    return RPOFlatRewardConfig(
        scales={
            "track_lin_vel_xy_exp": 1.0,
            "track_ang_vel_z_exp": 1.0,
            "lin_vel_z_l2": -0.2,
            "ang_vel_xy_l2": -0.1,
            "flat_orientation_l2": -1.0,
            "action_rate_l2": -0.02,
            "action_smoothness_l2": -0.02,
            "joint_torques_l2": -1.0e-5,
            "joint_vel_l2": -2.0e-4,
            "dof_acc_l2": -2.5e-7,
            "energy": -1.0e-4,
            "undesired_contacts": -1.0,
            "dof_pos_limits": -1.0,
            "termination_penalty": -200.0,
            "feet_air_time": 0.25,
            "feet_contact_without_cmd": 0.1,
            "feet_height": 0.2,
            "feet_orientation_l2": -0.1,
            "feet_distance": 0.1,
            "knee_distance": 0.1,
            "stand_still": -0.2,
            "upward": 0.4,
            "joint_deviation_hip": -0.03,
            "joint_deviation_legs": -0.01,
            "joint_deviation_torso": -0.5,
            "joint_deviation_arms": -0.06,
        }
    )


@registry.envcfg("RPOFlat")
@dataclass
class RPOFlatCfg(RPOBaseCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "rpo" / "scene_flat.xml")
        )
    )
    max_episode_seconds: float = 20.0
    commands: Commands = field(
        default_factory=lambda: Commands(
            vel_limit=[
                [-0.6, -0.5, -1.57],
                [1.0, 0.5, 1.57],
            ]
        )
    )
    actor_obs_history_length: int = 10
    critic_obs_history_length: int = 10
    contact_force_threshold: float = 1.0
    rel_standing_envs: float = 0.2
    command_resample_interval: float = 10.0
    reward_config: RPOFlatRewardConfig = field(default_factory=_default_reward_config)


@registry.env("RPOFlat", sim_backend="mujoco")
class RPOFlatEnv(RPOBaseEnv):
    _cfg: RPOFlatCfg

    def __init__(self, cfg: RPOFlatCfg, num_envs: int = 1, backend_type: str = "mujoco"):
        backend = create_backend(
            backend_type,
            cfg.scene,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.asset.base_name,
            **env_backend_kwargs(cfg),
        )
        super().__init__(cfg, backend, num_envs)
        self._backend.materialize()

        self._enable_reward_log = True
        self._reward_cfg = cfg.reward_config
        self._actor_hist_len = max(1, int(cfg.actor_obs_history_length))
        self._critic_hist_len = max(1, int(cfg.critic_obs_history_length))
        dtype = get_global_dtype()
        actor_dim = 78
        critic_dim = 133  # 139 - 6 (removed feet_contact_force 3D×2)
        self._actor_hist = np.zeros((num_envs, self._actor_hist_len, actor_dim), dtype=dtype)
        self._critic_hist = np.zeros((num_envs, self._critic_hist_len, critic_dim), dtype=dtype)
        self._last_foot_contact = np.zeros((num_envs, 2), dtype=bool)
        self._current_air_time = np.zeros((num_envs, 2), dtype=dtype)
        self._current_contact_time = np.zeros((num_envs, 2), dtype=dtype)
        self._foot_pos_w = np.zeros((num_envs, 2, 3), dtype=dtype)
        self._feet_pos_b = np.zeros((num_envs, 2, 3), dtype=dtype)
        self._knee_pos_b = np.zeros((num_envs, 2, 3), dtype=dtype)
        self._foot_quat = np.zeros((num_envs, 2, 4), dtype=dtype)
        joint_range = self._backend.get_joint_range()
        self._joint_range = (
            np.asarray(joint_range, dtype=dtype) if joint_range is not None else None
        )
        # Joint indices for grouped deviation rewards (IsaacLab RPO style).
        # Actuator order: 0..5 left leg, 6..11 right leg, 12 torso,
        # 13..17 left arm, 18..22 right arm.
        self._hip_joint_idx = np.array([0, 1, 6, 7], dtype=np.intp)          # thigh_yaw/roll
        self._legs_joint_idx = np.array([2, 3, 4, 5, 8, 9, 10, 11], dtype=np.intp)  # thigh_pitch/knee/ankle
        self._torso_joint_idx = np.array([12, 14, 15, 16, 17, 19, 20, 21, 22], dtype=np.intp)  # torso+arm_roll/yaw+elbow
        self._arms_joint_idx = np.array([13, 18], dtype=np.intp)              # arm_pitch
        self._init_reward_functions()

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": 78 * self._actor_hist_len, "critic": 133 * self._critic_hist_len}

    def reset(self, env_indices: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        env_ids = np.asarray(env_indices, dtype=np.int32)
        num_reset = int(env_ids.shape[0])
        dtype = get_global_dtype()

        qpos = np.tile(self._init_qpos, (num_reset, 1))
        qvel = np.tile(self._init_qvel, (num_reset, 1))
        if num_reset:
            qpos[:, 0:2] += np.asarray(
                np.random.uniform(-0.5, 0.5, (num_reset, 2)), dtype=dtype
            )
        self._backend.set_state(env_ids, qpos, qvel)
        if num_reset:
            self._last_foot_contact[env_ids] = False
            self._current_air_time[env_ids] = 0.0
            self._current_contact_time[env_ids] = 0.0
            self._foot_pos_w[env_ids] = 0.0

        commands = self._sample_commands(num_reset)
        info_updates: dict[str, np.ndarray] = {
            "current_actions": np.zeros((num_reset, self._num_action), dtype=dtype),
            "last_actions": np.zeros((num_reset, self._num_action), dtype=dtype),
            "previous_actions": np.zeros((num_reset, self._num_action), dtype=dtype),
            "commands": commands,
            "torques": np.zeros((num_reset, self._num_action), dtype=dtype),
            "qacc": np.zeros((num_reset, self._num_action), dtype=dtype),
            "current_air_time": np.zeros((num_reset, 2), dtype=dtype),
            "current_contact_time": np.zeros((num_reset, 2), dtype=dtype),
            "terminated_raw": np.zeros((num_reset,), dtype=bool),
            "terminated_contact": np.zeros((num_reset,), dtype=bool),
        }

        if self._state is not None:
            self._state.info["steps"][env_ids] = 0
            if "current_actions" not in self._state.info:
                self._state.info["current_actions"] = np.zeros(
                    (self._num_envs, self._num_action), dtype=dtype
                )
            if "last_actions" not in self._state.info:
                self._state.info["last_actions"] = np.zeros(
                    (self._num_envs, self._num_action), dtype=dtype
                )
            if "previous_actions" not in self._state.info:
                self._state.info["previous_actions"] = np.zeros(
                    (self._num_envs, self._num_action), dtype=dtype
                )
            if "commands" not in self._state.info:
                self._state.info["commands"] = np.zeros((self._num_envs, 3), dtype=dtype)
            if "prev_dof_vel" not in self._state.info:
                self._state.info["prev_dof_vel"] = np.zeros(
                    (self._num_envs, self._num_action), dtype=dtype
                )
            if "torques" not in self._state.info:
                self._state.info["torques"] = np.zeros(
                    (self._num_envs, self._num_action), dtype=dtype
                )
            if "qacc" not in self._state.info:
                self._state.info["qacc"] = np.zeros(
                    (self._num_envs, self._num_action), dtype=dtype
                )
            if "current_air_time" not in self._state.info:
                self._state.info["current_air_time"] = np.zeros((self._num_envs, 2), dtype=dtype)
            if "current_contact_time" not in self._state.info:
                self._state.info["current_contact_time"] = np.zeros(
                    (self._num_envs, 2), dtype=dtype
                )
            if "terminated_raw" not in self._state.info:
                self._state.info["terminated_raw"] = np.zeros((self._num_envs,), dtype=bool)
            if "terminated_contact" not in self._state.info:
                self._state.info["terminated_contact"] = np.zeros((self._num_envs,), dtype=bool)
            self._state.info["current_actions"][env_ids] = info_updates["current_actions"]
            self._state.info["last_actions"][env_ids] = info_updates["last_actions"]
            self._state.info["previous_actions"][env_ids] = info_updates["previous_actions"]
            self._state.info["commands"][env_ids] = info_updates["commands"]
            self._state.info["torques"][env_ids] = info_updates["torques"]
            self._state.info["qacc"][env_ids] = info_updates["qacc"]
            self._state.info["current_air_time"][env_ids] = info_updates["current_air_time"]
            self._state.info["current_contact_time"][env_ids] = info_updates["current_contact_time"]
            self._state.info["terminated_raw"][env_ids] = info_updates["terminated_raw"]
            self._state.info["terminated_contact"][env_ids] = info_updates["terminated_contact"]
            self._state.terminated[env_ids] = False
            self._state.truncated[env_ids] = False

        gyro = self.get_gyro()
        base_quat = self._backend.get_base_quat()
        projected_gravity = self._projected_gravity(base_quat)
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        linvel = self.get_local_linvel()
        foot_pos = self.get_foot_pos()
        actor_current = self._build_actor_obs(
            info_updates,
            gyro=gyro[env_ids],
            projected_gravity=projected_gravity[env_ids],
            dof_pos=dof_pos[env_ids],
            dof_vel=dof_vel[env_ids],
        )
        critic_current = self._build_critic_obs(
            info_updates,
            actor_obs_clean=actor_current["critic_actor_clean"],
            linvel=linvel[env_ids],
            dof_pos=dof_pos[env_ids],
            dof_vel=dof_vel[env_ids],
            foot_pos=foot_pos[env_ids],
            feet_contact=self._last_foot_contact[env_ids],
            feet_air_time=self._current_air_time[env_ids],
            joint_torque=self._get_joint_torque()[env_ids],
            joint_acc=np.zeros_like(dof_vel[env_ids]),
        )
        self._fill_histories(env_ids, actor_current["actor"], critic_current)
        if self._state is not None:
            self._state.info["prev_dof_vel"][env_ids] = dof_vel[env_ids]
        obs = {
            "obs": self._actor_hist[env_ids].reshape(num_reset, -1),
            "critic": self._critic_hist[env_ids].reshape(num_reset, -1),
        }
        return obs, info_updates

    def update_state(self, state: NpEnvState) -> NpEnvState:
        gyro = self.get_gyro()
        base_quat = self._backend.get_base_quat()
        projected_gravity = self._projected_gravity(base_quat)
        linvel_yaw = self._linvel_yaw_frame(base_quat, self.get_local_linvel())
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        linvel = self.get_local_linvel()
        base_pos = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())
        base_height = base_pos[:, 2]
        foot_pos = self.get_foot_pos()
        knee_pos = self.get_knee_pos()
        self._foot_pos_w = np.asarray(foot_pos, dtype=get_global_dtype())
        self._feet_pos_b = self._body_pos_b_from_world(
            np.asarray(foot_pos, dtype=get_global_dtype()),
            base_pos=base_pos,
            base_quat=base_quat,
        )
        self._knee_pos_b = self._body_pos_b_from_world(
            np.asarray(knee_pos, dtype=get_global_dtype()),
            base_pos=base_pos,
            base_quat=base_quat,
        )
        self._foot_quat = np.asarray(self.get_foot_quat(), dtype=get_global_dtype())
        joint_torque = self._get_joint_torque()
        # G1-style contact: aggregate 4 sphere found-sensors per foot
        feet_contact = self._get_aggregated_foot_contact()
        self._current_air_time[~feet_contact] += float(self._cfg.ctrl_dt)
        self._current_air_time[feet_contact] = 0.0
        self._current_contact_time[feet_contact] += float(self._cfg.ctrl_dt)
        self._current_contact_time[~feet_contact] = 0.0
        self._last_foot_contact = feet_contact
        commands = state.info.get("commands")
        if commands is None or not isinstance(commands, np.ndarray) or commands.shape != (self._num_envs, 3):
            commands = np.zeros((self._num_envs, 3), dtype=get_global_dtype())
            state.info["commands"] = commands

        # Mid-episode command resampling (matching IsaacLab 10s interval)
        if self._cfg.command_resample_interval > 0:
            self._resample_commands_on_interval(state)

        actor_current = self._build_actor_obs(
            state.info,
            gyro=gyro,
            projected_gravity=projected_gravity,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
        )
        prev_dof_vel = state.info.get("prev_dof_vel")
        if prev_dof_vel is None or not isinstance(prev_dof_vel, np.ndarray) or prev_dof_vel.shape != dof_vel.shape:
            prev_dof_vel = None
        joint_acc = self._compute_joint_acc(dof_vel, prev_dof_vel)
        critic_current = self._build_critic_obs(
            state.info,
            actor_obs_clean=actor_current["critic_actor_clean"],
            linvel=linvel,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            foot_pos=foot_pos,
            feet_contact=feet_contact,
            feet_air_time=self._current_air_time,
            joint_torque=joint_torque,
            joint_acc=joint_acc,
        )
        self._push_histories(None, actor_current["actor"], critic_current)
        state.info["torques"] = joint_torque.copy()
        state.info["qacc"] = joint_acc.copy()
        state.info["current_air_time"] = self._current_air_time.copy()
        state.info["current_contact_time"] = self._current_contact_time.copy()
        state.info["prev_dof_vel"] = dof_vel.copy()
        terminated_contact = self._compute_contact_terminated()
        state.info["terminated_contact"] = terminated_contact.copy()
        terminated = self._compute_terminated(base_height, projected_gravity, terminated_contact)
        state.info["terminated_raw"] = terminated.copy()
        reward = self._compute_reward(
            state.info,
            linvel=linvel,
            linvel_yaw=linvel_yaw,
            gyro=gyro,
            projected_gravity=projected_gravity,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            base_height=base_height,
        )
        log = state.info.setdefault("log", {})
        log["termination/contact_rate"] = float(np.mean(terminated_contact.astype(get_global_dtype())))
        obs = {
            "obs": self._actor_hist.reshape(self._num_envs, -1),
            "critic": self._critic_hist.reshape(self._num_envs, -1),
        }
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        previous_current = state.info.get("current_actions", np.zeros_like(actions))
        previous_last = state.info.get("last_actions", np.zeros_like(actions))
        state.info["previous_actions"] = previous_last
        state.info["last_actions"] = previous_current
        state.info["current_actions"] = actions
        exec_actions = (
            state.info["last_actions"]
            if self._cfg.control_config.simulate_action_latency
            else actions
        )
        ctrl: np.ndarray = (
            exec_actions * self._cfg.control_config.action_scale + self.default_angles
        )
        return ctrl

    def _sample_commands(self, num_samples: int) -> np.ndarray:
        low = np.asarray(self._cfg.commands.vel_limit[0], dtype=get_global_dtype())
        high = np.asarray(self._cfg.commands.vel_limit[1], dtype=get_global_dtype())
        cmds = np.asarray(
            np.random.uniform(low=low, high=high, size=(num_samples, 3)),
            dtype=get_global_dtype(),
        )
        # rel_standing_envs: first N envs get zero command (standing)
        num_standing = max(1, int(num_samples * self._cfg.rel_standing_envs)) if num_samples > 1 else 0
        if num_standing > 0:
            cmds[:num_standing] = 0.0
            np.random.shuffle(cmds)  # shuffle so standing envs are not contiguous
        return cmds

    def _resample_commands_on_interval(self, state: NpEnvState) -> None:
        """Resample commands every ``command_resample_interval`` seconds."""
        steps = state.info.get("steps")
        if steps is None:
            return
        resample_every = max(1, int(self._cfg.command_resample_interval / self._cfg.ctrl_dt))
        need_resample = np.asarray((steps % resample_every) == 0)
        if not np.any(need_resample):
            return
        idx = np.where(need_resample)[0]
        state.info["commands"][idx] = self._sample_commands(len(idx))

    def _projected_gravity(self, base_quat: np.ndarray) -> np.ndarray:
        gravity_w = np.zeros((base_quat.shape[0], 3), dtype=get_global_dtype())
        gravity_w[:, 2] = -1.0
        return np_quat_apply_inverse(base_quat, gravity_w)

    def _linvel_yaw_frame(self, base_quat: np.ndarray, linvel: np.ndarray) -> np.ndarray:
        linvel_world = np_quat_apply(base_quat, linvel)
        yaw_quat = np_yaw_quat(base_quat)
        return np_quat_apply_inverse(yaw_quat, linvel_world)

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

    @staticmethod
    def _scalarize_sensor_values(sensor_values: np.ndarray) -> np.ndarray:
        sensor_array = np.asarray(sensor_values, dtype=get_global_dtype())
        if sensor_array.ndim == 1:
            return sensor_array
        if sensor_array.ndim == 2 and sensor_array.shape[1] == 1:
            return sensor_array[:, 0]
        raise ValueError(f"Expected scalar sensor values, got shape {sensor_array.shape}")

    def _get_aggregated_foot_contact(self) -> np.ndarray:
        """Aggregate 4 sphere found-sensors per foot into binary contact (G1-style)."""
        left = [self._scalarize_sensor_values(self._backend.get_sensor_data(name))
                for name in self._cfg.sensor.foot_contact_sensors_left]
        right = [self._scalarize_sensor_values(self._backend.get_sensor_data(name))
                 for name in self._cfg.sensor.foot_contact_sensors_right]
        left_contact = np.any(np.stack(left, axis=1) > 0.5, axis=1)
        right_contact = np.any(np.stack(right, axis=1) > 0.5, axis=1)
        return np.stack([left_contact, right_contact], axis=1)

    def _get_joint_torque(self) -> np.ndarray:
        return np.asarray(
            self._backend.get_sensor_data_batch(self._cfg.sensor.actuator_frc),
            dtype=get_global_dtype(),
        )

    def _build_actor_obs(
        self,
        info: dict,
        *,
        gyro: np.ndarray,
        projected_gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> dict[str, np.ndarray]:
        diff = dof_pos - self.default_angles
        commands = info.get("commands")
        if commands is None or not isinstance(commands, np.ndarray) or commands.shape != (gyro.shape[0], 3):
            commands = np.zeros((gyro.shape[0], 3), dtype=get_global_dtype())
        last_actions = info.get("current_actions", np.zeros_like(diff))

        noise_cfg = self._cfg.noise_config
        noisy_gyro = self._obs_noise(gyro, noise_cfg.scale_gyro)
        noisy_gravity = self._obs_noise(projected_gravity, noise_cfg.scale_gravity)
        noisy_diff = self._obs_noise(diff, noise_cfg.scale_joint_angle)
        noisy_dof_vel = self._obs_noise(dof_vel, noise_cfg.scale_joint_vel)
        actor = np.concatenate(
            [noisy_gyro, noisy_gravity, commands, noisy_diff, noisy_dof_vel, last_actions],
            axis=1,
            dtype=get_global_dtype(),
        )
        critic_actor_clean = np.concatenate(
            [gyro, projected_gravity, commands, diff, dof_vel, last_actions],
            axis=1,
            dtype=get_global_dtype(),
        )
        return {"actor": actor, "critic_actor_clean": critic_actor_clean}

    def _init_reward_functions(self) -> None:
        self._reward_fns: dict[str, Any] = {
            "track_lin_vel_xy_exp": self._reward_track_lin_vel_xy_exp,
            "track_ang_vel_z_exp": self._reward_track_ang_vel_z_exp,
            "lin_vel_z_l2": rewards.lin_vel_z,
            "ang_vel_xy_l2": rewards.ang_vel_xy,
            "flat_orientation_l2": rewards.orientation,
            "action_rate_l2": rewards.action_rate,
            "action_smoothness_l2": rewards.action_smooth,
            "joint_torques_l2": rewards.dof_torques_l2,
            "joint_vel_l2": self._reward_joint_vel_l2,
            "dof_acc_l2": rewards.dof_acc_l2,
            "energy": rewards.energy,
            "undesired_contacts": self._reward_undesired_contacts,
            "dof_pos_limits": rewards.joint_pos_limits,
            "termination_penalty": self._reward_termination_penalty,
            "feet_air_time": self._reward_feet_air_time,
            "feet_contact_without_cmd": self._reward_feet_contact_without_cmd,
            "feet_height": self._reward_feet_height,
            "feet_orientation_l2": self._reward_feet_orientation_l2,
            "feet_distance": self._reward_feet_distance,
            "knee_distance": self._reward_knee_distance,
            "stand_still": self._reward_stand_still,
            "upward": self._reward_upward,
            "joint_deviation_hip": self._reward_joint_deviation_hip,
            "joint_deviation_legs": self._reward_joint_deviation_legs,
            "joint_deviation_torso": self._reward_joint_deviation_torso,
            "joint_deviation_arms": self._reward_joint_deviation_arms,
        }

    def _compute_reward(
        self,
        info: dict,
        *,
        linvel: np.ndarray,
        linvel_yaw: np.ndarray,
        gyro: np.ndarray,
        projected_gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        base_height: np.ndarray,
    ) -> np.ndarray:
        cfg = self._reward_cfg
        ctx = RewardContext(
            info=info,
            linvel=linvel,
            gyro=gyro,
            dof_pos=dof_pos,
            num_envs=self._num_envs,
            default_angles=self.default_angles,
            tracking_sigma=float(cfg.tracking_sigma),
            base_height_target=0.0,
            base_height=base_height,
            gravity=projected_gravity,
            dof_vel=dof_vel,
            joint_range=self._joint_range,
            linvel_yaw=linvel_yaw,
        )
        return rewards.run_reward_dispatch(
            scales=cfg.scales,
            fns=self._reward_fns,
            ctx=ctx,
            info=info,
            enable_log=self._enable_reward_log,
            ctrl_dt=self._cfg.ctrl_dt,
        )

    def _compute_terminated(
        self,
        base_height: np.ndarray,
        projected_gravity: np.ndarray,
        contact_terminated: np.ndarray,
    ) -> np.ndarray:
        upright_cos = np.clip(-projected_gravity[:, 2], -1.0, 1.0)
        tilt = np.arccos(upright_cos)
        max_tilt_rad = np.deg2rad(self._reward_cfg.max_tilt_deg)
        return np.asarray(
            contact_terminated
            | (base_height < self._reward_cfg.min_base_height)
            | (tilt > max_tilt_rad),
            dtype=bool,
        )

    def _compute_contact_terminated(self) -> np.ndarray:
        count = self._contact_count_from_sensors(
            self._cfg.sensor.termination_contact_force,
            threshold=float(self._cfg.contact_force_threshold),
        )
        return np.asarray(count > 0, dtype=bool)

    def _upright_gate(self, projected_gravity: np.ndarray | None) -> np.ndarray:
        if projected_gravity is None:
            return np.ones((self._num_envs,), dtype=get_global_dtype())
        return np.asarray(
            np.clip(-projected_gravity[:, 2], 0.0, 0.7) / 0.7,
            dtype=get_global_dtype(),
        )

    def _reward_track_lin_vel_xy_exp(self, ctx: RewardContext) -> np.ndarray:
        track = rewards.track_lin_vel_xy_yaw_frame_exp(ctx)
        return np.asarray(track * self._upright_gate(ctx.gravity), dtype=get_global_dtype())

    def _reward_track_ang_vel_z_exp(self, ctx: RewardContext) -> np.ndarray:
        track = rewards.track_ang_vel_z_world_exp(ctx)
        return np.asarray(track * self._upright_gate(ctx.gravity), dtype=get_global_dtype())

    def _reward_joint_vel_l2(self, ctx: RewardContext) -> np.ndarray:
        assert ctx.dof_vel is not None
        return np.asarray(np.sum(np.square(ctx.dof_vel), axis=1), dtype=get_global_dtype())

    def _reward_undesired_contacts(self, ctx: RewardContext) -> np.ndarray:
        return self._contact_count_from_sensors(
            self._cfg.sensor.undesired_contact_force,
            threshold=float(self._reward_cfg.undesired_contact_threshold),
        )

    def _reward_termination_penalty(self, ctx: RewardContext) -> np.ndarray:
        terminated = np.asarray(
            ctx.info.get("terminated_raw", np.zeros((ctx.num_envs,), dtype=bool)),
            dtype=get_global_dtype(),
        )
        return terminated

    def _reward_feet_air_time(self, ctx: RewardContext) -> np.ndarray:
        air = np.asarray(
            ctx.info.get("current_air_time", np.zeros((ctx.num_envs, 2))), dtype=get_global_dtype()
        )
        contact = np.asarray(
            ctx.info.get("current_contact_time", np.zeros((ctx.num_envs, 2))), dtype=get_global_dtype()
        )
        in_contact = contact > 0.0
        in_mode_time = np.where(in_contact, contact, air)
        single_stance = np.sum(in_contact.astype(np.int32), axis=1) == 1
        masked = np.where(single_stance[:, None], in_mode_time, 0.0)
        reward = np.min(masked, axis=1)
        reward = np.clip(reward, 0.0, float(self._reward_cfg.feet_air_time_threshold))
        cmd = np.asarray(ctx.info.get("commands", np.zeros((ctx.num_envs, 3))), dtype=get_global_dtype())
        moving = (np.linalg.norm(cmd[:, :2], axis=1) + np.abs(cmd[:, 2])) > float(
            self._reward_cfg.feet_air_time_command_threshold
        )
        return np.asarray(reward * moving * self._upright_gate(ctx.gravity), dtype=get_global_dtype())

    def _reward_feet_height(self, ctx: RewardContext) -> np.ndarray:
        contacts = np.asarray(self._last_foot_contact, dtype=bool)
        single_stance = np.sum(contacts.astype(np.int32), axis=1) == 1
        ankle_height = float(self._reward_cfg.feet_height_ankle_height)
        threshold = float(self._reward_cfg.feet_height_threshold)
        foot_height = np.clip(self._foot_pos_w[:, :, 2] - ankle_height, 0.0, 1.0)
        rew_pos = foot_height > threshold
        reward = np.where((~contacts) & single_stance[:, None], rew_pos.astype(get_global_dtype()), 0.0).sum(axis=1)
        cmd = np.asarray(ctx.info.get("commands", np.zeros((ctx.num_envs, 3))), dtype=get_global_dtype())
        moving = (np.linalg.norm(cmd[:, :2], axis=1) + np.abs(cmd[:, 2])) > float(
            self._reward_cfg.feet_height_command_threshold
        )
        return np.asarray(reward * moving * self._upright_gate(ctx.gravity), dtype=get_global_dtype())

    def _reward_feet_contact_without_cmd(self, ctx: RewardContext) -> np.ndarray:
        cmd = np.asarray(ctx.info.get("commands", np.zeros((ctx.num_envs, 3))), dtype=get_global_dtype())
        cmd_norm = np.linalg.norm(cmd[:, :2], axis=1) + np.abs(cmd[:, 2])
        still = cmd_norm < float(self._reward_cfg.stand_still_command_threshold)
        both_contact = np.sum(self._last_foot_contact.astype(np.int32), axis=1) == 2
        return np.asarray(
            still * both_contact * self._upright_gate(ctx.gravity),
            dtype=get_global_dtype(),
        )

    def _reward_feet_orientation_l2(self, ctx: RewardContext) -> np.ndarray:
        gravity_w = np.zeros((ctx.num_envs, 3), dtype=get_global_dtype())
        gravity_w[:, 2] = -1.0
        flat = []
        for i in range(2):
            g_foot = np_quat_apply_inverse(self._foot_quat[:, i, :], gravity_w)
            flat.append(np.sum(np.square(g_foot[:, :2]), axis=1))
        return np.asarray(flat[0] + flat[1], dtype=get_global_dtype())

    def _body_distance_y_exp(self, pos_b: np.ndarray, *, min_dist: float, max_dist: float) -> np.ndarray:
        distance = np.abs(pos_b[:, 0, 1] - pos_b[:, 1, 1])
        d_min = np.clip(distance - float(min_dist), -0.5, 0.0)
        d_max = np.clip(distance - float(max_dist), 0.0, 0.5)
        return np.asarray(
            (np.exp(-np.abs(d_min) * 100.0) + np.exp(-np.abs(d_max) * 100.0)) / 2.0,
            dtype=get_global_dtype(),
        )

    def _reward_feet_distance(self, ctx: RewardContext) -> np.ndarray:
        return self._body_distance_y_exp(
            self._feet_pos_b,
            min_dist=float(self._reward_cfg.feet_distance_min),
            max_dist=float(self._reward_cfg.feet_distance_max),
        )

    def _reward_knee_distance(self, ctx: RewardContext) -> np.ndarray:
        return self._body_distance_y_exp(
            self._knee_pos_b,
            min_dist=float(self._reward_cfg.knee_distance_min),
            max_dist=float(self._reward_cfg.knee_distance_max),
        )

    def _reward_stand_still(self, ctx: RewardContext) -> np.ndarray:
        cmd = np.asarray(ctx.info.get("commands", np.zeros((ctx.num_envs, 3))), dtype=get_global_dtype())
        cmd_norm = np.linalg.norm(cmd[:, :2], axis=1) + np.abs(cmd[:, 2])
        body_lin_vel = np.linalg.norm(ctx.linvel[:, :2], axis=1)
        body_ang_vel = np.abs(ctx.gyro[:, 2])
        body_vel = body_lin_vel + body_ang_vel
        pos_reward = float(self._reward_cfg.stand_still_pos_weight) * np.sum(
            np.abs(ctx.dof_pos - ctx.default_angles), axis=1
        )
        assert ctx.dof_vel is not None
        vel_reward = float(self._reward_cfg.stand_still_vel_weight) * np.sum(np.abs(ctx.dof_vel), axis=1)
        reward = np.where(
            (cmd_norm > float(self._reward_cfg.stand_still_command_threshold))
            | (body_vel > float(self._reward_cfg.stand_still_body_vel_threshold)),
            0.0,
            pos_reward + vel_reward,
        )
        return np.asarray(reward * self._upright_gate(ctx.gravity), dtype=get_global_dtype())

    def _reward_upward(self, ctx: RewardContext) -> np.ndarray:
        assert ctx.gravity is not None
        return np.asarray(-ctx.gravity[:, 2], dtype=get_global_dtype())

    # ── joint deviation (IsaacLab RPO style, L1) ───────────────────

    def _reward_joint_deviation_l1(self, ctx: RewardContext, indices: np.ndarray) -> np.ndarray:
        """L1 penalty for deviation from default, summed over the given joint indices."""
        diff = ctx.dof_pos[:, indices] - ctx.default_angles[indices]
        return np.asarray(np.sum(np.abs(diff), axis=1), dtype=get_global_dtype())

    def _reward_joint_deviation_hip(self, ctx: RewardContext) -> np.ndarray:
        return self._reward_joint_deviation_l1(ctx, self._hip_joint_idx)

    def _reward_joint_deviation_legs(self, ctx: RewardContext) -> np.ndarray:
        return self._reward_joint_deviation_l1(ctx, self._legs_joint_idx)

    def _reward_joint_deviation_torso(self, ctx: RewardContext) -> np.ndarray:
        return self._reward_joint_deviation_l1(ctx, self._torso_joint_idx)

    def _reward_joint_deviation_arms(self, ctx: RewardContext) -> np.ndarray:
        return self._reward_joint_deviation_l1(ctx, self._arms_joint_idx)

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

    def _build_critic_obs(
        self,
        info: dict,
        *,
        actor_obs_clean: np.ndarray,
        linvel: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        foot_pos: np.ndarray,
        feet_contact: np.ndarray,
        feet_air_time: np.ndarray,
        joint_torque: np.ndarray,
        joint_acc: np.ndarray,
    ) -> np.ndarray:
        num_envs = actor_obs_clean.shape[0]
        feet_height = np.clip(foot_pos[:, :, 2] - 0.04, 0.0, 1.0).astype(get_global_dtype())

        return np.concatenate(
            [
                actor_obs_clean,
                linvel,
                np.asarray(feet_contact, dtype=get_global_dtype()),
                np.asarray(feet_air_time, dtype=get_global_dtype()),
                feet_height,
                joint_acc,
                joint_torque,
            ],
            axis=1,
            dtype=get_global_dtype(),
        )

    def _compute_joint_acc(
        self, dof_vel: np.ndarray, prev_dof_vel: np.ndarray | None
    ) -> np.ndarray:
        if prev_dof_vel is None:
            return np.zeros_like(dof_vel)
        return np.asarray((dof_vel - prev_dof_vel) / float(self._cfg.ctrl_dt), dtype=get_global_dtype())

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
        env_ids: np.ndarray | None,
        actor_obs: np.ndarray,
        critic_obs: np.ndarray,
    ) -> None:
        sel = slice(None) if env_ids is None else env_ids
        self._actor_hist[sel, :] = actor_obs[:, None, :]
        self._critic_hist[sel, :] = critic_obs[:, None, :]
