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
from unilab.envs.locomotion.common.commands import Commands
from unilab.envs.locomotion.rpo.base import RPOBaseCfg, RPOBaseEnv
from unilab.utils.rotation import np_quat_apply_inverse


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
    reward_config: dict[str, Any] = field(default_factory=dict)


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

        self._actor_hist_len = max(1, int(cfg.actor_obs_history_length))
        self._critic_hist_len = max(1, int(cfg.critic_obs_history_length))
        dtype = get_global_dtype()
        self._actor_hist = np.zeros((num_envs, self._actor_hist_len, 78), dtype=dtype)
        self._critic_hist = np.zeros((num_envs, self._critic_hist_len, 139), dtype=dtype)
        self._feet_force = np.zeros((num_envs, 2, 3), dtype=dtype)
        self._last_foot_contact = np.zeros((num_envs, 2), dtype=bool)
        self._feet_air_time = np.zeros((num_envs, 2), dtype=dtype)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": 78 * self._actor_hist_len, "critic": 139 * self._critic_hist_len}

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
            self._feet_force[env_ids] = 0.0
            self._last_foot_contact[env_ids] = False
            self._feet_air_time[env_ids] = 0.0

        commands = self._sample_commands(num_reset)
        info_updates: dict[str, np.ndarray] = {
            "current_actions": np.zeros((num_reset, self._num_action), dtype=dtype),
            "last_actions": np.zeros((num_reset, self._num_action), dtype=dtype),
            "commands": commands,
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
            if "commands" not in self._state.info:
                self._state.info["commands"] = np.zeros((self._num_envs, 3), dtype=dtype)
            if "prev_dof_vel" not in self._state.info:
                self._state.info["prev_dof_vel"] = np.zeros(
                    (self._num_envs, self._num_action), dtype=dtype
                )
            self._state.info["current_actions"][env_ids] = info_updates["current_actions"]
            self._state.info["last_actions"][env_ids] = info_updates["last_actions"]
            self._state.info["commands"][env_ids] = info_updates["commands"]
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
            feet_contact_force=self._feet_force[env_ids],
            feet_air_time=self._feet_air_time[env_ids],
            joint_torque=self._get_joint_torque()[env_ids],
            prev_dof_vel=None,
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
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        linvel = self.get_local_linvel()
        foot_pos = self.get_foot_pos()
        self._feet_force = self._get_feet_contact_force()
        joint_torque = self._get_joint_torque()
        feet_contact = np.linalg.norm(self._feet_force, axis=2) > float(self._cfg.contact_force_threshold)
        self._feet_air_time[~feet_contact] += float(self._cfg.ctrl_dt)
        self._feet_air_time[feet_contact] = 0.0
        self._last_foot_contact = feet_contact
        commands = state.info.get("commands")
        if commands is None or not isinstance(commands, np.ndarray) or commands.shape != (self._num_envs, 3):
            commands = np.zeros((self._num_envs, 3), dtype=get_global_dtype())
            state.info["commands"] = commands

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
        critic_current = self._build_critic_obs(
            state.info,
            actor_obs_clean=actor_current["critic_actor_clean"],
            linvel=linvel,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            foot_pos=foot_pos,
            feet_contact=feet_contact,
            feet_contact_force=self._feet_force,
            feet_air_time=self._feet_air_time,
            joint_torque=joint_torque,
            prev_dof_vel=prev_dof_vel,
        )
        self._push_histories(None, actor_current["actor"], critic_current)
        state.info["prev_dof_vel"] = dof_vel.copy()
        obs = {
            "obs": self._actor_hist.reshape(self._num_envs, -1),
            "critic": self._critic_hist.reshape(self._num_envs, -1),
        }
        reward = np.zeros((self._num_envs,), dtype=get_global_dtype())
        terminated = np.zeros((self._num_envs,), dtype=bool)
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _sample_commands(self, num_samples: int) -> np.ndarray:
        low = np.asarray(self._cfg.commands.vel_limit[0], dtype=get_global_dtype())
        high = np.asarray(self._cfg.commands.vel_limit[1], dtype=get_global_dtype())
        return np.asarray(
            np.random.uniform(low=low, high=high, size=(num_samples, 3)), dtype=get_global_dtype()
        )

    def _projected_gravity(self, base_quat: np.ndarray) -> np.ndarray:
        gravity_w = np.zeros((base_quat.shape[0], 3), dtype=get_global_dtype())
        gravity_w[:, 2] = -1.0
        return np_quat_apply_inverse(base_quat, gravity_w)

    def _get_feet_contact_force(self) -> np.ndarray:
        forces = [self._backend.get_sensor_data(name) for name in self._cfg.sensor.foot_contact_force]
        return np.stack(forces, axis=1)

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
        feet_contact_force: np.ndarray,
        feet_air_time: np.ndarray,
        joint_torque: np.ndarray,
        prev_dof_vel: np.ndarray | None,
    ) -> np.ndarray:
        num_envs = actor_obs_clean.shape[0]
        feet_height = np.clip(foot_pos[:, :, 2] - 0.04, 0.0, 1.0).astype(get_global_dtype())

        if prev_dof_vel is None:
            joint_acc = np.zeros_like(dof_vel)
        else:
            joint_acc = (dof_vel - prev_dof_vel) / float(self._cfg.ctrl_dt)

        return np.concatenate(
            [
                actor_obs_clean,
                linvel,
                np.asarray(feet_contact, dtype=get_global_dtype()),
                np.asarray(feet_contact_force, dtype=get_global_dtype()).reshape(num_envs, -1),
                np.asarray(feet_air_time, dtype=get_global_dtype()),
                feet_height,
                joint_acc,
                joint_torque,
            ],
            axis=1,
            dtype=get_global_dtype(),
        )

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
