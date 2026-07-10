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
from unilab.envs.locomotion.rpo.base import RPOBaseCfg, RPOBaseEnv


@registry.envcfg("RPOFlat")
@dataclass
class RPOFlatCfg(RPOBaseCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "rpo" / "scene_flat.xml")
        )
    )
    max_episode_seconds: float = 20.0
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

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": 75}

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

        info_updates: dict[str, np.ndarray] = {
            "current_actions": np.zeros((num_reset, self._num_action), dtype=dtype),
            "last_actions": np.zeros((num_reset, self._num_action), dtype=dtype),
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
            self._state.info["current_actions"][env_ids] = info_updates["current_actions"]
            self._state.info["last_actions"][env_ids] = info_updates["last_actions"]
            self._state.terminated[env_ids] = False
            self._state.truncated[env_ids] = False

        obs = self._compute_obs(
            info_updates,
            gyro=self.get_gyro()[env_ids],
            linvel=self.get_local_linvel()[env_ids],
            dof_pos=self.get_dof_pos()[env_ids],
            dof_vel=self.get_dof_vel()[env_ids],
        )
        return obs, info_updates

    def update_state(self, state: NpEnvState) -> NpEnvState:
        obs = self._compute_obs(
            state.info,
            gyro=self.get_gyro(),
            linvel=self.get_local_linvel(),
            dof_pos=self.get_dof_pos(),
            dof_vel=self.get_dof_vel(),
        )
        reward = np.zeros((self._num_envs,), dtype=get_global_dtype())
        terminated = np.zeros((self._num_envs,), dtype=bool)
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _compute_obs(
        self,
        info: dict,
        *,
        gyro: np.ndarray,
        linvel: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> dict[str, np.ndarray]:
        diff = dof_pos - self.default_angles
        last_actions = info.get("current_actions", np.zeros_like(diff))
        obs = np.concatenate([gyro, linvel, diff, dof_vel, last_actions], axis=1, dtype=get_global_dtype())
        return {"obs": obs}
