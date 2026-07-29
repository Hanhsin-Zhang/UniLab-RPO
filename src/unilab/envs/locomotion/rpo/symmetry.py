"""MuJoCo-only RPO symmetry augmentation owned by the task/backend layer."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from unilab.base.augmentation import SymmetryAugmentation, SymmetryObsLayout


@dataclass(frozen=True)
class _ObsGroupTransform:
    dim: int
    joint_map: torch.Tensor
    sign_mask: torch.Tensor


class RPOSymmetryAugmentation(SymmetryAugmentation):
    """Runtime symmetry adapter aligned with the RoboParty RPO baseline."""

    batch_multiplier = 2

    _SYMMETRY_PAIRS = {
        "left_thigh_yaw_joint": "right_thigh_yaw_joint",
        "left_thigh_roll_joint": "right_thigh_roll_joint",
        "left_thigh_pitch_joint": "right_thigh_pitch_joint",
        "left_knee_joint": "right_knee_joint",
        "left_ankle_pitch_joint": "right_ankle_pitch_joint",
        "left_ankle_roll_joint": "right_ankle_roll_joint",
        "left_arm_pitch_joint": "right_arm_pitch_joint",
        "left_arm_roll_joint": "right_arm_roll_joint",
        "left_arm_yaw_joint": "right_arm_yaw_joint",
        "left_elbow_pitch_joint": "right_elbow_pitch_joint",
        "left_elbow_yaw_joint": "right_elbow_yaw_joint",
    }
    _NEGATIVE_ACTION_NAMES = (
        "thigh_yaw_joint",
        "thigh_roll_joint",
        "ankle_roll_joint",
        "arm_roll_joint",
        "arm_yaw_joint",
        "elbow_yaw_joint",
        "torso_joint",
    )

    def __init__(
        self,
        model,
        obs_layouts: dict[str, SymmetryObsLayout],
        *,
        device: str = "cuda",
    ):
        import mujoco

        actuator_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)
        ]
        name_to_idx = {name: i for i, name in enumerate(actuator_names)}

        joint_map: dict[int, int] = {}
        for left, right in self._SYMMETRY_PAIRS.items():
            if left in name_to_idx and right in name_to_idx:
                joint_map[name_to_idx[left]] = name_to_idx[right]
                joint_map[name_to_idx[right]] = name_to_idx[left]
        for i in range(len(actuator_names)):
            joint_map.setdefault(i, i)

        self._joint_map = torch.tensor(
            [joint_map[i] for i in range(len(actuator_names))],
            device=device,
            dtype=torch.long,
        )

        sign_mask = [1.0] * len(actuator_names)
        for i, name in enumerate(actuator_names):
            if any(pattern in name for pattern in self._NEGATIVE_ACTION_NAMES):
                sign_mask[i] = -1.0
        self._sign_mask = torch.tensor(sign_mask, device=device)
        self._obs_transforms = {
            group_name: self._build_obs_group_transform(layout, device=device)
            for group_name, layout in obs_layouts.items()
        }

    def _build_obs_group_transform(
        self,
        layout: SymmetryObsLayout,
        *,
        device: str,
    ) -> _ObsGroupTransform:
        obs_dim = sum(dim for _, dim in layout)
        flip_mask = torch.ones(obs_dim, device=device)
        joint_map = torch.arange(obs_dim, device=device, dtype=torch.long)
        joint_sign = torch.ones(obs_dim, device=device)
        idx = 0

        for key, dim in layout:
            if dim <= 0:
                raise ValueError(
                    f"Observation layout group {key!r} must have positive dim, got {dim}"
                )

            if key == "linvel":
                self._require_dim(key, dim, 3)
                flip_mask[idx + 1] = -1.0
            elif key == "gyro":
                self._require_dim(key, dim, 3)
                flip_mask[idx] = -1.0
                flip_mask[idx + 2] = -1.0
            elif key == "gravity":
                self._require_dim(key, dim, 3)
                flip_mask[idx + 1] = -1.0
            elif key in {"dof_pos", "dof_vel", "actions"}:
                self._require_dim(key, dim, int(self._joint_map.numel()))
                joint_map[idx : idx + dim] = self._joint_map + idx
                joint_sign[idx : idx + dim] = self._sign_mask
            elif key == "command":
                self._require_dim(key, dim, 3)
                flip_mask[idx + 1] = -1.0
                flip_mask[idx + 2] = -1.0
            elif key == "gait_phase":
                self._require_dim(key, dim, 2)
                joint_map[idx] = idx + 1
                joint_map[idx + 1] = idx

            idx += dim

        return _ObsGroupTransform(
            dim=obs_dim,
            joint_map=joint_map,
            sign_mask=flip_mask * joint_sign,
        )

    @staticmethod
    def _require_dim(group_name: str, actual: int, expected: int) -> None:
        if actual != expected:
            raise ValueError(
                f"Symmetry group {group_name!r} must have dim {expected}, got {actual}"
            )

    def mirror_action(self, action: torch.Tensor) -> torch.Tensor:
        return action[..., self._joint_map] * self._sign_mask

    def mirror_obs(self, obs: torch.Tensor, *, obs_group: str = "obs") -> torch.Tensor:
        transform = self._obs_transforms[obs_group]
        if obs.shape[-1] != transform.dim:
            raise ValueError(
                f"Symmetry obs group {obs_group!r} expects dim {transform.dim}, got {obs.shape[-1]}"
            )
        return obs[..., transform.joint_map] * transform.sign_mask

    def augment_obs(self, obs: torch.Tensor, *, obs_group: str = "obs") -> torch.Tensor:
        return torch.cat([obs, self.mirror_obs(obs, obs_group=obs_group)], dim=0)

    def augment_obs_and_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        *,
        obs_group: str = "obs",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.augment_obs(obs, obs_group=obs_group), torch.cat(
            [actions, self.mirror_action(actions)],
            dim=0,
        )
