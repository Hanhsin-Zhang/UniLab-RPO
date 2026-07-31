from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from unilab.base import registry
from unilab.base.registry import ensure_registries
from unilab.envs.locomotion.rpo.walk_flat import RPOWalkRewardConfig

pytest.importorskip("mujoco", reason="mujoco is required for RPO symmetry contract tests")


def _reward_config() -> RPOWalkRewardConfig:
    return RPOWalkRewardConfig(
        scales={"tracking_lin_vel": 2.0, "alive": 10.0},
        tracking_sigma=0.25,
        gait_frequency=1.5,
        feet_phase_swing_height=0.09,
        feet_phase_tracking_sigma=0.04,
        base_height_target=0.78,
        min_base_height=0.4,
        max_tilt_deg=65.0,
    )


def _make_env() -> Any:
    ensure_registries()
    return cast(
        Any,
        registry.make(
            "RPOWalkFlat",
            num_envs=1,
            sim_backend="mujoco",
            env_cfg_override={"reward_config": _reward_config()},
        ),
    )


def test_rpo_walk_flat_symmetry_contract_matches_obs_groups():
    env = _make_env()

    try:
        assert env.obs_groups_spec["obs"] == 78
        assert env.obs_groups_spec["critic"] == 135
        layouts = env.get_symmetry_obs_layouts()
        assert set(layouts) == {"obs", "critic"}
        for group_name, layout in layouts.items():
            assert sum(dim for _, dim in layout) == env.obs_groups_spec[group_name]
    finally:
        env.close()


def test_rpo_walk_flat_symmetry_mirrors_actions_and_obs_like_baseline():
    env = _make_env()

    try:
        augmentation = env.build_symmetry_augmentation(device="cpu")
        assert augmentation is not None

        actions = torch.arange(1, env.action_space.shape[0] + 1, dtype=torch.float32).unsqueeze(0)
        mirrored_actions = augmentation.mirror_action(actions)
        expected_actions = torch.tensor(
            [
                [
                    -7.0,
                    -8.0,
                    9.0,
                    10.0,
                    11.0,
                    -12.0,
                    -1.0,
                    -2.0,
                    3.0,
                    4.0,
                    5.0,
                    -6.0,
                    -13.0,
                    19.0,
                    -20.0,
                    -21.0,
                    22.0,
                    -23.0,
                    14.0,
                    -15.0,
                    -16.0,
                    17.0,
                    -18.0,
                ]
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(mirrored_actions, expected_actions)

        obs = torch.zeros((1, env.obs_groups_spec["obs"]), dtype=torch.float32)
        obs[:, 75:78] = torch.tensor([[1.0, 2.0, 3.0]])
        mirrored_obs = augmentation.mirror_obs(obs, obs_group="obs")
        torch.testing.assert_close(mirrored_obs[:, 75:78], torch.tensor([[1.0, -2.0, -3.0]]))

        critic_obs = torch.zeros((1, env.obs_groups_spec["critic"]), dtype=torch.float32)
        critic_obs[:, 78:81] = torch.tensor([[4.0, 5.0, 6.0]])
        mirrored_critic_obs = augmentation.mirror_obs(critic_obs, obs_group="critic")
        torch.testing.assert_close(
            mirrored_critic_obs[:, 78:81],
            torch.tensor([[4.0, -5.0, 6.0]]),
        )
    finally:
        env.close()
