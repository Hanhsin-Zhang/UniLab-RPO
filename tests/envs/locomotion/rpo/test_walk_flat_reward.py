from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.rpo.base import Sensor
from unilab.envs.locomotion.rpo.walk_flat import (
    RPOWalkDomainRandConfig,
    RPOWalkEnv,
    RPOWalkRewardConfig,
)


def _reward_config(*, probe_reduction: str = "min") -> RPOWalkRewardConfig:
    return RPOWalkRewardConfig(
        scales={"feet_height": 0.5},
        tracking_sigma=0.25,
        gait_frequency=1.5,
        feet_phase_swing_height=0.09,
        feet_phase_tracking_sigma=0.04,
        base_height_target=0.78,
        min_base_height=0.4,
        max_tilt_deg=65.0,
        feet_height_probe_reduction=probe_reduction,
    )


def _make_backend(sensor_data: dict[str, np.ndarray]) -> SimpleNamespace:
    return SimpleNamespace(get_sensor_data=lambda name: sensor_data[name])


def _make_probe_env(
    sensor_data: dict[str, np.ndarray], *, probe_reduction: str = "min", num_envs: int = 1
) -> RPOWalkEnv:
    env = object.__new__(RPOWalkEnv)
    env._num_envs = num_envs
    env._backend = _make_backend(sensor_data)
    env._cfg = SimpleNamespace(sensor=Sensor())
    env._reward_cfg = _reward_config(probe_reduction=probe_reduction)
    return env


def test_rpo_walk_flat_probe_height_supports_min_and_mean_reduction():
    sensor_data = {
        "left_foot_probe_0_pos": np.array([[0.0, 0.0, 0.04]], dtype=np.float32),
        "left_foot_probe_1_pos": np.array([[0.0, 0.0, 0.06]], dtype=np.float32),
        "left_foot_probe_2_pos": np.array([[0.0, 0.0, 0.08]], dtype=np.float32),
        "left_foot_probe_3_pos": np.array([[0.0, 0.0, 0.10]], dtype=np.float32),
        "right_foot_probe_0_pos": np.array([[0.0, 0.0, 0.01]], dtype=np.float32),
        "right_foot_probe_1_pos": np.array([[0.0, 0.0, 0.03]], dtype=np.float32),
        "right_foot_probe_2_pos": np.array([[0.0, 0.0, 0.05]], dtype=np.float32),
        "right_foot_probe_3_pos": np.array([[0.0, 0.0, 0.07]], dtype=np.float32),
    }

    env_min = _make_probe_env(sensor_data, probe_reduction="min")
    np.testing.assert_allclose(env_min._get_foot_height_from_probes(), [[0.04, 0.01]])

    env_mean = _make_probe_env(sensor_data, probe_reduction="mean")
    np.testing.assert_allclose(env_mean._get_foot_height_from_probes(), [[0.07, 0.04]])


def test_rpo_walk_flat_feet_height_reward_matches_ppo_style_gate():
    sensor_data: dict[str, np.ndarray] = {}
    left_contact = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    right_contact = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    for i in range(5):
        sensor_data[f"left_foot_contact_{i}"] = left_contact if i == 0 else np.zeros_like(left_contact)
        sensor_data[f"right_foot_contact_{i}"] = (
            right_contact if i == 0 else np.zeros_like(right_contact)
        )

    left_probe_heights = [0.01, 0.03, 0.02]
    right_probe_heights = [0.015, 0.01, 0.03]
    for i in range(4):
        sensor_data[f"left_foot_probe_{i}_pos"] = np.array(
            [[0.0, 0.0, left_probe_heights[0]], [0.0, 0.0, left_probe_heights[1]], [0.0, 0.0, left_probe_heights[2]]],
            dtype=np.float32,
        )
        sensor_data[f"right_foot_probe_{i}_pos"] = np.array(
            [
                [0.0, 0.0, right_probe_heights[0]],
                [0.0, 0.0, right_probe_heights[1]],
                [0.0, 0.0, right_probe_heights[2]],
            ],
            dtype=np.float32,
        )

    env = _make_probe_env(sensor_data, num_envs=3)
    ctx = RewardContext(
        info={
            "commands": np.array([[0.4, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        },
        linvel=np.zeros((3, 3), dtype=np.float32),
        gyro=np.zeros((3, 3), dtype=np.float32),
        dof_pos=np.zeros((3, 1), dtype=np.float32),
        num_envs=3,
        default_angles=np.zeros((1,), dtype=np.float32),
        gravity=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
    )

    reward = env._reward_feet_height(ctx)
    np.testing.assert_allclose(reward, np.array([0.75, 1.0, 0.0], dtype=np.float32))


def test_sac_rpo_flat_owner_enables_feet_height_and_standing_commands():
    root = Path(__file__).resolve().parents[4]
    cfg = OmegaConf.load(root / "conf" / "offpolicy" / "task" / "sac" / "rpo_flat" / "mujoco.yaml")

    assert OmegaConf.select(cfg, "reward.scales.feet_height") == pytest.approx(0.5)
    assert OmegaConf.select(cfg, "env.commands.rel_standing_envs") == pytest.approx(0.2)
    assert OmegaConf.select(cfg, "env.domain_rand.randomize_base_mass") is True
    assert OmegaConf.select(cfg, "env.domain_rand.push_robots") is True


def test_rpo_walk_flat_curriculum_scales_domain_rand_and_push():
    env = object.__new__(RPOWalkEnv)
    env._cfg = SimpleNamespace(
        domain_rand=RPOWalkDomainRandConfig(
            added_mass_range=[-3.0, 3.0],
            body_mass_multiplier_range=[0.9, 1.1],
            com_offset_x=[-0.025, 0.025],
            com_offset_y=[-0.025, 0.025],
            com_offset_z=[-0.05, 0.05],
            ground_friction_multiplier_range=[0.3, 1.6],
            dof_armature_multiplier_range=[0.5, 1.5],
            kp_multiplier_range=[0.9, 1.1],
            kd_multiplier_range=[0.9, 1.1],
            max_force=[1.0, 1.0, 0.5],
            reset_joint_qpos_range=[-0.05, 0.05],
            reset_base_qvel_range=[
                [-0.5, -0.5, -0.2, -0.52, -0.52, -0.78],
                [0.5, 0.5, 0.2, 0.52, 0.52, 0.78],
            ],
        )
    )
    env._domain_rand_curriculum_base = deepcopy(env._cfg.domain_rand)

    env._apply_domain_rand_curriculum_scale(0.5)

    dr = env._cfg.domain_rand
    assert dr.added_mass_range == pytest.approx([-1.5, 1.5])
    assert dr.body_mass_multiplier_range == pytest.approx([0.95, 1.05])
    assert dr.com_offset_y == pytest.approx([-0.0125, 0.0125])
    assert dr.ground_friction_multiplier_range == pytest.approx([0.65, 1.3])
    assert dr.dof_armature_multiplier_range == pytest.approx([0.75, 1.25])
    assert dr.kp_multiplier_range == pytest.approx([0.95, 1.05])
    assert dr.max_force == pytest.approx([0.5, 0.5, 0.25])
    assert dr.reset_joint_qpos_range == pytest.approx([-0.025, 0.025])
    np.testing.assert_allclose(
        dr.reset_base_qvel_range,
        [
            [-0.25, -0.25, -0.1, -0.26, -0.26, -0.39],
            [0.25, 0.25, 0.1, 0.26, 0.26, 0.39],
        ],
    )
