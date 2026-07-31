# Debug Session: sac-critic-inputs

Status: [OPEN]

## Goal
- Verify whether the new SAC critic features are actually wired into training for `RPOWalkFlat`.

## Hypotheses
- H1: The checkpoint critic input dimension is still the old shape, so the new features never entered the critic network.
- H2: The env reports the expanded critic observation dimension, but the replay/trainer path still slices or uses the old critic tensor.
- H3: The new critic feature slots exist, but runtime values are all zeros or near-constant due to missing `info` propagation or reset/update issues.
- H4: Some added signals are connected but mirrored/ordered incorrectly, making the critic input semantically noisy rather than useful.
- H5: Data is wired correctly, and the weak training result is instead caused by feature scale/statistics mismatch rather than integration failure.

## Plan
- Inspect run artifacts and checkpoint parameter shapes.
- Verify config snapshot and obs dimensions recorded by the run.
- Collect runtime evidence from a minimal env rollout and summarize per-feature statistics.
- Decide which hypotheses are confirmed or rejected.

## Evidence
- Checkpoint `model_6000.pt` loads successfully with keys `actor`, `qnet`, `qnet_target`, etc.
- Actor first layer shape is `(512, 80)`.
- Critic first layer shape is `(768, 160)`, matching `137 critic_obs + 23 action`.
- Runtime env probe reports `obs_groups_spec = {'obs': 80, 'critic': 137}`.
- Replay / learner path uses separate critic tensors:
  - collector: `obs_np, critic_np = split_obs_dict(state.obs)`
  - learner: `critic_obs = batch["critic"]`, `critic_next_obs = batch["next_critic"]`
- Runtime probe over reset + 5 steps shows:
  - `feet_height` is non-zero at reset and changes over steps.
  - `joint_torque` is non-zero at reset and changes over steps.
  - `joint_acc` is zero at reset and becomes non-zero after stepping.
  - `feet_contact`, `feet_air_time`, `feet_contact_time` become non-zero / change after several steps.

## Hypothesis Status
- H1: Rejected.
- H2: Rejected.
- H3: Rejected for `feet_height`, `joint_torque`, `joint_acc`; rejected in stepped rollout for contact-related features.
- H4: Not proven from current evidence.
- H5: Still plausible.
