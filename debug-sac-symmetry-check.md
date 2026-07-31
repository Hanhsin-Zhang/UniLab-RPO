# [OPEN] SAC symmetry check

## Symptom

- Config sets `algo.use_symmetry: true` in `conf/offpolicy/task/sac/rpo_flat/mujoco.yaml`.
- Suspected behavior: left-right mirroring augmentation is not taking effect during SAC training.

## Scope

- Task config compose path for off-policy SAC flat walking.
- Symmetry-related config plumbing, env metadata, learner hookup, and runtime evidence.

## Initial Hypotheses

1. The off-policy SAC training pipeline never reads or forwards `algo.use_symmetry`.
2. The symmetry flag is forwarded, but the env/task does not expose the symmetry metadata required by the learner.
3. The learner constructs symmetry augmentation only for PPO/on-policy codepaths, leaving SAC as a no-op.
4. The symmetry path exists, but a config compose mismatch or override resets the effective value to `false` or `None`.
5. The symmetry code executes, but the observation/action dimensions or mirror maps are invalid, causing silent fallback.

## Evidence Log

- Static trace confirms `conf/offpolicy/task/sac/rpo_flat/mujoco.yaml` sets `algo.use_symmetry: true`.
- `scripts/train_offpolicy.py` reads `cfg.algo.use_symmetry`, builds env-owned symmetry augmentation, and passes it into learner kwargs.
- `src/unilab/envs/locomotion/rpo/walk_flat.py` returns `RPOSymmetryAugmentation` only for MuJoCo backend.
- `src/unilab/algos/torch/fast_sac/learner.py` applies augmentation in both critic and actor update paths.
- `uv run pytest tests/envs/locomotion/rpo/test_symmetry_contract.py tests/algos/test_fast_sac_symmetry_contract.py tests/algos/test_offpolicy_runner_unit.py -k 'symmetry'` passed with `21 passed`.
- Runtime verification via Hydra compose on `task=sac/rpo_flat/mujoco` showed:
  - `cfg.training.task_name == RPOWalkFlat`
  - `cfg.training.sim_backend == mujoco`
  - `cfg.algo.use_symmetry == True`
  - env class `RPOWalkEnv`
  - augmentation class `RPOSymmetryAugmentation`
  - `batch_multiplier == 2`
  - `augment_obs` shape `(4, 240) -> (8, 240)`
  - `augment_obs(critic)` shape `(4, 411) -> (8, 411)`
  - learner `update_critic/update_actor` saw expanded tensors with leading batch dimension `8`, confirming runtime symmetry expansion is active.

## Hypothesis Status

1. The off-policy SAC training pipeline never reads or forwards `algo.use_symmetry`.
   - Rejected by static trace through `scripts/train_offpolicy.py`.
2. The symmetry flag is forwarded, but the env/task does not expose the symmetry metadata required by the learner.
   - Rejected for `rpo_flat/mujoco`; `walk_flat.py` provides `RPOSymmetryAugmentation`.
3. The learner constructs symmetry augmentation only for PPO/on-policy codepaths, leaving SAC as a no-op.
   - Rejected by `FastSACLearner.update_critic()` and `update_actor()` augmentation calls plus symmetry tests.
4. A config compose mismatch or override resets the effective value to `false` or `None`.
   - Not observed in the owner config path under inspection; still worth checking in the exact user launch command if extra CLI overrides are used.
5. The symmetry code executes, but the observation/action dimensions or mirror maps are invalid, causing silent fallback.
   - Rejected by current RPO symmetry contract tests covering layout dimensions and concrete mirrored obs/action outputs.

## Current Verdict

- For the owner config path `conf/offpolicy/task/sac/rpo_flat/mujoco.yaml`, symmetry support is wired and covered by passing runtime tests.
- Runtime evidence also confirms the effective path is active for the exact composed config: env builds augmentation, mirror transform changes action ordering/signs, and learner consumes doubled batches.
- The more likely failure mode, if your training effect still looks wrong, is not "flag ignored" but one of:
  - launch command accidentally composing a different owner config or override;
  - training metrics/logs do not expose whether augmented samples are used;
  - reward/command distribution makes symmetry benefit hard to see qualitatively.

## User Verification Checklist

1. Re-run the focused symmetry tests listed above and confirm they pass locally.
2. Verify your real training launch command resolves to the same owner config path and backend.
3. Check training startup logs for symmetry-enabled status and effective batch semantics.
4. If suspicion remains, add minimal runtime instrumentation around learner batch shapes and augmentation calls.

## Next Steps

1. Trace config -> runner -> learner wiring for `use_symmetry`.
2. Locate symmetry contract provider in env/task/backend stack.
3. Add minimal instrumentation if static inspection cannot prove the effective runtime path.
4. Produce a verification checklist and evidence summary for user validation.
