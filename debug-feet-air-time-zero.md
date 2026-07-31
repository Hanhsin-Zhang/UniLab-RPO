[OPEN] feet-air-time-zero

# Debug Session

- Session ID: `feet-air-time-zero`
- Symptom: SAC `feet_air_time` reward stays at 0 for the latest completed run.
- Checkpoint: `/data/hanxin/code/UniLab-RPO/logs/fast_sac/RPOWalkFlat/2026-07-29_17-23-31_mujoco/model_5000.pt`

# Hypotheses

1. `ctx.info["feet_air_time"]` is never populated in the SAC env path, so the reward always reads the zero fallback.
2. `feet_air_time` is populated, but its values almost never enter `(0.05, 0.5)`, so the in-range count stays 0.
3. The recently added `moving` gate masks out otherwise valid `feet_air_time` reward because command magnitudes stay below `feet_motion_command_threshold`.
4. Reward logging or dispatch configuration excludes `feet_air_time`, so training runs it but logs it as 0 or never logs it.
5. The latest checkpoint/run used code or config different from the current workspace, so the observed zero comes from a stale reward path.

# Plan

1. Inspect the completed run artifacts and reward logs for direct evidence.
2. Trace the SAC env path that writes `feet_air_time` into `info`.
3. If artifacts are inconclusive, add minimal instrumentation only to the reward-data path and reproduce with the latest checkpoint.

# Evidence

- `run_config.json` confirms the completed run enabled `reward.scales.feet_air_time = 0.3` and used `feet_motion_command_threshold = 0.05`.
- TensorBoard scalar `reward/feet_air_time` has 5000 samples with `min = 0.0`, `max = 0.0`, `nonzero_count = 0`.
- TensorBoard scalar `reward/feet_height` is non-zero in the same run, so the shared moving gate is not globally zeroing all foot-motion rewards.
- In `src/unilab/envs/locomotion/rpo/walk_flat.py`, `_reward_feet_air_time()` reads `ctx.info.get("feet_air_time", zeros(...))`.
- A codebase search finds no producer for `state.info["feet_air_time"]` in the SAC `RPOWalkFlat` path.
- PPO `src/unilab/envs/locomotion/rpo/flat.py` maintains `_current_air_time` each step and passes it into reward/critic paths, proving the expected producer shape exists there but is missing in SAC.

# Analysis

- Hypothesis 1 confirmed: SAC `RPOWalkFlat` never populates `info["feet_air_time"]`, so the reward always consumes the zero fallback.
- Hypothesis 2 rejected as primary cause: there is no runtime evidence that real air-time values are even reaching the reward, so threshold tightness is downstream of the actual break.
- Hypothesis 3 rejected for this run: `feet_height` is non-zero under the same command-threshold regime, so the moving gate is not the reason `feet_air_time` is identically zero.
- Hypothesis 4 rejected: reward dispatch/logging is active because `reward/feet_air_time` exists in TensorBoard; it is just always zero.
- Hypothesis 5 not needed for root cause: the saved run config matches the latest reward settings relevant to this symptom.

# Next Fix

1. Add SAC-side air-time state (`_current_air_time`) maintenance in `walk_flat.py`.
2. Write the maintained values into `state.info["feet_air_time"]` before reward computation.
3. Re-run a short checkpoint-based validation and compare `reward/feet_air_time` pre/post fix.

# Fix Applied

- Added SAC env state `self._current_feet_air_time` in `src/unilab/envs/locomotion/rpo/walk_flat.py`.
- Reset path now zeros the selected env rows and seeds `info_updates["feet_air_time"]`.
- Per-step update now derives fused left/right foot contact from the sole capsule sensors and accumulates air-time as `+ctrl_dt` while airborne, else resets to `0.0`.
- Reward path keeps consuming `info["feet_air_time"]`, but the key is now populated before `_compute_reward(...)`.

# Post-Fix Verification

- Static diagnostics for `walk_flat.py` are clean.
- Lightweight env validation using the real SAC `rpo_flat` Hydra config and random actions:
  - `reset_has_key = True`
  - `reset_max = 0.0`
  - Step 1: `max_air = 0.02`, `positive_entries = 256`
  - Step 10: `max_air = 0.20`, `positive_entries = 87`
  - Step 40: `max_air = 0.52`, `positive_entries = 127`
  - `summary_nonzero_steps = 60`
- This confirms the pre-fix zero fallback is gone and the env now emits non-zero air-time state continuously.
