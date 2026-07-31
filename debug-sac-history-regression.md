# Debug Session: sac-history-regression

Status: [OPEN]

## Goal
- Find why re-adding SAC history logic in `walk_flat.py` breaks training, even when both history lengths are set to `1`.

## Hypotheses
- H1: The history patch changed env reset/step lifecycle semantics, so the off-policy worker now sees different initial state than before.
- H2: The custom reset observation path bypasses side effects from the pre-history implementation, causing inconsistent `info` or internal state.
- H3: Even with history length `1`, the new history buffers alter observation assembly or ownership in a way that changes training behavior.
- H4: A non-history change bundled into the same `walk_flat.py` diff is the true regression, and the failure is only correlated with the history patch.

## Plan
- Inspect git history and diff for `walk_flat.py`.
- Compare the pre-history working version against the history-added version.
- Trace the off-policy worker lifecycle against the env lifecycle.
- Identify the smallest semantic change that explains the regression.

## Evidence
- `git log --oneline -- src/unilab/envs/locomotion/rpo/walk_flat.py` shows `02f8cc7c more critic obs and obs scale` as the last known good pre-history commit for this file.
- `git diff -- src/unilab/envs/locomotion/rpo/walk_flat.py` shows the history patch changes `build_reset_observation()` and `update_state()` from returning fresh `_compute_obs()` outputs to returning reshaped history buffers.
- Pre-history `git show 02f8cc7c:...walk_flat.py` shows:
  - `build_reset_observation()` delegates to `super().build_reset_observation(...)`
  - `update_state()` returns `obs = self._compute_obs(...)`
- History patch returns:
  - reset: `env._actor_hist[env_ids].reshape(...)`, `env._critic_hist[env_ids].reshape(...)`
  - step: `self._actor_hist.reshape(...)`, `self._critic_hist.reshape(...)`
- Runtime probe with `actor_obs_history_length=1` / `critic_obs_history_length=1` confirms:
  - `np.shares_memory(state0.obs["obs"], env._actor_hist) == True`
  - `np.shares_memory(state0.obs["critic"], env._critic_hist) == True`
  - after the next `env.step()`, the previous `state0.obs["obs"]` / `state0.obs["critic"]` values mutate in place
  - the old views become equal to the new state's observations
- Off-policy collector keeps previous-step numpy obs references across loop iterations before calling `replay_buffer.add(...)`, so mutating env-owned obs storage breaks the `(obs, action, reward, next_obs)` transition contract.

## Root Cause
- The regression is caused by returning env-owned mutable history-buffer views from `walk_flat.py`.
- This changes observation ownership semantics compared with the pre-history version, which returned fresh arrays from `_compute_obs()`.
- Because the off-policy worker retains previous obs arrays across `env.step()` calls, later history-buffer writes mutate earlier obs in place.
- This corrupts replay transitions even when history lengths are both set to `1`.

## Hypothesis Status
- H1: Confirmed.
- H2: Possible but secondary; not needed to explain the regression.
- H3: Confirmed.
- H4: Rejected as the primary explanation; the ownership change inside the history patch is sufficient.
