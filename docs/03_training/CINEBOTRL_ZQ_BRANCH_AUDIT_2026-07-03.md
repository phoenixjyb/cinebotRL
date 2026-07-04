# CineBotRL `zq*` Branch RL Audit

Date: 2026-07-03

Scope:
- Active repo on `.98`: `/mnt/g/wSpace/cinebotRL`, branch `win-recomoPro1`.
- Audited remote branches:
  - `zq/train_linux` at `56bd1fa`
  - `zq/1105test` at `62f56ad`
  - `zq/acc/1112` at `bde3e05`
  - `zq/filter/1112` at `53622f7`
  - `zq/hist/2123` at `30053dd`
- Audit method: shallow sparse clones under `/tmp/cinebot_zq_audit`, source/doc grep, and key-file comparison against the active branch.

## Executive Conclusion

Do not start another long PPO run blindly.

PPO is not disproven, but the current "PPO plus callback-style auxiliary training" path has saturated. The `zq*` branches do not show a clearly different `pi`/non-IsaacLab RL framework in the visible repo content. They mainly show an older IsaacLab + Stable-Baselines3 PPO stack, older 8D/PPR robot assumptions, and a long trail of reward/curriculum experiments.

The useful lesson from `zq*` is not "switch framework now". The useful lesson is that the problem needs a more structured formulation test:
- Can a single/few-trajectory policy track accurately on current Proto2 if trained with a supervised/recovery objective first?
- Does integrated PPO minibatch auxiliary loss beat the current callback-only aux approach?
- Does a reachability/workspace curriculum with explicit workspace state and progress-indexed teacher labels reduce tail failures?

## Verified Branch Facts

### Framework

The visible `zq*` branch source uses IsaacLab and Stable-Baselines3 PPO:
- `scripts/reinforcement_learning/sb3/train.py` imports `PPO`, `VecNormalize`, and IsaacLab `AppLauncher`.
- `src/rl_platform/tasks/mobile_mm/env.py` imports IsaacLab environment/assets/simulation APIs.
- `pyproject.toml` in `zq/train_linux` lists `torch`, `gymnasium`, and `stable-baselines3`.

Searches did not find a clear active alternative stack such as `rsl_rl`, `rl_games`, `skrl`, `torchrl`, `lerobot`, diffusion policy, or a `pi0/pi3/pi` training framework in the checked branch content. If such a framework exists, it is likely in another repository, unpublished local files, or non-sparse large artifacts, not in the visible `zq*` source tree.

### Robot/Action Contract

The `zq*` stack is older than the active Proto2 contract:
- `env.py` in the zq snapshots has `num_actions: int = 8`.
- It comments the contract as `6 arm joints + 2 base DOF (v_x, omega_z)`.
- It does not include the current Proto2 `base_vy` action.
- Docs and scripts reference older PPR-style assets such as `mobile_manipulator_PPR_base_corrected.urdf`, `mobile_manipulator_PPR_theta_before_x.urdf`, and `mobile_manipulator_PPR_theta_x_y.urdf`.

This means any good result from those branches is not directly transferable to the current `recomoProto2-1190` 9D contract.

### Reported Performance

The branch docs do not support the claim that the old general-policy result had small tracking error across the full task:
- Session 7c comparison records mean tracking error around `1.01 m`, still above the `0.30 m` target.
- Session 8 comparisons report position errors around `307.8 cm`, `311.0 cm`, `349.4 cm`, and `408.0 cm`.
- Session 8H docs report a best position error around `237.3 cm`, while orientation remained a fundamental issue.

There may have been a narrow demo, single trajectory, or visual qualitative case with small error, but the committed documentation for the broader RL runs shows meter-level errors.

## Transferable Lessons

The `zq*` work is still useful as failure analysis.

1. Workspace/reachability matters more than obstacle collision for the current plateau.
   - This matches our recent Proto2 recovery rounds: obstacle unsafe/collision stayed zero, while reachability/tail tracking remained bad.

2. Explicit workspace observations are worth reusing.
   - zq docs added heading cue and workspace comfort signals.
   - Current Proto2 should keep direct state about base-to-target geometry, reachable workspace distance, and whether the EE request is inside the current arm envelope.

3. Do not use brittle bell-shaped reachability as the main learning signal.
   - zq notes repeatedly reject the bell-shaped approach and prefer two-zone/linear reachability shaping.
   - For current Proto2, this should be tested as a curriculum/objective change, not as another blind long run.

4. Action smoothing/filtering is deployment-useful but not a reachability cure.
   - `zq/filter/1112` did not show material differences in the main key files compared with `zq/acc/1112` in the sparse audit.
   - Smoothness/EMA can reduce jerk, but it will not solve unreachable targets or wrong base positioning by itself.

5. Narrow-task success must be measured separately from generalization.
   - If the historical claim was "one/few trajectories tracked well", replicate that explicitly on current Proto2 before claiming the full policy is improving.

## Recommendation

Keep PPO available, but stop treating vanilla PPO as the next automatic step.

Recommended next experiment:

1. Build a current-Proto2 "single/few trajectory feasibility gate".
   - Use the exact current USD/URDF/action contract.
   - Include `base_vy`.
   - Use only trajectories with duration >= 5 s.
   - Report EE mean/p95, workspace hard percentage, obstacle unsafe/collision, base path, and action smoothness.

2. Add integrated minibatch auxiliary imitation loss inside the PPO update path.
   - The callback-style aux mining has saturated.
   - The next test should couple RL updates and teacher/recovery labels in the actual optimizer step, not only as side-channel callback training.

3. Use a staged objective:
   - Stage A: supervised/BC or recovery-assisted tracking on a tiny current-Proto2 set.
   - Stage B: short PPO fine-tune with the same tiny set.
   - Stage C: expand to hard categories only after Stage A/B prove small-error tracking on current geometry.

4. Treat zq as a baseline/failure reference.
   - Compare against its lessons on workspace, heading cue, reachability margins, and control timing.
   - Do not port old 8D PPR assumptions into current Proto2.

## Decision Gate Before Long Training

Do not launch a long run unless a short gate proves:
- Current action dim is 9.
- `base_vy` is active and not penalized as invalid lateral motion.
- Single/few-trajectory gate is meaningfully below current recovery tail error.
- Workspace hard percentage is reduced without obstacle unsafe/collision.
- PPO health metrics do not trigger the existing emergency pause behavior.

If the tiny current-Proto2 gate cannot achieve low error, the issue is formulation/control target quality, not training duration.

## Implemented Follow-Up

Implemented the first safe transferable bit from the audit in:

`scripts/reinforcement_learning/sb3/evaluate_recovery_candidate.py`

The evaluator now supports current-Proto2 narrow feasibility gates before any new training:
- `--trajectory_manifest` to evaluate an explicit manifest.
- `--trajectory_category` to restrict to one or more categories such as `crane_down`.
- `--trajectory_file_contains` to restrict by path/name substring.
- `--max_trajectories` to force one/few-trajectory gates.
- `--min_trajectory_duration` to keep the existing rule that trajectories shorter than 5 s are rejected.
- `--random_start_waypoint/--no-random_start_waypoint` and `--reset_base_to_trajectory_start/--no-reset_base_to_trajectory_start` to test deterministic replay versus randomized recovery starts.
- JSON output now records `trajectory_source` and `episode_group_summary`, grouped by category and file.

Validation performed without launching training:
- `py_compile` passed with `/mnt/g/isaaclab_venv/Scripts/python.exe -X utf8`.
- `--help` shows the new gate arguments.
- A helper smoke resolved `--trajectory_category crane_down --max_trajectories 2` to two current stage1 recovery trajectories and wrote `resolved_feasibility_manifest.txt`.
- Added `tests/test_recovery_candidate_eval.py`, which runs without Isaac Sim or pytest and covers manifest filtering, metric summarization, and per-file/per-category aggregation.
