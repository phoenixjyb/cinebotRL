# Proto2 Narrow Trajectory Gate Log

Date: 2026-07-04

Purpose: test the `zq`-inspired claim at the smallest useful scale: can the current Proto2 9D policy track one/few current trajectories before we launch any new training?

## Setup

- Host: `.98`, repo `/mnt/g/wSpace/cinebotRL`
- Branch: `win-recomoPro1`
- Checkpoint:
  `logs/sb3/recomoproto2trackee_v0/stage1_recovery_auxloss_round5_hardcats_obs_noassist_40k_seed20260703_20260703_1829/final_model.zip`
- VecNormalize:
  `logs/sb3/recomoproto2trackee_v0/stage1_recovery_auxloss_round5_hardcats_obs_noassist_40k_seed20260703_20260703_1829/vec_normalize.pkl`
- Gate set: `stage1_recovery`, category `crane_down`, first 2 trajectories.
- Selected files:
  - `trajectoryToLearn/world_json/cinematic_db/crane_down/crane_down_000.json`
  - `trajectoryToLearn/world_json/cinematic_db/crane_down/crane_down_019.json`
- Episodes/envs: 8 episodes, 8 envs
- Obstacles: enabled
- Duration filter: `min_trajectory_duration=5.0`

## Results

Raw policy output:

- JSON: `evaluation_results/proto2_narrow_gate_round5_cranedown2_20260704_151119/recovery_eval_raw-policy_20260704_151244.json`
- EE position mean: `1.4616 m`
- EE position p95 mean: `1.5407 m`
- EE orientation mean: `130.32 deg`
- Unreachable zone: `28.40%`
- Workspace hard exceed: `2.52%`
- Obstacle unsafe/collision: `0.0% / 0.0%`

Assisted base output:

- JSON: `evaluation_results/proto2_narrow_gate_round5_cranedown2_assisted_20260704_151256/recovery_eval_assisted_20260704_151357.json`
- EE position mean: `1.3515 m`
- EE position p95 mean: `1.4112 m`
- EE orientation mean: `142.79 deg`
- Unreachable zone: `6.87%`
- Workspace hard exceed: `0.49%`
- Obstacle unsafe/collision: `0.0% / 0.0%`

## Interpretation

The narrow gate did not reproduce a small-error result on current Proto2. Base assist improves reachability substantially, but EE tracking remains around `1.35 m` and orientation remains very poor. This means the immediate blocker is not just base reachability or obstacle avoidance.

Conclusion: do not start another broad PPO run from this state. The next useful update should target target/action formulation quality:

- Inspect the two selected trajectories against current Proto2 arm envelope and camera frame.
- Compare desired EE pose against actual reachable FK pose under the current arm/gimbal envelope.
- Add a deterministic non-RL replay/teacher sanity check for these two files before adding more PPO pressure.
- Only after that, test integrated imitation or BC-first training on the same two-trajectory gate.

## Runtime Notes

- Isaac emitted a non-fatal `hid` missing warning from `isaaclab_tasks` device imports but continued to run.
- Isaac emitted unresolved visual reference warnings for `base_footprint`; evaluation continued.
- Action contract was confirmed at runtime as current 9D `sim_6joint_gimbal_v1` with `base_vy`.
