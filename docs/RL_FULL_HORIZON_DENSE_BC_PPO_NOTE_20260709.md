# Full-Horizon Dense-BC PPO Note - 2026-07-09

## Context

The sparse no-obstacle79 BC policy was not trained densely enough for full
trajectory following. The dense-BC rebuild increased the no-obstacle79 dataset
from 2,803 rows to 14,066 rows by resampling accepted GIK teachers at 0.1 s.
Dense BC improved rendered behavior and completed case `0028`, but still failed
early on four of the five proof cases.

## Code Support Added

- `scripts/imitation/build_gik_obs_dataset.py --resample-dt`: builds dense
  full-horizon BC datasets aligned with the exported Isaac trajectory stage.
- `scripts/reinforcement_learning/sb3/record_rendered_recovery_rollout.py --episode_length_s`: lets proof clips exceed the old 20 s horizon.
- `scripts/reinforcement_learning/sb3/train.py --episode_length_s`: lets PPO train with the same full-horizon episode length.

## Dense BC Baseline

Dataset:

```text
data/gik_offline_teacher_obs/obs_dataset_no_obstacle79_dense01_full_masked_20260709.npz
rows: 14,066
resample_dt: 0.1 s
```

Policy:

```text
logs/bc/gik_no_obstacle79_dense01_masked9_80ep_20260709/bc_policy.zip
best val MSE: 0.000030
```

Rendered five-case coverage:

```text
case  executed_s  raw_s   coverage
0001  12.25       25.12   48.8%
0020  13.05       50.84   25.7%
0028  37.30       37.19   100.3%
0050   5.80       38.02   15.3%
0079  10.90       17.44   62.5%
```

## PPO Smoke 1: Random-Start Recovery Mix

Training:

```text
logs/sb3/fullhorizon_dense01_ppo_smoke_20260709
warm start: dense BC policy
timesteps: 131,072
num_envs: 128
episode_length_s: 60
random_start_waypoint: 0.0-0.9
reset_anchor_target_blend: 0.75
base assist: enabled, moderate
```

Rendered five-case coverage:

```text
case  dense_bc  ppo_smoke  delta
0001  48.8%     35.0%      -13.8%
0020  25.7%     28.8%       +3.1%
0028 100.3%     42.6%      -57.7%
0050  15.3%     40.1%      +24.8%
0079  62.5%     72.6%      +10.1%
```

Conclusion: do not scale this recipe directly. It learns some recovery-like
behavior but damages start-to-end tracking, especially `0028`.

## PPO Smoke 2: Start-Only Full-Trajectory Mix

Training:

```text
logs/sb3/fullhorizon_dense01_ppo_startonly_smoke_20260709
warm start: dense BC policy
timesteps: 131,072
num_envs: 128
episode_length_s: 60
random_start_waypoint: disabled
reset_anchor_target_blend: 0.0
base assist: weak
```

Observed training metrics:

```text
base-target unreachable zone: ~51.6% at 81,920 steps
workspace hard exceedance: ~22.7%
base actions saturated early in rollout debug
```

Conclusion: do not render or scale this recipe. It destabilizes base motion and
pushes the target outside the reachable envelope too often.

## Next Recommended Change

## PPO Smoke 3: Base-Slew Start-Only Full-Trajectory Mix

Training:

```text
logs/sb3/fullhorizon_dense01_ppo_base_slew_startonly_smoke_20260709
warm start: dense BC policy
timesteps: 65,536
num_envs: 128
episode_length_s: 60
random_start_waypoint: disabled
reset_anchor_target_blend: 0.0
base_action_delta_limit: 0.06 normalized/action step in SB3 wrapper
base_action_slew_limit: 0.06 normalized/action step in env
base_action_delta_penalty: 3.0
```

Training reached the planned timestep budget and saved both `final_model.zip`
and `vec_normalize.pkl`. The clamp worked mechanically: early rollout debug
showed normalized base actions ramping by about `0.06` per step instead of
instant saturation.

Rendered five-case coverage:

```text
case  dense_bc  ppo_smoke_random  ppo_base_slew  delta_vs_bc
0001  48.8%     35.0%             31.6%          -17.2%
0020  25.7%     28.8%             11.8%          -13.9%
0028 100.3%     42.6%             19.2%          -81.1%
0050  15.3%     40.1%             20.4%           +5.1%
0079  62.5%     72.6%             54.5%           -8.0%
```

Conclusion: keep the base clamp and penalty code as useful control primitives,
but reject this training recipe. The action slew limit prevents abrupt command
jumps, but it does not solve the policy collapse; it makes the full-start gate
worse on four of five cases and severely regresses the previously solved `0028`.

## Next Recommended Change

The next useful experiment should avoid plain PPO fine-tuning that is allowed
to drift away from the dense BC teacher:

1. Add a start-to-end gate callback or post-iteration gate so regressions on
   known good cases like `0028` stop the run early.
2. Train with an auxiliary imitation loss or a staged BC-refresh step so PPO
   cannot destroy the dense-BC trajectory-following behavior.
3. Train a two-head or router curriculum:
   - Stage A: preserve dense-BC full-start tracking with very low LR and minimal
     exploration.
   - Stage B: introduce random mid-trajectory starts only after Stage A passes
     the five-case full-start gate.
4. Only after no-obstacle full trajectories pass should obstacle cases be mixed
   back in.
