# Proto2 Grouped Teacher/Student Gate Log - 2026-07-06

## Purpose

First bounded HOVER-style bridge test for CineBotRL:

- grouped actor heads for `arm`, `gimbal`, and `base`
- GIK base-only teacher labels as behavioural-cloning warm start
- capped PPO gate, not open-ended training
- recovery evaluator as the promotion gate

## Teacher Dataset Check

Source:

`data/gik_stage1_recovery_gate20_basehome0_iter300_20260705/obs_dataset_base_only.npz`

Result:

- samples: `2,826`
- source observation dim: `85`
- action dim: `9`
- valid mask mean: `[0,0,0,0,0,0,1,1,1]`
- base-only labels are valid; full masked arm/gimbal labels remain diagnostic only.

The 85D dataset appends progress, while the PPO training environment currently exposes 84D policy observations. The first BC policy trained on 85D achieved low validation error, but could not warm-start PPO:

```text
[WARN] Failed to load BC pretrained policy: size mismatch ... [256, 85] vs [256, 84]
```

## Corrected BC Gate

Derived dataset:

`data/gik_stage1_recovery_gate20_basehome0_iter300_20260705/obs_dataset_base_only_obs84.npz`

Transformation:

- dropped the final progress column
- preserved actions and `action_valid_mask`
- set `observation_dim=84`

Command output:

- output policy: `logs/bc/gik_stage1_gate20_grouped_baseonly_obs84_20260706/bc_policy.zip`
- best validation MSE: `0.000082`
- architecture: grouped shared `[256,256]`, per-head hidden `128`

## Verified PPO Warm Start

Run:

`logs/sb3/recomoproto2trackee_v0/stage1_gate20_grouped_basebc_obs84_64k_20260706`

Gate:

- `64` envs
- `65,536` timesteps
- `stage1_recovery`
- `max_trajectories=20`
- `min_trajectory_duration=5.0`
- copied only base rows `[6,7,8]` from BC

Warm-start evidence:

```text
[OK] BC policy feature weights loaded
[OK] BC policy grouped shared encoder and selected action rows [6, 7, 8]; zeroed non-selected rows loaded
[OK] kept PPO log_std_init; set non-selected log_std=-2.0
```

Final training metrics:

- `approx_kl=0.010740501`
- `explained_variance=0.601`
- `value_loss=0.0677`
- `std=0.135`
- exit code: `0`

## Evaluator Repair

`scripts/reinforcement_learning/sb3/evaluate_recovery_candidate.py` now infers the checkpoint observation dim and drops one trailing observation column only when the evaluator env returns exactly `expected_dim + 1`.

Observed adapter evidence:

```text
[obs-adapter] Dropping final observation column for checkpoint compatibility: 85 -> 84
```

This fixes evaluation of older/current 84D PPO checkpoints against recorded-trajectory eval configs that append progress.

## Evaluation Result

Output:

`evaluation_results/recovery_candidate/stage1_gate20_grouped_basebc_obs84_64k_20260706/recovery_eval_raw-policy_20260706_234006.json`

Raw-policy metrics:

- `ee_pos_error_mean_m.mean=1.7677`
- `ee_pos_error_p95_m.mean=1.9410`
- `ee_ori_error_mean_deg.mean=166.6523`
- `unreachable_zone_pct.mean=91.2037`
- `workspace_hard_exceed_pct.mean=15.6481`
- `obstacle_unsafe_pct.mean=0.0`
- `obstacle_collision_pct.mean=0.0`

Comparison candidate:

`evaluation_results/recovery_candidate/fortykg_planar_smoke_20260706_2148/recovery_eval_raw-policy_20260706_214759.json`

- `ee_pos_error_mean_m.mean=1.1442`
- `ee_pos_error_p95_m.mean=1.3883`
- `ee_ori_error_mean_deg.mean=132.1662`
- `unreachable_zone_pct.mean=47.4211`
- `workspace_hard_exceed_pct.mean=3.7318`
- `obstacle_unsafe_pct.mean=0.0`
- `obstacle_collision_pct.mean=0.0`

## Decision

Do not promote `stage1_gate20_grouped_basebc_obs84_64k_20260706`.

The infrastructure works, but base-only BC warm-start by itself regressed recovery tracking versus the current yaw-assist candidate. The likely issue is that the student receives only base labels while arm/gimbal remain RL-only and undertrained in a short 64k gate. This is useful as a negative gate, not as a better policy.

## Next Recommendation

Keep the grouped architecture and evaluator adapter. Do not run a longer version of this same base-only gate blindly.

Next policy work should target one of:

- train a stronger teacher/student phase with validated arm/gimbal labels or replay-safe filtered labels
- add an auxiliary base imitation loss during PPO instead of only copying BC weights at initialization
- evaluate grouped architecture starting from the current best yaw-assist/recovery checkpoint if compatible, rather than starting from base-only BC

