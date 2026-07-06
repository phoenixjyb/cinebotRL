# GIK/ARCore Base-BC Warm-Start Recovery Gate - 2026-07-04

## Purpose

This gate tested whether the accepted GIK/ARCore base-only behavior-cloning policy can be used as a PPO warm-start for raw `stage1_recovery` training without runtime base assist.

The answer from this bounded run is no. The run is useful as a negative/research result, not as a promotion candidate.

## Inputs

- Branch: `win-recomoPro1`
- Repo: `/mnt/g/wSpace/cinebotRL`
- BC policy: `logs/bc/gik_accepted130_base_smoke_20260704/bc_policy.zip`
- BC dataset: `data/gik_offline_teacher_obs/obs_dataset_offline_base_only.npz`
- Dataset size: 4,557 samples
- Observation/action shape: 85D obs, 9D action
- Teacher source: 130 accepted GIK/ARCore demos
- Teacher mix: 79 no-obstacle demos and 51 one-obstacle demos
- Base action indices transferred into PPO: `6,7,8`
- Arm and gimbal labels were intentionally masked during BC.

## PPO Warm-Start Command

```bash
PYTHONUTF8=1 /mnt/g/isaaclab_venv/Scripts/python.exe -X utf8 \
  scripts/reinforcement_learning/sb3/train.py \
  --task RecomoProto2TrackEE-v0 \
  --num_envs 64 \
  --headless \
  --seed 20260704 \
  --total_timesteps 8192 \
  --n_steps 64 \
  --batch_size 256 \
  --n_epochs 2 \
  --learning_rate 0.0001 \
  --ent_coef 0.001 \
  --vf_coef 0.5 \
  --max_grad_norm 0.5 \
  --trajectory_stage stage1_recovery \
  --start_waypoint_min_fraction 0.25 \
  --start_waypoint_max_fraction 0.70 \
  --reset_anchor_target_blend 0.35 \
  --enable_obstacles \
  --obstacle_radius 0.20 \
  --obstacle_height 0.50 \
  --obstacle_x_range -0.35 0.35 \
  --obstacle_y_range 0.45 1.00 \
  --min_obstacle_start_clearance 0.10 \
  --disable_auto_base_assist \
  --disable_auto_base_assist_yaw \
  --base_assist_imitation_weight 0.0 \
  --base_assist_yaw_imitation_weight 0.0 \
  --pretrained_policy logs/bc/gik_accepted130_base_smoke_20260704/bc_policy.zip \
  --pretrained_action_indices 6,7,8 \
  --pretrained_unselected_log_std -2.0 \
  --log_dir logs/sb3/recomoproto2trackee_v0/gik_accepted130_basewarm_8k_seed20260704_20260704 \
  --save_freq 8192
```

## Training Artifacts

- Final policy: `logs/sb3/recomoproto2trackee_v0/gik_accepted130_basewarm_8k_seed20260704_20260704/final_model.zip`
- VecNormalize: `logs/sb3/recomoproto2trackee_v0/gik_accepted130_basewarm_8k_seed20260704_20260704/vec_normalize.pkl`

## Transfer Behavior Confirmed

- BC policy feature weights were loaded.
- Action head rows `6,7,8` were copied from BC into PPO.
- Non-selected action head rows were zeroed.
- Non-selected `log_std` values were set to `-2.0`.
- Runtime base assist and base-yaw assist were disabled.
- Obstacles were enabled with radius `0.20m` and height `0.50m`.

## Training Snapshot

At 8,192 PPO steps:

- `approx_kl`: `0.0150094265`
- `clip_fraction`: `0.166`
- `explained_variance`: `-0.0161`
- `value_loss`: `4.1`
- Step 0 reachability: `41/64` reachable, `23/64` unreachable
- Step 100 reachability: `24/64` reachable, `40/64` unreachable
- Average base-target distance moved from `0.937m` to `1.119m` during the observed snapshot.

This did not show evidence of recovery improvement during the tiny gate.

## Raw Recovery Evaluation Command

```bash
PYTHONUTF8=1 /mnt/g/isaaclab_venv/Scripts/python.exe -X utf8 \
  scripts/reinforcement_learning/sb3/evaluate_recovery_candidate.py \
  --checkpoint logs/sb3/recomoproto2trackee_v0/gik_accepted130_basewarm_8k_seed20260704_20260704/final_model.zip \
  --vec_normalize logs/sb3/recomoproto2trackee_v0/gik_accepted130_basewarm_8k_seed20260704_20260704/vec_normalize.pkl \
  --task RecomoProto2TrackEE-v0 \
  --num_envs 64 \
  --num_episodes 64 \
  --headless \
  --seed 20260704 \
  --trajectory_stage stage1_recovery \
  --min_trajectory_duration 5.0 \
  --random_start_waypoint \
  --start_waypoint_min_fraction 0.25 \
  --start_waypoint_max_fraction 0.70 \
  --reset_base_to_trajectory_start \
  --reset_anchor_target_blend 0.35 \
  --enable_obstacles \
  --output_dir evaluation_results/gik_accepted130_basewarm_8k_raw_smoke64_20260704
```

## Evaluation Artifact

- JSON: `evaluation_results/gik_accepted130_basewarm_8k_raw_smoke64_20260704/recovery_eval_raw-policy_20260704_234324.json`

## Overall Evaluation Result

- Episodes: `64`
- Steps: `399`
- `base_target_dist_mean`: mean `0.8056`, p95 `0.9588`
- `base_target_dist_max`: mean `1.5743`, p95 `1.7000`
- `optimal_zone_pct`: mean `11.0863`, p95 `25.1562`
- `acceptable_zone_pct`: mean `34.8684`, p95 `50.0000`
- `unreachable_zone_pct`: mean `54.0453`, p95 `70.3125`, max `73.4375`
- `workspace_soft_exceed_pct`: mean `35.9767`, p95 `56.2500`
- `workspace_hard_exceed_pct`: mean `7.6989`, p95 `14.0625`, max `20.3125`
- `workspace_distance_max`: mean `1.0156`, p95 `1.3860`
- `ee_pos_error_mean_m`: mean polluted by an extreme outlier, p95 `1.3365`
- `ee_pos_error_p95_m`: mean `1.7394`, p95 `1.8803`
- `ee_ori_error_mean_deg`: mean `135.2486`, p95 `143.6603`
- Obstacle unsafe/collision rates: `0.0 / 0.0`
- `obstacle_clearance_min`: mean `1.3571`, min `1.2226`

The obstacle metrics were safe in this run, but reachability and tracking were not acceptable.

## Category Breakdown

| Category | n | Unreachable mean pct | Unreachable p95 pct | Workspace hard mean pct | EE pos mean m | EE ori mean deg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `crane_down` | 8 | 71.13 | 98.88 | 15.26 | 1.50 | 133.08 |
| `crane_up` | 12 | 70.95 | 100.00 | 15.61 | outlier-polluted | 132.43 |
| `dolly_pull_out` | 32 | 37.12 | 70.53 | 1.39 | not promoted | 134.90 |
| `handheld_subtle` | 9 | 79.47 | 94.24 | 18.12 | not promoted | 132.39 |
| `scene_3` | 3 | 51.80 | 62.38 | 0.00 | not promoted | 141.83 |

## Decision

Reject this checkpoint as a policy candidate.

Do not continue longer training from this exact warm-start configuration. It is a smoke-test result showing that accepted GIK/ARCore base labels do not directly solve the raw `stage1_recovery` distribution when injected as a base-only PPO prior.

## Interpretation

The most likely issue is distribution mismatch:

- The accepted GIK/ARCore teachers came from their own trajectory and reset distribution.
- The recovery gate evaluates `stage1_recovery` cinematic trajectories with random starts.
- The transferred labels cover only base action rows `6,7,8`; arm and gimbal remain unresolved.
- Full-policy performance is still dominated by reachability, arm/gimbal action formulation, and trajectory feasibility.

This result supports the existing rule: use accepted GIK/ARCore data as learning material, not as proof that the full policy is ready.

## Next Useful Directions

1. Build an in-distribution GIK/ARCore eval stage and test whether the BC base policy works on trajectories matching its teacher source.
2. Generate `stage1_recovery`-compatible teacher labels before using BC warm-start for recovery.
3. Keep deterministic feasibility, action-envelope, and DJI camera-attitude/gimbal contract work ahead of more open-ended PPO.

## Operational Note

The run produced the usual non-fatal local warnings:

- `ModuleNotFoundError: No module named 'hid'` from `isaaclab_tasks`
- unresolved USD visual references for `base_footprint`

These warnings were not treated as the cause of the failed policy gate.

## 2026-07-05 In-Distribution Stage1 Teacher Follow-Up

The July 4 result above used accepted GIK/ARCore base teachers against the raw `stage1_recovery` distribution and was rejected. The follow-up generated teacher labels directly from the `stage1_recovery` request, then aligned export to the MATLAB StageC segment instead of the merged StageA+B+C prefix.

Teacher source on the Mac/GIK side:

```text
/Users/yanbo/Projects/gikWBC9DOF/outputs/rl_teacher_stage1_recovery_smoke6_basehome0_iter300_20260705
```

Imported dataset on `.98`:

```text
data/gik_stage1_recovery_smoke6_basehome0_iter300_20260705/obs_dataset_base_only.npz
```

Import result:

```text
cases: 6
samples: 916
obs_dim: 85
base-only label mask: [0,0,0,0,0,0,1,1,1]
full masked label mask: [1.0,0.1670,0.0,0.9683,0.9716,0.9694,1.0,1.0,1.0]
```

The base-only labels are suitable for a smoke gate. The full masked labels are not suitable for full 9D BC yet because some arm/gimbal channels have weak or zero validity.

Base-only BC smoke artifact:

```text
logs/bc/gik_stage1_smoke6_baseonly_20260705/bc_policy.zip
```

Training summary:

```text
transitions: 916
epochs: 10
best val MSE: 0.001297
```

Offline evaluator artifact:

```text
evaluation_results/bc/gik_stage1_smoke6_baseonly_20260705.json
```

Evaluator summary on action rows `6,7,8`:

```text
aggregate rmse: 0.031358
aggregate mae: 0.020651
action 6 vx rmse: 0.044872, mae: 0.033780, bias: -0.000473
action 7 vy rmse: 0.026196, mae: 0.016818, bias: 0.004419
action 8 wz rmse: 0.015818, mae: 0.011354, bias: 0.003138
worst source rmse: 0.045623
```

Decision: this passes the small in-distribution base-only BC gate. It should be used as a guarded Stage1 learning artifact, not as evidence that full 9D imitation is ready. Do not train from poor teachers, and do not promote the partial full-mask dataset until the arm/gimbal label validity problem is solved.

## 2026-07-06 Twenty-Case Stage1 Base-Only BC Gate

The 20-case Stage1 teacher export from the Mac/GIK side was synced after `.98` came back online. This is the preferred base-only Stage1 imitation artifact after the 6-case smoke.

Imported dataset:

```text
data/gik_stage1_recovery_gate20_basehome0_iter300_20260705/obs_dataset_base_only.npz
```

Import result:

```text
cases: 20
samples: 2826
obs_dim: 85
base-only label mask: [0,0,0,0,0,0,1,1,1]
full masked label mask: [1.0,0.1122,0.0,0.9625,0.9632,0.9632,1.0,1.0,1.0]
```

The full masked dataset remains diagnostic only because arm/gimbal validity is incomplete.

Base-only BC artifact:

```text
logs/bc/gik_stage1_gate20_baseonly_20260706/bc_policy.zip
```

Training summary:

```text
transitions: 2826
epochs: 10
best val MSE: 0.000643
```

Offline evaluator artifact:

```text
evaluation_results/bc/gik_stage1_gate20_baseonly_20260706.json
```

Evaluator summary on action rows `6,7,8`:

```text
aggregate rmse: 0.023468
aggregate mae: 0.015050
action 6 vx rmse: 0.029104, mae: 0.021763, bias: 0.003874
action 7 vy rmse: 0.025804, mae: 0.015832, bias: 0.004092
action 8 wz rmse: 0.011802, mae: 0.007554, bias: -0.000199
worst source rmse: 0.040504
```

Decision: this passes the 20-case in-distribution base-only BC gate and improves over the 6-case smoke. It can be used as a guarded Stage1 base prior or auxiliary signal. It is not evidence that full 9D imitation is ready.

## 2026-07-06 PPO Gate With Gate20 Base BC Aux

A bounded Stage1 PPO gate was run from the 20-case base-only BC policy. The purpose was to test whether the in-distribution base teacher can improve raw recovery without runtime base assist.

Training artifact:

```text
logs/sb3/recomoproto2trackee_v0/stage1_gate20_basebc_aux_64k_20260706/final_model.zip
logs/sb3/recomoproto2trackee_v0/stage1_gate20_basebc_aux_64k_20260706/vec_normalize.pkl
```

Key training settings:

```text
num_envs: 64
total_timesteps: 65536
pretrained_policy: logs/bc/gik_stage1_gate20_baseonly_20260706/bc_policy.zip
pretrained_action_indices: 6,7,8
base_assist_aux_dataset: data/gik_stage1_recovery_gate20_basehome0_iter300_20260705/obs_dataset_base_only.npz
base_assist_aux_action_indices: 6,7,8
runtime base assist: disabled
obstacles: enabled, radius=0.20m, height=0.50m
```

Evaluation artifact:

```text
evaluation_results/stage1_gate20_basebc_aux_64k_20260706/recovery_eval_raw-policy_20260706_100331.json
```

Evaluation result on 64 Stage1 recovery episodes:

```text
unreachable_zone_pct mean: 60.12
base_target_dist_mean: 0.847 m
base_target_dist_max mean: 1.547 m
workspace_hard_exceed_pct mean: 8.54
obstacle_unsafe_pct: 0.0
obstacle_collision_pct: 0.0
obstacle_clearance_min mean: 1.042 m
ee_pos_error_p95_m mean: 1.717
ee_ori_error_mean_deg mean: 133.61
```

Decision: reject as a policy candidate. The gate is safe around obstacles, but reachability and EE/camera tracking remain poor. The result says the 20-case base-only BC prior plus aux loss is not sufficient by itself when runtime base assist is disabled.

A headless rollout recorder was added for deterministic visual diagnostics:

```text
scripts/reinforcement_learning/sb3/record_recovery_rollout.py
```

Recorded diagnostic artifact for `crane_down_000`:

```text
evaluation_results/videos/stage1_gate20_basebc_aux_64k_20260706/crane_down_000_policy.gif
evaluation_results/videos/stage1_gate20_basebc_aux_64k_20260706/crane_down_000_policy.png
evaluation_results/videos/stage1_gate20_basebc_aux_64k_20260706/crane_down_000_policy.csv
evaluation_results/videos/stage1_gate20_basebc_aux_64k_20260706/crane_down_000_policy.json
```

The recorder confirms the failure mode visually on `crane_down_000`: base-target mean around `1.00m`, max `1.52m`, EE position error mean around `1.47m`.
