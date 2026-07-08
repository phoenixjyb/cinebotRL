# Proto2 Curriculum Rethink - 2026-07-07

## Why We Stop Blind Training

Recent bounded gates all regressed against the current yaw-assist baseline:

| Candidate | EE mean error | Ori mean error | Unreachable mean | Verdict |
| --- | ---: | ---: | ---: | --- |
| yaw_assist_baseline | 1.1442 m | 132.17 deg | 47.42% | baseline |
| yaw_assist_baseaux | 1.7962 m | 163.16 deg | 87.77% | regressed |
| grouped_no_aux | 1.7677 m | 166.65 deg | 91.20% | regressed |
| grouped_aux | 1.8187 m | 164.98 deg | 92.66% | regressed |

Audit artifact:

`evaluation_results/policy_regression_audit/curriculum_rethink_20260707/policy_regression_summary.csv`

Key observation:

Some failed policies have less negative evaluator reward while their tracking and reachability are worse. That means the current reward/curriculum can select behavior that looks numerically better but is functionally worse for EE tracking.

## Working Diagnosis

The current failures are probably not caused by network shape alone or insufficient timesteps.

Likely causes:

- Training uses strong base assist and shaping, but evaluation is raw-policy. A policy can learn to live under assist instead of owning the task.
- Base-only teacher labels over-constrain chassis motion while leaving arm/gimbal under-supervised. This can pull the robot into base behaviors that do not actually reduce EE error.
- The recovery task starts with targets often outside the arm workspace. If raw-policy early success is rare, PPO optimizes easier proxy terms.
- Obstacle and workspace safety are not the bottleneck right now. Tracking/reachability is.
- Reward scalar can be misleading. The audit showed better reward but worse tracking.

## RL-Only Hypothesis

RL-only may be better than IL-start for this phase, but only if the curriculum makes early raw-policy successes common.

Why RL-only could help:

- Avoids injecting poor or partial teacher labels.
- Lets the policy discover coordinated base/arm/gimbal behavior instead of copying a base-only controller.
- Avoids train/test mismatch caused by BC labels that include progress or frame assumptions not present in the current env observation.

Why RL-only can still fail:

- Exploration in 9D whole-body control is sparse and unstable.
- If the target begins far outside reachable workspace, the reward may be dominated by penalties/proxies.
- If base assist remains strong, the policy may still not learn raw recovery.

Conclusion:

Try RL-only, but not on the current full recovery task. Use a raw-policy-first curriculum where each stage has a deterministic evaluator gate.

## Proposed Curriculum

### Stage A: Raw Reachable Hold

Goal:

Teach the arm/gimbal policy to hold reachable EE targets without depending on base assist.

Settings:

- no obstacles
- no base assist
- no IL or BC
- target starts inside reachable workspace
- short fixed segments
- base either fixed or very small velocity range

Promotion gate:

- `ee_pos_error_mean_m < 0.25`
- `unreachable_zone_pct < 15%`
- `workspace_hard_exceed_pct = 0`
- raw-policy eval only

### Stage B: Raw Base Reposition

Goal:

Teach base movement only after Stage A tracking works.

Settings:

- no obstacles
- no IL or BC
- target starts near the edge of reachable workspace
- base action enabled
- base assist disabled or very low and decays to zero within the stage

Promotion gate:

- `ee_pos_error_mean_m < 0.45`
- `unreachable_zone_pct < 25%`
- base-target distance improves over episode without external assist

### Stage C: Recovery Segments

Goal:

Recover from moderate out-of-workspace starts.

Settings:

- no obstacles initially
- randomized start waypoint fraction bounded lower than current 0.25-0.70 range
- no base-only imitation
- use yaw-assist baseline only as a comparison, not as a teacher

Promotion gate:

- beat yaw-assist baseline on the same deterministic audit
- `unreachable_zone_pct < 47.42%`
- `ee_pos_error_mean_m < 1.1442 m`

### Stage D: Obstacles

Goal:

Add obstacle avoidance only after tracking/reachability is improving.

Settings:

- 40 cm diameter, 50 cm height obstacle
- obstacle randomized gradually
- keep raw-policy tracking gate active

Promotion gate:

- obstacle collision `0%`
- obstacle unsafe near `0%`
- no regression on Stage C tracking metrics

## Immediate Next Engineering Tasks

1. Add deterministic audit tooling.
2. Add an explicit raw RL-only Stage A config or CLI preset.
3. Add evaluator gates for Stage A and Stage B.
4. Run a tiny RL-only smoke first, then one bounded Stage A gate.

Do not run more base-only BC or base-only aux experiments unless a new audit proves the label distribution is aligned with raw-policy success.

## Stage A Smoke Result

Existing CLI flags are enough to express a first raw RL-only Stage A smoke:

```bash
PYTHONUTF8=1 NO_PROXY='*' no_proxy='*' /mnt/g/isaaclab_venv/Scripts/python.exe -X utf8 \
  scripts/reinforcement_learning/sb3/train.py \
  --headless \
  --num_envs 4 \
  --total_timesteps 32 \
  --n_steps 8 \
  --batch_size 8 \
  --n_epochs 1 \
  --trajectory_stage stage0_easy \
  --max_trajectories 4 \
  --min_trajectory_duration 5.0 \
  --disable_auto_base_assist \
  --disable_auto_base_assist_yaw \
  --save_freq 1000000 \
  --log_dir logs/sb3/recomoproto2trackee_v0/stage0_raw_rl_smoke_20260707
```

Smoke output:

- exit code: `0`
- observation dim: `84`
- action dim: `9`
- trajectory manifest: `trajectoryToLearn/stage0_easy/manifest.txt`
- selected trajectories: `4`
- category: `handheld_subtle`
- base assist: not enabled
- BC/IL: not enabled
- obstacles: not enabled
- reset reachability: `4/4` reachable, `0/4` unreachable

This proves the raw RL-only entrypoint is runnable.

## Proposed Next Bounded Stage A Gate

Run one small Stage A gate before any larger curriculum changes:

```bash
PYTHONUTF8=1 NO_PROXY='*' no_proxy='*' /mnt/g/isaaclab_venv/Scripts/python.exe -X utf8 \
  scripts/reinforcement_learning/sb3/train.py \
  --headless \
  --num_envs 64 \
  --total_timesteps 65536 \
  --n_steps 256 \
  --batch_size 1024 \
  --n_epochs 4 \
  --trajectory_stage stage0_easy \
  --max_trajectories 8 \
  --min_trajectory_duration 5.0 \
  --disable_auto_base_assist \
  --disable_auto_base_assist_yaw \
  --save_freq 32768 \
  --log_dir logs/sb3/recomoproto2trackee_v0/stage0_raw_rl_64k_20260707
```

Promotion gate:

- evaluate raw-policy only
- compare against the stage0 raw smoke/baseline once available
- require reachable starts to stay reachable rather than optimizing reward alone
- do not add obstacles or base-only imitation in this gate

## Stage A 64k Result

Run:

`logs/sb3/recomoproto2trackee_v0/stage0_raw_rl_64k_20260707`

Evaluation:

`evaluation_results/recovery_candidate/stage0_raw_rl_64k_20260707/recovery_eval_raw-policy_20260707_093937.json`

Training result:

- exit code: `0`
- final training `explained_variance=0.648`
- final `value_loss=0.058`
- final `approx_kl=0.0067758746`
- final `std=0.135`

Raw-policy evaluation:

- `ee_pos_error_mean_m.mean=1.1552`
- `ee_pos_error_p95_m.mean=1.3234`
- `ee_ori_error_mean_deg.mean=145.1841`
- `unreachable_zone_pct.mean=70.9273`
- `workspace_hard_exceed_pct.mean=0.0`
- `base_target_dist_mean.mean=0.7977`
- `base_target_dist_max.mean=0.9729`

Interpretation:

RL-only is runnable, but the first Stage A design is still too loose. At reset the easy trajectories are reachable, but the freely explored base motion quickly moves the robot away from the target. Training reward/value metrics improved, while raw-policy reachability remained poor. This is another example where PPO scalar health is not enough.

## Stage1 Obstacle Teacher Gate - 2026-07-08

Source teacher data:

- GIK Stage1 one-obstacle accepted subset: `data/gik_stage1_one_obstacle_accepted63_20260708/accepted_npz/manifest.json`
- Observation/action dataset: `data/gik_offline_teacher_obs/obs_dataset_stage1_one_obstacle_accepted63_full_masked_20260708.npz`
- Dataset shape: `obs=(1665,85)`, `actions=(1665,9)`, `sources=63`
- Obstacle contract: 40 cm diameter, 50 cm height exact-box obstacle metadata preserved from GIK

Base-only BC candidate:

- Policy: `logs/bc/gik_stage1_obstacle63_baseonly80_20260708/bc_policy.zip`
- Eval: `evaluation_results/bc/gik_stage1_obstacle63_baseonly80_20260708.json`
- Validation MSE: `0.001023`
- Base RMSE: `0.015518`
- Base MAE: `0.006399`
- Base max abs error: `0.541529`
- Labels used: `4913`

Interpretation:

The base-only BC head is good in offline teacher-label space, but that does not prove it is useful when inserted into the closed-loop sim policy.

### Row-Blend Closed-Loop Gate

Gate setup:

- Primary tracking checkpoint: `logs/bc/gik_no_obstacle79_masked9_smoke_20260708/bc_policy.zip`
- Base-head checkpoint: `logs/bc/gik_stage1_obstacle63_baseonly80_20260708/bc_policy.zip`
- Stage: `stage_gik_no_obstacle79_nominal`
- Env count/steps: `8 envs x 80 steps`
- Observation normalization disabled because both checkpoints are raw-observation BC policies
- Static obstacle: `radius=0.20 m`, `height=0.50 m`, `x=0.0`, `y=0.5`

Results:

| Candidate | Base blend | EE mean | EE final mean | EE p95 | Ori mean | Reward mean | Min obstacle clearance | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Primary BC only | 0.00 | 0.2590 m | 0.4766 m | 0.5283 m | 53.11 deg | -32.60 | 1.6994 m | baseline |
| Hybrid base head | 0.25 | 0.2402 m | 0.3523 m | 0.5076 m | 54.57 deg | -26.30 | 1.7560 m | candidate, small tracking gain |
| Hybrid base head | 0.50 | 0.3223 m | 0.4066 m | 0.6340 m | 57.65 deg | -26.51 | 1.7642 m | regressed tracking |
| Hybrid base head | 1.00 | 0.5130 m | 0.8252 m | 0.9582 m | 67.00 deg | -36.32 | 1.6533 m | reject |

Obstacle metrics were all collision-free and unsafe-free, but this is not evidence of obstacle avoidance. The fixed obstacle was more than `1.6 m` away from the active base path in this small gate, so the obstacle did not meaningfully challenge the policy.

Decision:

- Reject full Stage1 base-head replacement. It strongly worsens tracking and drives the chassis in a mismatched direction.
- Keep `0.25` row-blend only as a candidate for a representative Stage1 gate. It showed a small tracking improvement in this narrow no-obstacle-like closed-loop test.
- Do not run more row-blend sweeps on `stage_gik_no_obstacle79_nominal` with the fixed obstacle at `(0.0, 0.5)`. That setup cannot answer the avoidance question.

Next required gate:

Build a representative Stage1 obstacle evaluation before training or promoting any policy. Either:

- export the accepted Stage1 GIK trajectories into a sim-loadable `trajectoryToLearn/stage_gik_one_obstacle63_accepted` stage, or
- add evaluator support to place the obstacle along each loaded trajectory's base path or progress fraction using the GIK obstacle metadata.

The next promotion gate must measure both tracking and obstacle proximity on the same episodes:

- no obstacle collision
- near-obstacle unsafe percentage close to `0%`
- no tracking regression versus the primary BC baseline
- meaningful obstacle clearance, not a clearance floor above `1.5 m`

Updated Stage A requirement:

Before training arm/gimbal tracking, the base must be controlled:

- freeze base actions for the first raw reachable-hold stage, or
- mask base actions to zero in the environment, or
- add a very strong base-stationary penalty and action penalty.

Recommended next engineering change:

Add a CLI option such as `--freeze_base_actions` for Stage A. It should zero action rows `[6,7,8]` before they reach the env dynamics while preserving the 9D policy shape. Then rerun Stage A 64k with base frozen.

New Stage A promotion target:

- `unreachable_zone_pct.mean < 15%`
- `ee_pos_error_mean_m.mean < 0.5 m`
- `workspace_hard_exceed_pct.mean = 0`
- raw-policy eval only

## 2026-07-08 Stage1 Accepted Obstacle Teacher Dataset

The GIK handoff was updated after the raw RL-only rethink above. The current
teacher-data curriculum is:

- Stage 0 no obstacle: `79/79` accepted.
- Stage 1 one obstacle exact-box at 55% path progress: `63/79` accepted.
- Stage 1 holdout: 16 failed/debug episodes excluded from positive BC.
- Stage 2 two-obstacle: candidate only; not all79-proven.

The accepted Stage1 export is now synced on `.98`:

```text
data/gik_stage1_one_obstacle_accepted63_20260708/accepted_npz
data/gik_offline_teacher_obs/obs_dataset_stage1_one_obstacle_accepted63_full_masked_20260708.npz
```

Remote validation:

- exported trajectories: `63`
- exported action samples: `1665`
- export failures: `0`
- observation dataset: `observations=(1665,85)`, `actions=(1665,9)`
- source trajectories in obs dataset: `63`
- action range: `[-1.0, 1.0]`
- finite observations/actions: yes
- q mapping from GIK Proto2-PnC 13D logs: selected `[0,1,2,3,4,5,6,7,8]`; excluded virtual `[9,10,11,12]`

Use this as a bounded teacher-data gate before any long training run. It does
not override the earlier lesson that PPO reward alone is not enough; validate
tracking/reachability and obstacle metrics explicitly before promoting a policy.

## 2026-07-08 Stage1 Accepted Base-Only BC Gate

Ran a bounded base-only BC gate on the Stage1 accepted one-obstacle dataset.
Arm/gimbal labels were deliberately excluded from the loss with:

```text
--use_action_mask
--action_loss_weights 0,0,0,0,0,0,1,1,1
```

Artifacts on `.98`:

```text
logs/bc/gik_stage1_obstacle63_baseonly_smoke_20260708/bc_policy.zip
evaluation_results/bc/gik_stage1_obstacle63_baseonly_smoke_20260708.json
logs/bc/gik_stage1_obstacle63_baseonly80_20260708/bc_policy.zip
evaluation_results/bc/gik_stage1_obstacle63_baseonly80_20260708.json
```

Results:

| Candidate | Epochs | Best val MSE | Base RMSE | Base MAE | Base max abs | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `gik_stage1_obstacle63_baseonly_smoke_20260708` | 20 | `0.003876` | `0.052080` | `0.023596` | `0.692640` | undertrained |
| `gik_stage1_obstacle63_baseonly80_20260708` | 80 | `0.001023` | `0.015518` | `0.006399` | `0.541529` | useful base-head candidate |

Per-action RMSE for the 80-epoch checkpoint:

```text
base_vx/action[6]: 0.018625
base_vy/action[7]: 0.017514
base_wz/action[8]: 0.008507
```

Worst source for the 80-epoch checkpoint:

```text
source_index=30
source_name=60cd58a5e9_one_obstacle_055_exact_box_0041_vid_d728a8e2741d416b.npz
rmse=0.066708
max_abs=0.541529
```

Interpretation:

- The accepted Stage1 obstacle dataset is learnable for base-motion imitation.
- The 80-epoch base-only checkpoint is worth using as a bounded warm-start or
  distillation candidate.
- This is not proof that full 9D arm/gimbal imitation is ready; keep full-mask
  imitation diagnostic until arm/gimbal label quality is re-audited.
- Next gate should evaluate policy behavior in simulation, not only offline MSE.

## Offset-Follow Tail Recovery Probe - 2026-07-08

Question:

Can we teach the policy to move the chassis laterally in late-start `base040`
states by distilling a target-offset-follow teacher into base rows `[6,7]`?

Dataset:

`data/policy_envelope_fk_base040/offset_follow_tail_base025_20260708.npz`

Dataset facts:

- source checkpoint: `stage0_policy_envelope_fk_mix_contract_zpenalty80_lowstd_resume_32k_20260707/final_model.zip`
- stage: `stage0_policy_envelope_fk_base040`
- late-start range: `0.65-0.95`
- samples: `3383`
- observation dim: `84`
- action dim: `9`
- valid label rows: `[6,7]`
- teacher env action cap: `0.25`
- base action scale: `0.25`
- raw label clip fraction: `0.0`
- label space: raw policy action before base scaling

Teacher diagnostic:

The capped offset-follow teacher is learnable under the current
`base_action_scale=0.25` contract and still moves the base enough:

- assisted EE error: `0.0589 / 0.1108 / 0.2067 m`
- base XY motion: `0.1201 m`
- target XY motion: `0.1178 m`
- assist active: `90.4%`

Training probes:

| Candidate | Late-start EE mean/p95/max | Full-start EE mean/p95/max | Verdict |
| --- | ---: | ---: | --- |
| source checkpoint | `0.0904 / 0.1609 / 0.1940 m` | `0.0505 / 0.0749 / 0.1056 m` | current baseline |
| PPO aux rows `[6,7]` | `0.0929 / 0.1726 / 0.2019 m` | not run | reject; worse late-start |
| direct grouped-head distill rows `[6,7]` | `0.0706 / 0.1591 / 0.2165 m` | `0.0664 / 0.1244 / 0.1387 m` | reject; full-start regression and worse max |
| preserve-row grouped-head distill rows `[6,7]`, preserve `[8]` | `0.0634 / 0.1447 / 0.2048 m` | `0.0583 / 0.1060 / 0.1203 m` | diagnostic only; improves late-start mean/p95 but regresses full-start |

Artifacts:

- collector: `scripts/imitation/collect_offset_follow_tail_dataset.py`
- distill tool: `scripts/reinforcement_learning/sb3/distill_base_assist_head.py`
- best diagnostic checkpoint: `logs/sb3/recomoproto2trackee_v0/stage0_policy_envelope_fk_base040_offsetfollow_distill_preserve_20260708/final_model.zip`
- late-start gate: `evaluation_results/stage_rollout_gate/stage0_policy_envelope_fk_base040_offsetfollow_distill_preserve_tailstart_60step_20260708.json`
- full-start gate: `evaluation_results/stage_rollout_gate/stage0_policy_envelope_fk_base040_offsetfollow_distill_preserve_fullstart_60step_20260708.json`

Conclusion:

The offset-follow teacher is a valid diagnostic signal: it confirms the previous
failure mode that late-start tracking is dominated by base under-following.
However, directly distilling that signal into the grouped base head is not yet a
safe policy update. Even with row `[8]` preserved, the checkpoint regresses the
normal full-start `base040` gate and leaves a worse late-start max error.

Next rule:

Do not promote this checkpoint as the default `base040` policy. Use it only as a
recovery-teacher artifact. The next attempt should mix recovery labels with
normal-start preservation data, train a narrower final-layer/base-row adapter,
or route the recovery behavior conditionally instead of replacing the whole base
head.

## Mixed Recovery + Full-Start Preservation Probe - 2026-07-08

Question:

Can we keep the late-start recovery gains while preserving normal `base040`
behavior by mixing two datasets in the same grouped-head distill?

Added preservation dataset:

`data/policy_envelope_fk_base040/policy_preserve_fullstart_base040_20260708.npz`

Dataset facts:

- source checkpoint: `stage0_policy_envelope_fk_mix_contract_zpenalty80_lowstd_resume_32k_20260707/final_model.zip`
- stage: `stage0_policy_envelope_fk_base040`
- reset mode: full-start, no random start waypoint
- samples: `3840`
- valid label rows: `[6,7,8]`
- labels: current policy actions, not offset-follow teacher actions
- sample weight: uniform `1.5`
- source-state EE error: `0.0542 / 0.0913 / 0.1183 m`

Tooling update:

- `collect_offset_follow_tail_dataset.py` now supports `--policy_preserve_only`,
  `--policy_preserve_rows`, `--no-random_start_waypoint`, and
  `--sample_weight_scale`.
- `distill_base_assist_head.py` now supports multiple `--dataset` inputs and
  per-dataset `--dataset_weight_scales`.

Gate comparison:

| Candidate | Dataset scales | Late-start EE mean/p95/max | Full-start EE mean/p95/max | Verdict |
| --- | ---: | ---: | ---: | --- |
| source checkpoint | n/a | `0.0904 / 0.1609 / 0.1940 m` | `0.0505 / 0.0749 / 0.1056 m` | baseline |
| mixed preserve | `1,1` | `0.0533 / 0.1172 / 0.1765 m` | `0.0499 / 0.0997 / 0.1069 m` | good recovery, full-start p95 regression |
| mixed preserve | `1,2` | `0.0554 / 0.1128 / 0.1751 m` | `0.0465 / 0.0932 / 0.0986 m` | better full-start mean/max, p95 still high |
| mixed preserve | `1,4` | `0.0604 / 0.1136 / 0.1648 m` | `0.0472 / 0.0831 / 0.1026 m` | best compromise |
| mixed preserve | `1,8` | `0.0661 / 0.1132 / 0.1654 m` | `0.0485 / 0.0836 / 0.1094 m` | no better than `1,4` |

Current best checkpoint:

`logs/sb3/recomoproto2trackee_v0/stage0_policy_envelope_fk_base040_offsetfollow_mixed_preserve_w4_20260708/final_model.zip`

Interpretation:

The mixed-preservation approach is the first candidate that beats the source
checkpoint on all late-start metrics while also avoiding the earlier full-start
mean/max regression. The `1,4` preservation weighting is the best current
tradeoff: it improves late-start mean, p95, and max, and improves full-start
mean/max, but full-start p95 remains slightly worse than the source baseline.

Promotion rule:

Do not replace the default policy solely from this two-gate result. Treat the
`1,4` checkpoint as the current best `base040` recovery candidate. Before
promotion, run broader routed gates across the existing stage mix and confirm
that the full-start p95 increase does not show up as a visible trajectory
quality regression.

## Broader W4 Routed Validation - 2026-07-08

Question:

Does the `1,4` mixed-preservation checkpoint generalize beyond the two `base040`
gates, or is it overfit to the recovery probe?

Candidate:

`logs/sb3/recomoproto2trackee_v0/stage0_policy_envelope_fk_base040_offsetfollow_mixed_preserve_w4_20260708/final_model.zip`

Validation artifact:

`evaluation_results/stage_rollout_gate/broader_w4_validation_20260708/summary.json`

Compact gate settings:

- source and `w4` candidate
- `num_envs=8`
- `steps=60`
- `max_trajectories=8`
- `base_action_scale=0.25`
- full-start and late-start for each stage
- late-start range: `0.65-0.95`

Gate comparison:

| Stage | Mode | Source EE mean/p95/max | W4 EE mean/p95/max | Verdict |
| --- | --- | ---: | ---: | --- |
| `base025` | full | `0.0462 / 0.0682 / 0.0835 m` | `0.0384 / 0.0696 / 0.0749 m` | mean/max better, p95 flat |
| `base025` | late | `0.0569 / 0.0926 / 0.1144 m` | `0.0429 / 0.0743 / 0.1026 m` | better |
| `base040` | full | `0.0505 / 0.0749 / 0.1056 m` | `0.0472 / 0.0831 / 0.1026 m` | mean/max better, p95 worse |
| `base040` | late | `0.0904 / 0.1609 / 0.1940 m` | `0.0604 / 0.1136 / 0.1648 m` | much better |
| `large08` | full | `0.0452 / 0.0707 / 0.0902 m` | `0.0384 / 0.0706 / 0.0873 m` | better/flat |
| `large08` | late | `0.0471 / 0.0697 / 0.0829 m` | `0.0423 / 0.0731 / 0.0862 m` | mean better, p95/max slightly worse |
| `mix_large08_base025` | full | `0.0507 / 0.0754 / 0.0827 m` | `0.0440 / 0.0731 / 0.0857 m` | mean/p95 better, max slightly worse |
| `mix_large08_base025` | late | `0.0548 / 0.0816 / 0.1050 m` | `0.0389 / 0.0642 / 0.0715 m` | much better |

Interpretation:

The `w4` checkpoint passes broader validation as a recovery candidate. It is
not a clean global replacement because two small tail-risk regressions remain:

- `base040` full-start p95 worsens from `0.0749 m` to `0.0831 m`.
- `large08` late-start p95/max worsen slightly, despite a better mean.

Decision:

Keep source as the default safe policy. Use `w4` as the preferred conditional
recovery route when the robot is in late-start or base-under-following states.

Next engineering step:

Implement conditional routing rather than global replacement. The route should
prefer `w4` only when recovery evidence is present, such as late waypoint
fraction, larger base-target offset, or increasing EE position error. Then run a
rendered rollout check to see whether the remaining p95 tradeoff is visible.

## Conditional W4 Route Implementation - 2026-07-08

Implementation:

`scripts/reinforcement_learning/sb3/evaluate_stage_rollout_gate.py` now supports
an opt-in recovery checkpoint:

- `--recovery_checkpoint`
- `--recovery_route_min_waypoint_fraction`
- `--recovery_route_min_pos_error`
- `--recovery_route_min_base_target_distance`
- `--recovery_route_latch_once`

Runtime contract:

- Source checkpoint remains the default action source.
- Recovery checkpoint is evaluated only for envs that meet the route condition.
- With `--recovery_route_latch_once`, once an env enters recovery mode it stays
  routed for the rest of that rollout.
- The evaluator records route usage through `recovery_route_fraction` and related
  route diagnostics.

Validated route:

```bash
--checkpoint stage0_policy_envelope_fk_mix_contract_zpenalty80_lowstd_resume_32k_20260707/final_model.zip
--recovery_checkpoint stage0_policy_envelope_fk_base040_offsetfollow_mixed_preserve_w4_20260708/final_model.zip
--recovery_route_min_waypoint_fraction 0.65
--recovery_route_latch_once
--base_action_scale 0.25
```

Validation artifact:

`evaluation_results/stage_rollout_gate/routed_w4_latch_validation_20260708/summary.json`

Route result:

| Stage | Mode | Routed fraction | Routed EE mean/p95/max | Meaning |
| --- | --- | ---: | ---: | --- |
| `base025` | full | `0.0%` | `0.0462 / 0.0682 / 0.0835 m` | source preserved |
| `base025` | late | `99.8%` | `0.0428 / 0.0743 / 0.1026 m` | recovery route active |
| `base040` | full | `0.0%` | `0.0505 / 0.0749 / 0.1056 m` | source preserved |
| `base040` | late | `99.8%` | `0.0604 / 0.1139 / 0.1648 m` | recovery route active |
| `large08` | full | `0.0%` | `0.0452 / 0.0707 / 0.0902 m` | source preserved |
| `large08` | late | `99.8%` | `0.0426 / 0.0731 / 0.0862 m` | mean better, p95/max slightly worse than source |
| `mix_large08_base025` | full | `0.0%` | `0.0507 / 0.0754 / 0.0827 m` | source preserved |
| `mix_large08_base025` | late | `99.8%` | `0.0389 / 0.0638 / 0.0715 m` | recovery route active |

Decision:

The conditional route fixes the main blocker from global `w4` replacement:
normal/full-start gates remain exactly on the source policy and no longer inherit
the `w4` p95 tradeoff. Late-start gates get the recovery behavior. The remaining
caveat is `large08` late-start p95/max, where `w4` improves mean but slightly
worsens tail error. That is acceptable for the next visual check, but should not
be ignored.

Next step:

Record rendered rollouts for routed `base040` late-start and `large08` late-start
to inspect whether the remaining p95/max tradeoff is visible as jerk, drift, or
unnatural base motion.

## Routed W4 Rendered Rollouts - 2026-07-08

Implementation:

`scripts/reinforcement_learning/sb3/record_rendered_recovery_rollout.py` now
supports the same routed policy contract as the gate evaluator:

- `--recovery_checkpoint`
- `--recovery_route_min_waypoint_fraction`
- `--recovery_route_min_pos_error`
- `--recovery_route_min_base_target_distance`
- `--recovery_route_latch_once`
- `--disable_vec_normalize`
- `--base_action_scale`
- stage-specific reset offsets from `reset_config.json`

Rendered artifacts:

- `evaluation_results/videos_rendered/routed_w4_validation_20260708/base040_late_routed_w4_120/rl-video-step-0.mp4`
- `evaluation_results/videos_rendered/routed_w4_validation_20260708/large08_late_routed_w4_120/rl-video-step-0.mp4`

Local copies for review:

- `/Users/yanbo/Downloads/cinebotRL_routed_w4_20260708/base040_late_routed_w4_120.mp4`
- `/Users/yanbo/Downloads/cinebotRL_routed_w4_20260708/large08_late_routed_w4_120.mp4`

Render contract:

- 120 steps
- 20 FPS
- 1280x720 output
- raw observation policy path with `--disable_vec_normalize`
- `--base_action_scale 0.25`
- `--recovery_route_min_waypoint_fraction 0.65`
- `--recovery_route_latch_once`
- no obstacles
- late-start waypoint range `0.65-0.95`

Metadata:

- `base040_late_routed_w4_120`: `recovery_route_fraction=1.0`,
  `steps_executed=120`, `done_count=0`
- `large08_late_routed_w4_120`: `recovery_route_fraction=1.0`,
  `steps_executed=120`, `done_count=0`

Visual note:

Contact-sheet inspection shows stable robot rendering with no obvious detached
arm/gimbal or tip-over. The material is lighter/whiter than the earlier dark
polished renders, but the video is usable for motion validation.

## Stage A Freeze-Base Implementation

Implementation:

- `scripts/reinforcement_learning/sb3/train.py` supports `--freeze_base_actions`
- `scripts/reinforcement_learning/sb3/evaluate_recovery_candidate.py` supports `--freeze_base_actions`
- action rows `[6,7,8]` are zeroed immediately before `env.step(...)`
- the policy remains 9D, so this does not change checkpoint shape or action-contract naming

Tiny smoke:

```bash
PYTHONUTF8=1 NO_PROXY='*' no_proxy='*' /mnt/g/isaaclab_venv/Scripts/python.exe -X utf8 \
  scripts/reinforcement_learning/sb3/train.py \
  --headless \
  --num_envs 4 \
  --total_timesteps 32 \
  --n_steps 8 \
  --batch_size 8 \
  --n_epochs 1 \
  --trajectory_stage stage0_easy \
  --max_trajectories 4 \
  --min_trajectory_duration 5.0 \
  --disable_auto_base_assist \
  --disable_auto_base_assist_yaw \
  --freeze_base_actions \
  --save_freq 1000000 \
  --log_dir logs/sb3/recomoproto2trackee_v0/stage0_raw_rl_freezebase_smoke_20260707
```

Next bounded gate:

```bash
PYTHONUTF8=1 NO_PROXY='*' no_proxy='*' /mnt/g/isaaclab_venv/Scripts/python.exe -X utf8 \
  scripts/reinforcement_learning/sb3/train.py \
  --headless \
  --num_envs 64 \
  --total_timesteps 65536 \
  --n_steps 256 \
  --batch_size 1024 \
  --n_epochs 4 \
  --trajectory_stage stage0_easy \
  --max_trajectories 8 \
  --min_trajectory_duration 5.0 \
  --disable_auto_base_assist \
  --disable_auto_base_assist_yaw \
  --freeze_base_actions \
  --save_freq 32768 \
  --log_dir logs/sb3/recomoproto2trackee_v0/stage0_raw_rl_freezebase_64k_20260707
```

Evaluate the frozen-base gate with matching action semantics:

```bash
PYTHONUTF8=1 NO_PROXY='*' no_proxy='*' /mnt/g/isaaclab_venv/Scripts/python.exe -X utf8 \
  scripts/reinforcement_learning/sb3/evaluate_recovery_candidate.py \
  --headless \
  --checkpoint logs/sb3/recomoproto2trackee_v0/stage0_raw_rl_freezebase_64k_20260707/final_model.zip \
  --num_envs 64 \
  --num_episodes 128 \
  --trajectory_stage stage0_easy \
  --max_trajectories 8 \
  --min_trajectory_duration 5.0 \
  --no-enable_obstacles \
  --freeze_base_actions \
  --output_dir evaluation_results/recovery_candidate/stage0_raw_rl_freezebase_64k_20260707
```

## Stage A Freeze-Base 64k Result

Run:

`logs/sb3/recomoproto2trackee_v0/stage0_raw_rl_freezebase_64k_20260707`

Evaluation:

`evaluation_results/recovery_candidate/stage0_raw_rl_freezebase_64k_20260707/recovery_eval_raw-policy_20260707_095331.json`

Training result:

- exit code: `0`
- final training `explained_variance=0.801`
- final `value_loss=0.0465`
- final `approx_kl=0.005145751`
- final `std=0.135`
- action adapter confirmed rows `[6,7,8]` zeroed before env dynamics

Raw-policy freeze-base evaluation:

- `freeze_base_actions=true`
- `episodes_completed=128`
- `ee_pos_error_mean_m.mean=1.6781`
- `ee_pos_error_p95_m.mean=1.9667`
- `ee_ori_error_mean_deg.mean=167.6740`
- `unreachable_zone_pct.mean=82.2338`
- `workspace_soft_exceed_pct.mean=81.9517`
- `workspace_hard_exceed_pct.mean=0.0`
- `base_target_dist_mean.mean=0.8390`
- `base_target_dist_max.mean=1.0512`

Promotion result:

FAILED. The base-freeze mechanism itself works, but the `stage0_easy` recorded trajectories are not a true fixed-base arm/gimbal curriculum. At reset the targets can start reachable, yet the recorded camera path moves outside the stationary-base envelope, so frozen-base Stage A cannot satisfy the reachability target.

Lesson:

Do not continue training this exact frozen-base gate. The next Stage A design must either:

- construct fixed-base micro trajectories whose desired EE path stays inside the arm/gimbal reachable workspace, or
- use a teacher/base-assist policy for base placement while learning arm/gimbal tracking, or
- segment recorded trajectories into short reachable windows with base reset/re-anchor per segment.

## Stage A Fixed-Base Micro Curriculum

Implementation target:

- add `stage0_fixedbase_micro` as a separate curriculum stage
- generate deterministic short paths under `trajectoryToLearn/stage0_fixedbase_micro/generated`
- keep each path close to a known reachable target center `[1.05, 0.08, 0.86]`
- keep base actions frozen with `--freeze_base_actions`
- use this only as an arm/gimbal tracking gate, not as a cinematic trajectory benchmark

Generation:

```bash
PYTHONUTF8=1 python3 scripts/generate_fixedbase_micro_stage.py
```

Tiny smoke:

```bash
PYTHONUTF8=1 NO_PROXY='*' no_proxy='*' /mnt/g/isaaclab_venv/Scripts/python.exe -X utf8 \
  scripts/reinforcement_learning/sb3/train.py \
  --headless \
  --num_envs 4 \
  --total_timesteps 32 \
  --n_steps 8 \
  --batch_size 8 \
  --n_epochs 1 \
  --trajectory_stage stage0_fixedbase_micro \
  --max_trajectories 4 \
  --min_trajectory_duration 5.0 \
  --disable_auto_base_assist \
  --disable_auto_base_assist_yaw \
  --freeze_base_actions \
  --save_freq 1000000 \
  --log_dir logs/sb3/recomoproto2trackee_v0/stage0_fixedbase_micro_smoke_20260707
```

Bounded gate:

```bash
PYTHONUTF8=1 NO_PROXY='*' no_proxy='*' /mnt/g/isaaclab_venv/Scripts/python.exe -X utf8 \
  scripts/reinforcement_learning/sb3/train.py \
  --headless \
  --num_envs 64 \
  --total_timesteps 65536 \
  --n_steps 256 \
  --batch_size 1024 \
  --n_epochs 4 \
  --trajectory_stage stage0_fixedbase_micro \
  --max_trajectories 24 \
  --min_trajectory_duration 5.0 \
  --disable_auto_base_assist \
  --disable_auto_base_assist_yaw \
  --freeze_base_actions \
  --save_freq 32768 \
  --log_dir logs/sb3/recomoproto2trackee_v0/stage0_fixedbase_micro_64k_20260707
```

## Stage A Fixed-Base Micro 64k Result

Generated stage:

- `scripts/generate_fixedbase_micro_stage.py`
- `trajectoryToLearn/stage0_fixedbase_micro/manifest.txt`
- 25 generated trajectories
- 60 waypoints per trajectory
- 6.0 seconds per trajectory at loader `waypoint_dt=0.1`
- bounds: `x=[0.955, 1.145]`, `y=[0.030, 0.130]`, `z=[0.810, 0.910]`

Run:

`logs/sb3/recomoproto2trackee_v0/stage0_fixedbase_micro_64k_20260707`

Evaluation:

`evaluation_results/recovery_candidate/stage0_fixedbase_micro_64k_20260707/recovery_eval_raw-policy_20260707_103701.json`

Training result:

- exit code: `0`
- final training `explained_variance=0.683`
- final `value_loss=0.0367`
- final `approx_kl=0.0059682103`
- final `std=0.135`
- action adapter confirmed rows `[6,7,8]` zeroed before env dynamics

Raw-policy freeze-base evaluation:

- `freeze_base_actions=true`
- `episodes_completed=128`
- `episode_length_mean=399.0`
- `ee_pos_error_mean_m.mean=0.9691`
- `ee_pos_error_p95_m.mean=1.0246`
- `ee_ori_error_mean_deg.mean=152.8359`
- `unreachable_zone_pct.mean=0.0`
- `workspace_soft_exceed_pct.mean=0.0`
- `workspace_hard_exceed_pct.mean=0.0`
- `base_target_dist_mean.mean=0.5026`
- `base_target_dist_max.mean=0.5946`

Promotion result:

PARTIAL PASS. The curriculum design fixed the previous reachability failure: frozen-base micro trajectories stay inside the reachable workspace for full episodes. However, the policy itself has not learned usable EE tracking after only 64k PPO steps; mean position error remains about `0.97 m` and orientation error remains about `153 deg`.

Lesson:

This stage is now valid as a fixed-base Stage A learning environment, but pure PPO from random initialization is still too weak for arm/gimbal tracking. The next policy update should add supervised warm-start or action-space simplification before another long run:

- start with a BC/imitation seed for the 6 arm/gimbal rows on `stage0_fixedbase_micro`, or
- temporarily train only the hold/line micro paths before oval/figure-eight paths, or
- add a lower-level IK/PD teacher label export for these generated targets and use PPO only for refinement.

## Stage A BC/IK Teacher Diagnosis

Implemented diagnostic collectors:

- `scripts/imitation/collect_fixedbase_micro_zero_teacher.py`
- `scripts/imitation/collect_fixedbase_micro_diffik_teacher.py`

Zero-home teacher smoke:

- output: `data/fixedbase_micro_zero_teacher/smoke_zero_arm6.npz`
- samples: `16`
- observation dim: `84`
- valid action rows: `[0,1,2,3,4,5]`
- zero-action mean position error: `0.9866 m`
- zero-action p95 position error: `1.3332 m`

Conclusion: safe-home action `0` is not a useful teacher for the current micro targets. It reproduces the same roughly `1 m` tracking error as the failed PPO policy.

DiffIK teacher full diagnostic:

- output: `data/fixedbase_micro_diffik_teacher/obs_dataset_diffik_arm6.npz`
- samples: `8192`
- observation dim: `84`
- action dim: `9`
- action contract: `sim_6joint_gimbal_v1`
- base rows `[6,7,8]`: unlabelled, still frozen for Stage A

Valid-label counts:

- row `0` / `joint6_arm_yaw`: `0 / 8192`
- row `1` / `joint5_arm_pitch`: `0 / 8192`
- row `2` / `joint4_elbow_pitch`: `687 / 8192` (`8.39%`)
- row `3` / `joint3_gimbal_yaw`: `5346 / 8192` (`65.26%`)
- row `4` / `joint2_gimbal_roll`: `345 / 8192` (`4.21%`)
- row `5` / `joint1_gimbal_pitch`: `87 / 8192` (`1.06%`)

Conclusion:

Do not BC-train from this DiffIK dataset as a 6D arm/gimbal teacher. The current `stage0_fixedbase_micro` targets are reachable in the broad reach-map sense, but their DiffIK joint targets mostly fall outside the current RL-safe action envelope:

- `arm_safe_home = [0.0, 1.0, -1.2, 0.0, 0.0, 0.0]`
- `arm_action_radius = [1.0, 0.45, 0.8, 1.0, 0.8, 0.8]`

This explains why pure PPO sees reachable workspace metrics but still cannot learn good tracking: the target distribution is not guaranteed to be reachable by the policy action envelope.

Next required fix:

Generate Stage A targets from the policy envelope, not from hand-picked Cartesian offsets:

1. sample smooth normalized arm/gimbal action paths inside a conservative subset of `[-1,1]^6`
2. use Isaac FK to convert those action paths to EE target poses
3. write those FK poses as a new trajectory stage, for example `stage0_policy_envelope_fk`
4. save the sampled normalized actions as exact BC labels for rows `[0,1,2,3,4,5]`
5. train BC on that dataset, then PPO-refine with `--freeze_base_actions`

This makes the Stage A teacher self-consistent: every target waypoint is generated by an action that the policy is actually allowed to output.

## Stage A Policy-Envelope FK Expansion

The policy-envelope FK curriculum is now the active fixed-base Stage A path. It avoids the failed DiffIK label problem by sampling valid normalized arm/gimbal actions first, using Isaac FK to create EE target poses, and reusing those exact actions as BC labels for rows `[0,1,2,3,4,5]`.

Confirmed earlier gates:

- `stage0_policy_envelope_fk_slow` (`path_action_radius=0.03`) PPO gate: mean EE position error `0.030 m`, p95 `0.050 m`, max `0.0566 m`, mean orientation error `4.4 deg`.
- `stage0_policy_envelope_fk_medium` (`path_action_radius=0.06`) PPO gate: mean EE position error `0.036 m`, p95 `0.068 m`, max `0.079 m`, mean orientation error `4.9 deg`.

Rejected expansion:

- `stage0_policy_envelope_fk_large` (`path_action_radius=0.10`) generated valid labels for all six arm/gimbal rows.
- Teacher open-loop was borderline but acceptable: mean EE position error `0.0409 m`, p95 `0.0795 m`, max `0.0799 m`, mean orientation error `5.03 deg`.
- BC rollout failed the promotion intent: mean EE position error `0.0692 m`, p95 `0.1244 m`, max `0.1307 m`, mean orientation error `4.73 deg`.
- Decision: do not PPO-train or promote the `0.10 m` radius. Shrink the expansion instead.

Promoted expansion:

- `stage0_policy_envelope_fk_large08` (`path_action_radius=0.08`) generated `24` trajectories and `1440` observation/action rows.
- Label validity: `[1440, 1440, 1440, 1440, 1440, 1440, 0, 0, 0]`.
- Teacher open-loop: mean EE position error `0.0332 m`, p95 `0.0584 m`, max `0.0588 m`, mean orientation error `4.20 deg`.
- BC offline: RMSE `0.00741`, max abs action error `0.04855`.
- BC rollout: mean EE position error `0.0498 m`, p95 `0.0829 m`, max `0.0920 m`, mean orientation error `5.04 deg`.
- PPO 64k raw-observation/no-VecNormalize gate: mean EE position error `0.0350 m`, p95 `0.0648 m`, max `0.0713 m`, mean orientation error `3.12 deg`.

Training details:

- log dir: `logs/sb3/recomoproto2trackee_v0/stage0_policy_envelope_fk_large08_rawbc_novec_64k_20260707`
- warm start: `data/policy_envelope_fk_large08/bc_policy_policy_envelope_fk_large08_arm6.zip`
- flags: `--freeze_base_actions`, `--disable_vec_normalize`, `--disable_auto_base_assist`, `--disable_auto_base_assist_yaw`
- reachability stayed `64/64` during training
- final PPO KL was low (`approx_kl=0.0026930673`)

Decision:

Promote `stage0_policy_envelope_fk_large08` as the next fixed-base Stage A checkpoint. Do not jump straight to obstacles. The next curriculum step should cautiously unfreeze base behavior on a small, deterministic set while keeping the `large08` fixed-base gate as a regression test.

## Stage A Base-Scale Unfreeze Gate

The first base-unfreeze step should not expose full chassis authority. The fixed-base checkpoint has only been trained with rows `[6,7,8]` zeroed before environment dynamics, so its base head is not yet a trustworthy control policy.

Implementation:

- added `--base_action_scale` to `train.py`
- added `--base_action_scale` to `evaluate_stage_rollout_gate.py`
- added `--base_action_scale` to `evaluate_recovery_candidate.py`
- semantics: when `--freeze_base_actions` is set, base rows `[6,7,8]` are still zeroed; otherwise rows `[6,7,8]` are multiplied by `--base_action_scale` before `env.step(...)`
- valid scale range: `[0,1]`

Pre-training audits on the promoted `large08` checkpoint:

- frozen-base PPO gate: mean EE position error `0.0350 m`, p95 `0.0648 m`, max `0.0713 m`, mean orientation error `3.12 deg`
- no-freeze audit: mean EE position error `0.0457 m`, p95 `0.0697 m`, max `0.0771 m`, mean orientation error `3.34 deg`
- reduced base scale `0.25`: mean EE position error `0.0343 m`, p95 `0.0676 m`, max `0.0727 m`, mean orientation error `3.18 deg`

Training:

```bash
scripts/reinforcement_learning/sb3/train.py \
  --trajectory_stage stage0_policy_envelope_fk_large08 \
  --checkpoint logs/sb3/recomoproto2trackee_v0/stage0_policy_envelope_fk_large08_rawbc_novec_64k_20260707/final_model.zip \
  --disable_vec_normalize \
  --base_action_scale 0.25 \
  --learning_rate 1e-5 \
  --total_timesteps 65536
```

Run:

`logs/sb3/recomoproto2trackee_v0/stage0_policy_envelope_fk_large08_base025_resume_64k_20260707`

Training observations:

- exit code: `0`
- base action adapter confirmed rows `[6,7,8]` scaled by `0.25`
- final `approx_kl=0.0018498414`
- brief reachability dips were seen during training (`60/64`, `62/64`), so this still needs deterministic gates before promotion

Post-training gates:

- reduced base scale `0.25`: mean EE position error `0.0308 m`, p95 `0.0558 m`, max `0.0624 m`, mean orientation error `3.24 deg`
- fixed-base regression gate with `--freeze_base_actions`: mean EE position error `0.0328 m`, p95 `0.0598 m`, max `0.0694 m`, mean orientation error `3.30 deg`

Decision:

Promote the `--base_action_scale 0.25` adapter and checkpoint evidence as the first cautious base-unfreeze step. Do not move to obstacles yet. The next step should create a small deterministic base-required stage where the target is just outside the stationary arm envelope and require the policy to reduce base-target distance without regressing the `large08` fixed-base gate.

## Stage A Base-Required Micro Stage

Implemented a first deterministic base-required stage:

- generator: `scripts/imitation/generate_policy_envelope_fk_base_required_stage.py`
- stage: `trajectoryToLearn/stage0_policy_envelope_fk_base025`
- dataset: `data/policy_envelope_fk_base025/obs_dataset_policy_envelope_fk_base025_arm6_base3.npz`
- source stage: `stage0_policy_envelope_fk_large08`
- max generated base offset: `0.0618 m`
- base authority contract: train/eval with `--base_action_scale 0.25`
- valid action labels: all 9 rows `[0..8]`

Base-controller fix:

- `env.py` now explicitly integrates planar root `x/y/yaw` for direct root control.
- `_sanitize_base_root_state()` no longer wipes valid planar velocity command buffers for planar-only cleanup. It still zeroes buffers for finite/invalid-state repair.

Teacher and BC gates:

- Open-loop teacher with base labels: mean EE position error `0.0494 m`, p95 `0.0663 m`, max `0.0670 m`, mean orientation error `4.68 deg`.
- BC offline all-rows RMSE: `0.01534`, max abs action error `0.2625`; largest error is base `vy`.
- BC physical rollout: mean EE position error `0.0475 m`, p95 `0.0777 m`, max `0.0932 m`, mean orientation error `6.17 deg`.
- BC fixed-base regression on `large08` with `--freeze_base_actions`: mean EE position error `0.0366 m`, p95 `0.0710 m`, max `0.0829 m`, mean orientation error `4.43 deg`.

PPO result:

Do not promote PPO for this stage yet. Two PPO attempts were stopped early:

- default `log_std=-2` run: reachability dropped to `29/64` and `26/64`
- low-exploration `log_std=-4`, `ent_coef=0` run: reachability still dropped to `39/64`, `44/64`, and `39/64`

Interpretation:

The base-required teacher and BC policy are usable diagnostic assets, but PPO refinement is not ready. The current reachability monitor/reward appears to treat the moving-base target distribution as frequently unreachable during stochastic rollout, even though the deterministic teacher/BC gates track at centimeter scale. The next policy update should fix the moving-base reachability/reward accounting before another PPO run.

## Stage A Moving-Base Reachability Accounting

Problem:

The first `base025` PPO attempts collapsed in the reachability monitor even when deterministic teacher/BC rollouts tracked at centimeter scale. The generated FK stages naturally use a base-target working distance around `0.85-0.90 m`, while the older reward defaults treated `0.7 m` as the hard reachability margin and used a fixed reach-map query tolerance of `0.1 m`.

Implementation:

- added `reachability_query_tolerance` to the reward config and env reward-weight dictionary
- env reach-map query now uses `reward_weights["reachability_query_tolerance"]` instead of hardcoded `0.1`
- stage `reset_config.json` can now include `reward_overrides`
- `train.py`, `evaluate_stage_rollout_gate.py`, and `evaluate_recovery_candidate.py` apply stage-local reward overrides
- `stage0_policy_envelope_fk_base025/reset_config.json` now sets generated-FK-specific margins:
  - `reachability_query_tolerance=0.2`
  - `reachability_optimal_distance=0.85`
  - `reachability_hard_margin=1.05`
  - reduced far-target/base-command penalties for this stage only

Validation:

- BC gate with reward overrides still tracks the same: mean EE position error `0.0475 m`, p95 `0.0777 m`, max `0.0932 m`
- reward mean improved from about `25.0` to `35.8`, confirming the old far-target penalties were inappropriate for this generated stage
- low-exploration PPO probe, `32k` steps with `log_std=-4`, stayed `64/64` reachable through all logged checks
- PPO probe training remained stable: final `approx_kl=0.00756`, `std=0.0183`
- PPO probe rollout on `base025`: mean EE position error `0.0474 m`, p95 `0.0733 m`, max `0.0967 m`, mean orientation error `6.09 deg`
- PPO probe fixed-base regression on `large08`: mean EE position error `0.0374 m`, p95 `0.0659 m`, max `0.0801 m`, mean orientation error `4.55 deg`

Decision:

Promote the accounting fix and the stable PPO probe evidence. Do not claim policy improvement yet: PPO is now stable on `base025`, but it has not materially improved tracking beyond BC. The next update should target policy learning quality, not reachability plumbing.

## Stage A Learning-Quality Probe

Goal:

Improve the `base025` policy without reopening reachability plumbing. The stable reference remains the low-exploration `32k` PPO probe from the reward-override stage.

Weighted BC probe:

- added `--action_loss_weights` to `scripts/reinforcement_learning/bc/pretrain_bc.py`
- default behavior is unchanged; when specified, masked MSE is weighted per action dimension
- tested weights `1,1,1,1,1,1,3,3,1` to emphasize chassis `vx/vy`
- offline result improved strongly: RMSE `0.00367`, max abs action error `0.03563`
- physical rollout got worse: mean EE position error `0.0485 m`, p95 `0.0995 m`, max `0.1148 m`, mean orientation error `4.70 deg`

Decision:

Do not promote the weighted BC policy. The offline loss improvement did not transfer to the closed-loop Isaac rollout. This is a useful tool for future controlled experiments, but weighted BC artifacts should stay out of the promoted path.

Longer PPO continuation:

- resumed from the stable `32k` reward-override PPO checkpoint
- trained with `--learning_rate 5e-6`, `--policy_log_std_override -4.0`, `--disable_vec_normalize`, and `--base_action_scale 0.25`
- run: `logs/sb3/recomoproto2trackee_v0/stage0_policy_envelope_fk_base025_rewardoverride_lowstd_resume_96k_20260707`
- training stayed reachable through logged checks: `64/64`
- final logged `approx_kl=0.00511`
- moving-base `base025` gate improved: mean EE position error `0.0434 m`, p95 `0.0694 m`, max `0.0946 m`, mean orientation error `5.90 deg`
- fixed-base `large08` regression worsened: mean EE position error `0.0416 m`, p95 `0.0730 m`, max `0.0780 m`, mean orientation error `5.01 deg`

Decision:

Do not promote the `96k` continuation as the default policy. It improves the moving-base micro stage but erodes the fixed-base regression gate. The next improvement should use a mixed objective or mixed evaluation set during training, rather than optimizing only `base025` and checking `large08` after the fact.

## Stage A Mixed-Stage Probe

Goal:

Train with both fixed-base `large08` and moving-base `base025` trajectories in the same PPO run, so the optimizer sees the regression case during training instead of only during post-training evaluation.

Implementation:

- added `scripts/imitation/generate_policy_envelope_fk_mixed_stage.py`
- generated `stage0_policy_envelope_fk_mix_large08_base025`
- source stages are interleaved deterministically, not concatenated, so bounded runs with `--max_trajectories` see both sources
- trajectory count: `24` from `large08` plus `24` from `base025`
- reused `base025` reward overrides because the mixed stage still contains moving-base generated FK targets
- added the mixed stage to the `train.py` stage whitelist

Training:

- resumed from the stable `32k` reward-override PPO checkpoint
- trained `32k` more steps with `--learning_rate 3e-6`, `--policy_log_std_override -4.0`, `--disable_vec_normalize`, and `--base_action_scale 0.25`
- run: `logs/sb3/recomoproto2trackee_v0/stage0_policy_envelope_fk_mix_large08_base025_lowstd_resume_32k_20260707`
- training stayed reachable through logged checks: `64/64`
- final logged `approx_kl=0.00561`

Validation:

- mixed-stage baseline with the original `32k` policy: mean EE position error `0.0474 m`, p95 `0.0738 m`, max `0.0864 m`, mean orientation error `6.00 deg`
- mixed-stage trained candidate: mean EE position error `0.0446 m`, p95 `0.0690 m`, max `0.0840 m`, mean orientation error `5.65 deg`
- moving-base `base025` gate: mean EE position error `0.0462 m`, p95 `0.0752 m`, max `0.0952 m`, mean orientation error `5.84 deg`
- fixed-base `large08` regression: mean EE position error `0.0455 m`, p95 `0.0786 m`, max `0.0904 m`, mean orientation error `5.10 deg`

Decision:

Do not promote the mixed-stage PPO candidate. It improves the aggregate mixed gate, but the component gates show the same underlying issue: pure moving-base performance is weaker than the `96k` candidate, and fixed-base regression is worse than the original `32k` reference. The mixed stage is useful as a diagnostic generator, but the next policy update needs an explicit anti-regression objective or staged policy selection, not just mixed sampling.

## Stage A Tracking Diagnostics and Contract-Aware Mixed Probe

Goal:

Improve tracking accuracy without blindly extending PPO. The previous probes showed stable reachability but inconsistent transfer between moving-base and fixed-base gates.

Diagnostics implementation:

- added `scripts/reinforcement_learning/sb3/diagnose_stage_tracking.py`
- reports aggregate, per-source, per-time-bin, and worst-env tracking metrics
- captures EE position/orientation error, signed XYZ error, arm target lag, base/target XY motion, action magnitudes, and trajectory filenames

Reference diagnostic on the promoted `32k` reward-override policy:

- mixed-stage mean/p95/max EE position error: `0.0474 / 0.0738 / 0.0864 m`
- source split: `base025 = 0.0430 / 0.0673 / 0.0772 m`, `large08 = 0.0517 / 0.0754 / 0.0864 m`
- mean signed XYZ error: `[+0.0107, -0.0161, -0.0388] m`
- interpretation: the dominant systematic bias is vertical under-tracking, with EE roughly `3.5-4.0 cm` below target; arm target lag is modest at about `0.030 rad`

Vertical penalty probe:

- added optional reward weight `vertical_position_penalty`, default `0.0`
- tested mixed-stage override `vertical_position_penalty=80.0`
- training stayed stable and reachable: `64/64`, final `approx_kl=0.00479`
- `base025` gate improved versus the promoted `32k` reference: `0.0461 / 0.0709 / 0.0880 m` versus `0.0474 / 0.0733 / 0.0967 m`
- fixed-base `large08` still regressed: `0.0433 / 0.0748 / 0.0822 m` versus `0.0374 / 0.0659 / 0.0801 m`
- decision: do not promote; vertical shaping helps moving-base tracking but does not solve fixed-base regression

Contract-aware mixed adapter:

- added `--freeze_base_actions_for_non_base_required` to `train.py`
- behavior: during mixed FK training, trajectories whose metadata filename does not contain `base_required` get base rows `[6,7,8]` zeroed; `base_required` trajectories still use `--base_action_scale`
- reason: this matches the actual gate contract better than previous mixed sampling, where fixed-base samples were trained with base motion and then evaluated with base frozen

Contract-aware results:

- contract + vertical penalty: stable training, final `approx_kl=0.00465`
- contract + vertical `base025` gate: `0.0462 / 0.0682 / 0.0835 m`
- contract + vertical fixed-base `large08` gate: `0.0384 / 0.0706 / 0.0855 m`
- contract-only: stable training, final `approx_kl=0.00444`
- contract-only `base025` gate: `0.0468 / 0.0716 / 0.0995 m`
- contract-only fixed-base `large08` gate: `0.0386 / 0.0759 / 0.0872 m`

Decision:

Do not promote any new policy from this probe. The best moving-base metrics improved, but every candidate still regressed the fixed-base `large08` promotion gate against the `32k` reference. Keep the diagnostics and contract-aware adapter; next improvement should avoid replacing the whole policy with a single mixed PPO checkpoint. Prefer staged policy selection or DAgger-style relabeling on rollout states while preserving the fixed-base actor behavior.

## Stage A Routed Policy Selection

Goal:

Avoid the single-policy tradeoff by selecting the best available policy per trajectory regime:

- default/fixed-base trajectories: promoted `32k` reward-override checkpoint
- `base_required` trajectories: contract + vertical candidate checkpoint

Implementation:

- added `scripts/reinforcement_learning/sb3/evaluate_stage_policy_router.py`
- routes per-env actions by `trajectory_manager.current_trajectory_metadata`
- filenames containing `base_required` use `--checkpoint_base_required`
- all other filenames use `--checkpoint_default`
- default-policy base rows `[6,7,8]` are frozen before env dynamics
- base-required policy base rows `[6,7,8]` are scaled by `--base_required_action_scale`

Validation:

- mixed stage route count: `240` default samples, `240` base-required samples
- mixed stage: mean/p95/max EE position error `0.0414 / 0.0678 / 0.0754 m`, mean orientation error `5.34 deg`
- pure `base025`: mean/p95/max EE position error `0.0462 / 0.0682 / 0.0835 m`
- pure `large08`: mean/p95/max EE position error `0.0374 / 0.0659 / 0.0801 m`
- pure `large08` route count: `480` default samples, `0` base-required samples
- pure `base025` route count: `0` default samples, `480` base-required samples

Decision:

Promote routed policy selection as the current best operational evaluation strategy. It preserves the fixed-base `large08` reference exactly while using the better moving-base policy for `base_required` trajectories. This is not yet a single deployable checkpoint; it is a policy-routing contract. The next implementation step is to make the same routing available in any runtime/evaluation entrypoint that needs mixed fixed-base and moving-base trajectories.

## Stage B Larger Base-Required Curriculum Seed

Goal:

Start increasing moving-base target displacement before introducing obstacles. Stage B keeps the same validated FK source and action contract as Stage A, but asks the base to cover a larger generated offset.

Implementation:

- extended `scripts/imitation/generate_policy_envelope_fk_base_required_stage.py`
- added optional `--include_stage_reward_overrides`
- generated `stage0_policy_envelope_fk_base040`
- source stage: `stage0_policy_envelope_fk_large08`
- source dataset: `data/policy_envelope_fk_large08/obs_dataset_policy_envelope_fk_large08_arm6.npz`
- output dataset: `data/policy_envelope_fk_base040/obs_dataset_policy_envelope_fk_base040_arm6_base3.npz`
- base action scale: `0.25`
- base offset radius: `0.16 m`
- max generated base offset: `0.1649 m`
- max abs base action: `1.0`
- base action p95: about `0.133`, so the hard max is a spike rather than the typical command level

Baseline gates:

- current routed policy on `base040`: mean/p95/max EE position error `0.0505 / 0.0749 / 0.1056 m`, mean orientation error `6.35 deg`
- direct `base040` BC, 40 epochs, offline RMSE `0.0102`, max abs action error `0.2071`
- direct `base040` BC rollout: mean/p95/max EE position error `0.0495 / 0.0806 / 0.1119 m`, mean orientation error `6.84 deg`

Decision:

Do not promote `base040` yet. The routed Stage A policy can run it, but max error crosses `10 cm`, and direct BC does not improve the tail. Stage B is now a valid curriculum seed, not a solved stage. The next Stage B update should train a bounded PPO continuation from the routed/base-required candidate or use DAgger-style rollout relabeling, with Stage A routed gates kept as regressions.

## Stage B Base040 PPO Continuation Probe

Goal:

Test whether a short, low-exploration PPO continuation on the larger `base040` moving-base curriculum improves Stage B without regressing the Stage A routed gates.

Implementation:

- added `stage0_policy_envelope_fk_base040` to the `train.py` trajectory-stage whitelist
- resumed from the current base-required contract + vertical candidate checkpoint
- trained `32k` additional PPO steps on `stage0_policy_envelope_fk_base040`
- used `--learning_rate 3e-6`, `--policy_log_std_override -4.0`, `--disable_vec_normalize`, and `--base_action_scale 0.25`
- run: `logs/sb3/recomoproto2trackee_v0/stage0_policy_envelope_fk_base040_from_contractz_lowstd_32k_20260707`

Training health:

- CUDA device active
- stage reward overrides loaded from `stage0_policy_envelope_fk_base040/reset_config.json`
- base action rows `[6,7,8]` scaled by `0.25`
- training remained stable through the bounded run
- final logged `approx_kl=0.00431`, `std=0.0183`

Validation:

- direct `base040` candidate: mean/p95/max EE position error `0.0491 / 0.0761 / 0.1146 m`, mean orientation error `6.22 deg`
- previous routed `base040` baseline: `0.0505 / 0.0749 / 0.1056 m`
- routed mixed gate with new base-required checkpoint: `0.0406 / 0.0680 / 0.0754 m`
- routed `base025` gate with new base-required checkpoint: `0.0437 / 0.0681 / 0.0942 m`
- routed fixed-base `large08` gate: `0.0374 / 0.0659 / 0.0801 m`, unchanged because fixed-base still routes to the promoted default checkpoint

Decision:

Do not promote the `base040` PPO continuation. It improves some mean metrics and keeps fixed-base routing intact, but it fails the Stage B promotion criterion because the `base040` p95/max tail is worse than the routed baseline, and the `base025` max error also worsens versus the current routed reference. Keep the checkpoint as a diagnostic artifact only.

Next useful update:

Target tail-error reduction on base-required trajectories instead of more same-objective PPO. Prefer a bounded DAgger-style rollout relabeling pass or a tail-weighted evaluation/training objective that explicitly penalizes final-step and p95/max EE position error, while preserving the existing router and fixed-base default checkpoint.

## Stage B Tail-Error Probes

Goal:

Reduce `base040` tail error after the first PPO continuation showed better mean tracking but worse p95/max tail behavior.

Tail-weighted BC implementation:

- added optional per-sample weights to `scripts/reinforcement_learning/bc/pretrain_bc.py`
- default behavior remains unchanged with `--sample_weight_mode none`
- new mode `--sample_weight_mode trajectory_tail` ramps sample weights within each trajectory after `--tail_start_fraction`
- metadata `num_waypoints` is used automatically when `--trajectory_length` is omitted

Tail-weighted BC result:

- trained `base040` BC with `--sample_weight_mode trajectory_tail --tail_start_fraction 0.65 --tail_weight 3.0`
- offline weighted validation loss improved strongly: best weighted val MSE `0.000009`
- real Isaac `base040` rollout regressed badly: `0.0624 / 0.1110 / 0.1298 m`
- routed `base025` also regressed: `0.0506 / 0.0904 / 0.1128 m`
- diagnosis: tail-weighted BC overdrives the closed-loop policy, increases arm target lag to about `0.0546 rad`, and worsens tail behavior despite better offline loss

Decision:

Do not promote the tail-weighted BC policy. Keep the weighting option as tooling only; offline BC loss is not a sufficient selector for this stage.

Base action scale sweep:

- tested the current base-required checkpoint on `base040` with runtime base scales `0.50`, `0.75`, and `1.00`
- all larger scales regressed tracking; `0.50` already worsened to `0.0815 / 0.1662 / 0.2240 m`
- diagnosis: the policy is not merely under-scaled. Larger runtime base commands destabilize the closed-loop target/arm coordination.

Decision:

Keep `base_action_scale=0.25` for the current base-required route.

Vertical-shaping probe:

- diagnostic on the current `base040` reference showed persistent z undertracking, with mean signed XYZ error about `[+0.0112, -0.0029, -0.0336] m`
- added `vertical_position_penalty=80.0` to `stage0_policy_envelope_fk_base040/reset_config.json`
- trained one bounded `32k` continuation from the same contract + vertical base-required checkpoint
- run: `logs/sb3/recomoproto2trackee_v0/stage0_policy_envelope_fk_base040_zpenalty80_from_contractz_lowstd_32k_20260707`
- training stayed stable: final logged `approx_kl=0.00603`, `std=0.0183`

Validation:

- direct `base040`: `0.0482 / 0.0773 / 0.1087 m`
- previous routed `base040` baseline: `0.0505 / 0.0749 / 0.1056 m`
- routed `base025`: `0.0431 / 0.0688 / 0.0756 m`
- routed fixed-base `large08`: `0.0374 / 0.0659 / 0.0801 m`, unchanged because fixed-base still routes to the promoted default checkpoint

Decision:

Do not promote the vertical-shaping continuation as the Stage B base-required route yet. It improves `base040` mean error and improves `base025` max error, but `base040` p95/max are still slightly worse than the existing routed baseline. Keep `vertical_position_penalty=80.0` in the `base040` stage config for future Stage B training because the measured z bias is real.

Next useful update:

Move from offline weighting to rollout-state relabeling. The next candidate should collect actual failed rollout observations around the p95/max tail states and relabel them with a dynamics-aware expert or corrective controller. Do not use tail-weighted BC artifacts as a teacher, and do not increase runtime base scale above `0.25` without a fresh gate.

## Stage B Rollout-State DAgger Probe

Goal:

Collect actual failed rollout states from `base040` p95/max tail regions and relabel those visited observations with the generated-stage expert action for the same trajectory/waypoint. This avoids training only on offline teacher states that the closed-loop policy never visits.

Implementation:

- added `scripts/imitation/collect_stage_tail_dagger_dataset.py`
- the collector runs a checkpoint in Isaac, records pre-action policy observations, maps each env's current trajectory filename and waypoint index back to the generated expert dataset row, and writes a compact `.npz`
- selected samples are late/high-error states using `--tail_start_fraction` and `--error_percentile`
- fixed an initial parser bug by replacing regex filename parsing with direct basename/suffix parsing

Collected dataset:

- source checkpoint: `stage0_policy_envelope_fk_mix_contract_zpenalty80_lowstd_resume_32k_20260707/final_model.zip`
- stage: `stage0_policy_envelope_fk_base040`
- output: `data/policy_envelope_fk_base040/tail_dagger_contractz_p80_20260707.npz`
- source rollout samples: `480`
- selected samples: `64`
- selected criterion: `tail_start_fraction=0.65`, `error_percentile=80`
- selected error stats: mean/p95/max `0.0748 / 0.0980 / 0.1043 m`
- expert-vs-policy deltas show the largest corrections around base `vx/vy` and several arm rows, so the dataset is a real off-policy correction set rather than a copy of the existing policy

Naive relabel BC probe:

- combined original `base040` dataset with the 64 selected DAgger states repeated `4x`
- trained grouped BC for 40 epochs
- offline validation loss reached `0.000054`
- real Isaac `base040` rollout regressed to `0.0659 / 0.1048 / 0.1273 m`
- routed `base025` regressed to `0.0480 / 0.0743 / 0.0970 m`
- fixed-base `large08` stayed preserved only because router still uses the default fixed-base checkpoint

Decision:

Do not promote the naive DAgger-BC policy. The collector is useful and should be kept, but directly training a fresh BC policy on original + repeated tail states over-corrects closed-loop behavior. The next useful update should use the collected tail states as a constrained auxiliary loss or PPO regularizer while starting from the current base-required checkpoint, not as a standalone BC replacement.

## Stage B Tail-State Auxiliary PPO Probe

Goal:

Use the collected `base040` tail-state DAgger dataset as a constrained auxiliary loss during PPO, instead of training a standalone BC replacement.

Setup:

- source checkpoint: `stage0_policy_envelope_fk_mix_contract_zpenalty80_lowstd_resume_32k_20260707/final_model.zip`
- stage: `stage0_policy_envelope_fk_base040`
- aux dataset: `data/policy_envelope_fk_base040/tail_dagger_contractz_p80_20260707.npz`
- aux rows: base action rows `[6, 7, 8]`
- aux schedule: `8` supervised steps per PPO rollout, batch `64`, lr `1e-5`, grad clip `0.25`
- PPO: `32k` continuation, `learning_rate=3e-6`, `policy_log_std=-4.0`, `base_action_scale=0.25`
- run: `logs/sb3/recomoproto2trackee_v0/stage0_policy_envelope_fk_base040_taildagger_baseaux_from_contractz_lowstd_32k_20260708`

Training health:

- aux callback activated on grouped `base` head
- usable aux samples: `64/64`
- aux loss stayed stable around `0.0060`
- final PPO `approx_kl=0.00498`
- reachability did not collapse during the bounded run

Validation:

- direct `base040`: `0.0608 / 0.0853 / 0.0965 m`
- routed `base025`: `0.0590 / 0.0858 / 0.0982 m`
- routed fixed-base `large08`: `0.0374 / 0.0659 / 0.0801 m`, unchanged because fixed-base still routes to the promoted default checkpoint

Decision:

Do not promote the base-head auxiliary candidate. It improves neither `base040` nor `base025` mean tracking, even though max error is bounded. The tail-state expert corrections cannot be applied to the base head alone without disrupting arm/base coordination.

Next useful update:

If continuing this line, test a much weaker auxiliary path or a coordinated full action-head auxiliary loss with a very small learning rate and fewer steps. Do not reuse the base-only aux settings above as a promotion candidate.

## Stage B Weak Tail Auxiliary A/B Probe

Goal:

Compare two bounded low-gain uses of the collected `base040` tail-state DAgger dataset:

- weak base-only auxiliary loss on rows `[6, 7, 8]`
- weak coordinated full-action auxiliary loss on rows `[0, 1, 2, 3, 4, 5, 6, 7, 8]`

Common setup:

- source checkpoint: `stage0_policy_envelope_fk_mix_contract_zpenalty80_lowstd_resume_32k_20260707/final_model.zip`
- stage: `stage0_policy_envelope_fk_base040`
- aux dataset: `data/policy_envelope_fk_base040/tail_dagger_contractz_p80_20260707.npz`
- PPO: `32k` continuation, `learning_rate=3e-6`, `policy_log_std=-4.0`, `base_action_scale=0.25`
- gate: 60-step Isaac rollout, 8 envs, `max_trajectories=8`

Weak base-only candidate:

- run: `logs/sb3/recomoproto2trackee_v0/stage0_policy_envelope_fk_base040_taildagger_baseaux_weak_from_contractz_lowstd_32k_20260708`
- aux rows: `[6, 7, 8]`
- aux schedule: `2` supervised steps per PPO rollout, batch `64`, lr `2e-6`, grad clip `0.10`
- training health: aux loss `0.007267 -> 0.006554`, final PPO `approx_kl=0.00463`
- direct `base040`: `0.0503 / 0.0773 / 0.1063 m`
- routed `base025`: `0.0450 / 0.0698 / 0.0923 m`
- routed fixed-base `large08`: `0.0374 / 0.0659 / 0.0801 m`, unchanged because fixed-base still routes to the promoted default checkpoint

Weak full-action candidate:

- run: `logs/sb3/recomoproto2trackee_v0/stage0_policy_envelope_fk_base040_taildagger_fullaux_weak_from_contractz_lowstd_32k_20260708`
- aux rows: `[0, 1, 2, 3, 4, 5, 6, 7, 8]`
- aux schedule: `1` supervised step per PPO rollout, batch `64`, lr `1e-6`, grad clip `0.05`
- training health: aux loss `0.004044 -> 0.004009`, final PPO `approx_kl=0.00503`
- direct `base040`: `0.0495 / 0.0807 / 0.1139 m`
- routed `base025`: `0.0443 / 0.0739 / 0.0993 m`
- routed fixed-base `large08`: `0.0374 / 0.0659 / 0.0801 m`, unchanged because fixed-base still routes to the promoted default checkpoint

Reference metrics:

- existing routed `base040` baseline: `0.0505 / 0.0749 / 0.1056 m`
- existing routed `base025` baseline: `0.0462 / 0.0682 / 0.0835 m`
- existing routed fixed-base `large08`: `0.0374 / 0.0659 / 0.0801 m`

Decision:

Do not promote either weak auxiliary candidate. Both reduce mean error slightly on some routed checks, but both worsen p95/max tail behavior versus the existing routed baseline. The weak full-action variant is better coordinated than the prior strong base-head auxiliary probe, but it still does not solve the terminal tail error.

Lesson:

The collected tail-state dataset is diagnostically useful, but simply adding an auxiliary imitation loss around the tail states is not enough. The next update should stop increasing auxiliary weight and instead target the failure mode directly: terminal/tail recovery, trajectory-specific end-segment correction, or a small corrective controller/teacher that is evaluated on the visited closed-loop tail states before being mixed back into PPO.

## Stage B Tail-Start Recovery Probe

Goal:

Stop judging the terminal failure only through full-start rollouts. Add late-start gates and diagnostics so the final `65%-95%` of `base040` trajectories can be measured directly.

Tooling changes:

- `evaluate_stage_rollout_gate.py` now supports `--random_start_waypoint`, `--start_waypoint_min_fraction`, `--start_waypoint_max_fraction`, `--reset_base_to_trajectory_start`, and `--reset_anchor_target_blend`
- `diagnose_stage_tracking.py` supports the same late-start reset controls and records them in the output JSON
- `BaseAssistConfig` and `train.py` now support `--base_assist_mode target_direction|target_velocity`
- `target_direction` preserves the old base-assist behavior
- `target_velocity` uses the lookahead target displacement as the expert base direction, so it is opt-in and experimental

Tail-start baseline:

- checkpoint: `stage0_policy_envelope_fk_mix_contract_zpenalty80_lowstd_resume_32k_20260707/final_model.zip`
- gate: `base040`, random start `0.65-0.95`, no base-start anchor, `base_action_scale=0.25`
- result: `0.0904 / 0.1609 / 0.1940 m`
- this is far worse than the full-start `base040` gate, so the tail segment is a real standalone weakness

Diagnosis:

- output: `evaluation_results/stage_diagnostics/stage0_policy_envelope_fk_base040_tailstart_baseline_diag_20260708.json`
- target XY motion mean: `0.1178 m`
- base XY motion mean: `0.0155 m`
- base action abs mean: `0.0277`
- arm target lag is modest: mean/p95/max `0.0305 / 0.0660 / 0.0870 rad`
- average XYZ error is biased mostly in `y/z`: `[-0.0007, -0.0305, -0.0376] m`

Interpretation:

The tail-start failure is not primarily an arm-label imitation problem. The base under-moves while the target continues moving in XY, leaving the arm to absorb too much residual error.

Candidate A: tail-reset PPO only

- run: `logs/sb3/recomoproto2trackee_v0/stage0_policy_envelope_fk_base040_tailreset_65_95_lowstd_32k_20260708`
- setup: random start `0.65-0.95`, no auxiliary loss, no base assist
- training health: reachability stayed clean; final PPO `approx_kl=0.00503`
- late-start gate: `0.0905 / 0.1609 / 0.1951 m`
- decision: do not promote; changing reset distribution alone did not create enough learning signal at this budget

Candidate B: existing target-direction base assist

- run: `logs/sb3/recomoproto2trackee_v0/stage0_policy_envelope_fk_base040_tailreset_baseassist_lk5_32k_20260708`
- setup: random start `0.65-0.95`, base assist `target_direction`, blend `0.70->0.00`, lookahead `5`, imitation weight `30`
- training health: stable reachability, but expert commands chase the target center rather than preserving the intended base-target offset
- late-start gate: `0.0898 / 0.1604 / 0.1940 m`
- decision: do not promote; effectively unchanged from baseline

Candidate C: target-velocity base assist

- run: `logs/sb3/recomoproto2trackee_v0/stage0_policy_envelope_fk_base040_tailreset_velassist_lk5_32k_20260708`
- setup: random start `0.65-0.95`, base assist `target_velocity`, blend `0.70->0.00`, lookahead `5`, activation/full-speed displacement `0.005-0.030 m`
- training health: failed as a teacher; reachability dropped to `10/64` at step 100 and only recovered to `38/64` by step 500
- late-start gate: `0.0902 / 0.1621 / 0.1951 m`
- decision: do not promote; raw target-velocity following overdrives the base and breaks the base-target working offset

Decision:

Do not promote any tail-start recovery candidate from this pass. Keep the late-start gate/diagnostic tooling and the opt-in `target_velocity` assist mode because they are useful for controlled experiments, but do not use the generated checkpoints as routed policies.

Next useful update:

Implement an offset-preserving base-follow teacher rather than either existing target-direction assist or raw target-velocity assist. The expert should preserve the generated FK base-target offset while following target motion, then be tested first with the late-start diagnostic before any PPO continuation. A good smoke target is to raise base XY motion toward the target XY motion without pushing base-target distance outside the `base040` working band.

## Stage B Offset-Follow Teacher Probe

Goal:

Test an offset-preserving base-follow teacher that commands the base toward:

`desired_base_xy = target_xy - [reset_base_x_offset, reset_base_y_offset]`

This differs from the failed `target_direction` and `target_velocity` teachers:

- `target_direction` chases the target center and does not preserve the generated base-target offset
- `target_velocity` follows target displacement directly and overdrives the base
- `target_offset_follow` follows target motion while preserving the FK-generated base-target working offset

Tooling changes:

- added `target_offset_follow` to `BaseAssistConfig.mode`
- added `target_offset_follow` to `--base_assist_mode`
- extended `diagnose_stage_tracking.py` so base assist can be enabled during no-training diagnostics
- diagnostic output now records `base_target_distance_m`, `base_assist_coeff_mean`, and `base_assist_active_pct`

No-training teacher smoke:

- checkpoint: `stage0_policy_envelope_fk_mix_contract_zpenalty80_lowstd_resume_32k_20260707/final_model.zip`
- diagnostic: `base040`, random start `0.65-0.95`, no base-start anchor, `base_action_scale=0.25`
- assist: `target_offset_follow`, blend `0.70`, activation/full-speed `0.010-0.080 m`, max action `0.60`
- output: `evaluation_results/stage_diagnostics/stage0_policy_envelope_fk_base040_tailstart_offsetassist_diag_20260708.json`
- EE position error improved from baseline `0.0904 / 0.1609 / 0.1940 m` to `0.0555 / 0.0908 / 0.2045 m`
- base XY motion improved from `0.0155 m` to `0.1240 m`, matching target XY motion `0.1178 m`
- base-target distance stayed in-band: mean/p95/max `0.8407 / 0.8536 / 0.9811 m`
- assist was active `78.1%` of samples

Interpretation:

The teacher is directionally correct. It fixes the core base-under-following behavior without breaking the base-target working distance. The one max-error outlier means it is not perfect, but it is much better than the previous teachers as an executable controller.

PPO transfer probe:

- run: `logs/sb3/recomoproto2trackee_v0/stage0_policy_envelope_fk_base040_tailreset_offsetassist_32k_20260708`
- setup: random start `0.65-0.95`, `target_offset_follow`, blend `0.70->0.00`, activation/full-speed `0.010-0.080 m`, max action `0.60`, imitation weight `30`
- training health: reachability stayed `64/64`; final PPO `approx_kl=0.00333`
- unassisted late-start gate: `0.0896 / 0.1603 / 0.1936 m`

Decision:

Do not promote the PPO checkpoint. The offset-follow teacher works when executed, but the current PPO reward-imitation path did not transfer the teacher into the raw policy at `32k` timesteps.

Next useful update:

Stop trying to transfer this teacher through weak PPO reward shaping alone. Build a supervised offset-follow teacher dataset from visited late-start states, then train the base action head with a direct supervised/auxiliary loss while preserving the arm/gimbal heads. The acceptance test should be the unassisted late-start gate, not the assisted diagnostic.

## Long Cinematic Whole-Trajectory Probe - 2026-07-08

Goal:

Answer the visual concern that the rendered validation clips only showed small movements and did not prove whole-trajectory tracking.

Stage:

- added `trajectoryToLearn/stage_long_cinematic_probe_20260708/manifest.txt`
- selected 8 long cinematic trajectories from `trajectoryToLearn/world_json/cinematic_db`
- all selected trajectories are at least 12s long; the first 4-way gate used 4 trajectories with 240 control steps at 20 Hz
- copied the FK/base040 reset contract into `trajectoryToLearn/stage_long_cinematic_probe_20260708/reset_config.json`

Critical reset lesson:

The first run was invalid because the new probe stage had no `reset_config.json`, so the evaluator fell back to the historical default reset offset `[0.4415, 0.2405]`. That started the camera about `1.30 m` away from the target before the policy acted. With the FK reset config, initial error dropped to `0.136 m`, so future long-trajectory probes must include the FK reset config or the result is not meaningful.

Corrected gate results:

| Mode | Base scale | Initial mean | EE mean / p50 / p95 | Final mean | Dones | Verdict |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| source | `0.25` | `0.1361 m` | `0.7046 / 0.4788 / 1.9187 m` | `0.6351 m` | `3` | starts correctly, degrades over 12s |
| routed W4 | `0.25` | `0.1361 m` | `0.6236 / 0.5101 / 1.6116 m` | `0.6575 m` | `5` | better mean/p95, not a solved long tracker |
| source | `0.50` | `0.1361 m` | `0.9483 / 0.9988 / 1.7472 m` | `0.3133 m` | `4` | more base motion but worse average tracking |
| source | `1.00` | `0.1361 m` | `0.8363 / 0.7763 / 1.8507 m` | `0.2685 m` | `52` | unsafe/runaway, do not use as a shortcut |

Rendered evidence:

- remote: `evaluation_results/videos_rendered/long_cinematic_probe_20260708/orbit_left_078_routed_w4_fkreset_240/rl-video-step-0.mp4`
- local copy: `/Users/yanbo/Downloads/cinebotRL_long_probe_20260708/orbit_left_078_routed_w4_fkreset_240.mp4`
- render metadata: 240 frames, 20 FPS, 12.0s, selected `orbit_left_078.json`, `recovery_route_fraction=0.3542`

Interpretation:

The policy is not yet a whole-cinematic-trajectory tracker. The corrected reset proves the robot can start on the trajectory, but the current curriculum only solves short FK envelope and base-required recovery cases. Increasing `base_action_scale` is not the cure: `0.5` and `1.0` both destabilize average tracking, and `1.0` creates many terminations.

Next useful update:

Build the next curriculum around long-horizon base/EE coordination, not around another scalar base-speed change. Use this stage as the acceptance gate, but train on an easier ladder first: 2-3s long-motion snippets, then 5-6s, then 12s cinematic paths. Keep `base_action_scale=0.25` as the safety contract until a trained policy beats the corrected `0.25` baseline on mean, p95, final error, and done count.

## Correction: Use The 79 Live ARCore/GIK Trajectories, Not The 1000 Generated Cinematic Set - 2026-07-08

User correction:

Do not train the next curriculum from the initialized/generated 1000 cinematic trajectories. Use the 79 live ARCore/GIK trajectories instead.

Data audit:

- `data/gik_offline_teachers_20260701_142322/no_obstacle_all79_20260701_123049/work_json` contains 79 JSON files, but they are `first80` trimmed variants.
- The `first80` JSONs are all too short for the current curriculum rule: duration min/p50/mean/max is `1.42 / 1.50 / 1.50 / 1.58 s`, so `dur>=5s` count is `0/79`.
- The valid learning material is the accepted GIK teacher export under `data/gik_offline_teachers_20260701_142322/accepted_npz/manifest.json`.
- Filtering that manifest to `teacher_metadata.scenario == no_obstacle` gives the intended 79 accepted live trajectories: duration min/p50/mean/max is `5.98 / 13.76 / 17.65 / 58.65 s`, with `2803` action samples.

Artifacts created on `.98`:

- filtered manifest: `data/gik_offline_teachers_20260701_142322/accepted_npz/manifest_no_obstacle79.json`
- 79-only base dataset: `data/gik_offline_teacher_obs/obs_dataset_no_obstacle79_base_only.npz`
- BC smoke policy: `logs/bc/gik_no_obstacle79_base_smoke_20260708/bc_policy.zip`
- in-distribution eval: `evaluation_results/bc/gik_no_obstacle79_base_in_distribution_20260708.json`

79-only dataset check:

- trajectories: `79`
- samples: `2803`
- observation dimension: `85`
- action dimension: `9`
- base label mask mean: `[1.0, 1.0, 1.0]`
- arm/gimbal labels are masked out: `[0, 0, 0, 0, 0, 0]`

BC smoke result:

- training split: `2523 / 280`
- best validation MSE: `0.001325`
- in-distribution base RMSE/MAE/max: `0.04157 / 0.02430 / 0.56226`
- per-base RMSE: `vx=0.05371`, `vy=0.03948`, `wz=0.02724`

Decision:

The generated 1000 cinematic set is now diagnostic/evaluation-only unless explicitly requested. The next learning step should use the 79-only accepted GIK/ARCore teacher dataset above as the nominal source. Keep the `>=5s` rejection rule by filtering at the manifest/teacher level; do not use the `first80` work JSONs as training trajectories.

Next useful update:

Use the 79-only base teacher as a supervised base-head auxiliary/warm-start for the real RL policy, then evaluate on a 79-derived Isaac stage or on held-out teacher observations before any broad PPO run. Do not mix in one-obstacle or generated cinematic data until the nominal 79-case transfer is measured cleanly.

## 79-Only Base Auxiliary Transfer Smoke - 2026-07-08

Experiment:

- source checkpoint: `logs/sb3/recomoproto2trackee_v0/stage1_recovery_yaw_assist_gate_20260702_1508/final_model.zip`
- transfer run: `logs/sb3/recomoproto2trackee_v0/yaw_assist_noobs79_baseaux_lowp_8k_20260708`
- auxiliary dataset: `data/gik_offline_teacher_obs/obs_dataset_no_obstacle79_base_only_yawassist_vecnorm.npz`
- dataset normalization: observations were transformed with the source checkpoint sibling `vec_normalize.pkl`
- policy update: low-pressure PPO continuation, `8192` extra requested steps, `learning_rate=1e-5`, `target_kl=0.01`, one base-aux gradient step per rollout, base action rows `[6,7,8]`

Training signal:

- base auxiliary loss stayed high, around `0.54-0.59`
- rollout reachability stayed poor, often `75-100%` unreachable
- base assist was disabled during policy rollout; this was testing whether the policy itself absorbed the 79-only base labels

Raw-policy evaluation:

- output: `evaluation_results/recovery_candidate/yaw_assist_noobs79_baseaux_lowp_8k_20260708/recovery_eval_raw-policy_20260708_145554.json`
- stage: `stage1_recovery`, `20` selected trajectories, `64` completed episodes
- mean EE position error: `1.1224 m`
- p95 EE position error: `1.6544 m`
- mean EE orientation error: `138.73 deg`
- unreachable-zone mean: `96.72%`
- workspace hard-exceed mean: `30.66%`
- obstacle collision mean: `0.0%`

Comparison against the current yaw-assist baseline:

| candidate | mean pos err | mean ori err | unreachable | hard exceed | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| yaw_assist_baseline | `1.1442 m` | `132.17 deg` | `47.42%` | `3.73%` | keep as baseline |
| 79-only baseaux low-pressure transfer | `1.1224 m` | `138.73 deg` | `96.72%` | `30.66%` | reject |

Decision:

Do not continue this exact transfer path. The position mean is not enough to justify the run because the policy becomes much less reachable and much less workspace-safe than the source baseline. The 79 accepted GIK/ARCore trajectories remain useful learning material, but not as a direct base-only auxiliary injected into the current `stage1_recovery` distribution.

Next useful update:

Build a 79-derived Isaac replay/gate whose reset distribution, observation normalization, and trajectory source match the accepted GIK/ARCore teacher data, then test base-head learning in that nominal distribution before trying to transfer into `stage1_recovery`. If we need recovery behavior, generate teacher labels for the recovery reset distribution itself; do not assume nominal 79-only base labels are valid under random recovery starts.

## 79-Derived Nominal Isaac Stage - 2026-07-08

Implementation:

- added exporter: `scripts/imitation/export_gik_npz_stage.py`
- generated stage: `trajectoryToLearn/stage_gik_no_obstacle79_nominal`
- source manifest: `data/gik_offline_teachers_20260701_142322/accepted_npz/manifest_no_obstacle79.json`
- generated files: `79` trajectory JSONs plus `manifest.txt`, `reset_config.json`, and `export_summary.json`
- generated stage size: about `21M`

Exporter behavior:

- reads accepted GIK/ARCore NPZ teacher targets, not the `first80` work JSONs and not the generated cinematic corpus
- filters by `teacher_metadata.scenario == no_obstacle`
- keeps the `duration_s >= 5.0` rule at the manifest level
- resamples sparse NPZ target poses to the manifest duration at `0.1s` waypoint dt, because the raw NPZ rows alone would make many clips look shorter than 5s to Isaac
- converts stored `target_quat_wxyz` into JSON `xyzw`, matching `MultiTrajectoryLoader` expectations
- writes reset offsets from the teacher frame: `reset_base_x_offset=-0.000003`, `reset_base_y_offset=0.036461`

Validation:

- CPU loader smoke loaded all `79` trajectories after the `>=5s` filter
- loader lengths: min/mean/max `61 / 178.1 / 588` waypoints
- loader durations: min/mean/max `6.10 / 17.81 / 58.80 s`
- Isaac smoke command used `--trajectory_stage stage_gik_no_obstacle79_nominal --reset_base_to_trajectory_start --max_trajectories 4 --num_envs 4 --steps 40`
- smoke output: `evaluation_results/stage_gik_no_obstacle79_nominal/stage_loader_smoke_yaw_assist_20260708.json`

Important boundary:

This stage is a valid 79-live-trajectory routing/evaluation stage, not a solved policy result. The short smoke used the old yaw-assist policy and still had high tracking error: initial mean EE position error `1.433 m`, rollout mean `1.571 m`, p95 `1.921 m`, and `6` dones over `160` samples. The main reason is that the current Isaac reset can place the base using the GIK frame offset, but it does not reset the arm/gimbal to the per-trajectory GIK teacher joint state.

Next useful update:

Use this stage to build the nominal 79 curriculum, but do not call it solved until we add either a teacher-state reset path or a BC/offline warm-start path that initializes the arm/gimbal consistently with the GIK teacher state. For RL-only training on this stage, start with short bounded runs and judge them against the stage gate metrics above rather than against `stage1_recovery` cinematic metrics.

## 79 Teacher-State Reset And Masked BC Smoke - 2026-07-08

Implementation:

- `TrajectoryConfig.reset_arm_to_trajectory_metadata` now gates optional arm reset from per-trajectory JSON metadata.
- `MultiTrajectoryLoader` now preserves each trajectory JSON's `metadata` in `current_trajectory_metadata`.
- `export_gik_npz_stage.py` now writes `metadata.initial_arm_joint_pos` and enables `reset_arm_to_trajectory_metadata` in the generated stage `reset_config.json`.
- The train/stage-eval/diagnostic helpers now pass the stage reset flag into `TrajectoryConfig`.

Teacher-reset gate:

- command family: `evaluate_stage_rollout_gate.py --trajectory_stage stage_gik_no_obstacle79_nominal --reset_base_to_trajectory_start`
- old stage smoke before teacher arm reset: initial mean EE position error `1.433 m`
- 2-step smoke after teacher arm reset: initial mean EE position error `0.00316 m`, mean rollout EE error `0.0533 m`, `0` dones
- 40-step old yaw-assist policy after teacher arm reset: mean EE error `0.6373 m`, p95 `1.6802 m`, `4` dones
- 40-step masked-BC actor after teacher arm reset: mean EE error `0.1347 m`, p95 `0.3775 m`, `0` dones

Masked BC smoke:

- dataset: `data/gik_offline_teacher_obs/obs_dataset_no_obstacle79_full_masked.npz`
- policy: `logs/bc/gik_no_obstacle79_masked9_smoke_20260708/bc_policy.zip`
- offline eval: `evaluation_results/bc/gik_no_obstacle79_masked9_in_distribution_20260708.json`
- stage eval: `evaluation_results/stage_gik_no_obstacle79_nominal/stage_loader_teacherreset_40step_masked9bc_20260708.json`
- BC validation MSE after 10 epochs: `0.005042`
- offline masked RMSE/MAE/max: `0.0727 / 0.0449 / 0.7118`

Important boundary:

This is a sim-stage nominal tracking result, not a deployable DJI gimbal policy. The gimbal/action-contract issue still applies: row 3 has sparse/weak fit (`rmse=0.1677`, `956` labels), row 1 is only `514` valid labels, and the current action contract is still `sim_6joint_gimbal_v1`. Treat the masked-BC actor as a useful Isaac curriculum/warm-start artifact, not as proof that full 9D deployment semantics are solved.

Next useful update:

Use the teacher-reset stage and masked-BC actor as the nominal Stage-79 gate. The next training run should warm-start PPO from this BC actor on `stage_gik_no_obstacle79_nominal`, with short bounded steps first, raw-observation BC warm start, and no generated cinematic trajectories mixed in. Only after this gate stays stable should we add obstacle curricula or transfer back toward recovery/cinematic stages.

## 79 Masked-BC PPO Continuation Attempt - 2026-07-08

Implementation fix:

- `train.py` now accepts `stage_gik_no_obstacle79_nominal` as a named `--trajectory_stage`.
- when no `--checkpoint` is supplied, `train.py` now inspects `--pretrained_policy` observation space and passes that expected dimension into the IsaacLab-to-SB3 wrapper.
- this lets raw 84D env observations append the trajectory-progress column to match the 85D BC actor, the same way the stage gate already worked.

Rejected run:

- run: `logs/sb3/recomoproto2trackee_v0/stage_gik_noobs79_teacherreset_masked9bc_8k2_20260708`
- command intent: PPO continuation from `logs/bc/gik_no_obstacle79_masked9_smoke_20260708/bc_policy.zip` on `stage_gik_no_obstacle79_nominal`
- correct warm-start was confirmed: actor copied successfully and wrapper logged `84 -> 85`
- the run was manually stopped at about `4096` timesteps because it was degrading the BC prior
- rollout symptoms: by `2048-4096` steps, reachability was usually `7-8/8` unreachable and base-target distance was around `0.88-1.37 m`

Decision:

Do not continue this PPO configuration. The BC actor is currently better than the PPO continuation. The likely issue is that ordinary PPO rewards/critic initialization are too blunt for this nominal imitation stage, especially with sparse/weak arm-gimbal labels and large value losses.

Next useful update:

Keep the masked-BC actor as the current nominal Stage-79 policy artifact. If we do RL after BC, use a more conservative imitation-preserving update: lower learning rate, fewer PPO epochs, stronger KL/behavior regularization or an auxiliary BC loss on the same 79 dataset, and promote only if the 40-step teacher-reset gate beats the raw BC actor's `0.1347 m` mean / `0.3775 m` p95 / `0` dones baseline.

## 79 Masked-BC Aux-PPO Smoke - 2026-07-08

Attempted a more conservative PPO continuation after the plain PPO rejection:

- run: `logs/sb3/recomoproto2trackee_v0/stage_gik_noobs79_teacherreset_masked9bc_aux8k_20260708`
- warm start: `logs/bc/gik_no_obstacle79_masked9_smoke_20260708/bc_policy.zip`
- stage: `stage_gik_no_obstacle79_nominal`, 79 accepted live GIK/ARCore trajectories only
- PPO limits: `lr=1e-5`, `n_epochs=1`, `clip_range=0.03`, `target_kl=0.003`, `ent_coef=0`, `std=exp(-3)`
- aux BC stabilizer: `data/gik_offline_teacher_obs/obs_dataset_no_obstacle79_full_masked.npz`, action rows `0..8`, 8 supervised minibatches per rollout

Observed result before manual stop at `4096` timesteps:

- aux BC loss stayed around `0.0051-0.0055`, so the supervised head correction was active
- KL stayed tiny (`~5e-5` to `3e-4`), so this was not a large policy-update explosion
- rollout behavior still degraded: reachability was usually `7-8/8` unreachable and base-target distance ranged roughly `0.96-1.47 m`
- value loss remained very large (`~1.5e7-2.5e7`), suggesting the current PPO reward/value setup is not a good continuation target for this narrow nominal imitation stage

Decision:

Reject Aux-PPO continuation for the nominal 79 stage. The issue is not simply PPO learning rate, clipping, or missing BC regularization. The current RL objective is pulling against the BC behavior in ways that hurt trajectory tracking almost immediately.

Next useful update:

Do not run more nominal 79 PPO continuations until the RL objective is reworked. The safer next path is either (1) improve the 79 imitation policy directly with better labels/architecture/evaluation, or (2) build a separate teacher-forced or DAgger-style rollout collector where the policy is corrected against the live 79 teacher state distribution before any reward-only PPO is attempted. Obstacle curricula should wait until the no-obstacle 79 policy can track full trajectories, not just short local motion.

## 79 Weighted Masked-BC Candidate - 2026-07-08

Trained a stronger offline imitation candidate instead of continuing PPO:

- policy: `logs/bc/gik_no_obstacle79_masked9_weighted80_20260708/bc_policy.zip`
- dataset: `data/gik_offline_teacher_obs/obs_dataset_no_obstacle79_full_masked.npz`
- training: 80 epochs, `lr=2e-4`, masked action loss
- action weights: `1,2,1.25,2,1.25,1.25,1.5,1.5,1.25`
- offline validation MSE: `0.000171`
- offline full-dataset RMSE: `0.01243`, improved from the earlier 10-epoch masked-BC smoke RMSE `0.07273`

40-step teacher-reset Isaac gate on the same 4-trajectory protocol:

- output: `evaluation_results/stage_gik_no_obstacle79_nominal/stage_loader_teacherreset_40step_masked9bc_weighted80_20260708.json`
- initial EE error: `0.00316 m`
- mean EE error: `0.14694 m` versus previous BC10 `0.13472 m`
- p50 EE error: `0.13427 m` versus previous BC10 `0.08459 m`
- p95 EE error: `0.29419 m` versus previous BC10 `0.37746 m`
- max EE error: `0.36248 m` versus previous BC10 `0.44162 m`
- final EE error: `0.29756 m` versus previous BC10 `0.36056 m`
- orientation mean: `77.34 deg` versus previous BC10 `84.09 deg`
- dones: `0`

Decision:

Keep this as a candidate, not a clean replacement. It is better on tail/final/worst-case metrics and offline imitation, but worse on mean and median short-rollout tracking. This confirms that lower offline action MSE alone is not enough; the gate must measure closed-loop trajectory tracking.

Next useful update:

Add a longer rollout gate over more of the 79 trajectories and inspect per-trajectory failure cases. If the weighted BC candidate mainly improves tails but hurts early tracking, train/evaluate a hybrid BC objective rather than moving to PPO: preserve BC10-style early behavior while adding tail/final weighting only where the old policy drifts.

## 79 Sequential All-Trajectory Gate - 2026-07-08

Implementation update:

- `evaluate_stage_rollout_gate.py` now supports `--assign_loaded_trajectories_once`.
- This replaces reset-time random sampling-with-replacement with deterministic sequential assignment of loaded trajectory files to envs.
- With `--num_envs 79 --max_trajectories 79`, one rollout now evaluates each accepted live GIK/ARCore trajectory exactly once.
- The evaluator also writes a `per_env` block with trajectory file, length, waypoint indices, per-env position/orientation error, reward, and done counts.

Gate protocol:

- stage: `stage_gik_no_obstacle79_nominal`
- envs: `79`
- steps: `60`
- reset: `--reset_base_to_trajectory_start`
- normalization: disabled, raw-observation BC policies
- output summary: `evaluation_results/stage_gik_no_obstacle79_nominal/all79_seq60_bc_policy_comparison_20260708.json`

Aggregate results:

| policy | mean m | p50 m | p95 m | max m | final mean m | ori mean deg | reward mean | dones |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BC10 smoke | 0.24184 | 0.21236 | 0.54709 | 0.66708 | 0.49540 | 68.86 | -35.78 | 0 |
| BC80 weighted | 0.26580 | 0.16707 | 0.77909 | 1.11905 | 0.76414 | 74.93 | -82.58 | 0 |
| BC80 unweighted | 0.24354 | 0.18846 | 0.64923 | 0.81585 | 0.59110 | 65.67 | -56.21 | 0 |

Per-trajectory winner counts:

| metric | BC10 | BC80 weighted | BC80 unweighted |
|---|---:|---:|---:|
| mean position error | 43 | 9 | 27 |
| p95 position error | 56 | 2 | 21 |
| final position error | 63 | 2 | 14 |
| mean orientation error | 26 | 0 | 53 |

Decision:

Do not promote either BC80 candidate as the nominal policy. BC80 weighted is rejected. BC80 unweighted is a useful diagnostic: it improves orientation and p50 position error, but it regresses p95/final tracking. The current best position-tracking baseline remains BC10 smoke for the no-obstacle 79 stage.

Next useful update:

Use this all-79 deterministic gate as the promotion gate. For policy improvement, do not use global action weighting. Train a hybrid/imitation candidate that keeps BC10-style position tracking while borrowing the BC80-unweighted orientation improvement, then only promote if it beats BC10 on p95/final position without losing the orientation gain.

## 79 Row-Blend Hybrid Diagnostics - 2026-07-08

Implementation update:

- `evaluate_stage_rollout_gate.py` now supports `--row_blend_checkpoint`, `--row_blend_action_indices`, and `--row_blend_weight`.
- This is evaluator-only. It does not create a new policy artifact.
- Purpose: test whether rows from one BC policy can improve another before spending time on a new training objective.

Hybrid tests:

- primary: BC10 smoke, `logs/bc/gik_no_obstacle79_masked9_smoke_20260708/bc_policy.zip`
- secondary: BC80 unweighted, `logs/bc/gik_no_obstacle79_masked9_unweighted80_20260708/bc_policy.zip`
- protocol: same all-79 sequential 60-step gate
- output summary: `evaluation_results/stage_gik_no_obstacle79_nominal/all79_seq60_bc_policy_and_hybrid_comparison_20260708.json`

Aggregate results:

| policy | mean m | p50 m | p95 m | max m | final mean m | ori mean deg | reward mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| BC10 smoke | 0.24184 | 0.21236 | 0.54709 | 0.66708 | 0.49540 | 68.86 | -35.78 |
| BC80 unweighted | 0.24354 | 0.18846 | 0.64923 | 0.81585 | 0.59110 | 65.67 | -56.21 |
| hybrid rows 3,4,5 | 0.27059 | 0.24674 | 0.60437 | 0.73971 | 0.54937 | 65.62 | -40.39 |
| hybrid rows 0..5 | 0.24725 | 0.22485 | 0.54145 | 0.66857 | 0.46794 | 67.91 | -36.40 |

Decision:

The row `3,4,5` hybrid is rejected: it improves orientation but damages position tracking too much. The row `0..5` hybrid is a useful diagnostic candidate: keeping BC10 base rows while using BC80-unweighted arm/gimbal rows slightly improves p95 and final position error and modestly improves orientation, but it worsens mean and p50 position error.

Next useful update:

Do not promote the hybrid directly yet. Train a real hybrid/distillation candidate with BC10 as the anchor for position/base behavior and BC80-unweighted as a weak orientation/arm prior, then pass it through the all-79 sequential gate. Promotion target: beat BC10 on p95/final position, keep max error no worse, and improve orientation without regressing mean position materially.
