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
