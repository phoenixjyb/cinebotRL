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
