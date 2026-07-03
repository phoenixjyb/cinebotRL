# Stage1 Recovery Aux Policy Loop Log - 2026-07-03

This log records the current `RecomoProto2TrackEE-v0` stage1 recovery policy loop on the `.98` machine.

Remote workspace:

```text
ssh -p 2222 yanbo@192.168.100.98
repo: /mnt/g/wSpace/cinebotRL
branch: win-recomoPro1
python: /mnt/g/isaaclab_venv/Scripts/python.exe -X utf8
```

## Goal

Reduce raw-policy base-target unreachable-zone time for `stage1_recovery` while preserving obstacle safety.

Target for a deployable policy:

```text
unreachable_zone_pct mean < 15%
obstacle_unsafe_pct = 0
obstacle_collision_pct = 0
```

Current result: not achieved. Best mean so far is `25.1410%`, still above the target and with worse tail risk than the prior best.

## Source Fixes Landed

| Commit | Change | Reason |
| --- | --- | --- |
| `615ef60` | Added masked action dataset merge utility. | Needed to combine base-assist/distillation datasets without corrupting action masks. |
| `2bfb3b5` | Added masked action auxiliary PPO callback. | Let PPO continue while applying supervised updates only to selected action rows. |
| `74349a1` | Moved aux updates to rollout start. | Avoided mutating the policy between rollout collection and PPO train, which inflated KL and invalidated PPO data. |
| `571c61a` | Added sibling `vec_normalize.pkl` fallback when resuming from `final_model.zip`. | Fixed invalid resume where `final_model.pkl` was missing and training silently continued without normalization. |
| `1097cae` | Added `sample_weight` support in collection, merge, and aux training. | Allowed hard-state weighted datasets instead of only uniform sample expansion. |
| `ded87c0` | Added `--base_assist_aux_sample_weight_power`. | Allows softening hard-state weighting, e.g. `sample_weight^0.5`, without rewriting datasets. |

Repo status after these commits: clean and pushed to `origin/win-recomoPro1`.

## Policy Rounds

### Baseline: raw original 524k

Artifact:

```text
evaluation_results/recovery_candidate_startfrac045to070slow_524k/recovery_eval_raw-policy_20260703_143113.json
```

Result:

```text
unreachable_zone_pct mean = 52.4358%
p95 = 70.3125%
obstacle unsafe/collision = 0
```

Decision: failed. It established the starting point.

### Round 1: DAgger/base-head distillation

Checkpoint:

```text
logs/sb3/recomoproto2trackee_v0/stage1_recovery_basehead_dagger_round2_from_cont40k_20260703/basehead_dagger_round2_e200.zip
```

VecNormalize:

```text
logs/sb3/recomoproto2trackee_v0/stage1_recovery_basehead_dagger_round2_from_cont40k_20260703/basehead_dagger_round2_e200.pkl
```

Eval:

```text
evaluation_results/recovery_candidate_basehead_dagger_round2_from_cont40k_smoke64/recovery_eval_raw-policy_20260703_153903.json
```

Result:

```text
unreachable_zone_pct mean = 28.0898%
p95 = 43.7500%
max = 46.8750%
workspace_hard_exceed_pct mean = 0.9829%
obstacle unsafe/collision = 0
```

Decision: useful improvement, not deployable.

Lesson: base `vx/vy` imitation helps. This justified further base-only learning.

### Round 2: PPO continuation with runtime base assist

Eval:

```text
evaluation_results/recovery_candidate_dagger_round2_cont_assist030_40k_smoke64/recovery_eval_raw-policy_20260703_154313.json
```

Result:

```text
unreachable_zone_pct mean = 30.5216%
p95 = 54.6875%
obstacle_unsafe_pct mean = 0.0509%
```

Decision: reject.

Lesson: do not evaluate deployment readiness with runtime base assist enabled. It can make the raw policy worse and can introduce obstacle unsafe time.

### Round 3: yaw-inclusive base-head distillation

Dataset:

```text
data/base_assist_distill/stage1_recovery_dagger_round2_plus_yaw_merged_vxvywz.npz
```

Checkpoint:

```text
logs/sb3/recomoproto2trackee_v0/stage1_recovery_basehead_dagger_round2_yaw_from_round2_20260703/basehead_dagger_round2_yaw_e200.zip
```

Eval:

```text
evaluation_results/recovery_candidate_basehead_dagger_round2_yaw_from_round2_smoke64/recovery_eval_raw-policy_20260703_155158.json
```

Result:

```text
unreachable_zone_pct mean = 33.2463%
p95 = 51.5625%
workspace_hard_exceed_pct mean = 2.2132%
obstacle unsafe/collision = 0
```

Decision: reject.

Lesson: do not include yaw row `8` until there is separate evidence that yaw labels are valid. Current base learning should stay on rows `[6, 7]`.

### Round 4: first valid aux-loss PPO, vx/vy only

Dataset:

```text
data/base_assist_distill/stage1_recovery_dagger_round2_merged_vxvy.npz
```

Run:

```text
logs/sb3/recomoproto2trackee_v0/stage1_recovery_auxloss_vxvy_noassist_hookstart_40k_seed20260703_20260703_1606
```

Eval:

```text
evaluation_results/recovery_candidate_auxloss_vxvy_noassist_hookstart_40k_smoke64/recovery_eval_raw-policy_20260703_160142.json
```

Result:

```text
unreachable_zone_pct mean = 25.6383%
p95 = 35.9375%
max = 40.6250%
base_target_dist_mean = 0.5502
workspace_hard_exceed_pct mean = 1.5351%
obstacle unsafe/collision = 0
```

Decision: best risk-balanced policy so far. Not deployable.

Lesson: aux update at rollout start is valid and can improve the mean, but it is still far from the `<15%` target.

### Round 5: collect round3 dataset from current best and train with full aux pressure

New dataset:

```text
data/base_assist_distill/stage1_recovery_auxloss_round3_normobs_vxvy_256x128.npz
```

Merged dataset:

```text
data/base_assist_distill/stage1_recovery_auxloss_round3_merged_vxvy.npz
```

Run:

```text
logs/sb3/recomoproto2trackee_v0/stage1_recovery_auxloss_round3_noassist_40k_seed20260703_20260703_1608
```

Eval:

```text
evaluation_results/recovery_candidate_auxloss_round3_noassist_40k_smoke64/recovery_eval_raw-policy_20260703_160923.json
```

Result:

```text
unreachable_zone_pct mean = 28.4422%
p95 = 39.2187%
max = 48.4375%
obstacle unsafe/collision = 0
```

Decision: reject.

Lesson: adding more uniformly sampled aux data diluted or destabilized the useful signal. More samples alone are not a strategy.

### Round 6: lower-pressure round3 aux

Run:

```text
logs/sb3/recomoproto2trackee_v0/stage1_recovery_auxloss_round3_lowpressure_noassist_40k_seed20260703_20260703_1610
```

Eval:

```text
evaluation_results/recovery_candidate_auxloss_round3_lowpressure_noassist_40k_smoke64/recovery_eval_raw-policy_20260703_161214.json
```

Result:

```text
unreachable_zone_pct mean = 30.1574%
p95 = 42.1875%
max = 45.3125%
obstacle unsafe/collision = 0
```

Decision: reject.

Lesson: simply reducing aux LR/steps did not fix the round3 regression.

### Round 7: weighted hard-state aux, power 1.0

Weighted dataset:

```text
data/base_assist_distill/stage1_recovery_auxloss_round4_weighted_normobs_vxvy_256x128.npz
```

Dataset stats:

```text
rows = 25,825
valid rows = action 6/7 only
sample_weight min = 1.0
sample_weight mean = 1.3919
sample_weight max = 4.0
p90 = 2.6797
```

Merged weighted dataset:

```text
data/base_assist_distill/stage1_recovery_auxloss_round4_merged_weighted_vxvy.npz
```

Merged stats:

```text
rows = 70,028
sample_weight min = 1.0
sample_weight mean = 1.1445
sample_weight max = 4.0
p90 = 1.2292
```

Run:

```text
logs/sb3/recomoproto2trackee_v0/stage1_recovery_auxloss_round4_weighted_noassist_40k_seed20260703_20260703_1621
```

Eval:

```text
evaluation_results/recovery_candidate_auxloss_round4_weighted_noassist_40k_smoke64/recovery_eval_raw-policy_20260703_162312.json
```

Result:

```text
unreachable_zone_pct mean = 25.1410%
p95 = 40.7812%
max = 45.3125%
base_target_dist_mean = 0.5296
workspace_hard_exceed_pct mean = 0.7245%
workspace_soft_exceed_pct mean = 11.1176%
obstacle unsafe/collision = 0
```

Decision: best mean and workspace metrics, but worse tail than Round 4. Do not promote as generally safer.

Lesson: hard-state weighting can improve average base-target distance but may worsen tail robustness. Use tail metrics in promotion decisions, not mean alone.

### Round 8: softened weighted aux, power 0.5

Run:

```text
logs/sb3/recomoproto2trackee_v0/stage1_recovery_auxloss_round4_weightedpow05_noassist_40k_seed20260703_20260703_1625
```

Eval:

```text
evaluation_results/recovery_candidate_auxloss_round4_weightedpow05_noassist_40k_smoke64/recovery_eval_raw-policy_20260703_162659.json
```

Result:

```text
unreachable_zone_pct mean = 27.4514%
p95 = 40.6250%
max = 43.7500%
obstacle unsafe/collision = 0
```

Decision: reject.

Lesson: softening hard-state sampling did not recover the Round 4 tail and worsened the mean.

## Current Best Artifacts

Best risk-balanced policy:

```text
logs/sb3/recomoproto2trackee_v0/stage1_recovery_auxloss_vxvy_noassist_hookstart_40k_seed20260703_20260703_1606/final_model.zip
logs/sb3/recomoproto2trackee_v0/stage1_recovery_auxloss_vxvy_noassist_hookstart_40k_seed20260703_20260703_1606/vec_normalize.pkl
```

Best mean/workspace policy, but worse tail:

```text
logs/sb3/recomoproto2trackee_v0/stage1_recovery_auxloss_round4_weighted_noassist_40k_seed20260703_20260703_1621/final_model.zip
logs/sb3/recomoproto2trackee_v0/stage1_recovery_auxloss_round4_weighted_noassist_40k_seed20260703_20260703_1621/vec_normalize.pkl
```

Promotion recommendation:

```text
Do not promote either as deployable.
If forced to choose a conservative candidate, use the risk-balanced Round 4 aux-loss policy, not the weighted Round 7 policy.
```

## What Has Not Been Achieved

- The `<15%` unreachable-zone target has not been met.
- Tail robustness is not solved; weighted mean improvements can hide worse p95/max behavior.
- Arm/gimbal imitation remains out of scope for this policy loop.
- Yaw/action row `8` is not validated and should not be mixed into the base distillation path.
- Runtime base assist is not a deployment-ready substitute for a raw policy.

## Lessons For Future Loops

1. Always load the matching `vec_normalize.pkl`. If resuming from `final_model.zip`, use the sibling vecnorm fallback and verify the log prints `[OK] VecNormalize stats loaded successfully`.
2. Run aux updates before rollouts, not after rollouts. PPO rollouts must match the policy used for PPO training.
3. Keep raw-policy evaluation clean: use `--disable_auto_base_assist`, zero base-assist blend, and zero imitation weight.
4. Judge by raw eval JSON, not by a single training-window printout. AutoPause can trigger even when raw eval improves, and training windows can look good while eval tails regress.
5. Track mean, p95, max, and safety together. Mean-only promotion is unsafe for this task.
6. Do not include yaw row `8` until it has its own validation evidence.
7. Do not keep collecting larger uniform datasets if the previous larger round regressed. More samples can dilute the hard states.
8. Weighted hard-state sampling improves average distance but can worsen tails. Any weighted candidate must be evaluated against p95/max before promotion.
9. Do not spend more time on callback-only aux tuning unless there is a specific hypothesis. The next meaningful lever should be either an integrated PPO minibatch auxiliary loss or environment/reward design.
10. Generated datasets and training artifacts are experiment outputs; source changes should be committed/pushed, but dataset artifacts should not be treated as source.

## Stratified Failure Analysis Update

Reusable tooling added:

```text
scripts/reinforcement_learning/sb3/evaluate_recovery_candidate.py
scripts/reinforcement_learning/sb3/analyze_recovery_eval_strata.py
```

The evaluator now writes `episode_details` with trajectory file/category, start waypoint fraction, end waypoint fraction, per-episode unreachable percentage, workspace metrics, obstacle metrics, and EE tracking metrics. The analyzer groups enriched evals by source checkpoint, trajectory category, start-fraction bucket, and worst trajectory file.

Generated analysis artifacts:

```text
evaluation_results/recovery_stratified_analysis_20260703.md
evaluation_results/recovery_stratified_analysis_20260703.json
```

These artifacts are ignored by git, so the source scripts and this summary are the durable tracked record.

Enriched 64-episode rerun results:

| Candidate | Mean unreachable | P95 unreachable | Max unreachable | Workspace hard mean | Obstacle unsafe/collision |
| --- | ---: | ---: | ---: | ---: | --- |
| Round 4 risk-balanced aux | `36.8615%` | `97.8069%` | `100.0000%` | `1.6678%` | `0 / 0` |
| Round 7 weighted aux | `30.6306%` | `96.8494%` | `100.0000%` | `1.2940%` | `0 / 0` |

Important caveat: these enriched reruns sampled a different 64-episode set from the earlier smoke reports, so compare them as same-run strata evidence rather than replacing all earlier promotion numbers.

Worst combined categories:

| Category | Episodes | Mean unreachable | P95 unreachable |
| --- | ---: | ---: | ---: |
| `handheld_subtle` | `33` | `46.7041%` | `98.7234%` |
| `crane_up` | `14` | `44.2580%` | `100.0000%` |
| `crane_down` | `28` | `39.8897%` | `98.6000%` |
| `dolly_pull_out` | `50` | `20.0264%` | `38.5244%` |

Main lesson: this is now clearly a trajectory-category tail problem, not an obstacle-avoidance failure. Hard-category data should focus on `handheld_subtle`, `crane_up`, and `crane_down`; `dolly_pull_out` should not dominate the next aux dataset.

Collector update:

```text
scripts/reinforcement_learning/sb3/collect_base_assist_dataset.py
```

New options:

```text
--trajectory_categories handheld_subtle,crane_up,crane_down
--trajectory_files crane_down_035.json,handheld_subtle_087.json
```

The collector writes a temporary filtered manifest and records its path plus selected categories/files in dataset metadata. A small smoke with `4 envs x 2 steps` selected `145/256` hard-category files and produced a valid dataset:

```text
data/base_assist_distill/_smoke_category_filter_vxvy.npz
observations: (8, 85)
actions: (8, 9)
action_valid_mask: (8, 9)
sample_weight: (8,)
valid rows: action rows [6, 7]
```

### Round 9: hard-category weighted aux continuation

Goal: use the stratified report to mine the failing categories instead of adding another uniform dataset.

New hard-category dataset:

```text
data/base_assist_distill/stage1_recovery_hardcats_weighted_normobs_vxvy_256x128.npz
```

Collection config:

```text
checkpoint = Round 7 weighted aux final_model.zip
trajectory_categories = handheld_subtle, crane_up, crane_down
num_envs = 256
num_steps = 128
sample_weight_mode = base_distance
sample_weight_max = 5.0
valid action rows = [6, 7]
```

Dataset result:

```text
rows = 28,451
obs_dim = 85
sample_weight min/mean/max = 1.0 / 1.4038 / 5.0
valid action counts = row 6: 28,451, row 7: 28,451, row 8: 0
```

Merged dataset:

```text
data/base_assist_distill/stage1_recovery_hardcats_round5_merged_weighted_vxvy.npz
rows = 98,479
sample_weight min/mean/max = 1.0 / 1.2194 / 5.0
```

Failed launch to avoid repeating:

```text
logs/sb3/recomoproto2trackee_v0/stage1_recovery_auxloss_round5_hardcats_noassist_40k_seed20260703_20260703_1828
```

Failure:

```text
VecNormalize expected obs shape (85,), environment was (84,)
```

Cause: training was launched without `--enable_obstacles`, while the checkpoint and dataset/eval path use the 85D obstacle observation contract.

Valid training run:

```text
logs/sb3/recomoproto2trackee_v0/stage1_recovery_auxloss_round5_hardcats_obs_noassist_40k_seed20260703_20260703_1829
```

Important launch requirements:

```text
--enable_obstacles
--obstacle_radius 0.20
--obstacle_height 0.50
--obstacle_x_range -0.35 0.35
--obstacle_y_range 0.45 1.00
--min_obstacle_start_clearance 0.10
--disable_auto_base_assist
--disable_auto_base_assist_yaw
--base_assist_imitation_weight 0.0
--base_assist_yaw_imitation_weight 0.0
```

Training result:

```text
final_model.zip written
vec_normalize.pkl written
aux loss: about 0.085 -> 0.083
approx_kl: <= 0.0105
```

Eval:

```text
evaluation_results/recovery_candidate_auxloss_round5_hardcats_obs_noassist_40k_smoke64_enriched/recovery_eval_raw-policy_20260703_183121.json
evaluation_results/recovery_stratified_analysis_round5_20260703.md
evaluation_results/recovery_stratified_analysis_round5_20260703.json
```

Same-run enriched comparison:

| Candidate | Mean unreachable | P95 unreachable | Max unreachable | Workspace hard mean | Obstacle unsafe/collision |
| --- | ---: | ---: | ---: | ---: | --- |
| Round 4 risk-balanced aux | `36.8615%` | `97.8069%` | `100.0000%` | `1.6678%` | `0 / 0` |
| Round 7 weighted aux | `30.6306%` | `96.8494%` | `100.0000%` | `1.2940%` | `0 / 0` |
| Round 9 hard-category aux | `26.8675%` | `97.4762%` | `100.0000%` | `1.0201%` | `0 / 0` |

Round 9 category result:

| Category | Episodes | Mean unreachable | P95 unreachable | Max unreachable | Workspace hard mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `crane_up` | `10` | `42.3634%` | `100.0000%` | `100.0000%` | `2.6642%` |
| `crane_down` | `11` | `41.4291%` | `95.0914%` | `98.9130%` | `0.6958%` |
| `handheld_subtle` | `11` | `22.5917%` | `39.7773%` | `40.2062%` | `0.1865%` |
| `dolly_pull_out` | `23` | `20.4770%` | `44.7118%` | `50.0000%` | `0.7726%` |

Decision: research checkpoint only, not deployable and not a promotion candidate.

Lesson: category-aware mining worked for `handheld_subtle` but combining all hard categories in one aux bucket caused crane-tail failure to remain or worsen. Next policy work should separate crane-specific treatment from handheld-specific treatment instead of using one merged hard-category dataset.

### Round 10: worst-crane exact-file aux continuation

Goal: attack the exact crane files that carried the 100% tail after Round 9.

Dataset:

```text
data/base_assist_distill/stage1_recovery_worstcrane_round6_normobs_vxvy_256x128.npz
```

File filter:

```text
crane_up_021.json
crane_up_001.json
crane_up_098.json
crane_down_087.json
crane_down_035.json
crane_down_042.json
crane_down_069.json
crane_up_034.json
crane_down_065.json
crane_down_072.json
crane_up_012.json
crane_up_006.json
crane_up_016.json
```

Dataset result:

```text
rows = 25,046
obs_dim = 85
sample_weight min/mean/max = 1.0 / 2.0034 / 6.0
valid action rows = [6, 7]
start fraction = 0.45 to 0.70
```

Run:

```text
logs/sb3/recomoproto2trackee_v0/stage1_recovery_auxloss_round6_worstcrane_obs_noassist_40k_seed20260703_20260703_1850
```

Eval:

```text
evaluation_results/recovery_candidate_auxloss_round6_worstcrane_obs_noassist_40k_smoke64_enriched/recovery_eval_raw-policy_20260703_185215.json
evaluation_results/recovery_stratified_analysis_round6_20260703.md
evaluation_results/recovery_stratified_analysis_round6_20260703.json
```

Result:

| Candidate | Mean unreachable | P95 unreachable | Max unreachable | Workspace hard mean | Obstacle unsafe/collision |
| --- | ---: | ---: | ---: | ---: | --- |
| Round 9 hard-category aux | `26.8675%` | `97.4762%` | `100.0000%` | `1.0201%` | `0 / 0` |
| Round 10 worst-crane aux | `30.8861%` | `96.9898%` | `100.0000%` | `2.2388%` | `0 / 0` |

Category detail:

| Category | Mean unreachable | P95 unreachable | Max unreachable | Workspace hard mean |
| --- | ---: | ---: | ---: | ---: |
| `crane_up` | `30.7281%` | `65.3383%` | `85.7143%` | `1.7244%` |
| `crane_down` | `62.1532%` | `100.0000%` | `100.0000%` | `4.4438%` |
| `handheld_subtle` | `32.2785%` | `86.3636%` | `100.0000%` | `2.2664%` |

Decision: reject. It improved `crane_up` but made `crane_down`, handheld, and workspace worse.

Lesson: mixing `crane_up` and `crane_down` exact-file tails in one aux dataset is still not stable. The two crane modes need separate treatment or a non-aux objective.

### Round 11: crane-down-only aux continuation

Goal: preserve the Round 9 base while correcting only `crane_down`.

Dataset:

```text
data/base_assist_distill/stage1_recovery_cranedown_round7_normobs_vxvy_256x128.npz
data/base_assist_distill/stage1_recovery_cranedown_round7_merged_weighted_vxvy.npz
```

File filter:

```text
crane_down_035.json
crane_down_042.json
crane_down_065.json
crane_down_069.json
crane_down_072.json
crane_down_087.json
crane_down_043.json
crane_down_094.json
```

Dataset result:

```text
rows = 25,827 raw / 124,306 merged
sample_weight min/mean/max = 1.0 / 2.0180 / 6.0 raw
valid action rows = [6, 7]
start fraction = 0.45 to 0.70
```

Run:

```text
logs/sb3/recomoproto2trackee_v0/stage1_recovery_auxloss_round7_cranedown_obs_noassist_40k_seed20260703_20260703_1900
```

Eval:

```text
evaluation_results/recovery_candidate_auxloss_round7_cranedown_obs_noassist_40k_smoke64_enriched/recovery_eval_raw-policy_20260703_190503.json
evaluation_results/recovery_stratified_analysis_round7_20260703.md
evaluation_results/recovery_stratified_analysis_round7_20260703.json
```

Result:

| Candidate | Mean unreachable | P95 unreachable | Max unreachable | Workspace hard mean | Obstacle unsafe/collision |
| --- | ---: | ---: | ---: | ---: | --- |
| Round 9 hard-category aux | `26.8675%` | `97.4762%` | `100.0000%` | `1.0201%` | `0 / 0` |
| Round 11 crane-down-only aux | `28.3968%` | `93.8231%` | `100.0000%` | `2.7793%` | `0 / 0` |

Category detail:

| Category | Mean unreachable | P95 unreachable | Max unreachable | Workspace hard mean |
| --- | ---: | ---: | ---: | ---: |
| `crane_down` | `37.2874%` | `89.2909%` | `97.2222%` | `1.3005%` |
| `crane_up` | `48.2474%` | `99.6930%` | `100.0000%` | `12.9563%` |
| `handheld_subtle` | `35.5199%` | `77.5635%` | `100.0000%` | `0.6725%` |

Decision: reject as promotion. It improved `crane_down` p95 relative to Round 9, but it worsened mean reachability and badly worsened `crane_up`/workspace.

### Round 12: crane-up-only aux continuation

Goal: test the symmetric `crane_up`-only branch from the Round 9 base.

Dataset:

```text
data/base_assist_distill/stage1_recovery_craneup_round8_normobs_vxvy_256x128.npz
data/base_assist_distill/stage1_recovery_craneup_round8_merged_weighted_vxvy.npz
```

File filter:

```text
crane_up_001.json
crane_up_006.json
crane_up_011.json
crane_up_012.json
crane_up_016.json
crane_up_021.json
crane_up_030.json
crane_up_034.json
crane_up_081.json
crane_up_098.json
```

Dataset result:

```text
rows = 26,180 raw / 124,659 merged
sample_weight min/mean/max = 1.0 / 2.0530 / 6.0 raw
valid action rows = [6, 7]
start fraction = 0.45 to 0.70
```

Run:

```text
logs/sb3/recomoproto2trackee_v0/stage1_recovery_auxloss_round8_craneup_obs_noassist_40k_seed20260703_20260703_1908
```

Eval:

```text
evaluation_results/recovery_candidate_auxloss_round8_craneup_obs_noassist_40k_smoke64_enriched/recovery_eval_raw-policy_20260703_191146.json
evaluation_results/recovery_stratified_analysis_round8_20260703.md
evaluation_results/recovery_stratified_analysis_round8_20260703.json
```

Result:

| Candidate | Mean unreachable | P95 unreachable | Max unreachable | Workspace hard mean | Obstacle unsafe/collision |
| --- | ---: | ---: | ---: | ---: | --- |
| Round 9 hard-category aux | `26.8675%` | `97.4762%` | `100.0000%` | `1.0201%` | `0 / 0` |
| Round 12 crane-up-only aux | `36.1841%` | `98.4299%` | `100.0000%` | `1.7254%` | `0 / 0` |

Decision: reject. This is worse than Round 9 on mean, p95, and workspace hard.

Overall lesson from Rounds 10-12: callback-style aux mining has reached a tradeoff wall. It can move one category locally, but it does not produce a globally better raw policy and repeatedly worsens workspace or another category tail. Stop adding more aux-only exact-file datasets unless the training objective changes.

## Recommended Next Stage

Stop the current callback-only aux loop and move to one of these:

1. Integrate the masked supervised base loss into PPO minibatch training so the auxiliary objective is optimized in the same update as PPO instead of mutating the action head outside PPO.
2. Add direct environment/reward shaping for tail states, especially large base-target distance and repeated late-episode unreachable episodes.
3. Build a stratified evaluation report that breaks failures by trajectory category and start fraction before doing another training run.

Minimum gate for any next candidate:

```text
64-episode raw smoke:
  unreachable mean must beat 25.1410%
  p95 must not exceed 35.9375%
  obstacle unsafe/collision must remain 0
```

If a candidate only improves mean but worsens p95/max, mark it as research-only, not a promotion candidate.
