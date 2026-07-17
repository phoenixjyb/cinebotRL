# Handoff: CineBotRL-2Wheel-NewFormEvaluate

Date: 2026-07-17
Target session: `CineBotRL-2Wheel-NewFormEvaluate`
Target workspace on `.98`: `/mnt/g/wSpace/cinebotRL-two-wheel-riser`
Purpose: restart the riser/residual curriculum from authoritative trajectories
without losing the useful controller and data-collection work already completed.

## Decision

Do **not** resume the quarantined clean capture, behavior cloning, PPO, or the
current accepted-71 Gate-C batch. The authoritative source geometry is intact,
but the current executable plans are over-retimed and are not an admissible
curriculum under the user's duration-close requirement.

The session has **79 reference-only source trajectories** and 71 plans that pass
the old kinematic gate. It still has **0 training-ready trajectories**. The next
task is a CPU-only smoothed-plan rebuild that preserves source identity and EE
path intent while eliminating discontinuous base/RS4 decomposition choices and
keeping execution duration close to the source.

## Latest operational instruction

### 2026-07-17 bounded runtime result and timing pivot

Pushed commit `7019afbb89dc5a270238ccecaad7a74590b85876`
introduced a fresh, one-case-only recovery-v4 runtime contract. The complete
repository suite passed `258/258`; post-push validation passed all 21 identity
and authorization checks. Residual capture, BC, and PPO remained disabled.

Exactly one case-74 canary ran in namespace:

```text
/mnt/g/wSpace/cinebotRL-two-wheel-riser/artifacts/two_wheel_riser/
20260717_gate_c_case74_recovery_v4_runtime_v2_exclusive
```

It reached the pinned 1,200-second wall limit without producing a gate JSON.
The sealed summary remains `passed=false` and `valid_for_training=false`; its
`missing_runtime_json` classification means a bounded runtime timeout, not a
physical trajectory rejection. Do not rerun it with a larger timeout.

The reason is structural. Case 74 has `11.373883 s` of source trajectory but
`188.546638 s` of execution timing (`16.58x`). Its largest single scheduled
interval is `12.5909 s`, caused by a `3.1355 rad` base-yaw branch change and a
`3.1067 rad` proxy-gimbal branch change. The initial retimer spends
`84.9484 s` on raw path-heading changes; removing that term alone reduces its
computed schedule from `91.0728 s` to `26.5491 s` before plan refinement.

This is portfolio-wide, not case-74-only:

```text
old kinematic admissions:              71/79
training-ready trajectories:            0/79
execution/source ratio, median:          5.508x
execution/source ratio, p90:            12.749x
execution/source ratio, maximum:        17.129x
accepted plans at or below 2.0x:         2/71
```

Therefore pause all GPU Gate C work. First replace per-anchor path-heading
retiming with a globally continuous nonholonomic base/RS4 decomposition and
explicit smoothing around reversals. Do not solve discontinuities by adding
minutes of dwell time.

### Exact next bounded task for the riser session

1. Start from clean pushed `7019afb`; preserve the consumed timeout namespace.
2. Add a versioned smoothed-plan schema. Keep the immutable raw source arrays
   and hashes in a separate provenance block.
3. Remove raw `bidirectional_path_heading` interval changes as a direct retiming
   demand. Select base direction and continuous RS4 yaw branch over a window or
   whole trajectory, then densify the resulting smooth command.
4. Do not raise the existing `0.4 m/s` base-speed, `0.4 rad/s` yaw-rate,
   gimbal-rate, workspace, thermal, or tracking-error limits.
5. Run CPU-only cases `74,77,52` in a fresh namespace. Stop after the first
   contract failure and report its exact duration ratio, path drift, maximum
   base/proxy branch step, and failed physical check.
6. Only if all three pass, run the all-79 CPU planner and require at least 70
   timing-and-transition admissions. Do not launch Isaac as part of this task.

The exact-source Gate-B portfolio currently has 71/79 kinematic admissions,
but it is not a training corpus. A later case-74 canary than the
forbidden-contact run replayed semantic yaw continuously and removed the
`~720 deg` proxy-wrap failure. It ran without a safety termination to
`159.901892 / 188.546638 s`, but still failed completion and position tracking
(`1.065665 / 1.094696 m` p95/max). The remaining structural issue is
reverse-recovery/path-progress behavior, not proxy-yaw continuity.

The reviewed recovery-v4 controller parent is `ba8f4e0`; runtime-contract HEAD
is now `7019afb`. Recovery-v4 and the provisional thermal monitor did start
together, but the over-retimed plan hit the bounded wall limit before producing
dynamic evidence. Do not start the accepted-71 batch, case 77, residual capture,
BC, or PPO. The older CPU-only contract below is retained as provenance but is
superseded operationally by the runtime result above. No further runtime
authorization is appropriate until replacement plans pass the timing and
transition gates. This document requests no differential-session work.

### Historical CPU-only recovery-v4 evidence contract

Pushed revision `2aea3ea1ff902bf1d6ad46663ec2843ed6de3161` adds the
fail-closed case-74 recovery-v4 evidence contract. The obsolete
`AUTHORIZED_CASE74_RECOVERY_V4` token and the shared-runner case-74 route now
fail before Python starts; this revision intentionally contains **no valid
runtime authorization token**.

The canonical contract pins case 74, the reviewed controller parent
`ba8f4e0`, the exact source/portfolio/plan/gains/USD identities, recovery
profile and range, the fresh namespace, and all relevant code blobs. Its
validator derives `HEAD` and the configured upstream internally, requires the
canonical committed contract path/blob, and rejects dirty tracked state,
forged lineage, altered identities, or an existing namespace. Policy-rate
aggregate telemetry records recovery activation, direction changes/chatter,
yaw saturation, and candidate-versus-legacy command deltas without changing
commands or expanding the existing 1 Hz trace.

Focused CPU tests pass `29/29`; the authoritative CPU suite passes `257/257`
with two pre-existing pytest configuration warnings. Independent post-push
validation reports all contract checks true while
`runtime_authorized=false` and `valid_for_training=false`. `HEAD` equals its
configured upstream, the protected namespace is absent, and no Isaac, GPU,
capture, BC, or PPO work ran at that revision. Commit `7019afb` later supplied
the separate bounded authorization and produced the timeout evidence recorded
above. That consumed namespace must not be reused.

## Current stopped state

The July 16 clean residual capture was stopped after its active case completed.
It contains exactly 65 gate results and 65 NPZ files and is preserved at:

```text
/mnt/g/wSpace/cinebotRL-two-wheel-riser/artifacts/two_wheel_riser/
20260716_residual_all79_phase_v3_clean_QUARANTINED_UPSTREAM_TRUNCATED_SOURCE_20260717
```

No riser playback or dataset-gate process remained active at the handoff
snapshot. This directory is diagnostic history only. It must not be renamed
back, merged into a new corpus, or accepted because its individual Isaac gates
passed.

## Authoritative evidence

The all-79 audit compared each downstream `stage_gik_no_obstacle79_nominal`
reference against its original JSON:

| Metric | Authoritative sources | Old two-wheel stages |
| --- | ---: | ---: |
| Episodes | 79 | 79 |
| Exact-source passes | 79 required | **0** |
| Total poses | 71,038 | 14,066 |
| Total Cartesian path | 618.305 m | 83.306 m |
| Total duration | 1,388.425 s | 1,394.645 s |
| Median stage/source path-length ratio | — | 0.14625 |

All 79 stages have a pose-count mismatch. Passing balance, tracking, action
reconstruction, or residual-envelope gates cannot repair this upstream target
identity failure.

Audit files are available at:

```text
# Local
/Users/yanbo/Downloads/gikWBC9DOF_two_wheel_upstream_audit_20260717/evidence/

# .98
/mnt/g/wSpace/cinebotRL/data/gikWBC9DOF_two_wheel_upstream_audit_20260717/
```

Checksums:

```text
8014e3232b8c30881eea7a8653db0945a370cae205eab9609e906d672dc5ecf1  two_wheel_upstream_trajectory_integrity_all79.json
c4fcde3da9d463fb0a3743f1c123cb5c8e9a5856605f58678a72dfabddabbd95  two_wheel_upstream_trajectory_integrity_all79.csv
```

## New input package

Use this package as the only trajectory source for the next riser-plan build:

```text
# Local
/Users/yanbo/Downloads/gikWBC9DOF_exact_source_reference_all79_20260717

# .98
/mnt/g/wSpace/cinebotRL/data/gikWBC9DOF_exact_source_reference_all79_20260717
```

Manifest SHA-256:

```text
f265aa1bdd1cd6c762fd6e5367c00c7abcb7b19dea76bb30c6311885d2f3237d
```

The package contains all episodes 1–79, 71,038 ordered source poses, original
timestamps, normalized finite quaternions, per-file hashes, and the
`exact_source_v1` contract. Its package-level flags are deliberately:

```text
integrity_passed=true
quality_qualified_teacher=false
valid_for_training=false
```

This means it is ready for plan generation and retargeting, not BC/PPO.

The existing three-case transport canaries remain at:

```text
/mnt/g/wSpace/cinebotRL/data/gikWBC9DOF_exact_source_teacher_integrity_canaries_20260716
```

Their loader audit passes ep1/4/7 source identity with zero timestamp error,
while correctly rejecting all three from training.

## What remains reusable

Keep and reuse the following, subject to their existing tests:

- frozen LQR balance controller and push-recovery evidence;
- deterministic trajectory feed-forward;
- residual action definition `delta_vx`, `delta_wz`, `delta_riser_target`;
- 26-observation physical-state contract;
- pitch, position, and attitude phase governors;
- action scale `[0.30, 0.40, 0.10]` as a provisional controller-envelope
  result, to be re-audited on the replacement corpus;
- immutable admission records, source-commit checks, command-reconstruction
  checks, zero-clipping gate, and leakage checks;
- fail-fast dynamic-gate collector and accepted-only dataset merger.

Do not reuse as training truth:

- `20260716_corrected_all79_stage*` desired references;
- `20260716_all79_playback_inputs_v4` as an admitted source manifest;
- any `phase_v3_clean` labels or model derived from them;
- the former `split_teacher_v2` source package or its hash;
- any success count that does not include `exact_source_v1` admission.

## Replacement smoothed-plan contract

The raw input remains immutable and auditable. The executable plan may smooth
or resample difficult transitions; it does not have to command every raw anchor
as a controller waypoint. It must carry all of the following:

1. `source_manifest_sha256` equal to the new package hash above.
2. Per-episode `source_json_sha256` matching the package item.
3. A read-only source block containing `source_pose_count=N`, original
   `source_time_s[N]`, positions `N x 3`, and semantic DFR quaternions `N x 4`
   preserved verbatim for provenance and comparison.
4. A separately named smoothed target block and strictly increasing execution
   schedule. Never overwrite or relabel the raw source block as smoothed data.
5. The same first and final EE targets, source ordering, and motion direction.
   Smoothed EE path length must remain within 5% of the source. Position
   deviation from the source polyline must remain within the existing
   `0.15 m` p95 and `0.25 m` maximum envelope.
6. Per-case execution duration must be no more than `2.0x` source duration;
   target the portfolio median at or below `1.5x`. A physically infeasible case
   is an honest reject, not permission for 5x-17x retiming.
7. Base yaw and continuous proxy yaw must use one globally continuous branch.
   No pre-densification decomposition jump may exceed `0.25 rad`; final
   samples must also pass the existing velocity/rate limits.
8. Initialization/acquisition samples in a separate segment. They must not be
   counted as source anchors or residual labels.
9. `target_link=ee1_tool`, semantic DFR quaternion order `xyzw`, and `+Y`
   semantic forward-axis ownership retained.
10. Physical `cam_link` and gimbal solutions remain downstream diagnostics;
    physical DJI joint labels remain excluded from learned actions.
11. `trajectory_integrity_passed=true` only after raw-source provenance and
    smoothed-path comparison both pass.
12. `valid_for_training=false` until timing, transition, kinematic, thermal,
    and dynamic quality gates all pass.

Repeating the first or last waypoint to fill a shape mismatch is forbidden.
Reconstructing source time from a nominal sample rate is forbidden. A smoothed
plan must be explicitly identified as such; it must never masquerade as the raw
waypoint sequence.

## Required continuation sequence

### Gate A — admission hardening

- Finish the current fail-closed validator changes.
- Require the package hash, 79 episodes, `exact_source_v1`, all per-file hashes,
  and `valid_for_training=false` at reference-ingest time.
- Verify the old v4 plan and the three canaries cannot enter BC.
- Add tests for missing anchors, reordered anchors, timestamp replacement,
  initialization leakage, and a copied old package under a new directory name.

### Gate B — reference-plan regeneration

- Generate a new namespace; do not overwrite historical artifacts.
- Start with CPU canaries case 74 (known branch discontinuity), case 77 (healthy
  compatibility), and one long-duration case such as 52.
- Prove the raw source block remains exact while the executable target is
  explicitly smoothed and passes the path-length, deviation, duration-ratio,
  branch-continuity, and rate gates above.
- Render or numerically compare the full source and smoothed desired paths
  before any Isaac capture.
- Then generate all 79 plans and write one machine-readable admission summary.
- Report accepted and rejected IDs honestly. Require at least 70 admissions for
  the proposed 50/10/10 split; do not recover the count by relaxing the new
  timing or transition contract.

### Gate C — deterministic dynamic qualification

- Run the frozen controller and unchanged safety gates before collecting
  residual labels.
- Capture residual labels only from dynamically passing episodes.
- Record rejected episode IDs and failure modes; never force 79/79 by relaxing
  thresholds or inserting old-plan fallbacks.
- Recompute the raw residual envelope on the new corpus before freezing action
  scales. `[0.30, 0.40, 0.10]` must be confirmed, not assumed.

### Gate D — clean residual corpus

- Start a fresh single-commit capture only after all 79 plans finish Gate B.
- Require zero clipping and exact teacher-command reconstruction.
- Bind every NPZ to source JSON hash, plan hash, code commit, and dynamic result.
- Prove no quarantined filename, hash, or source directory appears in the
  merged manifest.

### Gate E — bounded BC, then later PPO

A small BC experiment may start only when all 79 inputs have been attempted
and the accepted exact-source set supports, at minimum:

- 50 diverse training episodes;
- 10 untouched validation episodes;
- 10 untouched holdout episodes;
- trajectory-shape, duration, height, and motion-direction coverage in each
  split rather than a random row split;
- no source video or episode leakage across splits.

These are minimum experiment gates, not permission for production promotion.
Run a tiny overfit/loader smoke first. Stop if validation does not improve or
if reconstructed residual commands disagree with the deterministic teacher.
PPO remains disabled until BC and the untouched dynamic holdout gate pass.

## Stop rules

Stop and report instead of continuing when any of these occurs:

- source manifest or per-file hash mismatch;
- any missing, duplicated, or reordered source anchor;
- initialization appearing in action labels;
- any attempt to admit the reference-only package directly to training;
- an old-lineage hash or quarantined path appearing in a new manifest;
- action clipping or command reconstruction beyond the existing numerical
  tolerance;
- a dynamic safety failure;
- fewer than 70 accepted episodes for the proposed 50/10/10 split;
- any plan with execution/source duration ratio above `2.0x`;
- any hidden raw-to-smoothed relabeling, path-length drift above 5%, or
  pre-densification base/proxy branch jump above `0.25 rad`.

## Expected reply to the GIK session

When the next bounded stage finishes, report:

1. new plan namespace and manifest SHA;
2. exact-source pass count out of 79;
3. dynamic accepted/rejected counts and episode IDs;
4. residual-envelope maxima and clipping/reconstruction results;
5. proposed train/validation/holdout episode lists;
6. explicit confirmation that BC/PPO did or did not start.

## Addendum: ep77 passed upstream retarget seed

A full-source no-obstacle ep77 seed is now available for planner diagnosis:

```text
# Local
/Users/yanbo/Downloads/gikWBC9DOF_ep77_exact_source_upstream_teacher_seed_20260717

# .98
/mnt/g/wSpace/cinebotRL/data/gikWBC9DOF_ep77_exact_source_upstream_teacher_seed_20260717
```

It preserves all 273 source anchors and original timestamps with an identity
anchor map. The position-GIK seed passes the upstream gates: maximum EE
position error `0.0634482 m`, base speed `0.391245 m/s`, yaw rate `0`, arm
speed `0.999832 rad/s`, and source attitude-target rate `0.0318751 rad/s`.

Hashes:

```text
2dc31d86325155fafb0dc3afe8870f9ae32ea3c58d5ed7d2671f43aa2d4d7404  manifest.json
cfb321784cab0b540610c51c2cd19b23fc5d898325c29dff3ea0b7b9bcb29d2c  episode_0077_exact_source_upstream_seed_v1.npz
5f39896865cb120197d4495605b25e531601e36ef3a16484b371082b13f0ba45  source.json
```

This artifact is deliberately `valid_for_training=false`. It is a holonomic
base+arm retarget seed with no learned actions, not a dynamically qualified
two-wheel rollout. Use it to compare or initialize the ep77 riser planner;
retain the authoritative source block, and regenerate executable riser timing
and dynamic labels under the existing Gate C rules.

## Addendum: vertical-workspace classification

The authoritative all-79 sources were audited against the current
`0.60..1.80 m` camera-Z workspace, allowing one constant vertical placement
offset per episode:

```text
# Local
/Users/yanbo/Downloads/gikWBC9DOF_exact_source_vertical_workspace_20260717

# .98
/mnt/g/wSpace/cinebotRL/data/gikWBC9DOF_exact_source_vertical_workspace_20260717
```

Results: `78/79` source height spans are compatible with some constant shift.
Episode 27 is the sole irreducible span reject: source Z is
`-0.687227..1.458746 m`, a `2.145973 m` span, exceeding the available
`1.20 m` workspace. This proves the 70-case target is not blocked by vertical
geometry alone, while ep27 must remain an honest rejection unless the physical
workspace changes.

```text
51c2e60e11e53cf8b1884d0d01bec61df4f8cacec9d7cf35bb3b5b08f81447ab  vertical_workspace_all79.json
9cbdbe679d17a6236c27d5f835391b23a554f0ff9896b9c48d137d19c7679777  vertical_workspace_all79.csv
```

A vertical-span pass is not planner or dynamic qualification. It only removes
source Z span as the explanation for the other 78 cases.

## Addendum: completed all-79 Gate B v2 result

The preview/coupled-retiming Gate B batch completed in a fresh namespace:

```text
/mnt/g/wSpace/cinebotRL-two-wheel-riser/artifacts/two_wheel_riser/
20260717_exact_source_all79_plans_v2_preview_coupled
```

The result proves trajectory integrity for the complete package, but it does
not meet the kinematic admission-count stop rule:

```text
attempted cases:                  79/79
exact-source integrity passes:   79/79
kinematic quality passes:        66/79
kinematic rejects:               13
valid_for_training:              false
training_started:                false

c37ab4762e91492309f7c80a54df61379137282395f8bc3b482adb605ceca296  manifest.json
58f2a1273a75d60c51277ca752d4a97caaff33d8cd129b37e63e0d9b60de0afb  summary.json
```

Rejected episode IDs are:

```text
6, 13, 18, 21, 22, 27, 52, 55, 64, 69, 74, 75, 76
```

The recoverable families are narrow and should be handled without rebuilding
the accepted 66:

- terminal XY maximum only: ep6 and ep18;
- proxy-rate only: ep13, ep52, ep55, and ep69;
- position-p95 only: the paired ep21 and ep22;
- mixed position-p95 and proxy-rate: ep64 and identical ep74/75/76;
- irreducible vertical workspace: ep27.

The current corpus is below the required `>=70` kinematic admissions.
Therefore Gate C, residual capture, BC, and PPO remain closed. Recover at
least four cases with unchanged source anchors and unchanged thresholds,
re-audit the merged manifest, and only then consider deterministic dynamic
qualification. Historical accepted artifacts must not be overwritten.

## Addendum: hash-audited 71-case Gate B portfolio

Targeted recovery preserved the original 66 passing plans and replaced only
episodes `52, 69, 74, 75, 76` with passing candidates. The composed portfolio
is stored at:

```text
/mnt/g/wSpace/cinebotRL-two-wheel-riser/artifacts/two_wheel_riser/
20260717_exact_source_all79_portfolio_v4_threshold71
```

Authoritative counts and hashes:

```text
case files and integrity passes:  79/79
kinematic accepted:               71/79
kinematic rejected:                8/79
recovery-selected cases:          52, 69, 74, 75, 76
remaining rejects:                6, 13, 18, 21, 22, 27, 55, 64
minimum Gate C candidates:        70
Gate C candidate count met:       true
Gate C dynamic quality started:   false
valid_for_training:               false
training started / PPO started:   false / false

851a7b2751cd397ba35daf57d1a8c6971fb14ed0186683af48d3c6109090570a  manifest.json
688b5bc23d801705c3132c511f009e1deb3d2af0a16a2a3ae33467764272db83  summary.json
```

An independent readback verified all 79 NPZ files against their per-item
SHA-256 values, recomputed 71 passing kinematic check dictionaries, and found
no missing cases, integrity failures, timestamp replacements, anchor-map
failures, initialization leakage, or training-enabled flags.

This closes the Gate B count stop rule and permits a bounded deterministic
Gate C run over the 71 accepted plans. It does not authorize residual capture
from failures, BC, or PPO. Gate C must remain fail-fast on the first dynamic
safety or quality failure and must record accepted/rejected IDs without
relaxing thresholds.

### Gate C canary progress

Gate C infrastructure was committed and pushed from the riser worktree at
revision `abc87fd`; local and upstream revisions matched and `63` focused
riser tests passed before execution. The fail-fast canary portfolio is
`1, 52, 74, 77`, with no residual dataset output.

Case 1 completed and passed every unchanged dynamic check:

```text
execution phase:                  77.833030 / 77.833030 s
simulation wall duration:        102.425 s
position error p95 / max:        0.127153 / 0.147211 m
attitude error p95 / max:        0.159065 / 0.226468 deg
pitch p95 / max:                 3.854715 / 4.759978 deg
proxy servo error p95 / max:     0.122178 / 0.195771 deg
internal proxy rate max:         39.426843 deg/s
internal attitude IK failures:   0
action/riser/proxy saturation:   0 / 0 / 0
termination:                     none
residual dataset written:        no
passed:                          true

2bfe794bd2329ff4822c929ec0c2f1ee5fb998b3a18f318fa1fdb4f7e086c6cc  gates/case_0001.json
```

The fail-fast runner then advanced to case 52. This proves case 1 only; Gate C
as a portfolio remains in progress and training remains disabled.

The current case JSON has one evidence-schema defect: its field named
`source_duration_s` contains `plan.time_s[-1]`, which is the retimed execution
duration (`77.833030 s`), not the immutable source duration (`4.634756 s`).
The simulation phase and horizon intentionally use execution time, so this
does not invalidate the dynamic metrics above. However, the runner must emit
separate `source_duration_s` and `execution_duration_s` fields, rename its
internal phase bound accordingly, and regenerate/reseal Gate C JSON artifacts
before a final Gate C manifest is accepted.

The same naming pattern also exists in
`smoke_all79_whole_body_playback.py`; the correction must cover both runtime
evidence paths and include a regression case where source and execution clocks
are intentionally unequal. Export and portfolio manifests already distinguish
the two durations correctly.

Case 52 subsequently completed and passed all unchanged dynamic checks:

```text
execution phase:                  292.740729 / 292.740729 s
simulation wall duration:        354.76 s
position error p95 / max:        0.108022 / 0.153767 m
attitude error p95 / max:        0.151288 / 0.211011 deg
pitch p95 / max:                 4.655275 / 5.569997 deg
proxy servo error p95 / max:     0.102525 / 0.208950 deg
internal proxy rate max:         39.893556 deg/s
internal attitude IK failures:   0
action/riser/proxy saturation:   0 / 0 / 0
termination / residual dataset:  none / none
passed:                          true

fbdbb90ddd28c71bfa42c3ba17d59d0eb7e396259a1355b36bc8640810c8034f  gates/case_0052.json
```

The fail-fast runner advanced to case 74. Case 52's metric result is also
provisional until the timing-field schema is corrected and resealed.

Case 74 then failed inside the dynamic loop, before applying the offending
residual action, in the residual-action envelope check:

```text
ValueError: residual action scale is too small:
[1.00200645 0.19818327 0.0356378]
```

The first normalized residual channel exceeds the frozen envelope by roughly
`0.200645%`. No `case_0074.json`, residual labels, or case-77 execution were
produced. This is an authoritative runtime envelope/zero-clipping stop of the
current runner, but not yet a physical dynamic rejection. The
provisional summary's `pre_execution_residual_action_reconstruction` label is
too broad because Isaac and the simulation loop had already started. The
repaired failure artifact must include completed step, elapsed time, phase
time, and explicit confirmation that the offending action was not applied.
Do not clip the command or silently enlarge the scale.

The exception path also left the Isaac bridge process resident and did not
write a machine-readable rejection artifact. Before any rerun, harden the
runner so a pre-execution fail-closed exception records the case, error class,
and training-disabled state, then releases the GPU. Complete the timing-schema
repair and reseal cases 1 and 52; afterward, recompute the raw residual envelope
from accepted plans before deciding whether a new action scale is justified.
Gate C is not passed and residual capture, BC, and PPO remain closed.

The stop also exposes a gate-order defect. `build_residual_action()` normalizes
a prospective learned label, while Gate C drives the deterministic controller
from separately computed raw commands and writes no residual dataset. Gate C
must therefore continue physical evaluation without clipping when the frozen
label scale is exceeded, recording two independent outcomes:

- deterministic dynamic quality and safety;
- residual-label envelope admission.

An envelope overflow must keep Gate D, BC, and PPO closed, but it must not
prevent collecting the raw residual maximum needed to recompute action scales
from dynamically passing episodes. Do not widen `[0.30, 0.40, 0.10]` yet.
Refactor and test this separation, then rerun case 74 under unchanged physical
commands and gates with no label dataset output.

The gate-order refactor is committed and pushed at `d4c2097`; `24` focused
tests prove the frozen envelope still rejects the prospective normalized label
while the raw deterministic command remains exact and unclipped. A corrected
machine-readable note must supersede the provisional summary's
`first_dynamic_reject` interpretation before the case-74 physical rerun.

That correction is now sealed:

```text
dbe43eadcfceac7a084a76237bf2edacd95cf862ded51aba15b92c3a62acb6dc  GATE_ORDER_CORRECTION.json
```

It binds the superseded summary hash, both repair commits, exception-smoke
hashes, `dynamic_quality_passed=null`, label-envelope failure, no applied
offending action, no labels/dataset, and all learning stages disabled.

The provisional fail-fast run is sealed as a rejected canary:

```text
d3c030da6bf238d45f11911ca8d26087f0c909f1090178118d11bc58d242a966  admission.json
b4b8444a885f3ab08bb8c90de9df9fc7bd3e10082ad0e65f36f469eb3cf7d84f  summary.json
425d93db2ffcd6f87ec8f52bc02be841d59673aaafbcab225873868923644083  logs/case_0074.log
```

The summary binds commit `abc87fd92b44c0a24698f51b4ecd07974c4d8e2a`,
the 71-case portfolio manifest, provisional case-1/52 JSON hashes, the timing
schema quarantine, case 74's exact overflow vector, runner exit code `4`,
forced GPU cleanup, and case 77 as not started. It explicitly records no
threshold relaxation, clipping, residual capture, BC, or PPO.

The Gate C clock/failure-schema repair is committed and pushed on the riser
branch at `af4a4da`. The corresponding shared whole-body clock separation is
committed at `48afd96`. These commits change evidence admission and naming,
not trajectory commands, controller behavior, residual scales, or gates.

### Final corrected case-74/77 Gate C result

The gate-order refactor, split source/execution clocks, and exclusive-GPU
guard were combined at pushed riser revision
`ec210a8638d0364c94ce2bcdc508190e92eb8585`. The corrected fail-fast run used
the unchanged case-74 plan and deterministic controller in a fresh namespace:

```text
/mnt/g/wSpace/cinebotRL-two-wheel-riser/artifacts/two_wheel_riser/
20260717_gate_c_case74_77_v3_exclusive_timing_resealed
```

Case 74 is now a final physical dynamic rejection, independently of its
residual-label result. It completed `13,659` simulation steps and reached only
`46.961485 / 188.546638 s` of the retimed execution before a forbidden-body
contact terminated the rollout. The immutable source duration remains
`11.373883 s`; the source trajectory was not shortened or substituted.

```text
dynamic quality passed:                    false
source / execution duration:               11.373883 / 188.546638 s
completed execution phase:                 46.961485 s
termination:                               forbidden_body_contact = 1
position error p95 / max:                  0.546011 / 1.842571 m
pitch max:                                 15.802575 deg
attitude error max:                        32.765379 deg
proxy servo error max:                     719.887930 deg

raw residual command abs max:              [0.401173, 0.308157, 0.012529]
normalized prospective label abs max:      [1.337243, 0.770392, 0.125295]
residual-label envelope passed:             false
prospective residual applied to commands:  false
executed residual action abs max:           [0, 0, 0]
residual dataset:                           none
```

Thus the prior provisional `ValueError` is superseded in two ways: the
corrected runtime was allowed to evaluate the deterministic physical dynamics,
and it recorded dynamic quality and label-envelope admission as separate false
outcomes. The normalized values exactly equal the raw maxima divided by the
still-frozen `[0.30, 0.40, 0.10]` scale; they were diagnostic values only.
No clipping or label action affected the plant.

The fail-fast rule correctly prevented case 77 from starting. The final sealed
hashes are:

```text
bd555b2b2298f0f6591b5f497e858e7ab3bfd66707324aa3932331a0f909c8b7  admission.json
9bec49cf68d37d100b800e6505f5d0e5b6df2d1af30cd5f4e89bbe10d7794eb4  gates/case_0074.json
eddd6b7532a33e4b1d8dc6a8baf6bbe372945239b898f89c30fa04ee02b02875  logs/case_0074.log
b4e6d3bd3e0ec8cebf0b1646fa57d6b65e527dddb68c0a9458d33689da29ad2b  summary.json
b6bbd2dc25783ddff8364bafea1a23b06555d7f2dfe089095dfad29304cde4ee  final_status.json
```

The GPU owner was released and the post-run playback-owner set was empty when
`final_status.json` was sealed. This run does not invalidate the 71/79 Gate B
portfolio, but it stops Gate C at case 74. Do not start the accepted-71 batch,
case 77, residual capture, BC, or PPO. The next permitted riser task is a
CPU-first structural diagnosis of case 74's reverse-recovery divergence and
proxy-yaw wrap, followed by a new bounded canary only after its controller
change and evidence contract are reviewed.

### Latest case-74 continuous-yaw and recovery-v4 status

The preceding forbidden-contact result remains valid historical evidence but
is superseded as the latest case-74 controller result. Commit
`f07669c681c1b2a6ae483177fa1893828913030b` replayed semantic yaw branches
statefully and removed the proxy-yaw discontinuity in a fresh namespace:

```text
/mnt/g/wSpace/cinebotRL-two-wheel-riser/artifacts/two_wheel_riser/
20260717_gate_c_case74_continuous_yaw_fix_v4_exclusive
```

The rollout did not terminate or report a forbidden contact. It progressed
substantially farther but still did not complete the exact-source execution:

```text
source / execution duration:               11.373883 / 188.546638 s
completed execution phase:                 159.901892 s
completed simulation steps:                75,420
position error p95 / max:                   1.065665 / 1.094696 m
attitude error max:                         0.458524 deg
pitch max:                                  7.645328 deg
proxy servo error max:                      0.500490 deg
termination:                                none
dynamic quality passed:                     false
residual-label envelope passed:             false
valid for training:                         false
```

Sealed evidence hashes:

```text
37371f47cac9d4056877c4c468fa60598f8d263300c86da00ea7f8d496b27040  admission.json
f5686d491cc3dff069a58f54fb974e718057993261a1924976e907b737fff65d  gates/case_0074.json
8f02ad60c1d81fad32fa3a70274f1d8c00ce736950cab2df8d9631d93fa8f2cb  logs/case_0074.log
3deb477ab7ee45cca9aafaac801b05ce4935523460d32f3cfd9e6b94cb37535f  summary.json
```

Subsequent pushed commits `fd949c5`, `656947e`, `9517633`, and `3dc87c1`
implement and guard a bounded recovery-v4 candidate that steers reverse
recovery from commanded motion and gates it to recovery-error conditions.
Commit `789be2d` adds the provisional motor-thermal admission evidence, and
`ba8f4e0` publishes the consolidated status. The full CPU suite passes `246`
tests. These are code and evidence-contract results only: recovery-v4 plus the
thermal gate has not run in Isaac and must not be described as dynamically
qualified.

Do not re-open proxy-yaw wrapping as the current root cause. Review the
recovery-v4 invariants against the sealed trace, then—only with explicit
case-74 authorization—run one fresh exclusive-GPU canary. Stop and audit on
any rejection; case 77 and all learning stages remain closed.

### Cross-session ep4 smoothing boundary

The differential-session ep4 investigation does not change this riser
session's admission state. A duration-preserving derived ep4 package now
exists with 192 poses, exact `14.042191 s` duration, `-0.0070%` Cartesian
length difference, unchanged orientation, and at most 20 mm of local
horizontal relief. Its CPU retarget canary completed only 49/191 transitions
before the bounded stop and produced no candidate.

Do not copy that ep4 package, its paired seed, or its partial checkpoint into
the riser corpus. It is marked invalid for training and dynamic evaluation;
it does not raise the riser admission count and does not authorize capture,
BC, PPO, case 77, or the accepted-71 batch. The riser session remains on the
case-74 recovery-v4 review and exclusive-GPU admission gate described above.

The shared CPU suite passes `248` tests after the derivation work. That is a
code-regression result only, not new dynamic evidence for either session.
