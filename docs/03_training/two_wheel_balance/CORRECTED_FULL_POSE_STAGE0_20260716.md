# Corrected Full-Pose Stage-0 Gate

## Current decision (updated 2026-07-17)

**STOP before corpus regeneration, BC, or PPO.** The former 78/79 and 79/79
schema-v3 retarget results are derived from an upstream-truncated teacher
lineage and are now marked:

```text
QUARANTINED_UPSTREAM_TRUNCATED_SOURCE_20260717
```

The quarantine does not invalidate the low-level engineering evidence. It does
invalidate every claim that these candidates preserve or learn the intended
authoritative end-effector trajectories.

## Quarantined source lineage

Source package:

```text
no_obstacle_monorepo_ee1_split_teacher_all79_moveit_urdf_v3_20260713
```

Manifest SHA-256:

```text
af035fb50f17322add90bf008427c9247dbbf08ee0bc38dd6d24172d9e3e14e4
```

Its `split_teacher_v2` inputs were produced from truncated/resampled GIK logs.
Nonholonomic retargeting, interpolation, balance checks, and dynamic safety
gates cannot reconstruct waypoints that were removed or replaced upstream.

Authoritative integrity mismatches include:

| Case | Quarantined stage | Authoritative source | Stage path | Source path |
|---|---:|---:|---:|---:|
| 1 | 253 poses / 25.123689 s | 256 poses / 4.634756 s | 1.092 m | 2.452 m |
| 4 | 224 poses / 22.291202 s | 723 poses / 14.042191 s | 2.007 m | 3.849 m |
| 7 | 174 poses / 17.247189 s | 663 poses / 12.940941 s | 2.260 m | 3.808 m |

All candidates and dynamic results carrying that package SHA are diagnostics
only, with `valid_for_training=false`, `runtime_approved=false`, and
`training_started=false`.

## Evidence that remains valid

The following results remain useful for controller and pretraining research:

- frozen-LQR two-wheel balance and the 28 kg provisional plant;
- physical camera option-B adapter,
  `R_world_cam = R_world_DFR * Rz(+pi/2)`;
- semantic `ee1_tool` position and world-DFR attitude contracts;
- gravity-aware acquisition, analytic gravity regression, interpolation checks,
  arm/gimbal limits, and nonholonomic transition checks;
- case-20 balance-governor success and the case-79 governor failure diagnosis;
- dynamic safety and quality gate implementation.

These show controller behavior against the quarantined references. They are not
evidence of tracking the requested source trajectories.

## Required replacement contract

No teacher package enters retargeting unless package and episode artifacts pass
all of these gates:

1. `trajectory_integrity_contract=exact_source_v1`.
2. The package is not in the quarantined SHA denylist.
3. For each episode, authoritative source pose count, reference pose count, and
   state count are all exactly `N`; action/transition count is exactly `N-1`.
4. Full desired timestamps are preserved exactly and transition `dt` equals
   their adjacent differences.
5. Ordered desired position and semantic-DFR attitude geometry is preserved.
   Explicit retiming may change timing only; it may not remove, replace, or
   reorder waypoints.
6. Initialization/acquisition is separate metadata and must not overwrite the
   authoritative semantic trajectory.
7. `trajectory_integrity_passed=true`, `teacher_quality_passed=true`,
   `teacher_approved_envelope=true`, `valid_for_candidate_training=true`, and
   `valid_for_training=true`.
8. Physical DJI gimbal joints remain internal actuator/IK variables, never
   teacher labels. Observation and reward use physical `cam_link` FK.

Only after these upstream gates pass may the existing retarget feasibility,
gravity, balance, actuator, interpolation, and dynamic quality/safety gates run.

## Exact-source reference and canaries

The authoritative reference package is now available at:

```text
/mnt/g/wSpace/cinebotRL/data/gikWBC9DOF_exact_source_reference_all79_20260717
```

Its manifest SHA-256 is:

```text
f265aa1bdd1cd6c762fd6e5367c00c7abcb7b19dea76bb30c6311885d2f3237d
```

It contains 79 episodes and 71,038 authoritative source poses under the
`exact_source_v1` integrity contract. It is a reference-only package:
`quality_qualified_teacher=false` and `valid_for_training=false`. The loader
may use it to define immutable source geometry and timestamps for retargeting,
but it must never expose policy actions or be admitted directly to training.

The `.98` transport canaries are at:

```text
/mnt/g/wSpace/cinebotRL/data/gikWBC9DOF_exact_source_teacher_integrity_canaries_20260716
```

Cases 1, 4, and 7 prove the loader/transport contract: exact counts, exact
timestamps, and near-zero source pose error. Their free-GIK states are solver
priors only. They are intentionally `valid_for_training=false` and must never
be played back or admitted directly to training.

The downstream exact-source candidate contract stores `M` executable states,
`M-1` controls, the immutable `N` source poses/timestamps, a complete ordered
source-anchor map, explicit acquisition metadata, and per-source-interval
retiming. Candidate admission independently verifies all of those fields and
the source package hash before any dynamic execution.

## Gate-1 bounded result

Gate 0 is implemented and fail-closed. The old `af035fb5...` lineage is denied,
the exact-source package and episode hashes are verified, initialization cannot
replace source anchor 0, and downstream playback/dynamic entrypoints reject
non-exact-source schemas.

The first bounded Gate-1 canaries remain rejected for training:

- Case 1 with the original free-GIK prior reached source interval 129 but could
  not satisfy the unchanged 0.05 m position gate. Its best bounded result was
  0.052198 m while static arm gravity remained 29.504716 Nm.
- Case 4 rejected at source interval 192 with 0.051691 m position error and
  29.503752 Nm static arm gravity.
- Case 7 preserved all 663 source anchors, exact source path length
  3.8081273358 m, and exact geometry/timestamps through a 1,763-state
  executable retarget. It is still rejected because the physical gimbal yaw
  branch jumps 179.994 degrees near interval 1578 at the physical +/-180 degree
  joint limit.

The branch-aware correction now enforces a physical gimbal margin throughout
semantic retargeting and in the final balanced-pitch IK. This converts the late
interpolation failure into a fail-closed solver decision. A bounded first run
rejected at source interval 569; proactive gimbal centering produced a distinct
branch but rejected at interval 557 with minimum margin 0.004721766 versus the
required 0.005. At that point position error was 0.006293 m, attitude error
0.005741 degrees, gravity 16.808017 Nm, and equilibrium pitch 0.242669 degrees,
so the physical joint-limit margin is the isolated blocker. The margin will not
be weakened merely to obtain a pass. A completed upstream multi-branch ep7 seed
is the next justified comparison.

The bounded case-1 upstream-seed A/B then closed the old forward-solver
blocker without changing any output gate:

- 256/256 immutable source anchors and exact timestamps were preserved;
- the exact source, mapped-anchor, and execution-target path lengths are all
  2.4522479583 m;
- 1,183 executable states and 1,182 transitions were generated, with
  initialization kept separate;
- position p95/max are 0.049322/0.049983 m, arm gravity is 29.504694 Nm,
  equilibrium pitch is 7.86576 degrees, and physical gimbal IK/interpolation
  max error is 0.099859 degrees;
- candidate SHA-256 is
  `461c14f018032c296b16bebb39b1b123a6ae747886352e68329a9a82e3dac070`.

Independent sealing and audit confirm integrity and offline executable quality.
The candidate is `valid_for_dynamic_evaluation=true` but remains
`valid_for_training=false`; dynamic execution has not yet passed and training
has not started. This A/B is evidence that a quality-qualified upstream
configuration branch can resolve a downstream nonholonomic solve boundary. It
does not make the upstream holonomic seed itself a policy teacher.

The first representative Isaac dynamic gate was safe but rejected. It ran all
16,368 bounded simulation steps with no termination and passed pitch, arm,
position, attitude, gimbal, action, and effort limits. However, the phase
governor stalled in acquisition at phase time 14.235399 s of 23.5 s, so neither
acquisition nor the semantic reference completed. The dynamic candidate remains
unapproved and `valid_for_training=false`. This is a Gate-3 acquisition/control
completion blocker, not an offline source-integrity regression.

Subsequent bounded diagnosis established that acquisition was unnecessarily
using the longer forward-only chassis route. The pose-equivalent route planner
now compares signed forward and reverse approaches and minimizes total yaw
travel without changing linear/yaw limits. For case 1 it selects reverse and
reduces planned yaw travel to 133.844064 degrees. The regenerated candidate:

- preserves the same 256 source anchors, source timestamps, source positions,
  source attitudes, and exact 2.4522479583 m source path;
- has 1,118 states / 1,117 transitions and 40.461261 s execution duration;
- passes position, gravity, pitch, physical-gimbal IK/rate/margin/interpolation,
  and exact-source gates;
- embeds the sealed route contract and schedule metadata in the NPZ itself;
- has execution-plan SHA-256
  `70c8e1abca453f67c771a33640a26309e57858cf28170fffd73019d4fbb41bc6`;
- remains `valid_for_training=false`.

Playback experiments that attempted stationary rotate/drive recovery were
rejected because they caused large chassis/camera excursions. The acquisition
task-space-arm v7 run is computationally inconclusive: it reached the explicit
900 s wall timeout while a separate riser canary owned the shared GPU and wrote
no result JSON. It is neither a dynamic pass nor a dynamic failure. The same v7
command may be rerun only as a single-GPU job against the sealed reverse-route
candidate.

That exclusive-GPU rerun physically passed from clean commit `d8ca698`. It used
the sealed reverse execution plan `70c8e1ab...`, acquisition-plus-semantic
bounded arm feedback, camera attitude gain 1.0, and a 25 degree bounded gimbal
feedback envelope. It completed acquisition and the full 40.461261 s reference
in 54.68 s wall time with no termination. Position p95/max were
0.040045/0.089092 m, camera attitude p95/max were 6.47546/7.51440 degrees,
peak pitch was 10.92131 degrees, and every arm/gimbal/action saturation and IK
gate passed. Its evidence schema nevertheless conflated the immutable
4.634756 s source clock with the 40.461261 s execution schedule and did not
embed the sealed plan identity and acquisition-route provenance. The physical
metrics are therefore provisional rather than a final dynamic admission. A
fresh rerun must preserve the same controller and gates while fail-closed
runtime admission validates and emits both clocks, execution-plan SHA-256,
schedule seal, and route metadata. `valid_for_training=false` and training has
not started.

That corrected rerun is now complete from clean commit `3e820e8`. The runtime
first verifies the exact NPZ SHA-256 and sealed route/schedule metadata, then
emits the immutable 4.634756 s source clock separately from the 40.461261 s
execution clock. An initial schema-v4 attempt reproduced the physical pass but
overlapped a later riser process and is retained only as provisional shared-GPU
evidence. The final-exclusive namespace was monitored every two seconds from
05:50:36 to 05:57:18, produced 201 ownership samples, and contained no foreign
playback command. It reproduced the same metrics and all 15 dynamic checks
passed. The final JSON, console, and ownership-log SHA-256 values are sealed in
the evidence audit. Case 1 therefore passes the representative dynamic gate,
but remains `valid_for_training=false`; this single case does not admit BC,
PPO, residual capture, or a corpus-wide teacher.

Case 7 therefore proves the corrected source-integrity path, not executable
teacher quality. Its gimbal failure requires a structural base/arm/gimbal
branch-selection correction; wrapping the exported physical joint angle would
hide a real actuator-limit crossing and is not allowed.

## Current status and next admission

- Active case-79 and full-corpus experiments were stopped without deleting
  evidence.
- The old source package and every known derived local/remote corpus are
  quarantined non-destructively.
- The loader, retargeter, assembler, and dynamic runner fail closed unless the
  exact-source and teacher-quality contract is present.
- PPO and BC remain blocked.
- Gate-1 exact-source retargeting is active only as bounded solver diagnosis.
  Case 1 now passes exact-source integrity and offline executable quality with
  its verified upstream solver seed. Case 4 still fails the position/gravity
  boundary and case 7 still fails physical gimbal branch continuity.
- Upstream ep1 and ep77 holonomic seeds may be used only as episode-specific
  solver priors after exact hash verification. They are not policy teachers and
  do not relax nonholonomic, source, gravity, timing, or output gates.
- Case 1 now passes exact-source offline and final-exclusive representative
  dynamic gates under the corrected schema-v4 evidence contract. The
  upstream ep7 branch search is closed without an admitted package: it
  crossed earlier branch failures but remained rate-disconnected at waypoint
  187. Ep7 therefore requires downstream `M>N` execution retiming with all 663
  source anchors, original source times, and complete ordered mapping retained.
  Gate-2 all-79 regeneration, a full dynamic corpus, BC, and PPO remain blocked.

The first downstream ep7 reserve-aware schedule canary crossed the upstream
branch failures at source waypoints 144, 158, 162, 167, and 187, then reached
source interval 552/662 under the unchanged local pose/rate gates. It hit the
explicit 1,800 s wall timeout because a gate-feasible but reserve-low interval
searched execution scales through 96 before selecting scale 1. No candidate or
result JSON was exported, so the run is computationally inconclusive rather
than an admission pass/failure. The next CPU fix bounds reserve-only search;
segments that are genuinely admission-infeasible retain the full execution
scale ladder. No second canary, playback, or learning stage has been started.

The authorized v4 CPU rerun used clean commit `385a8b8`, the same authoritative
manifest `f265aa1b...`, the same 1,800 s wall bound, and unchanged hard gates.
The reserve-only scale cap reduced the interval-552 search enough to complete
that interval, but the run timed out before interval 553 completed. Exit code
was 124 and console SHA-256 is `1ceeca34740f...`. The output namespace contains
no NPZ or result JSON, so no 663-anchor, source-time, ordered-map, physical
margin, interpolation, gravity, pitch, rate, or quality admission claim can be
made. V4 computationally supersedes v3 only as a search diagnostic; neither is
physically classified or valid for training. No Isaac playback or learning
stage was started.

Final-exclusive schema-v4 case-1 dynamic evidence is stored in:

```text
docs/03_training/two_wheel_balance/evidence_20260717_exact_source_ep1_reverse_dynamic_v7_schema_v4_final_exclusive
```

The older `evidence_20260717_exact_source_ep1_reverse_dynamic_v7` namespace is
historical/provisional evidence and must not be used as the final admission.

Machine-readable quarantine evidence is stored in:

```text
docs/03_training/two_wheel_balance/evidence_20260717_upstream_source_quarantine/quarantine.json
```

Independent case-1 audit evidence is stored in:

```text
docs/03_training/two_wheel_balance/evidence_20260717_exact_source_ep1_upstream_seed_independent_audit
```
