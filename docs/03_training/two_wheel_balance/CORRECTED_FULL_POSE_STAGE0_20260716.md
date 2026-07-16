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

## Exact-source canaries

The `.98` transport canaries are at:

```text
/mnt/g/wSpace/cinebotRL/data/gikWBC9DOF_exact_source_teacher_integrity_canaries_20260716
```

Cases 1, 4, and 7 prove the loader/transport contract: exact counts, `N-1`
actions, zero timestamp error, and near-zero source pose error. They are
intentionally `valid_for_training=false` because solver/envelope/trajectory
quality has not been approved. They must never be retargeted as training data.

## Current status and next admission

- Active case-79 and full-corpus experiments were stopped without deleting
  evidence.
- The old source package and every known derived local/remote corpus are
  quarantined non-destructively.
- The loader, retargeter, assembler, and dynamic runner fail closed unless the
  exact-source and teacher-quality contract is present.
- PPO and BC remain blocked.
- Next action is to receive a small quality-qualified `exact_source_v1` teacher
  smoke, validate cases 1/4/7 end to end, then regenerate a bounded pilot. A
  full accepted corpus is not regenerated until that pilot passes.

Machine-readable quarantine evidence is stored in:

```text
docs/03_training/two_wheel_balance/evidence_20260717_upstream_source_quarantine/quarantine.json
```
