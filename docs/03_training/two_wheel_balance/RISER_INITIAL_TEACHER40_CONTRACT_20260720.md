# Riser initial teacher-40 contract

Date: 2026-07-20

## Decision

The first policy-initialization experiment no longer waits for 70 dynamically
qualified trajectories. A fresh, homogeneous corpus of at least 40 qualified
trajectories is sufficient to start bounded behavior cloning (BC).

This changes the learning-entry count only. It does not relax source integrity,
trajectory geometry, physical dynamic gates, safety limits, camera-frame
semantics, action clipping, command reconstruction, or holdout requirements.
The 70-case milestone becomes a later coverage target, and 79/79 remains the
final portfolio-evaluation goal.

## Available selection set

The demonstrated dynamic union contains 42 cases:

```text
[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,
 28,30,31,32,33,34,36,37,41,52,53,66,67,68,70,74,77]
```

These are teacher-selection candidates, not an existing training corpus. The
historical Gate-C runs intentionally wrote no dataset and applied no residual.
The old seven-case dataset remains quarantined because its upstream lineage is
superseded.

All 42 candidates are physical dynamic passes. Under the provisional residual
scale `[0.30, 0.40, 0.10]`, only 39 also pass label-envelope admission. Cases
`10`, `28`, and `70` are physical passes with first-channel overflows. They must
not be clipped or silently discarded. A fresh capture must preserve raw teacher
commands, then recompute and freeze the initialization action scales before
normalization.

## Initialization admission

BC initialization requires all of the following:

1. At least 40 cases selected by exact plan hash from the sealed v16 portfolio
   and dynamically qualified evidence.
2. A fresh single-code-contract capture under the current robot, controller,
   observation, camera-frame, and execution-clock schemas.
3. Per-case physical dynamic, thermal, runtime, integrity, and no-termination
   passes during capture.
4. Semantic DFR/camera-attitude observations only; physical DJI gimbal joint
   values remain internal adapter state and are not learned actions.
5. Raw residual labels are audited before scales are frozen; clipping is zero
   and teacher-command reconstruction error is at most `2e-6`.
6. A case-disjoint split with at least 30 train, 5 validation, and 5 untouched
   holdout cases. If all 42 recapture cleanly, use the extra two for coverage
   rather than weakening holdout isolation.
7. BC is an initialization experiment only. PPO and learned online rollout
   remain closed until offline validation passes and a separate rollout route
   is authorized.

## Controller boundary

The learned policy remains above the inner loops. Deterministic LQR owns
two-wheel balance; deterministic supervisors own collision enforcement,
actuator/riser/gimbal limits, emergency handling, and final command clipping.
The initial learned action remains a bounded residual over base linear speed,
base yaw rate, and riser target. It does not command wheel torque or physical
gimbal joints.

## Exact next task

Run `audit_riser_initial_teacher_candidates.py` against the v16 portfolio and
the sealed Gate-C evidence. Seal the resulting 42-case selection manifest. Then
implement raw-command capture that is independent of the provisional action
scale, test it on a small accepted canary, and only afterward schedule the
fresh 40-plus-case capture. Do not start BC or PPO from historical gate traces.
