# Split Reference Observation V2 Gate (2026-07-14)

## Decision

`split_reference_v2` is implemented and its offline/runtime parity gate passes.
The first v2 BC checkpoint is rejected because closed-loop tracking regressed.
Do not promote it, continue BC on nominal trajectories, or resume PPO.

## Observation Contract

The contract is opt-in and leaves every legacy 84D checkpoint unchanged. It
adds 14 fields to produce a 98D no-obstacle observation:

- normalized trajectory progress: 1D;
- remaining trajectory time divided by 30 seconds: 1D;
- normalized world-frame target linear velocity: 3D;
- three future physical-camera attitude errors in axis-angle form: 9D.

The existing three future positions remain present. Their offline lookahead
stride is now derived from `lookahead_dt / trajectory_dt`: the 20 Hz teacher
uses a two-row stride to match Isaac's 0.10-second lookahead interval.

## Parity Gate

Dataset:

`artifacts/split_teacher_obs_all79_accepted_20260714/obs_dataset_split_teacher_accepted62_reference_v2.npz`

Stage:

`trajectoryToLearn/stage_gik_split_teacher_accepted62_reference_v2`

The dataset contains 62 accepted trajectories, 21,017 rows, 98D observations,
and the unchanged learned-action mask `[1,1,1,0,0,0,1,1,1]`.

Three Isaac observations were matched by actual waypoint index against episode
1 teacher rows. The formal parity artifact passed:

- future-position maximum absolute error: `0.0`;
- progress/time/target-velocity maximum absolute error: `5.96e-8`;
- initial future-attitude L2 error: `7.28e-4`.

Later future-attitude differences are expected during rollout because the
feature is expressed relative to the live physical camera rather than copied
from the teacher state.

Evidence:

`artifacts/split_teacher_bc_smoke_20260714/reference_v2_parity_comparison.json`

## Bounded BC Gate

The grouped 84D actor was transferred into the 98D actor with an explicit
first-layer mapping:

- legacy state and position-lookahead columns `0:65` remain in place;
- legacy action history `65:83` moves to `79:97`;
- legacy contact `83:84` moves to `97:98`;
- new reference columns `65:79` start with zero input weights.

The actor was trained once for ten epochs at `1e-4`. No PPO ran.

Offline trajectory-disjoint holdout RMSE improved from `0.02932` to `0.01588`.
This did not transfer to closed-loop execution:

| Policy | Position mean | Position p95 | Orientation mean | Reward mean |
|---|---:|---:|---:|---:|
| Primary 84D BC | 0.17373 m | 0.26433 m | 4.362 deg | 38.80 |
| Reference-v2 BC | 0.40597 m | 0.60182 m | 8.654 deg | 17.75 |

Rejected checkpoint:

`artifacts/split_teacher_bc_smoke_20260714/policy_grouped_accepted62_reference_v2.zip`

## Failure Diagnosis

The v2 actor is initially closer to the teacher than the old actor, so the
weight mapping is not the failure. On the first two parity states, learned-row
action RMSE was `0.061` and `0.021`, versus `0.125` and `0.054` for the old
actor.

The failure is off-teacher covariate shift:

- teacher future-attitude maximum-norm p99: `0.0312 rad`;
- rejected rollout future-attitude mean: `0.1489 rad`;
- rejected rollout future-attitude p95: `0.3150 rad`;
- rollout samples above teacher p99: `95.8%`;
- first teacher-p99 exceedance: step 5;
- rollout policy/teacher action RMSE mean: `0.3316`.

Evidence:

`artifacts/split_teacher_bc_smoke_20260714/reference_v2_covariate_shift_diagnosis.json`

## Next Data Requirement

Nominal GIK trajectories only describe states close to the desired camera path.
They cannot teach recovery after arm/base deviations. The next teacher export
must therefore provide coupled corrective arm/base labels from perturbed or
policy-visited physical states while enforcing the same Option-B `cam_link`
attitude objective. The DJI physical joint labels remain diagnostic and must
not become policy actions.

Acceptance gates for that export:

1. Corrective states must cover future-attitude errors beyond `0.031 rad`, with
   an initial bounded target range of at least `0.0-0.20 rad`.
2. Every label must satisfy the existing arm/base envelope and physical-camera
   FK checks.
3. Train, validation, and holdout remain trajectory-disjoint by source episode.
4. One bounded BC gate must improve both position p95 and orientation mean over
   the primary 84D BC baseline before any PPO discussion.

The executable request schema, MATLAB CSV handoff, and stop rules are defined
in `docs/03_training/CORRECTIVE_GIK_TEACHER_REQUEST_20260714_CN.md`.
