# RS4 + Obstacle V2 Implementation Audit - 2026-07-10

## Outcome

The accepted GIK teachers now produce a trajectory-disjoint, fixed-clock
94-dimensional BC dataset for the experimental `rs4_attitude_rate_v1` action
contract. The offline dataset and BC gates pass. On 2026-07-11, the baseline
Isaac scene and the RS4/two-obstacle scene were restored and verified on the
RTX PRO 4000 Blackwell host. This removes the scene-health blocker, but does
not by itself promote the BC policy for PPO or deployment.

## Contract Changes

- Replaced ZYX Euler finite differences with local-frame quaternion angular
  velocity plus closed-loop quaternion residual feedback.
- Applied the same RS4 rate and acceleration limits used by the simulator
  adapter: yaw/pitch `90 deg/s`, roll disabled, fixed control period `0.05 s`.
- Progress-retimed accepted teachers to `0.1 s` and rebuilt arm labels with
  `teacher_wide_v1`.
- Dereferenced MATLAB HDF5 cell arrays for zero, one, or two obstacles.
- Added two body-frame obstacle slots, each containing
  `[dx, dy, radius, clearance, valid]`.
- Added source-grouped train/validation/holdout splitting.  No trajectory can
  contribute rows to more than one split.

## Dataset Gate

```text
data/gik_offline_accepted274_rs4_npz_v2_20260710/
data/gik_offline_accepted274_rs4_obs_v2_20260710/obs_dataset_rs4_v2.npz
```

```text
accepted manifest rows: 274
exported trajectories:   269
skipped (<5 s):          5
failures:                0
retimed samples:         54,139
observation dimension:   94
arm valid:               1.000 / 0.943 / 1.000
RS4 valid:               1.000 / 1.000 / 0.000 (roll intentionally masked)
base valid:              1.000 / 0.9999 / 0.99996
RS4 rate p95:            19.5 / 21.6 / 0.0 deg/s
RS4 rate max:            90 / 90 / 0 deg/s
```

Obstacle slot validation:

```text
no obstacle:       79 sources -> [0, 0] valid slots
one obstacle:      64 sources -> [1, 0] valid slots
two obstacles A:   64 sources -> [1, 1] valid slots
two obstacles B:   62 sources -> [1, 1] valid slots
```

## BC Smoke

```text
logs/bc/rs4_v2_grouped_smoke_20260710/bc_policy.zip
logs/bc/rs4_v2_grouped_smoke_20260710/bc_policy.split.json
artifacts/rs4_v2_grouped_smoke_20260710/holdout_metrics.json
```

```text
trajectory groups: 215 train / 27 validation / 27 holdout
rows:              43,261 train / 5,657 validation / 5,221 holdout
best val MSE:      0.000207
holdout MSE:       0.000349
holdout RMSE:      0.01869
```

The main no/one/two-obstacle holdout families are between `0.00014` and
`0.00023` masked MSE.  The single WB-MPC/SQP recovery holdout remains a clear
outlier at about `0.0154` MSE and must not be hidden inside the aggregate.

## Isaac Scene Health Restoration - 2026-07-11

The earlier apparent stop at the `/World/Ground/SphereLight` warning was not a
scene-construction failure. The host was first cleared of stale Isaac process
trees, and the gates were rerun with unbuffered Python output (`python.exe -u`)
so reset and rollout progress could be observed directly.

Legacy no-obstacle checkpoint gate:

```text
artifact: artifacts/baseline_scene_health_20260711.json
environments / steps: 1 / 1
environment created: yes
reset and physics step: yes
initial EE position error: 0.00549 m
episode terminations: 0
```

Experimental RS4/two-obstacle open-loop gate:

```text
artifact: artifacts/rs4_two_obstacle_scene_health_clear_20260711.json
environments / steps: 1 / 5
action contract: rs4_attitude_rate_v1
observation mode: relative_two_v2 (94D)
obstacle centers: y=0.80 m and y=1.25 m
initial/final minimum clearance: 0.250 / 0.249 m
obstacle collision / unsafe: 0% / 0%
episode terminations: 0
```

The older `y=0.50 m` deterministic fixture was rejected as a health test: with
a `0.35 m` robot footprint radius and `0.20 m` obstacle radius it begins with
`-0.05 m` signed clearance. That result indicates fixture overlap, not broken
physics.

## Remaining Gates

- Run a longer multi-source RS4 rollout; the five-step replay proves scene and
  action-contract execution only, not trajectory tracking quality.
- Keep the grouped holdout outlier visible when judging BC promotion.
- Validate DJI hardware frame/sign behavior before deployment.
- Start PPO only after a bounded closed-loop BC gate passes the tracking and
  safety thresholds; scene health alone is not sufficient.

## Closed-Loop BC Gate and Contract Audit - 2026-07-11

The first bounded closed-loop evaluation exposed an observation contract bug:
the 94D policy could only receive its two obstacle feature slots when physical
obstacles were spawned, and one-obstacle episodes marked both slots valid. The
environment and rollout gate now separate fixed slot capacity from active
obstacle count. The validated combinations are:

```text
nominal:       2 slots, 0 active, no obstacle prims, 94D observation
one obstacle: 2 slots, 1 active, second prim moved out of the scene
two obstacle: 2 slots, 2 active
```

After that fix, the grouped BC checkpoint was evaluated without PPO:

```text
checkpoint: logs/bc/rs4_v2_grouped_smoke_20260710/bc_policy.zip

no-obstacle, 8 envs x 80 steps:
  mean / p95 / max position error: 0.689 / 1.637 / 1.983 m
  final mean position error:       1.257 m
  first-episode terminated envs:   1 / 8

one-obstacle, 8 envs x 80 steps:
  mean / p95 / max position error: 0.718 / 1.512 / 1.610 m
  final mean position error:       1.506 m
  collision / unsafe:              0% / 3.09%
  first-episode terminated envs:   0 / 8
```

Decision: reject this checkpoint as a closed-loop promotion candidate and do
not start PPO from it.

### Teacher-distribution diagnosis

The early nominal rollout was exported and compared with the 79 no-obstacle
teacher sources using `scripts/imitation/diagnose_rollout_teacher_shift.py`.
The nearest-teacher standardized RMS distance was already `6.37` in the first
half and increased to `6.48` in the second half. The largest shifts include
quaternion/axis-angle error channels `41-47` and action-history channels.

The source data explains the mismatch:

- `actual_ee_pos` in the accepted teacher NPZ is all zeros, while URDF FK and
  Isaac agree on the physical `cam_link` position.
- `actual_ee_quat_wxyz` carries the GIK camera-attitude frame rather than the
  physical `cam_link` FK quaternion.
- The offline BC loss is therefore low in a synthetic observation space that
  does not match the closed-loop simulator state.

### Camera-frame alignment audit

`scripts/imitation/audit_rs4_camera_frame_alignment.py` tested whether a single
fixed quaternion transform could map GIK camera attitude to URDF `cam_link` on
32 accepted sources. Both left- and right-multiplied transforms fail:

```text
fixed-transform attitude residual mean / p95 / max:
  110.79 / 164.70 / 179.92 deg

URDF FK versus target position mean / p95 / max:
  0.0125 / 0.0365 / 0.0400 m
```

This is evidence that the GIK virtual attitude branch contributes a
trajectory-dependent rotation. It cannot be repaired by a sign flip or a
constant camera-frame offset.

### Required next stage

Replace the experimental simulator's direct `[yaw, pitch, roll]` to
`[joint3, joint2, joint1]` axis permutation with a bounded differential-IK
adapter. The adapter must map semantic camera angular velocity to physical
gimbal joint velocity through the live `cam_link` rotational Jacobian, respect
joint/rate limits, and expose residual/singularity diagnostics. Validate it on
one trajectory before regenerating the 94D teacher dataset or training again.

The initial Jacobian audit passes and removes one implementation risk:

```text
artifact: artifacts/rs4_gimbal_jacobian_initial_20260711.json
cam_link Jacobian body index: 19 (body offset 0)
gimbal Jacobian columns:      joint ids + 6 floating-base columns
singular values:              1.2601 / 1.0000 / 0.6420
condition number:             1.963
10 deg/s axis probe residual: 0.00005 / 0.00007 / 0.00017 rad/s
```

This proves differential IK is numerically viable near the initial pose. It
does not resolve which GIK camera-attitude frame should be tracked. Do not wire
the Jacobian controller into the environment until that target frame is made
explicit; otherwise the controller would accurately track the wrong frame.

## Option-B Frame Contract and Corrected Export Smoke - 2026-07-11

The GIK frame audit defines the semantic-to-physical target conversion as:

```text
R_world_cam = R_world_DFR * Rz(+pi/2)
```

CineBotRL now applies this conversion at the trajectory target boundary under
`semantic_dfr_to_physical_cam_v1`. Observations and rewards continue to use the
physical Isaac `cam_link`. The RS4 dataset builder also:

- rejects legacy NPZs without `q_selection_meta`;
- requires the 13D selection `[0,1,2,3,4,5,10,11,12]`;
- recomputes physical `cam_link` state through URDF FK;
- excludes ramp-prefix poses before treating targets as semantic DFR poses;
- blocks retiming until ramp-aware joint/target interpolation exists;
- records physical-gimbal solve residual and success for every row.

Three corrected source exports were used for a native-row smoke. The old
export tree is rejected before output creation. The corrected smoke is finite
and 94D, but it is not training eligible because only `52/84` post-ramp rows
can be reproduced by the three physical gimbal joints within `2 deg` while
holding the exported base/arm state fixed.

### Contradiction in the MATLAB physical-only evidence

The supplied GIK response reports 15/15 physical-gimbal-only solves, but a
direct replay of its first no-obstacle sample shows that the solver changes the
supposedly fixed base and arm joints:

```text
seed first six:
[0.7669, 0.1015, 0.06395, 0.07946, 0.44133, -0.89672]

returned first six:
[0.7669, 0.1015, -0.2211, -0.2056, 0.7631, -1.2185]
```

The returned pose translation also differs strongly from the target because
that audit solve is orientation-only. Therefore its `15/15` result does not
prove three-gimbal-only reachability with base/arm fixed. PPO and full export
remain blocked. The next teacher must either:

1. regenerate coordinated physical arm plus gimbal states under the option-B
   target contract; or
2. prove a corrected three-gimbal-only solve whose base/arm deltas are checked
   numerically and remain below tolerance.

The retained smoke is diagnostic-only:

```text
corrected input: artifacts/gik_corrected_frame_smoke_20260711/
option-B output: artifacts/gik_corrected_option_b_obs_smoke_20260711/
rows: 84 post-ramp native rows
physical-gimbal rows <= 2 deg: 52/84
training_eligible: false
incomplete gimbal rows: 32 (gimbal action mask cleared)
```

A five-step Isaac open-loop contract smoke writes
`artifacts/option_b_runtime_smoke_20260711.json`, selects
`semantic_dfr_to_physical_cam_v1`, and completes without termination. Its mean
orientation error is `174.38 deg`; this is expected negative evidence from an
uncoordinated source, not a tracking pass. No PPO or BC training was started.
