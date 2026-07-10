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
