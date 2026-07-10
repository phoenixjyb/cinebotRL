# RS4 + Obstacle V2 Implementation Audit - 2026-07-10

## Outcome

The accepted GIK teachers now produce a trajectory-disjoint, fixed-clock
94-dimensional BC dataset for the experimental `rs4_attitude_rate_v1` action
contract.  This artifact is suitable for offline BC experiments, but it is not
yet promoted for PPO or deployment because the live Isaac process currently
exits during baseline scene initialization before any rollout step.

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

## Remaining Gate

The experimental open-loop RS4 replay and a one-step legacy checkpoint both
exit at the same Isaac scene-initialization point, before environment creation
finishes.  Therefore:

- dataset and offline BC gates pass;
- live Isaac execution remains unproven;
- DJI hardware frame/sign behavior remains unproven;
- PPO must not start until baseline Isaac scene health is restored and the
  open-loop RS4 replay writes a complete gate JSON.
