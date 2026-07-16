# Executed residual dataset smoke

This evidence establishes the first DNN-compatible data contract for the
arm-free two-wheel riser robot. It is a pipeline smoke over the seven repaired
trajectory families, not an all-79 training dataset and not PPO authorization.

## Control boundary

The learned layer predicts three normalized high-level residuals:

1. longitudinal command correction, scale `0.20 m/s`;
2. yaw-rate command correction, scale `0.40 rad/s`;
3. riser target increment, scale `0.10 m`.

The frozen cascaded LQR still converts the first two commands to wheel effort.
The riser target remains travel-clamped, and semantic DJI attitude realization
remains deterministic. The policy therefore cannot replace the balance or
camera-attitude safety layers.

## Capture gate

- Isaac cases: 18, 23, 24, 41, 50, 72, and 79.
- Dynamic replay: `7/7` passed, 39,430 dense pre-action rows.
- Observation dimension: 26; action dimension: 3.
- Source GIK actions used: no.
- Physical gimbal labels used as actions: no.
- Non-finite values: none.
- Action clipping ratio: zero on all channels.
- Absolute action maxima: `0.754922`, `0.594813`, `0.121189`.
- Teacher-command reconstruction maximum error: `1.1921e-7`.
- Case split: train `[24, 41, 50, 72, 79]`, validation `[23]`, holdout `[18]`.
- Trajectory leakage: none.

The merged dataset remains in the `.98` artifact tree rather than Git:

`artifacts/two_wheel_riser/20260716_residual_dataset_smoke_v1/repaired7_residual_dataset_v1.npz`

Its size is `4,610,317` bytes and SHA-256 is
`ea421303faa5b94eaf8c6933e8fa9c7616999c67c169b39ab701d3ae5998523f`.
The complete dynamic gate SHA-256 is
`0220b5863d76492340b53979ac1b3d4596c739fefb0294c477164fb5d61219a6`.

## Stop rule

Do not train or roll out a learned candidate from only these seven cases. Export
all 79 playback plans, collect all 79 passed Isaac executions, rerun the
case-disjoint dataset audit, and only then permit an offline BC candidate.
