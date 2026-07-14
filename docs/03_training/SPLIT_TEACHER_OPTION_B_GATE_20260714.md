# Split Teacher Option-B Gate (2026-07-14)

## Decision

The corrected split-teacher runtime is implemented and its teacher replay gate
passes. The current BC policy is not ready for PPO or deployment because it
still accumulates closed-loop tracking error. One bounded DAgger correction was
tested and rejected; do not promote that checkpoint.

## Frozen Contract

- Action contract: `split_base_arm_attitude_v1` (9D for SB3 compatibility).
- Learned rows: physical arm `0:3` and base `6:9`.
- Reserved rows: `3:6`; policy values are ignored and zeroed in action history.
- Attitude target: semantic DFR attitude converted once with
  `R_world_cam = R_world_DFR * Rz(+pi/2)`.
- Observation/reward frame: physical `cam_link` FK.
- Physical gimbal: runtime damped-least-squares attitude adapter using the live
  world-frame `cam_link` angular Jacobian and physical joint limits.
- Legacy teacher NPZ files remain quarantined.

## Corrected Data

Source export processed 79 no-obstacle episodes: 62 accepted, 17 rejected, and
zero export errors. Only accepted rows are admitted to primary BC. Every source
trajectory is at least 5 seconds long.

Control-rate dataset on `.98`:

`artifacts/split_teacher_obs_all79_accepted_20260714/obs_dataset_split_teacher_accepted62_control20hz.npz`

- 62 trajectory groups, 21,017 rows, 84D observations, 9D actions.
- Exact ownership mask: `[1,1,1,0,0,0,1,1,1]`.
- Retimed synchronously at 0.05 seconds using original trajectory durations.
- Option-B and physical FK replay are validated during dataset construction.

## Gate Results

| Gate | Position mean | Position p95 | Orientation mean | Decision |
|---|---:|---:|---:|---|
| Exact teacher replay, episode 1, 120 steps | 0.03070 m | 0.04660 m | 0.958 deg | Runtime contract passes |
| Primary grouped BC, episode 1, 120 steps | 0.17373 m | 0.26433 m | 4.362 deg | Closed-loop drift; no PPO |
| DAgger round 1, episode 1, 120 steps | 0.13309 m | 0.27879 m | 11.593 deg | Rejected |

Primary BC trajectory-disjoint holdout RMSE was `0.02932`. The DAgger round-1
checkpoint changed it to `0.03045`, so held-out performance also regressed.

## Bounded DAgger Experiment

The evaluator captured 120 actual BC states with row-level environment,
episode, waypoint, and first-episode identity. Steps 20 through 119 were
relabelled with exact progress-aligned accepted-teacher arm/base actions. The
100 corrections were repeated ten times in a 22,017-row merged dataset and the
existing grouped actor was warm-started for eight epochs at `1e-4`.

The update reduced mean position error but worsened tail position error,
camera-attitude tracking, reward, and held-out RMSE. This is not a successful
policy update. Do not chain another DAgger round from it.

Evidence on `.98`:

- `artifacts/split_teacher_bc_smoke_20260714/dagger_round1_rollout_episode1_120.npz`
- `artifacts/split_teacher_bc_smoke_20260714/dagger_round1_correction_episode1_steps20_119.npz`
- `artifacts/split_teacher_bc_smoke_20260714/holdout_control20hz_dagger1.json`
- `artifacts/split_teacher_bc_smoke_20260714/isaac_dls_bc_episode1_control20hz_dagger1_120steps.json`
- Rejected checkpoint: `artifacts/split_teacher_bc_smoke_20260714/policy_grouped_accepted62_control20hz_dagger1.zip`

## Next Allowed Step

Do not resume PPO. The next experiment must address the structural coupling
shown by this gate: arm/base relabeling improved average position while camera
attitude degraded. Evaluate a reference-conditioned policy with explicit
trajectory progress, target velocity, and future physical-camera attitude, or
use a coupled corrective teacher that labels arm/base states while enforcing
the camera-attitude objective. Keep the next run bounded and compare against
both the exact-teacher replay and the original primary BC checkpoint.
