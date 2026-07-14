# Robust Tracking Closure Diagnosis

Date: 2026-07-14

## Objective

The requested next gate was to improve strict provisional tracking from
`91/112` to at least `106/112` while preserving all `112/112` balance
recoveries and keeping peak pitch below `12 deg`. PPO remained blocked.

## Asset repair

The URDF importer created reference arcs for `arm_mount_link/visuals` and
`upper_imu_link/visuals` even though those frame-only links intentionally have
no visual geometry. The conversion script now removes only those invalid arcs
after import. The regenerated base USD passes every existing asset check:

- `28.0000003 kg` authored mass;
- one floating articulation root and seven explicitly-massed rigid bodies;
- two `+Y` revolute wheel joints with zero stiffness and damping;
- `0.620 m` track and `0.2032 m` wheel diameter;
- no unresolved visual-reference warning when the repaired stage is opened.

The post-repair nominal `36`-scenario run has exactly the same aggregate and
per-scenario metrics as the accepted baseline. Asset composition changed;
physics behavior did not.

## Bounded controller sweep

Reference slew support was added to the controller state and evaluator. It is
disabled by default and therefore does not change the accepted controller.
Seven representative candidates were evaluated over the full signed
`112`-scenario provisional matrix.

| Candidate | Strict pass | Balance recovery | Peak pitch | Decision |
| --- | ---: | ---: | ---: | --- |
| Accepted baseline | 91/112 | 112/112 | 11.47 deg | Retain |
| `vx_ki=0.06` | 93/112 | 112/112 | 11.68 deg | Reject: insufficient |
| `vx_ki=0.075` | 95/112 | 111/112 | 11.99 deg | Reject: balance regression |
| `vx_ki=0.075`, slew `0.4/0.8` | 96/112 | 112/112 | 12.00 deg | Reject: below target and over limit |
| `vx_ki=0.10`, 5.5 deg cap, slew `0.2/0.8` | 98/112 | 110/112 | 12.46 deg | Reject: balance regression |
| `vx_ki=0.075`, stronger yaw, slew | 97/112 | 112/112 | 12.45 deg | Reject: tilt regression |
| pitch-rate gain `1.3x` | 96/112 | 110/112 | 12.80 deg | Reject: balance regression |
| pitch-rate gain `0.7x` | 0/112 | 112/112 | 13.19 deg | Reject: global regression |

The stronger longitudinal loop repairs low-COM tracking and lowers aggregate
velocity RMSE, while stronger yaw terms repair much of the corner yaw error.
Neither closes the adverse COM-x cases without consuming the pitch safety
margin. Pitch-rate coefficient mutation does not solve that coupling.

## Decision and next structural step

No gain candidate is promoted. Defaults remain `vx_ki=0.05`, `wz_ki=0.10`,
`wz_feedforward=0.60`, no reference slew, and the accepted inner LQR.

The next controller change must be structural rather than another scalar gain
sweep. The recommended bounded option is a transparent path-progress governor:
estimate the equilibrium pitch bias during startup, retime trajectory progress
when requested chassis motion reinforces that bias, and report requested versus
admitted speed separately. This preserves spatial end-effector tracking by
slowing progress instead of hiding a velocity miss. A gain-scheduled LQI/MPC
alternative should wait for measured COM/inertia because its model bank would
otherwise encode the same guesses more deeply.

Arm/end-effector integration and obstacle avoidance remain blocked until the
governor has a separate acceptance contract and the chassis safety gate is
reaccepted. No PPO checkpoint is produced or promoted by this work.
