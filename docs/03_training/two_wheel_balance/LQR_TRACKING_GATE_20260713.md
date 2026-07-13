# Cascaded LQR Chassis-Tracking Gate

Date: 2026-07-13

## Result

The first low-speed chassis-tracking layer passes on the corrected 8-inch nominal Isaac plant. The controller is scripted and deterministic; no PPO policy was trained or resumed.

| Metric | Selected result | Gate |
| --- | ---: | ---: |
| Scenario survival | 32/32 (100%) | at least 95% |
| Nonzero `vx` RMSE | 0.0743 m/s | at most 0.08 m/s |
| Nonzero `wz` RMSE | 0.1055 rad/s | at most 0.12 rad/s |
| Peak pitch | 6.096 deg | at most 10 deg |
| Peak wheel speed | 3.272 rad/s | recorded, no gate |
| Action saturation | 0% | at most 10% |

An identical selected-candidate repeat produced zero difference in every aggregate metric. The unchanged push regression also reproduced the accepted push-gate summary exactly.

## Command contract

Each 10-second scenario starts upright and receives:

```text
vx commands:          -0.2, 0, +0.2 m/s
wz commands:          -0.4, 0, +0.4 rad/s
first command:         1.0 to 4.0 s
zero interval:         4.0 to 5.0 s
reversed command:      5.0 to 8.0 s
final zero interval:   8.0 to 10.0 s
metric settling time:  0.5 s after each command transition
```

The 32 environments repeat all nine `vx/wz` combinations. Tracking metrics use nonzero command windows only, so zero-command scenarios cannot make the RMSE gates easier.

## Controller structure

The nominal balance LQR remains frozen at 50 Hz. A deployable outer loop uses wheel odometry and IMU yaw rate:

```text
vx estimate = wheel_radius * mean_wheel_velocity
vx error    -> bounded pitch reference
wz command  -> bounded feedforward + yaw-rate feedback
```

Selected outer-loop values:

```text
vx_kp:                    0.6 rad per (m/s)
wz_kp:                    0.25 action per (rad/s)
wz_feedforward:           0.6 action per (rad/s)
vx_ki, wz_ki:             0
wheel_difference_kp:      0
pitch-reference limit:    +/-6 deg
normalized action limit:  +/-0.8
```

Simulator base `vx` is evaluation truth only. It is not part of the controller observation or feedback path.

## Diagnostic lessons

- The first outer-loop pitch-reference sign was wrong. It gave `vx` response gain `-0.261`; stronger gains amplified motion in the wrong direction. The corrected forward command first pulls the wheels back to create a forward lean, then the frozen LQR catches the body.
- Yaw-rate P alone under-responded at low gain and oscillated at high gain.
- Added differential-wheel P and yaw integral candidates did not pass combined `vx+wz` tracking and were rejected.
- Bounded yaw command feedforward solved the sustained turn-authority deficit without integral windup. A narrow final sweep selected the lowest normalized gate score, not merely the first passing candidate.

## Regression and boundaries

- The selected repeat exactly matches the accepted tracking metrics.
- The original upright push gate remains unchanged and exactly reproducible: 32/32 recovery, 0.415 s worst recovery, 2.084 deg peak pitch, and 0.0382% saturation.
- Tracking and external push are not yet combined in the same rollout.
- The run used the `30.0 kg` PhysX-resolved plant, not the `26.0 kg` authored-mass sum or the approximately `40 kg` complete robot.
- Plant uncertainty, slope, rough terrain, arm motion, end-effector tracking, obstacles, sensor noise, delay, and hardware are not included.
- PPO remains blocked until scripted-controller robustness and the articulated model boundary are established.

Machine-readable accepted evidence is in `evidence_20260713_lqr_tracking/tracking_gate.json`.
