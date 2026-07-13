# Cascaded LQR Tracking-Plus-Push Gate

Date: 2026-07-13

Branch: `codex/two-wheel-balance-rl`

Result: **passed**

## Purpose

This gate checks whether the frozen cascaded LQR chassis controller can preserve
balance and resume velocity tracking when an upper-body impulse occurs during an
active command. It combines the previously separate low-speed tracking and
upright push gates. It does not train or evaluate a neural policy.

The tested Cartesian product contains 36 deterministic scenarios:

- longitudinal commands: `vx = -0.2, +0.2 m/s`
- yaw commands: `wz = -0.4, 0.0, +0.4 rad/s`
- signed upper-body impulses: `-6, -4, -2, +2, +4, +6 N s`
- impulse duration: `0.1 s`
- equivalent application height above the base: `0.5 m`
- controller rate: `50 Hz`, held into a `200 Hz` policy loop

Each command starts before the impulse and remains active through recovery. The
controller receives wheel odometry and body angular state; simulator root `vx`
and roll are used only for evaluation.

## Acceptance contract

A scenario passes only when all of the following hold:

- the episode survives the full horizon;
- balance and tracking recover within `2.0 s` after the impulse;
- absolute tilt remains below `12 deg`;
- late post-impulse `vx` RMSE is at most `0.10 m/s`;
- late post-impulse `wz` RMSE is at most `0.15 rad/s`;
- action saturation remains at or below `10%`;
- at least `95%` of all scenarios pass.

The balance recovery envelope is measured from each scenario's settled
pre-impulse operating state, with margins of `0.5 deg` tilt and `0.05 rad/s`
angular rate. A fixed return-to-vertical threshold is invalid during active
velocity tracking because the controller intentionally commands a non-zero body
pitch. The independent `12 deg` safety limit remains fixed.

## Accepted result

| Metric | Result | Limit |
| --- | ---: | ---: |
| Scenario success | `36/36` (`100%`) | `>=95%` |
| Survival | `36/36` (`100%`) | required per scenario |
| Balance recovery | `36/36` (`100%`) | required per scenario |
| Tracking recovery | `36/36` (`100%`) | required per scenario |
| Worst balance recovery | `0.81 s` | `<=2.0 s` |
| Worst tracking recovery | `1.07 s` | `<=2.0 s` |
| Peak pitch | `7.825 deg` | `<12 deg` |
| Aggregate post-impulse `vx` RMSE | `0.0711 m/s` | `<=0.10 m/s` |
| Aggregate post-impulse `wz` RMSE | `0.1139 rad/s` | `<=0.15 rad/s` |
| Action saturation | `0.1297%` | `<=10%` |

Every signed `2/4/6 N s` impulse group passed all six command combinations. The
accepted run was repeated with the same seed; aggregate summaries and all
recorded per-scenario metrics were exactly equal.

The controller was not retuned for this gate. It retains:

```text
inner LQR gain scale: 0.6
vx_kp:               0.6
wz_kp:               0.25
wz_feedforward:      0.6
pitch reference:     +/-6 deg
action limit:        +/-0.8
```

## Rendered check

The selected rendered case uses `vx=+0.2 m/s`, `wz=-0.4 rad/s`, and a `-6 N s`
impulse. It records all 600 policy steps at 50 FPS, producing a 4x slow-motion
clip. The rollout survived without termination or truncation and reached a peak
pitch of `6.440 deg`.

## Evidence

- `evidence_20260713_lqr_tracking_push/combined_gate.json`
- `evidence_20260713_lqr_tracking_push/combined_gate_repeat.json`
- `evidence_20260713_lqr_tracking_push/recording.json`
- `scripts/two_wheel_balance/evaluate_lqr_tracking_push.py`
- `scripts/two_wheel_balance/record_lqr_tracking_push.py`

## Scope boundary

This is a simulation-only result for the current simplified `26.0 kg` nominal
plant, not the approximately `40 kg` complete robot. It does not establish
robustness to mass, COM, inertia, wheel friction, motor torque, sensor noise,
control delay, slopes, arm motion, end-effector tracking, obstacles, or hardware.
No PPO was started.

The next gate is deterministic plant-parameter robustness around this frozen
controller. Arm and end-effector motion should be introduced only after that
gate identifies a credible stability envelope.
