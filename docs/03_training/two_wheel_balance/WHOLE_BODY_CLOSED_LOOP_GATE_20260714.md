# Two-Wheel Whole-Body Closed-Loop Gate (2026-07-14)

## Decision

The no-obstacle representative dynamic gate passes for cases 1, 20, 28, 50,
and 79. This promotes a deterministic whole-body playback controller, not a
learned policy and not a hardware-ready controller.

PPO remains blocked. The next breadth gate is dynamic playback of all 79
retargeted references with the frozen settings below. Obstacle avoidance remains
after that no-obstacle gate.

## Promoted controller

- Frozen `structural_robust_v1` wheel LQR at 200 Hz.
- Bounded chassis pose feedback around retargeted `v/wz` feed-forward.
- Damped-least-squares world-position feedback on the three physical arm joints.
- Arm target slew and URDF joint-limit enforcement.
- Live rigid-body COM equilibrium-pitch feed-forward, clipped by the existing
  balance-controller safety limit.
- Full URDF-tree arm gravity feed-forward, including the physical DJI/camera
  branch beyond semantic `ee1_tool`.
- Arm implicit-drive stiffness/damping `400/40`, with the authored `30 Nm` hard
  effort limit unchanged.
- Phase retiming disabled by default and retained only as an explicit diagnostic.

The direct elbow smoke proved that gravity feed-forward is additive in Isaac:
at the softer `200/20` diagnostic setting, a requested `-0.2 rad` motion achieved
`-0.2125 rad` with a `4.223 deg` peak pitch. The previous no-feed-forward
`400/40` smoke achieved only about 68% of the requested motion.

## Representative results

All cases completed the original source duration with no termination, no wheel
action saturation, and no arm effort saturation.

| Case | Duration (s) | Pitch peak (deg) | Arm error peak (deg) | Arm torque peak (Nm) | Tool p95 (m) | Tool max (m) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 25.124 | 4.180 | 2.012 | 22.620 | 0.0389 | 0.0622 |
| 20 | 50.836 | 5.677 | 2.633 | 23.244 | 0.0701 | 0.0801 |
| 28 | 37.194 | 9.756 | 4.965 | 25.162 | 0.0775 | 0.0890 |
| 50 | 38.017 | 9.343 | 3.891 | 25.547 | 0.0776 | 0.0925 |
| 79 | 17.436 | 4.239 | 2.171 | 23.009 | 0.0375 | 0.0581 |

Acceptance limits were `12 deg` pitch, `10 deg` arm servo error, `0.15 m` tool
p95, `0.25 m` tool maximum, and `0.20` saturation ratio for both wheel action
and arm effort.

## Why the prior playback failed

The first dynamic runner replayed planned arm joints open loop and assumed the
base followed its kinematic path exactly. In physics, arm/camera gravity changed
the balance equilibrium, the wheel controller moved the base to reject that
disturbance, and the arm position drive had to build large position error before
producing gravity torque. The result was a balanced robot with poor tool
tracking and arm torque saturation.

Task-space feedback fixed geometric drift. Live COM feed-forward fixed the
moving balance equilibrium. Explicit full-tree gravity feed-forward removed the
arm's steady gravity burden without raising the 30 Nm limit. The `400/40` drive
then improved dynamic target realization without saturation.

## Reproduction

Run from `/mnt/g/wSpace/cinebotRL-two-wheel-balance` on `.98`:

```bash
env PYTHONPATH='G:\wSpace\cinebotRL-two-wheel-balance\src' \
/mnt/g/isaaclab_venv/Scripts/python.exe \
  'G:\wSpace\cinebotRL-two-wheel-balance\scripts\two_wheel_balance\smoke_all79_whole_body_playback.py' \
  --gains 'G:\wSpace\cinebotRL-two-wheel-balance\docs\03_training\two_wheel_balance\evidence_20260714_28kg\lqr_gains.json' \
  --retarget-dir 'G:\wSpace\cinebotRL-two-wheel-balance\evaluation_results\two_wheel_all79_playback\retargeted' \
  --urdf 'G:\wSpace\cinebotRL-two-wheel-balance\assets_own\recomoProto2_two_wheel_whole_body\recomoProto2_two_wheel_whole_body.urdf' \
  --cases 1,20,28,50,79 \
  --output 'G:\wSpace\cinebotRL-two-wheel-balance\evaluation_results\two_wheel_all79_playback\representative_promoted_defaults.json' \
  --headless
```

The full unabridged traces remain under
`evaluation_results/two_wheel_all79_playback/`. The compact committed evidence is
`evidence_20260714_whole_body_closed_loop/summary.json`.

## Stop rules and next gate

1. Do not resume PPO from any old checkpoint.
2. Freeze the promoted scripted controller while running all 79 no-obstacle
   references dynamically.
3. Stop and diagnose by failure family if any case violates balance, contact,
   completion, tracking, or saturation limits. Do not hide failures by relaxing
   thresholds or enabling phase retiming globally.
4. Only after all-79 no-obstacle acceptance, add obstacle sensing/planning and a
   bounded avoidance residual or arbitration layer.
5. Hardware readiness still requires measured COM, inertia, wheel torque,
   friction, delay, and a separate sim-to-real acceptance gate.

The resumable breadth gate is `scripts/two_wheel_balance/run_all79_dynamic_gate.sh`.
It preflights all 79 retargeted NPZs, validates each result JSON independently,
stops on the first failed case, and writes `summary.json` plus `COMPLETE` only
after all cases pass. Windows/WSL process status alone is not accepted as proof.

Case 7 later exposed an acquisition-prefix tracking failure and passed after a
case-local `1.25x` acquisition retime. See
`ALL79_DYNAMIC_CASE7_RECOVERY_20260715.md`; no controller or semantic-path timing
was changed.
