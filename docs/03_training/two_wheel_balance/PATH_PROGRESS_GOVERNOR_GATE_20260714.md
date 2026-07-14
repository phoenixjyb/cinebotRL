# Path Progress Governor Gate

Date: 2026-07-14

## Objective

The bounded experiment tested whether a transparent trajectory-progress
governor could close strict provisional tracking from `91/112` to at least
`106/112` without changing the accepted LQR gains, losing any balance
recoveries, exceeding `12 deg` tilt, or retaining less than `75%` path
progress. PPO remained blocked.

## Contract implemented

The controller estimates and latches its startup equilibrium pitch bias. When
the sign of requested longitudinal motion reinforces that bias, the governor
scales `vx` and `wz` by the same factor. Scaling both commands preserves the
requested planar path geometry while reducing progress rate.

- Governor is disabled by default.
- Scaling starts above `0.5 deg` equilibrium bias and reaches its floor above
  `2.5 deg`.
- Minimum admitted path-progress scale is `0.75`.
- Requested and admitted references and RMSE are reported separately.
- Recovery and pass/fail can explicitly select `requested` or `admitted`
  reference semantics.
- A separate minimum-progress gate prevents success by stopping the robot.
- Reference slew remains a separate, default-disabled mechanism.

The inner LQR and accepted outer-loop gains were not changed.

## Full provisional result

The complete signed matrix covered `vx=+/-0.2 m/s`, `wz=+/-0.4 rad/s`,
`+/-20 N` pushes for `0.1 s`, and all 14 provisional plant variations.

| Metric | Result | Requirement | Decision |
| --- | ---: | ---: | --- |
| Survival | 112/112 | 112/112 | Pass |
| Balance recovery | 112/112 | 112/112 | Pass |
| Strict admitted tracking | 99/112 | >=106/112 | **Fail** |
| Peak pitch | 9.23 deg | <12 deg | Pass |
| Minimum progress scale | 0.759 | >=0.75 | Pass |
| Requested `vx/wz` RMSE | 0.0904 / 0.0841 | Report only | Preserved |
| Admitted `vx/wz` RMSE | 0.0850 / 0.0734 | 0.10 / 0.15 | Pass |

The governor activated in 12 bias-reinforcing cases and all 12 passed. It
repaired eight strict failures relative to the accepted `91/112` baseline,
including the adverse COM-x and positive-velocity corner cases. This is useful
negative evidence: the sign-aware mechanism works in the cases it targets,
but it is not sufficient for the entire assumed plant envelope.

The 13 remaining failures were unaffected because their progress scale stayed
at `1.0`:

- eight `com_z_minus_0p03` cases missed the longitudinal error/recovery limit
  by a small but repeatable margin;
- four negative-velocity `corner_provisional_v1` cases missed yaw tracking;
- one positive-velocity `inertia_1p15` case recovered tracking in `2.05 s`,
  just outside the `2.0 s` limit.

Lowering the progress floor cannot repair failures where the governor never
activates, so no floor or gain sweep was run.

## Regression proof

The governor-off requested-reference nominal matrix was rerun over all 36
signed command and `+/-2/4/6 N s` push cases. All legacy aggregate fields are
bit-for-bit equal to the accepted post-asset baseline:

- 36/36 scenarios passed;
- peak pitch `10.91089153289795 deg`;
- `vx` RMSE `0.0818930801200202 m/s`;
- `wz` RMSE `0.05204797026867061 rad/s`;
- saturation ratio `0.0021308134148601073`.

This proves the new evaluator and controller state do not alter the accepted
default behavior.

## Decision and next structural step

Do not enable or promote the governor. Keep its reusable implementation and
requested/admitted evaluator contract default-disabled. Do not resume PPO and
do not run another scalar gain or governor-floor sweep.

The next bounded diagnosis should address the two non-bias failure families:
the low-COM longitudinal steady-state envelope and the signed corner yaw
coupling. A dynamic balance-margin scheduler or model-based controller is only
worth testing after its admission rule is specified independently of measured
tracking error; otherwise it can hide error by moving the target. Hardware COM,
inertia, wheel torque, friction, and delay measurements remain the preferred
way to determine whether these simulated corner cases belong in the real
operating envelope.

Arm/end-effector tracking and obstacle avoidance remain downstream of chassis
reacceptance. No policy was trained and no PPO checkpoint was produced.

Compact evidence is in
`evidence_20260714_path_progress_governor/summary.json`; the full generated
results remain under
`artifacts/two_wheel_balance/path_progress_governor/` on the evaluation host.
