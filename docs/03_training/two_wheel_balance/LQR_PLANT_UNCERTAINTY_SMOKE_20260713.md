# LQR Plant-Uncertainty Smoke

> **Superseded 2026-07-14:** this document describes the unintended 30 kg
> runtime plant and obsolete 40 kg stress assumption. Use
> `LQR_28KG_MODEL_AND_OUTER_LOOP_GATE_20260714.md` for the corrected 28 kg
> model, controller, and uncertainty boundary.

Date: 2026-07-13

Branch: `codex/two-wheel-balance-rl`

Result: **failed; controller tuning blocked by mass-contract mismatch**

## Purpose

This deterministic smoke checks the frozen cascaded LQR against one-factor and
combined plant changes while executing the accepted difficult case:

```text
vx command:       +0.2 m/s
wz command:       -0.4 rad/s
upper-body push:  -6 N s at 0.5 m equivalent height
controller:       accepted 50 Hz cascaded LQR, unchanged
```

No PPO or other policy training was started.

## Matrix

The 16 cases cover nominal behavior, total-mass scaling, base-link COM offsets,
inertia scaling, robot-collider friction, reduced torque, command delay, and one
combined hard corner. The 40 kg cases calculate their scale from the mass
resolved by the live articulation rather than assuming an authored mass.

| Variation | Result | Main observation |
| --- | --- | --- |
| Nominal | Pass | `wz` RMSE `0.1419 rad/s` |
| Mass `0.85x` | Pass | Tracking recovered in `0.41 s` |
| Mass `1.15x` | Fail | `wz` RMSE `0.2204 rad/s` |
| Uniform total mass `40 kg` | Fail | `wz` RMSE `0.2551 rad/s` |
| COM X `-0.03 m` | Fail | `vx` RMSE `0.1622 m/s` |
| COM X `+0.03 m` | Fail | `wz` RMSE `0.2065 rad/s` |
| COM Z `-0.05 m` | Pass | Tracking recovered in `0.46 s` |
| COM Z `+0.05 m` | Fail | `wz` RMSE `0.1759 rad/s` |
| Inertia `0.8x` | Pass | Tracking recovered in `0.30 s` |
| Inertia `1.2x` | Fail | `wz` RMSE `0.1780 rad/s` |
| Low friction `0.65/0.55` | Pass | Tracking recovered in `1.035 s` |
| High friction `1.10/1.00` | Fail | `wz` RMSE `0.1651 rad/s` |
| Torque `0.8x` | Fail | `wz` RMSE `0.2440 rad/s` |
| Delay `10 ms` | Pass | Tracking recovered in `0.93 s` |
| Delay `20 ms` | Pass | Tracking recovered in `0.945 s` |
| Combined 40 kg hard corner | Fail | Terminated before the scheduled push |

Aggregate result:

- scenario success: `7/16` (`43.75%`), below the `95%` requirement;
- survival: `15/16` (`93.75%`);
- balance recovery: `15/16` (`93.75%`);
- tracking recovery: `7/16` (`43.75%`);
- peak pitch among measured post-push cases: `7.493 deg`;
- aggregate post-push `vx` RMSE: `0.0783 m/s`;
- aggregate post-push `wz` RMSE: `0.1652 rad/s`;
- requested-action saturation: `0.2792%`.

The result shows that the inner balance loop is substantially more robust than
the outer velocity loop. Most one-factor failures survive and remain upright,
but miss yaw tracking or fail to re-enter the tracking envelope. This supports a
future bounded `vx/wz` integral-action study, not an inner-LQR retune or PPO run.

## Mass-contract defect

The smoke exposed a source/runtime discrepancy that must be fixed first:

```text
URDF explicitly authored mass: 26.0 kg
PhysX resolved runtime mass:    30.0 kg
resolved body masses:           [23, 1, 1, 1, 1.5, 1.5, 1] kg
```

`arm_mount_link`, `imu_link`, `laser_link`, and `upper_imu_link` have no URDF
inertial blocks. PhysX assigns each a `1.0 kg` fallback. Consequently, all
accepted nominal LQR, push, tracking, and combined gates actually ran on a
`30.0 kg` plant, despite the static asset audit reporting `26.0 kg`.

The evaluator now resolves a requested `40 kg` total from the live articulation.
The earlier internal attempts that assumed `40/26` and/or overwrote nominal
friction are quarantined and are not evidence.

## Stop rule and next action

Do not tune outer-loop integral gains, regenerate LQR gains, or resume PPO until:

1. the team provides credible masses, COMs, and inertias for the four missing links;
2. those inertials are authored in the URDF and the USD is regenerated;
3. static authored mass and live PhysX-resolved mass agree;
4. the accepted nominal and combined gates pass on the corrected plant.

After those checks, rerun this exact 16-case matrix. If the same steady-state
tracking pattern remains, perform a small deterministic `vx_ki/wz_ki` sweep with
the inner LQR frozen.

## Evidence

- `evidence_20260713_plant_uncertainty/nominal_regression.json`
- `evidence_20260713_plant_uncertainty/plant_uncertainty_smoke.json`
- `scripts/two_wheel_balance/evaluate_lqr_tracking_push.py`
- `src/rl_platform/tasks/two_wheel_balance/metrics.py`
