# 28 kg Model and Outer-Loop Gate

Date: 2026-07-14

Branch: `codex/two-wheel-balance-rl`

## Result

- explicit URDF-to-PhysX mass contract: **passed**;
- fresh 28 kg inner-LQR identification and signed-pitch recovery: **passed**;
- nominal combined tracking-plus-push gate: **36/36 passed**;
- exact deterministic repeat: **passed**;
- plant-uncertainty gate: **failed at 7/16**, with **16/16 survival**;
- PPO or other policy training: **not started**.

## Physical correction

The current two-wheel robot is approximately `28 kg`, not `40 kg`. The prior
URDF explicitly authored `26 kg`, but four fixed sensor/reference links had no
inertials and each received a PhysX fallback mass of `1 kg`, producing an
unintended `30 kg` runtime plant.

The simplified model treats `base_link` as the aggregate rigid upper body. The
four child links remain frames rather than independent equipment bodies, so they
receive explicit `1 g` inertials only to suppress importer fallback mass:

| Link | Mass |
| --- | ---: |
| `base_link` aggregate | `24.996 kg` |
| left wheel | `1.500 kg` |
| right wheel | `1.500 kg` |
| `arm_mount_link` frame | `0.001 kg` |
| `imu_link` frame | `0.001 kg` |
| `laser_link` frame | `0.001 kg` |
| `upper_imu_link` frame | `0.001 kg` |
| **Total** | **`28.000 kg`** |

The generated USD passes the complete static gate: seven positive explicit
inertials, `28.0000003 kg` authored mass, 620 mm wheel track, 8-inch wheels,
`+Y` wheel axes, floating articulation, and zero wheel-drive
stiffness/damping. The live articulation resolves to `27.9999981 kg`.

USD regeneration must use:

```text
--mesh-scale 1.0 --default-drive-type none
```

Omitting `--default-drive-type none` recreates position drives with stiffness
`625` and invalidates effort control.

## Inner-LQR correction

The 30 kg-era inner gain was not accepted on the corrected plant. Under the
corrected initial-condition gate it survived all six signed
`+/-2/5/8 deg` starts, but recovered `0/6` and saturated `24.67%` of requested
actions.

A fresh small-signal identification on the live 28 kg PhysX articulation
selected LQR scale `0.6`. The nominal tuner reported pitch p95
`1.690 deg`, pitch max `7.991 deg`, and zero saturation. The stronger signed
initial-pitch gate then passed `6/6`: the worst `+/-8 deg` recovery was
`0.825 s`, peak pitch was the authored `8 deg` initial condition, and action
saturation remained zero.

The recovery evaluator was corrected during this audit. For zero-push initial
conditions it now starts measurement and recovery timing at step zero and
includes the authored initial pitch in the peak. The prior implementation
waited until the unused push window and could report misleading zero-time
recovery.

## Outer-loop correction

With the fresh inner gain, zero outer integral passed only `12/36` combined
cases. All 36 survived and recovered balance; the turning cases missed because
steady-state `vx` error remained just above the `0.10 m/s` limit.

The newly identified inner LQR, proportional gains, feedforward, limits, and
50 Hz update rate were then frozen. The bounded integral correction remains:

```text
vx_kp = 0.60
vx_ki = 0.05
wz_kp = 0.25
wz_ki = 0.05
wz_feedforward = 0.60
pitch reference limit = +/-6 deg
action limit = +/-0.8
```

The corrected 36-case gate covers `vx=+/-0.2 m/s`,
`wz=-0.4/0/+0.4 rad/s`, and signed `2/4/6 N s` impulses.

| Metric | Result | Limit |
| --- | ---: | ---: |
| Success and survival | `36/36` | `>=95%` success |
| Worst balance recovery | `0.765 s` | `<=2.0 s` |
| Worst tracking recovery | `0.785 s` | `<=2.0 s` |
| Peak pitch | `10.980 deg` | `<=12 deg` |
| Post-push `vx` RMSE | `0.0762 m/s` | `<=0.10 m/s` |
| Post-push `wz` RMSE | `0.0746 rad/s` | `<=0.15 rad/s` |
| Requested-action saturation | `0.2131%` | `<=10%` |

The seeded repeat is exactly equal for the summary and every recorded
per-scenario metric.

## Uncertainty boundary

The corrected uncertainty matrix replaces the obsolete 40 kg cases with
`0.85x`, `1.15x`, and `1.25x` mass stress cases around the 28 kg plant. With
`vx_ki=wz_ki=0.05`:

- all `16/16` scenarios survived and recovered balance;
- `7/16` met every tracking, recovery, tilt, and saturation requirement;
- aggregate post-push `vx` RMSE was `0.0864 m/s`;
- aggregate post-push `wz` RMSE was `0.1370 rad/s`;
- peak pitch was `10.417 deg`;
- requested-action saturation was `0%`.

The remaining failures are tracking failures under mass variation, reduced
torque, `+/-3 cm` fore-aft COM shifts, vertical COM shifts, and the combined
corner case. All cases still recover balance. The earlier `0.10` integral
experiment used the rejected 30 kg-era inner gain, so it is retained only as
historical diagnostic evidence and is not part of the fresh-controller gate.
Blind integral escalation remains prohibited.

## Next action

The nominal controller baseline is accepted. The uncertainty gate remains a
controller-architecture and system-identification task:

1. measure or bound aggregate COM, yaw inertia, available wheel torque, and real control delay;
2. replace broad assumed ranges with credible hardware ranges;
3. evaluate torque/inertia-normalized yaw feedforward, anti-windup, or gain scheduling;
4. keep the accepted 28 kg inner balance LQR frozen unless hardware-range tests fail balance;
5. do not resume PPO until the deterministic uncertainty gate is acceptable.

## Evidence

- `evidence_20260714_28kg/asset_audit.json`
- `evidence_20260714_28kg/linear_model.json`
- `evidence_20260714_28kg/lqr_gains.json`
- `evidence_20260714_28kg/lqr_candidates.json`
- `evidence_20260714_28kg/lqr_nominal_gate.json`
- `evidence_20260714_28kg/inner_recovery_gate.json`
- `evidence_20260714_28kg/old_inner_recovery_regression.json`
- `evidence_20260714_28kg/combined_baseline_no_integral.json`
- `evidence_20260714_28kg/combined_gate.json`
- `evidence_20260714_28kg/combined_gate_repeat.json`
- `evidence_20260714_28kg/plant_uncertainty_gate.json`
- `evidence_20260714_28kg/recording.json`
