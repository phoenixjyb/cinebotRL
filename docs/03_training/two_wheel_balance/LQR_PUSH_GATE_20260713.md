# Scripted LQR Push-Recovery Gate

Date: 2026-07-13

## Result

The frozen nominal LQR passes the first deterministic fore/aft push-recovery gate on the corrected 8-inch Isaac plant. No policy was trained or resumed.

| Metric | Result | Gate |
| --- | ---: | ---: |
| Scenario success | 32/32 (100%) | at least 95% |
| Survival | 32/32 (100%) | required per scenario |
| Recovery | 32/32 (100%) | required per scenario |
| Worst recovery time | 0.415 s | at most 2.0 s |
| Worst pitch excursion | 2.084 deg | at most 15 deg |
| Peak wheel speed | 3.028 rad/s | recorded, no gate |
| Aggregate action saturation | 0.0382% | at most 10% |

An identical second run with seed `20260713` produced exactly the same summary and zero difference for every scenario's recovery time, peak pitch, peak pitch rate, peak wheel speed, and saturation ratio.

## Disturbance contract

The gate starts upright and applies a horizontal global-X force for `0.1 s` at an equivalent point `0.5 m` above the base-link center of mass. Because the Isaac articulation API accepts a wrench at the body COM, the evaluator supplies both the horizontal force and the corresponding pitch torque.

```text
force:             -60, -40, -20, +20, +40, +60 N
linear impulse:    -6,  -4,  -2,  +2,  +4,  +6 N s
pitch torque:      -30, -20, -10, +10, +20, +30 N m
push start:        1.0 s
push duration:     0.1 s
evaluation horizon: 10 s
```

Recovery means that absolute pitch is at most `2 deg` and absolute pitch rate is at most `0.2 rad/s` continuously for `0.25 s` after the push ends. A recovery time of zero is valid when the response remains inside this recovery envelope for the first complete hold window.

## Controller contract

The gate does not retune the selected LQR:

```text
physics rate:      1000 Hz
policy rate:       200 Hz
controller update:   50 Hz
action limit:      +/-0.8
selected scale:    0.6
```

The same gain matrix in `evidence_20260713_lqr_nominal/lqr_gains.json` is used without modification.

## Boundaries

- This result used the `30.0 kg` PhysX-resolved plant. The URDF explicitly authors `26.0 kg`, but four fixed links without inertials resolve to `1.0 kg` each. It does not validate the approximately `40 kg` complete robot.
- Mass distribution, COM, inertia, tire friction, motor torque, transmission behavior, sensing delay, and control delay remain provisional.
- The force-height model is a deterministic wrench approximation, not a contact-object impact test.
- This gate starts upright. Simultaneous initial-angle recovery plus push is intentionally deferred until the single-disturbance layers are established.
- No slope, velocity command, arm motion, end-effector tracking, obstacle avoidance, actuator fault, or hardware test is included.
- PPO remains blocked. Passing this scripted-controller gate is not evidence that PPO is needed or ready.

## Next controller layer

The next bounded gate is low-speed chassis command tracking under the frozen balance LQR: signed `vx` and `wz` commands, including command transitions, while preserving the same pitch, wheel-speed, saturation, and survival limits. Plant-parameter robustness must then be checked before the articulated arm and end-effector objective are reintroduced.

Machine-readable accepted evidence is in `evidence_20260713_lqr_push/push_gate.json`.
