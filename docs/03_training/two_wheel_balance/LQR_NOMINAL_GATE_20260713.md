# Nominal Scripted LQR Gate

Date: 2026-07-13

## Result

The scripted LQR gate passes on the corrected 8-inch nominal Isaac plant. This is a simulation baseline, not a hardware controller release and not PPO training.

| Candidate scale | Success | Mean duration | Pitch p95 | Saturation |
| ---: | ---: | ---: | ---: | ---: |
| 0.4 | 32/32 | 1,999 steps (environment timeout) | 4.253 deg | 0.000 |
| 0.5 | 32/32 | 1,999 steps (environment timeout) | 3.222 deg | 0.000 |
| **0.6** | **32/32** | **1,999 steps (environment timeout)** | **2.765 deg** | **0.000** |

The selected candidate also records `7.993 deg` maximum pitch and `0.586` action-magnitude p95. Isaac reports the 2,000-step episode timeout on the 1,999th recorded transition, equivalent to the configured 10-second horizon within one 5 ms policy step.

## Nominal assumptions

```text
wheel diameter: 0.2032 m
wheel track:    0.620 m
URDF-authored mass: 26.0 kg
PhysX-resolved mass: 30.0 kg
torque limit:   20.0 Nm per wheel
physics rate:   1000 Hz
policy rate:    200 Hz
LQR update:      50 Hz, four policy steps per zero-order hold
```

Four fixed links have no authored URDF inertial and resolve to `1.0 kg` each in PhysX. The accepted run therefore used a `30.0 kg` runtime plant, despite the older static audit reporting `26.0 kg`. Mass distribution, COM, inertia, friction, actuator torque, and delay remain provisional. The selected gain must not be copied to hardware without a measured-plant review.

## Identification and synthesis

The script obtains a central-difference discrete model directly from Isaac instead of relying on a hand-derived inverted-pendulum equation. A four-step zero-order hold is required because wheel torque does not produce a reliable root-yaw response at the first 5 ms observation boundary.

The six deployable measured states remain:

```text
pitch, pitch_rate, mean_wheel_position,
mean_wheel_velocity, wheel_velocity_difference, yaw_rate
```

The controllable LQR coordinates are:

```text
common effort: pitch, pitch_rate, mean_wheel_velocity
yaw effort:    wheel_velocity_difference
```

Absolute wheel position and yaw rate remain observed and logged but are not independently penalized in this first regulator because they are neutral or nearly redundant under the nominal no-slip contact model.

The selected normalized action law is `u = -Kx`, clipped to `+/-0.8`, with:

```text
K = [
  [-3.99899718, -2.03419798, 0, 0.00025472, 0, 0],
  [ 0,           0,          0, 0,          0.00648425, 0]
]
```

The longitudinal Riccati equation uses SciPy's deterministic `solve_discrete_are` fallback because the dependency-free fixed-point iteration does not converge for the stiff identified plant. The yaw scalar still converges with the fixed-point solver.

## Recovery scenarios

The final gate runs 32 environments covering repeated combinations of:

```text
initial pitch:    -8, -5, -2, +2, +5, +8 deg
initial yaw rate: -0.3, 0, +0.3 rad/s
horizon:          10 s
```

Passing requires at least 95% timeout survival, pitch p95 at most 10 degrees, action saturation at most 10%, and finite state throughout. The selected candidate passes all criteria with 100% survival and zero saturation.

## Boundaries and next gate

- No mass, COM, inertia, friction, torque, or latency randomization is included yet.
- No external push, slope, low-speed velocity command, or hardware test is included.
- PPO remains blocked; no checkpoint was created or resumed.
- Isaac still emits existing visual-reference warnings for `arm_mount_link` and `upper_imu_link`; these did not invalidate physics but should be fixed before rendered proof.
- The next controller task is a bounded plant-parameter and disturbance sweep around this frozen LQR baseline.

Machine-readable evidence is in `evidence_20260713_lqr_nominal/`.
