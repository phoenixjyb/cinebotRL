# Two-Wheel Geometry and Axis Correction

> **Mass note 2026-07-14:** the geometry and axis correction remains current,
> but the provisional mass statement below is superseded by the explicit 28 kg
> contract in `LQR_28KG_MODEL_AND_OUTER_LOOP_GATE_20260714.md`.

Date: 2026-07-12

## Corrected contract

```text
wheel diameter:       0.2032 m (8 inches)
wheel radius:         0.1016 m
wheel track:          0.620 m center-to-center
left center:          [0, +0.310, 0.1016] m
right center:         [0, -0.310, 0.1016] m
left/right axis:      [0, +1, 0]
positive wheel speed: chassis +X
```

The wheel cylinder inertia values were regenerated from the 8-inch radius. Total provisional model mass remains 26.0 kg.

## Why the axis also changed

With the old axis `[0,-1,0]`, positive angular velocity gives a positive contact-surface velocity at the wheel bottom, requiring chassis motion toward `-X` under no slip. The old URDF comment claiming positive speed drove `+X` was therefore inconsistent.

The first 8-inch common-effort replay exposed this directly: both wheel velocities were positive while root `vx` was negative. Both axes were changed to `+Y`, and the action mixer returned to:

```text
tau_left  = common - yaw
tau_right = common + yaw
```

After correction:

```text
+0.1 common: left/right wheel velocity approximately +1.63 rad/s, vx > 0
+0.1 yaw:    left approximately -1.00, right approximately +0.99 rad/s, wz +0.296 rad/s
```

## Validation

- Static URDF/USD audit passes track, diameter, center-height, axis, mass, inertia, floating-root, and effort-drive checks.
- Passive 2-degree perturbation: 113 mean policy steps before body contact.
- Scripted sign-correct PD: 409 mean policy steps, versus 113 passive.
- Two `32 x 2048` direct-mode smokes are byte-identical.
- Vector smoke: 295 accounted contacts and zero non-finite values.

Evidence is under `evidence_20260712_8in/`.

## Training boundary

All earlier PPO, residual-PPO, PD, and rendered evidence used the obsolete wheel geometry/axis convention. Do not resume those checkpoints. The next learning run must start from scratch on this corrected plant, after the remaining provisional mass, COM, inertia, and torque assumptions are accepted or replaced with measurements.
