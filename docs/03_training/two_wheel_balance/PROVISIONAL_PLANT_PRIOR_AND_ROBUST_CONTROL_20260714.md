# Provisional Plant Prior and Robust Controller

Date: 2026-07-14

Status: simulation engineering baseline pending hardware measurement

## Purpose

COM, inertia, available wheel torque, tire friction, and control delay are not
yet measured. Work can continue only if guessed values are explicit,
replaceable, and never presented as hardware validation. The source of truth is
`PLANT_PRIOR_PROVISIONAL_V1.json`.

## Three envelopes

The nominal model remains the explicit 28 kg URDF: COM at
`[0, 0, 0.40] m`, base diagonal inertia `[1.533, 1.613, 0.694] kg m^2`,
20 Nm per wheel, friction `0.9/0.8`, and zero modeled delay.

The primary provisional operating envelope assumes:

- mass `26.6-29.4 kg`;
- COM x `+/-0.02 m` and COM z `0.37-0.43 m`;
- inertia scale `0.85-1.15`;
- available torque `18-20 Nm` per wheel;
- static/dynamic friction from `0.70/0.60` to `1.00/0.90`;
- command delay `0-10 ms`.

The broader stress envelope retains mass scale `0.85-1.25`, COM x
`+/-0.03 m`, COM z `0.35-0.45 m`, 16 Nm minimum torque, friction down to
`0.65/0.55`, and 20 ms delay. It is diagnostic and is not a guessed hardware
specification.

## Controller decision

The 28 kg inner LQR remains frozen. Conditional anti-windup was added to the
outer loops. A bounded equilibrium-pitch estimator adapts once at startup while
both commands are zero and pitch rate is low, then freezes during motion. A
calibration latch prevents re-entry when a later command returns to zero. This
uses the IMU to identify COM-x bias without knowing COM in advance. The selected
provisional controller changes yaw adaptation and startup bias handling:

```text
vx_kp = 0.60
vx_ki = 0.05
vx integral limit = 0.50
wz_kp = 0.25
wz_ki = 0.10
wz integral limit = 2.00
wz feedforward = 0.60
pitch reference limit = +/-6 deg
pitch bias adaptation rate = 5.0 /s
pitch bias limit = +/-4 deg
action limit = +/-0.8
```

An aggressive longitudinal candidate used `vx_ki=0.075`, integral limit
`1.5`, and an 8 degree lean cap. It improved the single-sign provisional
matrix to 12/14 but regressed the nominal gate to 34/36 because two pushes did
not return to the pre-push balance envelope. It is rejected.

## Current boundary

The selected controller preserves nominal 36/36 tracking-plus-push success for
signed `vx=+/-0.2 m/s`, `wz=-0.4/0/+0.4 rad/s`, and 2/4/6 N s pushes. The exact
repeat matches every recorded metric. Peak pitch is `10.91 deg`, post-push
`vx/wz` RMSE is `0.0819/0.0520`, and saturation is `0.213%`.

The full signed provisional operating gate covers 112 combinations:
`vx=+/-0.2 m/s`, `wz=+/-0.4 rad/s`, `+/-2 N s` pushes, and all 14 guessed plant
variations. All 112 survive and recover balance, peak pitch is `11.47 deg`, and
saturation is zero. Strict tracking passes 91/112. The misses are concentrated
in direction-dependent COM-x, low-COM, and high-inertia cases.

The broader single-sign 6 N s stress matrix retains 16/16 survival and balance
recovery; 11/16 pass every strict tracking limit, peak pitch is `10.28 deg`, and
saturation is zero. The aggressive longitudinal controller is still rejected.

This is a complete provisional **balance-safety** simulation baseline, not a
claim that every guessed plant tracks at full speed. Safety priority remains:
balance first, collision avoidance second, trajectory accuracy third. Hardware
measurements must replace the guesses before deployment gains are frozen.

## Replacement procedure

1. Measure total travel-pose mass and COM x/z.
2. Estimate yaw/pitch inertia from CAD or a bounded excitation test.
3. Derive continuous and peak wheel torque from current limit, motor constant,
   gearbox ratio, and efficiency;
4. measure command-to-torque delay and wheel/ground friction bounds;
5. update the JSON as v2 and rerun nominal, provisional, stress, signed-pitch,
   and full signed-command gates;
6. keep PPO blocked until the deterministic baseline is reaccepted.
