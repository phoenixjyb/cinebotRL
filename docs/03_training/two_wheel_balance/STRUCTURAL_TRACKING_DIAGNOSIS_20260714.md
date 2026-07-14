# Two-wheel structural tracking diagnosis (2026-07-14)

## Scope and stop rule

This gate diagnoses the remaining chassis-only tracking failures on the provisional 28 kg plant model. It does not validate arm/end-effector tracking, obstacle avoidance, hardware transfer, or PPO. PPO remains blocked.

Acceptance requires:

- 112/112 survival and balance recovery on `provisional_prior_v1`.
- At least 106/112 strict admitted-tracking passes.
- Peak pitch below 12 degrees.
- Minimum admitted path-progress scale at least 0.75.
- No hidden wheel-mixer authority loss in settled tracking.
- 36/36 nominal tracking/push passes with no governor activation.

## Diagnosis

The v4 evaluator now mirrors the environment's common/yaw-to-wheel allocation and records wheel clipping, effective common/yaw authority, signed velocity means, pitch reference, yaw correction, and final controller state.

The 112-case baseline reproduced 91/112 strict passes, 112/112 balance recovery, and 11.47 degrees peak pitch. Wheel mixing was not the cause: wheel saturation and common/yaw authority loss were both zero, with a maximum pre-clip wheel command of 0.942.

The failures separated into three mechanisms:

1. `com_z_minus_0p03` failed 8/8 because longitudinal integral authority saturated at `+/-0.5`; post `vx` RMSE was about 0.105 m/s.
2. The negative-velocity half of `corner_provisional_v1` failed 4/4 because a +2.18 degree equilibrium bias coupled into yaw tracking. The original governor only admitted less progress when motion reinforced the bias.
3. Eight adverse COM-x cases were already repaired by the existing bias-based path-progress governor. One inertia case was a 2.05 s marginal recovery miss.

## Rejected candidates

- Conditional integral reserve `0.7` plus opposing-bias yaw feedforward boost `0.1`: safe, but only 99/112 strict passes. It improved errors without crossing the gate.
- Integral-state path governor: 112/112 strict robust passes, but it unnecessarily governed all 36 nominal cases and drove 31 to the 0.75 floor.
- Error-gated integral governor: nominal mean scale improved to 0.995, but threshold chatter removed tracking recovery from one healthy nominal case.
- Persistent integral limit `0.7` plus bidirectional bias governor at the old `vx_ki=0.05`: 103/112 strict passes. Low-COM post RMSE passed, but recovery remained slower than 2 s.

These candidates are not runtime profiles.

## Accepted profile

`structural_robust_v1` is explicit and opt-in. Conservative dataclass defaults remain unchanged.

```text
vx_ki = 0.075
vx_integral_limit = 0.7
path_progress_governor_enabled = true
governor_include_opposing_bias = true
governor_minimum_progress_scale = 0.75
```

Use `--controller-profile structural_robust_v1` together with `--tracking-reference admitted --minimum-path-progress-scale 0.75` for its acceptance gate.

## Results

| Gate | Strict | Balance | Peak pitch | Progress floor | Result |
|---|---:|---:|---:|---:|---|
| Baseline provisional | 91/112 | 112/112 | 11.47 deg | 1.000 | fail |
| Accepted provisional | 111/112 | 112/112 | 9.95 deg | 0.752 | pass |
| Baseline nominal | 36/36 | 36/36 | 10.91 deg | 1.000 | pass |
| Accepted nominal | 36/36 | 36/36 | 11.31 deg | 1.000 | pass |
| Diagnostic stress | 11/16 | 16/16 | 10.40 deg | 0.785 | diagnostic boundary |

The sole provisional miss recovered in 2.025 s, 25 ms beyond the strict deadline. Nominal post `vx` RMSE improved from 0.0819 to 0.0678 m/s and the governor activated in 0/36 nominal cases. The diagnostic stress gate remains 11/16 strict, while eventual tracking recovery improved from 12/16 to 15/16.

## Boundary and next work

This closes the provisional chassis tracking gate, not the complete robot objective. The next implementation step is to make the arm/end-effector controller consume the admitted chassis reference and preserve the balance safety hierarchy. Obstacle avoidance follows only after that integration passes no-obstacle whole-body tracking. Hardware claims remain blocked until COM, yaw inertia, wheel torque, friction, and delay are measured.

