# Riser Smoothed-Plan CPU Result

Date: 2026-07-17
Workspace: `/mnt/g/wSpace/cinebotRL-two-wheel-riser`
Planner commit: `d10668c8c7b86752e173fdbd48c0b9c34a6627bf`

## Decision

The bounded CPU-only smoothed-plan task is complete, but the all-79 Gate-B
threshold did not pass. Keep Isaac, deterministic dynamic qualification,
residual capture, BC, PPO, and differential-session work closed.

The new planner keeps the `exact_source_v1` arrays verbatim in a separate
provenance block, derives a separately named smoothed target, chooses a
continuous RS4 branch over the complete sequence, and retimes from the actual
base/riser/proxy command demand. It does not carry the obsolete per-anchor
`bidirectional_path_heading` dwell into the execution schedule. Existing
physical and quality thresholds were not raised.

Authoritative CPU tests at the final planner revision:

```text
focused smoothed-plan tests:  7 passed
full repository suite:        265 passed, 2 pre-existing config warnings
```

## Three-case canary

Namespace:

```text
/mnt/g/wSpace/cinebotRL-two-wheel-riser/artifacts/two_wheel_riser/
20260717_smoothed_plan_canaries_v2_committed_cpu
```

Hashes:

```text
19b6bb8467b55f603e4331d5d6cbfc1652ae85f94f1254640a6aefd9d06d6d82  manifest.json
1e4c3aee1dadde2b478e6b7b504e9ac88c726aa2b0e65514ccb3a03ffac835b5  summary.json
```

All three required canaries passed timing, source-path, motion-direction,
transition, and kinematic gates:

| Case | Duration ratio | Path drift | Base/proxy max step (rad) | Position p95/max (m) |
| ---: | ---: | ---: | ---: | ---: |
| 74 | 1.973508 | -3.4954% | 0.022625 / 0.025343 | 0.117345 / 0.121121 |
| 77 | 1.000000 | 0.0000% | 0.000561 / 0.000564 | 0.000252 / 0.000252 |
| 52 | 1.370001 | 0.0000% | 0.032385 / 0.033186 | 0.087714 / 0.097464 |

Every canary preserved source timestamps, positions, semantic DFR attitudes,
per-source hashes, identity anchor ordering, first/final target, and local
motion direction. `valid_for_training=false` remained explicit.

## All-79 result

Namespace:

```text
/mnt/g/wSpace/cinebotRL-two-wheel-riser/artifacts/two_wheel_riser/
20260717_smoothed_plan_all79_v1_committed_cpu
```

Hashes:

```text
9ad04e7e9088d930b54b4287575c4b1ed2b7ca23ab6ab38097533173aef0bcb4  manifest.json
64a34b80293a68a11e835dbe2489bac6567f0e94c351999812901570bf7aa00d  summary.json
```

```text
attempted:                         79/79
exact-source provenance passes:   79/79
timing/transition/kinematic pass: 63/79
required admission count:         70
portfolio gate passed:            false
accepted duration-ratio median:   1.490993x
accepted duration-ratio maximum:  1.999969x
all-case duration-ratio median:   1.543629x
valid for training:               false
```

Rejected episode IDs:

```text
1, 20, 27, 28, 29, 35, 36, 38, 39, 40, 45, 64, 71, 72, 73, 78
```

Failure families:

- Duration plus position p95/max: `1, 20, 71, 73`.
- Duration plus position and vertical workspace: `27`.
- Position p95/max only: `28, 29, 35, 45, 64, 78`.
- Position max only: `36`.
- Duration only: `38, 39, 40`.
- Local source-motion direction only: `72`.

No plan hash mismatch was found across the 79 NPZ files. All accepted plans
preserve source motion direction and pass the unchanged `0.4 m/s` base-speed,
`0.4 rad/s` yaw-rate, gimbal-rate, workspace, and tracking-error gates.

## Stop state

The `>=70` admission requirement is not met, so the exact continuation point
is a CPU-only targeted recovery of at least seven rejected cases. Do not relax
the duration, path, transition, workspace, or tracking thresholds and do not
replace a reject with an old-plan fallback. Preserve the accepted 63 unchanged.

No dynamic accepted/rejected count exists because Isaac was not launched. No
residual envelope was collected, no train/validation/holdout split was
constructed, and BC/PPO did not start.
