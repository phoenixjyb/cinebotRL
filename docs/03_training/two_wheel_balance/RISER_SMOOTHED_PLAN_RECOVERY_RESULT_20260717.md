# Riser Smoothed-Plan Recovery CPU Result

Date: 2026-07-17
Workspace: `/mnt/g/wSpace/cinebotRL-two-wheel-riser`
Planner commit: `85c16455d3f65fa06b6c938402fdf1c4a9e4cb90`

## Decision

The bounded CPU recovery improved timing/transition/kinematic admission from
`63/79` to `69/79`, but the required `>=70` Gate-B count is still not met.
Keep Isaac, Gate C, residual capture, BC, PPO, and differential-session work
closed. All exported plans remain `valid_for_training=false`.

No source arrays, source hashes, physical limits, duration threshold, path
threshold, position threshold, or gimbal-rate threshold were changed. Existing
baseline candidates remain first in the deterministic search order.

## Implemented recovery

Two bounded mechanisms are appended only after the baseline candidates fail:

- derive a forward or reverse reset yaw from the first `0.50 m` of immutable
  source motion, while solving the same first camera target;
- for one high-curvature family, blend the sigma-16 path by `0.45`, retaining
  endpoints and limiting path shortening to less than 5%.

The exporter records source yaw, selected reset yaw/mode, smoothing sigma and
blend, every rejected attempt, and all original source provenance. The full
CPU suite passes `270/270` with two pre-existing pytest configuration warnings.

Newly admitted cases relative to the original 63-case corpus:

| Case | Duration ratio | Path drift | Position p95/max (m) | Recovery |
| ---: | ---: | ---: | ---: | --- |
| 28 | 1.706262 | -3.6247% | 0.04057 / 0.04899 | reverse reset yaw |
| 36 | 1.733733 | -2.7542% | 0.12121 / 0.22516 | wider low-gain preview |
| 64 | 1.927540 | 0.0000% | 0.09256 / 0.10256 | forward reset yaw |
| 72 | 1.928550 | 0.0000% | 0.10971 / 0.12059 | wider low-gain preview |
| 73 | 1.842642 | -4.9530% | 0.14387 / 0.15946 | blended path + forward reset yaw |
| 78 | 1.419314 | 0.0000% | 0.10688 / 0.23914 | reverse reset yaw |

## Authoritative evidence

Four-case committed recovery canary:

```text
/mnt/g/wSpace/cinebotRL-two-wheel-riser/artifacts/two_wheel_riser/
20260717_smoothed_plan_reset_recovery_canaries_v1_committed_cpu
```

```text
bb4e819423020d047c7197ac69b2c739dfd128d547796b70376a5a78883fa5c0  manifest.json
```

Final all-79 single-commit export:

```text
/mnt/g/wSpace/cinebotRL-two-wheel-riser/artifacts/two_wheel_riser/
20260717_smoothed_plan_all79_v3_reset_recovery_committed_cpu
```

```text
29f167c4810202d5c121757851d1697d1ccffa554af8a4714a916c6be60bc9e1  manifest.json
411dea82f8e52611e87d21e26bf5dd442f801c68b44099858f73796e7f5d3b17  summary.json
```

Final all-79 result:

```text
attempted:                       79/79
admitted:                        69/79
required:                        70
accepted duration median:         1.493855x
accepted duration maximum:        1.999969x
portfolio gate passed:           false
valid for training:              false
```

Rejected cases:

```text
1, 20, 27, 29, 35, 38, 39, 40, 45, 71
```

Case 27 remains the honest vertical-workspace reject. Cases 38/39/40 share an
attitude/proxy-rate duration limit: the best inspected plan has a proxy-only
bound above 2.0x at the unchanged 24 deg/s filming limit. Do not recover them
by raising that limit. Cases 1, 35, 45, and 71 remain broad structural rejects.

## Exact continuation

Recover one more case before any runtime work. Case 20 is the closest remaining
structural candidate, but bounded parameter search stopped at a position p95 of
about `0.185 m` under an admissible `1.9998x` duration. The next change should
be a CPU-only curvature-feedforward or nonholonomic path decomposition change,
with explicit regression coverage and unchanged thresholds. Do not perform
another blind preview/smoothing sweep.

Only after a fresh single-commit all-79 export proves `>=70` may the session
prepare a bounded deterministic Gate-C canary. That later step still requires
exclusive GPU ownership and must not create residual labels or start BC/PPO.
