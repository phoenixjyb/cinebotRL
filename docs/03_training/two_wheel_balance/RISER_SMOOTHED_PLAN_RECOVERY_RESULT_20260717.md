# Riser Smoothed-Plan Recovery CPU Result

Date: 2026-07-18
Workspace: `/mnt/g/wSpace/cinebotRL-two-wheel-riser`
Planner commit: `26e90ed96e2460ad1f33293166f5c06dedf1c9fc`

## Decision

The bounded CPU recovery improved timing/transition/kinematic admission from
`63/79` to `70/79`. The required `>=70` Gate-B count is now met. A bounded,
deterministic Gate-C canary may be prepared under exclusive GPU ownership, but
residual capture, BC, PPO, and differential-session work remain closed. All
exported plans remain `valid_for_training=false`.

No source arrays, source hashes, physical limits, duration threshold, path
threshold, position threshold, or gimbal-rate threshold were changed. Existing
baseline candidates remain first in the deterministic search order.

## Implemented recovery

Three bounded mechanisms are appended only after the baseline candidates fail:

- derive a forward or reverse reset yaw from the first `0.50 m` of immutable
  source motion, while solving the same first camera target;
- for one high-curvature family, blend the sigma-16 path by `0.45`, retaining
  endpoints and limiting path shortening to less than 5%.
- for a near-gate seed that fails only position p95, optimize bounded residual
  signed base velocity/yaw-rate controls over the complete unicycle rollout.
  The optimizer has no lateral control, includes the frozen retiming demand in
  its objective, and re-runs global RS4 branch selection before final gating.

The exporter records source yaw, selected reset yaw/mode, smoothing sigma and
blend, batch-recovery contract, every rejected attempt, and all original source
provenance. The full CPU suite passes `273/273` with two pre-existing pytest
configuration warnings.

Newly admitted cases relative to the original 63-case corpus:

| Case | Duration ratio | Path drift | Position p95/max (m) | Recovery |
| ---: | ---: | ---: | ---: | --- |
| 20 | 1.992700 | -4.9000% | 0.13994 / 0.16555 | batch unicycle recovery |
| 28 | 1.706262 | -3.6247% | 0.04057 / 0.04899 | reverse reset yaw |
| 36 | 1.733733 | -2.7542% | 0.12121 / 0.22516 | wider low-gain preview |
| 64 | 1.927540 | 0.0000% | 0.09256 / 0.10256 | forward reset yaw |
| 72 | 1.928550 | 0.0000% | 0.10971 / 0.12059 | wider low-gain preview |
| 73 | 1.842642 | -4.9530% | 0.14387 / 0.15946 | blended path + forward reset yaw |
| 78 | 1.419314 | 0.0000% | 0.10688 / 0.23914 | reverse reset yaw |

## Authoritative evidence

Three-case committed batch-recovery canary:

```text
/mnt/g/wSpace/cinebotRL-two-wheel-riser/artifacts/two_wheel_riser/
20260718_smoothed_plan_batch_recovery_cases20_29_52_v1_committed_cpu
```

```text
e490c04ae9a3e9081b65bb655b74a1c428c6aa8f267dd03eb5f625ff5984db69  manifest.json
a46fbe962192e17c650af8446df9bb9fc099b121114b04d2a4ca98fd1c22a706  summary.json
```

Final all-79 single-commit export:

```text
/mnt/g/wSpace/cinebotRL-two-wheel-riser/artifacts/two_wheel_riser/
20260718_smoothed_plan_all79_v4_batch_recovery_committed_cpu
```

```text
9044dc360ad1a9a59fa84ec9adf0b00a30b1039751e959fd26ec2edc23a684dd  manifest.json
367778df2225687b707abebdbb49422376b6a22a6530e28c612d093031158a66  summary.json
```

Final all-79 result:

```text
attempted:                       79/79
admitted:                        70/79
required:                        70
accepted duration median:         1.497221x
accepted duration maximum:        1.999969x
portfolio gate passed:           true
valid for training:              false
```

Rejected cases:

```text
1, 27, 29, 35, 38, 39, 40, 45, 71
```

Case 27 remains the honest vertical-workspace reject. Cases 38/39/40 share an
attitude/proxy-rate duration limit: the best inspected plan has a proxy-only
bound above 2.0x at the unchanged 24 deg/s filming limit. Do not recover them
by raising that limit. Cases 1, 35, 45, and 71 remain broad structural rejects.

## Post-checkpoint diagnosis and closure

Recover one more case before any runtime work. The post-checkpoint CPU audit
closed the previously proposed local recovery paths without changing code:

- Case 20 remains at `0.185358 m` position p95 with `1.999788x` duration for
  the best causal preview candidate. Its error is predominantly cross-track
  (`0.179272 m` p95, versus `0.062042 m` along-track), while both base linear
  speed and yaw rate reach their frozen limits.
- Blending an exact Cartesian correction into case 20 first meets the position
  gate at blend `0.20`, but requires `2.557315x` duration and saturates the
  permitted lateral-slip rate on `40.21%` of transitions. This is not a valid
  two-wheel recovery and must not be implemented.
- Exact fixed-point decomposition reaches near-zero Cartesian error only at
  `17.870599x` to `21.047178x` duration with lateral saturation on `79%` to
  `86%` of transitions. The existing joint-adaptive and short-horizon MPC
  prototypes also diverged above `1.9 m` p95; none is an admissible candidate.
- Cases 38/39/40 remain a shared duration-only family. A wider reset-yaw and
  preview scan improved the best ratio to `2.058564x` while retaining position
  p95/max of `0.141703/0.225114 m`, but still cannot meet the frozen `2.0x`
  limit. Proxy pitch contributes `39.894797 s` of motion and wins `34.523234 s`
  of the interval lower bound; non-overlapping roll/yaw/base demands raise the
  complete minimum schedule to `42.500894 s` versus the `41.291786 s` budget.
- Case 29 has duration headroom. A coarse reset-yaw scan improved its p95 from
  `0.330248 m` to `0.236754 m`; refinement plus admissible smoothing reached
  `0.220182 m` with max error and duration passing. Candidates below `0.20 m`
  violated the unchanged source-motion-direction gate.

The implemented batch optimizer closed case 20 without weakening these
constraints. The committed three-case canary recovered case 20, preserved case
29 as a reject, and left healthy case 52 on its original baseline candidate.
The single-commit all-79 export then reproduced the same case-20 plan hash:

```text
ec0bb2845c948d17daec8abef6b00b205f6f56fe6cb9e4c42aa9395c6b66336d
```

The recovered plan has effectively zero lateral velocity
(`7.84e-15 m/s`), base linear/yaw rates at the unchanged `0.4` limits, riser
rate at the unchanged `1.0 m/s` limit, and proxy rate at the unchanged
`24 deg/s` limit.

## Exact continuation

Re-read the current Gate-C handoff and runtime authorization contract. Then,
only with exclusive GPU ownership, run one bounded deterministic canary from
the accepted plans and stop on the first physical/safety/quality failure.
Gate C must record source and execution clocks separately, raw residual-envelope
status independently from dynamic quality, and no residual labels or dataset.
Do not start case batches, residual capture, BC, or PPO from this CPU result.
