# Riser Smoothed-Plan Recovery CPU Result

Date: 2026-07-18
Workspace: `/mnt/g/wSpace/cinebotRL-two-wheel-riser`
Planner commit: `26e90ed96e2460ad1f33293166f5c06dedf1c9fc`

## Decision

The bounded CPU recovery improved timing/transition/kinematic admission from
`63/79` to `70/79`. The required `>=70` Gate-B count is met. The first
smoothed-plan Gate-C canary, case 77, is a deterministic dynamic rejection
because it did not complete under the frozen phase governor. Residual capture,
BC, PPO, and differential-session work remain closed. All exported plans and
runtime evidence remain `valid_for_training=false`.

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

## Smoothed Gate-C case-77 result

The first isolated namespace stopped before dynamics because the generic
playback loader did not yet understand the nested smoothed-plan metadata:

```text
/mnt/g/wSpace/cinebotRL-two-wheel-riser/artifacts/two_wheel_riser/
20260718_gate_c_smoothed_case77_v1_exclusive

runtime exception: KeyError('vertical_shift_m')
dynamic / thermal result: null / null
dataset / BC / PPO: none / false / false
```

This namespace is preserved as a consumed schema-failure audit. Commit
`cd5ed94802165718d73266c2a52a527ad9c37045` added fail-closed support for the
versioned `smoothed_target` metadata block. The authoritative CPU suite then
passed `281/281` tests in two explicit chunks (`123 + 158`).

The corrected one-case run used a new token and namespace:

```text
/mnt/g/wSpace/cinebotRL-two-wheel-riser/artifacts/two_wheel_riser/
20260718_gate_c_smoothed_case77_v2_exclusive

ff064be8368f19b3508cd14e3c4d583f9ff970300a8d9ca1d6115d96955c341b  admission.json
0d6133125f78831c93820e0a00ba1bbe6803ebfb217a3262e81174cbbd722fa4  gates/case_0077.json
423a54577e74d82fbf750121428eac0bdd8eb1c3a5f71c2aaaf340076ccd9089  logs/case_0077.log
59771700ce07e47726ce412777f496a76e77b277cb3e40dd7f01f629e2fc4a2e  summary.json
```

Case 77 passed every physical check except reference completion:

```text
source / execution duration:             5.431279 / 5.431279 s
completed phase / 2x runtime horizon:     3.753394 / 10.862558 s
position p95 / max:                       0.096049 / 0.101564 m
pitch p95 / max:                          6.779128 / 7.253576 deg
attitude p95 / max:                       0.124822 / 0.169612 deg
thermal admission:                        pass
action/riser/proxy saturation:            0 / 0 / 0
termination:                              none
dynamic quality:                          fail (completed_reference only)
```

The trace identifies a phase-governor lock rather than geometric divergence.
Pitch settles near `6.27 deg`; the unchanged `3..8 deg` balance governor then
holds progress near `0.345` even while camera error remains below `0.10 m`.
The prospective raw residual envelope independently fails on `delta_vx`:

```text
raw residual abs max:                     [0.314832, 0.003819, 0.008910]
normalized prospective label abs max:     [1.049442, 0.009548, 0.089099]
residual applied to commands:              false
residual dataset:                          none
```

No playback or compute owner remained after the run. This is the first final
dynamic reject for the smoothed portfolio, so no additional Gate-C episode may
start from this result.

## Case-77 dynamic-margin retime CPU result

Commit `66f0e6aeca22d75d5a716039cb87197754274f78` adds a generic,
evidence-bound uniform execution retimer and a CPU-only derivation command.
The authoritative CPU suite passes `284/284` tests in two explicit chunks
(`123 + 161`). No Isaac application was imported or launched.

The fresh case-77 candidate is stored at:

```text
/mnt/g/wSpace/cinebotRL-two-wheel-riser/artifacts/two_wheel_riser/
20260718_case77_dynamic_margin_retime_1p4_v1_cpu

7b0c9b5330733a5e6740023048c308491dd2bd85fd9f30e5fd0cc16901dbe5b7  case_0077_dynamic_margin_retime_v1.npz
ef2e85f8c3ef2ae13b67efde6c2377e41285afea03844790ef7c9cef498a66c5  manifest.json
4494f399f1e3a945375c67957eea2439562c57f9ed0f62536895f03ae90a285a  summary.json
```

The derivation is bound to the rejected v2 Gate-C JSON and summary hashes. An
independent readback recomputed the output hash and every kinematic gate. All
immutable source, target, base, riser, proxy, anchor-map, and initialization
arrays are bit-identical to the parent. Only the execution clock and matching
base/riser/proxy feed-forward derivatives changed.

```text
source / execution duration:             5.431279 / 7.603791 s
execution/source ratio:                  1.400000
prospective accepted count:              70
prospective portfolio median:            1.497221x
maximum base speed:                       0.279130 m/s
maximum base yaw rate:                    0.034685 rad/s
maximum riser rate:                       0.032118 m/s
maximum proxy rate:                       0.034769 rad/s
path drift / source deviation:            0 / 0
failed CPU checks:                        none
```

Controller, phase governor, thresholds, and source geometry are explicitly
recorded as unchanged. Dynamic and thermal qualification remain false because
this is a derived CPU candidate, not runtime evidence. Residual capture, BC,
and PPO remain disabled.

## Case-77 1.499x portfolio and dynamic qualification

The 1.4x candidate improved the phase-governed run but still stopped at
`7.042292 / 7.603791 s`. All physical checks passed and its prospective
residual label entered the frozen envelope. Because another uniform retime to
1.5x would violate the portfolio median by `0.000294x`, the final CPU candidate
uses `1.499x`:

```text
/mnt/g/wSpace/cinebotRL-two-wheel-riser/artifacts/two_wheel_riser/
20260718_case77_dynamic_margin_retime_1p499_v2_cpu

a45892c98311cdd6e6f2096b6821ef760759504138edc2f9c7caa9b1ac90f559  case_0077_dynamic_margin_retime_v1.npz
deb3466636b1ef40196c5bd3108e0ca7495b70412617b97999bf6de008a39111  manifest.json
edeb6d58345b52589741ec93955b9692423e6af7906cafc4272b529bc2fa5aab  summary.json
```

All 12 immutable source/geometry/state arrays are bit-identical to the 1.4x
parent. Only the execution clock, corresponding feed-forward rates, and
metadata changed. The v6 portfolio replaces only case 77:

```text
/mnt/g/wSpace/cinebotRL-two-wheel-riser/artifacts/two_wheel_riser/
20260718_smoothed_plan_all79_v6_case77_1p499_cpu

73121d240ccf54fa65783fc1cf47eed4d805af3e6bedbdfff847719c92f2130b  manifest.json
df9977ed7eaefb1ddc5afdff1bb681380779db95a2264ef1bc3807f67b015179  summary.json

hash-verified plans:                      79/79
changed cases from v5:                    [77]
accepted / rejected:                      70 / 9
accepted duration median:                 1.499794x
```

The first v6 canary at the unchanged 2.0x runtime horizon was again a narrow
completion-only reject: `7.977911 / 8.141487 s`, with position p95/max
`0.084045 / 0.084785 m` and all other checks passing. Its stable tail required
an estimated `0.334 s` more wall time. Commit
`3b7edebcb87edd9cc5f7e2329f51a0c7fa0ce6fa` therefore adds a separately
reported, fail-closed 2.05x completion horizon. This changes neither plan
duration nor source, commands, controller, phase governor, or quality/safety
thresholds. The complete CPU suite passes `285/285` tests with two pre-existing
pytest configuration warnings.

The fresh v5 canary passes:

```text
/mnt/g/wSpace/cinebotRL-two-wheel-riser/artifacts/two_wheel_riser/
20260718_gate_c_smoothed_case77_v5_completion_grace_exclusive

850c01a6a976fbc2688c3cfada24f12510a9ee9a6b51cb8412e7b1d8205cb875  admission.json
e3ddb05e5f1287b8945cabaefeac9aca54951d38b24802bd94f29f2c51551475  gates/case_0077.json
b5abe0f6f77d2203af081872a2f35c20bea1b22923022224aba0774e7b11e684  logs/case_0077.log
03b4327565d1a149cf2663c388fe64baec0459ed7fae0206da8059b1ad5b3aba  summary.json

source / execution duration:             5.431279 / 8.141487 s
completed phase / runtime:                8.141487 / 16.635000 s
bounded maximum runtime:                  16.690049 s
position p95 / max:                       0.084030 / 0.084785 m
pitch p95 / max:                          5.816333 / 6.514108 deg
attitude p95 / max:                       0.123348 / 0.165009 deg
action/riser/proxy saturation:            0 / 0 / 0
dynamic / thermal / label envelope:       pass / pass / pass
```

No residual was applied, no dataset was written, and residual capture, BC, and
PPO remain false. The GPU was empty after the run. This is one qualified
episode, not an all-70 dynamic qualification and not training admission.

## Exact continuation

Preserve all consumed namespaces and the v6 portfolio hashes. The next bounded
task is to define and run a small representative deterministic Gate C expansion
from the accepted v6 plans, beginning with the already important cases 52 and
74. It must use fresh authorization and namespaces, stop on the first dynamic
reject, retain the 2.05x completion-horizon evidence contract, and write no
residual dataset. Only after representative canaries pass should the accepted
70 be qualified sequentially.

Do not start residual capture, BC, or PPO. `valid_for_training` remains false
until the required deterministic dynamic corpus and subsequent capture gates
are complete.
