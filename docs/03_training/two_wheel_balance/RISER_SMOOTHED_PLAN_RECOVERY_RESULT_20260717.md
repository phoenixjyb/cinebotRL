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

## Representative Gate C results

Case 52's first namespace is an infrastructure timeout audit, not physical
evidence. Its fixed 300-second wall timeout expired exactly from `12:00:00` to
`12:05:00` after normal Isaac setup, before any runtime JSON was written:

```text
20260718_gate_c_smoothed_case52_v1_completion_grace_exclusive
classification: missing_runtime_json
GPU owner after cleanup: none
```

The fresh 420-second-bounded retry at commit
`f4fee5fe73fc8704afeda4164a4417ebc78cee35` produced complete evidence and
passed:

```text
20260718_gate_c_smoothed_case52_v2_timeout420_exclusive

d4c1c785ecf46c0d841ebdcdfdeb00e419048b79f35c029b2f4432ce204660f8  admission.json
3a8a52f3b2f6b4ec4ab4cd1d8b4e0b1195b87a438749e82c0a1bbe22cabef64a  gates/case_0052.json
13d919d60b5c6355448124d24ac2b6408a71bd9da4ceb08b80fb01d70f5a10e9  logs/case_0052.log
10f5f508b52ceaaac0d215ad32933368b00dedc7b4bef4da4f2046f5523a1b6a  summary.json

source / execution / runtime:            22.924931 / 31.407182 / 51.130000 s
position p95 / max:                       0.137522 / 0.146467 m
pitch p95 / max:                          6.354373 / 6.673738 deg
dynamic / thermal / label envelope:       pass / pass / pass
```

Case 74 then ran alone under the same deterministic contract and became the
first representative dynamic reject:

```text
20260718_gate_c_smoothed_case74_v1_representative_exclusive

703d9afa2f2493b81894f65b53be798a305c1da1967305e64901e2aad2b6aa4b  admission.json
ad900914351809a96b4cd34298daa9b57c981218d324ebcf0027fe721164f3c3  gates/case_0074.json
87ca442f4e76204673a26751e8044b1ef799bf6ba2b8b5e932dfb5781db12174  logs/case_0074.log
c1f3ee2470d3b825b2406bb056b1a023da1e702578202011780206d336ac1a9e  summary.json

source / execution / runtime:            11.373883 / 22.446453 / 43.760000 s
completed reference:                      pass
position p95 / 0.15 m gate:               0.163698 / fail
position max / 0.25 m gate:               0.171750 / pass
pitch max / attitude max:                 6.673746 / 0.218295 deg
termination / saturation:                 none / zero
thermal / label envelope:                 pass / pass
```

The miss is position-p95 only by `0.013698 m`. No residual was applied and no
dataset was written in either case. The fail-fast representative sequence
therefore stops at case 74; the accepted-70 batch remains closed.

## Case-74 CPU structural diagnosis

The sustained high-error window is execution phase `17.9..21.3 s`, inside
anchors `394..572` (`15.517..22.022 s` execution and `7.62..11.08 s` source).
The camera error is horizontal: Z remains about `0.022..0.027 m`, while XY
reaches about `0.168 m`. Base XY error remains only `0.058..0.068 m`, but base
yaw lags by `5..10 deg`; the arm/camera offset amplifies that yaw lag at the
camera. The reference `wz` repeatedly reaches the unchanged `+/-0.4 rad/s`
limit. Direction-recovery telemetry remains entirely inactive because base XY
error never reaches its `0.2 m` activation threshold, so this is not another
reverse-recovery failure.

The current execution clock cannot be redistributed. Recomputing every
transition's minimum duration from base linear/lateral/yaw, riser, and proxy
limits shows all `589/589` transitions at their demand-derived floor. Total
reclaimable slack is approximately `1.3e-13 s`. Slowing the late interval while
preserving the current geometry would therefore either exceed the frozen 2.0x
duration limit or violate a rate limit.

The one bounded candidate is `case74_localized_heading_relief_v1`: modify only
the derived horizontal smoothed geometry and resulting base-heading allocation
over anchors `394..572`, with a tapered boundary, to reduce late yaw curvature
and commanded yaw-rate saturation. Recompute the complete execution schedule
and feed-forward arrays from demands. The candidate must preserve:

- all authoritative source positions, timestamps, and semantic attitudes;
- source start/final targets, ordered one-to-one anchor mapping, and 590 states;
- unchanged camera Z and derived gimbal attitude contract;
- path-length drift within 5%, source-polyline deviation within 0.15 m p95 and
  0.25 m maximum, and no opposed source segment;
- execution/source ratio at or below 2.0x and portfolio median at or below
  1.5x;
- base `0.4 m/s`, yaw `0.4 rad/s`, proxy 24 deg/s, riser 1.0 m/s, transition,
  workspace, and all existing dynamic/thermal thresholds.

CPU tests must prove anchors outside the tapered edit window are unchanged,
source arrays are bit-identical, the clock is strict and unambiguous, all
derivatives match the new clock, and every integrity/kinematic gate passes.
The first candidate is diagnostic-only and `valid_for_training=false`.

## Exact continuation

Review the bounded `case74_localized_heading_relief_v1` contract above. If
accepted, implement only the CPU derivation and its negative/invariant tests,
then generate one fresh case-74 candidate and recompose the all-79 portfolio.
Do not relax the `0.15 m` p95 gate or launch another canary until the CPU
candidate, portfolio count/median, and hashes are reviewed.

Do not start residual capture, BC, or PPO. `valid_for_training` remains false
until representative and then accepted-corpus deterministic qualification is
complete.

## 2026-07-18 localized-relief result

The CPU derivation and negative/invariant tests were implemented and pushed at
`b0b0f300543bbc0e140f472ee4c9d3142284a906`. The authoritative `.98` suite
passed `291/291` before generation. The selected candidate changes only XY
geometry for interior anchors `395..571`; the maximum displacement from its
parent is `0.033328 m`. Camera Z and all authoritative source positions,
timestamps, semantic attitudes, and anchor indices are byte-identical.

```text
20260718_case74_localized_heading_relief_v1_cpu

0acc088a695ff53f9eccfde73107b0748e5de12ffbb6b048efa467455071bf90  case_0074_localized_heading_relief_v1.npz
e0242123a87e1550cf92f85065b8c9adc54543c7dd4a1813c9867330f7f03d9a  manifest.json
df3ee7bc8f32171d5e2f14d28081e378ad3589965b938e4d52f903dc3c7a9ee9  summary.json

execution/source ratio:                     1.960151
kinematic position p95 / max:               0.118713 / 0.122610 m
path-length relative drift:                -0.040477
source-polyline p95 / max:                  0.034883 / 0.064719 m
selected sigma / blend:                    24.0 / 0.75
localized anchors:                         394..572
```

The candidate replaced only case 74 in a new all-79 portfolio. An independent
audit recomputed every plan hash and case JSON, verified 79 contiguous cases,
and confirmed that only case 74 changed. The portfolio admits exactly 70 cases
and preserves the accepted duration median `1.4997940737652151`.

```text
20260718_smoothed_plan_all79_v7_case74_relief_cpu

0fe4b517d2629a1bca413162378708c2985cf5a42a1da8746de0a662f2fab00c  manifest.json
02facb78f65dc39ebc4170fcd0f4c9a5c1a745385d2670bf01100d22351d7868  summary.json

accepted: 70
rejected: [1, 27, 29, 35, 38, 39, 40, 45, 71]
Isaac / capture / BC / PPO: false / false / false / false
valid_for_training: false
```

## Localized-relief dynamic result

One isolated deterministic canary ran from the fresh hash-bound wrapper at
`f9c0435ec7dcfaf90f5d18b95046a280503bf097`. It completed the full execution
clock without termination or saturation. Thermal and residual-label envelope
gates passed, but dynamic quality still failed only `position_p95_bounded`.

```text
20260718_gate_c_smoothed_case74_relief_v2_exclusive

47075cb6513957e78800a17c46be013764758f524d205c73a5d2f4ec7f9ef89b  admission.json
265e851337617c278e7443b80fd34060b5e2eb9da918dfb13059e4c6701a5514  gates/case_0074.json
1062256cc73d6b51fab2e0cd9c3788567a019c7ba3c715bebdb1542ed41ea513  logs/case_0074.log
6b9558cee28449ba2175e643387f8a7fe2745dc6a12669051e63f12b2a07a653  summary.json

source / execution / wall:                 11.373883 / 22.294527 / 43.08 s
completed steps:                           8616
position p95 / max:                        0.161081 / 0.169124 m
pitch p95 / max:                           6.406132 / 6.601540 deg
attitude p95 / max:                        0.163848 / 0.224864 deg
dynamic / thermal / label envelope:        fail / pass / pass
termination / action saturation:           none / zero
dataset / residual applied:                none / false
```

Compared with the v1 canary, position p95 improved by only `0.002616 m` and
peak base-yaw error fell from `9.882863 deg` to `9.134248 deg`. In the dominant
phase `17.2..19.0 s`, mean yaw error fell by about `0.55 deg`, while mean base
XY error increased slightly. More path smoothing is therefore not the next
preferred intervention.

The trace instead shows a low-level yaw-rate tracking deficit. Near the peak,
the outer loop requests approximately `-0.3685 rad/s`, while measured yaw rate
is approximately `-0.1915 rad/s`; no action saturation occurs. The next
bounded candidate changes only cascaded-controller `wz_kp` from `0.25` to
`0.40`. A pure CPU regression at that sealed state proves common/balance action
is unchanged, yaw action changes in the corrective direction by approximately
`0.0266`, and remains below the `0.8` limit. The wrapper and tests are pushed at
`f6262e8a17760c6f1bec74fac806a3ae88678f7a`; the authoritative `.98` suite
passes `292/292`.

After the separate two-wheel-balance task released its bounded CPU job, the
riser worktree was synchronized to clean pushed commit `cedc1ee`, and the full
authoritative `.98` suite passed `293/293`. The v3 canary then ran exclusively,
completed both clocks, and released the GPU cleanly. Increasing `wz_kp` to
`0.40` improved the same metrics but did not cross the unchanged p95 gate.

```text
20260718_gate_c_smoothed_case74_relief_wzkp040_v3_exclusive

51fdbbbba494390e7afcf73e149cc2cb8692a509a8ff9fafd1f7fec14b567785  admission.json
e9a9c1052c537b997d2b9eb0d9c651bf9163ace14eb38b7d9b656836b2f802e7  gates/case_0074.json
40ec2089a355e4cc5c7292825ccb33154567c046a89870d5a86d348db41c114b  logs/case_0074.log
9a495b13943b55365ceb967a2d6268edf0d4fb9fc4110731d246578ca0889cc0  summary.json

source / execution / wall:                 11.373883 / 22.294527 / 43.155 s
completed steps:                           8631
position p95 / max:                        0.158452 / 0.166571 m
peak base XY / yaw error:                  0.153066 m / 8.137107 deg
pitch p95 / max:                           6.420319 / 6.560083 deg
attitude p95 / max:                        0.163086 / 0.225301 deg
dynamic / thermal / label envelope:        fail / pass / pass
termination / action saturation:           none / zero
dataset / residual applied:                none / false
```

The only failed check remains `position_p95_bounded`: `0.158452 m` versus the
unchanged `0.15 m` limit. The hard interval remains phase `17.6..19.0 s`, where
the requested yaw rate is about `-0.36 rad/s` and the achieved rate remains
about `-0.19..-0.20 rad/s`. Compared with v2, p95 improved by `0.002630 m`,
maximum error by `0.002553 m`, and peak yaw error by `0.997141 deg`; this is a
monotonic low-level yaw-authority response rather than a new geometry failure.

One second and final scalar-only candidate changes `wz_kp` from `0.40` to
`0.90`. At the sealed initial, hard-turn, and reverse-turn states, CPU tests
prove the common/balance action is identical, the yaw delta always opposes the
measured yaw-rate error, and absolute yaw action remains below `0.52`, versus
the unchanged `0.8` action limit. The v3 authorization is retired. A future
canary may use only the fresh namespace
`20260718_gate_c_smoothed_case74_relief_wzkp090_v4_exclusive` after clean
commit/upstream, full CPU-suite, GPU/process, namespace, and thermal preflight.

Regardless of the v4 result, do not start accepted-corpus qualification,
residual capture, BC, or PPO automatically.
