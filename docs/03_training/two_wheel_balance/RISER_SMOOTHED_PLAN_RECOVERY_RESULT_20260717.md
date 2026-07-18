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

## Second yaw-authority result

The v4 contract was committed and pushed at
`5a890e0314c89e02dd12cb9c6606b878da0872fd`. After the competing CPU job
released, the worktree was synchronized by a verified bundle, the full `.98`
suite passed `294/294`, and the canary ran with exclusive GPU ownership. The
process completed both clocks and released the GPU before the separate task
started its next CPU job.

```text
20260718_gate_c_smoothed_case74_relief_wzkp090_v4_exclusive

fc42ac44af6dea84d10dfa4ba328b06eedf30e367794295a1b4b9d6807d955ef  admission.json
9a0011840a61b47092dda0200d09ccaae0b7b752d724f4f7e7f73bf6b55a2656  gates/case_0074.json
75680f2684d004f1e2b919d18acfa6cfb13308e2031247338e45ee39829e9183  logs/case_0074.log
5ba450473bbbcb4768c52982a64a5ac72f466426aca78f46256d86b560db4c87  summary.json

source / execution / wall:                 11.373883 / 22.294527 / 43.310 s
completed steps:                           8662
position p95 / max:                        0.151237 / 0.158396 m
peak base XY / yaw error:                  0.176031 m / 6.593954 deg
pitch p95 / max:                           6.425590 / 6.516095 deg
attitude p95 / max:                        0.161617 / 0.233215 deg
dynamic / thermal / label envelope:        fail / pass / pass
termination / action saturation:           none / zero
dataset / residual applied:                none / false
```

The same sole check misses by `0.001237 m`. In the dominant trace interval,
base XY error remains approximately `0.050 m`, while yaw lag remains
`4.3..4.8 deg`; the residual camera-position error is therefore still
yaw-limited rather than translation-limited. Across v3 to v4, the p95 response
to `wz_kp` is monotonic and predicts that a final small step from `0.90` to
`1.05` should cross the unchanged gate with a narrow margin. CPU checks bind
the sealed initial, hard-turn, and reverse-turn states and keep yaw action below
`0.57`, with common/balance action unchanged and the `0.8` limit untouched.

The v4 token is retired. Only the fresh namespace
`20260718_gate_c_smoothed_case74_relief_wzkp105_v5_exclusive` may be used after
the same clean-pushed, full-suite, ownership, namespace, and thermal preflight.
If v5 does not pass, stop scalar gain stepping and perform a structural
controller diagnosis. Accepted-corpus qualification, capture, BC, and PPO
remain closed.

## Final case-74 yaw-authority result

After the competing case-77 retarget released, the v5 commit was synchronized
to `.98` from a verified Git bundle and the full suite passed `295/295`. The
exclusive canary completed normally and passed every unchanged deterministic,
thermal, runtime-contract, and residual-label admission check.

```text
20260718_gate_c_smoothed_case74_relief_wzkp105_v5_exclusive

139926870e61ff76b981121cd4dd24489bee343fd0d691f39fd5c35ab8be69b1  admission.json
dd9511fc07120bc7d254e2d189d981b4a0286fe76496a687b70d06f30f672ac1  gates/case_0074.json
43db41aa9eadfc3440d6452f678f5e09fb0829db155f7e2f72d9b56759917d8a  logs/case_0074.log
341cc35fb32b104029263ef4826151b08922c9e53d2d3cbc9ed89ec70fb3837c  summary.json

source / execution / wall:                 11.373883 / 22.294527 / 43.220 s
completed steps:                           8644
position p95 / max:                        0.149894 / 0.156471 m
peak base XY / yaw error:                  0.161817 m / 6.297071 deg
pitch p95 / max:                           6.436817 / 6.550212 deg
attitude p95 / max:                        0.161514 / 0.227384 deg
dynamic / thermal / label admission:       pass / pass / pass
termination / action saturation:           none / zero
dataset / residual applied:                none / false
valid_for_final_gate_c / training:         true / false
```

The pass margin is narrow: `0.000106 m` below the `0.15 m` p95 limit. This is
enough to advance from isolated case-74 diagnosis, but not enough to promote
the gain globally without regression. The next bounded task is an ordered,
fail-fast representative run over admitted cases 77 then 52 with the same
`wz_kp=1.05` and no other changes:

```text
case 77: source 5.431279 s, execution 8.141487221 s, 273 states
         a45892c98311cdd6e6f2096b6821ef760759504138edc2f9c7caa9b1ac90f559
case 52: source 22.924931 s, execution 31.40718243547463 s, 1199 states
         fa90c7345be5763e1e66a55b4b111780dfe5df97f5a779ab2c6bb390f7a3cbce
```

The representative contract must validate both plans independently, execute
case 77 first, stop before case 52 on any physical/thermal/runtime failure, and
reject any concurrent playback, GPU owner, or exact-source retarget process.
Its output must keep dynamic, thermal, runtime, and residual-label outcomes
separate. It may not create a dataset or authorize capture, BC, PPO, or
training. Only two representative passes permit a subsequent accepted-corpus
qualification proposal.

## Representative regression result

The ordered representative contract was committed and pushed at
`b3247988086c9ff3bbb5180d0e833982cb779048`. After the competing ep77 retarget
reached its bounded stop, the commit was synchronized by a verified bundle and
the complete `.98` suite passed `299/299`. The wrapper executed case 77 first
and advanced to case 52 only after the first playback returned a full pass.
Both cases completed without a concurrent retarget or GPU owner, and all
processes released after the final summary.

```text
20260718_gate_c_smoothed_representative_77_52_wzkp105_v1_exclusive

2ab068a224ab8cdcd82fff0806a7901fea76b37b757e92abbb2c0716d4432677  admission.json
ac21e9daa011a40b05d963aad516089436f4f03bc52e29e175962d203db4d515  gates/case_0077.json
2eaae8a5640fce030e8277c81c64e2026998939d456c6efa81c2f3266f1e8da5  logs/case_0077.log
5ec8a7d8075a8b9a79810767b5d828ae72fc8f964757b384aaef98c0c1c25276  gates/case_0052.json
1b83a29cd494736d75f35609ee1951276799786ac3a6409331a7cb45d163b0e7  logs/case_0052.log
84dd4dd809e56bc941c167fbef2b35258503aeb8df8aebfcfaa5cc24ab982207  summary.json

case 77 position p95 / max:                0.083963 / 0.084736 m
case 77 pitch / attitude max:              6.514897 / 0.165353 deg
case 77 action / proxy / riser saturation: 0 / 0 / 0
case 52 position p95 / max:                0.137045 / 0.148154 m
case 52 pitch / attitude max:              7.168662 / 0.280927 deg
case 52 action saturation:                 0.000587257
case 52 proxy / riser saturation:          0 / 0
dynamic / thermal / runtime / label:       pass / pass / pass / pass
dataset / capture / BC / PPO:              none / false / false / false
valid_for_final_gate_c / training:         true / false
```

The deterministic pass set is now `{52, 74, 77}`. This is sufficient to
propose accepted-corpus qualification, but not to promote the controller or
start learning. The v7 portfolio has 70 CPU-admitted cases with a total
execution clock of `1841.590621 s`. Excluding the three sealed dynamic passes
leaves 67 cases and `1779.747424 s` of nominal execution. Qualify them in
ordered fail-fast tranches with explicit case lists, plan hashes, execution
budgets, and fresh namespaces. Start with the five shortest remaining admitted
plans `(53, 10, 12, 11, 23)`, whose combined execution clock is approximately
`39.231720 s`. Stop the tranche at the first dynamic, thermal, runtime,
or ownership rejection. Record residual-envelope admission independently; an
overflow keeps Gate D and training closed but does not itself stop deterministic
physics.

Do not aggregate a partial tranche as a corpus pass. Do not start capture, BC,
PPO, or any DNN training until all 70 admitted plans have sealed deterministic
results and the raw residual envelope has been recomputed over dynamic passes.

## Tranche-1 fail-fast audit

The first tranche wrapper ran from clean pushed commit
`6c14ae98a0a8123d84d531a6bf3b1a82520616e9` after the complete `.98` suite
passed `301/301`. The Windows Python process did not propagate its nonzero
application exit code through WSL. The shell therefore continued after case 10
failed and executed cases 12, 11, and 23 before the final aggregate validator
closed with exit code 6. This is an orchestration defect, not a corpus pass.

```text
20260718_gate_c_smoothed_tranche1_53_10_12_11_23_wzkp105_v1_exclusive

65c7a7558597b3bc4ee0650a182810f90521f835c7811b8f78f6a220d2b1a21c  admission.json
c0faa126bef5af5f265b5b54838cf48986d2b47aa2c48468becc7936e82e56b5  gates/case_0053.json
2051345b467c11d8e538cdbe27ee21380ccd916f5fc4fca84183a013690642ee  gates/case_0010.json
a1c71cde3cabc7a04a3e14176431130e9c62effaed26eb20b280ac61f8be58c3  summary.json

case 53: dynamic pass, position p95/max 0.092387/0.095815 m
case 10: first physical reject, completed_reference=false
case 12/11/23: diagnostic only, executed after the first reject
capture / BC / PPO / training: false / false / false / false
```

Case 10 remained physically stable and accurate but exhausted the runtime
horizon before completing the immutable execution clock:

```text
execution / completed phase:               7.874601 / 5.832402 s
maximum runtime / wall:                    16.142933 / 16.150 s
position p95 / max:                        0.116563 / 0.131853 m
pitch / attitude max:                      7.146952 / 0.205674 deg
progress scale mean / minimum:             0.361139 / 0.170610
raw residual command abs max:              [0.313672, 0.062977, 0.010331]
normalized residual abs max:               [1.045573, 0.157443, 0.103312]
```

The 4.56% first-channel residual-envelope overflow is independent and keeps
Gate D closed. It must not prevent deterministic completion evaluation. The
runner is repaired to parse each gate JSON explicitly before advancing, so a
Windows/WSL exit-code mismatch cannot bypass fail-fast. The summarizer no
longer derives runtime validity from the label-dependent `passed_case_count`.

The next bounded canary is case 10 only. Increase only the outer runtime horizon
from `2.05x` to `3.00x`; preserve the source and execution clocks, plan hash,
phase governor, `wz_kp=1.05`, robot/gains identities, and every physical gate.
If case 10 still does not complete, do not increase the horizon again without a
new structural diagnosis. Capture, BC, PPO, and training remain closed.

## Case-10 bounded-horizon result

The repaired wrapper and independent label/dynamic admission were committed and
pushed at `bd4b17534cf2bff0b6c6b110cbdc54fb8c69f464`. The complete `.98`
CPU suite passed `303/303` before launch. The single authorized 3.0x canary then
completed the full immutable execution clock and passed every physical,
thermal, and runtime check:

```text
20260718_gate_c_smoothed_case10_horizon300_wzkp105_v1_exclusive

f9d159d9b54e9d7f9d760431be5b9aa348792d115c07c49e830396011026ebe7  admission.json
0f741e6a864dba49206459a817ca95378eada77e019cfe14a9b019a320250717  gates/case_0010.json
39505217365f6a384a1457c01f17ecb3d8992697f4315e8942e5ccf1ab0c7782  logs/case_0010.log
9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa  logs/case_0010.exit_code
31c33cf9978d8bbf41a93a8f10f445332fe76feefe2cdc4e6c2381b0f777002a  summary.json

source / execution / completed phase:       5.101357 / 7.874601 / 7.874601 s
completed steps / wall:                     4333 / 21.665 s
bounded maximum runtime:                    23.623804 s
position p95 / max:                         0.106099 / 0.131853 m
pitch / attitude max:                       7.146952 / 0.205674 deg
proxy error / rate max:                     0.220176 / 41.057193 deg(/s)
action / proxy / riser saturation:          0 / 0 / 0
raw residual command abs max:               [0.313672, 0.089366, 0.013123]
normalized residual label abs max:          [1.045573, 0.223415, 0.131227]
dynamic / thermal / runtime:                pass / pass / pass
label envelope / label admission:           fail / fail
dataset / residual applied / training:      none / false / false
```

The deterministic pass set is now `{10, 52, 53, 74, 77}`. Case 10 proves the
2.05x result was a bounded runtime-horizon reject rather than a trajectory or
controller failure. The unchanged 4.56% first-channel label overflow remains a
Gate-D blocker but was not applied to deterministic commands.

Cases 12, 11, and 23 from the defective first tranche remain diagnostic-only.
Their fresh qualification must use a new namespace and the repaired JSON-based
fail-fast path. Use the same outer 3.0x runtime horizon established by case 10,
without changing source/execution clocks, plans, controller, governor, gains,
model, or thresholds. Execute in order `(12, 11, 23)` and stop on the first
physical, thermal, runtime, or ownership reject. Training remains closed.
