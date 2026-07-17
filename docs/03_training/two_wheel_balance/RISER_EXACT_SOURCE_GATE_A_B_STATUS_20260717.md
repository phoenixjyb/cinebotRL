# Riser exact-source Gate A/B status - 2026-07-17

## Decision

Gate A and Gate B are complete. A hash-audited portfolio now has `71/79`
kinematic candidates, so the numerical count requirement for Gate C is met.
Gate C dynamic qualification has not started. Residual capture, BC, and PPO
remain unauthorized.

The authoritative package is admitted only as reference input:

- package: `gikWBC9DOF_exact_source_reference_all79_20260717`;
- manifest SHA-256:
  `f265aa1bdd1cd6c762fd6e5367c00c7abcb7b19dea76bb30c6311885d2f3237d`;
- contract: `exact_source_v1`;
- source cases/poses/path/duration: `79`, `71,038`, `618.304657 m`,
  `1,388.425333 s`;
- `quality_qualified_teacher=false` and `valid_for_training=false`.

## Gate A - admission hardening

`validate_riser_exact_source_manifest.py` now has separate
`reference_ingest` and `training` modes. Reference ingest verifies the pinned
package hash, contiguous cases, every bundled source hash, pose counts,
strict original timestamps, finite normalized `xyzw` quaternions, and the
explicit non-training flags. Training mode requires downstream mapping,
quality, initialization-separation, plan/source hash, and clean-lineage proof.

The real package passes reference ingest on `.98` and is rejected by training
mode with exit code `6`. Unit tests also reject modified source files,
timestamp replacement, duplicate cases, missing/reordered anchors,
initialization leakage, incomplete mapping, and quarantined lineage.

## Gate B - integrity-preserving plans

The new exporter stores two separate contracts in every NPZ:

- immutable source arrays: original `source_time_s`, positions, and semantic
  DFR quaternions in source `xyzw` order;
- explicit execution arrays: retimed increasing schedule, retargeted state,
  and a complete source-anchor-to-execution index map.

There is one execution state per source anchor in this version (`M=N`). The
map is exactly `0..N-1`; initialization is a separate empty segment and is not
an anchor or label. The target remains semantic `ee1_tool`, forward axis `+Y`.
Physical proxy/gimbal states are downstream diagnostics only.

Execution timing is allowed to differ from source timing but is explicit. It
is conservatively stretched for horizontal, vertical, attitude, and
nonholonomic heading-rate demand. Source timestamps remain verbatim and are
never reconstructed from a nominal sample rate.

## Canary result

Namespace:

```text
/mnt/g/wSpace/cinebotRL-two-wheel-riser/artifacts/two_wheel_riser/
20260717_exact_source_plan_canaries_v3_heading_retimed
```

Episodes 1, 4, and 7 preserve all `1,642` anchors and `10.109224 m` of source
path. Mapped position error is exactly zero. Ep7 passes the provisional pure
kinematic gate; ep1 fails only proxy-rate quality; ep4 remains above the
position threshold. These failures do not invalidate source integrity and do
not authorize dynamic capture.

## All-79 result

Namespace:

```text
/mnt/g/wSpace/cinebotRL-two-wheel-riser/artifacts/two_wheel_riser/
20260717_exact_source_all79_plans_v1_heading_retimed
```

Plan manifest SHA-256:

```text
940434d8caa5f85eb8c67d38d09a0894927a50b51fbb380b570d9e724fffe001
```

Results:

- exact-source integrity: `79/79`;
- NPZ count: `79`;
- preserved source poses: `71,038`;
- preserved source path: `618.304657 m`;
- preserved source duration: `1,388.425333 s`;
- explicit execution duration: `7,279.140145 s`;
- provisional pure-kinematic passes: `28/79`;
- provisional pure-kinematic rejects: `51/79`;
- dynamic Isaac qualification: not started;
- residual labels: not collected;
- BC/PPO: not started.

Provisional kinematic passes:

```text
2,3,7,8,9,10,11,12,14,15,19,23,24,36,37,45,46,47,48,49,50,53,56,57,61,62,63,77
```

Provisional kinematic rejects:

```text
1,4,5,6,13,16,17,18,20,21,22,25,26,27,28,29,30,31,32,33,34,35,38,39,40,41,42,43,44,51,52,54,55,58,59,60,64,65,66,67,68,69,70,71,72,73,74,75,76,78,79
```

The all-79 plan manifest is also rejected by training-mode admission, as
required. It has integrity evidence but no dynamic quality authorization.

## Superseding Gate B v2 and targeted recovery

The v1 result above is retained as historical evidence. Structural changes
were then made without changing any source anchor or acceptance threshold:

- added preview steering so the nonholonomic base corrects cross-track error
  before reaching the current waypoint;
- coupled explicit execution timing to selected base, riser, and RS4 proxy
  demand;
- selected a constant vertical placement within the fixed camera envelope
  `[0.60, 1.80] m`, including safe negative shifts when required;
- retained older strategy labels in the loader so historical v2 plans remain
  composable and auditable.

The upstream vertical audit is bound by SHA-256
`51c2e60e11e53cf8b1884d0d01bec61df4f8cacec9d7cf35bb3b5b08f81447ab`.
It independently proves `78/79` trajectories are span-compatible. Ep27 is the
only irreducible vertical reject: source span `2.145973 m` exceeds the robot's
`1.20 m` camera-height range.

All-79 v2 namespace and hashes:

```text
20260717_exact_source_all79_plans_v2_preview_coupled
manifest c37ab4762e91492309f7c80a54df61379137282395f8bc3b482adb605ceca296
summary  58f2a1273a75d60c51277ca752d4a97caaff33d8cd129b37e63e0d9b60de0afb
```

V2 preserves `79/79` exact-source plans and improves the unchanged kinematic
gate from `28/79` to `66/79`. Targeted recovery then passes cases
`52,69,74,75,76` using an explicit `6 deg/s` proxy-retiming design margin;
the unchanged acceptance limit remains `24 deg/s`.

Final portfolio namespace and hashes:

```text
20260717_exact_source_all79_portfolio_v4_threshold71
manifest 851a7b2751cd397ba35daf57d1a8c6971fb14ed0186683af48d3c6109090570a
summary  688b5bc23d801705c3132c511f009e1deb3d2af0a16a2a3ae33467764272db83
training rejection audit 77a0b80c1e6a4b97da67b09a2f381d86c774758e9275709af787505407f7e8ee
```

Final Gate B result:

- exact-source integrity: `79/79`;
- unchanged kinematic gate: `71/79`;
- Gate C candidates: `71`;
- explicit rejects: `6,13,18,21,22,27,55,64`;
- dynamic quality-qualified cases: `0`;
- residual labels: none;
- training admission: rejected;
- BC/PPO: not started.

The ep77 upstream seed manifest
`2dc31d86325155fafb0dc3afe8870f9ae32ea3c58d5ed7d2671f43aa2d4d7404`
was independently checked. Its source timestamps and positions match exactly;
its seed and riser base paths are `2.125 m` and `2.122 m`. It confirms ep77 as
a useful planner canary but remains `valid_for_training=false` and contributes
no learned actions.

## Stop and next action

The `>=70` pre-dynamic count rule is now satisfied by 71 hash-audited plans.
The next bounded task is Gate C deterministic dynamic qualification of those
71 cases from one clean code commit. Gate C must preserve the frozen LQR and
all existing safety gates, record dynamic rejects honestly, and collect no
residual label from a failing case. BC and PPO remain blocked until dynamic
qualification, clean residual capture, case-disjoint split, command
reconstruction, zero-clipping, and untouched holdout gates all pass.

## Gate C canary correction and current stop

The first canary namespace, `20260717_gate_c_canary_v1`, is preserved as
provisional audit evidence only. Cases 1 and 52 passed every physical dynamic
check, but their JSON mislabeled retimed execution duration as source duration.
Case 74 then stopped because the runtime normalized a prospective residual
label against the frozen `[0.30, 0.40, 0.10]` scale before completing dynamic
evaluation. The first normalized vector was
`[1.00200645, 0.19818327, 0.0356378]`; no failing action or dataset was written.

This was a gate-order bug, not a physical dynamic reject. The correction is
sealed beside the v1 evidence as `GATE_ORDER_CORRECTION.json` with SHA-256
`dbe43eadcfceac7a084a76237bf2edacd95cf862ded51aba15b92c3a62acb6dc`.
The timing-schema repair is commit `af4a4da`; raw residual diagnostics were
decoupled from Gate C dynamics in `d4c2097`. Gate C now reports independent
`dynamic_quality_passed` and `residual_label_envelope_passed` outcomes. It
does not clip, widen scales, apply raw labels, or write residual datasets.

The first reseal attempt was interrupted because a differential whole-body
Isaac run already owned the GPU. Namespace
`20260717_gate_c_canary_v2_timing_resealed` is explicitly
`interrupted_shared_gpu`, has no completed case, and is invalid for Gate C.
Commit `f08271e` adds a live exclusive-GPU admission guard; it was proven to
exit `5` before creating an artifact namespace while the competing process was
active.

The differential run then explicitly released the GPU. The corrected riser
continuation started case 74 alone before case 77 in a fresh exclusive
namespace:

```text
20260717_gate_c_case74_77_v3_exclusive_timing_resealed
```

Case 74 is the first final Gate C dynamic reject. Evidence hashes are:

```text
admission     bd555b2b2298f0f6591b5f497e858e7ab3bfd66707324aa3932331a0f909c8b7
case JSON     9bec49cf68d37d100b800e6505f5d0e5b6df2d1af30cd5f4e89bbe10d7794eb4
runtime log   eddd6b7532a33e4b1d8dc6a8baf6bbe372945239b898f89c30fa04ee02b02875
summary       b4e6d3bd3e0ec8cebf0b1646fa57d6b65e527dddb68c0a9458d33689da29ad2b
final status  b6bbd2dc25783ddff8364bafea1a23b06555d7f2dfe089095dfad29304cde4ee
```

The two clocks are correctly separate: source `11.373883 s`, execution
`188.546638 s`. Runtime stopped at phase `46.961485 s` after `13,659` steps
on one forbidden-body contact. Position p95/max reached
`0.546011/1.842571 m`, pitch max `15.802575 deg`, attitude max
`32.765379 deg`, and proxy servo error max `719.887930 deg`.

The residual-label result is independently false but did not cause the
physical failure. Raw maxima were
`[0.4011729574,0.3081566074,0.0125294506]`; normalized maxima were
`[1.3372431914,0.7703915184,0.1252945059]`. The raw label was not applied,
the executed residual action stayed `[0,0,0]`, and no dataset, BC, or PPO was
started. Case 77 did not start. The final ownership audit records no playback
owner after the run.

CPU diagnosis found a continuous-joint representation bug in the ideal DJI
proxy adapter. Near failure, semantic yaw was `+401.749 deg` while PhysX
reported the equivalent `-317.933 deg`; the runtime treated the branch
difference as about `720 deg`, saturated proxy effort, and destabilized the
reverse recovery. The repair preserves the unwrapped semantic target for
attitude continuity and audit, but sends the nearest equivalent yaw branch to
PhysX and computes continuous-yaw servo error modulo `2*pi`. The exact branch
pair is regression-tested, and the full CPU-only riser/two-wheel suite passes
`154` tests. The canary runner also now seals a summary when Isaac returns a
failed JSON with process status zero.

A sealed portfolio-wide CPU audit is stored at
`20260717_gate_c_continuous_yaw_scope_audit_v1`. Summary SHA-256 is
`7e357237237cb459a5b5c47f630852b4a179b955e022f4d55835c4949413fcbe`;
CSV SHA-256 is
`67bbfe164f6376189843517e9195437fb90b51e0186e8d1dfa32c4db15fc55cc`.
It proves `45/71` admitted plans require nearest-branch handling and all 45
contain canonical branch crossings. Case 71 has the largest possible naive
branch discrepancy at `720 deg`. Cases 74/75/76 share the largest valid
unwrapped waypoint step at `178.000692 deg`, making case 74 the strongest
bounded structural canary. Every plan hash and continuity check passes;
nearest-branch orientation error is at most `2.22e-16`. Multi-turn semantic
attitude remains authoritative and is not rejected or flattened.

The original v1 evidence above remains byte-identical. A stronger stateful
preflight is separately sealed at
`20260717_gate_c_continuous_yaw_stateful_preflight_v2`. It replays every one
of the 71 accepted semantic-yaw sequences from five equivalent PhysX branch
references (`-2,-1,0,+1,+2` turns), then verifies semantic delta preservation,
orientation equivalence, and a strictly positive margin below a half-turn
branch ambiguity. All 71 pass. Cases 74/75/76 remain the tightest at
`1.999307731 deg` margin. Summary SHA-256 is
`458e3f819c140d92d5664fa9118d0b1699b06ab55ac742ac3ee3ea7f48f2c2c6`;
CSV SHA-256 is
`88912658ecb77b1df462c65def2271d4ad566ad9fb1baac03b1fee3cae2d5786`.
This is CPU structural evidence only; it does not promote case 74 or authorize
case 77, residual capture, BC, or PPO.

The case-74 causal timeline is sealed as `CPU_FAILURE_TIMELINE_AUDIT.json`,
SHA-256
`d4d85cd7f2b18a376a40e20d41bc92194536688310ba7e2313818f8bc45d424e`.
Leaving the principal branch at `17 s` caused no error, so multi-turn attitude
is not itself a failure. At `58 s`, the target first exceeded one full turn:
semantic target `365.722 deg`, reported state `-353.857 deg`, false raw error
`719.580 deg`, and proxy effort exactly `10 Nm`. Position error was still only
`0.128 m` and pitch `0.275 deg`. Base XY failed at `59 s`, the governor stopped
at `61 s`, position failed at `64 s`, and forbidden contact occurred at
`68.29 s`. Nearest-equivalent mapping at the first fault is `-354.278 deg`
with only `-0.420 deg` wrapped error. This is strong temporal evidence, not yet
dynamic proof that the repair cures case 74.

A second hash-bound CPU audit separates valid reverse tracking from downstream
recovery. `CPU_REVERSE_RECOVERY_AUDIT.json` has SHA-256
`0a8e611a567640451103a80b05de955601718f5c63aa19d0e21fe9ab475df60a`.
Before the yaw fault, all `54` sampled reverse-motion points remained below
chassis-command saturation: maximum absolute `vx` reference was
`0.233591 m/s`, base error was at most `0.144051 m`, and camera position error
was at most `0.182615 m`. After the fault, `7/11` sampled commands saturated at
`0.4 m/s`, the recovery command reversed direction, and base error grew to
`1.361215 m`. The sealed 1 Hz trace therefore supports yaw-branch mismatch as
the primary observed precursor and full-scale bidirectional recovery as a
downstream response. It does not authorize a reverse-controller change; a
corrected-yaw dynamic canary is required first. The expanded CPU-only
riser/two-wheel suite now passes `157` tests.

After explicit authorization, corrected case 74 ran alone from pushed commit
`f07669c` in
`20260717_gate_c_case74_continuous_yaw_fix_v4_exclusive`. Evidence hashes are:

```text
admission   37371f47cac9d4056877c4c468fa60598f8d263300c86da00ea7f8d496b27040
case JSON   f5686d491cc3dff069a58f54fb974e718057993261a1924976e907b737fff65d
runtime log 8f02ad60c1d81fad32fa3a70274f1d8c00ce736950cab2df8d9631d93fa8f2cb
summary     3deb477ab7ee45cca9aafaac801b05ce4935523460d32f3cfd9e6b94cb37535f
```

The yaw repair is physically effective: forbidden contact is absent, proxy
servo p95/max is `0.114312/0.500490 deg`, proxy saturation is
`0.000044197`, pitch max is `7.645328 deg`, and attitude max is
`0.458524 deg`. Case 74 nevertheless remains a final Gate C reject. The
`377.1 s` bounded horizon ends at phase `159.901892/188.546638 s`; position
p95/max is `1.065665/1.094696 m`. There is no termination or dataset. Raw
residual labels remain unapplied, and case 77, residual capture, BC, and PPO
remain closed.

A new CPU audit at
`20260717_gate_c_case74_v4_motion_direction_audit/summary.json`, SHA-256
`39306409b323a8c6849d7d8de84e92641e20044e14a6c8b903330e3de5e524f5`,
localizes the remaining failure. In `206/378` sampled states, position
feedback commands motion opposite to the feedforward sign; this includes
`155/176` samples above the `0.25 m` position gate. At the `1.090352 m` peak,
feedforward is `+0.004799 m/s` while commanded velocity is `-0.054150 m/s`.
The legacy cross-track sign nearly cancels yaw recovery and commands only
`+0.107292 rad/s`; motion-command-aware steering requests the bounded
`-0.4 rad/s` direction instead.

The CPU-only candidate uses commanded velocity, smoothly blended over
`0.05 m/s`, to choose cross-track steering direction. Future traces expose
the reconstructed along/cross/yaw errors and both direction values directly.
This candidate changes no plan, source anchor, gate, LQR gain, residual scale,
or learned action and is not dynamically validated. A new case-74-only canary
requires separate explicit GPU authorization; case 77 and all learning remain
closed.
