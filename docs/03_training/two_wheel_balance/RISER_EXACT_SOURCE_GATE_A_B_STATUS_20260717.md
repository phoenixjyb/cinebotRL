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

All riser GPU/Isaac launches are now held while the differential session runs
its final exclusive rerun from commit `3e820e8`. Next riser continuation is the
fresh namespace `20260717_gate_c_canary_v3_exclusive_timing_resealed`, only
after explicit GPU release. Rerun cases `1,52,74,77`; stop only on
physical/safety/quality failure. A label-envelope failure keeps Gate D,
residual capture, BC, and PPO closed, but no longer stops Gate C dynamics.
