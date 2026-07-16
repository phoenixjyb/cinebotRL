# Riser exact-source Gate A/B status - 2026-07-17

## Decision

Gate A and Gate B are complete. Gate C, residual capture, BC, and PPO are
blocked by the handoff stop rule.

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

## Stop and next action

The handoff requires at least 70 accepted episodes before a proposed
50/10/10 split. The current provisional quality count is 28, so starting Isaac
capture for labels, BC, or PPO would violate the stop rule.

The next bounded task is planner/retarget diagnosis on the 51 rejects. It must
improve nonholonomic position tracking and proxy-rate feasibility without
dropping anchors, changing source geometry, relaxing quality thresholds, or
falling back to quarantined plans. After a structural fix, rerun Gate B in a
new namespace and only enter Gate C if at least 70 cases pass the pre-dynamic
quality screen.
