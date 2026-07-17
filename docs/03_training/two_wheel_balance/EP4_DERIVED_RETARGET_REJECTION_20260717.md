# Ep4 Derived Retarget Rejection

Date: 2026-07-17

## Decision

The current ep4 duration-preserving, reduced-anchor diagnostic is rejected at
derived semantic interval 50. It is not valid for dynamic evaluation or
training. Do not rerun this exact configuration, launch Isaac playback, or
start capture, BC, PPO, or residual learning from it.

This result does not modify or supersede the immutable 723-pose authoritative
ep4 reference. The evaluated package is explicitly derived and
training-disabled.

## Bound inputs

- Commit: `b9a8f7a98bd9ccc6114b865d5685a0e6d543c005`
- Code contract SHA-256:
  `b41b80c6d715b89bb51dcc4205e774a5cb90b955151fe8501dc05ddf9a179676`
- Derived package:
  `G:\wSpace\cinebotRL\data\gikWBC9DOF_ep04_relief20mm_stride4_localstride2_b9a8f7a_20260717`
- Derived manifest SHA-256:
  `c3e670981aff556e9dc4779d6b3364d2dc9babf1f5ab7a07af1b6fe4000a2241`
- Derived source SHA-256:
  `d3e07f94144f6da7e6f054c4a24fb0736d926c3b6e3efbf4ba06c3d6807616aa`
- Source/derived poses: `723 / 192`
- Duration: unchanged at `14.042191 s`
- Maximum local position derivation: `0.020000 m`
- Cartesian arc-length relative change: `-0.00007023`

The reduced package preserves an ordered subset, exact endpoint duration, and
the semantic DFR attitude contract. It does not preserve all authoritative
anchors, so it is a solver diagnostic and cannot satisfy the exact-source
Gate-1 contract.

## CPU regression

- Focused checkpoint/reference/retarget tests: `80 passed`, two pre-existing
  pytest configuration warnings.
- Repository-wide tests: `248 passed`, the same two warnings.

## Bounded result

Namespace:

```text
evaluation_results/two_wheel_exact_source_v1/
gate1_ep4_relief20mm_reservecap8_b9a8f7a_resume_v2_20260717
```

The run started from the sealed interval-50 checkpoint and ended with exit code
2 after approximately 1,163 seconds. The solver rejected interval 50 because
the best bounded solve exceeded the unchanged local position gate:

| Metric | Result | Gate | Decision |
| --- | ---: | ---: | --- |
| EE position max | `0.057255 m` | `0.050000 m` | fail |
| Camera attitude max | `0.033310 deg` | `0.100000 deg` | pass |
| Arm gravity max | `29.503980 Nm` | `29.510000 Nm` effective | pass |
| Equilibrium pitch max | `3.928917 deg` | `10.000000 deg` | pass |
| Physical gimbal margin | `0.016184765` | `0.005000000` minimum | pass |

The gravity-aware recovery family generated 231 candidates, selected none, and
did not change the accepted prefix. No candidate NPZ was emitted.

Checkpoint SHA-256 remained:

```text
d00f4aaf88a74f8c370de9ee094b723a222064317beb5233228fc325de705861
```

It still contains 95 states, 94 controls, 50 mapped derived anchors, and 15
retimed intervals. Independent validation confirms strict ordered mapping,
exact map-expanded execution clock, matching prefix positions/attitudes, all
array hashes, and `valid_for_training=false`.

## Ownership note

Preflight found no competing owner. A later CPU process from the separate
`G:\wSpace\cinebotRL-two-wheel-riser` workspace overlapped the final minutes.
It was not killed and did not access this repository or checkpoint. The solver
produced an explicit deterministic rejection, but the wall-time measurement is
not strictly exclusive.

## Admission state

- Offline derived candidate: rejected.
- Exact-source ep4 Gate 1: not passed.
- Physical playback: not started.
- Dynamic evaluation: not authorized.
- Training: not started and not authorized.
- Existing ep1 final-exclusive evidence remains the only sealed differential
  dynamic pass; it is still insufficient for corpus learning admission.

The next bounded task should be read-only structural diagnosis of derived
interval 50 and its corresponding raw ep4 neighborhood. Do not increase local
relief, decimate more anchors, relax the 0.05 m gate, or rerun the same search
without evidence identifying a distinct feasible branch.

## Read-only structural diagnosis

The derived waypoint map is explicit and hash-bound. Derived transition 50 is
derived anchor `49 -> 50`, which maps to authoritative raw anchors
`188 -> 190`. It is therefore earlier than the historical raw
`197 -> 198` hard transition; the current reduction/relief did not remove the
structural boundary, but moved the first rejection earlier.

At accepted derived anchor 49:

- physical-gate-safe state EE position error is `0.0473395 m`;
- peak arm gravity is `29.503574 Nm`;
- the next target displacement is `0.0174841 m` over `0.0609351 s`;
- derived Cartesian target speed is only `0.286930 m/s`.

The paired holonomic prior is not a connected escape branch. Its anchor-49 and
anchor-50 FK errors are `0.0876381 m` and `0.0880647 m`, and peak gravity is
`33.6457 Nm` and `33.6420 Nm`, respectively. Moving from the accepted state to
that branch in one interval would also require arm and chassis rates far above
the unchanged limits.

This evidence classifies the rejection as accumulated greedy branch continuity
at the gravity-position boundary. The current state is still gate-feasible but
has too little position/gravity slack for the next target. Gimbal margin,
camera attitude, target speed, and geometric smoothness are not the active
blockers.

The smallest justified solver experiment is an opt-in bounded multi-branch
lookback around this neighborhood: retain several hard-feasible base/arm states
for the preceding small window and rank them by future position reachability
plus gravity slack before committing a single prefix. It must preserve the
existing source/derived geometry, timestamps, rate limits, 0.05 m position
gate, gravity/pitch/gimbal gates, and deterministic ordering. It should first
be implemented and unit-tested as a CPU-only diagnostic; no new ep4 run is
authorized by this note alone.
