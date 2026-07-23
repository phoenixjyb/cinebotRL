# Case-23 Corrective Capture CPU Review

Date: 2026-07-23

## Decision

**GO for exactly one separately authorized case-23 corrective-label capture.**

This review does not issue an authorization token and does not authorize
Isaac, GPU use, dataset conversion, corpus construction, BC, PPO, holdout
access, or training. The committed runtime contract remains no-token and
fail-closed.

## Reviewed identities

- CPU-ready contract commit:
  `b52024d175404803744304144f659c934ad68664`
- Passed case-23 pair final SHA-256:
  `67c8e99a0629a4b1cb4a2981abfe8360c5d9979c4757582dab6d4fb22cd00deb`
- Case-23 plan SHA-256:
  `ad76ada4cdb9f874da615aa0c6e441be62d9a768b813c597c5dc4e20894042b6`
- No-token capture contract SHA-256:
  `1d290b6be77e86e69a5ecf025f616bf5cd3c53c336b2ca4d94185b76f5422756`
- Fresh namespace, still absent:
  `20260723_model_based_corrective_teacher_case23_capture_v1_exclusive`

## Runtime contract audit

The wrapper is pinned to case 23, the accepted case-23 smoothed plan, frozen
LQR gains, the tracked case-23 corrective and wrench profiles, and residual
scales `[0.05, 0.05, 0.02]`. It uses the model-based planner plus the bounded
corrective teacher, captures requested and effective post-supervisor commands,
and consumes a one-use authorization before Isaac when and only when a later
authorization commit provides one.

The current wrapper contains an empty authorization SHA-256 and exits before
Python/Isaac in `--execute` mode. Its preflight verifies clean
`HEAD == upstream`, the canonical committed contract blob, the reviewed-parent
lineage, every pinned source/asset hash, an absent output namespace, the passed
pair thresholds, closed holdouts, and closed learning paths.

The finalizer independently reopens the archive and requires:

- exactly case 23 and the train split;
- complete source and execution clocks;
- dynamic, thermal, controller, perturbation, heartbeat, and GPU-release gates;
- the exact plan, runtime, profile, pair, contract, and capture identities;
- 20 perturbation-active rows and normalized action magnitude below `0.95`;
- requested/effective command separation and clipping telemetry;
- no initialization rows and no legacy/raw/normalized capture path;
- dataset, BC, PPO, and training states still closed.

## Downstream audit

The real case-23 finalizer and converter path is now exercised without mocks.
The synthetic audit proves that:

- a correctly labeled case-23 archive finalizes and is only admitted for a
  later conversion review;
- the default case-30 conversion route rejects the case-23 archive;
- explicit `expected_case=23` conversion preserves case identity;
- effective post-supervisor actions, not requested actions, become targets;
- previous-action observation channels are rebuilt from the previous effective
  action;
- a case-30-labeled archive presented to the case-23 finalizer is rejected.

The corpus path remains independently closed until at least four admitted train
cases and two admitted validation cases exist. It excludes reserved holdouts
`[3, 5, 13, 19, 24]`, verifies every case-dataset SHA-256, keeps train and
validation case-disjoint, and leaves the merged corpus invalid for training
until a later BC admission review.

## Remaining runtime boundary

The next runtime change, if explicitly authorized, must modify only the
canonical contract and wrapper to bind one fresh mode-`0600`, SHA-256-verified,
one-use token. It must retain the same case, plan, profiles, controller, gates,
scales, namespace, maximum runtime, and capture-only semantics. Any drift or
existing namespace is a no-go.

After the capture, stop regardless of outcome. A pass opens a separate
conversion review; a reject opens diagnosis. Neither outcome starts BC or PPO.
