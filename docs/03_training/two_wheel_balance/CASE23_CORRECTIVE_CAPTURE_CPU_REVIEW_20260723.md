# Case-23 Corrective Capture CPU Review

Date: 2026-07-23

## Decision

**GO for exactly one separately authorized case-23 corrective-label capture.**

This review does not issue an authorization token and does not authorize
Isaac, GPU use, dataset conversion, corpus construction, BC, PPO, holdout
access, or training. The committed runtime contract remains no-token and
fail-closed.

## Reviewed identities

- Superseded pre-drive-profile contract commit:
  `b52024d175404803744304144f659c934ad68664`
- Current CPU-ready contract commit:
  `eff05387a93fd8281ae32482121c67105d85819d`
- Passed case-23 pair final SHA-256:
  `67c8e99a0629a4b1cb4a2981abfe8360c5d9979c4757582dab6d4fb22cd00deb`
- Case-23 plan SHA-256:
  `ad76ada4cdb9f874da615aa0c6e441be62d9a768b813c597c5dc4e20894042b6`
- No-token capture contract SHA-256:
  `18210efd27ba7d6001dc2d81f070f95df5dacd36f78ae417184771d2208f05d8`
- No-token capture contract Git blob:
  `01a7f0e957b49c21ccaf68a91c8e476bfaa894aa`
- Active 400 W drive-profile evidence SHA-256:
  `39a700de3985175e4e8415f1f23beef4264b103daa7ce8847f4ac0fe69f879f7`
- Authoritative `.98` CPU suite at commit
  `b801ae02c6beb03cc05cfa70017683541057d23e`:
  `892 passed, 12 skipped, 2 warnings` in `81.10 s`
- Fresh namespace, still absent:
  `20260723_model_based_corrective_teacher_case23_capture_v1_exclusive`

## Runtime contract audit

The wrapper is pinned to case 23, the accepted case-23 smoothed plan, frozen
LQR gains, the tracked case-23 corrective and wrench profiles, the active
`leadshine_400w_engineering_sample_v1` drive profile, and residual scales
`[0.05, 0.05, 0.02]`. It uses the model-based planner plus the bounded
corrective teacher, captures requested and effective post-supervisor commands,
and consumes a one-use authorization before Isaac when and only when a later
authorization commit provides one.

The current wrapper contains an empty authorization SHA-256 and exits before
Python/Isaac in `--execute` mode. Its preflight verifies clean
`HEAD == upstream`, the canonical committed contract blob, the reviewed-parent
lineage, every pinned source/asset hash, an absent output namespace, the passed
pair thresholds, closed holdouts, and closed learning paths.

The drive-profile validator additionally requires the active actuator limit to
remain `300 N / 1.0 m/s`, the first-order 400 W thermal contract to remain
active, and the 750 W production candidate to remain disabled for simulation,
runtime, and training. A silent plant upgrade or any attempt to reuse this
capture contract after a plant switch revokes CPU readiness before Isaac.

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
