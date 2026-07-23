# Case-23 Corrective Capture V3 CPU Review

Date: 2026-07-23

## Decision

**GO for a later, separately authorized, exactly-one case-23 v3
corrective-label capture.**

This review issues no authorization token and does not authorize Isaac, GPU
use, conversion, corpus merge, BC, PPO, holdout evaluation, or training.

## Repair

The consumed v2 route failed because playback called the generic capture loader
through its case-30 default. V3 requires a capture split independently of the
admission payload and calls the loader with:

- `expected_case=corrective_teacher_case`
- `expected_split=args.corrective_teacher_capture_split`

The v3 wrapper pins `--corrective-teacher-capture-split train`. Missing the
directory, admission, or split fails before `AppLauncher`.

## Identities

- Reviewed repair parent:
  `f444c7b88d84f66acf6857fff759ee618a7f633b`
- Implementation and authoritative CPU commit:
  `90d329cdb1ebaadefefd3696862873eb49f5fd37`
- Namespace:
  `20260723_model_based_corrective_teacher_case23_capture_v3_exclusive`
- Contract SHA-256:
  `990a20518288f2878fbd7c495dcdc17b8972ffab067f012db7e70a20cc9e3c7c`
- Contract Git blob:
  `6b25a90beb4b7f2c6f7ba769a60e246466c131dd`
- Consumed-v2 rejection manifest SHA-256:
  `2217f799fe2699fe10b5d2f458c2a2de099c9cfc1124669a622dc193161d0aeb`

## Verification

- Focused v3/rejection/runtime tests: `37 passed`, two configuration warnings.
- Authoritative `.98` suite: `949 passed`, `12 skipped`, two configuration
  warnings in `84.07 s`.
- `.98` no-token preflight passed at clean `HEAD == upstream == 90d329c`.
- V3 namespace remains absent.
- `runtime_authorized=false`
- `gpu_launch_authorized=false`
- `label_capture_authorized=false`
- `normalized_training_dataset_created=false`
- `bc_authorized=false`
- `ppo_authorized=false`
- `training_started=false`

## Exact Next Gate

A future instruction must explicitly authorize exactly one case-23 v3
corrective-label capture. It must use a new out-of-repository mode-`0600` token
and out-of-band lowercase SHA-256. The attempt stops after one pass or
rejection. A pass opens only a separate conversion audit.
