# Case-23 Corrective Capture V4 CPU Review

Date: 2026-07-23

## Decision

**GO for a later, separately authorized, exactly-one case-23 v4
corrective-label capture.**

This review issues no authorization token and does not authorize Isaac, GPU
use, conversion, corpus merge, BC, PPO, holdout evaluation, or training.

## Repairs

V3 reached the end of the trajectory phase but failed before writing the
dynamic gate and capture archive. Playback omitted the admitted case and split
when it called `save_corrective_capture`, so the save API used its case-30
compatibility default. The v3 wrapper also invoked a v2-hardcoded finalizer.

V4 repairs both route boundaries:

- Playback passes `case=int(corrective_capture_admission["case"])`.
- Playback passes `split=args.corrective_teacher_capture_split`.
- The wrapper invokes the v4-specific case-23 finalizer.
- The finalizer pins the v4 namespace and case-23 capture archive.

The case-30 API default remains unchanged for compatibility. No controller,
plan, gains, corrective/wrench profile, robot asset, safety gate, or capture
schema changed.

## Identities

- Reviewed repair/evidence parent:
  `472130ef622ef90afd6f470783f834d014e41ac0`
- Implementation and authoritative CPU commit:
  `2eb9604b7e2c030a867d9ab64e536240561c652f`
- Archive-to-finalizer test and latest authoritative CPU commit:
  `d4fb8b4fea89f953a699e4a090d33049c49936dc`
- Namespace:
  `20260723_model_based_corrective_teacher_case23_capture_v4_exclusive`
- Contract SHA-256:
  `23d4a9a3ac3ef3f6a0c52d25c9c5b156deaa86ca2b7a79753a3e24360ddc1b36`
- Contract Git blob:
  `b133dd843999ad9cc8e52ccbc9ab3ed82c7c19d7`
- Consumed-v3 rejection manifest SHA-256:
  `ae0043063e508c119a346654550b6415519cc24f7e0220c2da14d3f08494cb2d`

## Verification

- Focused v4/runtime/evidence tests: `37 passed`, two configuration warnings.
- Real case-23 archive/save/finalizer/conversion-focused tests: `33 passed`,
  two configuration warnings.
- The real-path fixture reopens the v4 archive and requires every archive,
  dynamic-gate, namespace, and contract check to pass while conversion remains
  separate and training remains closed.
- Latest authoritative `.98` suite: `960 passed`, `12 skipped`, two
  configuration warnings in `82.03 s`.
- `.98` no-token preflight passed at clean `HEAD == upstream == d4fb8b4`.
- V4 namespace remains absent.
- GPU compute owners: `0`
- V4 runtime processes: `0`
- `runtime_authorized=false`
- `gpu_launch_authorized=false`
- `label_capture_authorized=false`
- `dataset_creation_authorized=false`
- `bc_authorized=false`
- `ppo_authorized=false`
- `training_started=false`

## Exact Next Gate

A future instruction must explicitly authorize exactly one case-23 v4
corrective-label capture. It must use a new out-of-repository mode-`0600` token
and an out-of-band lowercase SHA-256. The attempt stops after one pass or
rejection. A pass opens only a separate conversion audit.
