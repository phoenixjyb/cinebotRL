# Case-23 Corrective Capture V3 Result

Date: 2026-07-23

## Decision

The exactly-one authorized case-23 v3 corrective-label capture was consumed.
Isaac completed the trajectory phase, but post-execution archive validation
rejected the capture. V3 must not be retried.

No capture archive, normalized dataset, corpus merge, BC, PPO, holdout
evaluation, or training was created or authorized.

## Runtime Evidence

- Runtime commit: `71ed62558dc4588b4f9a39a3b598e3faf636bd5f`
- Completed steps: `3273`
- Phase/execution duration: `9.929694 s`
- Peak position error in the final heartbeat: `0.067912 m`
- Peak attitude error: `0.257883 deg`
- Peak pitch: `5.642538 deg`
- Action saturation ratio: `0.001222`
- Termination pending: `false`
- Capture files: `0`
- GPU owners after finalization: `0`

These heartbeat values show that the full execution phase was reached without a
pending safety termination. They are not a finalized dynamic-quality pass,
because archive validation raised before the gate result could be written.

## Failure

Playback loaded the admission with explicit case `23` and split `train`, but
then omitted those identities when calling `save_corrective_capture`. The save
API retained its case-30 compatibility default and rejected the case-23 rows:

`corrective capture mixes or opens an unreviewed case`

The v3 wrapper also invoked the v2 case-23 finalizer, so the final status
reported the obsolete v2 namespace. This is an independent evidence-routing
defect.

Neither defect is evidence against the trajectory, corrective teacher,
controller, robot physics, or GPU runtime.

## Safety Boundary

- The one-use authorization token was deleted.
- V3 retry authorization is `false`.
- `dynamic_quality_passed=false`
- `normalized_training_dataset_created=false`
- `bc_authorized=false`
- `ppo_authorized=false`
- `training_started=false`
- `valid_for_training=false`

The immutable evidence is in
`evidence_20260723_case23_corrective_capture_v3_rejected_save_route`.
A future attempt requires a fresh v4 namespace, corrected save and finalizer
routing, CPU validation, and a new explicit v4 authorization.
