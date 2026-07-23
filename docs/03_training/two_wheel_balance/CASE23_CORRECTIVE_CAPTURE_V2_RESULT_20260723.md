# Case-23 Corrective Capture V2 Result

Date: 2026-07-23

## Decision

The exactly-one authorized case-23 v2 corrective-label capture was consumed
and rejected before Isaac initialization. The v2 route must not be retried.

No capture archive, normalized dataset, corpus merge, BC, PPO, holdout
evaluation, or training was created or authorized.

## Failure

The guarded admission itself correctly identified case `23`, split `train`,
the exact plan/profile/pair identities, the clean runtime commit, and the
out-of-band authorization checks. Playback then called the generic capture
loader without forwarding the expected case. That loader retained its
case-30 compatibility default and rejected the valid case-23 admission:

`invalid corrective capture admission: case_split=false`

This is a pre-Isaac route-integration defect. It is not evidence against the
case-23 trajectory, corrective teacher, controller, robot physics, or GPU
runtime.

## Safety Outcome

- Runtime commit: `526952133a784ad653f4cfebd3e618a23fd4b291`
- Playback exit code: `2`
- Finalizer exit code: `6`
- Capture files: `0`
- Authorization token remaining: `0`
- GPU owners after finalization: `0`
- `normalized_training_dataset_created=false`
- `bc_authorized=false`
- `ppo_authorized=false`
- `training_started=false`
- `valid_for_training=false`

The immutable evidence is in
`evidence_20260723_case23_corrective_capture_v2_rejected_case_split`.

## Repair Boundary

Playback must receive an expected capture split independently from the
admission payload and must call the loader with both:

- `expected_case=corrective_teacher_case`
- `expected_split=corrective_teacher_capture_split`

The repair requires focused and authoritative CPU tests plus a fresh namespace,
contract, validator, wrapper, and separate explicit authorization. This
consumed v2 authorization does not authorize that future v3 runtime.
