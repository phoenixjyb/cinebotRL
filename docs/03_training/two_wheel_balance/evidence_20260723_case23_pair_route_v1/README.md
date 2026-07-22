# Case-23 paired-canary route v1

This evidence records the CPU-only implementation and validation of the
guarded case-23 baseline/corrective-teacher pair route.

- Implementation commit: `1a626d37ebcf0b8e4fafc9ff814aaa2512c1a354`
- Reviewed controller parent: `9d79e355394b01ae330e5288d9c11f77bb920dfe`
- Namespace: `20260723_model_based_corrective_teacher_case23_pair_v1_exclusive`
- Contract SHA-256: `94e87b8eaae122f0c236c3812d9f2c184b54e1dd609713d52a9eb313ab59dd67`
- Contract Git blob: `33115952da2c70cadec2275b8561f4915d55b8ac`
- Authoritative `.98` suite: `808 passed, 11 skipped, 2 warnings`
- Canonical no-token preflight: passed
- Execute negative proof: exit `4`, `runtime_authorization_not_issued`

The contract pins the proposal, tranche selection, case-23 plan, perturbation,
corrective profile, robot assets, controller/runtime code, wrapper, validator,
and finalizer. It does not issue an authorization token. No namespace, Isaac
process, capture, dataset, BC, PPO, or training run was created.

`preflight_summary.json` is a compact durable summary. The canonical contract
and its validator remain the authoritative field-level evidence.
