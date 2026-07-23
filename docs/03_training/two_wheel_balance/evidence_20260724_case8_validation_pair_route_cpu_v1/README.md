# Case 8 validation paired runtime route

This evidence seals the CPU-only case-8 held-out validation route at runtime
commit `a1b04ca5d55dce74288b6d42b55c2117ed9f80d8`.

The route binds 19 identities: the validation selection, readiness/profile
evidence, exact case-8 plan, dedicated corrective and `18 N` perturbation
profiles, frozen controller/assets, shared playback, validation assessor,
wrapper, validator, and finalizer. The reset seed is `20260724`.

The future paired order is:

1. Complete model-based planner plus exact-zero residual.
2. The same plan, seed, physics, thresholds, and perturbation plus the bounded
   case-8 validation corrective profile.

The candidate remains unreachable unless the baseline passes the unchanged
dynamic gate and releases the GPU. The result is assessed as held-out
validation; it cannot admit a teacher, labels, a dataset, BC, PPO, or training.

Mac and `.98` preflights pass every route and document check with all 19
identities matching. The `.98` focused suite passes `27 passed, 2 warnings in
11.07 s`. The authoritative `.98` CPU suite passes
`1259 passed, 12 skipped, 2 warnings in 157.03 s`.

On `.98`, `--execute` returns code `4` with
`runtime_authorization_not_issued`, `python_started=false`, and
`isaac_started=false`. The runtime namespace remains absent.

This is a reviewed execution route, not a dynamic validation result and not a
teacher dataset.
