# Case 7 paired runtime route

This evidence seals the CPU-only case-7 baseline/corrective paired route at
implementation commit
`7f311139c93247b78e0e4e503525ebe002f825c0`.

The route binds 18 identities, including the selected exact-source plan,
readiness and profile evidence, case-specific corrective and wrench profiles,
the frozen LQR gains, robot assets, shared playback/runtime modules, wrapper,
validator, and finalizer. The case-7 reset seed is `20260723`.

The future paired execution order is:

1. Complete model-based planner plus exact-zero policy residual.
2. Complete model-based planner plus bounded case-7 corrective teacher.

Both runs use the same plan, seed, physics configuration, dynamic thresholds,
and deterministic `20 N`, `0.10 s` pulse. The candidate is unreachable unless
the baseline passes its dynamic gate and releases the GPU.

The `.98` preflight passes every contract check with all 18 identities
matching. The route regression suite passes `50 passed, 2 warnings in
22.87 s` on `.98`. The authoritative `.98` CPU suite passes
`1224 passed, 12 skipped, 2 warnings in 150.03 s`.

`--execute` returns code `4` before Python or Isaac with
`runtime_authorization_not_issued`. No authorization token or runtime
namespace exists. Label capture, dataset creation, BC, PPO, and training
remain disabled.

This is a reviewed execution route, not a dynamic result and not a teacher
dataset.
