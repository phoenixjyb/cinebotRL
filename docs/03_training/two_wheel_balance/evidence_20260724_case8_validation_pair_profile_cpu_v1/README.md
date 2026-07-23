# Case 8 validation paired-profile proposal

This CPU-only proposal derives dedicated corrective and deterministic
perturbation profiles from the hash-bound case-8 validation readiness
evidence. It uses the audited profile formula engine, but does not reuse a
case-30, case-23, case-6, case-2, or case-7 profile file or parameter set.

The validation corrective envelope retains 40% of the observed raw case-8
residual. With policy scales `[0.05, 0.05, 0.02]`, the resulting maximum
residuals are `[0.015366, 0.008148, 0.001003]`. Each channel reaches its
maximum over a `0.40 s` slew horizon. The effective normalized action remains
below `0.307323`.

The deterministic `18 N`, `20`-step longitudinal pulse starts at execution
phase time `2.851306 s` in low-motion source window `27..125`. Its nominal
duration is `0.10 s`, covers four source samples, and leaves more than
`15.16 s` for recovery. The selected pulse window is fully unclipped.

For the provisional `28 kg` plant, the pulse is `1.8 Ns`. Its ideal free-body
displacement lower model is `0.003214 m`; this is an observability screen, not
a prediction of Isaac response.

The deterministic safety projection remains contractive. Base and yaw
residuals are unclipped over the full plan. Four initial negative riser
transitions project at the lower bound. Any future label route must store only
the effective post-supervisor residual.

Every validation-selection, identity, shape, formula, timing, headroom,
non-reuse, plant, and closed-state check passes.
`pair_profile_cpu_ready=true`, but `runtime_route_implemented=false`. No
authorization token, runtime, GPU launch, label capture, conversion, merge,
BC, PPO, or training is enabled.

The focused case-8/case-7/readiness suite passes
`34 passed, 2 warnings in 0.72 s` on macOS. Authoritative `.98` verification
is deferred until this implementation is committed and synchronized.
