# Case 7 paired-profile proposal

This CPU-only proposal derives case-specific corrective and deterministic
perturbation profiles from the hash-bound case-7 readiness evidence. It does
not reuse the case-23, case-6, or case-2 profiles.

The corrective envelope retains 50% of the observed raw case-7 residual. With
policy scales `[0.05, 0.05, 0.02]`, the resulting maximum residuals are
`[0.019165, 0.010077, 0.001263]`. Each channel reaches its maximum over a
`0.35 s` slew horizon.

The deterministic `20 N`, `20`-step longitudinal pulse starts at execution
phase time `2.851306 s` in low-motion source window `27..125`. Its nominal
duration is `0.10 s`, covers four source samples, and leaves more than
`15.16 s` for recovery. The local base/yaw headroom is
`0.100130/0.394402`; the local riser target remains approximately
`0.0702 m` above its lower limit.

For the provisional `28 kg` plant, the pulse is `2 Ns`. Its ideal free-body
displacement lower model is `0.003571 m`; this is an observability screen, not
a prediction of Isaac response.

The deterministic safety projection remains contractive. Base and yaw
residuals are unclipped over the full plan. Four initial negative riser
transitions project at the lower bound, while the selected pulse window is
fully unclipped. Any future label route must store only the effective
post-supervisor residual.

Every input, shape, formula, timing, headroom, plant, and closed-state check
passes. `pair_profile_cpu_ready=true`, but `runtime_route_implemented=false`.
No token, runtime authorization, GPU launch, label capture, conversion, merge,
BC, PPO, or training is enabled.

The focused profile, corrective-teacher, and perturbation suite passes
`47 passed, 2 warnings in 0.39 s` on macOS and
`47 passed, 2 warnings in 0.72 s` on `.98`. Both hosts have identical
builder, test, profile, and proposal hashes.

The authoritative `.98` CPU suite passes
`1205 passed, 12 skipped, 2 warnings in 137.71 s` at implementation commit
`9d4777ac7e9b90d3a57061c6f8e1bb2b38b8a0c5`.
