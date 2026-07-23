# Case 6 paired-profile proposal

This CPU-only proposal derives case-specific corrective and perturbation
profiles from the hash-bound case-6 readiness evidence. It does not reuse the
case-23 profile.

The corrective envelope uses one reproducible rule:

`maximum_residual = observed_raw_case6_envelope * 0.75`

where the raw envelope is the zero-residual Gate C normalized label envelope
multiplied by policy scales `[0.05, 0.05, 0.02]`. The resulting maximum
residuals are `[0.028768, 0.007953, 0.001787]`. Each channel reaches its
maximum over a `0.30 s` slew horizon.

The deterministic `20 N`, `20`-step pulse starts at phase time
`17.185169 s`. Its nominal `0.10 s` phase interval fits inside the only
verified low-motion window, which ends at `17.290169 s`. The local base/yaw
headroom is `0.110688/0.182392`, riser-target headroom is `0.279227 m`, and
`0.452106 s` remains after the pulse for recovery.

For the provisional `28 kg` plant, the pulse is `2 Ns`, matching the accepted
plant-prior impulse envelope. Its free-body displacement lower model is
`0.003571 m`; this is an observability screen, not a prediction of Isaac
response.

Every input, shape, formula, timing, headroom, plant, and closed-state check
passes. `pair_profile_cpu_ready=true`, but `runtime_route_implemented=false`.
No token, runtime authorization, GPU launch, label capture, conversion, merge,
BC, PPO, or training is enabled.
