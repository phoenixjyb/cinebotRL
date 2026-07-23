# Case 6 paired-canary readiness audit

This CPU-only audit prepares the next diverse training-split candidate after
cases 30 and 23. It pins the selected case-6 exact-source smoothed plan and its
passing zero-residual dynamic evidence, but it does not authorize or launch a
paired canary.

The plan contains 807 source states and 806 transitions. It preserves a
15.942736 s source clock and uses a 17.737275 s execution clock. Camera height
stays within 0.600000-1.528812 m. The zero-residual run passed its existing
dynamic, thermal, and controller gates with 0.118125/0.127080 m position
p95/max error and 6.393436 degrees peak pitch.

Case 6 is not profile-ready. The plan reaches the base-linear, base-yaw, and
proxy-rate command limits, and its camera lever-arm correction saturation ratio
is 0.958710. A case-specific corrective and perturbation profile must therefore
be designed and CPU-reviewed. Reusing the case-23 profile is forbidden.

The only conservative low-motion window found is 0.105 s near the end of the
execution. This is a design constraint, not runtime admission.

Runtime, GPU launch, label capture, dataset conversion/merge, BC, PPO, and
training remain disabled. The already-consumed case-23 v4 capture remains the
sole v4 capture and is not retryable.
