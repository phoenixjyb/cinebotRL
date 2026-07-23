# Case 2 paired-canary readiness audit

This CPU-only audit pins the selected case-2 exact-source smoothed plan and its
passing zero-residual dynamic evidence. It does not authorize or launch a
paired canary.

The plan contains 480 source states and 479 transitions. It preserves a
9.439314 s source clock and uses an 18.241928 s execution clock. The source
anchor map is one-to-one, initialization remains separate and empty, and
camera height stays within 1.174803-1.362249 m.

The zero-residual run passed its existing dynamic, thermal, and controller
gates with 0.139830/0.153514 m position p95/max error and 6.632458 degrees
peak pitch. The p95 margin is only 0.010170 m. Base-linear, base-yaw, and
proxy-rate commands reach their frozen limits, and the camera lever-arm
correction saturation ratio is 0.943907.

Case 2 is not profile-ready. No interval of at least 0.1 s satisfies the
existing conservative low-motion envelope. Reusing the case-23 or case-6
profile is forbidden. The next bounded task is a CPU-only structural
case-2 corrective-profile design that does not perturb saturated segments.

The focused readiness suite passes 15 tests on macOS and `.98`. The
authoritative `.98` CPU suite passes
`1149 passed, 12 skipped, 2 warnings in 124.40 s` at implementation commit
`b7d02c1a5b72e61394921f837d9845a850fc5e64`. Both hosts regenerate
byte-identical `summary.json` evidence at SHA-256
`5e346af90c7d4888914baf6bcf8adaef65957bb20a97a11b8666590ca9067f5c`.

Runtime, GPU launch, label capture, dataset conversion/merge, BC, PPO, and
training remain disabled. The consumed case-23 v4 capture was not repeated.
