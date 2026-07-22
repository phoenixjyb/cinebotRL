# Case-23 paired-canary readiness audit v1

This CPU-only audit closes two pre-runtime questions for the sealed case-23
pair route:

1. The perturbation loader now accepts a non-default case only when playback
   explicitly supplies the sole requested case. Case 30 remains the default.
2. The exact case-23 midpoint pulse is checked against plan motion, historical
   dynamic margins, the provisional 28 kg plant impulse envelope, and the
   passed case-30 paired precedent.

The audit recommends exactly one bounded case-23 baseline/candidate canary.
It does not predict the closed-loop response and does not authorize runtime,
capture, dataset creation, BC, PPO, or training.

Audit SHA-256:
`335e4cd0181edce3c1bb0dd7c0b13c99df040f648b788d65b0fce1a34bf3a198`

Key results:

- Pulse: `20 N` for `0.10 s`, or `2.0 N*s`, at `0.5 m` height.
- Midpoint local limit use: base `59.2%`, yaw `1.2%`, riser `7.7%`, proxy `12.2%`.
- Historical case-23 p95/max position margins: `0.04455 m` / `0.14436 m`.
- Historical case-23 pitch margin: `6.255 deg`.
- Case-30 same-pulse p95 improvement: `0.006324 m` (`4.456%`).
- Decision: `recommend_exactly_one_bounded_case23_pair_canary`.
