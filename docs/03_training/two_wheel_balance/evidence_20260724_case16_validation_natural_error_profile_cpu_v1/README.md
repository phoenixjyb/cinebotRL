# Case 16 validation natural-error profile

This CPU-only evidence defines a structural corrective profile for held-out
validation case 16. The selected plan has no low-motion perturbation window and
already reaches the frozen base-linear, yaw, and proxy-rate limits, so the
profile uses the existing zero-residual tracking error rather than an external
wrench.

The profile retains the smaller of 40% and the case-specific p95 position-gate
margin fraction. For the sealed case-16 gate this is 40%, producing maximum
physical residuals of:

- `0.004255377317959039 m/s` longitudinal velocity;
- `0.007046567106873897 rad/s` yaw rate;
- `0.0010219869160504214 m` riser target.

The 0.40 s slew horizon and deterministic safety projection remain mandatory.
Projection is contractive and keeps every command inside the frozen limits.
The full plan projects 607 positive longitudinal transitions, 20 negative yaw
transitions, and 174 positive yaw transitions; no riser transition projects.
Only effective post-supervisor residuals may be assessed.

This is a held-out validation profile, not a teacher admission. It creates no
external perturbation, label archive, dataset, runtime namespace, authorization
token, BC job, PPO job, or training artifact. A separate CPU-only paired route
contract is required before any runtime authorization can be reviewed.

Implementation commit `77139d631ee05b3432d368ce478ff8f8af7bca93`
passes the 19-test focused suite on macOS and `.98`. Windows regenerates the
profile and proposal byte-for-byte. The authoritative `.98` CPU suite passes
`1278 passed, 12 skipped, 2 warnings in 162.65 s`.
