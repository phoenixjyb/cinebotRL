# Case-2 natural-error paired route CPU evidence

This package records the CPU-only completion of the disabled case-2
natural-error paired-canary route.

The contract pins 19 identities, including the exact-source case-2 plan,
zero-residual gate, natural-error corrective profile, safety projection,
frozen controller and robot assets, byte-stable shared playback, a
case-specific projection telemetry adapter, validator, guarded wrapper, and
paired-result finalizer.

## Result

- CPU preflight passed on macOS and `.98`.
- All 19 SHA-256 and Git blob identities matched.
- The shared playback and corrective-teacher runtime remain byte-identical to
  the historical case-23 v4 contract.
- The adapter observes requested and effective residual commands but returns
  the supervisor output unchanged.
- Baseline and candidate use the same plan, `20260718` case reset seed,
  controller, physics, and dynamic gates.
- No external wrench is used. Candidate execution requires a dynamically
  passing baseline.
- macOS corrective/runtime regression set: 320 passed.
- `.98` focused route tests: 28 passed.
- `.98` full tests: 1183 passed, 12 skipped.
- `--execute` rejects before Python or Isaac with exit code 4.
- The reserved runtime namespace was not created.

No authorization token exists. This evidence does not authorize Isaac, label
capture, dataset creation, BC, PPO, or training.
