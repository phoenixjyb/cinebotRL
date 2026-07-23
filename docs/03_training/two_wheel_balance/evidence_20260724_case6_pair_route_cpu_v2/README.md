# Case-6 paired-canary route CPU evidence

This package records the CPU-only completion of the disabled case-6 paired
canary route.

The v2 contract uses a fresh namespace and pins 18 identities, including the
case-6 plan and profiles, controller assets, playback, validator, guarded
wrapper, and paired-result finalizer.

## Result

- CPU preflight passed on macOS and `.98`.
- All 18 identities matched their SHA-256 and Git blob IDs.
- The guarded route contains deterministic baseline and corrective rollouts,
  unchanged dynamic gates, exclusive GPU checks, bounded timeouts, and a
  fail-closed finalizer.
- Synthetic finalizer tests cover successful admission and rejection of weak
  improvement, clock mismatch, profile mismatch, capture, failed GPU release,
  and unauthorized admission.
- `.98` focused tests: 50 passed.
- `.98` full tests: 1141 passed, 12 skipped.
- `--execute` rejected before Python or Isaac with exit code 4.
- The v2 runtime namespace was not created.

No authorization token exists. This evidence does not authorize Isaac, label
capture, dataset creation, BC, PPO, or training.
