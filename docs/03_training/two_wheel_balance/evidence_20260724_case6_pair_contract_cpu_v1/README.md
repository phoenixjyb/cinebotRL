# Case-6 paired-canary CPU contract evidence

This package records the CPU-only implementation and verification of the
fail-closed case-6 paired-canary contract.

The contract is hash-bound to the accepted case-6 selection, readiness audit,
smoothed plan, profile proposal, corrective and wrench profiles, provisional
28 kg LQR gains, robot assets, playback, and runtime support code.

## Result

- CPU contract preflight: passed on macOS and `.98`.
- Bound identities: 17/17.
- Focused `.98` tests: 26 passed.
- Full `.98` tests: 1133 passed, 12 skipped, 2 configuration warnings.
- `--execute`: rejected before Python or Isaac with exit code 4.
- Runtime namespace: not created.
- Runtime token: not issued.
- Execution route: intentionally incomplete.
- Isaac/GPU launch, label capture, dataset creation, BC, and PPO: not
  authorized.

This evidence does not authorize a case-6 paired canary and is not training
evidence.
