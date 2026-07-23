# Case-23 Corrective Capture Result

Date: 2026-07-23

## Result

The exactly-one authorized case-23 corrective-label capture was consumed and
rejected before Isaac simulation initialization. It must not be retried without
a new explicit authorization.

The admission contract passed at runtime commit
`037aa8dbeb04790276a0515a2203e3d575cb010b`: the token was mode `0600`, its
SHA-256 matched, the namespace was fresh, `HEAD == upstream`, the tracked tree
was clean, the GPU was exclusive, and the active simulation plant remained
`leadshine_400w_engineering_sample_v1`.

Playback then exited with code `2`. The shell wrapper used:

```bash
readonly OUTPUT_WIN="$WIN_ROOT\artifacts\two_wheel_riser\$NAMESPACE"
```

The backslash escaped the dollar sign, so the Python argument contained the
literal path `two_wheel_riser$NAMESPACE`. Argument validation could not open
`admission.json`; the simulation did not initialize and no dynamic gate ran.
The finalizer exited fail-closed with code `6`.

## Safety And Learning Boundary

- No case JSON or corrective-label archive was created.
- No normalized dataset was created and no conversion was authorized.
- No BC, PPO, or other training started.
- No source plan, controller command, residual scale, or plant identity changed.
- The authorization token was consumed before the Python process and is absent.
- No playback, Isaac, Kit, or compute process remains on the GPU.
- The rejected namespace is preserved as immutable attempt evidence.

The wrapper path is repaired by using `\\$NAMESPACE`, and the canonical
contract is returned to a no-token state. That repair is CPU-only and does not
authorize a second runtime.

## Evidence

The sealed evidence is under
`docs/03_training/two_wheel_balance/evidence_20260723_case23_corrective_capture_v1_rejected`.
Its `final_status.json` records the hashes of the copied runtime admission,
contract, logs, and exit-code files.

