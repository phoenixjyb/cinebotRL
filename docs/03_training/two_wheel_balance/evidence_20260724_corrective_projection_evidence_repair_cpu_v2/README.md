# Corrective projection evidence repair

This CPU-only checkpoint replaces the unreliable post-Isaac adapter injection
as the admission source for natural-error corrective pairs. The shared
extractor audits atomic fields already written by playback:

- requested normalized residual action maxima;
- effective post-supervisor residual action maxima;
- projection-delta maxima and affected-sample count;
- completed policy steps and the bound action scales.

It does not intercept or modify commands. Missing, non-finite, nonzero
disabled-route, out-of-bound, or inconsistent-count evidence fails closed.

The archived case-2 pair now has valid projection evidence over `9,204`
candidate steps with `65` projection-affected samples. It remains rejected:
position p95 improvement is only `0.0002637322 m` (`0.1886096%`), below the
unchanged `0.003 m` and `2%` gates.

The consolidated pending-route preflight passes for cases 7, 8, and 16 with
`62` pinned identities. macOS and `.98` produced byte-identical route reports
at SHA-256
`b58ebb7725f483777b732b05ff88ae0ec6aca0477ffc1f7cfe42de2872046f5d`.
The affected suite passed `68 passed, 2 warnings in 11.61s` locally and
`68 passed, 2 warnings in 83.31s` on `.98`.

Implementation commit
`d51702fa25ec90ffc8c64d23bd92ebbfe1b9c620` is pushed and synchronized.
No runtime token, namespace, Isaac/GPU workload, label capture, conversion,
merge, BC, PPO, checkpoint, or training run was created.
