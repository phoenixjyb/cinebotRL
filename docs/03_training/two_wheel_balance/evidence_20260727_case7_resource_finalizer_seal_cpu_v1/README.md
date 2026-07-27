# Case-7 resource finalizer seal

This CPU-only checkpoint closes the evidence gap between the shared Windows
resource guard and case-7 capture finalization at implementation commit
`0f2de2b8175e59395cd61b45d37a49a071ad81e5`.

The case-7 finalizer now fails closed unless `resource_admission.json` is
present, hashable, schema-correct, generated before runtime and authorization
consumption, and proves the pinned CAD, Windows RAM, and GPU headroom checks.
The accepted identity and observed snapshot are copied into final status.

Focused tests passed `23 passed, 2 warnings in 0.82 s` locally and
`23 passed, 2 warnings in 13.20 s` with native Windows Python on `.98`.
The clean synchronized tokenless preflight passes all 20 pinned identities.
The command-equivalence audit passes and reports no case-7 command mismatch.
No plan, controller, profile, dynamic gate, action scale, split, or playback
option changed.

The fresh live guard remains closed: SolidWorks and Siemens NX are active,
Windows free memory is `5.761 GiB`, and GPU free memory is `10,491 MiB`.
Therefore no token, runtime namespace, Isaac workload, capture, conversion,
dataset, BC, PPO, or training operation was created. The user's authorization
for exactly one case-7 corrective-label capture remains unused.
