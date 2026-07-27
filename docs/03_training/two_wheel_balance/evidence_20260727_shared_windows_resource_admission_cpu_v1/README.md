# Shared Windows resource admission v1

This CPU-only checkpoint binds a fail-closed shared-host resource guard to the
pending case-7 corrective-label capture route at implementation commit
`2b4bd88791a2e183b2a9bd4b9d4a0e8374b49fe1`.

The guard runs before namespace creation and authorization-token consumption.
It requires:

- no active Siemens NX, SolidWorks, or Creo frontend process;
- at least `12 GiB` free Windows physical memory;
- at least `16,384 MiB` free GPU memory;
- a valid single-GPU and Windows process/memory probe.

These are conservative shared-machine launch-admission thresholds, not robot
or training hyperparameters. They do not change the case-7 plan, reset seed,
controller, wrench/corrective profiles, dynamic gates, action scales, capture
split, or playback command.

The live `.98` report fails closed because `ugraf.exe` and `sldworks.exe` are
active, Windows free memory is below `12 GiB`, and GPU free memory is below
`16,384 MiB`. It records `runtime_started=false` and
`authorization_consumed=false`.

Focused tests passed `21 passed, 2 warnings in 0.70 s` locally and
`21 passed, 2 warnings in 10.73 s` with native Windows Python on `.98`.
The tokenless `.98` route preflight passes with `20` pinned identities while
runtime and label-capture authorization remain false. The committed
corrective-capture command-equivalence audit still passes all eight checks and
reports case 7 as command-compatible. No token, runtime namespace, Isaac/GPU
workload, label capture, conversion, corpus merge, BC, PPO, checkpoint, or
training output was created.

The full macOS test command is not authoritative for this worktree: six
unrelated `mobile_mm` modules cannot collect because `gymnasium` is absent.
After excluding those modules, the broad local run produced `1,368 passed`
and `16 failed`; all failures require remote-only generated URDF/mesh/hardware
artifacts absent from this registered worktree. The full Windows suite was
deliberately deferred while CAD users occupied the shared host.
