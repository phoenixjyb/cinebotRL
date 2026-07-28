# Case-8 validation corrective-label capture v1

This evidence preserves exactly one authorized case-8 validation
corrective-label capture at runtime commit
`83b8fbc251f05e309db41ccb6acb5e373c2a5fa8`.

The capture completed all `6,607` steps and the full `18.1173174 s`
execution clock. Position p95/max error was `0.12587/0.14144 m`; attitude
p95/max error was `0.1511/0.2580 deg`; maximum pitch was `6.1623 deg`.
Action, riser, and proxy saturation ratios were zero.

SolidWorks and Siemens NX remained active. Launch admission passed with
`5.490 GiB` free Windows RAM and `10,262 MiB` free GPU memory. The runtime
monitor passed all `48` samples with minimum headroom `2.980 GiB` RAM and
`7,801 MiB` GPU memory.

The finalizer reopened every archive row, verified the validation split,
source/execution clocks, plan/profile/pair/runtime identities, reserved
action margin, supervisor telemetry, and initialization exclusion. The
archive is admitted only for a separately authorized CPU conversion.
Conversion, corpus merge, BC, PPO, holdout execution, and training remain
closed, and the capture remains `valid_for_training=false`.
