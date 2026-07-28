# Case-8 validation paired canary v2

This evidence preserves exactly one authorized case-8 held-out validation
baseline/corrective pair at runtime commit
`b9bf4b7192d80d92ecbd83951e5f10b9d4aeba9c`.

SolidWorks and Siemens NX remained active. Launch admission passed with
`5.280 GiB` free Windows RAM and `10,253 MiB` free GPU memory. The baseline
and candidate resource monitors collected `45` and `41` samples. Their
minimum observed headroom was `2.957/2.979 GiB` RAM and `7,803 MiB` GPU
memory. Neither monitor requested termination, and both observed clean
process exit.

Both rollouts completed the full `18.117 s` execution clock. The baseline
position p95/max error was `0.13114/0.14333 m`; the candidate result was
`0.12587/0.14144 m`. The candidate therefore improved position p95 by
`0.00527 m` or `4.02%` without violating the paired regression gates.

The wrapper created no label capture or dataset. Teacher admission, capture,
conversion, corpus merge, BC, PPO, holdout execution, and training remain
closed. A separate authorization is required before any case-8 corrective
label capture.
