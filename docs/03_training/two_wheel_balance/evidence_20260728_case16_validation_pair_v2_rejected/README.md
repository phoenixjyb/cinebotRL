# Case-16 validation paired canary v2: rejected

This evidence preserves exactly one authorized case-16 held-out validation
baseline/corrective pair at runtime commit
`16bb0ef1fd9172fd76e9d75d7811317ba52840fd`.

SolidWorks and Siemens NX remained active. Launch admission passed with
`6.457 GiB` free Windows RAM and `10,182 MiB` free GPU memory. The baseline
and candidate monitors collected `67` and `62` samples. Their minimum
observed headroom was `2.986/3.126 GiB` RAM and `7,748 MiB` GPU memory.
Neither monitor requested termination, both observed clean process exit, and
the GPU was released.

Both rollouts passed their dynamic gates and completed the full
`26.028629743 s` execution clock. Baseline position p95/max error was
`0.0805995/0.0814922 m`; candidate p95/max was
`0.0780321/0.0785788 m`. The candidate improved p95 by `0.0025675 m`
or `3.1855%`.

The frozen paired admission rejected the candidate because absolute p95
improvement was below the `0.003 m` minimum by about `0.000433 m`, and
candidate action saturation was `0.00019146` while baseline saturation was
zero. This is an honest paired-admission rejection, not a dynamic failure.

The wrapper created no labels or dataset. Capture, conversion, corpus merge,
BC, PPO, holdout policy execution, checkpoints, and training remain closed.
The next bounded work is CPU-only diagnosis; no runtime retry is authorized.
