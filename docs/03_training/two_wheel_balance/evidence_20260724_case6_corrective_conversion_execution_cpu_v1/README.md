# Case 6 corrective-label CPU conversion

This directory preserves the sole authorized CPU-only conversion of the
admitted case-6 corrective capture at clean synchronized runtime commit
`1283e1b0da405653564382d0aa47d767ce2f925b`.

The one-use mode-`0600` authorization was stored outside the repository and
deleted before the converter started. The converter ran exactly once in the
fresh namespace
`20260724_model_based_corrective_case6_conversion_v1_cpu`; both the converter
and finalizer exited `0`.

The finalizer reopened 7,933 rows with 65 observation features and three
effective post-supervisor residual targets. Effective actions, requested
action audit values, non-history observations, case IDs, source time,
execution time, and elapsed time all match the sealed source capture exactly.
Previous-action channels contain zero on the first row and the prior effective
action thereafter. The retained clipping audit is `[0, 146, 0]`.

The canonical dataset SHA-256 is
`ac138c9790eda983643ae17cc5b3dcf33cfe4634841760aada929df367acb809`.
The final-status SHA-256 is
`5a8662dbb883ae084c8ef8c3a174d6aa8c166b7f27bbae9122109607ba9e2a02`.

This conversion makes case 6 eligible only for a later reviewed case-merge
operation. It does not merge a corpus, authorize BC or PPO, start training,
authorize Isaac/GPU work, or permit another conversion.
