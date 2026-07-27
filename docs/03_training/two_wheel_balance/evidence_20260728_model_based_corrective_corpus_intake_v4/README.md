# Model-based corrective corpus intake v4

This CPU-only audit reopens the sealed case-6, case-7, case-23, and case-30
corrective datasets without merging them. It binds each dataset to its
conversion evidence and requires effective post-supervisor targets,
reconstructed previous effective actions, matching source/runtime identities,
and closed merge/BC/PPO/training state.

Converted train cases are `[6, 7, 23, 30]`, satisfying the four-case train
count. Converted validation cases remain empty; cases `8` and `16` are still
required. Therefore the corpus manifest and all learning stages remain closed.

The next data-producing operation requires separate authorization:

`Authorize exactly one case-8 validation paired canary.`
