# Case-8 validation corrective CPU conversion

This package preserves the one authorized CPU-only conversion of the admitted
case-8 validation corrective capture. The conversion ran at clean synchronized
commit `97076069300512380e99e1d2ab55a95d0b4ebf5e`.

The external mode-`0600` authorization token was consumed before conversion.
The wrapper, converter, and finalizer each ran once and exited `0`; no token
remains and no retry was performed.

The finalizer reopened `6,607` validation rows with 65 observation features and
three effective post-supervisor residual targets. Effective and requested
actions, non-history observations, previous-action recurrence, case IDs,
clipping, and source/execution/elapsed clocks all match the sealed capture. An
independent local reopen under the original runtime namespace passed every
check and reproduced the dataset SHA-256.

The converted case is valid only for a later reviewed case-disjoint merge
operation. This package does not merge a corpus, authorize BC or PPO, execute a
holdout policy, start training, or mark the dataset valid for training.
