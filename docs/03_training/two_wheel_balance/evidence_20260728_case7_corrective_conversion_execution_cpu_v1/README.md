# Case-7 corrective CPU conversion

This package preserves the one authorized CPU-only conversion of the admitted
case-7 v2 corrective capture. The conversion ran at clean synchronized commit
`847597c2e1e9dac199357faf62576ecb634159f7`.

The external mode-`0600` authorization token was consumed before conversion.
The wrapper, converter, and finalizer each ran once and exited `0`; no token
remains. A preliminary shell guard had self-matched before token creation, and
an explicit state audit proved that it created no token, namespace, log, or
converter process before the single wrapper invocation.

The finalizer reopened `6,597` rows with 65 observation features and three
effective post-supervisor residual targets. Effective and requested actions,
non-history observations, previous-action recurrence, case IDs, clipping, and
source/execution/elapsed clocks all match the sealed capture. An independent
local reopen passed the same checks and reproduced the dataset SHA-256.

The converted case is valid only for a later reviewed case-merge operation.
This package does not merge a corpus, authorize BC or PPO, start training, or
mark the dataset valid for training.
