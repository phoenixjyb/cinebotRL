# Case 30 effective-label conversion v1

This directory preserves the sole CPU-only conversion of the admitted v2
case-30 corrective capture. The conversion uses effective post-supervisor
residual actions as candidate labels. Requested pre-supervisor actions remain
audit-only.

The converter rebuilt the previous-action observation fields from the previous
effective label. It changed 841 history rows while preserving all other
observation fields, case IDs, and source/execution/elapsed clocks exactly.

The output contains 11,411 rows with 65 observation features and three bounded
residual actions. Its SHA-256 is
`191a44147bc44038a0645bf48a63609463bf280d97b37ddaf884200bd8b52447`.

This artifact is valid only for a later reviewed case-merge operation. It is
not a merged dataset, is not valid for training, and does not authorize BC,
PPO, holdout evaluation, or another capture.
