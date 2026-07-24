# Case 23 v4 corrective-label CPU conversion

This directory preserves the one authorized CPU-only conversion of the
admitted case-23 v4 corrective capture at synchronized runtime commit
`11fd27698955d277f4b926151bcca0cda2f4b27c`.

The converter ran exactly once and produced 3,273 rows with 65 observation
features and three effective post-supervisor residual targets. All clocks,
case IDs, non-history observations, requested-action audit values, and
effective-action targets reopen exactly. No row is clipped.

The original wrapper escaped `$NAMESPACE` in its Windows output path. The
converter succeeded, but the finalizer could not find the admission file and
the wrapper exited `1`. No retry was performed. The produced dataset was
copied byte-for-byte from the literal-dollar path into the authorized
namespace, then the finalizer alone reopened and sealed it. Both copies have
SHA-256
`ee55db1c02e504e035a47532df8141ab142bb68a2c47b2795b6fd5ad7283ef01`.
The original failure log is retained under `logs/`.

The final status passes every conversion and provenance check and marks the
case eligible only for a later reviewed case-merge operation. It does not
merge a corpus, authorize BC or PPO, start training, or permit another
conversion.
