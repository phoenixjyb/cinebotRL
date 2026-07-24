# Generic corrective-capture finalizer

This CPU-only evidence validates the generic corrective-capture finalizer
introduced at commit `04a9ccc0a4f1217f225ed61cf02a9d900310cfaf`.

The finalizer derives case, split, namespace, capture filename, and the unique
plan identity from the admitted capture evidence. It then delegates to the
existing archive-reopening finalizer with those values. It supports both
`train` and `validation` splits and fails closed on ambiguous plan identities,
invalid cases/splits, open dataset or learning permissions, or a noncanonical
runtime commit.

Read-only re-finalization passed against the original `.98` artifacts:

- Case 6: 7,933 samples.
- Case 23: 3,273 samples.
- Case 30: 11,411 samples.

The generic code did not replay Isaac, capture labels, rewrite any source
archive, convert a dataset, merge a corpus, or start BC/PPO/training. The
case-specific launch wrappers remain authoritative until their controller
arguments are extracted into and validated against a proposal contract.

Focused coverage passes `31 passed, 2 warnings in 0.22 s` locally and
`31 passed, 2 warnings in 0.84 s` on `.98`. The authoritative `.98`
Windows-Python suite passes
`1427 passed, 12 skipped, 2 warnings in 246.17 s`.
