# Generic Corrective Conversion Execution Route

This package proves one reusable, fail-closed CPU conversion route for any
tracked and committed corrective-conversion proposal.

## Contract

- The proposal supplies the case, split, source capture, final status, sample
  metrics, clocks, and expected output name.
- The validator reopens the admitted capture and recomputes those values.
- The namespace and output path are derived, not supplied by the operator.
- A committed canonical contract pins the builder, preparer, validator,
  converter, dataset module, capture module, wrapper, and finalizer.
- Execution requires a separate external mode-`0600` token. The wrapper deletes
  that token before invoking the converter.
- The finalizer reopens the produced dataset and compares actions,
  observations, case IDs, clocks, clipping, and source hashes to the capture.

## Verification

- Cases 6, 23, and 30 pass the same route over 22,617 total samples.
- macOS and `.98` native Windows Python produced byte-identical preflights.
- The actual `.98` wrapper reproduced the case-6 preflight hash.
- Tokenless `--execute` returned code 4 and created no namespace.
- The compatibility suite passed 51 tests on both hosts.
- The authoritative `.98` suite passed 1,410 tests with 12 skips.

## Boundary

No execution token was issued. No conversion output, corpus merge, BC, PPO, or
training artifact was created. The route is ready for a future separately
authorized conversion only after a capture has passed and its proposal has
been committed.

The next data-producing operation remains:

`Authorize exactly one case-7 corrective-label capture.`
