# Generic Corrective Conversion Proposals

This package proves that the three already admitted corrective captures use one
case-parameterized conversion path. It does not execute another conversion and
does not create or modify a case dataset.

## Result

- Cases 6, 23, and 30 all pass the same proposal schema.
- The proposals cover 22,617 samples with 65 observations and three bounded
  residual actions.
- Training targets are the effective post-supervisor residuals.
- Previous-action observations are rebuilt from the previous effective action.
- Requested-versus-effective clipping and both source and execution clocks are
  audited.
- macOS and `.98` native Windows Python produced byte-identical proposal files.
- The `.98` full CPU suite passed 1,396 tests with 12 skips.

## Boundary

These reports are output-free conversion proposals. They contain no runtime
token and grant no conversion, corpus-merge, BC, PPO, or training authorization.
The existing converted datasets for cases 6, 23, and 30 are unchanged.

The next data-producing operation remains separately authorized:

`Authorize exactly one case-7 corrective-label capture.`
