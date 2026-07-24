# Case-2 natural-error paired canary execution

This package preserves the only execution authorized by the user statement
`Authorize exactly one case-2 paired canary`.

## Execution

- Runtime commit: `9363f2818688653c2c6db60699caba496a0c8d3a`
- Namespace:
  `20260724_model_based_corrective_teacher_case2_natural_error_pair_v1_exclusive`
- Baseline and candidate both exited `0` and independently passed the unchanged
  dynamic safety/quality gates.
- Baseline position p95/max: `0.1398296339 / 0.1535142853 m`.
- Candidate position p95/max: `0.1395659017 / 0.1530445505 m`.
- Observed p95 improvement: `0.0002637322 m` (`0.1886096%`), below both
  required gates of `0.003 m` and `2%`.

## Fail-closed result

The finalizer exited `6`. The case-specific projection observer did not inject
its expected `corrective_teacher_projection_telemetry` block into either
runtime JSON, so the official paired evidence contract rejected the run.
The candidate's ordinary corrective telemetry is present, but it is not a
substitute for the pinned observer contract.

Even if that evidence-contract defect were repaired, the measured tracking
improvement is below the immutable paired-admission thresholds. Case 2 is
therefore rejected for corrective-label capture.

The one-use mode-`0600` token was deleted before Isaac started. GPU release
passed. No label archive, dataset, BC, PPO, checkpoint, or training run was
created.
