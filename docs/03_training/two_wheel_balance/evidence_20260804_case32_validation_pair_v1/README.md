# Case 32 Validation Natural-Error Pair Evidence

This directory preserves the single explicitly authorized case-32 validation
pair executed on `.98` on 2026-08-04.

- Runtime commit: `ef21116db96259318ab838064d97ece626aaab95`
- Plan SHA-256: `71b1986633613fdb13585ac4c12870addc553ad12e895b05cc424a83cf4e037f`
- Baseline p95/max: `0.1024188226 / 0.1339963098 m`
- Candidate p95/max: `0.0991989044 / 0.1308894101 m`
- Absolute/relative p95 improvement: `0.0032199182 m / 3.1438735%`
- Baseline/candidate saturation ratio: `0 / 0`
- Final status SHA-256: `ca725adcf38550b3c581380d566612fc46c61608cc4075b432142b76a2d825e2`

Both dynamic rollouts passed under the same source plan, clocks, seed, physics,
and unchanged gates. The corrective projection was measured for `12,991`
samples and affected `374` samples without the observer modifying commands.

This pair is validation evidence only. It created no label capture, dataset,
BC checkpoint, PPO run, or training admission. Any corrective-label capture
requires a new explicit authorization.
