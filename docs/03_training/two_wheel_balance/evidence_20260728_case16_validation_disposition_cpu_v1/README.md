# Case-16 Validation Disposition

This CPU-only evidence classifies case 16 after its sealed paired rejection
and selects the next validation candidate without changing any controller,
plant, plan, or gate.

Case 16 is not intrinsically hard in realized dynamics. Its baseline p95
error is `0.08059953 m`, comfortably inside the unchanged `0.15 m` gate.
The candidate improves p95 by `3.1855%`, but the baseline is already strong,
so the absolute improvement misses the frozen `0.003 m` requirement by only
`0.00043253 m`. Candidate saturation represents two estimated samples.

Further profile tuning would reuse held-out validation feedback. Case 16 is
therefore retained as a calibration diagnostic and is not a teacher.

Case 32 is the preferred replacement candidate. Its historical exact-source
playback passed with `0.10241882 m` p95 error, zero action saturation, and a
`9.575589 m` source path. The historical files under `source/` are
hash-pinned selection evidence only; they do not admit a new runtime or
dataset.

The next bounded task is CPU-only case-32 provenance/readiness and profile
preparation. Isaac, capture, conversion, merge, BC, PPO, and training remain
closed.
