# Case 23 corrective capture v4

This directory preserves the sole authorized v4 case-23 corrective-label
capture. The one-use runtime commit was
`31bb9afbf3e9ce6c17e0fc1d2f06b5990e130d1c`. The external mode-`0600`
authorization token was consumed before Isaac started and was absent after the
run. No retry, conversion, corpus merge, BC, PPO, or training run was launched.

The finalizer passed every admission, dynamic, thermal, controller,
perturbation, heartbeat, GPU-release, archive, clock, and identity check. The
archive is admitted only for a separately reviewed CPU conversion step; it is
not a normalized training dataset and is not yet valid for training.

Key evidence:

- 3,273 aligned samples with separate source and execution clocks ending at
  9.929694 s.
- Exactly 20 perturbation-active rows; initialization contributes zero samples.
- Position error p95/max is 0.053413/0.067912 m.
- Attitude error p95/max is 0.148905/0.257883 degrees.
- Peak pitch is 5.642538 degrees and peak riser effort is 22.614313 N.
- Requested and effective normalized residual maxima are
  `[0.216787, 0.084475, 0.284235]`.
- No command or amplitude clipping occurred. Eight riser-label rows were
  slew-limited and are explicitly recorded.
- Capture SHA-256 is
  `f0ea5c59e1f2f0e5f6f91336788d0e0228d079f74a53a4a50d442751b8b23796`.
- Final-status SHA-256 is
  `8f7589cdc31b5b6369fea8fda7fbd8b743b57afa78709cf03b2bd600a25833e3`.

Use `SHA256SUMS` to verify the preserved runtime evidence.

