# Case 6 corrective capture v1

This directory preserves the sole authorized case-6 corrective-label capture.
The one-use runtime commit was
`c528cd2a3ccfabcb38fffd99a5c1613d306161e9`. The external mode-`0600`
authorization token was consumed before Isaac started and was absent after the
run. No retry, conversion, corpus merge, BC, PPO, checkpoint, or training run
was launched.

The finalizer passed every admission, dynamic, thermal, controller,
perturbation, heartbeat, GPU-release, archive, clock, and identity check. The
archive is admitted only for a separately authorized CPU conversion step; it
is not a normalized training dataset and is not valid for training.

Key evidence:

- 7,933 aligned samples with source and execution clocks ending at
  `15.942736 s` and `17.737274606 s`.
- Exactly 20 perturbation-active rows; initialization contributes zero samples.
- Position error p95/max is `0.110560/0.126411 m`.
- Attitude error p95/max is `0.158318/0.453032 degrees`.
- Peak pitch is `6.383017 degrees`, peak riser effort is `21.256136 N`, and
  peak riser servo error is `0.010080 m`.
- Requested and effective normalized residual maxima are
  `[0.462029, 0.159056, 0.089326]`.
- Supervisor clipping rows are `[0, 146, 0]`; requested and effective values
  are both preserved, and the effective post-supervisor residual remains the
  only future training target.
- Capture SHA-256 is
  `c51411a9686909c47af7eeabf46a61672d8b09432cfb35daabc46af5a5913f85`.
- Final-status SHA-256 is
  `843981c82609d8d07cf1b532ce5978e279872649f4f7cd499092a0c7261376f9`.

Use `SHA256SUMS` to verify the preserved runtime evidence.
