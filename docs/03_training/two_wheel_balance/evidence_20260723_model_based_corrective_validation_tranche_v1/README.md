# Model-based corrective validation tranche v1

This CPU-only selection closes the planning gap for the two validation cases
required by the corrective residual corpus. It selected cases `[8,16]` as the
maximally separated pair among the dynamically qualified validation candidates
`[8,16,22,32]` using the same sealed 12-feature contract as the training
tranche.

Case `78` remains excluded because it is not present in the dynamic-qualified
selection. The selector did not weaken the position gate or substitute the
planner-imitation result for dynamic qualification.

Both selected cases require independent same-seed baseline/candidate canaries
before any label capture may be considered. This artifact does not authorize
GPU execution, label capture, conversion, dataset merge, BC, PPO, or training.

- Selection SHA-256:
  `5576c696e304eb9b9a173970e5fed06e887eccefe2d65a20678415148e22fa0b`
- Selector SHA-256:
  `8f95c022743cd633d2399953060a8836d7e901f26ae7265258ffb8b72e8dd460`
