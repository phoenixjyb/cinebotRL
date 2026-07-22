# Model-based corrective pair tranche v1

This CPU-only selection proposes five diverse training cases:
`[30, 23, 6, 2, 7]`. Case 30 is the already converted pilot anchor. Cases 23,
6, 2, and 7 require independent same-seed baseline/candidate paired canaries
before any label capture may be considered.

The selector verified 31 eligible training cases against the sealed plan
portfolio, per-case dynamic evidence, admitted split, plan hashes, and case-30
conversion audit. It excluded validation cases `[8,16,22,32,78]` and holdout
cases `[3,5,13,19,24]`.

This artifact does not authorize reuse of the case-30 profile, creation of a
generic profile, GPU execution, label capture, dataset merge, BC, PPO, or
training. Selection SHA-256 is
`93aa25e99409ad926d4c4cf0b15075e6ea3532dd1c01d5217a45daffc37db7c0`.
