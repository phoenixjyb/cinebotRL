# Model-based corrective corpus intake v3

This CPU-only audit reopens the sealed case-6, case-23, and case-30 corrective
case datasets without merging them. macOS and `.98` Windows/Isaac Python
produced byte-identical reports at implementation commit
`c1e45027ec14f7aeaf4c39b066cd99418d6f116b`.

Converted train cases are `[6, 23, 30]`. The minimum four-case train tranche
still lacks case `2`; validation still lacks cases `8` and `16`. Case `7`
remains an additional train candidate. The corpus manifest is not ready.

The audit preserves the special case-23 path-recovery proof and independently
binds the case-6 dataset, final status, consumed authorization admission,
closed execution contract, and conversion result. It requires effective
post-supervisor labels, reconstructed previous effective actions, matching
source hashes/runtime identity, and closed merge/BC/PPO/training state.

The next data-producing operation is a separately authorized case-2 paired
canary. This audit does not authorize that canary, dataset merge, BC, PPO,
Isaac/GPU work, capture, or training.

The summary SHA-256 is
`d151a8c144005737eb2a86d4038fda19b9ed6da2286cc9e1f3132024bb8fdea8`.
