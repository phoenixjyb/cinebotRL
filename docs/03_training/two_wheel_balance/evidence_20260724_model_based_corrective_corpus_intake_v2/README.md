# Model-based corrective corpus intake v2

This CPU-only audit reopens the sealed case-23 v4 and case-30 corrective case
datasets without merging them. macOS and `.98` produced byte-identical reports
at implementation commit
`4f370b4b6cc71a338386cc6f760a42b7e32ff085`.

Converted train cases are now `[23, 30]`. The minimum four-case train tranche
still lacks cases `6` and `2`; validation still lacks cases `8` and `16`.
Case `7` remains an additional train candidate. The corpus manifest is not
ready.

The audit binds the case-23 dataset, final status, and path-recovery evidence,
including exactly one converter invocation and no converter retry. Dataset
merge, BC, PPO, Isaac/GPU work, capture, and training remain unauthorized.

The summary SHA-256 is
`dc5ee5f97882c0551521703267d9897ce79731845cbd73e6883bc121183bcb62`.
