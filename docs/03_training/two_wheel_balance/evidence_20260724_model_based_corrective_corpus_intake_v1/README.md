# Model-based corrective corpus intake v1

This directory preserves the CPU-only intake audit for the real model-based
corrective corpus state at commit
`e6a3688de943864f043691f407de90eb0e51f75d`.

The audit reopens the real case-30 converted dataset, validates the fixed train
and validation tranche selections, validates the passed case-23 v4 capture and
its closed conversion route, and reports:

- converted train cases: `[30]`, or `1/4`;
- converted validation cases: `[]`, or `0/2`;
- pending minimum train cases: `[23, 6, 2]`;
- pending validation cases: `[8, 16]`;
- optional additional train candidate: `[7]`;
- reserved unopened holdouts: `[3, 5, 13, 19, 24]`.

The next bounded action is exactly one separately authorized case-23 v4 CPU
conversion. A future case-23 archive advances this intake only when accompanied
by a passing final conversion status with matching dataset hash and closed
training state.

The report is byte-identical on macOS and `.98` Windows Python. Its SHA-256 is
`3d8f3da9c23ddb9d63a26afb3bec15324d8ce61a3e0b900c7cdf67f91c9e20bf`.

No conversion, merge, capture, runtime, BC, PPO, or training is authorized.
The final corpus manifest is not ready and the goal remains incomplete.
