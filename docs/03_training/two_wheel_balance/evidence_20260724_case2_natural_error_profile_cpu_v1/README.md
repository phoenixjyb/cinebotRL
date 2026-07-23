# Case 2 projection-aware natural-error profile

This CPU-only profile resolves the structural constraint found by the case-2
readiness audit without weakening command limits or injecting a disturbance
into a trajectory that already runs near its dynamic gate.

The corrective envelope retains 25% of the observed raw case-2 residual:
`[0.010247, 0.004541, 0.000614]` for linear velocity, yaw rate, and riser
target. Its slew horizon is 0.4 s. The existing baseline supplies natural
excitation: 42 of 47 trace samples exceed 0.03 m position error, with a
0.153167 m trace maximum. No external wrench profile is created.

The profile requires the deterministic model-based residual safety projection
before actuation. Offline envelope checks show that outward negative linear
corrections require projection on 430 transitions and outward positive yaw
corrections on 103 transitions. Projection remains contractive and keeps every
effective command within `[-0.4, 0.4] m/s`, `[-0.4, 0.4] rad/s`, and
`[0.0, 1.2] m`. Any future capture must store the effective projected
residual, not an unavailable requested command.

The focused profile/projection suite passes 50 tests on macOS and `.98`. The
authoritative `.98` CPU suite passes
`1161 passed, 12 skipped, 2 warnings in 122.95 s` at commit
`2911a0bdb45cf4c83d57a490dfdff9d9a9e90a58`. Both hosts regenerate
byte-identical profile and proposal artifacts.

The profile is CPU-ready, but no runtime route or authorization exists.
Runtime, GPU launch, capture, conversion, merge, BC, PPO, and training remain
disabled.
