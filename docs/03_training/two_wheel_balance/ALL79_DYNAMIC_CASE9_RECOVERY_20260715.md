# All-79 Dynamic Case-9 Acquisition Recovery (2026-07-15)

## Decision

Case 9 passes after increasing only its generated acquisition-prefix scale from
`1.25x` to `1.5x`. The frozen controller, semantic trajectory, acceptance
thresholds, and actuator limits are unchanged. PPO remains blocked.

## Evidence

The original dynamic run completed without termination and passed pitch, arm,
effort, and saturation gates. Tool p95 was `0.1533 m`, only `0.0033 m` above the
limit, and the maximum occurred at `2.235 s` inside the generated acquisition.

| Metric | Acquisition `1.25x` | Acquisition `1.5x` |
| --- | ---: | ---: |
| Acquisition duration | 6.991 s | 8.389 s |
| Full duration | 14.781 s | 16.179 s |
| Tool p95 | 0.1533 m | 0.0842 m |
| Tool maximum | 0.1625 m | 0.0941 m |
| Peak pitch | 10.026 deg | 9.913 deg |
| Peak arm error | 5.741 deg | 5.197 deg |
| Peak arm effort | 25.055 Nm | 24.725 Nm |
| Wheel/arm saturation | 0% / 0% | 0% / 0% |
| Dynamic gate | Fail | Pass |

The promoted v3 payload contains 79 passing kinematic candidates and differs
from v1 only for cases 7 and 9. Its explicit overrides are `7:1.25,9:1.5`; its
summary SHA-256 is
`e0e02898eafc3808b956f6a1cea9eb785e83cc4c3db28e67a94ce224e275793b`.

The breadth runner preserves accepted cases 1-8, reruns case 9 against v3, and
continues to case 10 only after the repeated case-9 result passes.
