# All-79 Dynamic Case-62 Acquisition Recovery (2026-07-15)

## Decision

Case 62 passes after increasing only its generated acquisition-prefix scale from
`1.0x` to `1.25x`. The frozen controller, semantic trajectory, acceptance
thresholds, and actuator limits are unchanged. PPO remains blocked.

## Evidence

The original dynamic run completed without termination and passed pitch, arm,
effort, maximum-position-error, and saturation gates. Tool p95 was `0.1796 m`,
and the maximum occurred at `1.805 s` inside the `2.991 s` generated
acquisition.

| Metric | Acquisition `1.0x` | Acquisition `1.25x` |
| --- | ---: | ---: |
| Acquisition duration | 2.991 s | 3.739 s |
| Full duration | 5.982 s | 6.730 s |
| Tool p95 | 0.1796 m | 0.0996 m |
| Tool maximum | 0.1874 m | 0.1060 m |
| Peak pitch | 9.242 deg | 6.969 deg |
| Peak arm error | 5.764 deg | 4.989 deg |
| Peak arm effort | 25.048 Nm | 23.163 Nm |
| Wheel/arm saturation | 0% / 0% | 0% / 0% |
| Dynamic gate | Fail | Pass |

The promoted v4 payload contains 79 passing kinematic candidates and differs
from v3 only for case 62. Its explicit overrides are
`7:1.25,9:1.5,62:1.25`; its summary SHA-256 is
`8a7d9b83ab82d140048540bee4046293fa4af9bfa6fd52fcab5459f29f50e625`.

The breadth runner preserves accepted cases 1-61, reruns case 62 against v4,
and continues to case 63 only after the repeated case-62 result passes.
