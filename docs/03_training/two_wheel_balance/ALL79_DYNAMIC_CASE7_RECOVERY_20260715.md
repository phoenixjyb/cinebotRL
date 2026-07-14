# All-79 Dynamic Case-7 Acquisition Recovery (2026-07-15)

## Decision

Case 7 passes after retiming only its generated acquisition prefix by `1.25x`.
The frozen whole-body controller, semantic trajectory timing, acceptance limits,
and actuator limits are unchanged. PPO remains blocked.

The promoted all-79 retarget payload is
`evaluation_results/two_wheel_all79_playback/retargeted_all79_v2_case7_scale125`.
It contains 79 NPZs and differs from `retargeted_all79_v1` only for case 7.

## Failure and diagnosis

The original case 7 completed and remained balanced, but tool-position p95 was
`0.1736 m` against the `0.15 m` limit. Its peak error occurred in the first six
seconds while the generated home-to-semantic-start acquisition drove a rapid arm
and COM transition. Peak pitch, arm error, torque, termination, and saturation
all passed. This was an acquisition-feasibility failure, not a semantic-path or
global-controller failure.

## Recovery result

| Metric | Original | Acquisition `1.25x` |
| --- | ---: | ---: |
| Acquisition duration | 7.178 s | 8.973 s |
| Full duration | 17.247 s | 19.042 s |
| Tool p95 | 0.1736 m | 0.0704 m |
| Tool maximum | 0.1818 m | 0.0826 m |
| Peak pitch | 10.089 deg | 9.959 deg |
| Peak arm error | 5.691 deg | 4.715 deg |
| Peak arm effort | 24.331 Nm | 24.337 Nm |
| Wheel/arm saturation | 0% / 0% | 0% / 0% |
| Dynamic gate | Fail | Pass |

The new kinematic candidate also passes all limits. Its acquisition maximum
kinematic error is `0.00226 m`; semantic p95/max errors remain below
`1.5e-7 m`, confirming that semantic samples were not modified.

## Contract change

`retarget_all79_nonholonomic.py` now accepts explicit overrides such as:

```text
--acquisition-time-scale-overrides 7:1.25
```

Overrides are validated, apply only to selected cases, and replace the automatic
scale search for those cases. Duplicate cases, scales below one, non-finite
values, and overrides for unselected cases are rejected.

The promoted 79-case summary SHA-256 is
`b5d5506d35f477d3441f443e7fb9c8c82d56074c25040138b13e2f5df8b465c2`.
The resumable dynamic gate preserves accepted cases 1-6, reruns case 7 against
the promoted payload, and proceeds from case 8 only if case 7 passes again.
