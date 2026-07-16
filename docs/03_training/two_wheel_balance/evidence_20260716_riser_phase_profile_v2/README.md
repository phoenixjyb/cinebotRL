# Riser phase-consistent profile v2

Accepted deterministic evidence for the arm-free two-wheel riser robot.

- Playback contract: `derivatives_scaled_by_progress_v1`
- Tracking profile: `riser_phase_consistent_v2`
- Tracking gains: along `1.6`, cross `1.5`, yaw `1.2`
- Case-15 planning yaw cap: `0.20 rad/s`
- Public yaw-rate limit: unchanged at `0.40 rad/s`
- Public camera-position gates: unchanged at p95 `0.15 m`, max `0.25 m`
- Representative gate: cases 1, 15, 31, 73 pass `4/4`
- Repaired-family gate: cases 18, 23, 24, 41, 50, 72, 79 pass `7/7`
- Repaired-family steps: `34002`
- Repaired-family worst position p95/max: `0.122759 / 0.127505 m`
- Repaired-family worst attitude p95: `0.199269 deg`
- Repaired-family worst pitch: `4.876883 deg`
- Terminations: `0`
- Action saturation: `0`

Runtime artifacts on `.98`:

- `artifacts/two_wheel_riser/20260716_all79_playback_inputs_v2`
- `artifacts/two_wheel_riser/20260716_phase_profile_representative4_v1/gate.json`
- `artifacts/two_wheel_riser/20260716_phase_profile_repaired7_v1/gate.json`

Training remains unauthorized until the fresh all-79 v2 dynamic dataset gate
and case-disjoint merge pass.
