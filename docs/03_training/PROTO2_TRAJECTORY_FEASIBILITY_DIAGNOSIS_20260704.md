# Proto2 Trajectory Feasibility Diagnosis

This report is deterministic and non-training. It enumerates every randomized reset waypoint in the configured range and replays one static-base episode through the current reachability-map transform.

## Runtime Contract
- `reach_map`: `matlab/reach_map_mobile_mm_arm_only.mat`
- `start_fraction_min`: `0.25`
- `start_fraction_max`: `0.7`
- `reset_anchor_target_blend`: `0.35`
- `control_dt`: `0.05`
- `waypoint_dt`: `0.1`
- `episode_length_s`: `20.0`
- `tolerance`: `0.1`
- reset base xy: `reset_anchor.xy - [0.4415, 0.2405]`
- arm frame: subtract `[0.16, 0, 0.9465]`, then rotate by `-90 deg` around z (`x_arm=y_mount`, `y_arm=-x_mount`).
- RL-safe arm/gimbal envelope used for nearest `qExample` audit: `[-1.0, 0.55, -2.0, -1.0, -0.8, -0.8]` to `[1.0, 1.45, -0.4, 1.0, 0.8, 0.8]`.

## Results
### `crane_down_000.json`
- waypoints/duration: `120` / `12.00s`
- position range: `[1.05, 0.08, 0.86]` to `[1.996926, 0.410283, 1.259396]`
- orientation: `1` unique wxyz quaternion(s), first `[-0.5, 0.5, -0.5, 0.5]`, Euler XYZ deg `[-90.0, 0.0, -90.0]`
- reset starts enumerated: `30..83` (`54` starts)
- unreachable mean/min/max: `67.81%` / `51.00%` / `80.00%`
- workspace distance p95 mean: `0.7700m`; max-distance mean: `0.7723m`
- ideal base motion from reset: max displacement mean/min/max `0.847m` / `0.721m` / `0.957m`; p95 speed mean `0.154m/s`
- nearest qExample envelope violation mean: `100.00%`
- nearest qExample per-joint violation mean: `[0.0, 100.0, 0.0, 35.213, 11.111, 71.032]`
- nearest qExample min/max over windows: `[-0.575161, 2.111926, -1.729132, -2.618109, -1.047969, -2.592411]` to `[0.397079, 3.227312, -0.416226, 2.355071, 1.152538, 2.288496]`
- best reset start: `83` with unreachable `51.00%`, p95 distance `0.6479m`, max distance `0.6499m`
- worst reset start: `40` with unreachable `80.00%`, p95 distance `0.8449m`, max distance `0.8469m`
- worst sample target world/arm: `[1.996926, 0.41016, 1.252763]` / `[0.541844, -1.151266, 0.306263]` at waypoint `114`

### `crane_down_019.json`
- waypoints/duration: `120` / `12.00s`
- position range: `[1.05, -0.332521, 0.86]` to `[1.971608, 0.08, 1.296528]`
- orientation: `1` unique wxyz quaternion(s), first `[-0.5, 0.5, -0.5, 0.5]`, Euler XYZ deg `[-90.0, 0.0, -90.0]`
- reset starts enumerated: `30..83` (`54` starts)
- unreachable mean/min/max: `68.23%` / `51.25%` / `80.50%`
- workspace distance p95 mean: `0.7245m`; max-distance mean: `0.7336m`
- ideal base motion from reset: max displacement mean/min/max `0.862m` / `0.756m` / `0.961m`; p95 speed mean `0.131m/s`
- nearest qExample envelope violation mean: `100.00%`
- nearest qExample per-joint violation mean: `[54.796, 100.0, 0.0, 56.194, 9.657, 69.833]`
- nearest qExample min/max over windows: `[-1.842158, 2.182273, -1.702596, -2.618109, -1.047969, -2.867167]` to `[0.68053, 3.22739, -0.548397, 2.417548, 1.225611, 2.513845]`
- best reset start: `83` with unreachable `51.25%`, p95 distance `0.6220m`, max distance `0.6301m`
- worst reset start: `34` with unreachable `80.50%`, p95 distance `0.8068m`, max distance `0.8219m`
- worst sample target world/arm: `[1.971608, -0.325537, 1.296478]` / `[-0.148552, -1.146218, 0.349978]` at waypoint `119`

## Interpretation

For these two `crane_down` files, the static reset contract is not feasible: even the best randomized reset start leaves about half the 20s episode outside the reach map, and the average start leaves about two thirds unreachable. The required ideal base motion is modest relative to the configured base velocity limit, so base motion should be treated as required but not sufficient.

The nearest reach-map `qExample` values also sit outside the current conservative action envelope, especially `joint5_arm_pitch` and the simulated wrist/gimbal axes. That means a base-only curriculum can improve reachability but still leave the policy unable to command the arm/gimbal posture implied by the reach map.

The trajectory orientation is constant at wxyz `[-0.5, 0.5, -0.5, 0.5]` / Euler XYZ `[-90, 0, -90]`. This script does not prove whether that camera-frame orientation is correct for the current `cam_link`; that remains a separate FK/camera-frame check.

- If unreachable percentages are low while policy EE error is high, the primary blocker is unlikely to be static reachability or chassis `vy`; inspect action-to-FK semantics, camera-frame target orientation, and whether qExample labels sit inside the training envelope.
- If nearest qExample violation is high, imitation labels or reach-map examples are outside the RL-safe envelope even when positions are geometrically reachable.
- If unreachable percentages are high, adjust reset-anchor/base-assist geometry before burning more PPO budget.

