# codex Reward Design Reference

This note captures the current reward shaping for the mobile manipulator tracking task. It is based on the latest implementation in `src/rl_platform/tasks/mobile_mm/rewards.py` and default weights from `src/rl_platform/tasks/mobile_mm/config.py`.

## Primary reward terms

- **End-effector position tracking** – exponential score on Cartesian error, scaled by `position_tracking` (default 50.0). See `position_tracking_reward` at `src/rl_platform/tasks/mobile_mm/rewards.py:8` and weight in `src/rl_platform/tasks/mobile_mm/config.py:79`.
- **End-effector orientation tracking** – exponential score on quaternion distance, scaled by `orientation_tracking` (2.0). See `src/rl_platform/tasks/mobile_mm/rewards.py:29` and weight at `src/rl_platform/tasks/mobile_mm/config.py:80`.
- **Progress bonus** – positive clamp of the reduction in position error, encouraging step-to-step improvement. Defined at `src/rl_platform/tasks/mobile_mm/rewards.py:58` and weighted by `progress_bonus` (1.0) in `src/rl_platform/tasks/mobile_mm/config.py:81`.
- **Base mobilization reward** – rewards chassis motion that actually shortens base-to-target distance, only when the goal is outside arm reach. Includes a 0.2 m/step cap to prevent reward spikes. Implementation at `src/rl_platform/tasks/mobile_mm/rewards.py:75` with weight `base_progress_reward` (150.0) in `src/rl_platform/tasks/mobile_mm/config.py:82`.
- **Obstacle clearance reward (optional)** – only active if obstacle distances are supplied; contributes via `obstacle_distance_reward` (called inside `compute_combined_reward`) and weighted by `min_obstacle_distance_weight` (1.0) in `src/rl_platform/tasks/mobile_mm/config.py:107`.

## Distance management around arm reach

- **Target distance penalty** – linear penalty on planar base-to-target separation beyond 0.6 m, with a 90 % discount when the base is already moving. Defined at `src/rl_platform/tasks/mobile_mm/rewards.py:131` and weighted by `target_distance_penalty` (5.0) in `src/rl_platform/tasks/mobile_mm/config.py:83`.
- **Excessive base movement penalty** – clamps per-step chassis motion to 0.1 m and penalizes the excess to prevent reward farming. See `src/rl_platform/tasks/mobile_mm/rewards.py:176`; weight `excessive_base_movement_penalty` (10.0) is in `src/rl_platform/tasks/mobile_mm/config.py:84`.

These two terms form the “outer workspace” logic: the distance penalty supplies a strong negative signal when the target is out of reach, while the mobilization reward plus excessive-motion clamp ensure progress is rewarded but large jumps stay bounded.

## Action quality penalties

All action penalties operate on the policy outputs before scaling by joint limits or velocity caps:

- Magnitude (`action_magnitude_penalty` at `src/rl_platform/tasks/mobile_mm/rewards.py:203`) weighted by 0.005 (`src/rl_platform/tasks/mobile_mm/config.py:86`).
- Rate of change (`action_rate_penalty` at `src/rl_platform/tasks/mobile_mm/rewards.py:223`) weighted by 0.01 (`src/rl_platform/tasks/mobile_mm/config.py:87`).
- Jerk / smoothness (`action_smoothness_penalty` at `src/rl_platform/tasks/mobile_mm/rewards.py:244`) weighted by 0.05 (`src/rl_platform/tasks/mobile_mm/config.py:88`).

Together these regularise arm and base commands so the policy does not exploit abrupt control.

## Physical constraint penalties

- **Velocity limits** – penalises linear base speed and arm joint velocities that exceed configured limits. See `velocity_limit_penalty` at `src/rl_platform/tasks/mobile_mm/rewards.py:267`; weight 5.0 at `src/rl_platform/tasks/mobile_mm/config.py:92`.
- **Acceleration limit** – clamps commanded forward acceleration vs. `max_linear_acceleration`. Implemented via `acceleration_limit_penalty` (`src/rl_platform/tasks/mobile_mm/rewards.py:296`) with weight 5.0 (`src/rl_platform/tasks/mobile_mm/config.py:93`).
- **Jerk limit** – penalises acceleration changes against `max_linear_jerk`. See `jerk_penalty` at `src/rl_platform/tasks/mobile_mm/rewards.py:320`; weight 0.05 (`src/rl_platform/tasks/mobile_mm/config.py:94`).
- **Joint limit** – encourages staying within soft joint bounds using the six arm joints. Implemented at `src/rl_platform/tasks/mobile_mm/rewards.py:352` and weighted by 10.0 (`src/rl_platform/tasks/mobile_mm/config.py:95`).
- **Lateral motion** – penalises sideways base velocity inconsistent with differential drive kinematics, defined at `src/rl_platform/tasks/mobile_mm/rewards.py:384` with weight 2.0 (`src/rl_platform/tasks/mobile_mm/config.py:96`).
- **Stability** – dampens oscillatory body motion via `stability_penalty` (`src/rl_platform/tasks/mobile_mm/rewards.py:461`) weighted 0.1 (`src/rl_platform/tasks/mobile_mm/config.py:101`).

## Safety penalties

- **Self-collision** – uses contact forces to apply a large penalty (50.0 weight) whenever link contacts exceed the `self_collision_threshold`. See `src/rl_platform/tasks/mobile_mm/rewards.py:407` and weight at `src/rl_platform/tasks/mobile_mm/config.py:99`.
- **External collision (placeholder)** – `collision_penalty` exists but is currently inactive in `compute_combined_reward`. The default weight remains 10.0 (`src/rl_platform/tasks/mobile_mm/config.py:100`) for future use.

Episodes also terminate on self-collision or excessive tracking error (`src/rl_platform/tasks/mobile_mm/env.py:1154` and `src/rl_platform/tasks/mobile_mm/config.py:157`), amplifying the safety incentives.

## Aggregation

`compute_combined_reward` (`src/rl_platform/tasks/mobile_mm/rewards.py:590`) gathers all components each step, applies the configured weights, stores diagnostic breakdowns, and returns the final scalar reward. The environment passes the current and historical state needed for these terms at `src/rl_platform/tasks/mobile_mm/env.py:1081`.

## Default scaling summary

| Term | Default scale | Notes |
|------|---------------|-------|
| Position tracking | 50.0 | dominates near-goal behaviour |
| Orientation tracking | 2.0 | secondary to position |
| Base mobilization | 150.0 | capped at 30 reward per step (0.2 m × 150) |
| Target distance penalty | 5.0 | linear metres beyond 0.6 m, 90 % discount while moving |
| Excessive movement penalty | 10.0 | per metre beyond 0.1 m per step |
| Action magnitude / rate / smoothness | 0.005 / 0.01 / 0.05 | applied to raw [-1,1] actions |
| Velocity / acceleration / jerk limits | 5.0 / 5.0 / 0.05 | ensure feasibility of commanded base motion |
| Joint limit penalty | 10.0 | arm joints only |
| Lateral motion penalty | 2.0 | keeps chassis aligned with diff-drive model |
| Self-collision penalty | 50.0 | large enough to dominate reward |
| Stability penalty | 0.1 | mild damping on oscillations |

These numbers were last tuned during “Session 5b” adjustments, as noted in the inline comments of `src/rl_platform/tasks/mobile_mm/config.py`.
