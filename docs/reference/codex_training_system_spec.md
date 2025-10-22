# codex Training & System Specification

This document summarises how the control loop, observations, actions, and training stack are configured to work with the current reward design.

## Control timing and simulation

- Physics runs at 200 Hz (`SimulationCfg(dt=0.005)` in `src/rl_platform/tasks/mobile_mm/env.py:117`), and the task decimates by 10 to expose a 20 Hz control step (`decimation = 10` at `src/rl_platform/tasks/mobile_mm/env.py:61`).
- The environment caches `control_dt = physics_dt × decimation = 0.05 s` for reward calculations and rate limiting (`src/rl_platform/tasks/mobile_mm/env.py:329`).
- Trajectories advance on this same 50 ms increment by default (`dt: float = 0.05` in `src/rl_platform/tasks/mobile_mm/trajectories.py:23`), keeping target updates aligned with actuator timing.

## Observation vector

`compose_observation` concatenates features in world frame (`src/rl_platform/tasks/mobile_mm/observations.py:8`). With the default config (`use_lookahead=True`, `lookahead_steps=3`, `include_action_history=True`, `action_history_length=2`) the observation dimension is:

| Component | Dims | Source |
|-----------|------|--------|
| Base pose & velocities | 13 | `src/rl_platform/tasks/mobile_mm/observations.py:56` |
| Arm joint positions & velocities (6 joints) | 12 | `src/rl_platform/tasks/mobile_mm/observations.py:62` |
| End-effector pose & velocities | 13 | `src/rl_platform/tasks/mobile_mm/observations.py:67` |
| EE-to-target error | 7 | `src/rl_platform/tasks/mobile_mm/observations.py:70` |
| Base-to-target planar delta, distance, reach flag | 4 | `src/rl_platform/tasks/mobile_mm/observations.py:74` |
| Lookahead positions (3 steps × 3 axes) | 9 | `src/rl_platform/tasks/mobile_mm/observations.py:85` |
| Two-step action history (2 × 8 actions) | 16 | `src/rl_platform/tasks/mobile_mm/observations.py:92` |

Total: 74 floats per environment, matching the expectation used by the PPO policy definition comment at `scripts/reinforcement_learning/sb3/train.py:826`.

Base linear and angular velocities are normalised by the configured limits before entering the observation (`src/rl_platform/tasks/mobile_mm/env.py:908`), so the policy sees values roughly in [-1, 1] without manual scaling downstream.

## Action interface

- The policy outputs eight continuous actions in [-1, 1] (`num_actions = 8` in `src/rl_platform/tasks/mobile_mm/env.py:70`): the first six map to arm joint position targets, the last two to base forward velocity and yaw rate (`src/rl_platform/tasks/mobile_mm/env.py:718`).
- Arm commands are scaled into soft joint limits with a 5 % safety margin (`src/rl_platform/tasks/mobile_mm/env.py:821`).
- Base commands are converted to physical velocities using `max_linear_velocity` (1.5 m/s) and `max_angular_velocity` (2.0 rad/s) from the task config (`src/rl_platform/tasks/mobile_mm/env.py:761`).
- Command deltas are rate limited so acceleration never exceeds configured bounds (`src/rl_platform/tasks/mobile_mm/env.py:767`). The resulting forward and yaw velocities are integrated over the 0.05 s control step to produce PPR joint targets for `joint_x`, `joint_y`, and `joint_theta` (`src/rl_platform/tasks/mobile_mm/env.py:790`).
- The environment records both commanded and realised velocities for reward terms that check acceleration, jerk, and lateral slip (`src/rl_platform/tasks/mobile_mm/env.py:785`, `src/rl_platform/tasks/mobile_mm/env.py:1059`).

## Reward integration

`compute_combined_reward` is called every control step with the latest observations, previous actions, and historical state (`src/rl_platform/tasks/mobile_mm/env.py:1081`). It uses the weights defined in `RewardWeights` (`src/rl_platform/tasks/mobile_mm/config.py:74`) and evaluates all components summarised in *codex_reward_design.md*. Diagnostics per term are exposed through `extras["reward_components"]` for logging (`src/rl_platform/tasks/mobile_mm/env.py:1123`).

## Trajectory management

- `TrajectoryManager` handles both parametric and recorded trajectories, with per-environment phase state (`src/rl_platform/tasks/mobile_mm/trajectories.py:12`).
- Recorded trajectories advance every two control steps by default because `waypoint_dt` sets 100 ms spacing (`src/rl_platform/tasks/mobile_mm/config.py:143`), and `_recorded_time_accum` accumulates `dt` until it exceeds that threshold (`src/rl_platform/tasks/mobile_mm/trajectories.py:176`).
- Environment resets align the robot base with the first target waypoint to avoid unreachable starts (`src/rl_platform/tasks/mobile_mm/env.py:104`).

## Training stack (Stable Baselines3 PPO)

- The environment created by `train.py` is wrapped to convert Isaac Lab’s dict/tensor observations into numpy arrays suited for SB3 (`scripts/reinforcement_learning/sb3/train.py:503`) and then normalised with `VecNormalize` for both observations and rewards (`scripts/reinforcement_learning/sb3/train.py:739`).
- PPO uses a custom MLP with separate policy and value heads sized `[256, 256, 128]` (`scripts/reinforcement_learning/sb3/train.py:829`) and an initial log standard deviation of −1.0 to moderate exploration (`scripts/reinforcement_learning/sb3/train.py:835`).
- Default hyperparameters target long training runs (up to 100 M steps): 128-step rollouts, batch size 512, learning rate 3e−4, γ = 0.99, λ = 0.95 (`scripts/reinforcement_learning/sb3/train.py:851`).
- Optional schedulers include entropy decay and adaptive KL thresholds to keep policy updates smooth across reward regime shifts (`scripts/reinforcement_learning/sb3/train.py:771`, `scripts/reinforcement_learning/sb3/train.py:784`).
- Checkpoints include the `VecNormalize` state so reward scaling learned during training remains in sync at evaluation time (`scripts/reinforcement_learning/sb3/train.py:761`).

## Alignment with reward design

- The 20 Hz control step and capped acceleration guarantee the reward terms tied to commanded velocities and jerk operate on physically plausible motion, making the penalties meaningful.
- Observation features explicitly expose base-to-target geometry and the “out-of-reach” flag, providing the information needed to exploit the base mobilization reward (see `src/rl_platform/tasks/mobile_mm/observations.py:74`).
- Action penalties operate on the raw [-1, 1] commands, consistent with storing action history prior to scaling (`src/rl_platform/tasks/mobile_mm/env.py:707`), so PPO gradients see the exact signals that will be regularised.
- `VecNormalize` keeps exponentially weighted reward components (especially the 50-point position tracking score) within a manageable numeric range, which stabilises PPO’s value losses relative to the large negative penalties.

This setup allows the control policy to learn when to mobilise the chassis, exploit arm reach, and stay within safety/feasibility constraints defined in the reward scheme.
