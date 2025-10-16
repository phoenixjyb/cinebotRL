# Codex Inspection Report

- Date: 2025-10-15
- Commit: 23dd54067650f95b8b5ee1725fc464d6195ad4dc

## Findings

1. **Critical – Trajectory never advances:** The environment only advances the reference path once during reset (`self.trajectory_manager.step()`), so the policy keeps chasing the first waypoint. See `src/rl_platform/tasks/mobile_mm/env.py:584`. In recorded mode the waypoint index is reset to zero (`src/rl_platform/tasks/mobile_mm/trajectories.py:73`) and never incremented when `get_target_pose()` reads from `self.current_waypoint_idx` (`src/rl_platform/tasks/mobile_mm/trajectories.py:224`). Result: end-effector tracking of full trajectories is impossible.
2. **Critical – Base remains passive:** The last two action dimensions are intended for chassis velocity, but the implementation is still a TODO. Actions for `v_x` and `omega_z` are computed and then dropped (`src/rl_platform/tasks/mobile_mm/env.py:359-374`), preventing the robot from repositioning its base to reach wider trajectories.
3. **High – Unscaled joint targets:** PPO outputs in `[-1, 1]` are used directly as absolute joint position targets (`src/rl_platform/tasks/mobile_mm/env.py:371`) without mapping to the physical limits defined in `task_spec.py`. This clips reach to a tiny band, encourages saturation, and ignores the required safety margin near hard stops.
4. **High – Self-collision detection disabled:** Contact forces are hard-coded to zero before reward calculation (`src/rl_platform/tasks/mobile_mm/env.py:469-477`), and the termination branch for self-collision simply `pass`es (`src/rl_platform/tasks/mobile_mm/env.py:533-538`). As a result, neither penalties nor early termination fire when the robot hits itself.
5. **Medium – Jerk/smoothness penalty neutered:** The reward call passes identical tensors for `actions`, `prev_actions`, and `prev_prev_actions` (`src/rl_platform/tasks/mobile_mm/env.py:485-486`), so the jerk term always evaluates to zero and action history signals are lost.
6. **Medium – Potential VecEnv misconfiguration:** The Gym registration captures a single `MobileMMTrackEEEnvCfg()` with `num_envs=1` (`src/task_spec.py:223-226`). Unless Isaac Lab clones the config per call, `gym.make(..., num_envs=k)` may still launch one environment; this should be validated for large Windows runs.
7. **Medium – Parametric timing edge case:** `TrajectoryManager.step()` advances phase by `speed / amplitude` (`src/rl_platform/tasks/mobile_mm/trajectories.py:147-152`); zero or tiny amplitudes (e.g., recorded trajectories with fixed points) stall the phase update. Combined with the missing waypoint increment, lookahead predictions (`src/rl_platform/tasks/mobile_mm/trajectories.py:118-138`) keep repeating the same pose.

