# Codex Analysis — 2025-10-19

## What We’ve Fixed So Far
- **Reward/unit consistency** – The jerk/acceleration penalties now operate in physical units, and the environment feeds them rate-limited commanded velocities instead of raw PhysX spikes. (: `src/rl_platform/tasks/mobile_mm/env.py`, `rewards.py`)
- **Command rate limiting** – `_pre_physics_step` clamps chassis commands to the configured acceleration limits and stores them for penalties/diagnostics.
- **Trajectory dataset confirmed** – All 1,038 recorded trajectories share the same start pose; the environment aligns the base to that waypoint at reset so episodes begin in a feasible configuration.
- **Debug tooling** – Base/EE tracking logs and reward diagnostics now show sane magnitudes (jerk penalties near zero, positive evaluation reward once base motion is allowed).

## Gaps & Risks Still Open
- **Self-collision detection** – Contact forces remain zero; the current filter also removes base links from the termination check. Arm–base collisions are effectively unmonitored.
- **Initial EE pose** – The first trajectory waypoint is inside the mast, so without base motion the arm collides immediately. No IK/heavy safeguards are in place.
- **Base joint soft limits** – `joint_theta` has a soft limit of 0 rad; any yaw motion trips the warning and may cap useful rotation.
- **Visualization disabled** – `_setup_visualization_markers` still short-circuits, depriving us of path/EE visuals during GUI debugging.
- **Legacy policy** – Existing checkpoints were trained before the fixes and keep the base frozen. They should not be reused.

## Next Steps (Priority Order)
1. **Restore collision feedback**
   - Enable or verify contact sensors on the base/arm links.
   - Adjust `_get_dones()` to include base links once sensor data is valid.
2. **Fix the start configuration**
   - Either solve IK for the first waypoint during reset or offset the chassis so the EE isn’t inside the frame.
3. **Update base joint limits**
   - Widen `joint_theta` soft limits in the URDF (or override them post-load) to allow realistic yaw motion.
4. **Re-enable visualization helpers**
   - Reinstate sphere/line-strip markers for trajectories when running with GUI.
5. **Retrain from scratch**
   - Launch a new PPO run (e.g. 4096 envs, 128 steps) and monitor base actions early on.
   - Use `diagnose_reward_components.py` / `analyze_policy_actions.py` to ensure vx/wz are exploring.

Optional quality-of-life:
- Log base distance travelled per episode to track progress.
- Add a pre-reset safety check that terminates if EE starts within a chassis clearance threshold.

Once these issues are closed and a fresh policy is trained, the chassis should contribute meaningfully to trajectory tracking without the arm slamming into the robot.
