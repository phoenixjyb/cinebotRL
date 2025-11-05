# Mobile MM Trajectory Tracking — Debug & Training Playbook

> **Purpose:** A drop‑in checklist of high‑leverage fixes to get the chassis + arm policy to actually move the base, then converge to smooth, accurate end‑effector tracking. All changes below reference the current codebase and keep frame conventions intact.

---

## TL;DR — apply these first

1) **Write the base state once per step (pose+vel together).** Replace the current pattern that writes root velocity and then root pose in the same step. This removes a control conflict that can make the base look “numb.” fileciteturn0file0  
2) **Let the base move when it needs to.** Gate stability/limit penalties when the target is far; keep the strong mobilization/alignment signals. fileciteturn0file4  
3) **Add a heading cue to observations.** Append _sin/cos of base→target yaw error_ so the policy immediately knows which way to turn. fileciteturn0file2

Once those are in, enable the KL/entropy schedules in PPO and lengthen rollouts. fileciteturn0file8

---

## 1) Control-path fix (must‑do)

**File:** `env.py` → `_pre_physics_step`

**Issue:** You call `write_root_link_velocity_to_sim()` and then **immediately** call `write_root_pose_to_sim()` to clamp Z, which likely wipes or fights the just‑set velocity. Replace both with a single `write_root_state_to_sim()` per step. fileciteturn0file0

**Patch (diff-style):**
```diff
@@ def _pre_physics_step(self, actions: torch.Tensor):
- # Apply velocity command to root
- self.robot.write_root_link_velocity_to_sim(root_vel_w)
-
- # Clamp Z and re-write pose (overwrites velocity)
- current_root_pos = self.robot.data.root_pos_w.clone()
- current_root_pos[:, 2] = 0.0
- root_pose = torch.cat([current_root_pos, self.robot.data.root_quat_w], dim=-1)
- self.robot.write_root_pose_to_sim(root_pose)
+ # Single write: pose (with Z clamp) + velocities in one atomic state
+ root_state = torch.zeros(self.num_envs, 13, device=self.device)
+ # position
+ root_state[:, 0:3]  = self.robot.data.root_pos_w
+ root_state[:, 2]    = 0.0                     # keep on ground
+ # orientation
+ root_state[:, 3:7]  = self.robot.data.root_quat_w
+ # linear + angular velocities
+ root_state[:, 7:10] = root_vel_w[:, 0:3]
+ root_state[:, 10:13]= root_vel_w[:, 3:6]
+ self.robot.write_root_state_to_sim(root_state)
```

> Keep PPR base joints zeroed at reset (you already do) and never mix them as world pose. The world pose for rewards/obs remains `root_*_w`. fileciteturn0file0

---

## 2) Reward shaping that doesn’t fight the chassis

**File:** `rewards.py` → `compute_combined_reward(...)`

You already have the right mobilization/alignment pieces; the problem is that constant stability/limit/jerk penalties dominate **exactly when the target is far**. Gate those by distance to switch from “go there fast” to “be precise” as the base enters arm reach. fileciteturn0file4

**Insert near the top (after you have `base_target_distance`):**
```python
# Smooth gate: far targets → ~0.0 (penalties off), near targets → ~1.0 (penalties on)
stability_gate = torch.sigmoid((0.55 - base_target_distance) * 10.0)
```

**Apply the gate to the motion-suppressing terms:**
```diff
- stab_penalty = stability_penalty(base_lin_vel, base_ang_vel, scale=weights["stability_penalty"])
+ stab_penalty = stability_gate * stability_penalty(base_lin_vel, base_ang_vel, scale=weights["stability_penalty"])

- vel_limit_penalty = velocity_limit_penalty(...)
+ vel_limit_penalty = stability_gate * velocity_limit_penalty(...)

- accel_limit_penalty = acceleration_limit_penalty(...)
+ accel_limit_penalty = stability_gate * acceleration_limit_penalty(...)

- jerk_penalty_val = jerk_penalty(...)
+ jerk_penalty_val = stability_gate * jerk_penalty(...)
```

This lets the mobilization and alignment terms you already implemented actually win when they should, and the polish terms dominate near the goal. fileciteturn0file4

> Keep your **smart distance penalty** (90% discount while moving) and the **inner margin** at small weight early; you can raise them later (see §5 schedules). fileciteturn0file4

---

## 3) Observations: add base→target yaw error (sin/cos)

**File:** `observations.py` → `compose_observation(...)`

Add a 2‑D feature to tell the policy how misaligned the chassis is relative to the target bearing. This removes the “should I turn left or right?” ambiguity. fileciteturn0file2

**Patch:**
```diff
@@ def compose_observation(...):
     base_to_target_xy = target_pos[:, :2] - base_pos[:, :2]
     base_to_target_dist = torch.norm(base_to_target_xy, dim=-1, keepdim=True)
     arm_reach = 0.6
     out_of_reach = (base_to_target_dist > arm_reach).float()
-    components.extend([base_to_target_xy, base_to_target_dist, out_of_reach])
+    components.extend([base_to_target_xy, base_to_target_dist, out_of_reach])
+
+    # NEW: heading error (sin/cos of yaw error)
+    w, x, y, z = base_quat[:, 0], base_quat[:, 1], base_quat[:, 2], base_quat[:, 3]
+    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
+    bearing = torch.atan2(base_to_target_xy[:, 1], base_to_target_xy[:, 0])
+    yaw_err = torch.remainder(bearing - yaw + torch.pi, 2 * torch.pi) - torch.pi
+    components.append(torch.stack([torch.sin(yaw_err), torch.cos(yaw_err)], dim=-1))
```

**And bump the obs dim by `+2`:**
```diff
@@ def get_observation_dimensions(...):
-    # Base-to-target info: dx(1) + dy(1) + distance(1) + out_of_reach_flag(1) = 4
-    dim += 4
+    # Base-to-target info: dx, dy, distance, out_of_reach, sin(yaw_err), cos(yaw_err)
+    dim += 6
```

> Your README still says **46 dims**; with current components it is already higher, and adding heading makes it +2. Update the doc to prevent confusion. fileciteturn0file3

---

## 4) Reset: face the first segment of the track

**File:** `env.py` → `_reset_idx`

Quickly rotate the base toward the first trajectory segment so early steps aren’t wasted spinning-in-place. Only apply if you have ≥2 waypoints. fileciteturn0file0

```python
# After computing first_target_pos and before write_root_state_to_sim(...)
if self.trajectory_manager.recorded_positions is not None:
    env_ids_cpu = env_ids.cpu().tolist()
    for k, e in enumerate(env_ids_cpu):
        wps = self.trajectory_manager.recorded_positions[e]
        if wps.shape[0] >= 2:
            delta = wps[1, :2] - wps[0, :2]
            heading = torch.atan2(delta[1], delta[0])
            half = 0.5 * heading
            # quaternion (w, x, y, z) for pure yaw
            new_root_state[k, 3] = torch.cos(half)
            new_root_state[k, 4] = 0.0
            new_root_state[k, 5] = 0.0
            new_root_state[k, 6] = torch.sin(half)
```

---

## 5) Training script: un‑choke PPO early, then tighten

**File:** `scripts/reinforcement_learning/sb3/train.py`

- **Turn on** the built‑in schedules and lengthen rollouts:  
  `--enable_kl_schedule --enable_entropy_decay --n_steps 1024 --ent_coef 0.001` fileciteturn0file8  
- Keep `VecNormalize` (already present) and the larger policy net. fileciteturn0file8

**Example launch (Windows Isaac Lab launcher):**
```powershell
I:\isaaclab\isaaclab-3090.bat -p scripts/reinforcement_learning/sb3/train.py `
  --task MobileMMTrackEE-v0 `
  --num_envs 2048 `
  --headless `
  --trajectory_type multi_recorded `
  --use_all_trajectories `
  --n_steps 1024 `
  --ent_coef 0.001 `
  --enable_kl_schedule `
  --enable_entropy_decay
```

**Optional:** tiny reward schedule callback (early→mid→late). Add this class and append to `callbacks`. It mutates `env.unwrapped.reward_weights` safely through SB3’s VecEnv wrapper. fileciteturn0file8

```python
from stable_baselines3.common.callbacks import BaseCallback
class RewardScheduleCallback(BaseCallback):
    def __init__(self, t1=2_000_000, t2=6_000_000, verbose=0):
        super().__init__(verbose); self.t1, self.t2 = t1, t2
    def _on_rollout_end(self) -> bool:
        # unwrap vec env chain
        env = self.training_env
        while hasattr(env, "venv"):
            env = env.venv
        isaac_env = getattr(env, "unwrapped", None)
        if isaac_env and hasattr(isaac_env, "reward_weights"):
            w = isaac_env.reward_weights
            t = self.num_timesteps
            # Early: encourage mobilization
            if t < self.t1:
                w["base_progress_reward"] = 12.0
                w["base_target_alignment"] = 8.0
                w["stability_penalty"] = 0.05
                w["acceleration_limit_penalty"] = 0.0
                w["jerk_limit_penalty"] = 0.0
            # Mid: shape accuracy
            elif t < self.t2:
                w["position_tracking"] = 12.0
                w["stability_penalty"] = 0.2
                w["acceleration_limit_penalty"] = 0.1
                w["jerk_limit_penalty"] = 0.05
            # Late: polish
            else:
                w["position_tracking"] = 16.0
                w["stability_penalty"] = 0.3
                w["velocity_limit_penalty"] = 0.05
                w["acceleration_limit_penalty"] = 0.2
                w["jerk_limit_penalty"] = 0.1
        return True
# ... then in setup: callbacks.append(RewardScheduleCallback())
```

---

## 6) Trajectory cadence & dataset hygiene

- **Recorded cadence:** Set `trajectory_dt` to match the JSON sampling cadence so waypoints don’t “jump/creep”. It’s passed into `TrajectoryManager(..., waypoint_dt=...)`. fileciteturn0file0 fileciteturn0file5  
- **Multi‑trajectory loader:** Already pads by repeating the final waypoint and samples one per env; keep using `--use_all_trajectories` for diversity. fileciteturn0file1  
- **Reachability map:** If available, keep it enabled; the `world_to_arm_frame(...)` is consistent with your base pose definition and arm mount offset. fileciteturn0file7

---

## 7) Safety & sensors

You’ve configured two filtered contact sensors (arm↔base, arm↔ground) and exclude base↔ground support loads from the penalty/termination path. Keep the filter prim paths consistent with the ground prim (`/World/Ground`). fileciteturn0file0

---

## 8) Small but valuable robustness tweaks

- **Indices → names for limits:** Don’t assume arm joints are `[3:9]` forever; derive indices from names (you already do this elsewhere). fileciteturn0file0 fileciteturn0file6  
- **Docs:** Update the README’s observation dimension and note that base control is integrated. fileciteturn0file3

---

## 9) Quick checkpoint (copy/paste)

1. Apply §1 diff in `env.py`. fileciteturn0file0  
2. Add §2 gating in `rewards.py`. fileciteturn0file4  
3. Add §3 features in `observations.py` (+2 dims). fileciteturn0file2  
4. Add §4 reset‑heading (optional but helpful). fileciteturn0file0  
5. Launch with §5 flags; optionally add the `RewardScheduleCallback`. fileciteturn0file8  
6. Verify `trajectory_dt` for your dataset; prefer `--trajectory_type multi_recorded --use_all_trajectories`. fileciteturn0file5 fileciteturn0file1

---

### Appendix — references to the current codebase
- Environment, base control, resets, sensors: `env.py`. fileciteturn0file0  
- Multi‑trajectory dataset loader: `multi_trajectory.py`. fileciteturn0file1  
- Observation builder & dims: `observations.py`. fileciteturn0file2  
- Task README (update dims & base control status): `README.md`. fileciteturn0file3  
- Rewards & penalties: `rewards.py`. fileciteturn0file4  
- Trajectory manager (recorded lookahead/cadence): `trajectories.py`. fileciteturn0file5  
- Mobile MM asset helpers & joint mapping robustness: `mobile_mm.py`. fileciteturn0file6  
- Reachability map & transforms: `reachability_map.py`. fileciteturn0file7  
- SB3 training script & wrappers/schedules: `train.py`. fileciteturn0file8

---

**End of playbook.**