# Critical Bugs Fixed - Chassis Movement Enabled

**Date**: October 18, 2025  
**Commit**: ea0d528  
**Impact**: 100M training run was wasted - these bugs prevented chassis from ever moving

## Summary

Four critical bugs were discovered and fixed that completely prevented the mobile base from moving during training. The 100M timestep training run (--trajectory_type multi_recorded) actually trained on **static targets** instead of the 1,038 diverse recorded trajectories.

## The 4 Critical Bugs

### 1. Trajectory Waypoints Never Advanced ⚠️ CRITICAL

**File**: `src/rl_platform/tasks/mobile_mm/trajectories.py:156`

**Problem**: 
- `TrajectoryManager.step()` only updated sinusoid phase for circle/line trajectories
- Never incremented `current_waypoint_idx` for recorded/multi_recorded types
- Result: Robot saw only the **first waypoint** of each trajectory, forever
- Targets appeared static at `[1.05, 0.08, 0.86]` - reachable without moving base

**Fix**:
```python
def step(self) -> None:
    """Advance trajectory by one timestep."""
    # ... existing phase update for circle/line ...
    
    # For recorded/multi_recorded trajectories, advance waypoint index
    if self.traj_type in ["recorded", "multi_recorded"] and self.recorded_positions is not None:
        self.current_waypoint_idx += 1
        max_idx = self.recorded_positions.shape[1] - 1
        self.current_waypoint_idx = torch.clamp(self.current_waypoint_idx, 0, max_idx)
```

**Verification**:
- Before: Total movement 0.0000 m over 100 steps
- After: Total movement **3.0215 m** over 100 steps ✅

### 2. Lateral Motion Penalty Killed Rotation 🚫 CRITICAL

**File**: `src/rl_platform/tasks/mobile_mm/rewards.py:328`

**Problem**:
- Penalty used world-frame velocity (`root_lin_vel_w`) directly
- As soon as base rotated away from world X-axis, **forward motion appeared in world-Y**
- Weight-2 penalty punished legitimate forward driving
- Policy learned: "Never rotate base to avoid lateral penalty"

**Example**:
- Base rotates 90° clockwise (facing world -Y direction)
- Drives forward at 1 m/s (correct behavior)
- World-frame velocity: `[0, -1, 0]` (forward in robot frame, but -Y in world)
- Lateral penalty: **2.0** (massive penalty for correct behavior!)

**Fix**:
```python
def lateral_motion_penalty(
    base_lin_vel: torch.Tensor,  # World frame
    base_quat: torch.Tensor,      # World frame orientation
    scale: float = 1.0,
) -> torch.Tensor:
    """Penalty for lateral motion in ROBOT frame."""
    # Extract yaw and rotate velocity to robot frame
    w, x, y, z = base_quat[:, 0], base_quat[:, 1], base_quat[:, 2], base_quat[:, 3]
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    
    # Rotate to robot frame
    vel_y_robot = -sin_yaw * base_lin_vel[:, 0] + cos_yaw * base_lin_vel[:, 1]
    
    # Now penalize lateral motion in robot frame
    return scale * vel_y_robot.abs() ** 2
```

**Impact**: Policy can now rotate and drive without false penalties

### 3. Self-Collision False Positives on Ground Contact ⚠️

**File**: `src/rl_platform/tasks/mobile_mm/rewards.py:135`

**Problem**:
- `net_contact_forces` includes **ground contact** on base link
- Base naturally has large ground contact forces (supporting robot weight)
- Self-collision threshold triggered on ground contact
- Policy learned: "Don't move base to avoid collision penalty"

**Fix**:
```python
def self_collision_penalty(
    net_contact_forces: torch.Tensor,
    threshold: float = 1.0,
    scale: float = 1.0,
    continuous: bool = True,
    exclude_base: bool = True,  # NEW: Filter ground contact
) -> torch.Tensor:
    contact_force_mag = torch.norm(net_contact_forces, dim=-1)
    
    # Exclude base link (index 0) which has ground contact
    if exclude_base and contact_force_mag.shape[1] > 1:
        contact_force_mag = contact_force_mag[:, 1:]  # Only check arm links
    
    # ... rest of penalty logic ...
```

**Impact**: Only checks arm-arm and arm-base collisions, ignores ground contact

### 4. Evaluate.py Configuration Bug ✅ FIXED

**File**: `scripts/reinforcement_learning/sb3/evaluate.py:300`

**Problem**:
- Used `gym.make(..., trajectory_type="multi_recorded")` - kwargs ignored
- Environment always defaulted to "circle" trajectory
- Evaluation showed frozen chassis because testing on simple circles

**Fix**:
- Use same config approach as train.py:
```python
from rl_platform.tasks.mobile_mm import MobileMMTrackEEEnvCfg, MobileMMTrackEEEnv
from rl_platform.tasks.mobile_mm.config import TrajectoryConfig

env_cfg = MobileMMTrackEEEnvCfg()
env_cfg.task_config.trajectory = TrajectoryConfig(
    type=args.trajectory_type,
    trajectory_dir=args.trajectory_dir,
    # ... other config ...
)

env = MobileMMTrackEEEnv(cfg=env_cfg)
```

**Impact**: Evaluation now uses correct trajectories matching training

## Verification Results

### Before Fixes
```
Target trajectory analysis:
  Start position: [1.05 0.08 0.86]
  End position:   [1.05 0.08 0.86]
  Total movement: 0.0000 m           ❌ STATIC!
  
Resetting 5 times:
  Reset 1: [1.05 0.08 0.86]           ❌ SAME EVERY TIME
  Reset 2: [1.05 0.08 0.86]
  Reset 3: [1.05 0.08 0.86]
  Reset 4: [1.05 0.08 0.86]
  Reset 5: [1.05 0.08 0.86]

❌ Using simple trajectory (circle/line), NOT multi_recorded!
```

### After Fixes
```
Target trajectory analysis:
  Start position: [1.055649 0.074351 0.859996]
  End position:   [4.05 -0.32136 0.777339]
  Total movement: 3.0215 m           ✅ MOVING!
  Max distance: 3.0314 m
  
Resetting 5 times:
  Reset 1: [1.055649 0.079983 0.859954]  ✅ DIFFERENT
  Reset 2: [1.050784 0.079889 0.860084]
  Reset 3: [1.050212 0.080026 0.859917]
  Reset 4: [1.050188 0.080008 0.860003]
  Reset 5: [1.044351 0.074351 0.859995]

✅ 1038 trajectories loaded and advancing!
   -> Multi-trajectory system is WORKING
   -> Training with 1038 diverse trajectories
```

## Impact on 100M Training Run

The completed 100M timestep training was **wasted**:

1. **Trajectories never advanced** - Only saw first waypoint of each trajectory
2. **Targets were nearly static** - Movement < 0.01m
3. **No chassis movement needed** - Simple arm reaches worked
4. **Lateral penalty suppressed rotation** - Policy learned arm-only tracking
5. **Self-collision on ground** - Policy avoided base movement

**Result**: Policy learned to track **static targets** with **arm only**, never using the 1,038 diverse trajectories requiring chassis movement.

## Next Steps

### Immediate
1. ✅ Verify fixes with diagnostic script
2. ✅ Commit fixes to repository
3. ⏳ Run short test (1M steps) to verify chassis actually moves
4. ⏳ Check reward components during training

### Short-term Training Test (1-10M steps)
- Verify chassis actions != 0
- Check lateral_motion_penalty doesn't spike
- Confirm self_collision_penalty reasonable
- Monitor tracking performance on moving targets

### Full Retraining (100M steps)
- Same hyperparameters as before
- Same trajectory settings: `--trajectory_type multi_recorded --use_all_trajectories`
- Now will actually train on 1,038 diverse trajectories
- Chassis should learn coordinated motion

## Files Modified

1. `src/rl_platform/tasks/mobile_mm/trajectories.py`
   - Added waypoint advancement in `step()`
   - Fixed shape mismatch in `_resample_multi_trajectories()`

2. `src/rl_platform/tasks/mobile_mm/rewards.py`
   - Convert velocity to robot frame for lateral penalty
   - Add `exclude_base` parameter for self-collision
   - Add `base_quat` parameter to `compute_combined_reward()`

3. `src/rl_platform/tasks/mobile_mm/env.py`
   - Pass `base_quat` to reward function

4. `scripts/reinforcement_learning/sb3/evaluate.py`
   - Use proper config approach (not kwargs to gym.make)

5. `scripts/verify_trajectory_loading.py`
   - Fixed to use proper config
   - Fixed conclusion logic
   - Added trajectory advancement test

## Lessons Learned

1. **Always verify assumptions** - We assumed training worked based on good metrics
2. **Diagnostic tools are critical** - verify_trajectory_loading.py revealed the truth
3. **World vs robot frame matters** - Penalties must use correct reference frame
4. **Contact filtering is essential** - Ground contact != self-collision
5. **Test end-to-end** - Beautiful training curves can hide fundamental bugs

## Credits

Analysis and bug identification by user, implementation and verification by development team.
