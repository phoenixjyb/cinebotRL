# Updated Reward System - Quick Summary

## ✅ All Updates Complete

### Physical Constraints Added

```python
# Mobile Base (Differential Drive)
max_linear_velocity = 1.5 m/s
max_angular_velocity = 2.0 rad/s (yaw only)
max_linear_acceleration = 5.0 m/s²
max_linear_jerk = 5.0 m/s³

# Arm Joints
max_joint_velocity = 2.0 rad/s (motor speed)
max_joint_acceleration = 10.0 rad/s²
joint_limit_margin = 0.1 radians

# Control
control_frequency = 10 Hz (matches trajectory 100ms spacing)
physics_frequency = 200 Hz
```

### 🚨 Self-Collision Detection (CRITICAL!)

```python
# Contact sensors: ENABLED
activate_contact_sensors = True

# Penalty configuration
self_collision_penalty: 50.0  # High weight!
self_collision_threshold: 1.0 N  # Start penalizing
self_collision_termination_threshold: 10.0 N  # End episode

# Behavior
- Continuous penalty scaling with force magnitude
- Episode terminates on hard collision (>10 N)
- Prevents arm hitting base, self-intersections
```

### All Penalty Components (14 total)

| Component | Weight | Type |
|-----------|--------|------|
| Position tracking | 10.0 | Reward ✅ |
| Orientation tracking | 2.0 | Reward ✅ |
| Progress bonus | 1.0 | Reward ✅ |
| Action magnitude | 0.01 | Penalty |
| Action rate | 0.01 | Penalty |
| Action smoothness (jerk) | 0.05 | Penalty |
| Velocity limit | 5.0 | Penalty |
| Acceleration limit | 5.0 | Penalty |
| Jerk limit | 3.0 | Penalty |
| Joint limit | 10.0 | Penalty |
| Lateral motion | 2.0 | Penalty |
| **Self-collision** | **50.0** | **Penalty 🚨** |
| Stability | 0.1 | Penalty |
| Obstacle distance | 1.0 | Reward (disabled) |

### Expected Reward Range

**With new constraints**:
- Perfect: +10-12 per step (all constraints satisfied)
- Good: +5-8 per step (minor violations)
- Poor: +1-3 per step (frequent violations)
- Collision: -50 per step! (self-collision)

**Episode total** (20s @ 10Hz = 200 steps):
- Perfect: ~2,000 cumulative reward
- Good: ~1,000-1,500
- Poor: ~200-600
- With collisions: Can go negative!

### Files Modified

1. **config.py**:
   - Added `RobotLimits` dataclass
   - Updated `RewardWeights` with 6 new penalties
   - Added `self_collision_*` parameters
   - Changed decimation to 20 (10Hz control)
   - Added `trajectory_dt = 0.1s`

2. **rewards.py**:
   - Added `velocity_limit_penalty()`
   - Added `acceleration_limit_penalty()`
   - Added `jerk_penalty()`
   - Added `joint_limit_penalty()`
   - Added `lateral_motion_penalty()`
   - Added `self_collision_penalty()` ← CRITICAL
   - Updated `compute_combined_reward()` signature

3. **env.py**:
   - Enabled contact sensors: `activate_contact_sensors=True`
   - Added state history tracking (prev_prev_actions, prev_base_accel, etc.)
   - Extract joint limits from USD in `_setup_scene()`
   - Updated `_get_rewards()` to compute derivatives
   - Updated `_get_dones()` to check self-collision
   - Added self-collision termination logic

### Testing Checklist

Before training:

```bash
# 1. Test environment loading
python scripts/test_mobile_mm_env.py --num_envs 4

# Expected output:
# - Joint limits extracted: Lower/Upper arrays
# - Control frequency: 10.0 Hz
# - Trajectory dt: 0.100s
# - Reward components: 14 items
# - Self-collision penalty present

# 2. Check reward computation
# Should see in logs:
extras["reward_components"] = {
    "position_tracking": X,
    "orientation_tracking": X,
    ...
    "self_collision_penalty": X,  # ← Should be present!
}

# 3. Monitor during training
tensorboard --logdir logs/sb3/MobileMMTrackEE-v0
# Watch:
# - self_collision_penalty should decrease to ~0
# - velocity/acceleration penalties should decrease
# - tracking rewards should increase
```

### Termination Conditions

1. **Self-collision** (NEW!): Contact force > 10 N
2. **Tracking error**: Distance > 2.0 m
3. **Timeout**: Episode reaches 200 steps (20s @ 10Hz)

### Key Behavior Changes

**Before**:
- No physical limits enforced
- No self-collision detection
- Agent could violate physics freely

**After**:
- Velocity capped at 1.5 m/s, 2 rad/s
- Acceleration capped at 5 m/s²
- Jerk limited to 5 m/s³
- **Self-collision strongly penalized (weight 50!) and terminates episode**
- No lateral motion (differential drive constraint)
- Joint limits respected with 0.1 rad margin
- Control runs at realistic 10 Hz

### Documentation

- Full details: `docs/reference/robot_constraints_updated.md`
- Reward cheatsheet: `docs/reference/reward_cheatsheet.md` (needs update)
- Multi-trajectory guide: `docs/workflows/multi_trajectory_training.md`

---

## Ready to Train! 🚀

Your system now:
- ✅ Uses your custom USD robot
- ✅ Tracks with left_gripper_link
- ✅ Enforces realistic physical constraints
- ✅ **Prevents self-collision (CRITICAL!)**
- ✅ Matches trajectory timing (10 Hz = 100ms waypoints)
- ✅ Supports 1000+ trajectories for diverse training
- ✅ Logs all 14 reward components for monitoring

Next step: Test with `scripts/test_mobile_mm_env.py`!
