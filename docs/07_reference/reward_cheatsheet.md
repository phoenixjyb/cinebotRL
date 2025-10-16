# Reward System Quick Reference

## Visual Breakdown 📊

```
┌─────────────────────────────────────────────────────────────────┐
│                    TOTAL REWARD COMPUTATION                     │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────────┐
│   TRACKING REWARDS   │  ← Primary objectives
├──────────────────────┤
│ Position    : +10.0  │  ● Exponential: exp(-error²)
│ Orientation : +2.0   │  ● Quaternion distance
│ Progress    : +1.0   │  ● Improvement bonus
└──────────────────────┘
         ↓ MAXIMIZE

┌──────────────────────┐
│  ACTION PENALTIES    │  ← Efficiency & smoothness
├──────────────────────┤
│ Magnitude   : -0.01  │  ● Energy cost: ||actions||²
│ Rate        : -0.01  │  ● Smoothness: ||Δactions||²
└──────────────────────┘
         ↓ MINIMIZE

┌──────────────────────┐
│  SAFETY PENALTIES    │  ← Constraints
├──────────────────────┤
│ Collision   : -10.0  │  ● Contact forces (disabled)
│ Stability   : -0.1   │  ● Base motion ||velocity||²
│ Obstacles   : ±1.0   │  ● Distance sigmoid (disabled)
└──────────────────────┘
         ↓ MINIMIZE

═══════════════════════════════════════════════════════════════
EXPECTED RANGES:
  Perfect:  +12.0 / step  →  12,000 / episode (20s)
  Good:     +6.0  / step  →   6,000 / episode
  Poor:     +1.0  / step  →   1,000 / episode
═══════════════════════════════════════════════════════════════
```

## Reward Response Curves 📈

### Position Tracking (weight: 10.0)
```
Reward
  10 │ ●
     │  ●
   8 │   ●●
     │     ●●
   6 │       ●●
     │         ●●●
   4 │            ●●●
     │               ●●●●
   2 │                   ●●●●●
     │                        ●●●●●●●●●
   0 └───────────────────────────────────────→ Error (m)
     0   0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1.0

Sharp reward for precise tracking!
```

### Orientation Tracking (weight: 2.0)
```
Reward
   2 │ ●
     │  ●●
 1.5 │    ●●
     │      ●●
   1 │        ●●
     │          ●●●
 0.5 │             ●●●●
     │                 ●●●●●
   0 └───────────────────────────────────────→ Error (deg)
     0   10   20   30   40   50   60   70   80   90

Forgiving for small angles, strict for large
```

## Configuration Example 🔧

```python
# src/rl_platform/tasks/mobile_mm/config.py

@dataclass
class RewardWeights:
    # TRACKING (positive rewards)
    position_tracking: float = 10.0     # ← Most important!
    orientation_tracking: float = 2.0   # ← Secondary
    progress_bonus: float = 1.0         # ← Incentivize improvement
    
    # EFFICIENCY (penalties)
    action_magnitude: float = 0.01      # ← Energy cost
    action_rate: float = 0.01           # ← Smoothness
    
    # SAFETY (penalties)
    collision_penalty: float = 10.0     # ← Hard constraint
    stability_penalty: float = 0.1      # ← Soft constraint
    
    # OBSTACLES (rewards/penalties)
    min_obstacle_distance_weight: float = 1.0
    safety_radius: float = 0.2  # meters
```

## Component Status 🚦

```
✅ Position Tracking        : ACTIVE
✅ Orientation Tracking     : ACTIVE
✅ Progress Bonus           : ACTIVE
✅ Action Magnitude Penalty : ACTIVE
✅ Action Rate Penalty      : ACTIVE (needs prev_action fix)
✅ Stability Penalty        : ACTIVE
⚠️  Collision Penalty       : IMPLEMENTED but DISABLED
⚠️  Obstacle Distance       : IMPLEMENTED but DISABLED
```

## Tuning Cheat Sheet 🎛️

| Problem | Solution | Adjustment |
|---------|----------|------------|
| **Agent not learning** | Increase tracking rewards | `position_tracking = 20.0` |
| **Jerky motion** | Increase smoothness penalty | `action_rate = 0.05` |
| **Too much base motion** | Increase stability penalty | `stability_penalty = 1.0` |
| **Slow convergence** | Decrease action penalties | `action_magnitude = 0.005` |
| **Overfitting to trajectory** | Use multi-trajectory mode | See `multi_trajectory_training.md` |

## Logging & Monitoring 📊

### During Training

All reward components are logged in `extras["reward_components"]`:

```python
extras = {
    "reward_components": {
        "position_tracking": 8.5,           # Current values
        "orientation_tracking": 1.8,
        "progress_bonus": 0.12,
        "action_magnitude_penalty": -0.015,
        "action_rate_penalty": -0.008,
        "collision_penalty": 0.0,
        "stability_penalty": -0.02,
        "obstacle_reward": 0.0,
    }
}
```

### TensorBoard Metrics

```bash
tensorboard --logdir logs/sb3/MobileMMTrackEE-v0

# Check these plots:
# - rollout/ep_rew_mean        : Total episode reward
# - reward_components/*        : Individual component breakdown
# - train/loss                 : Policy loss
# - train/explained_variance   : Value function quality
```

## USD Asset Confirmation ✅

**Your robot is loaded from**:
```
assets_own/usd/mobile_manipulator_PPR_base_corrected.usd
```

**End-effector tracked**:
```
/Robot/left_arm_link6/left_gripper_link
```

**Isaac Lab configuration** (env.py:84-96):
```python
robot_usd_path = str(get_mobile_mm_usd_path())
robot_cfg = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=robot_usd_path,  # ← Your USD file
        activate_contact_sensors=False,
    ),
    ...
)
```

---

## Next Steps 🚀

1. **Test environment**: `python scripts/test_mobile_mm_env.py`
2. **Monitor rewards**: Check `extras["reward_components"]` output
3. **Start training**: Use baseline config, monitor convergence
4. **Tune if needed**: Adjust weights based on learning curves
5. **Enable features**: Add contact sensors, obstacles when stable

**Documentation**: See `docs/reference/reward_system.md` for full details!
