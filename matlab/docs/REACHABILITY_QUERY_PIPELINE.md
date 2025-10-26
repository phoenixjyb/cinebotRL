# Reachability Query Pipeline: Coordinate Transforms

## Complete Flow During RL Training

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RL TRAINING STEP                             │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 1. TRAJECTORY WAYPOINT (Mobile Base Frame)                          │
│    - Isaac Lab env provides target EE pose                          │
│    - Coordinate system: abstract_chassis_link (mobile base)         │
│    - Example: target_ee = [0.66, 0.0, 0.9465] m                    │
│                           └─ 0.66m forward from mobile base center  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 2. QUERY REACHABILITY MAP                                           │
│    Python: rmap.query_batch(target_ee, in_mobile_frame=True)       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 3. AUTO-TRANSFORM: Mobile Base → Arm Base                          │
│    target_arm = target_mobile - arm_offset                          │
│    [0.66, 0.0, 0.9465] - [0.16, 0, 0.9465] = [0.5, 0, 0]          │
│                                                                      │
│    Meaning: Target is 0.5m in front of shoulder, at shoulder height│
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 4. KDTREE LOOKUP (Arm Frame)                                       │
│    - Find nearest voxel to [0.5, 0, 0] in arm frame                │
│    - Return reachScore from pre-computed map                        │
│    - Score ∈ [0, 1]: 0=unreachable, 1=fully reachable              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 5. USE IN REWARD SHAPING                                            │
│    Option A: Scale tracking reward                                  │
│      shaped_reward = tracking_reward * reach_score                  │
│                                                                      │
│    Option B: Filter unreachable targets                             │
│      valid = reach_score > 0.5                                      │
│      target_ee = target_ee[valid]                                   │
│                                                                      │
│    Option C: Bonus for high manipulability                          │
│      bonus = manip_score * 0.1                                      │
│      total_reward = tracking_reward + bonus                         │
└─────────────────────────────────────────────────────────────────────┘
```

## Coordinate Frame Summary

### Frame Hierarchy
```
World Frame
    └─> Mobile Base Frame (abstract_chassis_link)
            └─> Arm Base Frame (left_arm_base_link)
                    └─> End-Effector Frame (left_gripper_link)
```

### Transform: Mobile Base → Arm Base
```python
# From URDF: <origin xyz="0.160 0.000 0.9465" rpy="0 0 0"/>
ARM_OFFSET = [0.16, 0.0, 0.9465]  # [x_forward, y_left, z_up]

# Transform equation
target_in_arm_frame = target_in_mobile_frame - ARM_OFFSET
```

### Numerical Example

**Scenario:** Target 0.5m in front of shoulder, at shoulder height

| Frame | Coordinates [x, y, z] | Reference Point |
|-------|----------------------|-----------------|
| **Mobile Base** | [0.66, 0.0, 0.9465] | Mobile base center (ground) |
| **Arm Base** | [0.5, 0.0, 0.0] | Shoulder mount point |

**Calculation:**
```
Arm Base = Mobile Base - Offset
[0.5, 0, 0] = [0.66, 0, 0.9465] - [0.16, 0, 0.9465]
```

**Physical Interpretation:**
- Mobile base frame: "0.66m forward from chassis, at height 0.9465m"
- Arm base frame: "0.5m in front of shoulder, at shoulder level"
- **Same physical point, different reference frames**

## Code Integration Examples

### Example 1: Simple Reward Scaling
```python
# In your environment's reward computation
from scripts.reachability_utils import ReachabilityMap

class MyRobotEnv:
    def __init__(self):
        self.rmap = ReachabilityMap(
            "matlab/reach_map_arm.mat",
            arm_offset=[0.16, 0, 0.9465],
            device=self.device
        )
    
    def compute_rewards(self):
        # Target from trajectory (mobile base frame)
        target_ee = self.trajectory.get_waypoint(self.step)  # (N, 3)
        
        # Query reachability
        reach_scores, _, _ = self.rmap.query_batch(
            target_ee,
            in_mobile_frame=True  # Auto-transform
        )
        
        # Tracking reward
        tracking_error = torch.norm(self.current_ee - target_ee, dim=-1)
        tracking_reward = -tracking_error
        
        # Scale by reachability
        shaped_reward = tracking_reward * reach_scores
        
        return shaped_reward
```

### Example 2: Curriculum Learning
```python
# Start with easy (highly reachable) targets, progress to harder ones
class CurriculumEnv:
    def reset(self):
        # Get trajectory waypoints
        all_waypoints = self.trajectory.get_all_waypoints()  # (T, 3)
        
        # Query reachability for all waypoints
        reach_scores = self.rmap.query_batch(
            all_waypoints,
            in_mobile_frame=True
        )
        
        # Get threshold based on training progress
        threshold = self.get_curriculum_threshold(self.global_step)
        # Early: 0.8 (only very reachable)
        # Late: 0.3 (include harder targets)
        
        # Filter waypoints
        valid_mask = reach_scores >= threshold
        filtered_waypoints = all_waypoints[valid_mask]
        
        # Use filtered trajectory
        self.trajectory.set_waypoints(filtered_waypoints)
        
        return self.get_obs()
```

### Example 3: Manipulability Bonus
```python
# Reward not just reaching target, but doing so with good manipulability
def compute_rewards(self):
    target_ee = self.get_target()
    
    # Query both reachability and manipulability
    reach_scores, manip_scores, _ = self.rmap.query_batch(
        target_ee,
        in_mobile_frame=True,
        return_manipulability=True
    )
    
    # Tracking reward
    tracking_reward = -torch.norm(self.current_ee - target_ee, dim=-1)
    
    # Manipulability bonus (encourage dexterous poses)
    manip_bonus = manip_scores * 0.1  # Small bonus
    
    # Combined reward
    total_reward = tracking_reward * reach_scores + manip_bonus
    
    return total_reward
```

## Key Takeaways

✅ **During RL Training:**
1. Target waypoint is in **mobile base frame** (from Isaac Lab)
2. Python `query_batch()` **auto-transforms** to arm frame
3. Lookup happens in **arm frame** (where map is built)
4. Result: reachability score [0,1] for that target

✅ **You Don't Need to Manually Transform!**
- Just pass `in_mobile_frame=True` (default)
- Python handles the subtraction: `arm = mobile - offset`

✅ **Why This Design?**
- **Map building:** Clean arm-only kinematics (no mobile base joints)
- **Training:** Natural mobile base frame (where Isaac Lab works)
- **Query:** Automatic transform (best of both worlds)

## Verification

To verify the transform is correct, you can manually check:

```python
# Test case
target_mobile = torch.tensor([[0.66, 0.0, 0.9465]])  # Mobile base frame

# Query with auto-transform
score_auto = rmap.query_batch(target_mobile, in_mobile_frame=True)

# Manual transform and query
target_arm = target_mobile - torch.tensor([[0.16, 0, 0.9465]])  # = [0.5, 0, 0]
score_manual = rmap.query_batch(target_arm, in_mobile_frame=False)

# Should be identical
assert torch.allclose(score_auto, score_manual)
print(f"✓ Transform verified: {score_auto.item():.3f} == {score_manual.item():.3f}")
```

## Summary

**Question:** "Do we convert waypoint into arm base frame to check reachability?"

**Answer:** Yes! The Python loader does this automatically:

```python
# You provide: mobile base frame coordinates
target = [0.66, 0.0, 0.9465]

# Loader transforms: subtract arm offset
target_arm = [0.66-0.16, 0.0-0.0, 0.9465-0.9465] = [0.5, 0, 0]

# Lookup happens: in arm frame
score = map_query([0.5, 0, 0])  # Returns reach score
```

**Just use:** `rmap.query_batch(targets, in_mobile_frame=True)` and it handles everything! 🎉
