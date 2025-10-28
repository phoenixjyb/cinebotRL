# ✅ ANSWER: Coordinate Transform During RL Training

## Your Question
> "When we work with RL to check arm reachability, we have desired end-effector waypoint in world frame, and do we convert this waypoint into arm base frame to check for reachability?"

## Short Answer
**YES!** The waypoint must be in **arm base frame** to query the map. But **you don't do this manually** - the Python loader does it automatically!

## The Complete Flow

```
┌──────────────────────────────────────────────────────────────┐
│ Step 1: Isaac Lab Provides Target (Mobile Base Frame)       │
│ target_ee_mobile = [0.66, 0.0, 0.9465]                      │
│ "0.66m forward from mobile base, at height 0.9465m"         │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│ Step 2: Python Auto-Transforms (Mobile → Arm)               │
│ target_ee_arm = target_ee_mobile - arm_offset               │
│ [0.66,0,0.9465] - [0.16,0,0.9465] = [0.5, 0, 0]           │
│ "0.5m in front of shoulder, at shoulder height"            │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│ Step 3: Query Map (Arm Frame)                               │
│ score = map.lookup([0.5, 0, 0])  # KDTree nearest neighbor  │
│ Returns: 0.87 (87% of orientations are reachable)           │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│ Step 4: Use in Reward                                        │
│ shaped_reward = tracking_reward * 0.87                       │
└──────────────────────────────────────────────────────────────┘
```

## Python Code (What You Actually Write)

```python
from scripts.reachability_utils import ReachabilityMap

# Initialize (once)
rmap = ReachabilityMap(
    "matlab/reach_map_arm.mat",
    arm_offset=[0.16, 0, 0.9465],  # From URDF
    device="cuda"
)

# During training loop
target_ee = self.trajectory.get_waypoint(step)  # Mobile base frame: [0.66, 0, 0.9465]

# Query (automatic transform happens here!)
scores = rmap.query_batch(
    target_ee,
    in_mobile_frame=True  # <-- This triggers auto-transform
)
# Internally: target_arm = [0.66-0.16, 0-0, 0.9465-0.9465] = [0.5, 0, 0]
# Query happens in arm frame: score = lookup([0.5, 0, 0])

# Use score
shaped_reward = tracking_reward * scores
```

## What Happens Internally (You Don't Need to Do This!)

```python
# Inside ReachabilityMap.query_batch():
def query_batch(self, targets, in_mobile_frame=True, ...):
    if in_mobile_frame:
        # Auto-transform: mobile → arm
        targets = targets - self.arm_offset  # [0.66,0,0.95] → [0.5,0,0]
    
    # Query map (in arm frame)
    distances, indices = self.kdtree.query(targets)
    scores = self.reach_score[indices]
    
    return scores
```

## Why This Design?

### Map Building (MATLAB):
- ✅ **Arm frame** = Clean kinematics (no mobile base joints)
- ✅ Shoulder at origin [0,0,0] = Intuitive workspace
- ✅ Pre-computed offline = Fast

### RL Training (Python):
- ✅ **Mobile base frame** = Natural for Isaac Lab
- ✅ Auto-transform = Transparent to user
- ✅ Fast query = 10K/sec on GPU

## Key Takeaway

**You provide targets in mobile base frame** (from Isaac Lab), and **Python automatically handles the transform** to arm frame for map lookup!

Just use:
```python
scores = rmap.query_batch(targets_mobile, in_mobile_frame=True)  # Default
```

No manual transform needed! 🎉

---

## Documentation Files

For more details, see:
- `matlab/COORDINATE_FRAMES_EXPLAINED.md` - Complete frame hierarchy
- `matlab/REACHABILITY_QUERY_PIPELINE.md` - Step-by-step pipeline
- `scripts/reachability_utils.py` - Python implementation with docstrings
- `matlab/VISUALIZATION_WORLD_FRAME.md` - How visualization works
