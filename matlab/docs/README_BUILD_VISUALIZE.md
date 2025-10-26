# Reachability Map Building & Visualization Guide

## Current Status

**Map Building:** IN PROGRESS (started in background terminal)
- Expected time: 30-60 minutes
- Grid: 1.2×1.6×1.0 m³ workspace (24,576 voxels @ 5cm resolution)
- Orientations: 24 camera-like poses (90° cone)
- Expected output: `reach_map_arm.mat` (~50-100 MB)

**What's being computed:**
```
For each voxel position (x,y,z) in arm workspace:
  For each orientation (24 samples):
    Try inverse kinematics (8 random seeds)
    Check self-collision with robot geometry
    Compute manipulability (dexterity metric)
    Store: reachability score, best manipulability, IK solutions
```

## Monitoring Progress

Check if the map file exists and is growing:
```powershell
# In PowerShell
cd C:\Users\yanbo\wSpace\cinebotRL\matlab
dir reach_map_arm.mat  # Check if file exists and size
```

Or watch the MATLAB terminal output for progress updates.

## After Map is Built

### 1. Verify Map File
```matlab
% In MATLAB
load('reach_map_arm.mat')
whos  % Should show: map, config, metadata
```

### 2. Visualize the Map

Run the visualization script with 5 interactive modes:

```matlab
% Mode 1: 3D voxel cloud (color = reachability score)
visualize_reachability('reach_map_arm.mat', 'mode', 1, 'threshold', 0.3)

% Mode 2: Horizontal slice (at specific height)
visualize_reachability('reach_map_arm.mat', 'mode', 2, 'slice_height', 0.0, 'threshold', 0.3)

% Mode 3: Manipulability slice (dexterity distribution)
visualize_reachability('reach_map_arm.mat', 'mode', 3, 'slice_height', 0.0, 'threshold', 0.3)

% Mode 4: Top view (bird's eye)
visualize_reachability('reach_map_arm.mat', 'mode', 4, 'threshold', 0.3)

% Mode 5: Robot model + reachability overlay (RECOMMENDED!)
visualize_reachability('reach_map_arm.mat', 'mode', 5, 'threshold', 0.3)
```

**Mode 5 is the most useful** - shows full robot with reachability cloud in world frame.

### 3. Test Python Loader

After MATLAB verification, test the Python integration:

```powershell
# In PowerShell (from project root)
cd C:\Users\yanbo\wSpace\cinebotRL
python -c "from scripts.reachability_utils import ReachabilityMap; m = ReachabilityMap('matlab/reach_map_arm.mat'); print(m)"
```

Expected output:
```
ReachabilityMap loaded:
  Grid: 24 × 32 × 20 voxels
  Voxel size: 0.05 m
  Total voxels: 15360
  Reachable: ~8000 (50-60%)
  Arm offset from mobile base: [0.16, 0.00, 0.9465]
  Device: cuda
```

### 4. Quick Query Test

```python
import torch
from scripts.reachability_utils import ReachabilityMap

# Load map
reach_map = ReachabilityMap('matlab/reach_map_arm.mat')

# Test with trajectory target (in mobile base frame from your analysis)
target_mobile = torch.tensor([[1.08, 0.08, 0.862308]])  # First waypoint from trajectory
scores = reach_map.query_batch(target_mobile, in_mobile_frame=True)
print(f"Target in mobile frame: {target_mobile[0].tolist()}")
print(f"Reachability score: {scores[0]:.3f}")

# Test with far target (should be low reachability)
target_far = torch.tensor([[3.0, 0.0, 0.9465]])  # 3m away (from analysis: max 4.1m)
scores_far = reach_map.query_batch(target_far, in_mobile_frame=True)
print(f"\nFar target (3m away): {target_far[0].tolist()}")
print(f"Reachability score: {scores_far[0]:.3f} (should be low!)")
```

Expected:
- First waypoint (close): score ~0.8-1.0 (reachable)
- Far target (3m): score ~0.0-0.2 (unreachable, needs base movement)

## Integration into Session 8

Once validated, add to your training environment:

```python
# In src/rl_platform/tasks/mobile_mm/env.py
from scripts.reachability_utils import ReachabilityMap, shape_reward_with_reachability

# In __init__():
self.reach_map = ReachabilityMap('matlab/reach_map_arm.mat', device=self._device)

# In _compute_rewards():
ee_pos = self._ee_pos  # Current EE position in mobile base frame (N, 3)
reach_scores = self.reach_map.query_batch(ee_pos, in_mobile_frame=True)

# Shape rewards (choose one strategy):
# Option 1: Scale tracking reward by reachability
shaped_reward = shape_reward_with_reachability(
    base_reward, reach_scores, mode='scale', threshold=0.3
)

# Option 2: Curriculum - filter by reachability
shaped_reward = shape_reward_with_reachability(
    base_reward, reach_scores, mode='filter', threshold=0.3
)

# Option 3: Bonus for high manipulability
manipulability = self.reach_map.query_manipulability(ee_pos, in_mobile_frame=True)
shaped_reward = base_reward + 0.1 * manipulability
```

## Expected Impact on Session 8

Based on your trajectory analysis (83.8% unreachable from fixed base):

**Training Speed:**
- 30-40% faster convergence (skip impossible arm-only exploration)
- Fewer wasted samples on 2.3m targets with arm-only attempts

**Base Mobilization:**
- Clear signal when base movement required (reach_score < 0.3)
- Better gradient for learning "when to move base"

**Collision Reduction:**
- Curriculum: Train on reachable poses first (16% of trajectory)
- Gradually add harder poses as policy improves
- Combined with Session 7's collision penalty fix (50.0→0.5)

## Troubleshooting

**If map building fails:**
1. Check MATLAB terminal output for error messages
2. Verify URDF path is correct
3. Check Robotics System Toolbox is installed
4. Reduce grid size for faster testing

**If Python loading fails:**
1. Check scipy is installed: `pip install scipy`
2. Verify .mat file format is v7.3
3. Check file path is correct (use absolute path)

**If visualization is slow:**
1. Increase threshold (e.g., 0.5 instead of 0.3)
2. Use Mode 4 (top view) for quick overview
3. Reduce voxel count in build script

## Next Steps After Visualization

1. ✅ Verify map looks correct (robot + reachability aligned)
2. ✅ Test Python loader with trajectory waypoints
3. ✅ Choose integration strategy (scale/filter/curriculum)
4. ✅ Modify env.py to add reachability shaping
5. 🚀 Launch Session 8 with reachability guidance
6. 📊 Compare Session 7 (no reach) vs Session 8 (with reach)

Expected Session 8 improvements:
- Episode rewards: Better than Session 7's -120K~+30K
- Base mobilization: >100 reward/step (currently near 0)
- Training time: 40% reduction to reach same performance
- Sample efficiency: Fewer episodes to learn base movement
