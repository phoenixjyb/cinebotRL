# 🚀 Reachability Map Build - RUNNING!

## ✅ Status: Building Map Successfully

**Current State:** Computing reachability for 15,360 voxels  
**Started:** Just now  
**Expected Completion:** 45-90 minutes (serial processing)  
**Output File:** `matlab/reach_map_mobile_mm_arm_only.mat`

## ✅ Confirmed Working

MATLAB output shows successful initialization:

```
╔════════════════════════════════════════════════════════╗
║  MOBILE MANIPULATOR REACHABILITY MAP BUILDER          ║
╚════════════════════════════════════════════════════════╝

Loading URDF: ...mobile_manipulator_PPR_base_corrected.urdf
  Original DOF: 9
  Extracted arm subtree from "left_arm_base_link"
  Arm-only DOF: 6 (mobile base joints excluded)  ✅
  Links found: 8
  DOF: 6
  Joint limits:
         left_arm_joint1: [-2.88, +2.88] rad
         left_arm_joint2: [+0.00, +3.23] rad
         left_arm_joint3: [-3.32, +0.00] rad
         left_arm_joint4: [-2.88, +2.88] rad
         left_arm_joint5: [-1.66, +1.66] rad
         left_arm_joint6: [-2.88, +2.88] rad
  Orientation bins: 25 (90° cone)
  Grid: 24x32x20 = 15360 voxels @ 50mm resolution  ✅
  Bounds: X[-0.57, 0.58], Y[-0.78, 0.78], Z[-0.38, 0.58] m
  Building self-collision pairs... 21 pairs  ✅

Computing reachability map...
  Progress: [RUNNING]
```

**Key Points:**
- ✅ Arm-only workspace (mobile base joints excluded as requested)
- ✅ 6 DOF arm kinematics loaded correctly
- ✅ Self-collision checking enabled (21 collision pairs)
- ✅ Processing started

## Monitor Progress

### Check Output File Growth

Run this command every 5-10 minutes to see progress:

```powershell
cd C:\Users\yanbo\wSpace\cinebotRL\matlab
dir reach_map_mobile_mm_arm_only.mat
```

**Expected Timeline:**
- **5-10 min:** File appears (~5-10 MB)
- **20-30 min:** File grows to ~25 MB (30% complete)
- **40-60 min:** File grows to ~50 MB (60% complete)
- **60-90 min:** Final size 50-100 MB (100% complete)

### Automatic Monitoring

```powershell
python matlab/monitor_build.py
```

Will check every 30 seconds and report progress.

## What Happens Next

Once the `.mat` file reaches ~50-100 MB and stops growing:

### 1. Verify in MATLAB (Quick Check)

```matlab
cd C:\Users\yanbo\wSpace\cinebotRL\matlab
matlab

% In MATLAB:
load('reach_map_mobile_mm_arm_only.mat')
whos

% Should show:
%   map      - struct with grid, reach_score, manipulability
%   config   - build configuration
%   metadata - robot info
```

### 2. Visualize (Interactive 3D)

```matlab
% Mode 5 is BEST - shows robot + reachability in world frame
visualize_reachability('reach_map_mobile_mm_arm_only.mat', 'mode', 5, 'threshold', 0.3)

% Other modes:
% Mode 1: 3D voxel cloud (color-coded by reachability)
% Mode 2: Horizontal slice at specific height
% Mode 3: Manipulability distribution slice
% Mode 4: Top-down view with workspace circles
```

**What to expect in visualization:**
- Black circle at [0, 0, 0] = mobile base (ground)
- Red star at [0.16, 0, 0.9465] = arm shoulder
- Color gradient: Blue (low reach) → Red (high reach)
- Robot model aligned with reachability cloud

### 3. Test Python Loader

```powershell
cd C:\Users\yanbo\wSpace\cinebotRL

# Quick test
python -c "from scripts.reachability_utils import ReachabilityMap; m = ReachabilityMap('matlab/reach_map_mobile_mm_arm_only.mat'); print(m)"
```

**Expected output:**
```
ReachabilityMap loaded:
  Grid: 24 × 32 × 20 voxels
  Voxel size: 0.05 m
  Total voxels: 15360
  Reachable: ~8000 (50-60%)
  Arm offset from mobile base: [0.16, 0.00, 0.9465]
  Device: cuda
```

### 4. Test with Real Trajectory Data

```python
import torch
from scripts.reachability_utils import ReachabilityMap

# Load map
reach_map = ReachabilityMap('matlab/reach_map_mobile_mm_arm_only.mat')

# From your trajectory analysis:
# - First waypoint: [1.08, 0.08, 0.862308] (close, should be reachable)
# - Mean distance: 2.3m (far, should be unreachable)
# - Max distance: 4.1m (very far, definitely unreachable)

# Test close target (first waypoint)
close_target = torch.tensor([[1.08, 0.08, 0.862308]])
score_close = reach_map.query_batch(close_target, in_mobile_frame=True)
print(f"Close target (1.08m): reachability = {score_close[0]:.3f}")
# Expected: 0.6-1.0 (reachable)

# Test far target (typical from trajectory)
far_target = torch.tensor([[2.5, 0.0, 0.9]])
score_far = reach_map.query_batch(far_target, in_mobile_frame=True)
print(f"Far target (2.5m): reachability = {score_far[0]:.3f}")
# Expected: 0.0-0.3 (unreachable, needs base movement)
```

**This validates the key finding from `analyze_trajectory_reach.py`:**
- 83.8% of waypoints beyond arm reach → Need base movement
- Reachability map will show low scores for these → Triggers base mobilization

## Why This Matters for Your Training

**Session 7 (no reachability):**
```python
# Agent sees target at [2.5, 0, 0.9] (2.5m away)
# No reachability signal
# Tries arm-only reach → Fails → Penalty
# Tries again → Fails → Penalty
# Wastes 100+ samples before exploring base movement
```

**Session 8 (with reachability):**
```python
# Agent sees target at [2.5, 0, 0.9]
reach_score = 0.15  # Low! (from map)
# Clear signal: "arm can't reach this, need base movement"
# Bonus reward for base mobilization
# Learns faster: base movement necessary for far targets
```

## Integration Plan for Session 8

Once map is built and tested:

```python
# In src/rl_platform/tasks/mobile_mm/env.py

# Add at class level:
from scripts.reachability_utils import ReachabilityMap

# In __init__():
reach_map_path = Path(__file__).parent.parent.parent.parent / "matlab" / "reach_map_mobile_mm_arm_only.mat"
self.reach_map = ReachabilityMap(str(reach_map_path), device=self._device)

# In _compute_rewards():
# Query reachability for current EE target
target_ee = self._target_ee_pos  # (N, 3) in mobile base frame
reach_scores = self.reach_map.query_batch(target_ee, in_mobile_frame=True)

# Strategy 1: Scale tracking reward by reachability
# (Reduces reward for unreachable targets → encourages base movement)
tracking_reward_shaped = tracking_reward * (0.3 + 0.7 * reach_scores)

# Strategy 2: Bonus for base mobilization when reachability low
# (Clear signal: "move base when target is far")
base_bonus = torch.where(
    reach_scores < 0.3,  # Unreachable
    base_mobilization * 2.0,  # Double bonus
    base_mobilization * 1.0   # Normal reward
)

# Add to total reward
total_reward = total_reward + base_bonus
```

**Expected Session 8 Improvements:**
- 30-40% faster learning (skip impossible arm-only attempts)
- Base mobilization > 100 reward/step (vs ~0 in Session 7)
- Better final performance (uses both arm + base optimally)

## Current Build Progress

**Check terminal:** `9874d4a1-6b1c-4922-9233-d7f2b4f21e46`  
**Check file:** Every 10 minutes for size growth  
**Estimated completion:** In 45-90 minutes from start

---

**Last updated:** Just now  
**Status:** ✅ Building successfully  
**Next check:** In 10-15 minutes
