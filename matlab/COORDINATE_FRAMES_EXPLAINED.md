# Reachability Map Coordinate Frames

## Problem Statement

The reachability map needs to be **anchored to the arm base** (shoulder), not the ground/mobile base. This ensures:
1. Map doesn't "sit on the ground" when mobile base moves
2. Easy coordinate transformations during RL training
3. Clear separation between mobile base motion and arm workspace

## Coordinate Frame Hierarchy

```
World Frame
    ↓
Mobile Base Frame (abstract_chassis_link)
    ↓ [translation: 0.16m forward, 0.9465m up]
Arm Base Frame (left_arm_base_link)  ← **MAP IS BUILT HERE**
    ↓ [kinematic chain: 7 joints]
End-Effector Frame (left_gripper_link)
```

## Key Transforms

### From URDF (mobile_manipulator_PPR_base_corrected.urdf)

**Arm Base Position (relative to mobile base):**
```xml
<origin xyz="0.160 0.000 0.9465" rpy="0 0 0"/>
```

This means:
- X: 0.16m forward from mobile base center
- Y: 0.0m (centered laterally)
- Z: 0.9465m above mobile base center (shoulder height)

### Transform Equations

**Mobile Base → Arm Base:**
```
arm_base_position = mobile_base_position + [0.16, 0, 0.9465]
```

**Target Transform (Mobile → Arm):**
```
target_arm_frame = target_mobile_frame - [0.16, 0, 0.9465]
```

**Example:**
- Target in mobile base: `[0.66, 0.0, 0.9465]`
- Target in arm frame: `[0.66-0.16, 0.0-0.0, 0.9465-0.9465] = [0.5, 0.0, 0.0]`
  - Meaning: 0.5m in front of shoulder, at shoulder height

## MATLAB Reachability Map

### Configuration (build_reachability_map.m)

```matlab
% BASE_LINK points to arm base (shoulder)
BASE_LINK = "left_arm_base_link";  

% Grid relative to ARM BASE (shoulder origin)
GRID_ORIGIN = [-0.6, -0.8, -0.4];   % min [x,y,z] from shoulder
GRID_SIZE   = [ 1.2,  1.6,  1.0];   % workspace: 1.2m×1.6m×1.0m
VOXEL       = [ 0.05, 0.05, 0.05];  % 5cm resolution
```

### What This Means

The map covers a workspace **around the shoulder**:
- X: -0.6m to +0.6m (behind and in front of shoulder)
- Y: -0.8m to +0.8m (left and right of shoulder)
- Z: -0.4m to +0.6m (below and above shoulder)

**Important:** All coordinates are in the **arm base frame**. The shoulder is at the origin `[0, 0, 0]` in this frame.

## Python Integration (reachability_utils.py)

### Loading Map

```python
from scripts.reachability_utils import ReachabilityMap

# Load with arm offset
rmap = ReachabilityMap(
    "matlab/reach_map_arm.mat",
    arm_offset=[0.16, 0, 0.9465],  # From URDF
    device="cuda"
)
```

### Querying During Training

```python
# Targets from RL environment (in mobile base frame)
target_ee_mobile = torch.tensor([[0.66, 0.0, 0.9465]])  # (N, 3)

# Query map (auto-transforms to arm frame)
scores, manip, seeds = rmap.query_batch(
    target_ee_mobile,
    in_mobile_frame=True  # Default: True
)

# scores[0] = reachability at [0.5, 0.0, 0.0] in arm frame
```

### Manual Transform (if needed)

```python
# If you already have targets in arm frame
target_ee_arm = rmap.transform_mobile_to_arm_frame(target_ee_mobile)
# target_ee_arm = [0.5, 0.0, 0.0]

# Query without transform
scores = rmap.query_batch(target_ee_arm, in_mobile_frame=False)
```

## Visualization (visualize_reachability.m)

When you run the visualization:

```matlab
visualize_reachability()
```

You will see:
- **Red star** at `[0, 0, 0]`: This is the **shoulder** (arm base origin)
- **Voxels** clustered around origin: Reachable workspace relative to shoulder
- **Grid bounds**: -0.6 to +0.6m in X, -0.8 to +0.8m in Y, -0.4 to +0.6m in Z

The visualization is in **arm base frame**, so the origin is the shoulder mount point.

## Common Pitfalls (AVOID!)

### ❌ WRONG: Map relative to mobile base
```matlab
% BAD: Uses abstract_chassis_link as BASE_LINK
BASE_LINK = "abstract_chassis_link";  
GRID_ORIGIN = [0, 0, 0];  % Origin at ground (mobile base)
```
**Problem:** Map "sits on ground". When mobile base moves, the map doesn't move with the arm!

### ✅ CORRECT: Map relative to arm base
```matlab
% GOOD: Uses left_arm_base_link as BASE_LINK
BASE_LINK = "left_arm_base_link";
GRID_ORIGIN = [-0.6, -0.8, -0.4];  % Origin at shoulder
```
**Benefit:** Map anchored to shoulder. Mobile base can move freely, just apply offset transform!

## Verification Checklist

✓ **MATLAB builder:**
- [ ] `BASE_LINK = "left_arm_base_link"` (not abstract_chassis_link)
- [ ] `GRID_ORIGIN` centered around `[0, 0, 0]` (symmetric around shoulder)
- [ ] Robot subtree excludes virtual mobile joints (joint_x, joint_y, joint_theta)

✓ **Python loader:**
- [ ] `arm_offset = [0.16, 0, 0.9465]` matches URDF
- [ ] `in_mobile_frame=True` by default in `query_batch()`
- [ ] Transform subtracts offset: `target_arm = target_mobile - offset`

✓ **Visualization:**
- [ ] Red star at `[0, 0, 0]` shows shoulder position
- [ ] Voxels clustered around origin (not ground)
- [ ] Grid bounds symmetric: X±0.6m, Y±0.8m, Z[-0.4,+0.6]m

## Summary

**Key Point:** The reachability map is **anchored to the arm base (shoulder)**, at coordinates `[0.16, 0, 0.9465]` relative to the mobile base. This design:
1. Keeps map independent of mobile base motion
2. Simplifies coordinate transforms (just subtract offset)
3. Matches physical intuition (arm workspace around shoulder)

During RL training:
- Isaac Lab gives targets in **mobile base frame**
- Python loader **auto-transforms** to arm frame
- Map lookup happens in **arm frame**
- Result: reachability score [0,1] for that target

**Visual Check:** When you visualize the map, the shoulder should be at the center (origin), with reachable voxels forming a hemisphere/shell around it.
