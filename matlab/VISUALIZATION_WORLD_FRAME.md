# Reachability Map Visualization - World Frame Integration

## Problem Solved

The reachability map is built in **arm base frame** (shoulder origin), but when visualizing with the robot model, we need to show everything in **world frame** (mobile base at ground). This update transforms the map voxels from arm frame → world frame for proper visualization.

## Key Changes

### 1. Arm Offset Parameter
```matlab
ARM_OFFSET = [0.16, 0.0, 0.9465];  % Shoulder position relative to mobile base
```

### 2. Full Robot Model Loading
- **Before:** Loaded arm-only subtree (shoulder as root)
- **After:** Loads full robot with mobile base (includes 3 virtual planar joints)
- **Why:** Need full robot to show mobile base on ground with arm mounted on top

### 3. Coordinate Transform in All Visualization Modes

#### Voxel Cloud
```matlab
% Map voxels are in arm frame
X_arm = map.grid.origin(1) + ...
Y_arm = map.grid.origin(2) + ...
Z_arm = map.grid.origin(3) + ...

% Transform to world frame
X_world = X_arm + ARM_OFFSET(1);
Y_world = Y_arm + ARM_OFFSET(2);
Z_world = Z_arm + ARM_OFFSET(3);
```

#### Slices
- Converts slice height from world → arm frame for indexing
- Plots grid in world coordinates
- Shows arm base marker at correct world position

#### Top View
- Transforms X,Y grid to world frame
- Shows both mobile base (at origin) and arm base (at offset)
- Workspace radius centered at arm base

#### Robot + Reach
- Robot model at world origin (mobile base on ground)
- Reachability voxels transformed to world frame
- Semi-transparent overlay shows reachable workspace around arm
- Markers: Red star/circle = arm base, Green circle = end-effector

## Visualization Elements

### Markers in World Frame

**Mobile Base (Ground Level):**
- Black circle with + at `[0, 0, 0]`
- Label: "Mobile Base"

**Arm Base (Shoulder):**
- Red star + circle at `[0.16, 0, 0.9465]`
- Label: "ARM BASE (Shoulder)"
- RGB coordinate axes (X=red, Y=green, Z=blue)

**End-Effector (when robot shown):**
- Green circle at actual EE position from FK

## What You'll See

### Mode 5: Robot + Reach (Most Important!)

```
┌─────────────────────────────────────┐
│  Z ↑                                │
│    │   🔴 Arm Base (shoulder)       │
│    │   ╱╲ (at 0.9465m height)      │
│    │  ╱  ╲                          │
│    │ ╱ ··· ╲ Reachable voxels      │
│    │╱  ·····  ╲ (cloud around arm) │
│    ◉─────────→ Y                    │
│   ╱│  Mobile Base                   │
│  ╱ │  (at ground)                   │
│ X  │                                │
│    └─ Robot Model                   │
│       (7-DOF arm on mobile base)    │
└─────────────────────────────────────┘
```

**Expected View:**
1. **Mobile base** sitting on ground (Z=0)
2. **Arm** mounted 0.9465m above ground
3. **Reachability voxels** forming hemisphere around shoulder (red star)
4. **Coordinate axes** at shoulder showing arm frame orientation
5. **End-effector** (green) at home position

## Verification

### Correct Visualization ✅
- Mobile base at ground level (Z=0)
- Arm base marker at Z=0.9465m
- Reachability voxels centered around shoulder (not ground)
- Voxels extend roughly 0.6-0.8m from shoulder in all directions
- Robot model matches voxel positions

### Wrong Visualization ❌
- Map voxels floating disconnected from robot
- Map centered at ground instead of shoulder
- Arm base marker at origin instead of elevated
- Voxels don't align with robot arm reach

## Running the Visualization

```matlab
cd C:\Users\yanbo\wSpace\cinebotRL\matlab

% Build map first (if not done yet)
build_reachability_map()  % 30-60 min

% Visualize
visualize_reachability()
```

**Controls:**
- Button 1-4: Different views (cloud, slices, top-down)
- **Button 5: "Robot + Reach"** ← Use this to verify alignment!
- Sliders: Adjust threshold and slice height
- Mouse: Rotate, pan, zoom

## Technical Details

### Why Transform?

The map is built in **arm base frame** because:
1. Arm workspace is independent of mobile base motion
2. IK solver works in arm frame
3. Easier to pre-compute offline

But for visualization with the full robot:
1. Robot model is in **world frame** (mobile base at origin)
2. Need to show map in same coordinate system
3. Transform: `world_coords = arm_coords + arm_offset`

### Frame Definitions

**World Frame (Visualization):**
- Origin: Mobile base center at ground
- Z=0: Ground plane
- Robot model root: abstract_chassis_link

**Arm Frame (Map Data):**
- Origin: Shoulder (left_arm_base_link)
- Z=0 at shoulder height (0.9465m above ground)
- Map grid centered around [0,0,0]

**Transform:**
```
[x_world]   [x_arm]     [0.16  ]
[y_world] = [y_arm]  +  [0.00  ]
[z_world]   [z_arm]     [0.9465]
```

## Python Integration (No Change)

The Python loader still expects targets in **mobile base frame** and auto-transforms to **arm frame** for queries. The visualization change only affects MATLAB display, not the RL training pipeline.

```python
# Training code unchanged
rmap = ReachabilityMap("matlab/reach_map_arm.mat", 
                       arm_offset=[0.16, 0, 0.9465])
scores = rmap.query_batch(targets_mobile_frame, in_mobile_frame=True)
```

## Summary

✅ **Before:** Map displayed in arm frame (disconnected from robot)  
✅ **After:** Map displayed in world frame (aligned with robot model)

The visualization now correctly shows:
- Mobile base on ground (black marker at Z=0)
- Arm mounted at shoulder height (red marker at Z=0.9465m)
- Reachability workspace around shoulder (voxels transformed to world)
- Full robot model matching the reachable region

Use **Mode 5: "Robot + Reach"** to verify that the arm and reachability map are properly aligned! 🎉
