# Visualization with Robot URDF

## Overview

The `visualize_fk_map.m` function now loads and displays the robot URDF alongside the reachability map, making it much easier to understand the workspace geometry.

## What You'll See

### Left Plot: Binary Reachability
- **Red point cloud**: Reachable voxels (positions where EE can reach)
- **Robot mesh**: Mobile manipulator at home configuration
  - Mobile base (black pentagon) at ground level (0, 0, 0)
  - Arm mounted at height 0.9465m
- **Blue dot**: Arm shoulder (base of arm)
- **Grid box**: Workspace boundary

### Right Plot: Manipulability Index
- **Color gradient**: Blue (low) → Red (high)
  - High values = joint config far from limits (good dexterity)
  - Low values = near joint limits (poor dexterity)
- **Robot mesh**: Same home configuration
- **Same markers**: Mobile base + arm shoulder

## Key Visual Insights

### Hemisphere Shape
The reachable workspace should form a **hemisphere** around the arm shoulder:
- Center: (0.16, 0, 0.9465) in world frame
- Radius: ~0.6-0.8m (arm reach)
- NOT at ground level (that would be wrong!)

### Robot Geometry Context
Seeing the robot helps understand:
1. **Why certain regions are reachable**: Arm can extend there without collision
2. **Why gaps exist**: Collisions prevent reaching those voxels
3. **Workspace orientation**: Relative to mobile base and arm mounting

## Usage

```matlab
% After successful build:
visualize_fk_map('reach_map_mobile_mm_arm_only.mat')

% Or use quick launcher:
quick_viz
```

## Robot Display Details

- **Model source**: Loaded from URDF path stored in `config.urdf_path`
- **Configuration**: Home configuration (all joints at default positions)
- **Rendering**:
  - `Visuals: on` - Shows mesh geometry
  - `Collisions: off` - Hides collision spheres (cleaner view)
  - `Frames: off` - Hides coordinate frame axes (less clutter)

## Troubleshooting

### If robot doesn't show:
- Check URDF path in map file: `load('reach_map...'); config.urdf_path`
- Ensure URDF file exists at that path
- Function will warn but continue without robot if URDF missing

### If robot looks wrong:
- Check home configuration: `robot = importrobot(...); homeConfiguration(robot)`
- Verify mobile base is at ground (z=0)
- Verify arm base is at correct height (z=0.9465m)

## Benefits Over Previous Version

**Without robot URDF:**
- ❌ Hard to judge if workspace looks correct
- ❌ No context for gap/hole locations
- ❌ Difficult to explain to others

**With robot URDF:**
- ✅ Immediately see if workspace is at correct height
- ✅ Understand collision-induced gaps
- ✅ Better intuition for mobile base + arm coordination
- ✅ Publication-quality visualizations

## Next Steps

After visualizing:
1. **Verify geometry**: Robot at correct pose, workspace at correct height
2. **Check coverage**: ~40-60% reachable (8,000-9,000 voxels expected)
3. **Inspect gaps**: Should correlate with arm self-collision regions
4. **Test Python integration**: Confirm loader works with trajectory queries
