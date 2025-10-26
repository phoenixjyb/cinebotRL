# Grid Expansion for FK Reachability Map

## Coordinate Frame (CRITICAL!)

**Grid origin is RELATIVE TO the ARM BASE LINK frame**, NOT mobile base (chassis) frame!

From URDF (line 234):
```xml
<joint name="arm_mount_joint" type="fixed">
    <parent link="abstract_chassis_link"/>
    <child link="left_arm_base_link"/>
    <origin xyz="0.15995 0.0 0.9465" rpy="0 0 -1.5708"/>
</joint>
```

Coordinate transform chain:
```
Mobile Base (chassis):        [0, 0, 0] in world frame
↓ (arm_mount_joint offset)
Arm Base Link (shoulder):     [0.16, 0, 0.9465] in world frame
  ↓ (grid defined RELATIVE to this frame!)
  Grid Origin:                [-0.8, -1.0, -0.6] in arm base frame
  Grid Corner in World:       [0.16-0.8, 0-1.0, 0.9465-0.6] = [-0.64, -1.0, 0.3465]
```

**IMPORTANT**: Grid origin `[-0.8, -1.0, -0.6]` means:
- The workspace extends from -0.8m to +0.8m in arm's local X
- From -1.0m to +1.0m in arm's local Y  
- From -0.6m to +0.8m in arm's local Z
- Center of grid is roughly at arm base link origin (0, 0, 0)

This means:
- FK calculations use arm base link as reference frame
- Grid positions are relative to arm shoulder, not ground
- When visualizing, we add ARM_OFFSET to transform to world frame

## Grid Size Comparison

### Original Grid (Conservative)
```
Origin: [-0.60, -0.80, -0.40] m
Size:   [ 1.20,  1.60,  1.00] m
→ End:  [ 0.60,  0.80,  0.60] m

Voxels: 24 × 32 × 20 = 15,360
Coverage: ~0.6-0.8m reach from shoulder
```

### Expanded Grid (+40cm each axis)
```
Origin: [-0.80, -1.00, -0.60] m
Size:   [ 1.60,  2.00,  1.40] m
→ End:  [ 0.80,  1.00,  0.80] m

Voxels: 32 × 40 × 28 = 35,840
Coverage: ~0.8-1.0m reach from shoulder
```

**Expansion details:**
- X-axis: -0.6→0.6 to **-0.8→0.8** (+20cm each side)
- Y-axis: -0.8→0.8 to **-1.0→1.0** (+20cm each side)
- Z-axis: -0.4→0.6 to **-0.6→0.8** (+20cm each side)

## Why Expand?

### Potential Benefits:
1. **Arm might reach farther** than initial 0.6-0.8m estimate
2. **Check edge cases** near joint limits
3. **Capture extreme poses** that might be useful for specific tasks
4. **Verify workspace boundary** empirically (where FK stops finding valid configs)

### Trade-offs:
- **More voxels**: 15,360 → 35,840 (2.3× increase)
- **Same samples**: Still 100K FK evaluations
- **Slightly longer**: ~1-2 min → ~1.5-2.5 min (more voxels to check)
- **Larger file**: ~50-100 MB → ~120-240 MB (2.3× more data)

## Expected Outcome

With expanded grid, we'll see:

### Scenario 1: Arm Reaches Farther (Best Case)
- Reachable voxels increase from ~8,000 to ~15,000-20,000
- Coverage extends to 0.9-1.0m from shoulder
- More poses near joint limits discovered
- **Action**: Keep expanded grid, better coverage!

### Scenario 2: Same Reach, Empty Edges (Most Likely)
- Reachable voxels stay ~8,000-10,000
- Outer 20cm shell mostly empty (unreachable)
- Coverage still ~0.6-0.8m from shoulder
- **Action**: Can revert to original grid to save space, or keep for safety margin

### Scenario 3: Collisions Prevent Expansion
- Reachable voxels ~8,000-9,000 (minimal increase)
- Self-collision blocks extreme poses
- Coverage ~0.6-0.8m (same as before)
- **Action**: Revert to original grid, expansion not useful

## How to Check After Build

```matlab
% Load map
load('reach_map_mobile_mm_arm_only.mat')

% Get reachable voxel positions in arm frame
[ix, iy, iz] = ind2sub(config.grid_dims, find(reachScore > 0));
x_arm = config.grid_origin(1) + (ix - 0.5) * config.voxel_size;
y_arm = config.grid_origin(2) + (iy - 0.5) * config.voxel_size;
z_arm = config.grid_origin(3) + (iz - 0.5) * config.voxel_size;

% Check max reach distance from arm base
dist = sqrt(x_arm.^2 + y_arm.^2 + z_arm.^2);
fprintf('Arm reach statistics (from shoulder):\n');
fprintf('  Min: %.3f m\n', min(dist));
fprintf('  Mean: %.3f m\n', mean(dist));
fprintf('  Max: %.3f m\n', max(dist));

% Check edge utilization (how much of expanded space is used)
fprintf('\nEdge utilization:\n');
fprintf('  X-range: [%.2f, %.2f] (grid: [%.2f, %.2f])\n', ...
        min(x_arm), max(x_arm), config.grid_origin(1), ...
        config.grid_origin(1) + config.grid_size(1));
fprintf('  Y-range: [%.2f, %.2f] (grid: [%.2f, %.2f])\n', ...
        min(y_arm), max(y_arm), config.grid_origin(2), ...
        config.grid_origin(2) + config.grid_size(2));
fprintf('  Z-range: [%.2f, %.2f] (grid: [%.2f, %.2f])\n', ...
        min(z_arm), max(z_arm), config.grid_origin(3), ...
        config.grid_origin(3) + config.grid_size(3));
```

## Build Time Impact

| Grid Size | Voxels | FK Samples | Expected Time | File Size |
|-----------|--------|------------|---------------|-----------|
| **Original** | 15,360 | 100K | 1-2 min | 50-100 MB |
| **Expanded** | 35,840 | 100K | 1.5-2.5 min | 120-240 MB |

The extra time comes from:
- More voxel IDs to compute (35,840 vs 15,360)
- More array indexing operations
- Larger arrays to save to disk

But FK computation time is **same** (still 100K samples), so impact is minimal!

## Decision Point

After this build completes:
1. **Check reach statistics** (max distance from shoulder)
2. **Check edge utilization** (% of grid volume used)
3. **If edges are empty**: Can revert to original grid in future builds
4. **If edges are used**: Keep expanded grid, arm reaches farther than expected!

## Summary

✅ **Grid origin**: Fixed to arm base link frame (shoulder)  
✅ **Expansion**: +40cm (20cm each side) on all axes  
✅ **Purpose**: Test if arm reaches farther than initial estimate  
✅ **Cost**: 2.3× more voxels, ~30% longer build, 2× larger file  
✅ **Benefit**: Empirically determine true workspace boundary  

Let's build and see what the arm can actually reach! 🎯
