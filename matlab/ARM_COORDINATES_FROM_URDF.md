# ARM COORDINATE FRAME - VERIFIED FROM URDF

## Arm Mount Joint (Line 231-235 in URDF)

```xml
<joint name="arm_mount_joint" type="fixed">
    <parent link="abstract_chassis_link"/>
    <child link="left_arm_base_link"/>
    <origin xyz="0.15995 0.0 0.9465" rpy="0 0 -1.5708"/>
    <axis xyz="1 0 0"/>
</joint>
```

## What This Means

### Position
- **X = 0.15995 m** (0.16m forward from chassis center)
- **Y = 0.0 m** (centered on chassis)
- **Z = 0.9465 m** (0.95m above ground - shoulder height!)

### Orientation
- **Roll = 0 rad** (no tilt)
- **Pitch = 0 rad** (no tilt)
- **Yaw = -1.5708 rad** (-90° = -π/2, rotated counterclockwise around Z)

This rotation means the arm's +X axis points in the chassis's **+Y direction** (to the left when viewed from behind).

## Grid Origin Clarification

**Grid origin `[-0.8, -1.0, -0.6]` is NOT the position of the arm base link!**

Instead, it's the **minimum corner** of the workspace **expressed in the arm base link's coordinate frame**.

### Transform Chain

1. **Mobile base frame** (world): Origin at [0, 0, 0]
2. **Arm base link frame**: Origin at [0.16, 0, 0.9465] in world
3. **Grid origin**: [-0.8, -1.0, -0.6] **relative to arm base link**
4. **Grid origin in world**: [0.16-0.8, 0-1.0, 0.9465-0.6] = **[-0.64, -1.0, 0.3465]**

### Grid Bounds in Arm Frame

```
Origin: [-0.8, -1.0, -0.6]  (min corner)
Size:   [ 1.6,  2.0,  1.4]  (extent)
End:    [ 0.8,  1.0,  0.8]  (max corner)
```

This creates a workspace that:
- Extends **±0.8m** in arm's local X (forward/backward)
- Extends **±1.0m** in arm's local Y (left/right)
- Extends **-0.6m to +0.8m** in arm's local Z (down/up)
- Is roughly **centered** around the arm shoulder (0, 0, 0 in arm frame)

### Grid Bounds in World Frame

```
Origin in world: [-0.64, -1.0, 0.3465]
End in world:    [ 0.96,  1.0, 1.7465]
Center in world: [ 0.16,  0.0, 1.0465]  (slightly above shoulder)
```

## Why This Matters for FK

When we do FK:
1. Build joint config `q_full` with arm joints
2. Call `getTransform(robot, q_full, "left_gripper_link", "left_arm_base_link")`
3. Get EE position **in arm base link frame**: `[x, y, z]`
4. Check if `[x, y, z]` is inside grid bounds: `[-0.8, -1.0, -0.6]` to `[0.8, 1.0, 0.8]`
5. Convert to voxel index if inside

When we visualize:
1. Load reachable voxel positions in arm frame
2. Add ARM_OFFSET = [0.16, 0, 0.9465] to transform to world frame
3. Plot with mobile base at ground and arm at correct height

## Common Confusion

❌ **WRONG**: "Grid origin is at the arm base link"
- This would mean workspace starts AT the shoulder

✅ **CORRECT**: "Grid origin is -0.8m in X, -1.0m in Y, -0.6m in Z **relative to** the arm base link"
- This means workspace extends 0.8m **backward** from shoulder, 1.0m **to the right**, and 0.6m **down**

## Visual Summary

```
            Z (up)
            ↑
            |
            |   [Grid in arm frame]
     0.8 ───┼──────────────
            │   ╱╱╱╱╱╱╱╱  │  Reachable
            │  ╱╱╱╱╱╱╱╱   │  workspace
     0.0 ───●──────────────  ← Arm base link origin
            │              │  (shoulder)
   -0.6 ────┼──────────────
            │
        -1.0    0.0   1.0  → Y (left/right)
               -0.8  0.8   → X (forward/back)

    [World frame]
    Ground = Z: 0.0
    Shoulder = Z: 0.9465
    Grid bottom = Z: 0.3465  (shoulder - 0.6)
    Grid top = Z: 1.7465     (shoulder + 0.8)
```

The arm can reach:
- Down to 0.35m above ground (not quite touching ground)
- Up to 1.75m above ground (full extension upward)
- ±0.8m forward/back and ±1.0m left/right from shoulder

This is why we see a **hemisphere** at shoulder height, not ground level!
