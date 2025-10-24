# Visualizing the FK-Generated Map

## The Issue

The old `visualize_reachability()` function expected a different data format (with `map` structure). The FK version outputs simpler arrays: `reachScore`, `manipMax`, `qExample`.

## Solution: New Visualizer

Created `visualize_fk_map.m` that works with FK output format.

## How to Visualize

### Option 1: Quick launcher (recommended)
```matlab
cd C:\Users\yanbo\wSpace\cinebotRL\matlab
quick_viz
```

### Option 2: Direct call
```matlab
cd C:\Users\yanbo\wSpace\cinebotRL\matlab
visualize_fk_map('reach_map_mobile_mm_arm_only.mat')
```

### Option 3: Default (no arguments)
```matlab
cd C:\Users\yanbo\wSpace\cinebotRL\matlab
visualize_fk_map
```

## What You'll See

**Two subplots:**
1. **Left - Binary Reachability:**
   - Red points = reachable voxels
   - Black dot at origin = arm base (shoulder)
   - Dashed box = workspace bounds

2. **Right - Manipulability Heatmap:**
   - Color-coded by joint configuration quality
   - Blue = low manipulability (near joint limits)
   - Red = high manipulability (centered configs)
   - Shows best regions for dexterous manipulation

## Frame Convention

- **Origin (0,0,0)** = Arm base (shoulder joint)
- **Grid origin** = [-0.6, -0.8, -0.4] relative to arm base
- **Workspace** = 1.2 × 1.6 × 1.0 m³ around shoulder

This is the **arm base frame**, not the mobile base frame!

## Expected Result

You should see a hemisphere of reachable points centered around the origin, extending roughly 0.6-0.8m in all directions (arm reach).

---

**Ready to visualize!** Just run `quick_viz` after the build completes! 🚀
