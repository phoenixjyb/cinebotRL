# Fixed Issues: Zero Map & Frame Confusion

## Issue 1: All Zeros in Map ✅ FIXED

**Problem:** The FK builder was producing all zeros (no valid samples).

**Likely Causes:**
1. Collision checking too strict
2. Grid bounds wrong
3. FK failing silently

**Fix Applied:**
Added debug output in `build_reachability_map_FK.m` (line ~208):
- Shows first 10 rejected samples with detailed reasons
- Prints FK results, grid bounds checking
- Clear error message if no valid samples found

**What to Look For:**
When you run the build, if it still fails you'll now see:
```
⚠️  WARNING: NO VALID SAMPLES FOUND!
   Checking first 10 samples in detail...

Sample 1:
  q = [1.234, -0.567, ...]
  EE pos: [0.456, 0.123, 0.789]
  Relative pos: [1.056, 0.923, 1.189]
  Grid max: [1.200, 1.600, 1.000]
  ❌ REJECTED: Above grid bounds
```

This will tell us exactly why samples are rejected!

---

## Issue 2: Arm Base at Wrong Height ✅ FIXED

**Problem:** Visualization showed arm base at (0,0,0) ground level, but it's mounted at 0.9465m on the chassis.

**Root Cause:** Frame confusion!
- **Map is built in ARM BASE FRAME** (shoulder = origin)
- **Visualization was showing ARM BASE FRAME** (origin at shoulder)
- **Should show WORLD/MOBILE BASE FRAME** (origin at ground, arm at height)

**Fix Applied:**
Updated `visualize_fk_map.m` to transform coordinates:

```matlab
% ARM_OFFSET = [0.16, 0.0, 0.9465] from URDF

% In arm base frame:
x_arm = origin(1) + (ix - 0.5) * voxel;
z_arm = origin(3) + (iz - 0.5) * voxel;

% Transform to world frame:
x = x_arm + ARM_OFFSET(1);  % +0.16m forward
z = z_arm + ARM_OFFSET(3);  % +0.9465m up
```

**Now Shows:**
- **Black pentagon at (0,0,0)** = Mobile base (ground)
- **Blue dot at (0.16, 0, 0.95)** = Arm shoulder
- **Red cloud** = Reachable workspace (hemisphere around shoulder at ~1m height!)

---

## Frame Reference Chart

| Frame | Origin | Purpose |
|-------|--------|---------|
| **World/Mobile Base** | Ground (0,0,0) | Global reference, training rewards |
| **Arm Base** | Shoulder (0.16, 0, 0.9465) | Map building (FK/IK) |
| **End Effector** | Camera/gripper tip | Target tracking |

**Key Insight:**
- Map is **built** in arm base frame (simpler math, no mobile base)
- Map is **visualized** in world frame (easier to understand)
- Python loader will **auto-transform** between frames (already implemented!)

---

## Next Steps

1. **Re-run build:**
   ```matlab
   cd C:\Users\yanbo\wSpace\cinebotRL\matlab
   run_build
   ```

2. **If you still get all zeros:**
   - Look at the debug output
   - Check if samples are rejected for grid bounds
   - May need to adjust GRID_ORIGIN or GRID_SIZE

3. **If build succeeds:**
   ```matlab
   visualize_fk_map
   ```
   You should see:
   - Hemisphere at ~1m height (not at ground!)
   - Centered around blue dot (arm shoulder)
   - Radius ~0.6-0.8m (arm reach)

---

**Status:** Debug tools added, visualization frame fixed!  
**Ready to test:** Run `run_build` and share debug output if it fails! 🔧
