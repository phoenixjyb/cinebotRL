# Quick Test of FK Builder

## Run in MATLAB:

```matlab
cd C:\Users\yanbo\wSpace\cinebotRL\matlab
run_build
```

## Expected Output:

```
========================================
   Reachability Map Builder (FK)
========================================

✓ Parallel Computing Toolbox detected
  Will use parallel processing for faster build

Starting FK-based build (100K samples)...
Expected time: 1-2 minutes
════════════════════════════════════════

=== Building Reachability Map (FK-based) ===

Loading robot from: ..\assets_own\mobile_manipulator_PPR_base_corrected.urdf
  Total bodies: 10
  Arm joints: 6
  DOF (arm only): 6
  Joint limits:
    Joint 1: [-2.967, 2.967] rad
    Joint 2: [-2.094, 2.094] rad
    Joint 3: [-2.967, 2.967] rad
    Joint 4: [-2.094, 2.094] rad
    Joint 5: [-2.967, 2.967] rad
    Joint 6: [-2.094, 2.094] rad

Grid setup:
  Origin: [-0.60, -0.80, -0.40] m
  Size: [1.20, 1.60, 1.00] m
  Voxel size: 0.050 m
  Grid dimensions: 24 × 32 × 20 = 15360 voxels

  Using 8 workers

=== Sampling 100000 random configurations ===
Computing FK and checking collisions...
  Processed 5000/100000 samples (5.0%)
  Processed 10000/100000 samples (10.0%)
  ...
```

## If It Works:

You should see:
- ✅ Robot loads successfully
- ✅ 6 arm joints detected
- ✅ Parallel pool starts (8 workers)
- ✅ Progress updates every 5%
- ✅ Completes in ~1-2 minutes

## If It Fails:

Share the error message and I'll fix it!

---

**Status:** Fixed robot.Bodies indexing issue  
**Ready:** Yes! Try it now! 🚀
