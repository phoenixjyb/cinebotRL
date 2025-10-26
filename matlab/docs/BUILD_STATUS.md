# Reachability Map Build - Quick Status

## Build Status: 🔄 RUNNING (Serial Mode)

**Started:** Just now  
**Expected Duration:** 45-90 minutes (serial processing, 15,360 voxels × 24 orientations)  
**Progress Indicator:** Watch for `.mat` file creation in `matlab/` directory

## What's Happening Right Now

```
MATLAB is computing reachability for each voxel:
- Grid: 24×32×20 = 15,360 voxels
- Per voxel: 24 orientations × 8 IK attempts = 192 computations
- Total: ~2.9 million IK + collision checks
- Serial processing: ~200-300 voxels/minute
```

## Monitor Progress

### Option 1: Watch File Size
```powershell
# In PowerShell, run this every few minutes:
cd C:\Users\yanbo\wSpace\cinebotRL\matlab
dir reach_map_mobile_mm_arm_only.mat
```

File will appear when first voxels complete, then grow to 50-100 MB.

### Option 2: Python Monitor
```powershell
python matlab/monitor_build.py
```

Checks every 30 seconds and estimates progress.

### Option 3: MATLAB Terminal
Check the terminal where `matlab -batch` is running for progress messages.

## Expected Timeline

| Time | Expected Progress |
|------|------------------|
| 0-5 min | MATLAB startup, URDF loading, grid setup |
| 5-10 min | First voxels processed, .mat file appears |
| 10-30 min | Steady progress, file growing to ~20 MB |
| 30-60 min | File growing to ~50 MB (50% complete) |
| 60-90 min | Final voxels, file reaches 50-100 MB |

## Why Serial Processing?

Changed from parallel (`USE_PARFOR = true`) to serial (`USE_PARFOR = false`) because:
- First attempt hit parallel pool error
- Serial is more stable (no worker communication issues)
- Slower but reliable (45-90 min instead of 30-60 min)
- Can re-enable parallel later once serial works

## What Happens After Build Completes?

1. **MATLAB exits** with message showing final statistics
2. **File created:** `reach_map_mobile_mm_arm_only.mat` (50-100 MB)
3. **Verify:** Load in MATLAB to check contents
4. **Visualize:** Run visualization script (5 interactive modes)
5. **Test Python:** Load with `ReachabilityMap` class
6. **Integrate:** Add to Session 8 training

## If Build Fails

**Symptoms:**
- MATLAB exits with error before file created
- File created but very small (<10 MB)
- Process hangs for >2 hours

**Solutions:**
1. Check MATLAB terminal output for specific error
2. Verify Robotics System Toolbox installed: `matlab -batch "ver('robotics')"`
3. Test URDF loading separately
4. Reduce grid size for faster testing:
   ```matlab
   GRID_SIZE = [0.8, 1.0, 0.6];  % Smaller workspace
   VOXEL = [0.08, 0.08, 0.08];    % Coarser resolution
   ```

## Current Configuration

```matlab
% Grid (arm base frame)
GRID_ORIGIN = [-0.6, -0.8, -0.4]  % min [x,y,z]
GRID_SIZE   = [1.2, 1.6, 1.0]      % size [dx,dy,dz]
VOXEL       = [0.05, 0.05, 0.05]   % 5cm resolution

% Orientations
N_ORIENT = 24                       % Camera-like poses
ORIENT_CONE_DEG = 90                % 90° cone (hemisphere)

% IK
IK_ATTEMPTS = 8                     % Seeds per orientation
IK_POS_TOL = 3mm
IK_ORI_TOL = 5°

% Collision
DO_SELF_COLLISION = true            % Essential!

% Processing
USE_PARFOR = false                  % Serial (stable)
```

## Estimated Output

Based on similar robots:

```
Expected reachability map:
- Total voxels: 15,360
- Reachable voxels: ~7,000-9,000 (45-60%)
- File size: 50-100 MB
- Load time (Python): ~2-5 seconds
- Query time (batch): ~1-5 ms per 4096 targets
```

## Next Steps After Build

See `README_BUILD_VISUALIZE.md` for complete workflow.

Quick version:
1. ✅ Build completes → `.mat` file created
2. 📊 Visualize in MATLAB → Mode 5 (robot + reach)
3. 🐍 Test Python loader → Quick query
4. 🚀 Integrate Session 8 → Add to env.py
5. 📈 Compare results → Session 7 vs Session 8

---

**Status last updated:** Just now  
**Expected completion:** In 45-90 minutes  
**Check again in:** 10-15 minutes to see if file appeared
