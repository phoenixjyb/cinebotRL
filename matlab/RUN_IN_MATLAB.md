# Running Reachability Map Build in MATLAB

## ✅ UPDATED: Parallel Computing Enabled!

The script now uses **parallel processing** for much faster computation.

## Quick Start (Easiest Method)

1. **Open MATLAB**
2. **Change to matlab directory:**
   ```matlab
   cd C:\Users\yanbo\wSpace\cinebotRL\matlab
   ```

3. **Run the build launcher:**
   ```matlab
   run_build
   ```

That's it! The script will:
- ✅ Auto-start parallel pool (if available)
- ✅ Show detailed progress every 5%
- ✅ Display ETA and speed
- ✅ Save results when complete
- ✅ Handle errors gracefully

## Expected Performance

### With Parallel (8 cores):
- **Total voxels:** 15,360
- **Processing rate:** ~20-30 voxels/second
- **Build time:** 10-15 minutes ⚡
- **Progress updates:** Every 5% (every ~30 seconds)

### Without Parallel (serial):
- **Processing rate:** ~3-5 voxels/second
- **Build time:** 45-90 minutes
- **Progress updates:** Every 30 seconds

## Progress Tracking Features

The updated script now shows:

```
Computing reachability map...
  Using 8 workers for parallel computation
  Total voxels: 15360
  Expected time: 8-13 minutes (with 8 cores)

  Setting up parallel progress tracking...
  Starting parallel computation...

  [ 5.0%] Voxel   768/15360 | Elapsed:   0.5 min | ETA:   9.5 min | Speed: 25.6 vox/s
  [10.0%] Voxel  1536/15360 | Elapsed:   1.0 min | ETA:   9.0 min | Speed: 25.6 vox/s
  [15.0%] Voxel  2304/15360 | Elapsed:   1.5 min | ETA:   8.5 min | Speed: 25.6 vox/s
  ...
  [95.0%] Voxel 14592/15360 | Elapsed:  10.0 min | ETA:   0.5 min | Speed: 24.3 vox/s
  [100.0%] Voxel 15360/15360 | Elapsed:  10.5 min | ETA:   0.0 min | Speed: 24.4 vox/s

  ✅ All voxels processed (parallel)!
```

## Monitor Progress

While MATLAB is running, you can:

1. **Watch the console** - Updates every 5%
2. **Check file size** - Open another PowerShell:
   ```powershell
   cd C:\Users\yanbo\wSpace\cinebotRL\matlab
   while ($true) {
       dir reach_map_mobile_mm_arm_only.mat | select Name, @{N='Size (MB)';E={$_.Length/1MB}}
       Start-Sleep -Seconds 30
   }
   ```

3. **Check CPU usage** - Task Manager should show ~800% (8 cores at 100%)

## If Build Fails

The script has error handling. If it fails:

1. **Check error message** in MATLAB console
2. **Common issues:**
   - Parallel pool error → Disable parallel: Edit `build_reachability_map.m`, line 78: `USE_PARFOR = false;`
   - URDF not found → Check path in line 23
   - Out of memory → Reduce grid size or use serial mode

3. **Retry with serial mode:**
   ```matlab
   % Edit build_reachability_map.m
   USE_PARFOR = false;  % Line 78
   
   % Then re-run
   run_build
   ```

## After Build Completes

You'll see:
```
╔═══════════════════════════════════════════════════════════╗
║                  BUILD COMPLETED! ✅                      ║
╚═══════════════════════════════════════════════════════════╝

Next steps:
  1. Visualize: visualize_reachability('reach_map_mobile_mm_arm_only.mat', 'mode', 5)
  2. Test Python: python -c "from scripts.reachability_utils import ReachabilityMap; ..."
```

**File created:** `reach_map_mobile_mm_arm_only.mat` (~50-100 MB)

## Visualization

Still in MATLAB:
```matlab
% Best view - robot + reachability in world frame
visualize_reachability('reach_map_mobile_mm_arm_only.mat', 'mode', 5, 'threshold', 0.3)

% Or use quick launcher
quick_viz
```

## Parallel Pool Tips

**If you get parallel pool errors:**

1. **Close existing pools:**
   ```matlab
   delete(gcp('nocreate'))
   ```

2. **Check toolbox:**
   ```matlab
   ver('parallel')
   ```

3. **Manually start pool:**
   ```matlab
   parpool(8)  % Or however many cores you have
   ```

4. **Then run build:**
   ```matlab
   run_build
   ```

## Summary

| Method | Speed | Time | Complexity |
|--------|-------|------|------------|
| **run_build** (parallel) | ⚡⚡⚡ | 10-15 min | Easy (recommended) |
| run_build (serial) | ⚡ | 45-90 min | Easy |
| matlab -batch | ⚡⚡ | Varies | Hard (no interaction) |

**Recommended:** Just run `run_build` in MATLAB - it handles everything!

---

**Updated:** Parallel computing enabled, progress tracking enhanced
**Status:** Ready to run! 🚀
