# Base Movement Validation Script

**File**: `scripts/test_base_movement_fix.py`  
**Status**: ✅ Ready to run  
**Created**: After commit 7f6a94d (base movement fix)

## Purpose

Validates that the base movement control fix (commit 7f6a94d) works correctly by:

1. **Measuring root state** (NOT joint state) - `root_pos_w`, `root_lin_vel_w`
2. **Commanding forward motion** - 0.5 m/s for 10 seconds
3. **Verifying displacement** - ~5.0m expected
4. **Checking PPR joints** - Should stay near zero (< 1cm drift)

## Key Differences from Before

### ❌ **Before Fix (would measure incorrectly)**:
```python
# WRONG: Measures joint offsets (stay at zero with new fix!)
displacement = robot.data.joint_pos[:, 0:3] - initial_joint_pos
# Result: Would report 0m displacement even when base moves!
```

### ✅ **After Fix (measures correctly)**:
```python
# CORRECT: Measures world position from root state
displacement = robot.data.root_pos_w - initial_root_pos
# Result: Reports actual world displacement (should be ~5m)
```

## Run Command

```powershell
I:\isaaclab\isaaclab.bat -p scripts/test_base_movement_fix.py --headless
```

## Expected Output

If fix is working correctly:

```
================================================================================
RESULTS
================================================================================
Base displacement (from root_pos_w):
  X: 5.000 m          ← Should be ~5.0m (0.5 m/s * 10s)
  Y: 0.000 m          ← Should be near 0 (no lateral drift)
  Z: 0.000 m          ← Should be near 0 (no vertical drift)
Final velocity (from root_lin_vel_w):
  X: 0.500 m/s        ← Should be ~0.5 m/s (commanded velocity)
  Y: 0.000 m/s        ← Should be near 0
Max PPR joint drift: 0.00100 m  ← Should be < 0.01m (1cm)
================================================================================

VALIDATION:
  1. Forward displacement: 5.000m (expected 5.0±0.5m) - ✅ PASS
  2. Forward velocity: 0.500m/s (expected 0.5±0.1m/s) - ✅ PASS
  3. PPR joint drift: 0.00100m (threshold 0.01m) - ✅ PASS
  4. Lateral drift: 0.000m (threshold 0.5m) - ✅ PASS
  5. Vertical drift: 0.000m (threshold 0.1m) - ✅ PASS

================================================================================
✅ ALL CHECKS PASSED - Base movement fix is working correctly!
================================================================================
```

## 5 Validation Checks

1. **Forward displacement**: 5.0m ± 0.5m  
   - Tests: `robot.data.root_pos_w` actually updates  
   - Before: Would show ~0.002m (frozen root)  
   - After: Should show ~5.0m (moving base)

2. **Forward velocity**: 0.5 m/s ± 0.1 m/s  
   - Tests: `robot.data.root_lin_vel_w` matches command  
   - Verifies velocity integration works

3. **PPR joint drift**: < 0.01m (1cm)  
   - Tests: PPR joints stay at zero offset  
   - Before: Would accumulate to -6.3m  
   - After: Should stay near 0m

4. **Lateral drift**: < 0.5m  
   - Tests: Base moves straight (no Y-axis drift)  
   - Verifies frame transforms correct

5. **Vertical drift**: < 0.1m  
   - Tests: Base stays at ground level  
   - Verifies Z-axis stability

## What It Proves

### ✅ **If all checks pass**:
- Base movement fix is working correctly
- Root state updates via velocity integration
- PPR joints correctly zeroed (no dual control)
- Frame transformations correct
- **Ready for Session 7c launch**

### ❌ **If checks fail**:
- Check 1 fails → Root position not updating (velocity integration broken)
- Check 2 fails → Velocity commands not being applied correctly
- Check 3 fails → PPR joints still accumulating (dual control still present)
- Check 4/5 fail → Frame transformation issues

## Next Steps After Validation

### If test passes:
1. ✅ Mark todo item 4 complete
2. 🚀 Launch Session 7c with confidence:
   ```powershell
   .\scripts\launch_training_windows.ps1 -Task MobileMMTrackEE-v0 -NumEnvs 8192 -Headless
   ```
3. 📊 Monitor first 1000 steps:
   - Base displacement should be >0.1m (vs 0.002m in Session 7b)
   - Mean tracking error should decrease
   - Base alignment should be positive

### If test fails:
1. 🔍 Analyze which checks failed
2. 🐛 Debug specific issue:
   - Check 1/2 fail → Review velocity integration in `_pre_physics_step`
   - Check 3 fails → Review PPR joint zeroing in `_reset_idx`
   - Check 4/5 fail → Review frame transformations (quat_to_yaw)
3. 🔧 Apply additional fixes
4. ♻️ Re-run validation

## Technical Details

### Test Configuration:
- **Num envs**: 4 (lightweight test)
- **Duration**: 10 seconds
- **Command**: Constant 0.5 m/s forward, 0.0 rad/s rotation
- **Timestep**: ~0.1s (physics_dt * decimation)
- **Total steps**: ~100 steps

### Measurements (from robot.data):
```python
# Root state (world coordinates)
root_pos_w       # [N, 3] - XYZ position in world frame
root_quat_w      # [N, 4] - Orientation quaternion (w, x, y, z)
root_lin_vel_w   # [N, 3] - Linear velocity in world frame

# Joint state (for monitoring only)
joint_pos[:, 0:3]  # PPR joint offsets (should stay ~0)
```

### Key Functions Used:
- `quat_to_yaw()` - Same helper as in env.py
- Matches exact implementation from base movement fix
- Ensures consistent orientation extraction

## Related Files

- **Fix implementation**: `src/rl_platform/tasks/mobile_mm/env.py` (commit 7f6a94d)
- **Analysis docs**: 
  - `docs/tracking/frame_transformation_analysis.md`
  - `docs/tracking/findings_verification.md`
- **Todo tracking**: Updated in conversation
