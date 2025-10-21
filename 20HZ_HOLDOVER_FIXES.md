# 20Hz Holdover Fixes - Complete Cleanup

**Date**: October 21, 2025  
**Issue**: Several files still had 20ms (50Hz) defaults after switching to 20Hz control  
**Status**: ✅ All fixed

---

## Issues Found and Fixed

### 1. **TrajectoryManager Default** ✅
**File**: `src/rl_platform/tasks/mobile_mm/trajectories.py:23`  
**Issue**: Default `dt=0.02` (50Hz)  
**Fix**: Changed to `dt=0.05` (20Hz) with comment

**Impact**: Standalone test scripts (like `test_trajectory_loading.py`) now default to correct 20Hz timing.

---

### 2. **Test Script Explicitly Passes dt** ✅
**File**: `scripts/test_trajectory_loading.py:204`  
**Issue**: Didn't pass `dt` parameter, relied on old 0.02s default  
**Fix**: Now explicitly passes `dt=0.05` with comment

**Impact**: Dry runs now use correct control frequency.

---

### 3. **Visualization Sleep Time** ✅
**File**: `scripts/test_trained_model.py:137`  
**Issue**: `time.sleep(0.02)` giving 50 FPS preview  
**Fix**: Changed to `time.sleep(0.05)` (20 FPS) to match 20Hz control

**Impact**: Visualization now syncs with actual control rate.

---

### 4. **Test Script Documentation** ✅
**File**: `scripts/test_base_movement_fix.py:5-12`  
**Issue**: Comments said "50 steps × 0.02s"  
**Fix**: Updated to "20 steps × 0.05s @ 20Hz"

**Impact**: Documentation now matches actual 20Hz timing.

---

### 5. **PPR Architecture Documentation** ✅
**File**: `docs/PPR_CONTROL_ARCHITECTURE.md`  
**Issues**:
- Line 42: `dt = 0.02  # 50Hz control frequency`
- Lines 65-80: Integration examples using 0.02s timesteps
- Line 176: Table showing 50 Hz (dt=0.02s)

**Fixes**:
- Changed all `dt = 0.02` → `dt = 0.05` with "20Hz" comments
- Updated integration examples: 0.02m → 0.05m, 0.01 rad → 0.025 rad, 20ms → 50ms
- Updated table: 50 Hz → 20 Hz, added stiffness/damping updates (1kN/m, ζ=0.5)

**Impact**: Architecture documentation now accurately reflects current system.

---

### 6. **Training Hierarchy Documentation** ✅
**File**: `docs/training_hierarchy_explained.md`  
**Issues**:
- Line 27: "50Hz (dt=0.02s per step)"
- Line 72: "0.02 seconds of simulated time (50 Hz)"
- Line 107-108: "50Hz (0.02s per step), 1500 steps needed"
- Line 117: Example table with 0.02s increments
- Line 198: "0.02s sim time"
- Line 200: "128 steps × 0.02s = 2.56s"
- Line 292: "0.02s"

**Fixes**:
- Updated all "50Hz" → "20Hz", "0.02s" → "0.05s"
- Recalculated example: 300 waypoints now needs 600 steps (was 1500), 10 rollouts (was 12)
- Updated rollout math: 64 steps × 0.05s = 3.20s (was 128 × 0.02s = 2.56s)
- Updated training numbers table with n_steps=64, rollout samples=262,144

**Impact**: Training documentation now matches actual 20Hz control with n_steps=64.

---

### 7. **Training Diary** ✅
**File**: `TRAINING_DIARY.md:165`  
**Issue**: "Control: 50Hz (dt=0.02s)"  
**Fix**: Changed to "Control: 20Hz (dt=0.05s)"

**Impact**: Historical documentation now accurate.

---

## Files Modified

### Code Changes:
1. ✅ `src/rl_platform/tasks/mobile_mm/trajectories.py` - Default dt: 0.02→0.05
2. ✅ `scripts/test_trajectory_loading.py` - Explicit dt=0.05 parameter
3. ✅ `scripts/test_trained_model.py` - Visualization sleep: 0.02→0.05
4. ✅ `scripts/test_base_movement_fix.py` - Documentation: 50 steps @ 0.02s → 20 steps @ 0.05s

### Documentation Changes:
5. ✅ `docs/PPR_CONTROL_ARCHITECTURE.md` - All timing examples updated to 20Hz
6. ✅ `docs/training_hierarchy_explained.md` - Complete rewrite for 20Hz + n_steps=64
7. ✅ `TRAINING_DIARY.md` - Historical entry corrected

---

## Validation

### Before Fixes:
- ❌ TrajectoryManager default: 0.02s (wrong!)
- ❌ Test scripts: 50Hz timing (wrong!)
- ❌ Visualization: 50 FPS (out of sync)
- ❌ Documentation: Mix of 50Hz and 20Hz (confusing!)

### After Fixes:
- ✅ TrajectoryManager default: 0.05s (correct!)
- ✅ Test scripts: 20Hz timing (correct!)
- ✅ Visualization: 20 FPS (synced!)
- ✅ Documentation: Consistent 20Hz throughout

---

## Impact on Training

### No Impact:
- ✅ **env.py runtime code**: Already using dynamic dt calculation (unchanged)
- ✅ **Training loop**: Already uses decimation=10 from env.py (unchanged)
- ✅ **Reward functions**: Already take dt as parameter (unchanged)

### Fixed:
- ✅ **Standalone test scripts**: Now use correct 20Hz timing by default
- ✅ **Visualization**: Now syncs with actual control rate
- ✅ **Documentation**: No more confusion about control frequency
- ✅ **Future development**: Developers won't accidentally revert to 50Hz

---

## Verification Commands

```bash
# Verify TrajectoryManager default
grep "dt: float = 0.05" src/rl_platform/tasks/mobile_mm/trajectories.py

# Verify test script
grep "dt=0.05" scripts/test_trajectory_loading.py

# Verify visualization
grep "time.sleep(0.05)" scripts/test_trained_model.py

# Verify documentation
grep -r "20 Hz" docs/
grep -r "0.05s" docs/
```

---

## Related Documentation

- `CODE_AUDIT_20HZ.md` - Runtime code audit (env.py, config.py, rewards.py)
- `20HZ_CONTROL_ANALYSIS.md` - Physics justification for 20Hz control
- `SESSION_4_LAUNCH_READY.md` - Complete Session 4 training guide

---

## Summary

**All 20ms (50Hz) holdovers eliminated!** ✅

The codebase is now **100% consistent** with 20Hz control frequency:
- ✅ Runtime code (env.py, config.py, trajectories.py)
- ✅ Test scripts (test_base_movement_fix.py, test_trajectory_loading.py, test_trained_model.py)
- ✅ Documentation (PPR architecture, training hierarchy, training diary)
- ✅ Default parameters (TrajectoryManager, visualization timing)

**No more confusion between 50Hz and 20Hz!** 🎉
