# Session 8i Training Failure Analysis & Recovery Plan

## Training Failure Summary

**Actual Training**:
- Started: 2025-11-07 16:42
- Stopped: 2025-11-07 18:30 (Emergency Pause)
- Steps Completed: 29.36M / 40M (73%)
- Trigger: AutoPauseCallback detected instability

**Metrics @ 12.6M (Healthy)**:
- KL Divergence: 0.0165 ✅
- Explained Variance: 0.854 ✅
- FPS: 4945

**Metrics @ 29.36M (Collapsed)**:
- KL or Variance crossed threshold
- Emergency checkpoint saved

---

## Root Cause Analysis

### Suspected Issues

#### 1. **Distance-Gated Weight Jump Too Aggressive** 🔴 HIGH PRIORITY
```python
# Current config:
orientation_tracking_far: 4.0    # >0.7m
orientation_tracking_close: 30.0  # <0.7m
# Jump ratio: 7.5x (30 / 4 = 7.5)
```

**Problem**: 
- Robot crossing 0.7m threshold experiences 7.5x weight change
- May cause policy instability at boundary
- Gradient discontinuity leads to learning oscillation

**Fix**: Reduce jump ratio to 3-4x
```python
orientation_tracking_far: 8.0    # >0.7m  (doubled from 4)
orientation_tracking_close: 30.0  # <0.7m (unchanged)
# New jump ratio: 3.75x (smoother)
```

#### 2. **New Reward Components Not Validated** 🟡 MEDIUM PRIORITY
```python
orientation_progress_bonus: 2.0    # NEW in Session 8i
angular_velocity_penalty: 1.0      # NEW in Session 8i
```

**Problem**:
- These components added on top of `orientation_tracking`
- May create conflicting gradients
- No A/B testing to isolate their effect

**Fix Option A** (Conservative): Disable new components first
```python
orientation_progress_bonus: 0.0   # DISABLED for testing
angular_velocity_penalty: 0.0      # DISABLED for testing
```

**Fix Option B** (Moderate): Reduce weights by 50%
```python
orientation_progress_bonus: 1.0    # Reduced from 2.0
angular_velocity_penalty: 0.5      # Reduced from 1.0
```

#### 3. **Auto-Pause Threshold Too Sensitive** 🟢 LOW PRIORITY
```python
variance_threshold: -0.3  # Triggers pause if variance < -0.3
```

**Analysis**: 
- Auto-pause worked as designed - prevented catastrophic collapse
- Threshold seems reasonable (-0.3 allows some negative variance)
- This is a *feature*, not a bug

**Action**: Keep as-is

---

## Recovery Strategy

### Option 1: Conservative Restart ⭐ RECOMMENDED

**Goal**: Validate distance-gated approach WITHOUT new components

**Changes**:
1. Smooth out weight jump: `orientation_tracking_far: 8.0` (was 4.0)
2. Disable new components:
   ```python
   orientation_progress_bonus: 0.0
   angular_velocity_penalty: 0.0
   ```
3. Start fresh 40M run

**Rationale**:
- Isolate distance-gating as the ONLY variable
- If 40M passes → distance-gating works
- If 40M fails → distance-gating itself is flawed
- Can re-enable new components in Session 8i.1 after validation

**Time Cost**: 2.5-3 hours (40M steps @ 4945 fps)

---

### Option 2: Moderate Restart

**Goal**: Keep all features, reduce aggressiveness

**Changes**:
1. Smooth weight jump: `orientation_tracking_far: 8.0`
2. Reduce new components by 50%:
   ```python
   orientation_progress_bonus: 1.0   # was 2.0
   angular_velocity_penalty: 0.5     # was 1.0
   ```
3. Start fresh 40M run

**Rationale**:
- Less conservative than Option 1
- Still testing distance-gating with less aggressive new components
- Higher risk of another collapse

**Time Cost**: 2.5-3 hours (40M steps)

---

### Option 3: Rollback & Continue (❌ NOT RECOMMENDED)

**Goal**: Continue from pre-collapse checkpoint

**Actions**:
1. Find last stable checkpoint (variance > 0.3)
2. Continue from there to 40M

**Why NOT recommended**:
- We don't have TensorBoard data to identify exact collapse point
- Risk inheriting degraded policy state
- Better to start clean with fixes

---

## Recommended Action Plan

### Phase 1: Conservative Validation (NOW)

1. **Modify config.py**:
   ```python
   # Distance-gated weights (smoother transition)
   orientation_tracking_far: 8.0     # was 4.0 (reduced jump)
   orientation_tracking_close: 30.0  # unchanged
   
   # Disable new components for A/B testing
   orientation_progress_bonus: 0.0   # was 2.0 (DISABLED)
   angular_velocity_penalty: 0.0     # was 1.0 (DISABLED)
   ```

2. **Launch clean 40M run**:
   ```powershell
   .\scripts\launch_session_8i.ps1 -Phase short
   ```

3. **Success Criteria @ 40M**:
   - Orientation: <110° (20% improvement)
   - Position: <250cm
   - Variance: >0.3 (stable)
   - KL: <0.1 (stable)

### Phase 2: Re-enable Components (IF Phase 1 succeeds)

1. **If 40M passes all criteria**:
   - Re-enable new components with reduced weights
   - Launch Session 8i.1 with full feature set

2. **If 40M fails**:
   - Distance-gating approach is fundamentally flawed
   - Need to rethink strategy (gradual weight transition?)

---

## Implementation

See `config_fixes_session_8i.md` for exact code changes.

---

## Lessons Learned

1. ✅ **Auto-pause saved us** - Session 8g collapsed @ 100M without it
2. ⚠️ **Too many variables** - Changed 3 things at once (gating + 2 new components)
3. ⚠️ **Large weight jumps risky** - 7.5x jump may be too aggressive
4. 📊 **Need TensorBoard** - Can't diagnose without metric history

---

## Next Steps

**Your decision**:
- [ ] Option 1: Conservative (disable new components) ⭐ RECOMMENDED
- [ ] Option 2: Moderate (reduce all weights by 50%)
- [ ] Option 3: Diagnose first (install TensorBoard, analyze collapse point)

**Time estimate**:
- Config changes: 5 minutes
- New 40M run: 2.5-3 hours
- Evaluation: 30 minutes

**Total time to recovery**: ~3-4 hours
