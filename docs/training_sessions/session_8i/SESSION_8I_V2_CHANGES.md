# Session 8i v2 - Configuration Changes

## Date: 2025-11-08

## Problem: Session 8i v1 Failed @ 29.36M Steps

**Training Log**:
- Started: 2025-11-07 16:42
- Emergency Pause: 2025-11-07 18:30 (29.36M / 40M steps)
- Trigger: AutoPauseCallback (KL > 0.1 or Variance < -0.3)
- Directory: `logs\sb3\mobilemmtrackee_v0\20251107_164255`

**Metrics Before Collapse** (@ 12.6M):
- ✅ KL Divergence: 0.0165 (excellent)
- ✅ Explained Variance: 0.854 (excellent)
- ✅ FPS: 4945

**Root Cause Analysis**:
1. 🔴 **Weight jump too aggressive**: 7.5x (4.0 → 30.0) at 0.7m threshold
2. 🟡 **Untested new components**: orientation_progress_bonus, angular_velocity_penalty
3. 🟡 **Potential gradient conflicts**: New components may interfere with orientation_tracking

---

## Solution: Conservative Configuration v2

### Changes Applied

#### 1. Smoothed Distance-Gated Weight Transition
```python
# v1 (FAILED):
orientation_tracking_far: 4.0   # 7.5x jump → unstable
orientation_tracking_close: 30.0

# v2 (FIXED):
orientation_tracking_far: 8.0   # 3.75x jump → smoother (DOUBLED from 4.0)
orientation_tracking_close: 30.0
```

**Rationale**:
- Reduced jump ratio from 7.5x to 3.75x
- Still maintains strong orientation focus in close zone
- Less abrupt policy change at threshold boundary

#### 2. Disabled New Reward Components (A/B Testing)
```python
# v1 (FAILED):
orientation_progress_bonus: 2.0  # NEW component, untested
angular_velocity_penalty: 1.0    # NEW component, untested

# v2 (FIXED):
orientation_progress_bonus: 0.0  # DISABLED for isolation test
angular_velocity_penalty: 0.0    # DISABLED for isolation test
```

**Rationale**:
- Isolate distance-gating as the ONLY variable change
- Validate core mechanism before adding enhancements
- If v2 succeeds → distance-gating works, re-enable in Session 8i.1
- If v2 fails → distance-gating itself is flawed, need new approach

---

## Testing Strategy

### Phase 1: Validate Core Distance-Gating (NOW)
```powershell
.\scripts\launch_session_8i.ps1 -Phase short  # 40M steps, ~3 hours
```

**Success Criteria @ 40M**:
- ✅ Orientation: <110° (20% improvement from 135.1°)
- ✅ Position: <250cm (maintain ~237cm)
- ✅ Explained Variance: >0.3 (stable learning)
- ✅ KL Divergence: <0.1 (stable policy)
- ✅ No emergency pause triggered

### Phase 2: Re-enable Components (IF Phase 1 succeeds)
- Launch Session 8i.1 with reduced component weights:
  ```python
  orientation_progress_bonus: 1.0   # 50% of original
  angular_velocity_penalty: 0.5     # 50% of original
  ```

---

## Expected Outcomes

### If v2 Succeeds @ 40M ✅
- **Conclusion**: Distance-gating works, weight jump was the issue
- **Next Step**: Session 8i.1 with re-enabled components
- **Confidence**: High (isolated variable)

### If v2 Fails @ 40M ❌
- **Conclusion**: Distance-gating itself is problematic
- **Possible Issues**:
  - Hard threshold creates discontinuity
  - Need gradual weight transition (sigmoid?)
  - Threshold location incorrect (0.7m too close/far?)
- **Next Step**: Redesign approach (Session 8j?)

---

## Files Modified

- `src/rl_platform/tasks/mobile_mm/config.py`:
  - Line 162: `orientation_tracking_far: 8.0` (was 4.0)
  - Line 164: `orientation_progress_bonus: 0.0` (was 2.0)
  - Line 165: `angular_velocity_penalty: 0.0` (was 1.0)

---

## Launch Command

```powershell
# Start fresh 40M validation run
.\scripts\launch_session_8i.ps1 -Phase short

# Expected time: ~3 hours @ 4945 fps
# Monitor: Variance should stay positive, KL should stay <0.1
```

---

## Monitoring Checklist

During training (check every 10M):
- [ ] @ 10M: Variance > 0.3, KL < 0.05
- [ ] @ 20M: Variance > 0.3, KL < 0.05
- [ ] @ 30M: Variance > 0.3, KL < 0.05
- [ ] @ 40M: Variance > 0.3, KL < 0.05

If any checkpoint fails:
- Stop training immediately
- Analyze logs
- Roll back to previous stable checkpoint
- Re-evaluate approach

---

## Backup Plan

If v2 also fails, consider:

**Option A: Sigmoid Weight Transition**
```python
# Smooth transition instead of hard threshold
weight = far_weight + (close_weight - far_weight) * sigmoid((threshold - distance) / smoothness)
```

**Option B: Gradual Linear Ramp**
```python
# Linear interpolation in transition zone
if distance > 0.8:  # Far zone
    weight = 8.0
elif distance < 0.6:  # Close zone
    weight = 30.0
else:  # Transition zone (0.6-0.8m)
    weight = 8.0 + (30.0 - 8.0) * (0.8 - distance) / 0.2
```

**Option C: Disable Distance-Gating**
```python
# Go back to constant weight
orientation_tracking: 15.0  # Middle ground
```

---

## Next Steps

1. ✅ Config modified (v2 applied)
2. ⏳ Launch training: `.\scripts\launch_session_8i.ps1 -Phase short`
3. ⏳ Monitor @ 10M, 20M, 30M, 40M
4. ⏳ Evaluate @ 40M
5. ⏳ Decide: Continue to 120M or adjust approach

**Status**: Ready to launch! 🚀
