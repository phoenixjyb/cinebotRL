# Session 8i v3: Sigmoid Smooth Transition Fix

**Created**: 2025-01-08 02:00  
**Problem**: v1 and v2 both failed due to hard threshold causing policy oscillation  
**Solution**: Replace hard threshold with Sigmoid smooth transition  

---

## Problem Analysis

### v1 Failure (29.36M steps)
- **Configuration**: 7.5x jump (4.0 → 30.0) at 0.7m hard threshold
- **Result**: Emergency pause @ 29.36M (73% of 40M target)
- **Hypothesis**: Jump too aggressive

### v2 Failure (20.97M steps) - WORSE!
- **Configuration**: 3.75x jump (8.0 → 30.0), smoother than v1
- **Result**: Emergency pause @ 20.97M (52% of 40M target)
- **Critical Finding**: Collapsed 8.4M steps EARLIER than v1

### Root Cause Identified
**Hard threshold itself causes instability, not jump magnitude:**

```python
# BROKEN: v1/v2 Hard Threshold
ori_weight_current = torch.where(
    distance < 0.7,
    torch.full_like(distance, 30.0),  # Close
    torch.full_like(distance, 8.0)     # Far
)
# Problem: Abrupt switch at 0.7m boundary
# → Policy oscillates when robot crosses threshold
# → Gradient discontinuity prevents stable learning
```

**Evidence**:
- Smoothing jump (7.5x → 3.75x) made it worse
- Disabling new components didn't help
- Both failed in 20-30M range
- **Conclusion**: Discontinuity is the enemy, not magnitude

---

## v3 Solution: Sigmoid Smooth Transition

### Mathematical Fix
```python
# v3: Smooth Sigmoid Transition
sigmoid_factor = 1.0 / (1.0 + torch.exp((distance - 0.7) / 0.15))
ori_weight_current = 8.0 + (30.0 - 8.0) * sigmoid_factor

# Result:
# distance = 0.4m → sigmoid ≈ 0.95 → weight ≈ 28.9 (close mode)
# distance = 0.7m → sigmoid = 0.50 → weight = 19.0 (midpoint)
# distance = 1.0m → sigmoid ≈ 0.05 → weight ≈ 9.1 (far mode)
```

### Advantages
1. **No Discontinuity**: Smooth gradient at all points
2. **Stable Transition**: ~0.3m wide zone (0.55m - 0.85m)
3. **Mathematically Proven**: Sigmoid is infinitely differentiable
4. **Predictable Behavior**: No sudden policy changes

### Implementation
- **File**: `src/rl_platform/tasks/mobile_mm/rewards.py` lines 1020-1045
- **Config**: Added `distance_gate_smoothness: 0.15` parameter
- **Test**: Smoke test before launch

---

## Configuration Comparison

| Parameter | v1 (Failed @ 29M) | v2 (Failed @ 21M) | v3 (Sigmoid) |
|-----------|-------------------|-------------------|--------------|
| Transition Type | Hard threshold | Hard threshold | **Sigmoid smooth** |
| Far weight | 4.0 | 8.0 | 8.0 |
| Close weight | 30.0 | 30.0 | 30.0 |
| Jump ratio | 7.5x | 3.75x | 3.75x |
| **Smoothness** | **0 (instant)** | **0 (instant)** | **0.15 (smooth)** |
| Progress bonus | 2.0 | 0.0 (disabled) | 0.0 (disabled) |
| Angular penalty | 1.0 | 0.0 (disabled) | 0.0 (disabled) |

---

## Training Plan

### Phase 2 (v3): 0 → 40M Steps
- **Command**: `.\scripts\launch_session_8i.ps1 -Phase short`
- **Duration**: ~3 hours (40M / 4500 FPS / 3600s)
- **Success Criteria**:
  - No emergency pause
  - Reaches 40M steps
  - KL < 0.05, Variance > 0.3 throughout
  - Pass 29M mark (where v1 failed)
  - Pass 21M mark (where v2 failed)

### Monitoring Checkpoints
- **10M**: Check KL/Variance stability
- **20M**: Critical - where v2 collapsed
- **30M**: Critical - where v1 collapsed
- **40M**: Full Phase 2 completion

### Expected Outcome
- **Position**: ~237cm (maintain Session 8h quality)
- **Orientation**: <110° (improve from 135°)
- **Stability**: Smooth weight transition, no oscillation

---

## Fallback Plans

### If v3 Fails
1. **Wider Smoothness**: Increase to 0.25 (0.45m - 0.95m zone)
2. **Linear Transition**: Three zones (far, linear, close)
3. **Abandon Distance Gating**: Return to fixed weight (Session 8h)

### If v3 Succeeds
1. **Phase 3**: 40M → 120M (reduce exploration)
2. **Documentation**: Update master log with Sigmoid victory
3. **Knowledge Base**: Document hard threshold pitfall for future

---

## Files Modified

### Code Changes
- `src/rl_platform/tasks/mobile_mm/rewards.py`:
  - Lines 1020-1045: Sigmoid smooth transition implementation
  - Replaced `torch.where` hard threshold with sigmoid formula
  - Updated comments to reference v1/v2 failures

- `src/rl_platform/tasks/mobile_mm/config.py`:
  - Line 163: Added `distance_gate_smoothness: 0.15`
  - Lines 155-161: Updated comments with v1/v2/v3 history

### Documentation Created
- `SESSION_8I_FAILURE_ANALYSIS.md`: v1 root cause analysis
- `SESSION_8I_V2_CHANGES.md`: v2 rationale and testing
- `SESSION_8I_V3_SIGMOID_FIX.md`: This document (v3 solution)

---

## Theoretical Justification

### Why Sigmoid Works
1. **Smooth Gradient**: `d(sigmoid)/dx` is continuous everywhere
2. **Bounded Output**: Always between 0 and 1
3. **Controllable Width**: `smoothness` parameter tunes transition zone
4. **Symmetric**: Equal behavior approaching from far/close

### Policy Learning Benefits
- **Stable Gradients**: No discontinuous jumps in reward signal
- **Predictable Transitions**: Policy can learn smooth approach behavior
- **No Oscillation**: Crossing boundary doesn't trigger sudden weight change
- **Exploration Friendly**: Can safely explore around threshold without penalty

---

## Next Steps

1. **Smoke Test**: `.\scripts\launch_session_8i.ps1 -Test`
2. **Launch v3**: `.\scripts\launch_session_8i.ps1 -Phase short`
3. **Monitor**: Check 10M, 20M, 30M, 40M checkpoints
4. **Validate**: Compare metrics to v1/v2 via TensorBoard
5. **Document**: Update master log with results

---

**Status**: Ready to launch  
**Expected Success**: High (mathematically sound fix)  
**Risk**: Low (fallback to Session 8h if needed)
