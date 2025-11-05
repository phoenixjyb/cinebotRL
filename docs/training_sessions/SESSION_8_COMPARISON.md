# Sessions 8d, 8e, 8f - Comparative Analysis

**Generated:** November 1, 2025  
**Purpose:** Cross-session comparison to guide Session 8g design

---

## 📊 Quick Performance Summary

### Position Error (Lower is Better)
```
Session 8d @ 109M:  311.0 cm   ████████████████████████████████
Session 8e @ 50M:   349.4 cm   ██████████████████████████████████████
Session 8f @ 100M:  307.8 cm ⭐ ███████████████████████████████  BEST
```

### Orientation Error (Lower is Better)
```
Session 8d @ 109M:  47.4°   ████████████████████████████████████
Session 8e @ 50M:   48.5°   █████████████████████████████████████
Session 8f @ 100M:  46.5° ⭐ ██████████████████████████████████  BEST
```

### Mean Reward (Higher is Better)
```
Session 8d @ 109M:  -177k   ████████████████████████████
Session 8e @ 50M:   -259k   ████████████████████████████████████████
Session 8f @ 100M:  -126k ⭐ ████████████████  BEST
```

---

## 🔬 Reward Architecture Evolution

### Session 8d: Linear Reachability
```python
# Simple linear bonus
reachability_bonus = interpolate(distance, 0.3→0.9m)
workspace_mean = 0.402m  # Too close!
result: 7.06 pts bonus   # Good but cramped
```

### Session 8e: Bell-Shaped Comfort Zone
```python
# Gaussian peak at 0.5m
reachability_bonus = gaussian(distance, center=0.5m, sigma=0.15m)
workspace_mean = 0.52m → 0.58m  # Drifting away
result: 0.79 pts bonus   # COLLAPSED! (-89%)
```

### Session 8f: Two-Zone Linear + Distance Gating
```python
# Three zones: approach, plateau, decay
reachability_bonus = {
    0.35-0.45m: linear ramp up,
    0.45-0.55m: plateau (max bonus),
    0.55-0.9m: linear decay
}
# PLUS: penalties gated by distance
penalty_gate = sigmoid((0.55 - distance) * 10.0)
workspace_mean = 0.42m → 0.60m  # Still drifting
result: 0.64 pts bonus   # Low but better than 8e
```

---

## 📈 Workspace Distance Trends

### Session 8d (Linear)
```
Training: Unknown (logs not available)
Evaluation: 0.402m  ⚠️ Too close (below optimal 0.45-0.55m)
```

### Session 8e (Bell-Shaped)
```
  10M: 0.523m  ✅ Good
  20M: 0.552m  ✅ Good
  30M: 0.558m  ⚠️ Starting drift
  40M: 0.575m  ⚠️ Drifting
  50M: 0.575m  ⚠️ Stabilized but too far
  60M: 0.577m  ⚠️ Still too far
  70M: 0.578m  ⚠️ Still too far
  73M: 0.582m  ⚠️ Still too far

Average: 0.56m (10% above optimal)
```

### Session 8f (Two-Zone Linear + Gating)
```
  10M: 0.416m  ⚠️ Too close
  20M: 0.450m  ✅ Perfect!
  30M: 0.481m  ✅ Good
  40M: 0.489m  ✅ Good
  50M: 0.556m  ⚠️ Starting drift
  60M: 0.554m  ⚠️ Drifting
  70M: 0.625m  🚨 Too far!
  80M: 0.568m  ⚠️ Fluctuating
  90M: 0.600m  ⚠️ Still too far

Best window: 20-40M (0.45-0.49m)
Problem: Drift starts at 50M
```

---

## 🎯 Detailed Metrics Comparison

| Metric | Session 8d | Session 8e | Session 8f | Winner |
|--------|------------|------------|------------|--------|
| **Tracking Performance** | | | | |
| Position error (mean) | 311.0cm | 349.4cm | **307.8cm** | 🏆 8f |
| Position error (median) | N/A | 311.2cm | **298.7cm** | 🏆 8f |
| Orientation error (mean) | 47.4° | 48.5° | **46.5°** | 🏆 8f |
| Orientation error (median) | N/A | 47.2° | **40.7°** | 🏆 8f |
| **Reward Components** | | | | |
| Mean total reward | -177k | -259k | **-126k** | 🏆 8f |
| Position tracking | 32.9 | 25.6 | 21.8 | 🏆 8d |
| Orientation tracking | 170.2 | 139.4 | 144.7 | 🏆 8d |
| **Reachability** | | | | |
| Reachability bonus | **7.06** | 0.79 | 0.64 | 🏆 8d |
| Distance penalty | 360 | 529 | **232** | 🏆 8f |
| Workspace distance | 0.402m | 0.56m | 0.55-0.60m | ⚠️ None |
| **Mobilization** | | | | |
| Base mobilization | N/A | 0.15 | **0.32** | 🏆 8f |
| Base alignment | N/A | 0.056 | **0.36** | 🏆 8f |
| Inner margin penalty | N/A | 0.21 | **0.13** | 🏆 8f |

---

## 💡 Key Insights

### What We Learned from Each Session

**Session 8d:**
- ✅ Linear reachability works
- ✅ High reachability bonus (7.06)
- ❌ Workspace too close (0.402m)
- ❌ Orientation collapse (47.4°)

**Session 8e:**
- ✅ Bell-shaped concept reasonable
- ❌ **FAILED:** Reachability bonus collapsed 89%
- ❌ Workspace drifted away (0.52m → 0.58m)
- ❌ Worse tracking than 8d (349cm vs 311cm)
- **Lesson:** Bell-shaped too brittle for dynamic tasks

**Session 8f:**
- ✅ **BEST tracking:** 307.8cm, 46.5°
- ✅ **BEST reward:** -126k (51% better than 8e)
- ✅ Mobilization working (0.32 vs 0.15)
- ✅ Distance gating effective early (20-40M)
- ⚠️ Workspace still drifts after 50M
- ⚠️ Reachability bonus still low (0.64)
- **Lesson:** Playbook fixes work, but need stronger reachability pull

---

## 🚨 Persistent Problem: Workspace Drift

### All Three Sessions Show Drift!

**Pattern:**
1. Early training: random or too-close positioning
2. Mid-training: finds optimal zone (0.45-0.55m)
3. Late training: drifts away (>0.55m)

**Hypothesis:**
- Policy discovers: "staying farther = lower penalties"
- Reachability bonus not strong enough to fight this
- Distance gating helps but insufficient

**Evidence:**
```
Session 8d: Final 0.402m (too close)
Session 8e: 0.52m → 0.58m (drift +11%)
Session 8f: 0.45m → 0.60m (drift +33%)
```

**Root Cause Analysis:**
1. **Reachability bonus collapse:** 7.06 → 0.64 (91% drop from 8d to 8f)
2. **Penalty avoidance:** Policy learns farther = safer
3. **Weak gravity:** Two-zone linear gives ~0 bonus beyond 0.55m
4. **No explicit constraint:** Workspace distance not directly rewarded

---

## 🎯 Recommendations for Session 8g

### Priority 1: Stronger Reachability Gravity (RECOMMENDED)

**Change:**
```python
reachability_maintenance_reward: 40 → 80  # 100% increase
optimal_plateau_width: 0.45-0.55m → 0.40-0.60m  # Wider margin
decay_rate: current → steeper  # Punish harder beyond 0.60m
```

**Expected Impact:**
- Reachability bonus: 0.64 → 3-5 pts (closer to 8d's 7.06)
- Workspace distance: stable 0.45-0.55m throughout training
- Position/orientation: maintain or improve from 8f

**Risk:** Low (validated by 8d's success with strong reachability)

### Priority 2: Explicit Workspace Distance Reward

**New Component:**
```python
def workspace_distance_reward(distance):
    optimal_center = 0.50
    optimal_radius = 0.05  # ±5cm tolerance
    
    if abs(distance - optimal_center) <= optimal_radius:
        return +50.0  # Large bonus in sweet spot
    elif 0.40 <= distance <= 0.60:
        return +25.0  # Medium bonus nearby
    else:
        return -100 * (distance - optimal_center)**2  # Quadratic penalty
```

**Expected Impact:**
- Direct feedback on workspace distance
- Less reliance on reachability map
- Clearer learning signal

**Risk:** Medium (might conflict with reachability bonus)

### Priority 3: Progressive Weight Schedule

**Implementation:**
```python
# Gradually increase reachability importance
def get_reachability_weight(timestep):
    if timestep < 20_000_000:
        return 40  # Early: allow exploration
    elif timestep < 50_000_000:
        return 60  # Mid: start enforcing
    else:
        return 100  # Late: strict enforcement
```

**Expected Impact:**
- Early training: flexible positioning
- Late training: tight workspace control
- Prevents drift in final stages

**Risk:** Low (natural curriculum)

### Priority 4: Success-Based Curriculum

**Implementation:**
```python
# Dynamic workspace requirements
if position_error < 2.50 AND orientation_error < 0.61:  # 250cm, 35°
    workspace_tolerance = 0.05  # Strict: ±5cm
else:
    workspace_tolerance = 0.15  # Relaxed: ±15cm
```

**Expected Impact:**
- Don't fight workspace until tracking is good
- Prevents early-training confusion
- Gradual refinement

**Risk:** Low (proven in curriculum learning)

---

## 🏆 Session Rankings

### Overall Performance: **Session 8f** 🥇

**Justification:**
- Best tracking accuracy
- Highest reward
- Mobilization working
- All playbook fixes validated

### Reachability Bonus: **Session 8d** 🥇

**Justification:**
- 7.06 pts (11x better than 8f)
- Linear approach more robust
- But workspace too close (0.402m)

### Most Improved: **Session 8f** 🥇

**Justification:**
- +51% reward vs 8e
- -11.9% position error vs 8e
- Mobilization +113% vs 8e

---

## 📋 Next Steps

1. **Implement Session 8g** with stronger reachability gravity (Priority 1)
2. **Monitor workspace distance** closely (target: 0.45-0.55m stable)
3. **Evaluate at 50M** to catch drift early
4. **If drift persists:** Try Priority 2 (explicit workspace reward)
5. **If still issues:** Combine Priority 1 + Priority 3 (progressive schedule)

**Expected Outcome:**
- Position error: <300cm (maintain 8f level)
- Orientation error: <45° (maintain 8f level)
- Reachability bonus: 3-5 pts (recovery toward 8d's 7.06)
- Workspace distance: 0.45-0.55m stable throughout training

**Success Criteria:**
✅ Position error ≤ 310cm  
✅ Orientation error ≤ 47°  
✅ Reachability bonus ≥ 3.0  
✅ Workspace distance drift < 10% (0.45-0.50m)

---

## 📊 Visual Comparison

```
Reachability Bonus (Higher is Better):
8d: ███████████████████████████████████████████ 7.06  ⭐
8e: ████ 0.79  💥 COLLAPSED
8f: ███ 0.64  ⚠️ Still low

Workspace Distance (Target: 0.45-0.55m):
Target:  ████████████████████████████ 0.45-0.55
8d:      ████████████████████ 0.40  (too close)
8e:      ██████████████████████████████████ 0.56  (drifting)
8f:      ███████████████████████████████ 0.55-0.60  (drifting)

Position Error (Lower is Better):
8d: ████████████████████████████████ 311cm
8e: ██████████████████████████████████████ 349cm
8f: ███████████████████████████████ 308cm  ⭐ BEST

Mean Reward (Higher is Better):
8d: ████████████████████████████ -177k
8e: ████████████████████████████████████████ -259k
8f: ████████████████ -126k  ⭐ BEST
```

---

**Conclusion:** Session 8f is the best overall, but needs stronger reachability gravity to prevent workspace drift in Session 8g. 🚀
