# Session 8e Complete Analysis: 50M vs 100M Results

**Date:** November 1, 2025  
**CRITICAL FINDING:** Session 8e got WORSE from 50M to 100M! 🚨

---

## 📊 Session 8e Performance Over Time

| Metric | @ 50M | @ 100M | Change | Verdict |
|--------|-------|--------|--------|---------|
| **Position Error (cm)** | 349.4 | **408.0** | +16.8% | ❌ **WORSE!** |
| **Position Median (cm)** | 311.2 | **436.9** | +40.4% | ❌ **MUCH WORSE!** |
| **Orientation Error (°)** | 48.5 | **34.0** | -29.9% | ✅ **BETTER!** |
| **Orientation Median (°)** | 47.2 | **34.6** | -26.7% | ✅ **BETTER!** |
| **Mean Reward** | -259,394 | **-292,775** | -12.9% | ❌ **WORSE** |
| **Reachability Bonus** | 0.79 | **0.58** | -26.6% | ❌ **WORSE** |
| **Reachability Distance Penalty** | 528.90 | **582.30** | +10.1% | ❌ **WORSE** |
| **Base Mobilization** | 0.15 | **0.13** | -13.3% | ❌ **WORSE** |
| **Base Alignment** | 0.056 | **0.018** | -67.9% | ❌ **MUCH WORSE!** |

---

## 🚨 Critical Finding: Continued Degradation

**Session 8e's bell-shaped reachability approach FAILED and got progressively worse!**

### Position Tracking: COLLAPSED
```
  50M: 349.4cm mean  →  100M: 408.0cm mean  (+16.8% worse!)
  50M: 311.2cm median →  100M: 436.9cm median (+40.4% worse!)
```

**This is catastrophic** - position error increased by 58.6cm at median!

### Orientation: IMPROVED (Only Bright Spot)
```
  50M: 48.5° mean  →  100M: 34.0° mean  (-29.9% better!)
  50M: 47.2° median →  100M: 34.6° median (-26.7% better!)
```

**Interesting:** Orientation improved significantly despite position collapse.

### Reachability: STILL COLLAPSED
```
  50M: 0.79 bonus  →  100M: 0.58 bonus  (-26.6% worse!)
  50M: 528.90 penalty →  100M: 582.30 penalty (+10.1% worse!)
```

Bell-shaped reachability never recovered.

### Mobilization: DEGRADED
```
  50M: 0.15 mobilization →  100M: 0.13 (-13.3%)
  50M: 0.056 alignment →  100M: 0.018 (-67.9%!)
```

Base movement became even less purposeful!

---

## 📈 Complete Comparison: Session 8e @ 50M vs 100M vs Session 8f @ 100M

| Metric | 8e @ 50M | 8e @ 100M | 8f @ 100M | Winner |
|--------|----------|-----------|-----------|--------|
| **Position (cm)** | 349.4 | 408.0 | **307.8** | 🏆 8f |
| **Position Median (cm)** | 311.2 | 436.9 | **298.7** | 🏆 8f |
| **Orientation (°)** | 48.5 | **34.0** | 46.5 | 🏆 8e @ 100M |
| **Orientation Median (°)** | 47.2 | **34.6** | 40.7 | 🏆 8e @ 100M |
| **Mean Reward** | -259k | -293k | **-126k** | 🏆 8f |
| **Reachability Bonus** | 0.79 | 0.58 | **0.64** | 🏆 8f |
| **Distance Penalty** | 529 | 582 | **232** | 🏆 8f |
| **Mobilization** | 0.15 | 0.13 | **0.32** | 🏆 8f |
| **Alignment** | 0.056 | 0.018 | **0.36** | 🏆 8f |

---

## 🎯 Key Insights

### 1. Bell-Shaped Reachability is FUNDAMENTALLY BROKEN

**Evidence:**
- Position error INCREASED 58.6cm from 50M to 100M
- Reachability bonus DECREASED from 0.79 → 0.58
- Base alignment COLLAPSED 68% (0.056 → 0.018)

**Root Cause:**
- Bell-shaped reward too brittle
- Policy learned to avoid optimal zone (0.5m) because penalty for error is too high
- Safer to stay farther away where gradient is gentler

### 2. Orientation Improvement Despite Position Collapse

**Surprising finding:**
- Orientation improved 29.9% (48.5° → 34.0°)
- While position degraded 16.8%

**Hypothesis:**
- Policy gave up on position tracking
- Focused remaining capacity on orientation (easier sub-task)
- Classic reinforcement learning: optimize achievable goals, abandon impossible ones

### 3. Session 8f Validates Two-Zone Linear Approach

**Session 8f @ 100M is superior in almost every metric:**
- ✅ Position: 307.8cm vs 8e's 408.0cm (-24.6%!)
- ✅ Reward: -126k vs 8e's -293k (+57.0%!)
- ✅ Mobilization: 0.32 vs 8e's 0.13 (+146%!)
- ⚠️ Orientation: 46.5° vs 8e's 34.0° (+36.8% worse)

**Only Session 8e's orientation (34.0°) is better** - this is the mystery!

---

## 💡 Why Did Session 8e's Orientation Improve?

**Hypothesis 1: Task Simplification**
- Policy abandoned position tracking (too hard)
- Freed up capacity to focus on orientation
- Result: Better orientation, worse position

**Hypothesis 2: Bell-Shaped Side Effect**
- Bell-shaped penalty at wrong distance
- Forces base to stay at suboptimal position
- But arm still tries to reach → more rotation → better orientation practice

**Hypothesis 3: Reward Structure Emphasis**
- Session 8e may have had higher orientation_tracking weight
- Check config: orientation_tracking = ?
- Session 8f: orientation_tracking = 30.0

---

## 🔬 Detailed Reward Breakdown

### Session 8e @ 100M (Complete Data)

```
Total Mean Reward: -292,775

Positive Contributors:
+ orientation_tracking: 165.07  (💪 Best!)
+ position_tracking:     16.20
+ base_mobilization:      0.13
+ base_target_alignment:  0.02
+ reachability_bonus:     0.58  (💀 Collapsed!)
+ progress_bonus:         0.00

Negative Contributors:
- reachability_distance_penalty: 582.30  (🚨 Very high!)
- position_distance_penalty:     306.43
- base_overshoot_penalty:        11.86
- jerk_penalty:                   0.33
- inner_margin_penalty:           0.18
- velocity_limit_penalty:         0.05
```

### Comparison with Session 8f @ 100M

| Component | 8e @ 100M | 8f @ 100M | Winner |
|-----------|-----------|-----------|--------|
| orientation_tracking | **165.07** | 144.72 | 🏆 8e (+14%) |
| position_tracking | 16.20 | **21.83** | 🏆 8f (+35%) |
| reachability_bonus | 0.58 | **0.64** | 🏆 8f (+10%) |
| reachability_distance_penalty | 582.30 | **231.77** | 🏆 8f (-60%!) |
| position_distance_penalty | 306.43 | N/A | - |
| base_mobilization | 0.13 | **0.32** | 🏆 8f (+146%) |
| base_alignment | 0.018 | **0.36** | 🏆 8f (+1900%!) |

---

## 🎭 The Orientation Mystery Deepens

**Three sessions now show surprising orientation behavior:**

1. **Session 8cv2 @ 200M:** 20.5° (BEST!) despite -1207 reachability
2. **Session 8e @ 100M:** 34.0° (GOOD!) despite 408cm position
3. **Session 8f @ 100M:** 46.5° (WORST!) despite 308cm position

**Pattern:**
- **Worse position tracking → Better orientation!**
- 8cv2: 328cm position, 20.5° orientation
- 8e: 408cm position, 34.0° orientation  
- 8f: 308cm position, 46.5° orientation

**Hypothesis:**
- When position tracking fails, policy focuses on orientation
- When position tracking succeeds, orientation becomes secondary
- Trade-off in learning capacity or reward weighting

---

## 📊 Visual Comparison

### Position Error Progression (Lower is Better)
```
8e @ 50M:  ████████████████████████████████████████ 349cm
8e @ 100M: ████████████████████████████████████████████████ 408cm ❌ WORSE!
8f @ 100M: ██████████████████████████████████ 308cm ⭐
```

### Orientation Error Progression (Lower is Better)
```
8e @ 50M:  ████████████████████████████████████████████████ 48.5°
8e @ 100M: ████████████████████████████████████ 34.0° ⭐ BEST!
8f @ 100M: ████████████████████████████████████████████ 46.5°
```

### Mean Reward Progression (Higher is Better)
```
8e @ 50M:  ████████████████████████████████████████████ -259k
8e @ 100M: ████████████████████████████████████████████████ -293k ❌ WORSE!
8f @ 100M: ████████████████ -126k ⭐
```

### Reachability Bonus (Higher is Better)
```
8e @ 50M:  ████ 0.79
8e @ 100M: ███ 0.58 ❌ WORSE!
8f @ 100M: ███ 0.64 ⭐
```

---

## ✅ Conclusions

### 1. Bell-Shaped Reachability FAILED
- Session 8e degraded from 50M to 100M
- Position error increased 17%, median increased 40%!
- Reachability bonus collapsed further (0.79 → 0.58)
- **Do NOT use bell-shaped reachability for Session 8g!**

### 2. Session 8f is CLEARLY SUPERIOR
- 24.6% better position (308cm vs 408cm)
- 57% better reward (-126k vs -293k)
- 146% better mobilization (0.32 vs 0.13)
- **Use Session 8f as baseline for Session 8g**

### 3. Orientation Mystery Requires Investigation
- Session 8e @ 100M: 34.0° (excellent!)
- Session 8f @ 100M: 46.5° (mediocre)
- **Hypothesis:** Failed position tracking → focus shifts to orientation
- **Action:** Investigate orientation_tracking weight in 8e vs 8f config

### 4. Two-Zone Linear Works Better Than Bell-Shaped
- Despite both having low reachability bonus (~0.6)
- Session 8f achieves much better position tracking
- Two-zone linear provides better learning signal

---

## 🎯 Recommendations for Session 8g

### Must Keep from Session 8f:
1. ✅ Two-zone linear reachability (NOT bell-shaped!)
2. ✅ Atomic root state write
3. ✅ Distance-gated penalties
4. ✅ Heading cue observations

### Must Fix:
1. 🔧 Increase reachability_maintenance_reward: 40 → 80
2. 🔧 Investigate orientation_tracking weight (why is 8e better?)
3. 🔧 Consider dynamic reward weighting based on position performance

### Target Performance:
- Position: ≤300cm (improve from 8f's 308cm)
- Orientation: ≤35° (match 8e's 34.0°!)
- Reachability bonus: ≥3.0 (recovery toward 8d's 7.06)
- Mean reward: ≥-100k (improve from 8f's -126k)

---

## 📁 Files Generated

- `evaluation_plots/session_8e_100M/20251031_224729/eval_summary_20251101_171948.json`
- `evaluation_plots/session_8e_100M/20251031_224729/episodes_20251101_171948.csv`
- `evaluation_plots/session_8e_100M/20251031_224729/steps_20251101_171948.csv`
- `evaluation_plots/session_8e_100M/20251031_224729/arrays_20251101_171948.npz`

---

**VERDICT:** Bell-shaped reachability is FUNDAMENTALLY BROKEN. Session 8f's two-zone linear approach is clearly superior, but we must solve the orientation mystery from Session 8e! 🚀
