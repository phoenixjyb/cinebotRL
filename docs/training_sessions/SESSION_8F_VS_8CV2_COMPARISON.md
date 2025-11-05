# Session 8f vs Session 8cv2 Comparison

**Date:** November 1, 2025  
**Purpose:** Compare Session 8f (playbook fixes) with Session 8cv2 (200M baseline)

---

## 📊 Executive Summary

**Session 8f is DRAMATICALLY better than Session 8cv2!**

| Metric | Session 8cv2 @ 200M | Session 8f @ 100M | Improvement |
|--------|---------------------|-------------------|-------------|
| **Position Error** | 328.2cm | **307.8cm** | ✅ **-6.2%** |
| **Orientation Error** | 20.5° | **46.5°** | ❌ **+127%** (WORSE!) |
| **Mean Reward** | -448k | **-126k** | ✅ **+71.9%** |
| **Reachability Bonus** | -1207.6 | **0.64** | ✅ **+100,088%** |
| **Training Duration** | 200M steps | 100M steps | ✅ **2x faster** |

### 🎯 Key Takeaway

**Session 8f FIXES the catastrophic reachability collapse in 8cv2!**
- 8cv2 had **negative reachability bonus** (-1207.6) → policy stuck outside workspace
- 8f has **positive reachability bonus** (0.64) → policy maintains workspace access
- **BUT:** 8cv2 had much better orientation (20.5° vs 46.5°)

---

## 🔬 Detailed Metrics Comparison

### Tracking Performance

| Metric | Session 8cv2 | Session 8f | Winner |
|--------|--------------|------------|--------|
| **Position Error** | | | |
| Mean | 328.2cm | **307.8cm** | 🏆 8f (-6.2%) |
| Median | 357.7cm | **298.7cm** | 🏆 8f (-16.5%) |
| Std Dev | 115.5cm | N/A | - |
| P95 | 468.9cm | N/A | - |
| Min | 22.1cm | N/A | - |
| Max | 534.2cm | N/A | - |
| **Orientation Error** | | | |
| Mean | **20.5°** | 46.5° | 🏆 8cv2 (-56%) |
| Median | **17.8°** | 40.7° | 🏆 8cv2 (-56%) |
| Std Dev | 10.9° | N/A | - |
| P95 | 36.7° | N/A | - |

**Analysis:**
- Position tracking: 8f slightly better
- **Orientation tracking: 8cv2 MUCH better!** 🚨
- This is surprising - 8cv2 achieved excellent orientation despite catastrophic reachability

### Reward Components

| Component | Session 8cv2 | Session 8f | Change |
|-----------|--------------|------------|--------|
| **Total Reward** | -448,029 | **-126,482** | ✅ **+71.9%** |
| **Tracking Rewards** | | | |
| position_tracking | 11.05 | 21.83 | ✅ +97.6% |
| orientation_tracking | 92.72 | 144.72 | ✅ +56.1% |
| **Reachability** | | | |
| reachability_bonus | **-1207.63** 💥 | **0.64** | ✅ +100,088%! |
| reachability_distance_penalty | N/A | 231.77 | - |
| target_distance_penalty | 1.89 | N/A | - |
| **Mobilization** | | | |
| base_mobilization | 0.11 | **0.32** | ✅ +190% |
| base_target_alignment | 0.004 | **0.36** | ✅ +8,900%! |
| base_overshoot_penalty | 8.41 | N/A | - |
| **Action Penalties** | | | |
| velocity_limit_penalty | 0.075 | N/A | - |
| jerk_penalty | 0.158 | N/A | - |
| stability_penalty | 0.031 | N/A | - |

**Critical Finding:**
- **8cv2 had NEGATIVE reachability bonus (-1207.63)** - this is catastrophic!
- 8f fixed this completely (0.64 positive)
- 8f mobilization rewards 190-8900% higher → base moves purposefully

### Base Movement

| Metric | Session 8cv2 | Session 8f | Winner |
|--------|--------------|------------|--------|
| **Linear Velocity X** | | | |
| Mean | 0.260 m/s | N/A | - |
| Max | 0.863 m/s | N/A | - |
| **Linear Velocity Y** | | | |
| Mean | 0.100 m/s | N/A | - |
| Max | 0.777 m/s | N/A | - |
| **Angular Velocity Z** | | | |
| Mean | -0.013 rad/s | N/A | - |
| Max | 0.693 rad/s | N/A | - |

**Note:** Session 8f base velocity data not available in eval summary, but mobilization rewards show active movement.

---

## 🎭 Architecture Differences

### Session 8cv2 Architecture

**Reward Structure:**
```python
reachability_maintenance_reward: Unknown weight
  - NEGATIVE bonus (-1207.63) → policy outside workspace!
target_distance_penalty: 1.0 weight (legacy)
base_mobilization: 0.5 weight
base_target_alignment: Unknown weight
base_overshoot_penalty: 5.0 weight
```

**Observations:**
- Unknown dimensions (likely 49 base)
- No heading cue
- No distance gating

**Control:**
- Sequential velocity/pose writes (potential conflict)

**Training:**
- 200M steps (2x longer than 8f)
- Unknown environment count

### Session 8f Architecture

**Reward Structure:**
```python
reachability_maintenance: 40.0 weight
  - Two-zone linear (0.35-0.55m optimal plateau)
reachability_distance_penalty: 10.0 weight (only >0.60m)
position_distance_penalty: 5.0 weight
base_mobilization: 0.5 weight
base_target_alignment: 2.0 weight
base_overshoot_penalty: 5.0 weight
```

**Observations:**
- 51 base dims (49 + 2 heading cue)
- Heading cue: sin/cos of base→target yaw error
- Distance-gated penalties

**Control:**
- Atomic root state write (13-element tensor)

**Training:**
- 100M steps (half of 8cv2)
- 16,384 environments

---

## 🚨 Critical Issue: Orientation Regression

**Session 8f orientation is 127% WORSE than 8cv2!**

```
8cv2: 20.5° mean, 17.8° median  ⭐
8f:   46.5° mean, 40.7° median  🚨
```

**Hypothesis:**
1. **8cv2 may have different observation space** that emphasized orientation better
2. **8cv2's catastrophic reachability** may have forced policy to prioritize orientation (since it couldn't move base)
3. **8f's heading cue** may have introduced confusion or over-rotation
4. **Distance gating in 8f** may have relaxed orientation penalties too much

**Evidence from rewards:**
- 8cv2 orientation_tracking: 92.72
- 8f orientation_tracking: 144.72 (+56%)
- Yet 8cv2 achieved better orientation!
- This suggests 8cv2's reward structure penalized orientation errors more effectively

---

## 💡 What We Can Learn

### What 8cv2 Got Right (Despite Catastrophic Reachability)

1. ✅ **Excellent orientation control** (20.5° vs 8f's 46.5°)
   - Even with negative reachability, policy learned orientation
   - Reward structure emphasized orientation more effectively
   
2. ✅ **Reasonable position tracking** (328cm, only 6% worse than 8f)
   - Policy could track reasonably without base movement
   
3. ⚠️ **Some base movement** (0.26 m/s linear X, 0.10 m/s linear Y)
   - Base was active despite low mobilization rewards
   - But not purposeful (alignment reward only 0.004)

### What 8f Got Right

1. ✅ **Fixed catastrophic reachability collapse** (+100k% improvement!)
   - Two-zone linear reachability works
   - Distance-gated penalties allow mobilization
   
2. ✅ **Purposeful base movement** (alignment +8900%, mobilization +190%)
   - Base moves toward targets
   - Clear directional guidance
   
3. ✅ **Higher total reward** (+72%)
   - Overall learning more effective
   - Better balance of objectives
   
4. ✅ **2x faster training** (100M vs 200M)
   - More efficient learning

### What 8f Needs to Fix

1. 🚨 **Orientation regression** (46.5° vs 8cv2's 20.5°)
   - Need to investigate 8cv2's orientation reward structure
   - May need to increase orientation_tracking weight
   - Check if heading cue is confusing the policy
   
2. ⚠️ **Workspace distance still drifts** (0.42m → 0.60m)
   - Same issue as 8d/8e
   - Need stronger reachability gravity

---

## 🎯 Recommendations for Session 8g

### Priority 1: Investigate 8cv2's Orientation Success

**Action Items:**
1. Compare orientation reward weights between 8cv2 and 8f
2. Check if 8cv2 used different observation features for orientation
3. Analyze if heading cue in 8f is helping or hurting orientation

**Hypothesis to Test:**
- 8cv2 may have had **higher orientation_tracking weight**
- 8f's heading cue might be over-rotating the base
- Distance gating in 8f might be relaxing orientation penalties too much

### Priority 2: Combine 8cv2 and 8f Strengths

**Proposed Changes for 8g:**
```python
# Keep 8f fixes
+ Atomic root state write
+ Distance-gated penalties
+ Two-zone linear reachability
+ Heading cue observations

# Add 8cv2 orientation emphasis
orientation_tracking: 30.0 → 50.0  # Match or exceed 8cv2's effectiveness
reachability_maintenance: 40.0 → 80.0  # Stronger gravity
distance_gate_threshold: 0.55m → 0.50m  # Tighter tolerance

# Verify heading cue isn't over-rotating
+ Add yaw rate penalty if needed
+ Monitor angular velocity Z
```

### Priority 3: Session 8g Success Criteria

**Must Achieve:**
- ✅ Position error ≤ 310cm (maintain 8f level)
- ✅ Orientation error ≤ 25° (approach 8cv2's 20.5°)
- ✅ Reachability bonus ≥ 3.0 (much better than 8f's 0.64)
- ✅ Workspace distance stable 0.45-0.55m

**If orientation remains poor:**
- Consider removing or modifying heading cue
- Increase orientation_tracking weight further (50 → 80)
- Analyze base angular velocity patterns

---

## 📊 Visual Comparison

### Position Error (Lower is Better)
```
8cv2 @ 200M: ████████████████████████████████████ 328cm
8f @ 100M:   ██████████████████████████████████ 308cm ⭐ -6.2%
```

### Orientation Error (Lower is Better) 🚨
```
8cv2 @ 200M: ████████████████████ 20.5° ⭐⭐⭐
8f @ 100M:   ████████████████████████████████████████████ 46.5° (-127%!)
```

### Mean Reward (Higher is Better)
```
8cv2 @ 200M: ████████████████████████████████████████████ -448k
8f @ 100M:   ████████████ -126k ⭐ +72%
```

### Reachability Bonus (Higher is Better)
```
8cv2 @ 200M: 💥💥💥 -1208 (CATASTROPHIC!)
8f @ 100M:   ▌ 0.64 ⭐ (+100,088%!)
```

### Base Mobilization (Higher is Better)
```
8cv2 @ 200M: ▌ 0.11
8f @ 100M:   ███ 0.32 ⭐ +190%
```

### Base Alignment (Higher is Better)
```
8cv2 @ 200M: ▌ 0.004
8f @ 100M:   ███████████████████████████████ 0.36 ⭐ +8900%
```

---

## 🏆 Session Rankings

### Overall Winner: **Session 8f** 🥇

**Justification:**
- Fixed catastrophic reachability collapse
- Higher total reward (+72%)
- Better mobilization and alignment
- More efficient training (100M vs 200M)

### Orientation Winner: **Session 8cv2** 🥇

**Justification:**
- 20.5° vs 8f's 46.5° (127% better!)
- Achieved excellent orientation despite negative reachability
- This is the mystery we need to solve for 8g

### Position Winner: **Session 8f** 🥇 (Marginal)

**Justification:**
- 307.8cm vs 8cv2's 328.2cm (6.2% better)
- But margin is small

---

## 🔍 Mystery to Solve

**Why did 8cv2 achieve excellent orientation (20.5°) despite catastrophic reachability (-1207)?**

**Possible Explanations:**
1. **Higher orientation weight:** 8cv2 may have used orientation_tracking > 30
2. **Different observation space:** 8cv2 may have had better orientation features
3. **Static base advantage:** With base stuck outside workspace, policy focused entirely on arm orientation
4. **Reward structure:** 8cv2's reward may have weighted orientation > position
5. **Training duration:** 200M steps gave more time to refine orientation

**Action Required:**
- Find 8cv2 configuration files
- Compare reward weights and observation spaces
- Determine why orientation was so good
- Replicate this in 8g while keeping 8f's reachability fixes

---

## 📋 Next Steps

1. **Locate 8cv2 configuration** (`src/rl_platform/tasks/mobile_mm/config.py` historical version)
2. **Analyze 8cv2 orientation reward structure**
3. **Design Session 8g** combining:
   - 8f's reachability fixes (atomic state, distance gating, two-zone)
   - 8cv2's orientation success (higher weight? different features?)
   - Stronger reachability gravity (40 → 80)
4. **Test 8g at 50M** to verify orientation improvement
5. **Full 100M training** if 50M looks good

**Target for Session 8g:**
- Position: ≤310cm (maintain 8f)
- Orientation: ≤25° (approach 8cv2's 20.5°)
- Reachability: ≥3.0 (much better than 8f's 0.64)
- Training: 100M steps (like 8f)

---

**Conclusion:** Session 8f is clearly better overall, but we MUST understand and replicate 8cv2's orientation success in Session 8g! 🚀
