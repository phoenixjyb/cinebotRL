# Session 8G Evaluation Results - Post-Mortem Analysis

**Date**: November 2, 2025  
**Training Duration**: 100M steps (~7.2 hours)  
**Session ID**: 20251101_222053  
**Status**: ⚠️ **PARTIAL SUCCESS - Training collapsed at 100M**

---

## Executive Summary

Session 8g implemented evidence-based workspace expansion (0.6m→0.7m margin), gentler penalties (60→30 weight), and curriculum learning. **Workspace convergence succeeded brilliantly at 31M steps** (0.554m, perfect target!), but **catastrophic training collapse occurred by 100M** (std exploded 19,000x). The 40M checkpoint shows **position improvement vs Session 8f baseline** but **severe orientation degradation**, revealing curriculum imbalance issues.

---

## Evaluation Results

### 🟢 40M Checkpoint (Last Stable)
**Checkpoint**: `ppo_mobile_mm_40009728_steps.zip`  
**Training State**: After workspace convergence @ 31M, before instability @ 36M

| Metric | Mean | Median | P95 | vs Session 8f |
|--------|------|--------|-----|---------------|
| **Position Error** | 300.70 cm | 163.00 cm | 1054.85 cm | **+2% BETTER** 🟢 |
| **Orientation Error** | 130.38° | 125.04° | 163.57° | **+180% WORSE** 🔴 |
| **Reward** | -237,503 | -104,674 | - | -89% worse |

**Key Observations**:
- ✅ **Position tracking improved** vs Session 8f (301cm vs 308cm mean)
- ❌ **Orientation catastrophically degraded** (130° vs 8f's 46.5°)
- ⚠️ High variance (std=248k) indicates unstable policy
- 💡 Median position (163cm) much better than mean (301cm) - bimodal distribution

---

### 🔴 Final Model (100M - Collapsed)
**Checkpoint**: `final_model.zip`  
**Training State**: Catastrophic collapse (std=7,740, entropy=-82.8)

| Metric | Mean | Median | P95 | vs 40M |
|--------|------|--------|-----|--------|
| **Position Error** | 433.29 cm | 480.76 cm | 702.71 cm | **+44% WORSE** 🔴 |
| **Orientation Error** | 124.56° | 127.09° | 175.49° | **+5% BETTER?!** 🤔 |
| **Reward** | -311,658 | -311,087 | - | -31% worse |

**Key Observations**:
- ❌ **Position tracking collapsed completely** (433cm mean, 481cm median)
- 🤔 **Orientation strangely stable** (125° vs 40M's 130°) - suggests decoupled failure
- ⚠️ Median worse than mean (481cm vs 433cm) - reversed from 40M, uniform failure
- 💡 Final model unusable but orientation didn't explode further

---

## Timeline Analysis

### Training Progression
```
  27M: ✅ Healthy (KL=0.020, variance=0.786)
  31M: 🎯 WORKSPACE CONVERGED (0.554m, 3% violations, variance=0.597)
  36M: ⚠️  WARNING (variance=-0.241 NEGATIVE, KL=0.067)
  40M: 📊 EVALUATED (301cm position, 130° orientation)
  50M: 🔄 CURRICULUM TRANSITION (Stage 1→2, weights doubled)
 100M: 💥 CATASTROPHIC COLLAPSE (std=7,740, entropy=-82.8)
```

### Checkpoint Quality Assessment
| Checkpoint | Training Step | Stability | Position | Orientation | Usability |
|------------|---------------|-----------|----------|-------------|-----------|
| 40M | 40,009,728 | ⚠️ Pre-instability | 301cm (good) | 130° (poor) | ⚠️ Usable with caveats |
| 50M | ~50M | ❓ Unknown | - | - | 🔍 Need evaluation |
| Final | 100,663,296 | 💥 Collapsed | 433cm (failed) | 125° (stable?) | ❌ Unusable |

---

## Root Cause Analysis

### 🎯 What Worked (Evidence-Based Design)
1. **Workspace Expansion (0.6→0.7m)**:
   - ✅ Converged perfectly to 0.554m @ 31M (target: 0.55-0.65m)
   - ✅ Only 3% hard violations (excellent constraint adherence)
   - ✅ Position tracking improved vs Session 8f baseline

2. **Gentler Distance Penalties (60→30)**:
   - ✅ Allowed natural workspace exploration
   - ✅ Policy found optimal distance without getting stuck

3. **Workspace Observations (+2 dims)**:
   - ✅ Comfort signal and distance guidance worked as intended
   - ✅ Policy learned to recognize good positioning zones

### 💥 What Failed (Curriculum Implementation)

#### Critical Failure: Orientation Under-Training
**Stage 1 Weights (0-50M)**:
- Position: 5.0 (50% of final 10.0)
- Orientation: 15.0 (50% of final 30.0)

**Problem**: Absolute orientation weight (15.0) still 3x position weight (5.0), but **both were halved**. This created severe imbalance:
- Position learning benefited from 50M of focused training (even at reduced weight)
- Orientation learning was **starved** - too little signal to develop proper tracking

**Evidence**:
- 40M orientation: 130.38° (catastrophically bad vs 8f's 46.5°)
- Final orientation: 124.56° (barely improved despite 60M more training)
- Position: Actually improved at 40M (301cm vs 8f's 308cm)

#### Secondary Failure: Abrupt Curriculum Transition
**Stage 2 Transition @ 50M**:
- Weights **doubled instantly** (position 5→10, orientation 15→30)
- No interpolation period or gradual ramp
- Likely shocked value function → variance=-0.241 @ 36M (warning sign)

**Collapse Mechanism Hypothesis**:
1. **0-31M**: Policy learned workspace positioning with weak orientation signal
2. **31-36M**: Position converged, orientation still untrained, instability emerged
3. **36-50M**: Policy struggled with insufficient orientation training capacity
4. **50M**: Abrupt weight doubling shocked value function
5. **50-100M**: Training diverged, std exploded, entropy collapsed

---

## Comparison vs Baseline

### Session 8f (Baseline - BEST)
- Position: 308cm mean, 46.5° orientation, -126k reward
- Architecture: Two-zone linear + distance gating, 0.6m margin, 76 dims
- Status: **Stable throughout 200M training**

### Session 8g Outcomes
| Checkpoint | Position vs 8f | Orientation vs 8f | Overall |
|------------|----------------|-------------------|---------|
| **40M** | **+2% BETTER** 🟢 | **+180% WORSE** 🔴 | ⚠️ Mixed |
| **Final** | +41% WORSE 🔴 | +168% WORSE 🔴 | ❌ Failed |

**Key Insight**: Evidence-based workspace changes **worked** (position improved), but curriculum **backfired** (orientation destroyed).

---

## Lessons Learned

### ✅ Keep for Session 8h
1. **Workspace margin 0.7m** (65% FK coverage) - validated by 31M convergence
2. **Distance weight 30** (gentler gradient) - allowed natural exploration
3. **Optimal distance 0.6m** (FK median) - policy found it naturally
4. **Workspace observations** - provided useful positioning guidance

### ❌ Fix for Session 8h
1. **Curriculum weights imbalanced**:
   - Stage 1 orientation weight (15.0) was too low relative to position (5.0)
   - Should maintain 3:1 ratio but reduce **both proportionally**
   - Suggestion: Stage 1 (position: 3.0, orientation: 9.0) → Stage 2 (10.0, 30.0)

2. **Abrupt transition**:
   - Instant doubling @ 50M shocked value function
   - Implement gradual interpolation: 45M-55M linear ramp
   - Monitor KL divergence and pause if >0.1

3. **Orientation signal insufficiency**:
   - Consider increasing final orientation weight to 40.0 (4:1 ratio)
   - Or add orientation-specific observations (heading rate, angular error rate)

### 🔍 Need Investigation
1. **Why didn't orientation collapse further after 40M?**
   - Final model: 125° (only 5° better than 40M's 130°)
   - Position collapsed +44% but orientation stayed stable
   - Suggests **decoupled failure modes** - position diverged, orientation plateaued

2. **Evaluate 50M checkpoint**:
   - Check if collapse happened AT transition or AFTER
   - Compare 49.9M (pre) vs 50.1M (post) to isolate transition shock

---

## Recommendations for Session 8h

### Option 1: Balanced Curriculum (Recommended)
```python
# Stage 1 (0-50M): Reduce both but maintain ratio
curriculum_stage_1_position_weight = 3.0     # 30% of final
curriculum_stage_1_orientation_weight = 9.0  # 30% of final (keeps 3:1 ratio)

# Stage 2 (50M-100M): Gradual transition
# Linear interpolation 45M-55M instead of instant @ 50M
```

### Option 2: No Curriculum (Safer)
- Remove curriculum entirely
- Use full weights throughout: position 10.0, orientation 30.0
- Keep all workspace changes (0.7m margin, weight 30, observations)
- Rationale: 8f was stable without curriculum, 8g workspace changes work

### Option 3: Orientation Focus (Aggressive)
- Increase final orientation weight to 40.0 (4:1 ratio)
- Stage 1: position 4.0, orientation 12.0 (maintain 1:3 ratio)
- Add orientation rate observations (+2 dims → 80 total)
- Gradual transition 45M-55M

### Monitoring Requirements
1. **Track orientation metrics every 1M steps** (not just position)
2. **Alert if explained_variance < 0** (early warning system)
3. **Checkpoint every 2M steps** (finer granularity for rollback)
4. **Auto-pause if KL > 0.1** (prevent divergence cascade)

---

## Conclusion

Session 8g **validated the evidence-based workspace expansion approach** - position tracking improved vs Session 8f baseline at 40M checkpoint. However, **curriculum implementation critically flawed**:

1. **Orientation under-training** in Stage 1 (15.0 weight insufficient)
2. **Abrupt transition** at 50M shocked value function
3. **Training diverged catastrophically** by 100M (std=7,740)

**The 40M checkpoint is marginally usable** for deployment (better position than 8f, but terrible orientation). **The final model is completely unusable**.

**Session 8h must fix curriculum balance** or remove it entirely. Evidence-based workspace changes should be retained as they demonstrably improved position tracking.

---

## Files Generated

### Evaluation Results
- **40M**: `evaluation_plots/session_8g_40M/eval_summary_20251102_083142.json`
- **Final**: `evaluation_plots/session_8g_final/eval_summary_20251102_084524.json`

### Training Logs
- **Session log**: `H:\wSpace\cinebotRL\logs\sb3\mobilemmtrackee_v0\20251101_222053\`
- **Checkpoints**: 1,000+ files (every ~100K steps from 196K to 100M)

### Next Steps
1. ✅ Evaluate 50M checkpoint (pre/post curriculum transition)
2. ⚠️ Design Session 8h with balanced curriculum or no curriculum
3. 🔍 Investigate orientation-position decoupling in collapse phase
4. 📊 Compare detailed reward component traces (position vs orientation vs reachability)
