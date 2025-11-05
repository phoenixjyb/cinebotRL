# Session 8h Checkpoint Evaluation Results

**Evaluation Date**: November 4, 2025  
**Session**: Session 8h (100M steps, gradual curriculum 45-55M, LR 2e-4)  
**Evaluation Setup**: 16 parallel environments, 50 episodes per checkpoint  

## 📊 Visualization Plots

All evaluation plots are available in the `evaluation_plots/` directory:

### Individual Checkpoint Analysis
- **Session 8h @ 20M**: `evaluation_plots/session_8h_20M/`
- **Session 8h @ 40M**: `evaluation_plots/session_8h_40M/` ⭐ **Best Performance**
- **Session 8h @ 100M**: `evaluation_plots/session_8h_100M/`

Each directory contains:
- `tracking_errors.png` - Position and orientation error distributions
- `joint_angles.png` - Joint angle usage and limits
- `joint_velocities.png` - Joint velocity profiles
- `base_velocities.png` - Mobile base velocity analysis
- `reward_components.png` - Breakdown of reward contributions
- `episode_statistics.png` - Episode length and reward trends

### Comparison Plots
- **Session 8h Checkpoints**: `evaluation_plots/session_8h_comparison/`
  - Compares 20M vs 40M vs 100M checkpoints
- **Cross-Session Comparison**: `evaluation_plots/session_8h_vs_previous/`
  - Compares Session 8f, 8g, and all 8h checkpoints

---

## Executive Summary

Session 8h evaluation reveals **unexpected performance regression** at 100M compared to earlier checkpoints. The **40M checkpoint shows the best overall performance** with lowest position error, suggesting potential overfitting or curriculum transition issues in later training.

### Key Findings

✅ **40M checkpoint is the best performer**: Position error 237.3 cm (20% better than target)  
⚠️ **100M shows regression**: Position error 302.4 cm (worse than 40M by 27%)  
⚠️ **Orientation remains challenging**: All checkpoints >100° (target was <60°)  
✅ **Early checkpoint (20M) competitive**: Position 296.8 cm, near 100M performance

---

## Detailed Metrics Comparison

### Position Tracking Error (Target: <300 cm)

| Checkpoint | Mean (cm) | Median (cm) | P95 (cm) | Status | vs Target |
|------------|-----------|-------------|----------|---------|-----------|
| **20M** | **296.8** | 255.3 | 645.3 | ✅ Pass | +1.1% below |
| **40M** | **237.3** | 206.7 | 590.4 | ✅✅ Best | **-20.9% below** |
| **100M** | **302.4** | 256.8 | 706.8 | ❌ Fail | +0.8% above |

**Analysis**:
- ✅ **40M is the best checkpoint**: 237.3 cm mean error (20.9% below target)
- ⚠️ **100M regressed**: 302.4 cm (27.5% worse than 40M, fails target)
- ✅ **20M competitive**: 296.8 cm (only 1.1% better than target, near 100M)
- **Curriculum impact unclear**: No obvious improvement from 40M → 100M transition

### Orientation Tracking Error (Target: <60°)

| Checkpoint | Mean (°) | Median (°) | P95 (°) | Status | vs Target |
|------------|----------|------------|---------|---------|-----------|
| **20M** | 135.5 | 144.4 | 174.2 | ❌ Poor | +125.9% |
| **40M** | 135.1 | 142.3 | 172.8 | ❌ Poor | +125.2% |
| **100M** | 119.1 | 118.3 | 177.0 | ❌ Poor | +98.5% |

**Analysis**:
- ❌ **All checkpoints fail orientation target** (>100° vs 60° target)
- ✅ **100M shows improvement**: 119.1° (11.8% better than 20M/40M)
- ⚠️ **Still far from target**: 98.5% above 60° target
- **Problem**: Orientation tracking fundamentally underperforming

### Episode Rewards

| Checkpoint | Mean Reward | Std | Min | Max |
|------------|-------------|-----|-----|-----|
| **20M** | -1069.9 | 520.7 | -2395.4 | -217.8 |
| **40M** | -784.5 | 420.1 | -2043.5 | -127.3 |
| **100M** | -1031.0 | 489.4 | -2346.6 | -254.2 |

**Analysis**:
- ✅ **40M has highest mean reward**: -784.5 (26.7% better than 100M)
- ⚠️ **100M reward regression**: -1031.0 (31.4% worse than 40M)
- **Correlation with position error**: Best position → best reward (40M)

---

## Comparison with Baselines

### vs Session 8f (100M, instant curriculum @ 50M)
- Position: 308 cm, Orientation: 46.5°

| Checkpoint | Position | vs 8f | Orientation | vs 8f |
|------------|----------|-------|-------------|-------|
| 20M | 296.8 cm | ✅ **-3.6%** | 135.5° | ❌ +191.4% |
| **40M** | **237.3 cm** | ✅ **-23.0%** | 135.1° | ❌ +190.5% |
| 100M | 302.4 cm | ❌ +1.9% | 119.1° | ❌ +156.1% |

### vs Session 8g @ 40M (last stable before collapse)
- Position: 301 cm, Orientation: 130°

| Checkpoint | Position | vs 8g@40M | Orientation | vs 8g@40M |
|------------|----------|-----------|-------------|-----------|
| 20M | 296.8 cm | ✅ **-1.4%** | 135.5° | ❌ +4.2% |
| **40M** | **237.3 cm** | ✅ **-21.2%** | 135.1° | ❌ +3.9% |
| 100M | 302.4 cm | ❌ +0.5% | 119.1° | ✅ **-8.4%** |

---

## Training Progression Analysis

### Position Error Trend (20M → 40M → 100M)

```
300cm ┤      ╭─────────────100M (302.4)
      │     ╱
      │    ╱
280cm ┤   ╱
      │  ╱          20M (296.8)
260cm ┤ ╱
      │╱
240cm ┼────────────40M (237.3) ← BEST
      └────────────────────────────────
       20M      40M        100M
```

**Observations**:
1. ✅ **20M → 40M**: Strong improvement (-20.0%, -59.5 cm)
2. ❌ **40M → 100M**: Significant regression (+27.5%, +65.1 cm)
3. **Curriculum transition (45-55M)**: No visible benefit
4. **Best performance at 40M**: Suggests overfitting or curriculum issues

### Orientation Error Trend

```
140° ┤ 20M (135.5)──40M (135.1)
     │                            
120° ┤                      ╲     
     │                       ╲    
100° ┤                        100M (119.1)
     └────────────────────────────────
      20M      40M        100M
```

**Observations**:
1. ✅ **40M → 100M**: Orientation improved (-11.8%, -16.0°)
2. ❌ **Still far from target**: 119° vs 60° target
3. **Trade-off**: Position got worse while orientation improved

---

## Reward Component Analysis (40M - Best Checkpoint)

| Component | Mean | Analysis |
|-----------|------|----------|
| **Position tracking** | 0.66 | Good tracking performance |
| **Orientation tracking** | 0.92 | Moderate orientation penalty |
| **Progress bonus** | 0.02 | Low - suggests slow progress |
| **Base mobilization** | 0.44 | Chassis moving appropriately |
| **Reachability bonus** | 1.98 | Strong arm workspace usage |
| **Reachability distance penalty** | -110.5 | Moderate unreachable targets |
| **Position distance penalty** | -164.2 | Position error penalty |
| **Base overshoot penalty** | -6.5 | Chassis overshooting targets |

**Key Issues**:
1. ❌ **High position distance penalty**: -164.2 dominates reward
2. ❌ **High reachability distance penalty**: -110.5 suggests unreachable targets
3. ✅ **Reachability bonus**: +1.98 shows good arm usage
4. ⚠️ **Base overshoot**: -6.5 chassis moving too far

---

## Root Cause Analysis

### Why did 100M regress from 40M?

**Hypothesis 1: Curriculum Transition Issues (45-55M)**
- ✅ **Evidence**: Best performance at 40M (before transition)
- ✅ **Evidence**: Regression at 100M (after transition)
- **Mechanism**: Gradual transition (10.0, 30.0) may still shock policy
- **Recommendation**: Even gentler transition or pause training @ 40M

**Hypothesis 2: Overfitting to Early Trajectories**
- ✅ **Evidence**: Strong performance at 40M, then decay
- **Mechanism**: Policy overfits to easier stage-1 curriculum
- **Recommendation**: More diverse trajectories in stage-1

**Hypothesis 3: Learning Rate Too Low**
- ✅ **Evidence**: 2e-4 LR (Session 8h) vs 3e-4 (Session 8g)
- ⚠️ **Evidence**: 40M great but couldn't improve further
- **Mechanism**: Low LR prevents adaptation to stage-2 curriculum
- **Recommendation**: Increase LR during curriculum transition

**Hypothesis 4: Orientation-Position Trade-off**
- ✅ **Evidence**: Orientation improved 20M→100M while position regressed
- **Mechanism**: Curriculum weights (10.0, 30.0) may over-emphasize orientation
- **Recommendation**: Adjust final curriculum weights balance

---

## Recommendations

### Immediate Actions

1. **✅ Use 40M checkpoint for deployment**
   - Best position error: 237.3 cm (-20.9% below target)
   - Stable performance, no regression
   - Best overall reward: -784.5

2. **❌ Do not use 100M checkpoint**
   - Position error: 302.4 cm (+0.8% above target)
   - 27.5% worse than 40M
   - Reward regression: -1031.0 vs -784.5

3. **✅ Investigate 60M and 80M checkpoints**
   - May reveal when regression started
   - Could find better checkpoint than 40M
   - Helps understand curriculum transition impact

### Training Improvements for Future Sessions

**Session 8i Recommendations**:

1. **Pause training at 40M if performance peaks**
   - Monitor position error during training
   - Implement early stopping when error increases
   - Avoid unnecessary training that causes regression

2. **Adjust curriculum transition**
   - Option A: Stop at stage-1 (40M performs well)
   - Option B: Even gentler transition (40-60M instead of 45-55M)
   - Option C: Adaptive transition based on performance metrics

3. **Adjust final curriculum weights**
   - Current: (10.0, 30.0) - 1:3 ratio
   - Proposed: (12.0, 24.0) - 1:2 ratio (more position emphasis)
   - Rationale: Position regressed while orientation improved

4. **Consider learning rate schedule**
   - Stage 1 (0-40M): 2e-4 (current, works well)
   - Transition (45-55M): 1e-4 (gentler adaptation)
   - Stage 2 (55-100M): 2e-4 (resume normal learning)

5. **Add more trajectory diversity in stage-1**
   - Current: May be overfitting to easier trajectories
   - Proposed: Mix some stage-2 style trajectories even in stage-1
   - Rationale: Better generalization to harder targets

---

## Comparison Summary Table

| Metric | 20M | 40M | 100M | Target | Best |
|--------|-----|-----|------|--------|------|
| **Position (cm)** | 296.8 | **237.3** ✅ | 302.4 | <300 | **40M** |
| **Orientation (°)** | 135.5 | 135.1 | **119.1** | <60 | **100M** |
| **Reward** | -1069.9 | **-784.5** ✅ | -1031.0 | Higher | **40M** |
| **vs 8f (308cm, 46°)** | Better pos, worse ori | **Best pos**, worse ori | Worse pos, worse ori | - | **40M** |
| **vs 8g@40M (301cm, 130°)** | Better pos, worse ori | **Best pos**, worse ori | Worse pos, better ori | - | **40M** |

---

## Conclusion

Session 8h shows **unexpected regression from 40M to 100M**, contradicting the hypothesis that gradual curriculum transition would improve final performance. The **40M checkpoint is the clear winner** with:

✅ **Best position error**: 237.3 cm (20.9% below target)  
✅ **Best reward**: -784.5 (26.7% better than 100M)  
✅ **Best vs baselines**: -23.0% vs Session 8f, -21.2% vs Session 8g  

However, **orientation tracking remains a fundamental issue** across all checkpoints (>100° vs 60° target), suggesting the reward function or training approach needs rethinking for orientation control.

**Next Steps**:
1. Deploy 40M checkpoint as best available model
2. Evaluate 60M/80M to understand regression timeline
3. Run Session 8i with early stopping or modified curriculum
4. Consider separate training phases for position vs orientation

---

**Files**:
- 20M detailed results: `evaluation_results/session_8h_comparison/Session_8h_at_20M/checkpoints/eval_summary_20251104_094503.json`
- 40M detailed results: `evaluation_results/session_8h_comparison/Session_8h_at_40M/checkpoints/eval_summary_20251104_095126.json`
- 100M detailed results: `evaluation_results/session_8h_comparison/Session_8h_at_100M/checkpoints/eval_summary_20251104_095859.json`
