# Session 8h Evaluation Plots

**Generated**: November 5, 2025  
**Session**: Session 8h (100M steps, gradual curriculum 45-55M, LR 2e-4)  
**Evaluation**: 16 parallel environments, 50 episodes per checkpoint  

## Overview

This directory contains comprehensive visualization plots for Session 8h checkpoint evaluation. Three checkpoints were evaluated (20M, 40M, 100M) to understand training progression and identify the best performing model.

### Key Finding

**The 40M checkpoint is the best performer** with 237.3 cm position error (20.9% below target), while the 100M final checkpoint regressed to 302.4 cm (27.5% worse than 40M).

---

## Directory Structure

### Individual Checkpoint Plots

#### 📁 `session_8h_20M/`
Early training checkpoint (20M steps)
- **tracking_errors.png**: Position error 296.8 cm (mean), orientation 135.5°
- **joint_angles.png**: Joint angle distributions and usage patterns
- **joint_velocities.png**: Joint velocity profiles (all within limits)
- **base_velocities.png**: Mobile base velocity analysis
- **reward_components.png**: Breakdown showing reachability penalties dominate
- **episode_statistics.png**: Episode rewards (-1069.9 mean) and lengths

**Performance**: ✅ Passes position target (296.8 cm < 300 cm), competitive with 100M

---

#### 📁 `session_8h_40M/` ⭐ **BEST CHECKPOINT**
Mid-training checkpoint (40M steps) - **Recommended for deployment**
- **tracking_errors.png**: Position error 237.3 cm (mean), orientation 135.1°
- **joint_angles.png**: Well-distributed joint usage
- **joint_velocities.png**: Smooth velocity profiles
- **base_velocities.png**: Efficient base motion patterns
- **reward_components.png**: Best balance across all components
- **episode_statistics.png**: Best episode rewards (-784.5 mean)

**Performance**: ✅✅ Best overall - 237.3 cm (20.9% below target), highest rewards

---

#### 📁 `session_8h_100M/`
Final checkpoint after full training (100M steps)
- **tracking_errors.png**: Position error 302.4 cm (mean), orientation 119.1°
- **joint_angles.png**: Joint usage patterns similar to earlier checkpoints
- **joint_velocities.png**: Velocity profiles comparable to 40M
- **base_velocities.png**: Base motion slightly more conservative
- **reward_components.png**: Increased penalties vs 40M
- **episode_statistics.png**: Regressed rewards (-1031.0 mean)

**Performance**: ❌ Regression - 302.4 cm (0.8% above target), 27.5% worse than 40M

---

## Comparison Plots

### 📁 `session_8h_comparison/`
Direct comparison of Session 8h checkpoints (20M, 40M, 100M)

#### Plots:
- **session_comparison_tracking.png**
  - Position error progression: 296.8 → 237.3 → 302.4 cm (improvement then regression)
  - Orientation error: 135.5 → 135.1 → 119.1° (gradual improvement)
  - Shows clear performance peak at 40M

- **session_comparison_rewards.png**
  - Episode rewards: -1069.9 → -784.5 → -1031.0 (peak at 40M)
  - Reward components breakdown across checkpoints
  - Highlights curriculum transition impact

- **session_comparison_joints.png**
  - Joint velocity comparison across checkpoints
  - Joint usage patterns evolution
  - Shows consistent behavior across all checkpoints

- **comparison_report.txt**
  - Detailed numerical comparison table
  - Statistical summary for each metric

---

### 📁 `session_8h_vs_previous/`
Cross-session comparison: Session 8f, 8g, and all 8h checkpoints

#### Sessions Compared:
- **8f @ 100M**: Baseline session (308 cm position, 46.5° orientation)
- **8g @ 40M**: Previous best attempt (301 cm position, 130° orientation)
- **8h @ 20M**: Early 8h checkpoint (296.8 cm)
- **8h @ 40M**: Best 8h checkpoint (237.3 cm) ⭐
- **8h @ 100M**: Final 8h checkpoint (302.4 cm)

#### Plots:
- **session_comparison_tracking.png**
  - Shows Session 8h @ 40M beats all previous sessions
  - Position error: 8h@40M (237.3cm) < 8h@20M (296.8cm) < 8g@40M (301cm) < 8h@100M (302.4cm) < 8f@100M (308cm)
  - Demonstrates effectiveness of gradual curriculum at 40M checkpoint

- **session_comparison_rewards.png**
  - Episode reward trends across sessions
  - 8h @ 40M achieves best rewards (-784.5)
  - Shows reward degradation in later training

- **session_comparison_joints.png**
  - Joint behavior comparison across sessions
  - Similar joint usage patterns across all sessions
  - No significant joint velocity differences

- **comparison_report.txt**
  - Comprehensive session comparison table
  - Performance metrics vs baselines

---

## Key Insights from Plots

### 1. **Unexpected Regression Pattern**
- Position error: 296.8 → **237.3** → 302.4 cm (40M is best)
- Episode reward: -1069.9 → **-784.5** → -1031.0 (40M is best)
- Suggests curriculum transition (45-55M) caused performance degradation

### 2. **Orientation Still Challenging**
- All checkpoints: >100° vs 60° target
- 100M shows slight improvement: 119.1° vs 135.1° at 40M
- But position error worsened significantly
- May indicate position-orientation trade-off in curriculum

### 3. **Joint and Velocity Consistency**
- All checkpoints show similar joint usage patterns
- Velocities remain within safe limits across all checkpoints
- Base motion consistent across training progression

### 4. **Reward Component Analysis**
- Reachability distance penalty is largest component (negative)
- Position distance penalty second largest
- 40M checkpoint achieves best balance of all components
- 100M shows increased penalties, especially reachability

---

## Recommendations

### For Deployment
✅ **Use Session 8h @ 40M checkpoint**
- Checkpoint file: `logs/sb3/mobilemmtrackee_v0/20251103_235918/checkpoints/ppo_mobile_mm_40009728_steps.zip`
- Performance: 237.3 cm position error (20.9% below target)
- Best episode rewards: -784.5 mean
- Stable and reliable tracking

❌ **Do NOT use 100M checkpoint**
- Performance regressed: 302.4 cm (fails target)
- Worse rewards: -1031.0 (31.4% worse than 40M)
- Not suitable for robot deployment

### For Future Training (Session 8i)
1. **Implement early stopping**: Monitor position error, stop when it starts increasing
2. **Evaluate 60M and 80M**: Understand exactly when regression started
3. **Adjust curriculum**: Consider:
   - Stopping at stage-1 (40M already excellent)
   - Even gentler transition (40-60M instead of 45-55M)
   - Adjusted final weights: (12.0, 24.0) vs current (10.0, 30.0)
   - Learning rate schedule: reduce during transition
4. **Address orientation**: Separate training phase or different approach

---

## Plot Generation Commands

### Individual checkpoint plots:
```powershell
# 20M checkpoint
I:\isaaclab\isaaclab.bat -p scripts/reinforcement_learning/sb3/visualize_eval_results.py `
    --input "evaluation_results/session_8h_comparison/Session_8h_at_20M/checkpoints/eval_summary_20251104_094503.json" `
    --output_dir "evaluation_plots/session_8h_20M"

# 40M checkpoint
I:\isaaclab\isaaclab.bat -p scripts/reinforcement_learning/sb3/visualize_eval_results.py `
    --input "evaluation_results/session_8h_comparison/Session_8h_at_40M/checkpoints/eval_summary_20251104_094955.json" `
    --output_dir "evaluation_plots/session_8h_40M"

# 100M checkpoint
I:\isaaclab\isaaclab.bat -p scripts/reinforcement_learning/sb3/visualize_eval_results.py `
    --input "evaluation_results/session_8h_comparison/Session_8h_at_100M/checkpoints/eval_summary_20251104_095428.json" `
    --output_dir "evaluation_plots/session_8h_100M"
```

### Comparison plots:
```powershell
# Session 8h checkpoints comparison
I:\isaaclab\isaaclab.bat -p scripts/reinforcement_learning/sb3/compare_sessions.py `
    --sessions `
        "8h_20M:evaluation_results/session_8h_comparison/Session_8h_at_20M/checkpoints/eval_summary_20251104_094503.json" `
        "8h_40M:evaluation_results/session_8h_comparison/Session_8h_at_40M/checkpoints/eval_summary_20251104_094955.json" `
        "8h_100M:evaluation_results/session_8h_comparison/Session_8h_at_100M/checkpoints/eval_summary_20251104_095428.json" `
    --output_dir "evaluation_plots/session_8h_comparison"

# Cross-session comparison
I:\isaaclab\isaaclab.bat -p scripts/reinforcement_learning/sb3/compare_sessions.py `
    --sessions `
        "8f_100M:evaluation_plots/session_8f_100M/20251101_013539/eval_summary_20251101_151551.json" `
        "8g_40M:evaluation_plots/session_8g_40M/checkpoints/eval_summary_20251102_083142.json" `
        "8h_20M:evaluation_results/session_8h_comparison/Session_8h_at_20M/checkpoints/eval_summary_20251104_094503.json" `
        "8h_40M:evaluation_results/session_8h_comparison/Session_8h_at_40M/checkpoints/eval_summary_20251104_094955.json" `
        "8h_100M:evaluation_results/session_8h_comparison/Session_8h_at_100M/checkpoints/eval_summary_20251104_095428.json" `
    --output_dir "evaluation_plots/session_8h_vs_previous"
```

---

## Related Documentation

- **Full Analysis**: `docs/training_sessions/SESSION_8H_EVALUATION_RESULTS.md`
- **Training Plan**: `docs/training_sessions/SESSION_8H.md` (if exists)
- **Master Log**: `TRAINING_SESSIONS_MASTER_LOG.md`

---

**Last Updated**: November 5, 2025  
**Status**: ✅ Evaluation complete, plots generated, 40M recommended for deployment
