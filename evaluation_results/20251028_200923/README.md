# Evaluation Results - Session 7d (200M Timesteps)

**Date**: October 29, 2025  
**Checkpoint**: `logs\sb3\mobilemmtrackee_v0\20251028_200923\final_model.zip`  
**Episodes**: 200 | **Environments**: 64 parallel | **Trajectories**: 1,038 total

---

## 📁 Files in this Directory

### Data Files
- **`eval_summary_20251029_131728.json`** - Complete evaluation statistics (tracking errors, rewards, episodes)
- **`episodes_20251029_131728.csv`** - Per-episode data (200 rows)
- **`steps_20251029_131728.csv`** - Time-series step data (sampled every 10 steps)
- **`arrays_20251029_131728.npz`** - Raw numpy arrays for custom analysis

### Analysis
- **`ANALYSIS_REPORT.md`** - ⭐ **Comprehensive 2000+ word analysis report with findings and recommendations**

### Visualizations (../evaluation_plots/)
- `tracking_errors.png` - Position and orientation error distributions
- `joint_angles.png` - Joint angle utilization
- `joint_velocities.png` - Velocity profiles for all joints
- `reward_components.png` - Breakdown of reward components
- `episode_statistics.png` - Episode reward and length distributions
- `evaluation_report.txt` - Text summary with assessment

---

## 🎯 Quick Summary

### Overall Assessment: ⚠️ **NOT READY FOR DEPLOYMENT**

**Position Tracking**:
- Mean error: **3.64m** (364 cm) ❌ Target: < 20 cm
- Median: 1.20m (120 cm)
- Best case: 6.3 cm ✅
- Worst case: 19.22m ❌

**Orientation Tracking**:
- Mean error: **140.7°** ❌ Target: < 10°
- Robot pointing nearly **opposite direction** on average
- Policy learned to ignore orientation

**Rewards**:
- Mean: -5,120
- Penalties (velocity, jerk) overwhelm tracking rewards
- Reward imbalance is root cause

---

## 🔍 Key Findings

1. **Reward Imbalance**: Penalties (-40/step) >> Rewards (+0.07/step)
2. **Orientation Ignored**: Weight too small (0.19 vs 27.7 for position)
3. **Base Immobility**: Robot barely moves base (< 0.01 m/s)
4. **Velocity Violations**: Joint 5 hitting 4.0 rad/s limits constantly
5. **Outliers**: Some trajectories fail catastrophically (12-19m error)

---

## 🔧 Recommendations

**Priority 1: Retrain with Fixed Reward Weights**
```python
orientation_tracking_weight: 50.0  # Was: ~1.0 (increase 50×)
velocity_limit_penalty_weight: 5.0  # Was: 15-20 (reduce 3×)
jerk_penalty_weight: 2.0  # Was: 10-15 (reduce 5-7×)
base_mobilization_weight: 10.0  # Was: 1.0 (increase 10×)
```

**Priority 2: Curriculum Learning**
- Stage 1: Static targets (50M steps)
- Stage 2: Slow-moving (50M steps)
- Stage 3: Full cinematic (100M steps)

**Priority 3: Architecture Changes**
- Consider hierarchical control (base + arm)
- Add visual servoing for fine control
- Improve base-arm coordination

---

## 📊 Success Criteria for Next Training

**Minimum for deployment:**
- Mean position error: **< 20 cm** (currently 364 cm ❌)
- Mean orientation error: **< 10°** (currently 141° ❌)
- Velocity penalty: **< 2.0** (currently 15.5 ❌)
- Jerk penalty: **< 1.0** (currently 14.0 ❌)
- Success rate: **> 75%** (currently ~50% ❌)

---

## 🎬 Conclusion

The evaluation system is **working perfectly** ✅ - all metrics captured correctly. However, the **policy requires retraining** with corrected reward weights before deployment.

**Next Steps**:
1. ⏸️ **DO NOT deploy to Orin**
2. 🔧 **Retrain with fixed reward weights**
3. 📊 **Re-evaluate after retraining**
4. 🎯 **Deploy only after meeting criteria**

---

## 📚 Related Files

- **Evaluation System**: `scripts/reinforcement_learning/sb3/evaluate_quantitative.py`
- **Visualization**: `scripts/reinforcement_learning/sb3/visualize_eval_results.py`
- **Launcher**: `scripts/launch_evaluation_quantitative.ps1`
- **Documentation**: `scripts/reinforcement_learning/sb3/EVALUATION_README.md`
- **Environment Fix**: `src/rl_platform/tasks/mobile_mm/env.py` (lines 420-422, 1588-1595)
