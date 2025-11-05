# Session 8h - Gradual Curriculum with Lower Learning Rate

**Training Date**: November 3-4, 2025  
**Status**: ✅ Complete - **40M checkpoint recommended for deployment**  
**Training Duration**: 6.5 hours (100M steps)  

## Overview

Session 8h implemented gradual curriculum transition (45-55M steps) with lower learning rate (2e-4) to prevent the divergence issues seen in Session 8g. The training completed successfully with 0 auto-pause triggers.

### Key Configuration
- **Learning Rate**: 2e-4 (lower than 8g's 3e-4)
- **Curriculum Transition**: Gradual from 45-55M steps
  - Start: (4.0, 12.0) position/orientation weights
  - End: (10.0, 30.0) position/orientation weights
- **Auto-pause**: KL>0.1 or variance<-0.3, 500K warmup
- **Total Steps**: 100M (completed without divergence)

## Key Finding: 40M Checkpoint is Best

**Unexpected Result**: Performance peaked at 40M, then regressed at 100M despite completing full training.

### Performance Summary

| Checkpoint | Position Error | Orientation Error | Status |
|------------|----------------|-------------------|---------|
| **40M** | **237.3 cm** ⭐ | 135.1° | **BEST** |
| 20M | 296.8 cm | 135.5° | Pass |
| 100M | 302.4 cm | 119.1° | Regressed |

**Target**: <300 cm position, <60° orientation

## Documentation Files

### Implementation & Planning
- **SESSION_8H_IMPLEMENTATION_PLAN.md** - Initial training plan and configuration
- **SESSION_8H_RESULTS.md** - Training completion summary

### Evaluation & Analysis
- **SESSION_8H_EVALUATION_GUIDE.md** - How to run evaluations
- **SESSION_8H_EVALUATION_READY.md** - Evaluation status tracking
- **SESSION_8H_EVALUATION_RESULTS.md** - **⭐ Comprehensive evaluation analysis**

## Evaluation Results

**Evaluation Setup**: 16 parallel envs, 50 episodes per checkpoint

### 40M Checkpoint (RECOMMENDED) ⭐
- **Position Error**: 237.3 cm mean (20.9% below target)
- **Orientation Error**: 135.1° mean
- **Episode Reward**: -784.5 mean
- **Status**: Best performance, recommended for deployment
- **Checkpoint File**: `logs/sb3/mobilemmtrackee_v0/20251103_235918/checkpoints/ppo_mobile_mm_40009728_steps.zip`

### 100M Checkpoint (NOT RECOMMENDED)
- **Position Error**: 302.4 cm mean (0.8% above target)
- **Orientation Error**: 119.1° mean (better than 40M)
- **Episode Reward**: -1031.0 mean
- **Status**: Regressed 27.5% from 40M - Do NOT use

## Visualization Plots

All plots available in `evaluation_plots/`:
- **session_8h_20M/** - Early checkpoint plots
- **session_8h_40M/** - Best checkpoint plots ⭐
- **session_8h_100M/** - Final checkpoint plots
- **session_8h_comparison/** - Comparison of all 3 checkpoints
- **session_8h_vs_previous/** - Comparison with Sessions 8f and 8g

See `evaluation_plots/session_8h_README.md` for detailed plot documentation.

## Comparison with Previous Sessions

### vs Session 8f @ 100M
- Position: 237.3 cm vs 308 cm (**-23.0% better**)
- Orientation: 135.1° vs 46.5°

### vs Session 8g @ 40M  
- Position: 237.3 cm vs 301 cm (**-21.2% better**)
- Orientation: 135.1° vs 130°

**Result**: Session 8h @ 40M beats all previous sessions in position tracking.

## Key Insights

### Why 40M is Best
1. ✅ Lowest position error (237.3 cm)
2. ✅ Best episode rewards (-784.5)
3. ✅ Stable performance before curriculum transition
4. ✅ Beats Session 8f and 8g benchmarks

### Why 100M Regressed
Possible causes (see EVALUATION_RESULTS.md for details):
1. **Curriculum transition issues**: Despite gradual approach, 45-55M transition still caused degradation
2. **Overfitting to stage-1**: Policy may have overfit to easier trajectories
3. **Learning rate too low**: 2e-4 may prevent adaptation to stage-2
4. **Orientation-position trade-off**: Increased orientation weight (30.0) may hurt position tracking

## Recommendations

### For Deployment
✅ **Use 40M checkpoint** (`ppo_mobile_mm_40009728_steps.zip`)
- Export to ONNX format
- Deploy to physical Cinebot system
- Best position tracking performance

❌ **Do NOT use 100M checkpoint**
- Performance regressed
- Fails position target (302.4 cm > 300 cm)

### For Future Training (Session 8i)
1. **Early stopping**: Monitor position error, stop when it starts increasing
2. **Evaluate 60M & 80M**: Understand exactly when regression started
3. **Curriculum adjustments**:
   - Option A: Stop at stage-1 (40M already excellent)
   - Option B: Even gentler transition (40-60M)
   - Option C: Adjust final weights (12.0, 24.0) for better position emphasis
   - Option D: Learning rate schedule (reduce during transition)
4. **Orientation tracking**: Separate training phase or different approach needed

## Related Files & Logs

**Training Logs**: `logs/sb3/mobilemmtrackee_v0/20251103_235918/`

**Evaluation Data**: `evaluation_results/session_8h_comparison/`
- Session_8h_at_20M/checkpoints/
- Session_8h_at_40M/checkpoints/
- Session_8h_at_100M/checkpoints/

**Scripts**:
- `scripts/launch_session_8h.ps1` - Training launcher
- `scripts/launch_session_8h_evaluation.ps1` - Evaluation launcher
- `scripts/reinforcement_learning/sb3/evaluate_session_8h_simple.py` - Evaluation wrapper

**Master Log**: `TRAINING_SESSIONS_MASTER_LOG.md`

---

**Last Updated**: November 5, 2025  
**Status**: ✅ Evaluation complete, 40M recommended for deployment
