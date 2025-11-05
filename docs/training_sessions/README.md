# Training Sessions Documentation

This directory contains comprehensive documentation for all training sessions of the CinebotRL mobile manipulator project.

## 📂 Directory Structure

### Session-Specific Folders

Each session has its own folder containing all related documentation:

#### **session_8h/** ⭐ LATEST & BEST
Current best performing session (November 2025)
- **40M checkpoint: 237.3 cm** position error (best to date)
- Gradual curriculum transition with lower learning rate
- Files: Implementation plan, evaluation results, analysis
- **Status**: ✅ Complete, 40M recommended for deployment

#### **session_8g/**
Previous attempt with instant curriculum transition
- **40M checkpoint: 301 cm** position error  
- Diverged at 50M due to curriculum shock
- Files: Proposal, implementation, evaluation results
- **Status**: ✅ Complete, but outperformed by 8h

#### **session_8f/**
Baseline with adjusted weights
- **100M checkpoint: 308 cm** position error
- Files: Implementation, evaluation, comparison with 8cv2
- **Status**: ✅ Complete

#### **session_8e/**
Earlier session with different configuration
- Files: 50M and 100M evaluations, complete analysis
- **Status**: ✅ Complete

#### **session_8c/**
Session 8C series
- Files: Implementation, quick start, comparison with 8b
- **Status**: ✅ Complete

#### **session_8b/**
Session 8B series
- Files: Comparison with 8c
- **Status**: ✅ Complete

#### **session_7d/** & **session_7c/**
Legacy session 7 series
- **Status**: ✅ Complete (see folder for details)

#### **session_6/**
Session 6 series
- **Status**: ✅ Complete (see folder for details)

### Root-Level Documentation

Files at the root of `training_sessions/`:

- **SESSION_8_COMPARISON.md** - Cross-session comparison for Session 8 series
- **SESSION_FOLDER_MAPPING.md** - Mapping of old to new folder structure
- **TRAINING_DIARY.md** - Chronological training log
- **CONTROL_FREQUENCY_ANALYSIS.md** - Analysis of control frequency issues
- **20HZ_CONTROL_ANALYSIS.md** - 20Hz control update analysis
- **CODE_AUDIT_20HZ.md** - Code audit for 20Hz control
- **Various session status files** - Launch logs and status for older sessions

## 🏆 Performance Leaderboard

### Position Tracking Error (Target: <300 cm)

| Rank | Session | Checkpoint | Position Error | Status |
|------|---------|------------|----------------|---------|
| 🥇 1st | **Session 8h** | **40M** | **237.3 cm** | ✅ Best |
| 🥈 2nd | Session 8h | 20M | 296.8 cm | ✅ Pass |
| 🥉 3rd | Session 8g | 40M | 301.0 cm | ⚠️ Close |
| 4th | Session 8h | 100M | 302.4 cm | ❌ Fail |
| 5th | Session 8f | 100M | 308.0 cm | ❌ Fail |

**Winner**: Session 8h @ 40M (20.9% below target)

### Key Achievements
- ✅ First checkpoint to consistently beat 300cm target
- ✅ 23% improvement over Session 8f
- ✅ 21.2% improvement over Session 8g @ 40M

## 📊 Quick Reference

### Latest Results (Session 8h)

**Best Checkpoint**: 40M steps  
**Training Date**: November 3-4, 2025  
**Position Error**: 237.3 cm (20.9% below target)  
**Orientation Error**: 135.1° (still needs improvement)  
**Episode Reward**: -784.5 (best recorded)  

**Checkpoint File**: `logs/sb3/mobilemmtrackee_v0/20251103_235918/checkpoints/ppo_mobile_mm_40009728_steps.zip`

### Recommended Actions

**For Deployment**:
1. Export Session 8h @ 40M checkpoint to ONNX
2. Test on physical Cinebot system
3. Monitor real-world performance

**For Next Training (Session 8i)**:
1. Implement early stopping (stop at peak performance)
2. Evaluate 60M & 80M checkpoints to understand regression
3. Consider curriculum adjustments or stopping at stage-1
4. Address orientation tracking separately

## 📖 How to Use This Documentation

### Finding Information

1. **Latest Results**: Check `session_8h/README.md` and `session_8h/SESSION_8H_EVALUATION_RESULTS.md`
2. **Performance Comparison**: See `SESSION_8_COMPARISON.md`
3. **Training Configuration**: Check individual session folders for implementation details
4. **Chronological History**: See `TRAINING_DIARY.md`

### Session Folder Contents

Each session folder typically contains:
- **README.md** - Overview and quick reference
- **Implementation/Plan files** - Training configuration and setup
- **Results files** - Training completion summary
- **Evaluation files** - Quantitative evaluation results and analysis

## 🔗 Related Documentation

### Evaluation Plots
All visualization plots are in `evaluation_plots/` at the repository root:
- Individual session plots (tracking, joints, velocities, rewards)
- Cross-session comparisons
- See `evaluation_plots/session_8h_README.md` for comprehensive plot documentation

### Training Scripts
All training and evaluation scripts in `scripts/`:
- `launch_session_8*.ps1` - PowerShell training launchers
- `scripts/reinforcement_learning/sb3/train.py` - Main training script
- `scripts/reinforcement_learning/sb3/evaluate_*.py` - Evaluation scripts

### Master Logs
- **TRAINING_SESSIONS_MASTER_LOG.md** (repository root) - Complete training history
- **ROADMAP.md** (repository root) - Project roadmap and future plans

## 📋 Session Naming Convention

- **Session 8x**: Main training series with different configurations
  - 8a-8d: Early experiments
  - 8e: Configuration refinement
  - 8f: Adjusted weights baseline
  - 8g: Instant curriculum transition (diverged)
  - 8h: Gradual curriculum transition (current best)
- **Session 7x**: Previous training series
- **Session 6 and earlier**: Legacy sessions

## 🚀 Next Steps

Based on Session 8h results:

1. **Immediate**: Deploy 40M checkpoint to robot
2. **Analysis**: Evaluate 60M/80M to understand regression timeline
3. **Planning**: Design Session 8i with early stopping mechanism
4. **Research**: Investigate orientation tracking improvements

---

**Last Updated**: November 5, 2025  
**Current Best**: Session 8h @ 40M (237.3 cm position error)  
**Status**: Active development, ready for deployment testing
