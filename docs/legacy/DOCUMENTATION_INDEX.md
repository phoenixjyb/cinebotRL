# 📚 cinebotRL Documentation Index

**Quick access to all project documentation**

---

## 🚀 Quick Start (START HERE!)

### I want to train my robot RIGHT NOW
```powershell
# Phase 1: Conservative (2x speedup) - UPDATED for proper iterative learning
cd I:\isaaclab
.\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py `
  --task MobileMMTrackEE-v0 `
  --num_envs 2048 `
  --batch_size 512 `
  --n_steps 32 `
  --headless

# Monitor GPU
nvidia-smi -l 1
```

**Expected**: 2x faster, 5-7GB GPU usage, ~18 mins for 10M steps, 153 iterations (proper learning)

📖 **Full guide**: [`docs_archive/04_gpu_optimization/RTX3090_REFERENCE_CARD.md`](docs_archive/04_gpu_optimization/RTX3090_REFERENCE_CARD.md)

---

## 📂 Documentation Structure

### [`docs_archive/`](docs_archive/) - Complete Documentation Archive
All project documentation organized by topic:

1. **[`01_project_overview/`](docs_archive/01_project_overview/)** - Project status, roadmap, lessons
2. **[`02_bug_fixes/`](docs_archive/02_bug_fixes/)** - Bug analysis and fixes (all resolved ✅)
3. **[`03_training_guides/`](docs_archive/03_training_guides/)** - Training workflows and tuning
4. **[`04_gpu_optimization/`](docs_archive/04_gpu_optimization/)** - RTX 3090 optimization (latest work)
5. **[`05_historical_analysis/`](docs_archive/05_historical_analysis/)** - Historical reference

📖 **Full index**: [`docs_archive/README.md`](docs_archive/README.md)

---

## 🎯 Common Tasks

### Training
| Task | Document | Location |
|------|----------|----------|
| Quick start command | RTX3090_REFERENCE_CARD.md | `docs_archive/04_gpu_optimization/` |
| Full training guide | START_TRAINING_NOW.md | `docs_archive/03_training_guides/` |
| Hyperparameter tuning | TRAINING_TUNING_PLAN.md | `docs_archive/03_training_guides/` |
| CPU training | CPU_TRAINING_GUIDE.md | `docs_archive/03_training_guides/` |
| Visualization | VISUALIZATION_QUICK_START.md | `docs_archive/03_training_guides/` |

### GPU Optimization
| Task | Document | Location |
|------|----------|----------|
| **Quick reference** ⭐ | RTX3090_REFERENCE_CARD.md | `docs_archive/04_gpu_optimization/` |
| Step-by-step guide | RTX3090_QUICK_START.md | `docs_archive/04_gpu_optimization/` |
| Technical deep dive | RTX3090_CRITICAL_ANALYSIS.md | `docs_archive/04_gpu_optimization/` |
| Why advice was wrong | WHY_ADVICE_WAS_WRONG.md | `docs_archive/04_gpu_optimization/` |

### Bug Fixes (Historical)
| Task | Document | Location |
|------|----------|----------|
| All fixes summary | ALL_FIXES_COMPLETE.md | `docs_archive/02_bug_fixes/` |
| Verification report | FIXES_VERIFICATION_REPORT.md | `docs_archive/02_bug_fixes/` |
| Initial analysis | CODEX_ANALYSIS_ACTION_PLAN.md | `docs_archive/02_bug_fixes/` |

### Project Overview
| Task | Document | Location |
|------|----------|----------|
| Roadmap | ROADMAP.md | `docs_archive/01_project_overview/` |
| Branch status | BRANCH_STATUS.md | `docs_archive/01_project_overview/` |
| ROS2 lessons | lessons_learnt_ros2OnWindows.md | `docs_archive/01_project_overview/` |

---

## 🏆 Project Status Summary

### ✅ Completed (All Systems Go!)
- [x] **5 Critical Bugs Fixed** - Base mobility, action scaling, history, collision, trajectory
- [x] **Parameter Bug Fixed** - `target=` parameter corrected
- [x] **Training Verified** - Runs without crashes, metrics improving
- [x] **Hyperparameters Tuned** - clip_range_vf, ent_coef, target_kl optimized
- [x] **GPU Optimized** - TF32 + cuDNN benchmark + scaling strategy

### 🚀 Current Focus: GPU Utilization Scaling
**Problem**: Only 2.5GB / 24GB used (10% GPU)  
**Solution**: Scale environments 1024 → 2048 → 4096 → 6144  
**Expected**: 6-8x speedup (30 mins → 5 mins for 5M steps)

### 📊 Performance Targets

| Phase | num_envs | GPU Memory | Speed | Time (5M steps) |
|-------|----------|------------|-------|-----------------|
| Current | 1024 | 2.5GB (10%) | 20K/sec | 30-40 mins |
| Phase 1 | 2048 | 5-7GB (25%) | 40K/sec | 15-20 mins ⭐ |
| Phase 2 | 4096 | 10-15GB (60%) | 100K/sec | 5-10 mins |
| Phase 3 | 6144 | 18-22GB (80%) | 150K/sec | 3-5 mins |

---

## 💻 Development Environment

- **Isaac Sim**: 5.0.0-rc.45 @ `I:\isaaclab`
- **Isaac Lab**: 2.2.0, Python 3.11.13, PyTorch 2.7.0+cu128
- **Platform**: Windows 11 Pro (native, no WSL)
- **GPU**: RTX 3090 (24GB) + Quadro P2000 (display)
- **Robot**: 9 DOF mobile manipulator (6 arm + 3 base)
- **Task**: MobileMMTrackEE-v0
- **Algorithm**: PPO (Stable Baselines3)

---

## 🔧 Key Files & Directories

```
cinebotRL/
├── README.md                          # Main project README
├── DOCUMENTATION_INDEX.md             # This file (quick navigation)
├── docs_archive/                      # All documentation (organized)
│   ├── README.md                      # Documentation structure guide
│   ├── 01_project_overview/           # Project-level docs
│   ├── 02_bug_fixes/                  # Bug analysis & fixes
│   ├── 03_training_guides/            # Training workflows
│   ├── 04_gpu_optimization/           # GPU optimization (latest)
│   └── 05_historical_analysis/        # Historical reference
├── docs/                              # Architecture & setup guides
│   ├── setup/                         # Installation guides
│   ├── architecture/                  # System architecture
│   └── workflows/                     # Daily workflows
├── src/
│   └── rl_platform/
│       └── tasks/mobile_mm/
│           └── env.py                 # Main RL environment (all bugs fixed)
├── scripts/
│   └── reinforcement_learning/sb3/
│       └── train.py                   # Training script (optimized)
└── experiments/                       # Experiment configs
```

---

## 📞 Quick Commands Reference

```powershell
# Training (Phase 1 - Recommended start) - UPDATED for proper iterative learning
cd I:\isaaclab
.\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py `
  --task MobileMMTrackEE-v0 --num_envs 2048 --batch_size 512 --n_steps 32 --headless

# Monitor GPU
nvidia-smi -l 1

# TensorBoard
cd C:\Users\yanbo\wSpace\cinebotRL
tensorboard --logdir logs/sb3

# Git status
git status
git log --oneline -10
```

---

## 🎓 Key Learnings Documented

1. **Generic ML advice ≠ RL advice** ([WHY_ADVICE_WAS_WRONG.md](docs_archive/04_gpu_optimization/WHY_ADVICE_WAS_WRONG.md))
   - 80% of Stack Overflow advice was wrong for RL
   - Environment scaling > numerical tricks
   - TF32 yes, FP16 no (breaks PPO)

2. **Bug fixing methodology** ([ALL_FIXES_COMPLETE.md](docs_archive/02_bug_fixes/ALL_FIXES_COMPLETE.md))
   - Systematic verification (21/21 tests passed)
   - One bug at a time approach
   - Comprehensive testing

3. **GPU optimization strategy** ([RTX3090_CRITICAL_ANALYSIS.md](docs_archive/04_gpu_optimization/RTX3090_CRITICAL_ANALYSIS.md))
   - Problem: 10% GPU usage (too few environments)
   - Solution: Scale parallel environments 6x
   - Result: 6-8x speedup potential

---

## 🎯 Next Steps

1. **Run Phase 1 training** (2048 envs) - Verify 2x speedup
2. **Monitor GPU usage** - Should see 5-7GB (was 2.5GB)
3. **Scale to Phase 2** (4096 envs) if stable - Target 4-6x speedup
4. **Evaluate results** - Compare metrics, visualize policy

**Start here**: [`docs_archive/04_gpu_optimization/RTX3090_REFERENCE_CARD.md`](docs_archive/04_gpu_optimization/RTX3090_REFERENCE_CARD.md)

---

## 📝 Contributing to Documentation

When adding new docs:
- **Training guides** → `docs_archive/03_training_guides/`
- **GPU optimization** → `docs_archive/04_gpu_optimization/`
- **Bug fixes** → `docs_archive/02_bug_fixes/`
- **Project updates** → `docs_archive/01_project_overview/`

Keep root directory clean - move completed work to archive.

---

*Last updated: October 16, 2025*  
*Documentation organized: 25 files across 5 categories*
