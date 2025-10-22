# CinebotRL Documentation

**Comprehensive learning path for the CinebotRL reinforcement learning project.**

---

## 🚀 Quick Start

**New to the project?** Start here:
- **⚡ [Quick Start Guide](../START_TRAINING_NOW.md)** - Get training running in 3 commands!
- **🔧 [Quick Reference Card](QUICK_REFERENCE.md)** - One-page command cheat sheet
- **📊 [Reward System Design](reference/REWARD_SYSTEM_DESIGN.md)** - How rewards guide learning (start here to understand training goals)
- **🧠 [Model Architecture](reference/MODEL_ARCHITECTURE.md)** - PPO network, obs/action spaces, hyperparameters

---

## 📖 Learning Path

Follow these folders in order:

### 1. Setup
📁 [`01_setup/`](01_setup/)  
Install Isaac Sim/Lab, configure Windows/WSL environments.

### 2. Architecture
📁 [`02_architecture/`](02_architecture/)  
Understand system design, ROS2 communication, training pipeline.

### 3. Training
📁 [`03_training/`](03_training/)  
Train RL agents with multi-trajectory support.

### 4. Optimization
📁 [`04_optimization/`](04_optimization/)  
**RTX 3090 GPU optimization** - Essential for performance tuning.

### 5. Bug Fixes
📁 [`05_bug_fixes/`](05_bug_fixes/)  
Resolved issues and their solutions.

### 6. Workflows
📁 [`06_workflows/`](06_workflows/)  
Daily development and visualization workflows.

### 7. Reference
📁 [`07_reference/`](07_reference/)  
Technical references, reward functions, model architecture, troubleshooting.

**Essential References:**
- **[REWARD_SYSTEM_DESIGN.md](reference/REWARD_SYSTEM_DESIGN.md)** ⭐ - Complete reward function documentation (800+ lines)
- **[MODEL_ARCHITECTURE.md](reference/MODEL_ARCHITECTURE.md)** ⭐ - PPO network, obs/action spaces, hyperparameters (1000+ lines)
- **[reward_cheatsheet.md](reference/reward_cheatsheet.md)** - Quick reward formula reference
- **[troubleshooting.md](reference/troubleshooting.md)** - Common issues & solutions

### 8. Project History
📁 [`08_project_history/`](08_project_history/)  
Project evolution, training session logs, documentation changes.

**Training Session Logs:**
- **[TRAINING_SESSIONS_MASTER_LOG.md](training_sessions/TRAINING_SESSIONS_MASTER_LOG.md)** ⭐ - All training runs documented
- **[SESSION_5B_FIX_SUMMARY.md](training_sessions/SESSION_5B_FIX_SUMMARY.md)** ⭐ - Base mobility fixes explained (Session 5 → 5b)

### 9. Archive
📁 [`09_archive/`](09_archive/)  
Archived tracking documents and WSL troubleshooting.

---

## 🔍 Quick Navigation

**Most Important Documents:**
- **[Reward System Design](reference/REWARD_SYSTEM_DESIGN.md)** ⭐ - How rewards are designed (9 components)
- **[Model Architecture](reference/MODEL_ARCHITECTURE.md)** ⭐ - PPO network, obs/action spaces, hyperparameters
- **[Training Sessions Log](training_sessions/TRAINING_SESSIONS_MASTER_LOG.md)** ⭐ - All training runs documented
- **[Session 5b Fix Summary](training_sessions/SESSION_5B_FIX_SUMMARY.md)** ⭐ - Base mobility fixes

**By Topic:**
- **Training**: [Multi-Trajectory Training](03_training/multi_trajectory_training.md), [Command Reference](03_training/TRAINING_COMMAND_REFERENCE.md)
- **Architecture**: [PPR Control Architecture](02_architecture/PPR_CONTROL_ARCHITECTURE.md), [Training Pipeline](02_architecture/training_architecture.md)
- **Bug Fixes**: [Base Movement Bug](05_bug_fixes/BASE_MOVEMENT_BUG_ANALYSIS.md), [Frozen Base Summary](05_bug_fixes/FROZEN_BASE_SUMMARY.md)
- **Optimization**: [Policy Divergence at 200M](04_optimization/POLICY_DIVERGENCE_200M.md), [Trajectory Tracking](04_optimization/TRAJECTORY_TRACKING_IMPROVEMENTS.md)
- **Troubleshooting**: [Troubleshooting Guide](07_reference/troubleshooting.md)
- **Daily Use**: [Daily Workflow](06_workflows/daily_workflow.md), [Quick Reference](QUICK_REFERENCE.md)

---

## 📝 Documentation Updates

**Latest Changes (2025-10-22):**
- ✅ **Added comprehensive documentation**: [REWARD_SYSTEM_DESIGN.md](reference/REWARD_SYSTEM_DESIGN.md) (800+ lines) and [MODEL_ARCHITECTURE.md](reference/MODEL_ARCHITECTURE.md) (1000+ lines)
- ✅ **Major reorganization**: 37 files moved into 8 categorized directories (01_setup through 08_project_history)
- ✅ **Training session logs**: Session 5 failure documented, Session 5b fixes explained
- ✅ **Cross-references updated**: All documents now link to new file locations

**Previous Updates:**
- 2025-10-17: Added PPR control architecture documentation
- 2025-10-16: Updated base movement bug analysis with URDF fixes
- 2025-10-15: Created training success documentation and compatibility fixes

This structure consolidates all documentation from the original `docs/` and `docs_archive/` folders into a unified, numbered learning path.

**For project overview and quick start, see:** [`../README.md`](../README.md)

---

**💡 TIP:** Bookmark this page as your documentation hub! All essential documents are linked above.
