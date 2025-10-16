# Documentation Archive

This directory contains organized documentation from the cinebotRL project development process.

## Directory Structure

### 📁 01_project_overview/
Project-level documentation about status, roadmap, and lessons learned.
- `README.md` - Main project README (symlink)
- `ROADMAP.md` - Project roadmap and milestones
- `BRANCH_STATUS.md` - Git branch status and workflow
- `lessons_learnt_ros2OnWindows.md` - ROS2 on Windows lessons

### 📁 02_bug_fixes/
Documentation related to bug identification, fixes, and verification.
- `CODEX_ANALYSIS_ACTION_PLAN.md` - Initial bug analysis
- `codexInspect-23dd540.md` - Codex inspection report
- `BUG_FIXES_COMPLETE.md` - Bug fix completion summary
- `ALL_FIXES_COMPLETE.md` - All fixes verified
- `FIX5_TRAJECTORY_ADVANCEMENT_COMPLETE.md` - Specific fix details
- `FIXES_VERIFICATION_REPORT.md` - Verification test results
- `TIP_OVER_ROOT_CAUSE.md` - Tip-over issue analysis

### 📁 03_training_guides/
Active training documentation and quick-start guides.
- `START_TRAINING_NOW.md` - Quick start command reference
- `READY_TO_TRAIN.md` - Pre-training checklist and setup
- `TRAINING_SUCCESS.md` - Training success documentation
- `TRAINING_ACTION_PLAN.md` - Training workflow and plan
- `TRAINING_TUNING_PLAN.md` - Hyperparameter tuning strategy
- `QUICK_TUNING_SUMMARY.md` - Quick tuning reference
- `CPU_TRAINING_GUIDE.md` - CPU training options
- `VISUALIZATION_QUICK_START.md` - Visualization tools

### 📁 04_gpu_optimization/
RTX 3090 optimization documentation (most recent work).
- `RTX3090_REFERENCE_CARD.md` - **START HERE** - Quick reference
- `RTX3090_QUICK_START.md` - Commands and testing protocol
- `RTX3090_CRITICAL_ANALYSIS.md` - Technical deep dive
- `RTX3090_OPTIMIZATION_GUIDE.md` - Comprehensive optimization guide
- `WHY_ADVICE_WAS_WRONG.md` - Point-by-point analysis
- `OPTIMIZATION_SUMMARY.md` - Change log and summary

### 📁 05_historical_analysis/
Historical analysis and legacy documentation.
- Reference materials from earlier development phases

---

## Quick Navigation

### 🚀 I want to start training NOW
→ Read: `03_training_guides/START_TRAINING_NOW.md`  
→ Or: `04_gpu_optimization/RTX3090_REFERENCE_CARD.md`

### 🐛 I want to understand the bug fixes
→ Read: `02_bug_fixes/ALL_FIXES_COMPLETE.md`  
→ Then: `02_bug_fixes/FIXES_VERIFICATION_REPORT.md`

### ⚡ I want to optimize GPU performance
→ Read: `04_gpu_optimization/RTX3090_REFERENCE_CARD.md`  
→ Then: `04_gpu_optimization/RTX3090_QUICK_START.md`  
→ Deep dive: `04_gpu_optimization/RTX3090_CRITICAL_ANALYSIS.md`

### 📊 I want to understand the project history
→ Start: `01_project_overview/ROADMAP.md`  
→ Context: `02_bug_fixes/CODEX_ANALYSIS_ACTION_PLAN.md`

### 🎓 I want to learn what NOT to do
→ Read: `04_gpu_optimization/WHY_ADVICE_WAS_WRONG.md`

---

## Document Organization Rationale

### Why this structure?

1. **Chronological + Topical**: Follows project development flow
2. **Progressive detail**: Quick start → detailed analysis
3. **Clear separation**: Bug fixes vs optimization vs training
4. **Easy navigation**: Numbered folders, clear naming

### Active vs Archive

**Keep in root** (active development):
- Project README
- Current status files

**Move to archive** (reference only):
- Historical documentation
- Completed bug fix records
- Analysis documents

---

## Maintenance

When adding new documentation:

1. **Training guides** → `03_training_guides/`
2. **GPU optimization** → `04_gpu_optimization/`
3. **Bug fixes** → `02_bug_fixes/`
4. **Project status** → `01_project_overview/`

Keep root directory minimal for active development focus.

---

*Organized: October 16, 2025*
