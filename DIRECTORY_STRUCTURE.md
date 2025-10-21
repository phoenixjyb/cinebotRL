# CinebotRL Directory Structure

**Last Updated:** October 21, 2025  
**Purpose:** Quick reference for finding files in this repository

---

## 📁 Root Directory Structure

```
cinebotRL/
├── README.md                           # Main project documentation (START HERE)
├── pyproject.toml                      # Python project configuration
├── reporting.py                        # Logging/reporting utilities
│
├── .github/                            # GitHub-specific files
│   └── copilot-instructions.md         # AI coding agent context
│
├── src/                                # Source code (Python package)
│   ├── task_spec.py                    # Robot task specification
│   ├── rl_platform/                    # RL training platform
│   │   └── tasks/mobile_mm/            # Mobile manipulator task
│   │       ├── env.py                  # Main environment (CRITICAL FILE)
│   │       ├── config.py               # Task configuration
│   │       ├── observations.py         # Observation space composition
│   │       ├── rewards.py              # Reward functions
│   │       └── trajectories.py         # Trajectory manager
│   └── asset_inspector/                # Asset validation tools
│
├── scripts/                            # Executable scripts
│   ├── launch_training_windows.ps1     # Main training launcher (USE THIS)
│   ├── reinforcement_learning/sb3/     # SB3 training scripts
│   │   └── train.py                    # Core training script
│   ├── networking/                     # ROS2/FastDDS configuration
│   └── wsl/                            # WSL-specific scripts
│
├── docs/                               # Documentation (ORGANIZED)
│   ├── README.md                       # Documentation index
│   ├── QUICK_REFERENCE.md              # Quick command reference
│   ├── TRAJECTORY_TRACKING_IMPROVEMENTS.md  # Architecture analysis (NEW)
│   ├── IMPROVEMENT_CHECKLIST.md        # Action items checklist (NEW)
│   │
│   ├── setup/                          # Installation guides
│   │   ├── INSTALL_QUICK.md
│   │   ├── TRAIN_ON_WINDOWS.md
│   │   ├── windows_setup_guide.md
│   │   └── wsl_setup_guide.md
│   │
│   ├── workflows/                      # How-to guides
│   │   ├── daily_workflow.md
│   │   ├── multi_trajectory_training.md
│   │   └── visualization_options.md
│   │
│   ├── reference/                      # Technical references
│   │   ├── reward_cheatsheet.md
│   │   ├── reward_system.md
│   │   ├── robot_constraints_updated.md
│   │   └── troubleshooting.md
│   │
│   ├── architecture/                   # System architecture
│   │   ├── overview.md
│   │   ├── python_environments.md
│   │   ├── ros2_communication.md
│   │   └── training_architecture.md
│   │
│   ├── tracking/                       # Tracking system details
│   │   ├── ee_frame_alignment.md
│   │   ├── mobile_arm_asset_validation.md
│   │   └── phase0_environment.md
│   │
│   ├── troubleshooting/                # Problem-solving guides
│   │   ├── wsl2_cuda_fix_summary.md
│   │   └── wsl2_cuda_isaac_sim.md
│   │
│   ├── urdf_fixes/                     # URDF/USD fixes (ORGANIZED)
│   │   ├── URDF_FIXES_APPLIED.md       # Summary of all physics fixes
│   │   ├── PPR_MASS_FIX_SUMMARY.md     # PPR joint mass fixes
│   │   ├── PPR_STIFFNESS_TUNING.md     # Spring-damper tuning
│   │   ├── USD_REGENERATION_GUIDE.md   # How to regenerate USD
│   │   ├── USD_REGENERATION_CHECKLIST_V2.md
│   │   ├── ISAAC_SIM_GUI_GUIDE.md      # Isaac Sim import guide
│   │   └── ISAAC_SIM_IMPORT_CHECKLIST.md
│   │
│   ├── training_sessions/              # Training history (ORGANIZED)
│   │   ├── TRAINING_DIARY.md           # Complete training log
│   │   ├── TRAINING_SESSION_2_STATUS.md
│   │   ├── SESSION_4_LAUNCH_READY.md
│   │   ├── 20HZ_CONTROL_ANALYSIS.md    # Control frequency analysis
│   │   ├── 20HZ_HOLDOVER_FIXES.md      # 50Hz→20Hz fixes
│   │   ├── CODE_AUDIT_20HZ.md          # Code consistency audit
│   │   └── CONTROL_FREQUENCY_ANALYSIS.md
│   │
│   └── legacy/                         # Obsolete/archived docs
│       ├── codex_analysis_1019.md
│       ├── DOCUMENTATION_INDEX.md
│       ├── IMPORT_NOW.md
│       └── PROPOSED_DOCS_STRUCTURE.md
│
├── assets/                             # Processed assets
│   └── processed/mobile_arm_whole_body/
│
├── assets_own/                         # Custom robot assets
│   ├── mobile_manipulator_PPR_base_corrected.urdf  # CORRECTED URDF
│   ├── meshes/                         # Robot mesh files
│   └── usd/                            # Isaac Sim USD assets
│       └── mobile_manipulator_PPR_base_corrected.usd  # CORRECTED USD
│
├── trajectoryToLearn/                  # Training trajectories
│   ├── chassis_required_indices.txt    # Trajectory filter
│   ├── chassis_required_trajectories.txt
│   ├── 1_pull_world_scaled.json        # Example trajectory
│   └── world_json/                     # Trajectory database
│
├── logs/                               # Training logs
│   ├── sb3/                            # Stable-Baselines3 logs
│   │   └── MobileMMTrackEE-v0/         # Task-specific logs
│   └── evaluation/                     # Evaluation logs (ORGANIZED)
│       ├── eval_after_fix.txt
│       ├── eval_debug_output.txt
│       ├── eval_final.txt
│       ├── eval_fixed_output.txt
│       ├── eval_result.txt
│       ├── test_error.log
│       └── trajectory_analysis_results.csv
│
├── experiments/                        # Experimental code
├── stages/                             # Isaac Sim stage files
└── archive/                            # Archived/deprecated files
    └── projectSketch.md
```

---

## 🎯 Quick File Finder

### "I want to..."

**Start training:**
→ `scripts/launch_training_windows.ps1`

**Understand the system:**
→ `README.md` → `docs/README.md` → `docs/QUICK_REFERENCE.md`

**Fix base not moving:**
→ `docs/urdf_fixes/URDF_FIXES_APPLIED.md`

**Understand trajectory tracking:**
→ `docs/TRAJECTORY_TRACKING_IMPROVEMENTS.md` ⭐ NEW

**Follow improvement checklist:**
→ `docs/IMPROVEMENT_CHECKLIST.md` ⭐ NEW

**Modify the environment:**
→ `src/rl_platform/tasks/mobile_mm/env.py`

**Change rewards:**
→ `src/rl_platform/tasks/mobile_mm/rewards.py`

**Configure training:**
→ `src/rl_platform/tasks/mobile_mm/config.py`

**Debug URDF issues:**
→ `docs/urdf_fixes/` (all files)

**See training history:**
→ `docs/training_sessions/TRAINING_DIARY.md`

**Install Isaac Lab on Windows:**
→ `docs/setup/TRAIN_ON_WINDOWS.md`

**Set up WSL2:**
→ `docs/setup/wsl_setup_guide.md`

**Troubleshoot CUDA:**
→ `docs/troubleshooting/wsl2_cuda_fix_summary.md`

**Understand rewards:**
→ `docs/reference/reward_cheatsheet.md`

**Daily workflow:**
→ `docs/workflows/daily_workflow.md`

---

## 📋 File Categories

### Critical Files (Touch These Carefully)
- `src/rl_platform/tasks/mobile_mm/env.py` - Main environment logic
- `src/rl_platform/tasks/mobile_mm/config.py` - Task configuration
- `assets_own/mobile_manipulator_PPR_base_corrected.urdf` - Robot URDF
- `assets_own/usd/mobile_manipulator_PPR_base_corrected.usd` - Robot USD
- `scripts/launch_training_windows.ps1` - Training launcher

### Frequently Modified Files
- `src/rl_platform/tasks/mobile_mm/rewards.py` - Reward tuning
- `trajectoryToLearn/*.json` - Training trajectories
- `docs/training_sessions/TRAINING_DIARY.md` - Training log

### Read-Only References
- `docs/reference/*.md` - Technical references
- `docs/setup/*.md` - Installation guides
- `docs/urdf_fixes/*.md` - Physics fix documentation

### Generated/Output Files
- `logs/sb3/*/` - Training logs (TensorBoard)
- `logs/evaluation/` - Evaluation results
- `cinebotrl.egg-info/` - Python package metadata

---

## 🗂️ Recent Reorganization (Oct 21, 2025)

Moved scattered root files into organized directories:

**URDF Fixes** → `docs/urdf_fixes/`:
- URDF_FIXES_APPLIED.md
- PPR_MASS_FIX_SUMMARY.md
- PPR_STIFFNESS_TUNING.md
- USD_REGENERATION_GUIDE.md
- USD_REGENERATION_CHECKLIST_V2.md
- ISAAC_SIM_GUI_GUIDE.md
- ISAAC_SIM_IMPORT_CHECKLIST.md

**Training Sessions** → `docs/training_sessions/`:
- TRAINING_DIARY.md
- TRAINING_SESSION_2_STATUS.md
- SESSION_4_LAUNCH_READY.md
- 20HZ_CONTROL_ANALYSIS.md
- 20HZ_HOLDOVER_FIXES.md
- CODE_AUDIT_20HZ.md
- CONTROL_FREQUENCY_ANALYSIS.md

**Evaluation Logs** → `logs/evaluation/`:
- eval_*.txt (5 files)
- test_error.log
- trajectory_analysis_results.csv

**Legacy Docs** → `docs/legacy/`:
- codex_analysis_1019.md
- DOCUMENTATION_INDEX.md
- IMPORT_NOW.md
- PROPOSED_DOCS_STRUCTURE.md

---

## 📝 Documentation Conventions

### Naming Patterns
- `UPPERCASE.md` - Important standalone documents (README, QUICK_REFERENCE)
- `lowercase_with_underscores.md` - Technical references
- `PascalCase.md` - Historical/session documents (TRAINING_DIARY)

### Directory Structure
```
docs/
├── Top-level guides (README, QUICK_REFERENCE)
├── setup/ - Installation (how to set up environment)
├── workflows/ - How-to guides (how to do tasks)
├── reference/ - Technical details (reward system, constraints)
├── architecture/ - System design (how it works)
├── tracking/ - Tracking system specifics
├── troubleshooting/ - Problem solving
├── urdf_fixes/ - URDF/USD fixes (consolidated history)
├── training_sessions/ - Training logs (historical records)
└── legacy/ - Obsolete docs (keep for reference)
```

---

## 🔍 Search Tips

**Find files by content:**
```powershell
# Search in all markdown files
Get-ChildItem -Recurse -Filter "*.md" | Select-String "search term"

# Search in Python files
Get-ChildItem -Recurse -Filter "*.py" | Select-String "function_name"
```

**Find recent changes:**
```powershell
# Files modified in last 7 days
Get-ChildItem -Recurse -File | Where-Object {$_.LastWriteTime -gt (Get-Date).AddDays(-7)}

# Git changed files
git log --name-only --since="7 days ago"
```

---

## 🚀 Next Steps After Reorganization

1. ✅ Root directory is now clean (only essential files)
2. ✅ Documentation is organized by purpose
3. ✅ Training logs are archived
4. ✅ URDF fixes are consolidated
5. [ ] Review `docs/README.md` for updated navigation
6. [ ] Commit reorganization changes

**Recommended commit message:**
```
Reorganize repository structure for better maintainability

Moved scattered root documentation into organized directories:
- docs/urdf_fixes/ - URDF/USD physics fixes and regeneration guides
- docs/training_sessions/ - Training logs and control frequency analysis
- docs/legacy/ - Obsolete documentation (archived for reference)
- logs/evaluation/ - Evaluation results and test logs

Created DIRECTORY_STRUCTURE.md as quick reference guide.

Root directory now contains only essential project files:
- README.md, pyproject.toml (project config)
- src/, scripts/, docs/ (organized code/docs)
- assets_own/, trajectoryToLearn/ (robot data)

Benefits:
- Easier navigation for new contributors
- Clear separation of current vs historical docs
- Reduced root directory clutter (44 → 24 items)
```

---

**Maintained by:** CinebotRL Team  
**Questions?** See `docs/README.md` for documentation index
