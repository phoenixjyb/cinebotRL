# Training Sessions Master Log

**Project:** CinebotRL - Mobile Manipulator End-Effector Tracking  
**Repository:** phoenixjyb/cinebotRL  
**Branch:** train-windows  
**Last Updated:** October 21, 2025 19:30 +0800

---

## 🚀 **READY TO LAUNCH: Session 5**

**Status:** All fixes committed, documentation complete, ready for training launch  
**Session:** 5 - Base Mobilization Fix  
**Commits:** 6 total (URDF fixes → observation space fix → logging infrastructure)  
**Latest Commit:** `78216e1` - Training session logging infrastructure

### Quick Start
```powershell
# Option A: 4096 Environments (~12 hours)
cd I:\isaaclab
I:\isaaclab\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py --task MobileMMTrackEE-v0 --num_envs 4096 --n_steps 64 --batch_size 256 --total_timesteps 100000000 --learning_rate 3e-4 --ent_coef 0.001 --enable_entropy_decay --final_ent_coef 1e-4 --decay_start_timestep 50000000 --decay_duration_timesteps 50000000 --enable_kl_schedule --kl_warmup 0.25 --kl_main 0.15 --kl_finetune 0.07 --target_kl 1.0 --trajectory_type multi_recorded --use_all_trajectories --headless

# Option B: 8192 Environments (~9 hours, may be tight on VRAM)
cd I:\isaaclab
I:\isaaclab\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py --task MobileMMTrackEE-v0 --num_envs 8192 --n_steps 64 --batch_size 512 --total_timesteps 100000000 --learning_rate 3e-4 --ent_coef 0.001 --enable_entropy_decay --final_ent_coef 1e-4 --decay_start_timestep 50000000 --decay_duration_timesteps 50000000 --enable_kl_schedule --kl_warmup 0.25 --kl_main 0.15 --kl_finetune 0.07 --target_kl 1.0 --trajectory_type multi_recorded --use_all_trajectories --headless
```

### What's Fixed
✅ Observation space enhanced (+4 dims for base-to-target info)  
✅ Rewards rebalanced (base movement gives +15.7 points vs -5.1 for staying still!)  
✅ Smart penalty (90% reduction when moving)  
✅ All bugs from BASE_MOVEMENT_BUG_ANALYSIS.md addressed

### Expected Outcome
🎯 Base action std > 0.3 within 5 minutes  
🎯 Mean tracking error < 0.3m after 100M steps  
🎯 Policy learns to mobilize base when targets out of reach

---

This document maintains a chronological record of all training sessions, including git commits, configurations, commands, and outcomes.

---

## 📋 Quick Navigation

- [Session 1](#session-1-initial-training) - Initial baseline
- [Session 2](#session-2) - [TBD]
- [Session 3](#session-3) - [TBD]
- [Session 4a](#session-4a) - [TBD]
- [Session 4b](#session-4b-25m-steps-no-base-movement) - 25.7M steps, no base movement (stopped)
- [Session 5](#session-5-base-mobilization-fix) - **CURRENT** - Critical observation space fix

---

## Session 5: Base Mobilization Fix

**Status:** 🚀 READY TO LAUNCH  
**Date:** October 21, 2025  
**Objective:** Fix critical observation space bug preventing base mobilization learning

### Git Commits

#### Commit 1: URDF Physics Fixes
```
Commit: 00060bc492efe10a028fba0fa5a01570fa80afa6
Date: October 21, 2025 00:51:41 +0800
Message: Fix critical URDF physics issues for base mobility
```

**Changes:**
- Set PPR helper links (base_link_x/y) mass to 0.0 kg (was 0.001kg)
  - Eliminates numerical stiffness from micro-bodies + 10kN/m springs
  - PhysX computes composite mass from articulation tree (more stable)
- Fix joint_theta limits: -inf/+inf → ±6.283185 rad (±2π)
  - USD converter collapses infinite limits to zero-width (locked joint)
  - Finite limits allow base rotation within ±360°+ range
- Regenerated USD from corrected URDF with Isaac Sim 5.0
  - All PPR joints set to Position control (matches env.py)
  - Mesh scale 0.001 (mm→m conversion)
  - Moveable base enabled

**URDF Physics Summary (5 fixes total):**
1. Base mass: 0.0 → 20.0 kg (movable, not static)
2. Chassis mass: 50.96 → 30.96 kg (fixed duplication, 51kg total)
3. Base inertia: (1,1,1) → (0.833, 0.833, 1.2) (realistic)
4. PPR helpers: 0.001 → 0.0 kg (stable simulation)
5. joint_theta: -inf/+inf → ±6.28 rad (rotatable)

**Files Modified:**
- URDF files (PPR helper masses, joint limits)
- USD asset regeneration
- Documentation: `docs/urdf_physics_analysis.md`, `docs/URDF_PHYSICS_ISSUES_REMAINING.md`, `docs/PPR_CONTROL_ARCHITECTURE.md`, `URDF_FIXES_APPLIED.md`

---

#### Commit 2: PPR Helper Mass Fix
```
Commit: c59cda8b45306321d637c1ba352fb94097821f36
Date: October 21, 2025 09:07:55 +0800
Message: Fix PPR helper link masses: 0.0 → 1.0 kg for real base mobility
```

**Changes:**
- Changed PPR helper links mass from 0.0 kg → 1.0 kg
- Ensures PhysX propagates position updates through articulation tree
- Required for `root_pos_w` to update when PPR joints move

**Files Modified:**
- URDF files (PPR helper link masses)

---

#### Commit 3: USD Regeneration with 1.0kg Masses
```
Commit: 29d66686315649cdad5547c1bf79326bc845e219
Date: October 21, 2025 09:23:03 +0800
Message: Regenerate USD with 1.0kg PPR helper masses for real base mobility
```

**Changes:**
- Imported corrected URDF in Isaac Sim 5.0
- PPR helper links now have 1.0 kg mass (was 0.0 kg)
- All joints set to Position control
- Mesh scale 0.001 (mm to m)
- Moveable base enabled

**Files Modified:**
- `assets_own/usd/mobile_manipulator_PPR_base_corrected.usd`
- `assets_own/usd/configuration/` (if present)

---

#### Commit 4: Repository Reorganization
```
Commit: eca70435ba2970d66176f452f8379833da96eaed
Date: October 21, 2025 18:51:12 +0800
Message: Reorganize repository structure for better maintainability
```

**Changes:**
- Created organized directory structure: `docs/{legacy/, training_sessions/, urdf_fixes/}`, `logs/evaluation/`
- Moved 18 scattered files into appropriate directories
- Created `DIRECTORY_STRUCTURE.md` navigation guide
- Reduced root clutter by 45% (44 items → 24 items)

**Files Modified:**
- 28 files changed (moved to organized directories)
- Created `DIRECTORY_STRUCTURE.md`

---

#### Commit 5: Critical Observation Space Fix
```
Commit: 7ff0e8de8282bdd7457299cd999d3269f3169dc1
Date: October 21, 2025 19:12:01 +0800
Message: Fix CRITICAL observation space bug preventing base mobilization learning
```

**ROOT CAUSE IDENTIFIED:**
- 83.8% of trajectory waypoints OUT OF REACH (>0.6m from base)
- Mean distance: 2.3m, Max distance: 4.0m (trajectory REQUIRES base movement!)
- Policy had 0.958 explained variance but zero base movement after 25.7M steps
- Policy learned "I will get low rewards" but couldn't learn "HOW to fix it"

**CRITICAL FIXES:**

**Fix #1: Observation Space Enhancement (+4 dims)**
- `src/rl_platform/tasks/mobile_mm/observations.py`:
  - Added `base_to_target_xy` [2 dims]: XY offset from base to target
  - Added `base_to_target_dist` [1 dim]: Euclidean distance to target
  - Added `out_of_reach_flag` [1 dim]: Binary signal when target >0.6m from base
  - Total: 59 dims (was 55 dims with lookahead)

**Fix #2: Reward Rebalancing**
- `src/rl_platform/tasks/mobile_mm/config.py`:
  - `base_progress_reward`: 50.0 → 150.0 (3x stronger!)
  - `target_distance_penalty`: 10.0 → 3.0 (less harsh during movement)
  - `action_magnitude`: 0.01 → 0.005 (encourage base action exploration)
  - `jerk_limit_penalty`: 0.1 → 0.05 (reduce movement suppression)

**Fix #3: Smart Distance Penalty**
- `src/rl_platform/tasks/mobile_mm/rewards.py`:
  - 90% penalty reduction when base is moving (>1cm threshold)
  - Full penalty when stationary (maintains gradient)
  - Encourages exploration: "Try moving base → penalty drops dramatically!"

**Files Modified:**
- `src/rl_platform/tasks/mobile_mm/observations.py`
- `src/rl_platform/tasks/mobile_mm/config.py`
- `src/rl_platform/tasks/mobile_mm/rewards.py`
- `scripts/analyze_trajectory_reach.py` (NEW - trajectory analysis tool)
- `docs/BASE_MOVEMENT_BUG_ANALYSIS.md` (NEW - 10-page investigation report)

---

#### Commit 6: Training Session Logging Infrastructure
```
Commit: 78216e1ec98b86b84cd2e8deb46cf0cc85e8aa43
Date: October 21, 2025 19:30:00 +0800 (approx)
Message: Add comprehensive training session logging infrastructure
```

**Changes:**
- Created `TRAINING_SESSIONS_MASTER_LOG.md` (this file) - single source of truth
- Created `docs/training_sessions/SESSION_5_LAUNCH_LOG.md` - detailed tracking template
- Documented all 5 previous commits with full git hashes and timestamps
- Both launch command options (4096 and 8192 envs)
- Complete success criteria, monitoring plan, and failure indicators
- Session comparison matrix and template for future sessions

**Files Modified:**
- `TRAINING_SESSIONS_MASTER_LOG.md` (NEW)
- `docs/training_sessions/SESSION_5_LAUNCH_LOG.md` (NEW)
- `docs/training_sessions/SESSION_5_BASE_FIX.md` (added in this commit)

---

**ROOT CAUSE IDENTIFIED:**
- 83.8% of trajectory waypoints OUT OF REACH (>0.6m from base)
- Mean distance: 2.3m, Max distance: 4.0m (trajectory REQUIRES base movement!)
- Policy had 0.958 explained variance but zero base movement after 25.7M steps
- Policy learned "I will get low rewards" but couldn't learn "HOW to fix it"

**CRITICAL FIXES:**

**Fix #1: Observation Space Enhancement (+4 dims)**
- `src/rl_platform/tasks/mobile_mm/observations.py`:
  - Added `base_to_target_xy` [2 dims]: XY offset from base to target
  - Added `base_to_target_dist` [1 dim]: Euclidean distance to target
  - Added `out_of_reach_flag` [1 dim]: Binary signal when target >0.6m from base
  - Total: 59 dims (was 55 dims with lookahead)

**Fix #2: Reward Rebalancing**
- `src/rl_platform/tasks/mobile_mm/config.py`:
  - `base_progress_reward`: 50.0 → 150.0 (3x stronger!)
  - `target_distance_penalty`: 10.0 → 3.0 (less harsh during movement)
  - `action_magnitude`: 0.01 → 0.005 (encourage base action exploration)
  - `jerk_limit_penalty`: 0.1 → 0.05 (reduce movement suppression)

**Fix #3: Smart Distance Penalty**
- `src/rl_platform/tasks/mobile_mm/rewards.py`:
  - 90% penalty reduction when base is moving (>1cm threshold)
  - Full penalty when stationary (maintains gradient)
  - Encourages exploration: "Try moving base → penalty drops dramatically!"

**Files Modified:**
- `src/rl_platform/tasks/mobile_mm/observations.py`
- `src/rl_platform/tasks/mobile_mm/config.py`
- `src/rl_platform/tasks/mobile_mm/rewards.py`
- `scripts/analyze_trajectory_reach.py` (NEW - trajectory analysis tool)
- `docs/BASE_MOVEMENT_BUG_ANALYSIS.md` (NEW - 10-page investigation report)

---

### Training Configuration

**Environment:**
- Task: `MobileMMTrackEE-v0`
- Trajectory Type: `multi_recorded`
- All Trajectories: Enabled
- Rendering: Headless

**Hyperparameters:**
- Total Timesteps: 100,000,000 (100M)
- Number of Environments: **[4096 or 8192 - TBD]**
- Steps per Rollout: 64
- Batch Size: **[256 or 512 - TBD based on num_envs]**
- Learning Rate: 3e-4

**PPO Settings:**
- Initial Entropy: 0.001
- Entropy Decay: Enabled (final: 1e-4, start: 50M, duration: 50M)
- Target KL: 1.0
- KL Schedule: Enabled (warmup: 0.25, main: 0.15, finetune: 0.07)

### Launch Command

**Option A: 4096 Environments**
```powershell
cd I:\isaaclab
I:\isaaclab\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py `
  --task MobileMMTrackEE-v0 `
  --num_envs 4096 `
  --n_steps 64 `
  --batch_size 256 `
  --total_timesteps 100000000 `
  --learning_rate 3e-4 `
  --ent_coef 0.001 `
  --enable_entropy_decay `
  --final_ent_coef 1e-4 `
  --decay_start_timestep 50000000 `
  --decay_duration_timesteps 50000000 `
  --enable_kl_schedule `
  --kl_warmup 0.25 `
  --kl_main 0.15 `
  --kl_finetune 0.07 `
  --target_kl 1.0 `
  --trajectory_type multi_recorded `
  --use_all_trajectories `
  --headless
```

**Option B: 8192 Environments**
```powershell
cd I:\isaaclab
I:\isaaclab\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py `
  --task MobileMMTrackEE-v0 `
  --num_envs 8192 `
  --n_steps 64 `
  --batch_size 512 `
  --total_timesteps 100000000 `
  --learning_rate 3e-4 `
  --ent_coef 0.001 `
  --enable_entropy_decay `
  --final_ent_coef 1e-4 `
  --decay_start_timestep 50000000 `
  --decay_duration_timesteps 50000000 `
  --enable_kl_schedule `
  --kl_warmup 0.25 `
  --kl_main 0.15 `
  --kl_finetune 0.07 `
  --target_kl 1.0 `
  --trajectory_type multi_recorded `
  --use_all_trajectories `
  --headless
```

**Command Used:** [TO BE FILLED AT LAUNCH]

### Expected Performance

**Reward Improvement Per Step (7.5cm base movement):**
- Base Mobilization Reward: +11.0
- Distance Penalty Improvement: +4.6 (90% reduction when moving)
- Position Tracking Improvement: +0.1
- Action Penalties: -0.025
- **NET IMPROVEMENT: +15.7 points** (vs -5.1 for staying still)

**Success Criteria:**
- **Early (1M steps, 5 min):** base_vx_action_std > 0.3, base_mobilization_reward > 1.0
- **Intermediate (10M steps, 1 hr):** base_movement > 0.005m/step, explained_variance > 0.80
- **Final (100M steps):** mean_tracking_error < 0.3m, total_reward > 35, explained_variance > 0.92

### Training Timeline

- **Launch Time:** [TO BE FILLED]
- **Estimated Duration:** 9-12 hours (depending on num_envs)
- **Checkpoint 1 (5 min):** [TO BE FILLED]
- **Checkpoint 2 (1 hour):** [TO BE FILLED]
- **Checkpoint 3 (3 hours):** [TO BE FILLED]
- **Checkpoint 4 (6 hours):** [TO BE FILLED]
- **Completion:** [TO BE FILLED]

### Results

**Status:** Not Started

**Final Metrics:** [TO BE FILLED]

**Outcome:** [TO BE FILLED]

**Detailed Log:** [docs/training_sessions/SESSION_5_LAUNCH_LOG.md](docs/training_sessions/SESSION_5_LAUNCH_LOG.md)

---

## Session 4b: 25.7M Steps - No Base Movement

**Status:** ⏸️ STOPPED (Issue Identified)  
**Date:** [Date TBD]  
**Objective:** Continue training to 100M steps

### Issue Summary

**Symptoms:**
- Excellent metrics: 0.958 explained variance, 0.00443 value loss
- **ZERO base movement** despite training to 25.7M steps
- Policy correctly predicted low rewards but didn't learn how to fix it

**Root Cause:**
- Observation space missing explicit base-to-target distance signal
- Reward structure made base movement net negative even when necessary
- 83.8% of trajectory waypoints require base movement (>0.6m from base)

**Resolution:** Session 5 implements critical fixes

### Training Configuration

[TO BE FILLED FROM SESSION 4b LOGS]

### Results

**Final Metrics:**
- Steps Completed: 25,700,000 / 100,000,000
- Explained Variance: 0.958
- Value Loss: 0.00443
- Base Movement: ~0 mm (ZERO!)
- Decision: Stopped for investigation

**Detailed Log:** [TO BE CREATED IF NEEDED]

---

## Session 4a

**Status:** [TBD]  
**Date:** [TBD]  
**Objective:** [TBD]

[TO BE FILLED]

---

## Session 3

**Status:** [TBD]  
**Date:** [TBD]  
**Objective:** [TBD]

[TO BE FILLED]

---

## Session 2

**Status:** [TBD]  
**Date:** [TBD]  
**Objective:** [TBD]

[TO BE FILLED]

---

## Session 1: Initial Training

**Status:** [TBD]  
**Date:** [TBD]  
**Objective:** Establish baseline performance

[TO BE FILLED]

---

## 📊 Session Comparison Matrix

| Session | Steps | Envs | Obs Dims | Base Movement | Tracking Error | Explained Var | Notes |
|---------|-------|------|----------|---------------|----------------|---------------|-------|
| 1 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | Baseline |
| 2 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 3 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 4a | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 4b | 25.7M | [TBD] | 55 | ~0 mm | ~2.3m | 0.958 | Stopped - no base movement |
| 5 | 0M | 4096/8192 | 59 | **TBD** | **TBD** | **TBD** | **Observation space fix** |

---

## 🔧 Hardware & Environment Info

**System:**
- GPU: NVIDIA RTX 3090 (24GB VRAM)
- OS: Windows
- Isaac Lab: I:\isaaclab
- Isaac Sim: 5.0
- Workspace: C:\Users\yanbo\wSpace\cinebotRL

**Performance Benchmarks:**
- 4096 envs: ~8.3M steps/hour (~12 hours for 100M)
- 8192 envs: ~11M steps/hour (~9 hours for 100M)
- Memory usage: 18-20 GB (4096 envs), 22-23 GB (8192 envs)

---

## 📚 Key Documentation

### Session-Specific
- [Session 5 Launch Log](docs/training_sessions/SESSION_5_LAUNCH_LOG.md) - Detailed tracking
- [Session 5 Fix Documentation](docs/training_sessions/SESSION_5_BASE_FIX.md) - Fixes explained
- [Base Movement Bug Analysis](docs/BASE_MOVEMENT_BUG_ANALYSIS.md) - 10-page investigation

### General
- [Training Architecture](docs/architecture/training_architecture.md)
- [Reward System](docs/reference/reward_system.md)
- [Reward Cheatsheet](docs/reference/reward_cheatsheet.md)
- [Robot Constraints](docs/reference/robot_constraints_updated.md)
- [Training Diary](docs/training_sessions/TRAINING_DIARY.md)

### Setup Guides
- [Train on Windows](docs/setup/TRAIN_ON_WINDOWS.md)
- [Isaac Lab Windows Setup](docs/setup/isaaclab_windows.md)
- [Daily Workflow](docs/workflows/daily_workflow.md)

---

## 🎯 Success Metrics Definitions

### Base Action Activity
- **base_vx_action_std > 0.3**: Policy actively using forward velocity actions
- **base_wz_action_std > 0.2**: Policy actively using rotation actions
- **base_vel_x_mean > 0.01 m/s**: Real movement, not numerical noise

### Training Health
- **explained_variance > 0.80**: Value function learning properly
- **value_loss < 1.0**: Value predictions accurate
- **approx_kl < target_kl**: Policy updates stable

### Task Performance
- **mean_tracking_error < 0.3m**: Good end-effector tracking
- **total_reward > 35**: Net positive rewards (successful behavior)
- **base_mobilization_active > 70%**: Using base when needed

---

## 🚨 Common Issues & Resolutions

### Issue 1: No Base Movement (Session 4b)
**Symptoms:** Zero base action despite good explained variance  
**Root Cause:** Missing base-to-target observations  
**Resolution:** Added 4 observation dims in Session 5

### Issue 2: OOM (Out of Memory)
**Symptoms:** Training crashes, CUDA out of memory  
**Resolution:** Reduce num_envs from 8192 to 4096

### Issue 3: NaN Values
**Symptoms:** NaN in loss or rewards  
**Resolution:** Check for gradient explosion, reduce learning rate

---

## 📝 Template for New Session

```markdown
## Session X: [Session Name]

**Status:** [🚀 Active / ✅ Complete / ⏸️ Paused / ❌ Failed]  
**Date:** [Date]  
**Objective:** [Primary goal]

### Git Commits

#### Commit 1: [Commit Title]
```
Commit: [Git hash]
Date: [Date]
Message: [Commit message]
```

**Changes:**
- [Change 1]
- [Change 2]

**Files Modified:**
- [File 1]
- [File 2]

---

### Training Configuration

**Environment:**
- Task: [Task name]
- Trajectory Type: [Type]
- Rendering: [Headless/GUI]

**Hyperparameters:**
- Total Timesteps: [Number]
- Number of Environments: [Number]
- Learning Rate: [Value]
- [Other params]

### Launch Command

```powershell
[Full command]
```

**Command Used:** [Timestamp and confirmation]

### Results

**Status:** [In Progress / Complete / Failed]

**Final Metrics:**
- [Metric 1]: [Value]
- [Metric 2]: [Value]

**Outcome:** [Success / Failure / Partial]

**Detailed Log:** [Link to session-specific log]

---
```

---

**Last Updated:** October 21, 2025  
**Current Session:** 5 (Base Mobilization Fix)  
**Repository Status:** All changes committed, ready to launch
