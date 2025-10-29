# Session 8 Training Configuration Guide
**Prepared**: October 29, 2025  
**Based on**: Session 7d evaluation results (200M timesteps)  
**Target**: Fix catastrophic tracking failures with corrected reward weights

---

## 🎯 Session 7d Performance Analysis

### Current Issues (Session 7d - 200M timesteps)
❌ **Position Error**: 3.64m mean (18× worse than 0.2m target)  
❌ **Orientation Error**: 140.7° mean (14× worse than 10° target)  
❌ **Velocity Violations**: -15.5 penalty per step (massive)  
❌ **Jerk Penalties**: -14.0 penalty per step (huge)  
❌ **Base Immobility**: < 0.01 m/s (robot acts like fixed-base arm)  

### Root Cause
**Reward Imbalance**: Penalties (-40/step) completely overwhelm rewards (+0.07/step)
- Policy learned to **minimize penalties** (stand still, don't move)
- Instead of: **maximize tracking** (move base + arm smoothly to follow targets)

---

## 🔧 Recommended Changes for Session 8

### Strategy Overview
1. **Dramatically increase tracking rewards** (make them dominant)
2. **Significantly reduce motion penalties** (encourage exploration)
3. **Add base mobilization incentives** (overcome immobility)
4. **Implement curriculum learning** (easy → hard trajectories)

---

## 📊 Detailed Reward Weight Changes

### Current Weights (Session 7d)
```python
class RewardWeights:
    # Tracking rewards
    position_tracking: float = 100.0       # PRIMARY REWARD
    orientation_tracking: float = 2.0      # ❌ TOO SMALL (0.02× position!)
    progress_bonus: float = 1.0
    base_progress_reward: float = 250.0
    base_target_alignment: float = 10.0
    target_distance_penalty: float = 3.0
    
    # Motion quality penalties
    action_magnitude: float = 0.005
    action_rate: float = 0.01
    action_smoothness: float = 0.15
    
    # Constraint violation penalties
    velocity_limit_penalty: float = 5.0    # ❌ CAUSING -15.5/step!
    acceleration_limit_penalty: float = 5.0
    jerk_limit_penalty: float = 0.05       # ❌ CAUSING -14.0/step!
    joint_limit_penalty: float = 10.0
    lateral_motion_penalty: float = 2.0
    
    # Safety penalties
    self_collision_penalty: float = 0.5
    stability_penalty: float = 0.1
```

### Proposed Weights (Session 8)
```python
class RewardWeights:
    # ========================================
    # TRACKING REWARDS (Make these DOMINANT)
    # ========================================
    position_tracking: float = 150.0       # INCREASED 100→150 (50% boost)
                                           # Make position tracking more valuable
    
    orientation_tracking: float = 75.0     # 🔥 INCREASED 2.0→75.0 (37.5× boost!)
                                           # Now 50% of position weight (was 2%!)
                                           # Robot MUST learn to respect orientation
    
    progress_bonus: float = 5.0            # INCREASED 1.0→5.0 (5× boost)
                                           # Reward steady progress along trajectory
    
    base_progress_reward: float = 400.0    # INCREASED 250→400 (60% boost) 🔥
                                           # This scales the existing base_mobilization_reward() function
                                           # Strong incentive for strategic base movement toward targets
    
    base_target_alignment: float = 30.0    # INCREASED 10→30 (3× boost)
                                           # Reward moving toward target (goal-directed)
    
    # NOTE: No new "base_mobilization_reward" config field needed!
    # The existing base_mobilization_reward() function (rewards.py line 75)
    # is already scaled by base_progress_reward above.
    
    target_distance_penalty: float = 1.0   # REDUCED 3.0→1.0 (67% reduction)
                                           # Less punishment for being far = more exploration
    
    excessive_base_movement_penalty: float = 5.0  # REDUCED 10→5 (50% reduction)
                                                   # Allow more exploration
    
    # ========================================
    # MOTION QUALITY PENALTIES (Reduce these)
    # ========================================
    action_magnitude: float = 0.002        # REDUCED 0.005→0.002 (60% reduction)
                                           # Don't penalize large actions as much
    
    action_rate: float = 0.005             # REDUCED 0.01→0.005 (50% reduction)
                                           # Allow faster action changes
    
    action_smoothness: float = 0.05        # REDUCED 0.15→0.05 (67% reduction)
                                           # Was causing -1.72/step in Session 7d
                                           # Need less harsh penalty on action changes
    
    # ========================================
    # CONSTRAINT VIOLATIONS (Gentler penalties)
    # ========================================
    velocity_limit_penalty: float = 1.5    # 🔥 REDUCED 5.0→1.5 (70% reduction!)
                                           # Was causing -15.5/step in Session 7d
                                           # Still penalize but much gentler
    
    acceleration_limit_penalty: float = 1.5  # REDUCED 5.0→1.5 (70% reduction)
                                             # Match velocity penalty reduction
    
    jerk_limit_penalty: float = 0.01       # 🔥 REDUCED 0.05→0.01 (80% reduction!)
                                           # Was causing -14.0/step in Session 7d
                                           # Function only penalizes violations (jerk > max_jerk)
                                           # Need to reduce weight, not increase it!
    
    joint_limit_penalty: float = 5.0       # REDUCED 10.0→5.0 (50% reduction)
                                           # Softer joint limit enforcement
    
    lateral_motion_penalty: float = 1.0    # REDUCED 2.0→1.0 (50% reduction)
                                           # Less harsh on unintended lateral motion
    
    # ========================================
    # SAFETY PENALTIES (Keep reasonable)
    # ========================================
    self_collision_penalty: float = 1.0    # INCREASED 0.5→1.0 (2× boost)
                                           # Slightly more important than before
    
    collision_penalty: float = 10.0        # KEEP (external collisions - not used yet)
    
    stability_penalty: float = 0.2         # INCREASED 0.1→0.2 (2× boost)
                                           # Slightly more important
```

---

## 💡 Key Design Principles

### 1. **Reward/Penalty Ratio**
**Session 7d**: 
- Total tracking rewards: ~+28/step
- Total penalties: ~-40/step
- **Ratio: 0.7:1** ❌ (penalties win!)

**Session 8 Target**:
- Total tracking rewards: ~+80/step (position + orientation + progress)
- Total penalties: ~-5/step (gentler constraints)
- **Ratio: 16:1** ✅ (rewards dominate!)

### 2. **Orientation Weight Justification**
**Why 75.0 (50% of position)?**
- Film production needs: Orientation is **almost as important** as position
- Camera must point at subject, not just be at correct location
- Session 7d showed 140.7° error = robot pointing backwards!
- At 75.0 weight:
  - 10° error → penalty ~-5
  - 180° error → penalty ~-100
  - This makes orientation **unmissable** for the policy

### 3. **Velocity Penalty Reduction**
**Why reduce from 5.0 to 1.5?**
- Session 7d: Robot hitting limits constantly (-15.5/step)
- This was **correct behavior** - robot tried hard to catch up
- But penalty was so severe, policy learned to not move at all
- At 1.5: Still penalized, but won't dominate reward signal

### 4. **Base Mobilization Incentive**
**How it works in the existing code:**
- The reward function `base_mobilization_reward()` (rewards.py line 75) already exists
- It's scaled by the config parameter `base_progress_reward` (currently 250.0)
- **Session 8 increases this from 250.0 → 400.0** to boost base movement
- The function rewards progress toward out-of-reach targets (not just any movement)
- Formula: `reward = base_progress_reward × progress × out_of_reach_mask`
- Session 7d contribution: Only +0.49/step (too small!)
- Session 8 target: +2-5/step (by increasing weight 60%)

**Why not add a separate "movement incentive"?**
- Existing function already handles this correctly
- Just needs higher weight to overcome penalties
- Adding another reward term would complicate tuning

---

## 📈 Curriculum Learning (Recommended)

### Phase 1: Easy Trajectories (0-50M steps)
**Goal**: Learn basic tracking with minimal base movement required

**Trajectories**: 
- Filter to only trajectories with `max_distance_from_start < 1.0m`
- Use `chassis_required_indices.txt` subset (easier trajectories)
- ~30-40% of full trajectory set

**Reward Adjustments**:
- `position_tracking: 200.0` (even higher for easy cases)
- `orientation_tracking: 100.0` (also boosted)
- `velocity_limit_penalty: 1.0` (very gentle)

**Success Criteria**:
- Mean position error < 50cm
- Mean orientation error < 30°
- Mean reward > -500

### Phase 2: Medium Trajectories (50M-100M steps)
**Goal**: Add trajectories requiring moderate base movement

**Trajectories**:
- Add trajectories with `max_distance_from_start < 2.0m`
- ~60-70% of full trajectory set

**Reward Adjustments**:
- Transition to full Session 8 weights
- Start reducing scaffolding

**Success Criteria**:
- Mean position error < 30cm
- Mean orientation error < 20°
- Mean reward > -1000

### Phase 3: All Trajectories (100M-200M steps)
**Goal**: Handle full difficulty including distant targets

**Trajectories**:
- Use all 1,038 trajectories
- Full cinematographic complexity

**Reward Adjustments**:
- Use full Session 8 weights
- No scaffolding

**Success Criteria**:
- Mean position error < 20cm ✅
- Mean orientation error < 10° ✅
- Mean reward > 0 ✅
- Ready for deployment!

---

## 🔬 Hyperparameter Changes

### Learning Rate Schedule
```python
# Session 7d used constant 3e-4
# Session 8: Use learning rate annealing for better convergence

learning_rate: 3e-4  # Initial
lr_schedule: "linear"  # Anneal to 0 over training
```

### PPO Hyperparameters (Keep stable)
```python
# These worked well in Session 7d, keep them:
n_steps: 16  # Rollout length
batch_size: 512  # Mini-batch size
n_epochs: 4  # PPO epochs per update
gamma: 0.99  # Discount factor
gae_lambda: 0.95  # GAE parameter
clip_range: 0.2  # PPO clip range
ent_coef: 0.001  # Entropy coefficient (mild exploration)
vf_coef: 0.5  # Value function coefficient
max_grad_norm: 0.5  # Gradient clipping
```

### Episode Length (Consider reducing)
```python
# Session 7d: 20s episodes = 400 steps @ 20Hz
# Option: Reduce to 15s = 300 steps for faster iteration
episode_length_s: 15.0  # OPTIONAL: Faster training cycles
```

---

## 🚀 Implementation Steps

### Step 1: Update Config File
```bash
# Edit src/rl_platform/tasks/mobile_mm/config.py
# Update RewardWeights class with Session 8 weights
```

### Step 2: Add Base Mobilization Reward
```python
# In src/rl_platform/tasks/mobile_mm/rewards.py
# Add new reward component:

def compute_base_mobilization_reward(
    base_lin_vel: torch.Tensor,
    max_linear_velocity: float,
    weight: float = 20.0
) -> torch.Tensor:
    """Reward robot for moving base (overcome learned immobility).
    
    Args:
        base_lin_vel: Base linear velocity [num_envs, 3]
        max_linear_velocity: Maximum linear velocity (e.g., 1.5 m/s)
        weight: Reward weight
        
    Returns:
        Reward tensor [num_envs]
    """
    # Compute planar speed (ignore Z)
    base_speed = torch.norm(base_lin_vel[:, :2], dim=-1)
    
    # Reward when moving > 5 cm/s, proportional to speed
    threshold = 0.05  # 5 cm/s
    normalized_speed = base_speed / max_linear_velocity
    
    reward = torch.where(
        base_speed > threshold,
        weight * normalized_speed,
        torch.zeros_like(base_speed)
    )
    
    return reward
```

### Step 3: Curriculum Learning Setup
```python
# Create trajectory filter script
# scripts/reinforcement_learning/sb3/filter_trajectories_by_difficulty.py

import json
from pathlib import Path
import numpy as np

def analyze_trajectory_difficulty(traj_file):
    """Compute difficulty metric (max distance from start)."""
    with open(traj_file) as f:
        traj = json.load(f)
    
    positions = np.array([wp['position'] for wp in traj['waypoints']])
    start_pos = positions[0]
    distances = np.linalg.norm(positions - start_pos, axis=1)
    max_dist = np.max(distances)
    
    return max_dist

# Filter and create curriculum files
easy_threshold = 1.0  # meters
medium_threshold = 2.0  # meters

# Generate trajectory lists for each phase
# easy_trajectories.txt
# medium_trajectories.txt
# all_trajectories.txt
```

### Step 4: Launch Training
```powershell
# Phase 1: Easy trajectories (0-50M)
.\scripts\launch_training_windows.ps1 `
    -Task MobileMMTrackEE-v0 `
    -NumEnvs 128 `
    -MaxSteps 50000000 `
    -TrajectoryFile "trajectory_lists/easy_trajectories.txt" `
    -Headless `
    -SessionName "session_8a_curriculum_phase1"

# Phase 2: Medium trajectories (50M-100M)
# Load checkpoint from Phase 1 and continue with medium trajectories

# Phase 3: All trajectories (100M-200M)
# Load checkpoint from Phase 2 and continue with all trajectories
```

---

## 📊 Expected Results

### Session 8a (Phase 1 - Easy, 50M steps)
**Target**:
- Position error: < 50cm mean
- Orientation error: < 30° mean
- Base movement: > 0.1 m/s typical

### Session 8b (Phase 2 - Medium, 100M steps total)
**Target**:
- Position error: < 30cm mean
- Orientation error: < 20° mean
- Base movement: smooth, goal-directed

### Session 8c (Phase 3 - All, 200M steps total)
**Target**:
- Position error: **< 20cm** mean ✅ **(DEPLOYMENT READY)**
- Orientation error: **< 10°** mean ✅ **(DEPLOYMENT READY)**
- Velocity penalty: < 2.0 per step
- Jerk penalty: < 1.0 per step
- Mean reward: **> 0** ✅

---

## ⚠️ Monitoring & Checkpoints

### Key Metrics to Watch During Training

1. **Episode Reward Trend**
   - Should be **positive** and **increasing**
   - If stuck at negative: Check reward balance

2. **Reward Components (TensorBoard)**
   - `position_tracking`: Should be ~+100 to +150
   - `orientation_tracking`: Should be ~+50 to +75
   - `velocity_limit_penalty`: Should be **< 2.0** (not -15.5!)
   - `jerk_penalty`: Should be **< 1.0** (not -14.0!)

3. **Base Mobility Indicators**
   - `base_vel_x_mean`: Should be **> 0.1** m/s
   - `base_vel_x_max`: Should reach ~0.5-1.0 m/s
   - If still ~0.01: Increase `base_mobilization_reward` to 50.0

4. **Orientation Error**
   - Track this explicitly in logs
   - Should decrease from 140° → 30° → 10°
   - If stuck > 90°: Increase `orientation_tracking` to 100.0

### Evaluation Checkpoints
Run evaluation every 25M steps:
```powershell
.\scripts\launch_evaluation_quantitative.ps1 `
    -Checkpoint "logs\sb3\mobilemmtrackee_v0\session_8a\checkpoint_25M.zip" `
    -NumEpisodes 100 `
    -Headless
```

---

## 🎬 Success Criteria for Deployment

**Minimum Requirements** (from ANALYSIS_REPORT.md):
- ✅ Mean position error: **< 20 cm**
- ✅ Median position error: **< 10 cm**
- ✅ P95 position error: **< 50 cm**
- ✅ Mean orientation error: **< 10°**
- ✅ Velocity penalty: **< 2.0**
- ✅ Jerk penalty: **< 1.0**
- ✅ Success rate: **> 75%**

**Good Performance** (target):
- ⭐ Mean position error: **< 10 cm**
- ⭐ Mean orientation error: **< 5°**
- ⭐ P95 position error: **< 20 cm**
- ⭐ Success rate: **> 90%**

---

## 📝 Files to Modify

1. **`src/rl_platform/tasks/mobile_mm/config.py`**
   - Update `RewardWeights` class with Session 8 values

2. **`src/rl_platform/tasks/mobile_mm/rewards.py`**
   - Add `compute_base_mobilization_reward()` function
   - Update `compute_combined_reward()` to include it

3. **`src/rl_platform/tasks/mobile_mm/env.py`** (optional)
   - Add episode_length_s: 15.0 if reducing episode length

4. **New: `scripts/reinforcement_learning/sb3/train_curriculum.py`** (optional)
   - Script to handle curriculum phases automatically
   - Auto-switch trajectory files at milestone steps

---

## 🚦 Go/No-Go Decision

**Proceed to Session 8 if**:
- ✅ Reward weights updated in config.py
- ✅ Base mobilization reward implemented
- ✅ Curriculum trajectory lists prepared
- ✅ Evaluation system tested and working
- ✅ Session 7d results committed to git

**Do NOT proceed if**:
- ❌ Reward balance still favors penalties
- ❌ No mechanism to overcome base immobility
- ❌ No plan for curriculum or trajectory filtering
- ❌ Evaluation system not validated

---

## 📚 Related Documents

- **Session 7d Results**: `evaluation_results/20251028_200923/ANALYSIS_REPORT.md`
- **Current Config**: `src/rl_platform/tasks/mobile_mm/config.py`
- **Reward Implementation**: `src/rl_platform/tasks/mobile_mm/rewards.py`
- **Evaluation System**: `scripts/reinforcement_learning/sb3/EVALUATION_README.md`
- **Training Launcher**: `scripts/launch_training_windows.ps1`

---

**Status**: ⏸️ **READY TO IMPLEMENT**  
**Next Action**: Update config.py with Session 8 weights and implement base mobilization reward  
**Expected Start**: After config updates and testing  
**Expected Duration**: 200M steps @ ~15-20 hours (similar to Session 7d)
