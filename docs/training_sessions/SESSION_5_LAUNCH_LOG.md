# Training Session 5 - Launch Log

**Date:** October 21, 2025  
**Session ID:** Session_5_Base_Mobilization_Fix  
**Status:** Ready to Launch

---

## 🎯 Session Objectives

1. **Validate base mobilization fixes** (observation space + reward rebalancing)
2. **Achieve >0.3 base action std** within first 5 minutes (1M steps)
3. **Reduce mean tracking error** from 2.3m to <0.3m by end of training
4. **Complete 100M timesteps** (~9-12 hours depending on envs)

---

## 🔧 Configuration Summary

### Environment Configuration
- **Task:** `MobileMMTrackEE-v0`
- **Trajectory Type:** `multi_recorded` (all trajectories)
- **Trajectory Filter:** All trajectories enabled (`--use_all_trajectories`)
- **Rendering Mode:** Headless (`--headless`)

### Training Hyperparameters
- **Total Timesteps:** 100,000,000 (100M)
- **Number of Environments:** 4096 (or 8192 - TBD)
- **Steps per Rollout:** 64
- **Batch Size:** 256 (or 512 if 8192 envs)
- **Learning Rate:** 3e-4 (0.0003)

### PPO-Specific Settings
- **Initial Entropy Coefficient:** 0.001
- **Enable Entropy Decay:** Yes
  - Final Entropy: 1e-4 (0.0001)
  - Decay Start: 50M timesteps
  - Decay Duration: 50M timesteps
- **Target KL:** 1.0
- **Enable KL Schedule:** Yes
  - KL Warmup: 0.25
  - KL Main: 0.15
  - KL Finetune: 0.07

### Observation Space Changes (CRITICAL)
- **Previous Dimensions:** 55 dims (with lookahead)
- **New Dimensions:** 59 dims (with lookahead)
- **Added Features:**
  - `base_to_target_xy` [2 dims]: XY offset from base to target
  - `base_to_target_dist` [1 dim]: Euclidean distance to target
  - `out_of_reach_flag` [1 dim]: Binary signal when target >0.6m from base

### Reward Configuration Changes
- **base_progress_reward:** 50.0 → 150.0 (3x stronger!)
- **target_distance_penalty:** 10.0 → 3.0 (less harsh during movement)
- **action_magnitude:** 0.01 → 0.005 (encourage base action exploration)
- **jerk_limit_penalty:** 0.1 → 0.05 (reduce movement suppression)
- **Smart distance penalty:** 90% reduction when base is moving (>1cm threshold)

---

## 💻 Launch Commands

### Option A: 4096 Environments (Safer, Tested)
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

**Estimated Duration:** ~12 hours  
**Memory Usage:** ~18-20 GB VRAM  
**Throughput:** ~8.3M steps/hour

---

### Option B: 8192 Environments (Maximum Throughput)
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

**Estimated Duration:** ~9 hours  
**Memory Usage:** ~22-23 GB VRAM (may be tight on RTX 3090)  
**Throughput:** ~11M steps/hour

---

## 🎯 Success Criteria

### Early Validation (1M steps, ~5 minutes)
- [ ] `base_vx_action_std > 0.3` (policy actively using base forward velocity)
- [ ] `base_wz_action_std > 0.2` (policy actively using base rotation)
- [ ] `base_vel_x_mean > 0.01 m/s` (real movement, not just noise)
- [ ] `base_mobilization_reward > 1.0` (reward signal working)

### Intermediate Validation (10M steps, ~1 hour)
- [ ] `base_movement_per_step > 0.005m` (5mm average movement)
- [ ] `explained_variance > 0.80` (value function learning properly)
- [ ] `position_tracking_reward` increasing trend
- [ ] `target_distance_penalty` decreasing trend

### Final Success (100M steps, ~9-12 hours)
- [ ] `mean_tracking_error < 0.3m` (down from 2.3m initial)
- [ ] `base_mobilization_active > 70%` of time when target out of reach
- [ ] `total_reward > 35` average (positive rewards!)
- [ ] `explained_variance > 0.92` (strong value function)

---

## 📊 Monitoring Plan

### TensorBoard Metrics to Watch
1. **Base Action Statistics:**
   - `rollout/base_vx_action_mean`
   - `rollout/base_vx_action_std`
   - `rollout/base_wz_action_mean`
   - `rollout/base_wz_action_std`

2. **Reward Components:**
   - `rollout/base_mobilization_reward`
   - `rollout/position_tracking_reward`
   - `rollout/target_distance_penalty`
   - `rollout/total_reward`

3. **Training Health:**
   - `train/explained_variance`
   - `train/value_loss`
   - `train/policy_loss`
   - `train/entropy_loss`
   - `train/approx_kl`

4. **Movement Statistics:**
   - `rollout/base_vel_x_mean`
   - `rollout/base_pos_x_std` (should increase over time)
   - `rollout/tracking_error_mean`

### Checkpoints to Monitor
- **5 minutes** (1M steps): Quick validation - stop if no base actions
- **30 minutes** (5M steps): Check base movement trends
- **1 hour** (10M steps): Intermediate validation
- **3 hours** (30M steps): Mid-training health check
- **6 hours** (60M steps): Compare with Session 4b at same point
- **9-12 hours** (100M steps): Final evaluation

---

## 🚨 Failure Indicators (Stop Training If...)

1. **Base actions still zero** after 5M steps (30 minutes)
   - Indicates observation fix didn't work or gradient flow issue
   
2. **Explained variance < 0.5** after 10M steps
   - Value function not learning, potential numerical instability
   
3. **Total reward still negative** after 20M steps
   - Reward structure still not incentivizing correct behavior
   
4. **Training crashes** due to OOM
   - Reduce num_envs from 8192 to 4096
   
5. **NaN values** in loss or rewards
   - Numerical instability, check for exploding gradients

---

## 🔄 Changes From Session 4b

| Aspect | Session 4b | Session 5 |
|--------|-----------|-----------|
| **Observation Dims** | 55 | 59 (+4 base-to-target info) |
| **base_progress_reward** | 50.0 | 150.0 |
| **target_distance_penalty** | 10.0 | 3.0 |
| **action_magnitude** | 0.01 | 0.005 |
| **jerk_limit_penalty** | 0.1 | 0.05 |
| **Smart Distance Penalty** | No | Yes (90% reduction when moving) |
| **Total Timesteps** | 100M (stopped at 25.7M) | 100M (fresh start) |
| **Expected Base Movement** | 0 mm | >5 mm per step |

---

## 📝 Pre-Launch Checklist

- [x] All code changes committed to git
- [x] Observation space updated (+4 dims)
- [x] Reward weights rebalanced
- [x] Smart distance penalty implemented
- [x] Trajectory analysis completed (83.8% out of reach confirmed)
- [x] Documentation updated (BASE_MOVEMENT_BUG_ANALYSIS.md)
- [x] Session 5 documentation created (SESSION_5_BASE_FIX.md)
- [ ] GPU memory cleared (restart if needed)
- [ ] TensorBoard ready for monitoring
- [ ] Backup of Session 4b checkpoints (in case we need to compare)
- [ ] Launch command selected (4096 or 8192 envs)

---

## 📋 Post-Training Actions

After training completes:

1. **[ ] Analyze final metrics** - compare with Session 4b
2. **[ ] Visualize base trajectories** - did base move appropriately?
3. **[ ] Export checkpoint** for evaluation
4. **[ ] Run evaluation script** on held-out trajectories
5. **[ ] Update TRAINING_DIARY.md** with results
6. **[ ] Create Session 5 results summary** document
7. **[ ] Decide on next steps:**
   - If successful: Move to velocity tracking enhancement (Priority 2)
   - If failed: Investigate remaining issues (gradient flow, USD physics, etc.)

---

## 🗂️ Files Modified in This Session

1. `src/rl_platform/tasks/mobile_mm/observations.py` - Added base-to-target observations
2. `src/rl_platform/tasks/mobile_mm/config.py` - Rebalanced reward weights
3. `src/rl_platform/tasks/mobile_mm/rewards.py` - Smart distance penalty
4. `scripts/analyze_trajectory_reach.py` - Trajectory analysis tool (NEW)
5. `docs/BASE_MOVEMENT_BUG_ANALYSIS.md` - Investigation report (NEW)
6. `docs/training_sessions/SESSION_5_BASE_FIX.md` - Session documentation (NEW)
7. `docs/training_sessions/SESSION_5_LAUNCH_LOG.md` - This file (NEW)

---

## 💡 Expected Reward Improvement Per Step

When base moves 7.5cm (max speed 1.5 m/s × 0.05s timestep):

| Component | Change |
|-----------|--------|
| Base Mobilization Reward | +11.0 |
| Distance Penalty Improvement | +4.6 (90% reduction!) |
| Position Tracking Improvement | +0.1 |
| Action Penalties | -0.025 |
| **NET IMPROVEMENT** | **+15.7 points** 🚀 |

Compare to staying still: **-5.1 points** (distance penalty only)

**Conclusion:** Base movement is 20x more rewarding than staying stationary!

---

## 🎬 Launch Time

**Command Selected:** [TO BE FILLED BY USER]

**Actual Launch Time:** [TO BE FILLED]

**Initial Observations:** [TO BE FILLED AFTER 5 MINUTES]

---

## 📈 Training Progress Log

### Hour 0 (Launch)
- **Timestamp:** 
- **Status:** 
- **Notes:** 

### Hour 1 (~10M steps)
- **Timestamp:** 
- **Base Action Std:** 
- **Mean Reward:** 
- **Explained Variance:** 
- **Notes:** 

### Hour 3 (~30M steps)
- **Timestamp:** 
- **Base Movement:** 
- **Tracking Error:** 
- **Notes:** 

### Hour 6 (~60M steps)
- **Timestamp:** 
- **Status:** 
- **Notes:** 

### Final (100M steps)
- **Completion Time:** 
- **Final Metrics:** 
- **Success:** Yes/No
- **Next Steps:** 

---

**Last Updated:** October 21, 2025  
**Session Status:** READY TO LAUNCH 🚀
