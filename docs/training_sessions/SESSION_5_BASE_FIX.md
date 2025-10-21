# Training Session 5: Base Mobilization Fix

**Date:** October 21, 2025  
**Session:** 5  
**Previous:** Session 4b (25.7M/100M steps, 0.958 explained variance, ZERO base movement)

---

## 🎯 Mission

Restart training from scratch with **CRITICAL BUG FIXES** to enable base mobilization learning.

---

## 🐛 Root Cause Analysis

### Trajectory Requirements (DISCOVERED VIA ANALYSIS)
```
Trajectory: 1_pull_world_scaled.json
- Total waypoints: 210
- Within arm reach (<0.6m): 16.2% (34 waypoints)
- Out of reach (>0.6m): 83.8% (176 waypoints)  ← BASE MOVEMENT REQUIRED!
- Mean distance from base: 2.3 meters (4x arm reach!)
- Max distance: 4.0 meters
- Longest consecutive OOR: 176 waypoints (17.6 seconds!)

CONCLUSION: Trajectory ABSOLUTELY REQUIRES base movement to succeed!
```

### Why Policy Didn't Learn (25.7M Steps)
1. **Missing observations:** No explicit base-to-target distance → complex inference required
2. **Reward imbalance:** Base movement (-0.5 net) vs arm-only (+40) → arm-only wins
3. **Harsh penalties:** Distance penalty (-14) + action penalties (-1.5) = net negative even when moving!

### Training Metrics (Session 4b)
- ✅ Explained variance: 0.958 (near-perfect value function!)
- ✅ Value loss: 0.00443 (extremely low)
- ❌ Base movement: ~0mm (policy learned "I'll get low rewards" but not "how to fix it")

**Key insight:** Policy CORRECTLY learned to predict poor performance, but observation space didn't provide information needed to improve!

---

## ✅ Fixes Applied

### Fix #1: Explicit Base-to-Target Observations (+4 dims)
```python
# In observations.py - NOW policy sees:
base_to_target_xy: [dx, dy]  # 2 dims: XY offset to target
base_to_target_dist: float    # 1 dim: Euclidean distance
out_of_reach_flag: bool        # 1 dim: Binary signal when >0.6m

# Old observation: 55 dims
# New observation: 59 dims (with lookahead)
```

**Before:** Policy had to infer "should I move base?" through complex reasoning  
**After:** Policy sees explicit signal: `out_of_reach_flag=1` → MOVE BASE!

### Fix #2: Rebalanced Rewards
```python
# Reward changes in config.py:
base_progress_reward: 50.0 → 150.0  # 3x stronger! (overcome penalties)
target_distance_penalty: 10.0 → 3.0  # Less harsh (was killing movement)
action_magnitude: 0.01 → 0.005  # Encourage base exploration
jerk_limit_penalty: 0.1 → 0.05  # Reduce movement suppression
```

**New reward analysis:**
- Target out of reach + base moves 0.1m closer:
  - Base mobilization: +15.0 (was +5.0)
  - Distance penalty: -0.6 (was -4.0, with 90% reduction while moving!)
  - Action penalties: -0.8
  - **Net: +13.6** (was -0.5) ← NOW POSITIVE!

### Fix #3: Smart Distance Penalty
```python
# In rewards.py - Intelligent penalty reduction:
if base_is_moving:
    penalty *= 0.1  # 90% reduction!
else:
    penalty *= 1.0  # Full penalty
```

**Encourages exploration:** "Try moving base → penalty drops 90%!"

---

## 🚀 Training Configuration

### Environment
- **Task:** MobileMMTrackEE-v0
- **Num envs:** 8192 (maximized for RTX 3090)
- **Episode length:** 400 steps (20s @ 20Hz)
- **Control frequency:** 20Hz (0.05s timestep)

### Physics (From Session 4)
- Spring: k=1000 N/m
- Damping: ζ=0.5 (underdamped, 96% in 1 step)
- Base mass: 20kg
- PPR helpers: 1.0kg

### Algorithm (PPO)
- **Total timesteps:** 100M (reset from 0)
- **n_steps:** 64
- **batch_size:** 256
- **learning_rate:** 3e-4
- **Entropy decay:** 0.001→0.0001 (50M-100M)
- **KL schedule:** warmup 0.25, main 0.15, finetune 0.07

### Expected Metrics
- **FPS:** ~3000-3200 (8192 envs, RTX 3090)
- **Training time:** ~9 hours (100M / 3100 FPS / 3600 s/hr)
- **GPU memory:** ~4-5 GB / 24 GB

---

## 📊 Success Criteria

### Early Signs (First 1M steps, ~5 minutes)
- [ ] Base action std > 0.3 (active use of base DOFs)
- [ ] Base velocity mean > 0.01 m/s (real movement, not noise)
- [ ] base_mobilization reward mean > 1.0 (policy using reward)

### Intermediate (10M steps, ~1 hour)
- [ ] Base movement per step > 0.005m average
- [ ] Explained variance > 0.80
- [ ] Position tracking reward increasing
- [ ] Distance penalty decreasing

### Final (100M steps, ~9 hours)
- [ ] Base mobilizes for OOR targets (>0.6m)
- [ ] EE tracking error < 0.15m average
- [ ] Explained variance > 0.92
- [ ] Total reward > 30 average

---

## 🔍 Monitoring Plan

### Every 50 steps (via enhanced display):
- Overall statistics (health breakdown %)
- Base movement metrics (velocity, actions)
- Reward components (especially base_mobilization)
- 3 random + 1 best + 1 worst env samples

### Key metrics to watch:
```python
# Base action activity
base_vx_action_std  # Should be >0.3 (active)
base_wz_action_std  # Should be >0.2 (active)

# Base movement
base_vel_x_mean  # Should be >0.01 m/s
base_movement_per_step  # Should be >0.005m

# Reward balance
base_mobilization_mean  # Should be >1.0 (policy using it!)
position_tracking_mean  # Should increase over time
distance_penalty_mean  # Should decrease (getting closer!)
```

---

## 🚨 Failure Indicators

If after 5M steps (15 minutes):
- ❌ base_vx_action_std < 0.1 → Base actions not being used
- ❌ base_mobilization_mean < 0.1 → Reward signal too weak
- ❌ distance_penalty_mean > -5.0 → Still far from targets
- ❌ explained_variance < 0.5 → Policy not learning value function

**Action:** Stop training, investigate further (gradient flow, action masking, etc.)

---

## 📝 Launch Commands

### PowerShell (Preferred):
```powershell
.\scripts\launch_training_windows.ps1 `
  -Task MobileMMTrackEE-v0 `
  -NumEnvs 8192 `
  -Headless `
  -TotalTimesteps 100000000
```

### Direct Isaac Lab:
```powershell
I:\isaaclab\isaaclab.bat -p scripts/reinforcement_learning/sb3/train.py `
  --task MobileMMTrackEE-v0 `
  --num_envs 8192 `
  --headless true `
  --total_timesteps 100000000
```

---

## 📈 Expected Timeline

| Time | Steps | Event |
|------|-------|-------|
| 0:00 | 0 | Training start |
| 0:05 | 1M | First checkpoint - verify base actions active |
| 0:15 | 3M | Early learning - base mobilization emerging |
| 1:00 | 10M | Checkpoint - base movement established |
| 3:00 | 30M | Entropy decay begins (0.001→0.0001) |
| 5:00 | 50M | Mid-training checkpoint |
| 7:00 | 70M | KL finetune phase (0.15→0.07) |
| 9:00 | 100M | Training complete |

---

## 🎯 Predicted Outcome

Based on fixes:
1. **Early learning (1-10M):** Policy discovers base mobilization gives positive reward
2. **Skill acquisition (10-50M):** Learns when/how to move base efficiently
3. **Refinement (50-100M):** Fine-tunes coordination between base and arm

**Expected final performance:**
- EE tracking error: 0.10-0.15m average (down from 2.3m starting)
- Base mobilization: Active for >70% of OOR waypoints
- Total reward: 35-45 average (positive!)
- Explained variance: 0.92-0.96 (excellent value function)

---

## 📚 Reference Documents

- `docs/BASE_MOVEMENT_BUG_ANALYSIS.md` - Complete investigation (10 pages)
- `docs/TRAJECTORY_TRACKING_IMPROVEMENTS.md` - Architecture analysis (15 pages)
- `scripts/analyze_trajectory_reach.py` - Trajectory analysis tool
- `docs/training_sessions/TRAINING_DIARY.md` - Historical log

---

**Status:** Ready to launch! 🚀  
**Next:** Run training command and monitor for first 5 minutes to confirm base actions are active.
