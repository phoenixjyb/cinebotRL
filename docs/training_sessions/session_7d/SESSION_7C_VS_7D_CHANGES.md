# Session 7c vs 7d: Complete Change Summary

**Date:** 2025-10-28  
**Purpose:** Document all differences between Session 7c and Session 7d configurations

---

## 📊 Quick Comparison Table

| Aspect | Session 7c (Baseline) | Session 7d (Accelerated + Tuned) | Change |
|--------|----------------------|----------------------------------|--------|
| **Training Duration** | 100M timesteps (~11h) | 200M timesteps (~11h) | 2x longer training ⏱️ |
| **Environments** | 4,096 | 8,192 | 2x parallelism 🚀 |
| **n_steps** | 128 | 64 | Halved (maintain buffer) |
| **Batch Size** | 1024 | 1024 | Unchanged ✓ |
| **Rollout Buffer** | 524,288 timesteps | 524,288 timesteps | Unchanged ✓ |
| **Learning Rate** | 3e-4 | 3e-4 | Unchanged ✓ |
| **Expected Wall Time** | ~11 hours | ~11 hours | Same! ⚡ |

---

## 🎯 Category 1: Reward Weight Changes

### 1.1 Base Progress Reward (INCREASED)
```python
# Session 7c
base_progress_reward: float = 150.0

# Session 7d  
base_progress_reward: float = 250.0  # +67% boost
```

**Why:** 
- Session 7c mobilization rewards were too weak (0.0-2.3 pts observed)
- Max per-step reward: 150 × 0.2m = 30 pts (not enough to overcome penalties)
- **New max: 250 × 0.2m = 50 pts** (stronger incentive for strategic movement)

---

### 1.2 Base Target Alignment (NEW)
```python
# Session 7c
# (not implemented)

# Session 7d
base_target_alignment: float = 10.0  # NEW reward component
```

**What it does:**
- Rewards base moving in direction that brings target closer to reachable zone
- Computed in `rewards.py` line ~120+: `cos(angle) × distance × weight`
- **Goal:** Encourage goal-directed navigation, not just random movement

**Why needed:**
- Session 7c: Base moved (0.1-1.8m) but not toward targets
- Mobilization only triggered when target out of reach, no directional guidance
- **Alignment reward provides navigation signal** even when target already reachable

---

### 1.3 Target Distance Penalty (REDUCED)
```python
# Session 7c
target_distance_penalty: float = 5.0

# Session 7d
target_distance_penalty: float = 3.0  # -40% gentler
```

**Why:**
- Session 7c logs showed penalties of 13-20 pts (too harsh)
- High penalty discouraged exploration when target far away
- **Reduced penalty encourages trying to reach distant targets**

**Balance calculation:**
- Session 7c: 1m beyond reach → 5.0 pts penalty vs 30 pts max reward (unfavorable)
- Session 7d: 1m beyond reach → 3.0 pts penalty vs 50 pts max reward (favorable!)

---

### 1.4 Action Smoothness (INCREASED)
```python
# Session 7c
action_smoothness: float = 0.05

# Session 7d
action_smoothness: float = 0.15  # +3x
```

**Why:**
- Session 7c showed some jittery/oscillating behavior
- Stronger smoothness penalty → more fluid motion
- Helps with base stability during arm reaching

---

## ⚙️ Category 2: Training Hyperparameters

### 2.1 Total Timesteps (DOUBLED)
```python
# Session 7c
total_timesteps = 100_000_000  # 100M

# Session 7d
total_timesteps = 200_000_000  # 200M
```

**Why:**
- Session 7c plateaued early (~50M steps) due to poor reward balance
- Reward changes are significant → policy needs more time to adapt
- Entropy decay and KL schedule benefit from longer training
- **2x timesteps with 2x environments = same wall-clock time!**

---

### 2.2 Number of Environments (DOUBLED)
```python
# Session 7c
num_envs = 4096

# Session 7d
num_envs = 8192  # 2x parallelism
```

**Why:**
- Accelerate training: collect data 2x faster
- Compensate for 2x timesteps → same ~11 hour duration
- VRAM: 8192 × 3MB = ~24.6GB (fits in RTX 3090 with memory management)

**Safety:**
- Rollout buffer size maintained: 128 × 4096 = 64 × 8192 = 524K
- Sample efficiency unchanged (same gradient estimation quality)
- Standard PPO scaling practice

---

### 2.3 n_steps (HALVED)
```python
# Session 7c
n_steps = 128

# Session 7d
n_steps = 64
```

**Why:**
- Maintain rollout buffer size: n_steps × num_envs = 524K (unchanged)
- More environments → shorter rollout per env to keep memory constant
- **Key insight:** Sample efficiency depends on buffer size, not individual n_steps

---

### 2.4 Batch Size (UNCHANGED)
```python
# Session 7c
batch_size = 1024

# Session 7d  
batch_size = 1024  # Same
```

**Why keep same:**
- Rollout buffer size unchanged (524K)
- 1024 minibatch is appropriate for 524K buffer
- Could have increased to 2048, but kept conservative for stability

---

## 🔬 Category 3: Other Changes

### 3.1 Entropy Decay Settings (UNCHANGED)
```python
# Both Session 7c and 7d
ent_coef = 0.01
final_ent_coef = 0.0001
decay_start_timestep = 50_000_000
```

**Note for Session 7d:**
- Start decay at 50M (same absolute timestep)
- But this is now 25% of training (was 50% in Session 7c)
- More exploration time relative to total training duration

---

### 3.2 KL Schedule (UNCHANGED)
```python
# Both Session 7c and 7d
kl_warmup = 0.25      # Timesteps 0-50M
kl_main = 0.15        # Timesteps 50-160M  
kl_finetune = 0.07    # Timesteps 160M+
```

**Note for Session 7d:**
- Session 7c ended at 100M (during kl_main phase)
- Session 7d continues through 200M (enters kl_finetune phase)
- **Finetune phase with lower KL → more focused policy refinement**

---

### 3.3 Save Frequency (ADJUSTED)
```python
# Session 7c
save_freq = 100_000  # Every 100K timesteps

# Session 7d
save_freq = 2_048_000  # Every 2M timesteps
```

**Why:**
- Session 7c: 100M / 100K = 1,000 checkpoints (too many!)
- Session 7d: 200M / 2M = 100 checkpoints (reasonable)
- Saves disk space while keeping sufficient recovery points

---

## 📈 Expected Outcomes

### Session 7c Results (100M):
```
Reachability:        6% ❌
Mean Error:          1.01m ❌
Base Movement:       0.1-1.8m ✅
Mobilization Reward: 0.0-2.3 pts ❌
```

### Session 7d Targets (200M):
```
Reachability:        30-50% 🎯 (5-8x improvement)
Mean Error:          <0.70m 🎯 (30% reduction)
Base Movement:       0.5-2.5m 🎯 (more strategic)
Mobilization Reward: 10-30 pts 🎯 (10x higher)
```

**Key improvements expected:**
1. ✅ **Strategic base movement** (alignment reward guides navigation)
2. ✅ **Higher reachability** (stronger mobilization incentive)
3. ✅ **Lower tracking error** (more time to converge)
4. ✅ **Smoother motion** (3x action smoothness weight)

---

## 🔑 Key Insights

### 1. **Training Duration Doubled for Free**
- Session 7c: 100M @ 4096 envs = ~11 hours
- Session 7d: 200M @ 8192 envs = ~11 hours
- **Same wall-clock time, 2x learning budget!**

### 2. **Reward Balance is Critical**
Session 7c failed because:
- Mobilization reward (max 30 pts) < distance penalty (13-20 pts)
- No directional guidance (base moved but randomly)

Session 7d fixes:
- Mobilization reward (max 50 pts) > distance penalty (3-10 pts)
- Alignment reward provides navigation signal

### 3. **Sample Efficiency Maintained**
- Rollout buffer: 524K timesteps (unchanged)
- Learning rate: 3e-4 (unchanged)
- Batch size: 1024 (unchanged)
- **Doubling envs doesn't hurt training quality!**

### 4. **Longer Training Justified**
Session 7c plateaued not because it converged, but because:
- Reward structure was broken (penalties dominated)
- Policy couldn't find good strategy under those constraints

Session 7d with fixed rewards should:
- Continue improving beyond 100M
- Benefit from finetune KL phase (160M-200M)
- Reach better local optimum

---

## 🚀 Launch Commands Comparison

### Session 7c (Actual):
```powershell
I:\isaaclab\isaaclab.bat -p scripts/reinforcement_learning/sb3/train.py `
  --task MobileMMTrackEE-v0 `
  --num_envs 4096 `
  --n_steps 128 `
  --batch_size 1024 `
  --total_timesteps 100000000 `
  --learning_rate 3e-4 `
  --headless
```

### Session 7d (Accelerated):
```powershell
.\scripts\launch_session_7d_accelerated.ps1

# Or manually:
I:\isaaclab\isaaclab.bat -p scripts/reinforcement_learning/sb3/train.py `
  --task MobileMMTrackEE-v0 `
  --num_envs 8192 `
  --n_steps 64 `
  --batch_size 1024 `
  --total_timesteps 200000000 `
  --learning_rate 3e-4 `
  --headless
```

---

## 📝 Summary

**Session 7d is NOT just "Session 7c with more environments"!**

It combines:
1. **Reward tuning** (4 changes: stronger mobilization, new alignment, gentler penalty, smoother motion)
2. **Training extension** (100M → 200M for reward changes to converge)
3. **Acceleration** (4096 → 8192 envs to keep wall-clock time same)

**The acceleration is the HOW (train faster), not the WHY (fix reward structure).**

The primary goal is fixing Session 7c's low reachability (6%) by rebalancing rewards. The 8192 environments just make it feasible to do 200M timesteps in reasonable time (~11 hours instead of ~22 hours).
