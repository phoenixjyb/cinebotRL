# Training Tuning Plan - October 16, 2025

## Executive Summary

**Status:** Training paused after showing improvement but still unstable
**Progress:** Explained variance improved from -16.3 → +0.59 (positive trend!)
**Issue:** Critic instability, high clipping, insufficient exploration decay

**Strategy:** Apply targeted fixes to stabilize critic and improve policy learning

---

## 📊 Recent Training Analysis (Iterations 258-279)

### What's Working ✅
1. **Explained variance trending positive** (272-279: 0.3-0.59)
2. **No crashes** - all 5 bug fixes working
3. **Training progressing** - agent collecting experience

### What Needs Fixing ⚠️
1. **Explained variance unstable** (-16.3 to +0.59 fluctuations)
2. **Clip fraction too high** (11-18%, target: <10%)
3. **Entropy not decaying** (stuck at -9.6 to -9.9)
4. **Value loss too low** (~0.0005, critic not learning)
5. **Slow convergence** (policy gradient stable but weak)

---

## 🔧 Tuning Changes to Apply

### Priority 1: Enable Reward Normalization (CRITICAL)

**File:** `scripts/reinforcement_learning/sb3/train.py`

**Current code** (around line 270-280):
```python
# Create normalized wrapper
env = VecNormalize(
    env,
    norm_obs=True,
    norm_reward=False,  # ← CURRENTLY DISABLED
    clip_obs=10.0,
    gamma=0.99
)
```

**Change to:**
```python
# Create normalized wrapper with reward normalization
env = VecNormalize(
    env,
    norm_obs=True,
    norm_reward=True,      # ← ENABLE: Normalize rewards for stable critic learning
    clip_obs=10.0,
    clip_reward=10.0,      # ← ADD: Clip normalized rewards to ±10
    gamma=0.99,
    epsilon=1e-8           # ← ADD: Numerical stability
)
```

**Why:** Reward normalization is THE most impactful fix for critic instability. It stabilizes value function learning by scaling rewards to consistent range.

---

### Priority 2: Improve PPO Hyperparameters (HIGH IMPACT)

**File:** `scripts/reinforcement_learning/sb3/train.py`

**Current code** (around line 290-310):
```python
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=512,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
    device="cuda:0",
    tensorboard_log=log_dir
)
```

**Change to:**
```python
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,        # Keep: standard LR
    n_steps=4096,              # ← INCREASE: More data per update (was 2048)
    batch_size=256,            # ← DECREASE: More gradient updates (was 512)
    n_epochs=10,               # Keep: good balance
    gamma=0.99,                # Keep: standard discount
    gae_lambda=0.95,           # Keep: standard GAE
    clip_range=0.2,            # Keep: standard clipping
    clip_range_vf=1.0,         # ← ADD: Clip value function updates
    ent_coef=0.01,             # ← INCREASE: More exploration (was 0.0)
    vf_coef=0.5,               # Keep: value function coefficient
    max_grad_norm=0.5,         # Keep: gradient clipping
    target_kl=0.01,            # ← ADD: Early stopping if KL too large
    verbose=1,
    device="cuda:0",
    tensorboard_log=log_dir
)
```

**Why Each Change:**
- **n_steps=4096**: More experience per update → more stable GAE targets
- **batch_size=256**: Smaller batches → more gradient updates per epoch → better learning
- **clip_range_vf=1.0**: Prevents large value function updates → critic stability
- **ent_coef=0.01**: Encourages exploration → helps with entropy decay
- **target_kl=0.01**: Early stopping if policy changes too much → stability

---

### Priority 3: Add Learning Rate Schedule (MEDIUM IMPACT)

**File:** `scripts/reinforcement_learning/sb3/train.py`

**Add before model creation:**
```python
def linear_schedule(initial_value: float):
    """
    Linear learning rate schedule.
    
    :param initial_value: Initial learning rate.
    :return: schedule that computes current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0 (end).
        
        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value
    return func
```

**Then use it in model:**
```python
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=linear_schedule(3e-4),  # ← USE SCHEDULE instead of 3e-4
    # ... rest of parameters
)
```

**Why:** Gradually reduces learning rate → better convergence in later stages

---

### Priority 4: Improve Critic Network Architecture (OPTIONAL)

**File:** `scripts/reinforcement_learning/sb3/train.py`

**Add custom policy kwargs:**
```python
policy_kwargs = dict(
    net_arch=dict(
        pi=[256, 256, 256],  # Policy network: 3 layers of 256 units
        vf=[256, 256, 256]   # Value network: 3 layers of 256 units (was likely [64, 64])
    ),
    activation_fn=torch.nn.ReLU,
    ortho_init=True
)

model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,  # ← ADD
    # ... rest of parameters
)
```

**Why:** Larger value network → better value function approximation → more stable critic

---

## 📝 Implementation Steps

### Step 1: Apply Changes to Code

1. **Open:** `scripts/reinforcement_learning/sb3/train.py`
2. **Apply** Priority 1 (reward normalization) - lines ~270-280
3. **Apply** Priority 2 (PPO hyperparameters) - lines ~290-310
4. **Apply** Priority 3 (learning rate schedule) - add before model creation
5. **(Optional)** Apply Priority 4 (network architecture)

### Step 2: Test Configuration

**Quick test run (10 minutes):**
```powershell
cd I:\isaaclab
.\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py `
  --task MobileMMTrackEE-v0 `
  --num_envs 64 `
  --total_timesteps 500000 `
  --headless
```

**What to check:**
- [ ] No crashes (should complete)
- [ ] Explained variance: should stabilize around 0.3-0.7
- [ ] Clip fraction: should decrease to <12%
- [ ] Value loss: should increase initially (critic learning)
- [ ] Entropy: should start decreasing

### Step 3: Full Training Run

**If test succeeds, scale up:**
```powershell
cd I:\isaaclab
.\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py `
  --task MobileMMTrackEE-v0 `
  --num_envs 1024 `
  --total_timesteps 5000000 `
  --headless
```

**Expected improvements:**
- Explained variance: stable 0.5-0.8
- Clip fraction: 5-10%
- Value loss: gradual decrease
- Episode reward: steady increase
- Entropy: gradual decrease

---

## 📈 Monitoring Strategy

### Key Metrics to Watch (TensorBoard)

**Open TensorBoard:**
```powershell
cd I:\isaaclab
.\isaaclab.bat -p -m tensorboard --logdir H:\wSpace\cinebotRL\logs\sb3
# Browser: http://localhost:6006
```

**Critical Metrics:**

1. **rollout/ep_rew_mean** 📊
   - Should increase steadily
   - Target: Positive and growing

2. **train/explained_variance** 🎯
   - Should stabilize 0.5-0.8
   - RED FLAG: Below 0 or above 1

3. **train/value_loss** 📉
   - Should decrease gradually
   - Current ~0.0005 too low
   - Expect: 0.001-0.1 range initially

4. **train/clip_fraction** ✂️
   - Should be 5-10%
   - Current: 11-18% (too high)
   - Target: <10%

5. **train/entropy_loss** 🎲
   - Should gradually decrease
   - Current: -9.6 (stuck)
   - Want: Slow decrease over time

6. **rollout/ep_len_mean** ⏱️
   - Should stabilize at high values
   - Target: 800-1000 steps

---

## 🎓 Expected Training Phases

### Phase 1: Initial Learning (0-500K steps)
- **Explained variance**: Stabilizes to 0.3-0.6
- **Value loss**: Increases then decreases
- **Reward**: Fluctuates, slight upward trend
- **Clip fraction**: Decreases from 15% → 10%

### Phase 2: Skill Acquisition (500K-2M steps)
- **Explained variance**: Stable 0.6-0.8
- **Reward**: Clear upward trend
- **Episode length**: Increases to 600+
- **Entropy**: Starts decreasing

### Phase 3: Refinement (2M-5M steps)
- **Explained variance**: Stable 0.7-0.9
- **Reward**: Approaching plateau
- **Episode length**: Stable 900-1000
- **Entropy**: Continues decreasing

---

## 🚨 Red Flags & Interventions

### If Explained Variance Still Negative After 200K Steps:
**Action:** Stop and check:
- Reward scale (are rewards too small/large?)
- Observation normalization (check VecNormalize stats)
- Reward structure (sparse vs dense)

### If Clip Fraction Stays >15%:
**Action:** Reduce clip_range:
```python
clip_range=0.15,  # Reduce from 0.2
```

### If Value Loss Stays <0.001:
**Action:** Check reward magnitudes:
```python
# Print during training to verify
print(f"Reward range: {rewards.min():.3f} to {rewards.max():.3f}")
```

### If Training Crashes:
**Action:** Reduce batch size further:
```python
batch_size=128,  # Even smaller
```

---

## 💾 Checkpointing Strategy

**Save frequency:**
- Every 100K steps: Regular checkpoint
- Every 500K steps: Evaluation checkpoint
- Best model: When eval reward improves

**Evaluation:**
```python
from stable_baselines3.common.callbacks import EvalCallback

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=f"{log_dir}/best_model",
    log_path=f"{log_dir}/eval",
    eval_freq=50000,  # Evaluate every 50K steps
    deterministic=True,
    render=False
)

model.learn(
    total_timesteps=args.total_timesteps,
    callback=eval_callback  # Add callback
)
```

---

## 📊 Comparison: Before vs After Changes

| Metric | Current (Before) | Expected (After) |
|--------|------------------|------------------|
| **Explained Variance** | -16 to +0.6 (unstable) | 0.5-0.8 (stable) |
| **Clip Fraction** | 11-18% (too high) | 5-10% (good) |
| **Value Loss** | ~0.0005 (too low) | 0.001-0.1 (learning) |
| **Entropy** | -9.6 (stuck) | Gradually decreasing |
| **Convergence Speed** | Slow | 2-3x faster |

---

## 🔄 Iterative Improvement Cycle

### After Each Training Run:

1. **Analyze TensorBoard metrics**
2. **Identify bottleneck** (which metric is worst?)
3. **Apply targeted fix:**
   - Critic unstable → Adjust n_steps, clip_range_vf
   - Policy unstable → Adjust learning rate, clip_range
   - No exploration → Increase ent_coef
   - Slow learning → Increase batch updates (smaller batch_size)
4. **Test with 500K steps**
5. **If improved → Scale up to 5M**
6. **If not → Try next fix**

---

## 📁 Files to Modify

1. **scripts/reinforcement_learning/sb3/train.py** (MAIN CHANGES)
   - Add reward normalization
   - Update PPO hyperparameters
   - Add learning rate schedule
   - (Optional) Add policy kwargs

2. **No changes needed:**
   - src/rl_platform/tasks/mobile_mm/env.py (bug fixes already applied)
   - config files (using command-line args)

---

## ✅ Success Criteria

### Minimum (500K test):
- [ ] No crashes
- [ ] Explained variance > 0
- [ ] Clip fraction < 15%
- [ ] Value loss increasing from 0.0005

### Good (5M full run):
- [ ] Explained variance: 0.5-0.8
- [ ] Clip fraction: 5-10%
- [ ] Episode reward: steady increase
- [ ] Episode length: >800 steps
- [ ] Successful trajectory tracking visible

### Excellent:
- [ ] Explained variance: 0.7-0.9
- [ ] Episode length: 1000 (max)
- [ ] Success rate: >80%
- [ ] Converges in <2M steps

---

## 🚀 Next Steps (In Order)

1. **[ ] Review this plan** ← YOU ARE HERE
2. **[ ] Apply Priority 1 & 2 changes** (5 mins coding)
3. **[ ] Run test (64 envs, 500K steps)** (5-10 mins)
4. **[ ] Check TensorBoard** (review metrics)
5. **[ ] If good → Full run (1024 envs, 5M steps)** (35-40 mins)
6. **[ ] Monitor & iterate** (adjust if needed)
7. **[ ] Evaluate best policy** (visualize results)
8. **[ ] Commit & push to git**

---

## 📝 Notes

- **Start with Priority 1 & 2 ONLY** - Don't change everything at once
- **Test incrementally** - 500K test before 5M full run
- **Monitor closely** - First 100K steps tell you a lot
- **Be patient** - Critic learning takes time with reward normalization
- **Document changes** - Commit with clear messages

**Most Important:** Reward normalization (Priority 1) will likely have THE biggest impact!

