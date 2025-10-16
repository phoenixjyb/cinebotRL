# Quick Tuning Summary - Ready to Apply

## ✅ Good News: Reward Normalization Already Enabled!

Line 356 in `train.py` already has:
```python
norm_reward=True,
clip_reward=10.0,
```

This is the MOST IMPORTANT fix and it's already done! 🎉

---

## 🔧 Changes to Apply Now

### Change 1: Improve PPO Hyperparameters (Lines 399-412)

**Current:**
```python
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=args.learning_rate,
    n_steps=args.n_steps,             # Default: 2048
    batch_size=args.batch_size,        # Default: 512
    n_epochs=args.n_epochs,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,                      # ← No exploration bonus!
    vf_coef=0.5,
    max_grad_norm=0.5,
    tensorboard_log=args.log_dir,
    device=device,
    verbose=1,
)
```

**Change to:**
```python
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=args.learning_rate,
    n_steps=args.n_steps,              # Will use CLI arg (increase to 4096)
    batch_size=args.batch_size,        # Will use CLI arg (decrease to 256)
    n_epochs=args.n_epochs,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    clip_range_vf=1.0,                 # ← ADD: Clip value function
    ent_coef=0.01,                     # ← CHANGE: Add exploration bonus
    vf_coef=0.5,
    max_grad_norm=0.5,
    target_kl=0.01,                    # ← ADD: Early stopping
    tensorboard_log=args.log_dir,
    device=device,
    verbose=1,
)
```

---

## 🚀 Command to Run

### Test Run (5-10 minutes):
```powershell
cd I:\isaaclab
.\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py `
  --task MobileMMTrackEE-v0 `
  --num_envs 64 `
  --total_timesteps 500000 `
  --n_steps 4096 `
  --batch_size 256 `
  --headless
```

### Full Run (30-40 minutes):
```powershell
cd I:\isaaclab
.\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py `
  --task MobileMMTrackEE-v0 `
  --num_envs 1024 `
  --total_timesteps 5000000 `
  --n_steps 4096 `
  --batch_size 256 `
  --headless
```

---

## 📊 Expected Improvements

| Metric | Before | After |
|--------|--------|-------|
| **Explained Variance** | -16 to +0.6 | 0.5-0.8 stable |
| **Clip Fraction** | 11-18% | 5-10% |
| **Entropy** | Stuck at -9.6 | Gradual decrease |
| **Value Loss** | ~0.0005 | 0.001-0.1 |
| **Convergence** | Slow | 2x faster |

---

## 🎯 Key Changes Impact

1. **n_steps=4096** (from 2048)
   - More experience per update
   - More stable GAE estimates
   - Reduces variance in critic targets

2. **batch_size=256** (from 512)
   - More gradient updates per epoch
   - Better optimization
   - Faster convergence

3. **clip_range_vf=1.0** (new)
   - Prevents large value function jumps
   - Stabilizes critic learning
   - Reduces explained variance fluctuations

4. **ent_coef=0.01** (from 0.0)
   - Encourages exploration
   - Helps entropy decrease naturally
   - Prevents premature convergence

5. **target_kl=0.01** (new)
   - Early stopping if policy changes too much
   - Prevents destabilizing updates
   - Improves training stability

---

## ✅ What to Watch in TensorBoard

Open TensorBoard before training:
```powershell
cd I:\isaaclab
.\isaaclab.bat -p -m tensorboard --logdir H:\wSpace\cinebotRL\logs\sb3
# Browser: http://localhost:6006
```

**Success indicators (first 100K steps):**
- [ ] Explained variance: stays positive (>0.2)
- [ ] Clip fraction: decreases to <12%
- [ ] Value loss: increases from 0.0005 (critic learning!)
- [ ] Episode reward: shows upward trend
- [ ] No crashes

---

## 🔄 If Issues Persist

**If explained variance still negative:**
- Check reward scale in environment
- May need to adjust reward weights

**If clip fraction still high:**
- Can reduce clip_range to 0.15

**If training too slow:**
- Can increase learning_rate to 5e-4

But let's try these changes first - they should have big impact!

