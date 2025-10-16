# Training Action Plan - October 15, 2025

## Current Status ✅

**Training is NOW RUNNING with all 5 critical bug fixes:**
1. ✅ Base mobility enabled (fixed `target` parameter)
2. ✅ Action scaling to joint limits
3. ✅ Action history tracking (3 timesteps)
4. ✅ Collision detection enabled (with fallback)
5. ✅ Trajectory advancement working

**Current Test Run:**
- Command: 64 envs, 1M timesteps (test run)
- Log: `H:\wSpace\cinebotRL\logs\sb3\mobilemmtrackee_v0\20251015_232303\`
- Status: Collecting rollouts now

---

## Issues from Previous Training Analysis

Based on `docs/RL_Training_Summary-20251015night.md`, previous runs showed:

1. **Critic Instability**
   - Explained variance: negative and worsening (-7.28 → -19.3)
   - Value loss: low but stable (~0.01-0.02)
   - Problem: Critic not learning meaningful returns

2. **Policy Issues**
   - Increasing clipping (5-8%)
   - Entropy not decreasing (~11.3 constant)
   - Problem: Policy unstable, exploration not improving

3. **Reward Issues**
   - Sparse or inconsistent rewards
   - Improper scaling
   - Problem: Low variance in returns

---

## Phase 1: Monitor Current Run (ACTIVE NOW)

### What to Watch For:

**Key Metrics to Track:**
```bash
# In TensorBoard (open in another terminal):
cd I:\isaaclab
.\isaaclab.bat -p -m tensorboard --logdir H:\wSpace\cinebotRL\logs\sb3
# Then open: http://localhost:6006
```

**Critical Indicators:**
- [ ] **Explained Variance** - Should be positive (0-1), not negative
- [ ] **Episode Reward** - Should increase over time
- [ ] **Episode Length** - Should stabilize (not crash early)
- [ ] **Value Loss** - Should decrease initially
- [ ] **Policy Loss** - Should decrease and stabilize
- [ ] **Entropy** - Should gradually decrease (exploration → exploitation)

### Expected Improvements from Bug Fixes:

With our fixes, we should see:
1. **Longer episodes** - Collision detection prevents tip-overs
2. **Higher rewards** - Full workspace access (95% vs 50%)
3. **Smoother actions** - Action history prevents jerky movements
4. **Moving target** - Trajectory advancement enables actual tracking

---

## Phase 2: Quick Wins (IF Issues Persist)

### Priority 1: Reward Normalization
**File:** `scripts/reinforcement_learning/sb3/train.py`

**Current code** (lines ~270-280):
```python
# Create normalized wrapper
env = VecNormalize(
    env,
    norm_obs=True,
    norm_reward=False,  # ← CHANGE THIS
    clip_obs=10.0,
    gamma=0.99
)
```

**Fix:**
```python
# Enable reward normalization
env = VecNormalize(
    env,
    norm_obs=True,
    norm_reward=True,  # ← ENABLE reward normalization
    clip_obs=10.0,
    clip_reward=10.0,  # ← Add reward clipping
    gamma=0.99
)
```

### Priority 2: Better Hyperparameters

**Current** (in `train.py`):
```python
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=512,
    # ...
)
```

**Recommended changes:**
```python
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,      # Keep
    n_steps=4096,            # ← INCREASE (more data per update)
    batch_size=256,          # ← DECREASE (more gradient updates)
    n_epochs=10,             # Keep
    gamma=0.99,              # Keep
    gae_lambda=0.95,         # Keep
    clip_range=0.2,          # Keep
    clip_range_vf=1.0,       # ← ADD (clip value function updates)
    ent_coef=0.01,           # ← INCREASE slightly (more exploration)
    vf_coef=0.5,             # Keep
    max_grad_norm=0.5,       # Keep
    verbose=1
)
```

### Priority 3: Check Reward Structure

**File:** `src/rl_platform/tasks/mobile_mm/env.py`

Check reward components (lines ~520-590):
- Are rewards too sparse?
- Are they scaled appropriately?
- Is the balance between components correct?

---

## Phase 3: Iterative Improvements

### After First 100K Steps:

**Decision Point 1: Is training improving?**
- ✅ **YES** → Continue to 1M steps, then scale up to 512 envs
- ❌ **NO** → Apply Phase 2 fixes and restart

### After 1M Steps (Test Run Complete):

**Decision Point 2: Policy performance?**
- ✅ **GOOD** (reward increasing, episodes stable) → 
  - Scale up to 512-1024 envs
  - Train for 5M steps
  - Push all changes to git
  
- ⚠️ **MEDIOCRE** (some improvement but unstable) →
  - Apply Priority 1 & 2 from Phase 2
  - Run another 1M test
  
- ❌ **BAD** (no improvement, crashes) →
  - Deep dive into reward structure
  - Check observation normalization ranges
  - Review action space design

---

## Phase 4: Full Training Run

**Once test run succeeds:**

```powershell
cd I:\isaaclab
.\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py `
  --task MobileMMTrackEE-v0 `
  --num_envs 1024 `
  --total_timesteps 5000000 `
  --headless
```

**Monitoring:**
- Check TensorBoard every 30 minutes
- Save checkpoints every 500K steps
- Evaluate policy at 1M, 2.5M, 5M steps

**Expected Duration:**
- 1024 envs, 5M steps: ~35-40 minutes

---

## Phase 5: Evaluation & Iteration

### After Full Training:

1. **Visualize Best Policy:**
```powershell
.\scripts\visualize_policy.ps1 -Checkpoint "H:\wSpace\cinebotRL\logs\sb3\mobilemmtrackee_v0\<timestamp>\checkpoints\best_model.zip"
```

2. **Analyze Results:**
   - Does robot track trajectory smoothly?
   - Are there any failure modes?
   - What's the success rate?

3. **Fine-tune if needed:**
   - Adjust reward weights
   - Try different trajectory types
   - Implement curriculum learning

---

## Immediate Next Steps (NOW)

### Step 1: Let Current Test Run Complete ⏳
**Status:** Running now (started 15:22)
**Action:** Monitor TensorBoard, don't interrupt

### Step 2: Check Progress After ~5 Minutes
```powershell
# Check if training is progressing
Get-Content "H:\wSpace\cinebotRL\logs\sb3\mobilemmtrackee_v0\20251015_232303\PPO_1\progress.csv" -Tail 5
```

### Step 3: Open TensorBoard (Optional, in new terminal)
```powershell
cd I:\isaaclab
.\isaaclab.bat -p -m tensorboard --logdir H:\wSpace\cinebotRL\logs\sb3
# Open browser: http://localhost:6006
```

### Step 4: After Test Completes (~5-10 mins)
- Review metrics
- Decide: Continue with full run OR apply Phase 2 fixes
- Commit and push latest changes to git

---

## Git Management

**Current commits:**
- ✅ All 5 bug fixes
- ✅ Visualization tools
- ✅ Parameter name fix (`target`)

**After successful training:**
```bash
git add .
git commit -m "training: Successful test run with all bug fixes

- 64 envs, 1M steps completed
- All 5 critical fixes validated
- Ready for full scale training"
git push origin train-windows
```

---

## Success Criteria

**Minimum (Test Run):**
- ✅ No crashes for 1M steps
- ✅ Episode length > 100 steps average
- ✅ Reward trend: upward or stable
- ✅ Explained variance: > -5

**Good (Full Run):**
- ✅ Episode length: 800-1000 steps
- ✅ Success rate: > 50%
- ✅ Smooth trajectory tracking visible
- ✅ No tip-overs

**Excellent:**
- ✅ Episode length: stable at 1000
- ✅ Success rate: > 80%
- ✅ Precise end-effector tracking
- ✅ Fast convergence (< 2M steps)

---

## Notes

- **Warp CUDA warnings:** Harmless, ignore them
- **Gymnasium type warnings:** Handled by wrapper, ignore
- **Contact forces warning:** Using fallback method, expected
- **Negative explained variance:** Main concern from previous runs, watch this closely

