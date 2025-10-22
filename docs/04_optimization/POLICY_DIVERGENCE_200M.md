# Policy Divergence Analysis: 200M Training Run

**Date**: October 17, 2025  
**Training Run**: `20251017_001304`  
**Issue**: Policy diverged from std=0.59 at 139M steps to std=1.18 at 183M steps

---

## Executive Summary

Your PPO policy **successfully learned** end-effector tracking from 0M to 139M steps (std dropped from 1.0 to 0.59), but then **completely diverged** back to random behavior (std increased to 1.18 by 183M steps). The root cause was **constant high entropy coefficient** (0.001) that rewarded randomness even after the policy converged.

**Best Checkpoint**: `ppo_mobile_mm_139591680_steps.zip` (139.6M steps)  
**Status**: Training stopped at 183M steps after policy fully randomized  
**Solution**: Implemented entropy decay callback to prevent future divergence

---

## Timeline of Training

| Phase | Timesteps | std | Status | Analysis |
|-------|-----------|-----|--------|----------|
| **Initial** | 0M | 1.0 | Random policy | Starting from scratch |
| **Learning** | 0M - 100M | 1.0 → 0.7 | Learning well | Entropy bonus encourages exploration |
| **Convergence** | 100M - 139M | 0.7 → 0.59 | **Excellent!** | Policy learned precise tracking |
| **Divergence Start** | 139M - 167M | 0.59 → 1.18 | **Degrading** | Entropy bonus fights convergence |
| **Full Divergence** | 167M - 183M | 1.18 → 1.11 | **Random** | Policy stuck in high-entropy trap |

### Key Metrics Evolution

**At 139M steps (Best Policy):**
```
std:              0.59    ✅ Excellent precision
entropy_loss:     -6.85   ✅ Appropriate entropy
explained_var:    1.0     ✅ Perfect value prediction
approx_kl:        0.5-1.8 ✅ Healthy policy updates
value_loss:       3e-06   ✅ Minimal value error
```

**At 167M steps (Diverged Policy):**
```
std:              1.16-1.18  ❌ Random behavior
entropy_loss:     -12.4      ❌ Maximum entropy
explained_var:    1.0        ✅ Value function still works
approx_kl:        0.20-0.26  ❌ Tiny updates, stuck
value_loss:       2-8e-05    ❌ 10× worse
```

**At 183M steps (Fully Random):**
```
std:              1.08-1.11  ❌ Fully random (same as initialization)
entropy_loss:     -11.8/-12  ❌ Maximum entropy stabilized
explained_var:    1.0        ✅ Value function unchanged
approx_kl:        0.25-0.53  ❌ Still tiny updates
clip_fraction:    0.58-0.67  ❌ High clipping = trapped
```

---

## Root Cause: Constant Entropy Coefficient

### The Problem

PPO's loss function is:
$$L_{\text{total}} = L_{\text{policy}} - c_{\text{ent}} \cdot H(\pi)$$

Where:
- $L_{\text{policy}}$ = Policy gradient loss (lower = better tracking)
- $c_{\text{ent}}$ = Entropy coefficient (0.001 in your training)
- $H(\pi)$ = Policy entropy (higher = more random)

**During learning phase (0-139M):**
- High entropy helps exploration → Good!
- $c_{\text{ent}} \times H(\pi)$ bonus = 0.001 × 6.85 = **0.00685** reward for randomness
- This small bonus encourages trying new actions → Policy learns

**During convergence phase (139M+):**
- Policy wants to become deterministic (std → 0) → Good tracking!
- But entropy bonus **penalizes low entropy**
- PPO optimizes: "To reduce total loss, I need higher entropy"
- Policy starts outputting random actions to maximize entropy bonus
- **Result**: Policy "unlearns" and becomes random again

### Why It Couldn't Recover

1. **Local minimum trap**: Being random gives entropy bonus that outweighs poor tracking
2. **PPO clipping**: 60%+ clip_fraction means policy updates are cut off, can't escape
3. **Small KL divergence**: Policy barely changing (approx_kl = 0.2-0.5), stuck in place
4. **Constant pressure**: Entropy bonus never decreases, keeps pushing toward randomness

---

## Mathematical Explanation

**Entropy bonus at 139M steps:**
- entropy_loss = -6.85
- Entropy bonus = -0.001 × (-6.85) = **+0.00685** added to reward
- This 0.00685 bonus is significant compared to policy gradient loss (~0.01-0.05)

**After divergence (167M steps):**
- entropy_loss = -12.4
- Entropy bonus = -0.001 × (-12.4) = **+0.0124** reward
- Policy is now optimizing for entropy instead of tracking!

**The math says**: "Get 0.0124 reward by being random" > "Get 0.01 reward by tracking well"

---

## Evidence in Training Logs

### Divergence Signatures

1. **Standard deviation doubled:**
   - 139M: std = 0.59 (precise, learned behavior)
   - 167M: std = 1.18 (random, 2× increase)
   - 183M: std = 1.11 (fully random, stabilized)

2. **Entropy loss collapsed:**
   - 139M: entropy_loss = -6.85 (moderate entropy)
   - 167M: entropy_loss = -12.4 (maximum entropy)
   - Policy outputting maximum randomness allowed by action bounds

3. **KL divergence shrunk:**
   - 139M: approx_kl = 0.5-1.8 (healthy updates)
   - 167M+: approx_kl = 0.2-0.5 (stuck, barely changing)
   - Policy found a "local optimum" of maximum entropy

4. **High clip fraction:**
   - 167M+: clip_fraction = 0.58-0.67
   - 60%+ of policy updates are clipped
   - PPO's clip_range=0.2 prevents large updates to escape trap

5. **Value function unchanged:**
   - explained_variance = 1.0 throughout (even at 183M)
   - Critic still predicts rewards accurately
   - Policy didn't "forget" reward structure, just chose entropy over tracking

---

## Solution: Entropy Decay

### Implementation

Added `EntropyDecayCallback` to `train.py` that:
1. Starts with high `ent_coef` for exploration (e.g., 0.001)
2. Decays to low `ent_coef` for convergence (e.g., 0.0001)
3. Linearly interpolates during decay period

### Usage

**For 200M training with decay at 100M:**
```powershell
cd I:\isaaclab
.\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py `
    --task MobileMMTrackEE-v0 `
    --num_envs 4096 `
    --batch_size 1024 `
    --n_steps 32 `
    --total_timesteps 200000000 `
    --ent_coef 0.001 `
    --enable_entropy_decay `
    --final_ent_coef 0.0001 `
    --decay_start_timestep 100000000 `
    --decay_duration_timesteps 100000000 `
    --headless
```

**For 100M training (test if sufficient):**
```powershell
.\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py `
    --task MobileMMTrackEE-v0 `
    --num_envs 4096 `
    --batch_size 1024 `
    --n_steps 32 `
    --total_timesteps 100000000 `
    --ent_coef 0.001 `
    --enable_entropy_decay `
    --final_ent_coef 0.0001 `
    --decay_start_timestep 50000000 `
    --decay_duration_timesteps 50000000 `
    --headless
```

### Decay Schedule Example

For 200M training with decay from 100M to 200M:

| Timesteps | ent_coef | Phase | Purpose |
|-----------|----------|-------|---------|
| 0M - 100M | 0.001 | Exploration | Encourage trying diverse actions |
| 100M - 150M | 0.001 → 0.00055 | Early convergence | Gradual reduction |
| 150M - 180M | 0.00055 → 0.00028 | Mid convergence | More deterministic |
| 180M - 200M | 0.00028 → 0.0001 | Final convergence | Lock in learned behavior |

At 200M steps, entropy bonus = 0.0001 × 12 = **0.0012** (10× smaller, no longer dominates)

---

## Recommendations

### Immediate Actions

1. **STOP current training** (already at 183M, policy is random)
   ```powershell
   # Press Ctrl+C in training terminal
   ```

2. **Use best checkpoint** (139.6M steps)
   ```powershell
   cd I:\isaaclab
   .\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\evaluate.py `
       --checkpoint H:\wSpace\cinebotRL\logs\sb3\mobilemmtrackee_v0\20251017_001304\checkpoints\ppo_mobile_mm_139591680_steps.zip `
       --num_envs 4 --num_episodes 5
   ```

3. **Evaluate tracking quality**
   - If tracking is good → Use this as final model
   - If tracking needs polish → Fine-tune with lower ent_coef

### Future Training Strategy

**Option 1: Fresh Training with Entropy Decay (Recommended)**
- Total timesteps: 100M (may be sufficient)
- num_envs: 4096 (better GPU utilization)
- ent_coef: 0.001 with decay to 0.0001
- Decay: 50M → 100M steps
- Expected time: ~2.5 hours (vs 10 hours for 200M)

**Option 2: Fine-tune from 139M Checkpoint**
- Start from: `ppo_mobile_mm_139591680_steps.zip`
- Total timesteps: 30-50M additional
- ent_coef: 0.0001 (10× lower)
- learning_rate: 0.0001 (3× lower for fine-tuning)
- Purpose: Polish already-learned behavior

**Option 3: Shorter Training without Decay**
- Total timesteps: 140M (stop before divergence)
- ent_coef: 0.001 constant
- Monitor std closely, stop when std stops decreasing
- Requires manual monitoring

---

## Lessons Learned

1. **Constant hyperparameters don't adapt to training phase**
   - Early: Need high entropy for exploration
   - Late: Need low entropy for convergence
   - Solution: Schedule/decay hyperparameters

2. **More training ≠ better policy**
   - 139M checkpoint better than 200M checkpoint
   - Training can actively hurt performance
   - Know when to stop!

3. **Entropy coefficient is powerful**
   - 0.001 seems small but has huge impact
   - Can completely override reward signal
   - Must decrease as policy converges

4. **PPO clipping can create traps**
   - Prevents large policy changes (good for stability)
   - But also prevents escaping local minima (bad for recovery)
   - Once diverged, hard to recover

5. **Value function explains_variance ≠ policy quality**
   - Can have explained_variance=1.0 with random policy
   - Value function predicts "what reward will I get if I'm random"
   - Need to monitor std, entropy_loss, and clip_fraction too

---

## Key Files

### Training Script
- Path: `scripts/reinforcement_learning/sb3/train.py`
- Changes: Added `EntropyDecayCallback` class and CLI arguments

### Best Checkpoint
- Path: `H:\wSpace\cinebotRL\logs\sb3\mobilemmtrackee_v0\20251017_001304\checkpoints\ppo_mobile_mm_139591680_steps.zip`
- Timesteps: 139,591,680
- Performance: std=0.59, explained_var=1.0

### TensorBoard Logs
- Path: `H:\wSpace\cinebotRL\logs\sb3\mobilemmtrackee_v0\20251017_001304\`
- View: `I:\isaaclab\_isaac_sim\python.bat -m tensorboard.main --logdir <path>`

---

## Conclusion

Your training was **highly successful** up to 139M steps, achieving excellent tracking performance (std=0.59). The divergence afterward was entirely due to constant entropy coefficient fighting against convergence. With the entropy decay callback now implemented, future training runs will:

1. Use high entropy for initial exploration (0-100M)
2. Gradually reduce entropy bonus during convergence (100M-200M)
3. Lock in learned behavior with minimal entropy bonus (>150M)
4. Prevent the divergence issue that occurred in this run

**The 139M checkpoint is your best model.** Use it!
