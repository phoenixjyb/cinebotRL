# Quick Reference: Training with Entropy Decay

## TL;DR - Run This Command

**For 100M training (test if sufficient):**
```powershell
cd I:\isaaclab
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

**Expected results:**
- Training time: ~2.5 hours (vs 10 hours for 200M)
- Iterations: ~763 (vs 3,052 for 200M)
- GPU usage: ~90% (4096 envs vs 2048)
- Policy should converge around 80-100M steps

---

## What Changed

### Before (Diverged at 139M)
```powershell
--ent_coef 0.001  # Constant throughout training
# Result: std 0.59 → 1.18 (policy became random)
```

### After (With Decay)
```powershell
--ent_coef 0.001                      # Start high for exploration
--enable_entropy_decay                # Enable decay
--final_ent_coef 0.0001              # End low for convergence (10× lower)
--decay_start_timestep 50000000      # Start decay at 50M
--decay_duration_timesteps 50000000  # Decay over 50M steps (50M → 100M)
# Result: Entropy bonus shrinks 10× during convergence phase
```

### Decay Schedule

| Timesteps | ent_coef | Purpose |
|-----------|----------|---------|
| 0M - 50M | 0.001 | Full exploration |
| 50M - 75M | 0.001 → 0.00055 | Start converging |
| 75M - 90M | 0.00055 → 0.00028 | More deterministic |
| 90M - 100M | 0.00028 → 0.0001 | Lock in behavior |

---

## Monitoring Training

### Key Metrics to Watch

**Healthy Training:**
```
std:              1.0 → 0.6 → 0.5    ✅ Decreasing toward 0
entropy_loss:     -6 → -7 → -8       ✅ Gradually decreasing
explained_var:    0.8 → 0.95 → 1.0   ✅ Increasing toward 1.0
approx_kl:        0.5 - 2.0          ✅ Moderate updates
clip_fraction:    0.3 - 0.5          ✅ Some clipping, not too much
```

**Signs of Divergence (STOP if you see this):**
```
std:              0.5 → 0.8 → 1.1    ❌ INCREASING after convergence
entropy_loss:     -7 → -10 → -12     ❌ Collapsing rapidly
clip_fraction:    > 0.6              ❌ Too much clipping = stuck
approx_kl:        < 0.3              ❌ Tiny updates = stuck
```

### TensorBoard Command
```powershell
cd I:\isaaclab
.\_isaac_sim\python.bat -m tensorboard.main --logdir H:\wSpace\cinebotRL\logs\sb3\
```

---

## Alternative: Fine-tune from 139M Checkpoint

If you want to refine the already-learned 139M policy:

```powershell
cd I:\isaaclab
.\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py `
    --task MobileMMTrackEE-v0 `
    --num_envs 2048 `
    --batch_size 512 `
    --n_steps 32 `
    --total_timesteps 50000000 `
    --checkpoint H:\wSpace\cinebotRL\logs\sb3\mobilemmtrackee_v0\20251017_001304\checkpoints\ppo_mobile_mm_139591680_steps.zip `
    --ent_coef 0.0001 `
    --learning_rate 0.0001 `
    --headless
```

**Key changes for fine-tuning:**
- `--checkpoint`: Start from 139M policy
- `--ent_coef 0.0001`: 10× lower (no decay needed)
- `--learning_rate 0.0001`: 3× lower for gentle updates
- Total: 50M additional steps (189M total)

---

## Evaluation Command

After training, evaluate visually:

```powershell
cd I:\isaaclab
.\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\evaluate.py `
    --checkpoint <path_to_checkpoint>.zip `
    --num_envs 4 `
    --num_episodes 5
```

**What to look for:**
- Robot tracks trajectory smoothly ✅
- End-effector follows path closely ✅
- No erratic/random movements ✅
- Base moves appropriately to keep arm in workspace ✅

---

## Why This Prevents Divergence

### The Math

**Without decay (your 200M run):**
- At 139M: entropy_bonus = 0.001 × 6.85 = +0.00685
- At 167M: entropy_bonus = 0.001 × 12.0 = +0.012
- **Problem**: Bonus grows as policy becomes random!

**With decay (new training):**
- At 50M: entropy_bonus = 0.001 × 6.85 = +0.00685 (same)
- At 75M: ent_coef decays to 0.00055, bonus = 0.00055 × 8 = +0.0044
- At 100M: ent_coef decays to 0.0001, bonus = 0.0001 × 12 = +0.0012
- **Solution**: Bonus shrinks even if entropy increases!

At 100M steps, entropy bonus is **10× smaller**, no longer dominates policy loss.

---

## Expected Training Performance

### 100M Training (4096 envs)
- **Duration**: ~2.5 hours
- **Iterations**: 763
- **FPS**: ~11,000 (vs 4,200 with 2048 envs)
- **GPU usage**: ~90% (vs ~50% with 2048 envs)
- **Timesteps per iteration**: 4096 × 32 = 131,072
- **Checkpoints**: Every 100K steps = ~8 checkpoints

### GPU Utilization Comparison

| Config | Envs | FPS | GPU % | Training Time (100M) |
|--------|------|-----|-------|---------------------|
| Old | 2048 | 4,200 | ~50% | ~6.6 hours |
| **New** | **4096** | **~11,000** | **~90%** | **~2.5 hours** |

**Benefits of 4096 envs:**
- 2.6× faster training (better GPU utilization)
- More stable gradients (2× larger batch_size)
- More diverse experiences per iteration

---

## Success Criteria

### Training is successful if:
1. **std drops below 0.6** (from initial 1.0)
2. **explained_variance reaches 0.95+** (value function converged)
3. **std stops decreasing and stabilizes** (policy converged)
4. **Visual evaluation shows smooth tracking** (qualitative check)

### Stop training when:
- std stabilizes for 10M+ steps (no further improvement)
- std starts increasing after convergence (divergence!)
- explained_variance stays at 1.0 for 20M+ steps (fully converged)

**Don't overtrain!** More is not always better.

---

## Files Modified

- `scripts/reinforcement_learning/sb3/train.py`: Added `EntropyDecayCallback` class
- `docs/Policy_Divergence_Analysis_200M_Training.md`: Full analysis of divergence issue

## Best Checkpoint from 200M Run

- Path: `H:\wSpace\cinebotRL\logs\sb3\mobilemmtrackee_v0\20251017_001304\checkpoints\ppo_mobile_mm_139591680_steps.zip`
- Performance: std=0.59, explained_var=1.0
- Status: **This is your best model from that run**
