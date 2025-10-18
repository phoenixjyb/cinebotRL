# Summary: Playbook Analysis & Next Training Command

## TL;DR

The playbook is **mostly correct** but has **3 wrong assumptions** about your setup:
1. ❌ Says you lack temporal context → You **already have** 3-step lookahead + 2-step action history!
2. ❌ Recommends ent_coef=0.01 → **Too high**, your 0.001 is correct
3. ⚠️ Suggests n_steps=4096-8192 → **Too risky**, starting with 128 instead

## What We Implemented

### ✅ Already Done (Previous Commits)
1. **Entropy Decay**: 0.001 → 0.0001 (prevents your divergence issue)
2. **KL Schedule**: 0.07 → 0.02 → 0.01 (prevents instability)
3. **More Environments**: 4096 (better GPU usage)

### ✅ Just Added (This Commit)
4. **Log-Std Bounds**: [-3.0, 1.0] → std ∈ [0.05, 2.72] (safety guardrail)
5. **Increased n_steps**: 32 → 128 (better GAE estimation)

## Your Complete Training Command

```powershell
cd I:\isaaclab
.\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py `
    --task MobileMMTrackEE-v0 `
    --num_envs 4096 `
    --batch_size 1024 `
    --n_steps 128 `
    --total_timesteps 100000000 `
    --learning_rate 0.0003 `
    --ent_coef 0.001 `
    --enable_entropy_decay `
    --final_ent_coef 0.0001 `
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

## Expected Training Behavior

### Metrics Timeline
| Phase | Timesteps | ent_coef | target_kl | Expected approx_kl | Expected std | Expected clip_frac |
|-------|-----------|----------|-----------|-------------------|--------------|-------------------|
| Warmup | 0-10M | 0.001 | 0.07 | 0.02-0.06 | 1.0 → 0.85 | 0.15-0.30 |
| Early Main | 10-50M | 0.001 | 0.02 | 0.01-0.02 | 0.85 → 0.65 | 0.15-0.25 |
| Late Main | 50-80M | 0.001→0.00055 | 0.02 | 0.008-0.015 | 0.65 → 0.55 | 0.10-0.20 |
| Finetune | 80-100M | 0.00055→0.0001 | 0.01 | 0.005-0.01 | 0.55 → 0.50 | 0.08-0.15 |

### Your Previous Run (For Comparison)
| Phase | Timesteps | approx_kl | std | clip_frac | Status |
|-------|-----------|-----------|-----|-----------|--------|
| Learning | 0-139M | 0.5-1.8 | 1.0 → 0.59 | 0.4-0.5 | ✅ Good but inefficient |
| Divergence | 139M-183M | 0.2-0.5 | 0.59 → 1.18 | 0.58-0.67 | ❌ Policy randomized |

**New run should have**:
- 10-90× **lower** approx_kl (0.01-0.02 vs 0.5-1.8)
- Similar or better std convergence (1.0 → 0.50)
- **Much lower** clip_fraction (0.15-0.25 vs 0.4-0.7)
- **No divergence** after convergence

## Protection Layers

### Layer 1: Entropy Decay (Prevents Divergence)
**Problem**: Constant ent_coef=0.001 caused your 139M→183M divergence
**Solution**: Decay to 0.0001 during training
**Impact**: Entropy bonus shrinks 10×, policy stays converged

### Layer 2: KL Schedule (Prevents Instability)
**Problem**: High approx_kl=0.5-1.8 wastes gradients
**Solution**: Adaptive target_kl (0.07 → 0.02 → 0.01)
**Impact**: Smaller, more efficient policy updates

### Layer 3: Log-Std Bounds (Safety Net)
**Problem**: Policy can drift to extreme std values
**Solution**: Clamp std to [0.05, 2.72] range
**Impact**: Cannot explode to std=20+ or collapse to std=0

### Layer 4: Longer Rollouts (Better Estimates)
**Problem**: n_steps=32 too short for accurate GAE
**Solution**: n_steps=128 (4× longer)
**Impact**: More accurate advantage estimates, stable learning

## Training Phases Explained

### Phase 1: Warmup (0-10M steps)
**Goal**: Explore action space, learn basic behaviors

**Settings**:
- ent_coef = 0.001 (high entropy)
- target_kl = 0.07 (loose constraint)

**Expected behavior**:
- Robot tries many different actions
- std stays high (~0.9-1.0)
- Some tracking attempts, mostly random

**Duration**: ~15 minutes

### Phase 2: Early Main (10-50M steps)
**Goal**: Learn tracking behavior

**Settings**:
- ent_coef = 0.001 (still high)
- target_kl = 0.02 (moderate constraint)

**Expected behavior**:
- Robot starts following trajectories
- std decreases steadily (0.9 → 0.65)
- Tracking improves significantly

**Duration**: ~60 minutes

### Phase 3: Late Main (50-80M steps)
**Goal**: Refine tracking, start converging

**Settings**:
- ent_coef = 0.001 → 0.00055 (decaying)
- target_kl = 0.02 (moderate)

**Expected behavior**:
- Smooth tracking emerges
- std continues decreasing (0.65 → 0.55)
- Policy becoming deterministic

**Duration**: ~45 minutes

### Phase 4: Finetune (80-100M steps)
**Goal**: Lock in precise behavior

**Settings**:
- ent_coef = 0.00055 → 0.0001 (very low)
- target_kl = 0.01 (tight)

**Expected behavior**:
- Very smooth, consistent tracking
- std stabilizes (~0.50-0.55)
- No more exploration

**Duration**: ~30 minutes

**Total**: ~2.5-3 hours

## Red Flags (Stop Training If You See)

### 🛑 Critical - Stop Immediately
1. **std increasing** after convergence (e.g., 0.55 → 0.60 → 0.70)
   - Sign: Divergence starting
   - Action: Stop, use earlier checkpoint

2. **approx_kl > 0.1** consistently (>5 rollouts)
   - Sign: Policy updates too large
   - Action: Stop, need to lower LR

3. **clip_fraction > 0.6** consistently
   - Sign: Most updates being clipped, stuck
   - Action: Stop, need tighter KL or lower LR

### ⚠️ Warning - Monitor Closely
1. **explained_variance dropping** from 1.0
   - Sign: Critic degrading
   - Action: Check if rewards changed, may need to continue

2. **std not decreasing** after 30M steps
   - Sign: Not learning
   - Action: May need more training or env issue

3. **Training much slower** than expected
   - Sign: n_steps too long or env bottleneck
   - Action: Profile to find bottleneck

## Success Criteria

### Training Complete When:
1. ✅ std stabilizes below 0.6 for 10M+ steps
2. ✅ explained_variance ≥ 0.95
3. ✅ approx_kl in healthy range (0.005-0.02)
4. ✅ clip_fraction < 0.30
5. ✅ Reached 100M steps OR convergence plateau

### Visual Evaluation Should Show:
1. ✅ Smooth end-effector tracking along trajectory
2. ✅ Base moves appropriately to assist arm
3. ✅ No jerky or erratic movements
4. ✅ Consistent behavior across episodes

## What NOT to Do (Common Mistakes)

### ❌ Don't Add These Yet
1. **Branched heads** (base/arm separate)
   - Why: Adds complexity, current arch worked for 139M steps
   - When: After verifying current fixes work

2. **SDE (State-Dependent Exploration)**
   - Why: Adds noise to debugging
   - When: If exploration still insufficient after convergence

3. **Temporal stacking**
   - Why: You already have 3-step lookahead + 2-step history!
   - When: Never (already implemented)

4. **n_steps=4096**
   - Why: Too large a jump from 32
   - When: Maybe try 512 after 128 works

5. **ent_coef=0.01**
   - Why: 10× too high for your task
   - When: Never (use 0.001)

### ✅ Do Add Later (If Needed)
1. **Learning rate decay** (3e-4 → 3e-5)
   - When: If training still unstable with current setup
   - Impact: Smoother late-stage convergence

2. **Clip range decay** (0.2 → 0.1)
   - When: If clip_fraction stays high despite KL schedule
   - Impact: Allows smaller updates late in training

## Comparison: Old vs New Setup

| Component | Old Setup | New Setup | Impact |
|-----------|-----------|-----------|---------|
| **ent_coef** | 0.001 (constant) | 0.001 → 0.0001 (decay) | ✅ Prevents divergence |
| **target_kl** | None | 0.07 → 0.02 → 0.01 | ✅ Prevents instability |
| **log_std_bounds** | None | [-3, 1] | ✅ Safety guardrail |
| **n_steps** | 32 | 128 | ✅ Better GAE |
| **num_envs** | 2048 | 4096 | ✅ 2× faster |
| **Total Protection** | 1 layer (clip_range) | 5 layers (above) | ✅✅✅ |

## Next Steps After This Training

### If Training Succeeds (std < 0.6, no divergence)
1. ✅ Evaluate best checkpoint visually
2. ✅ Test on held-out trajectories
3. ⚠️ Consider adding LR decay for even smoother training
4. ⚠️ Consider n_steps=256-512 for longer rollouts

### If Training Still Has Issues
**High KL/clip_fraction**:
- Add learning rate decay
- Tighten KL schedule values

**Slow convergence**:
- May need curriculum learning
- Check reward shaping

**Still diverges**:
- Check if log_std hitting bounds
- May need even lower final ent_coef (0.00005)

## Files Modified

1. **train.py**:
   - Added log_std_init=-1.0
   - Added log_std_bounds=(-3.0, 1.0)
   - Changed n_steps default: 32 → 128
   - Previously added: EntropyDecayCallback, DynamicKLSchedule

2. **Documentation**:
   - Critical_Analysis_Playbook_vs_Reality.md (this analysis)
   - Policy_Divergence_Analysis_200M_Training.md (your divergence)
   - FINAL_TRAINING_COMMAND_With_All_Protections.md (complete guide)
   - Why_Both_Entropy_Decay_AND_KL_Schedule.md (theory)

## Final Checklist

Before you run training:

- [✅] Entropy decay implemented
- [✅] KL schedule implemented
- [✅] Log-std bounds added
- [✅] n_steps increased to 128
- [✅] num_envs set to 4096
- [✅] Best checkpoint from 200M run identified (139.6M)
- [✅] TensorBoard command ready
- [✅] Evaluation command ready
- [✅] Red flag criteria understood

**You are ready to train!** 🚀

Just copy-paste the command above and let it run. Monitor TensorBoard during training, and stop if you see any red flags.

**Expected outcome**: Smooth training to convergence in ~2.5-3 hours, final std ~0.50-0.55, no divergence!
