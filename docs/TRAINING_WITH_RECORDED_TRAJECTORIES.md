# Training with Recorded Trajectories - Complete Guide

## 🎯 Overview

This guide explains how to train your mobile manipulator on **all 1,038 recorded trajectories** for a diverse, robust policy. The trajectory analysis identified different types of movements, and exposing the robot to ALL of them ensures it learns when to use base movement vs arm-only strategies.

## 📊 Quick Summary: Why Use All Trajectories?

From our analysis of 1,038 trajectories:
- **519 (50%)** require chassis movement (X change ≥ 2.0m)
- **519 (50%)** can use arm-only (X change < 2.0m)

This **perfect 50/50 balance** is ideal for training because the robot learns:
1. **When to move the base** - for long-range movements (2.0m+)
2. **When to use arm only** - for short-range movements (< 2.0m)
3. **How to coordinate** - base + arm working together smoothly

## 🚀 Quick Start: Train on All Trajectories

### Recommended Command (Full Training)

```bash
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
    --trajectory_dir trajectoryToLearn/world_json `
    --use_all_trajectories `
    --headless
```

**This will train on ALL 1,038 trajectories!** ✅

## 📖 Trajectory Configuration Options

### Option 1: All Trajectories (Recommended for Training)

```bash
--trajectory_type multi_recorded \
--trajectory_dir trajectoryToLearn/world_json \
--use_all_trajectories
```

**Use for:**
- ✅ Main training runs
- ✅ Building robust, generalizable policy
- ✅ Learning when to use base vs arm-only

**Result:** Robot sees full diversity of movements (1,038 trajectories)

### Option 2: Chassis-Only Trajectories (For Testing Only)

```bash
--trajectory_type multi_recorded \
--trajectory_dir trajectoryToLearn/world_json \
--use_chassis_only
```

**Use for:**
- ✅ Testing that base movement works
- ✅ Visual validation of chassis fixes
- ❌ **NOT for main training** (biased toward base movement)

**Result:** Robot only sees 519 challenging trajectories

### Option 3: Limited Subset (For Debugging)

```bash
--trajectory_type multi_recorded \
--trajectory_dir trajectoryToLearn/world_json \
--use_all_trajectories \
--max_trajectories 100
```

**Use for:**
- ✅ Quick debugging runs
- ✅ Testing new reward functions
- ✅ Faster iteration during development

**Result:** Robot sees first 100 trajectories only

### Option 4: Simple Parametric (Original Behavior)

```bash
--trajectory_type circle
# (default - no additional flags needed)
```

**Use for:**
- ✅ Initial policy debugging
- ✅ Validating basic tracking ability
- ❌ **NOT for final training** (too simple, not realistic)

**Result:** Robot only sees synthetic circular trajectory

## 🎓 Training Strategies

### Strategy 1: Full Training (Recommended)

**Goal:** Build robust policy on all trajectories

```bash
# Train on all 1,038 trajectories for 100M steps
.\isaaclab.bat -p ... \
    --total_timesteps 100000000 \
    --trajectory_type multi_recorded \
    --use_all_trajectories \
    --headless
```

**Expected Results:**
- Training FPS: 8000-9000
- Convergence: ~50-100M timesteps
- Success rate: 75%+ on diverse trajectories
- Policy learns strategic base usage

### Strategy 2: Curriculum Learning

**Goal:** Gradually increase difficulty

```bash
# Stage 1: Arm-only trajectories (1-50M steps)
.\isaaclab.bat -p ... \
    --total_timesteps 50000000 \
    --trajectory_type circle

# Stage 2: Mixed trajectories (50-150M steps)
.\isaaclab.bat -p ... \
    --total_timesteps 100000000 \
    --trajectory_type multi_recorded \
    --use_all_trajectories \
    --checkpoint logs/.../checkpoint_50000000.zip
```

**Expected Results:**
- Faster initial learning on simple tasks
- Better generalization to complex trajectories
- Higher final success rate (80%+)

### Strategy 3: Focused Base Training

**Goal:** Specifically improve base movement

```bash
# Train heavily on chassis-requiring trajectories
.\isaaclab.bat -p ... \
    --total_timesteps 50000000 \
    --trajectory_type multi_recorded \
    --use_chassis_only
```

**Expected Results:**
- Robot becomes expert at base movement
- May struggle with arm-only tasks
- **Use for specialized applications only**

## 📊 Monitoring Training Progress

### Key Metrics to Watch

1. **Training FPS**
   - Target: 8000-9000 FPS
   - If < 1000: Check for frozen base issue

2. **Episode Rewards**
   - Should gradually increase
   - Positive rewards = successful tracking
   - Negative rewards = tracking failures

3. **Base Movement Diagnostics**
   - Check `base_vel_x`, `base_vel_y`, `base_vel_yaw`
   - Should be non-zero for chassis-requiring trajectories
   - Should be near-zero for arm-only trajectories

4. **Episode Completion Rate**
   - Target: 95%+ episodes run to max steps
   - Early termination = policy struggling

### Example Good Training Output

```
[INFO] Total timesteps: 10,485,760
[INFO] Training FPS: 8542
[INFO] Mean reward: +1234.56
[INFO] Episode length: 995.2 steps (out of 1000)
[INFO] Base distance traveled: 2.15m (avg)
```

### Example Bad Training Output

```
[WARNING] Total timesteps: 524,288
[WARNING] Training FPS: 856  ← Too low!
[WARNING] Mean reward: -125,431  ← Negative!
[WARNING] Episode length: 12.3 steps  ← Early termination!
[WARNING] Base distance traveled: 0.02m  ← Not moving!
```

## 🔧 Common Training Patterns

### Pattern 1: Short Test Run (1 hour)

```bash
# Quick validation that everything works
.\isaaclab.bat -p ... \
    --total_timesteps 10000000 \
    --num_envs 4096 \
    --trajectory_type multi_recorded \
    --use_all_trajectories \
    --max_trajectories 50 \
    --headless
```

**Duration:** ~1 hour
**Purpose:** Verify training pipeline works

### Pattern 2: Overnight Run (8-12 hours)

```bash
# Substantial training progress
.\isaaclab.bat -p ... \
    --total_timesteps 100000000 \
    --num_envs 4096 \
    --trajectory_type multi_recorded \
    --use_all_trajectories \
    --headless
```

**Duration:** 8-12 hours
**Purpose:** Get meaningful policy convergence

### Pattern 3: Multi-Day Run (Production)

```bash
# Full training to convergence
.\isaaclab.bat -p ... \
    --total_timesteps 500000000 \
    --num_envs 4096 \
    --trajectory_type multi_recorded \
    --use_all_trajectories \
    --wandb \
    --headless
```

**Duration:** 2-3 days
**Purpose:** Achieve state-of-the-art performance

## 🎯 Trajectory Statistics (Reference)

### Overall Distribution

| Metric | Value |
|--------|-------|
| Total trajectories | 1,038 |
| Require chassis | 519 (50.0%) |
| Arm-only | 519 (50.0%) |
| Mean X change | 2.089m |
| Median X change | 2.002m |
| Max X change | 3.000m |

### By Trajectory Type

| Type | Count | % Chassis | Avg X Change |
|------|-------|-----------|--------------|
| **arc_left_push** | 100 | 100% | 2.893m |
| **push** | 200 | 100% | 2.910m |
| **orbit_left** | 100 | 94% | 2.688m |
| **orbit_right** | 100 | 93% | 2.630m |
| **approach** | 11 | 100% | 2.990m |
| **arc_right** | 100 | 0% | 1.500m |
| **pull** | 200 | 0% | 1.500m |
| **retreat** | 11 | 0% | 1.500m |
| **round** | 9 | 33% | 1.660m |

### By Scene

| Scene | Trajectories | % Chassis | Avg Path Length |
|-------|-------------|-----------|-----------------|
| scene_1 | 7 | 0% | 3.095m |
| scene_2 | 9 | 33% | 10.292m |
| scene_3 | 11 | 0% | 1.860m |
| scene_4 | 11 | 100% | 3.008m |
| unknown | 1,000 | 50.5% | 2.659m |

## 🐛 Troubleshooting

### Issue: "No trajectory files found"

**Cause:** Incorrect trajectory directory path

**Solution:**
```bash
# Check directory exists
ls trajectoryToLearn/world_json

# Verify it contains scene folders
ls trajectoryToLearn/world_json/scene_1
ls trajectoryToLearn/world_json/scene_2
# etc.

# If missing, check you've unzipped the trajectory data
```

### Issue: "Out of memory"

**Cause:** Too many trajectories loaded at once

**Solution:**
```bash
# Limit trajectories to reduce memory
--max_trajectories 500

# Or reduce number of environments
--num_envs 2048  # Instead of 4096
```

### Issue: Training very slow (< 1000 FPS)

**Cause:** Possible base frozen issue still present

**Solution:**
1. Check base diagnostics in logs
2. Verify action scaling is applied
3. Check actuator configuration
4. Run visual test to confirm base moves

### Issue: All rewards negative

**Cause:** Policy not learning / reward function issues

**Solution:**
1. Check if episodes terminate early (< 100 steps)
2. Verify observation normalization is working
3. Reduce learning rate: `--learning_rate 0.0001`
4. Increase entropy for exploration: `--ent_coef 0.01`

## 📝 Comparing Trajectory Modes

| Feature | Circle | All Trajectories | Chassis-Only |
|---------|--------|------------------|--------------|
| **Diversity** | Very Low | ✅ Very High | Medium |
| **Realism** | Low | ✅ High | High |
| **Difficulty** | Medium | ✅ Varied | High |
| **Train Time** | Short | Long | Medium |
| **Use Case** | Debugging | ✅ Production | Testing |
| **Base Learning** | Some | ✅ Optimal | ✅ Focused |
| **Generalization** | Poor | ✅ Excellent | Limited |

## 🎓 Best Practices

### ✅ DO:
- Use `--use_all_trajectories` for main training
- Monitor base diagnostics during training
- Save checkpoints frequently (`--save_freq 100000`)
- Use wandb logging for long runs (`--wandb`)
- Verify base movement with visual tests first
- Train for sufficient timesteps (100M+)

### ❌ DON'T:
- Use `--use_chassis_only` for main training (biased!)
- Train on single trajectory type only
- Skip visual validation of base movement
- Ignore base diagnostics in logs
- Train without entropy decay/KL scheduling
- Stop training too early (< 50M steps)

## 🚀 Next Steps After Training

1. **Evaluate on Test Set**
   ```bash
   python scripts/reinforcement_learning/sb3/evaluate.py \
       --checkpoint logs/.../final_model.zip \
       --num_episodes 100
   ```

2. **Visual Validation**
   ```bash
   python scripts/test_chassis_trajectories.py \
       --checkpoint logs/.../final_model.zip \
       --num 20
   ```

3. **Analyze Performance by Trajectory Type**
   - Group trajectories by type (arc_left, push, etc.)
   - Calculate success rate per group
   - Identify weaknesses for additional training

4. **Deploy to Real Robot** (if applicable)
   - Export policy to ONNX
   - Test in simulation first
   - Gradual real-world validation

## 📚 Related Documentation

- [Trajectory Analysis Summary](TRAJECTORY_ANALYSIS_SUMMARY.md) - Full statistics
- [Testing Recorded Trajectories](TESTING_RECORDED_TRAJECTORIES.md) - Visual validation guide
- [Base Movement Analysis](BASE_MOVEMENT_COMPREHENSIVE_ANALYSIS.md) - Original problem & fixes

---

**Happy Training!** 🎉

Remember: Using all 1,038 trajectories ensures your robot learns a robust, generalizable policy that knows when to use base movement vs arm-only strategies. This is the key to real-world performance!
