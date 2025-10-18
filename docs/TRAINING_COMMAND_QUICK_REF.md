# Quick Command Reference: Training with All Trajectories

## 🎯 Main Training Command (USE THIS!)

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

**This trains on ALL 1,038 trajectories** ✅

## 📊 What's Different from Before?

| Old (Circle only) | New (All trajectories) |
|-------------------|------------------------|
| `--trajectory_type circle` (default) | `--trajectory_type multi_recorded` |
| (no trajectory flags) | `--use_all_trajectories` |
| 1 synthetic trajectory | 1,038 real trajectories |
| Low diversity | ✅ High diversity |
| 519 arm-only, 519 chassis-required | ✅ 50/50 balanced! |

## 🎬 Test vs Train

### For TESTING Base Movement (Visual Validation)

```bash
# Use chassis-only trajectories
python scripts/test_chassis_trajectories.py \
    --num 10 \
    --envs 4
```

**Purpose:** Prove visually that base CAN move

### For TRAINING (Main Runs)

```bash
# Use ALL trajectories
.\isaaclab.bat -p ... \
    --trajectory_type multi_recorded \
    --use_all_trajectories \
    --headless
```

**Purpose:** Train robust policy that learns when to use base

## 🔧 Quick Modifications

### Train Faster (Fewer Trajectories)

```bash
... \
    --use_all_trajectories \
    --max_trajectories 100
```

### Train Longer (Better Convergence)

```bash
... \
    --total_timesteps 500000000  # 500M instead of 100M
```

### Use Fewer Environments (Less Memory)

```bash
... \
    --num_envs 2048  # Instead of 4096
```

### Enable Logging

```bash
... \
    --wandb \
    --wandb_project cinebotrl
```

## ✅ Checklist Before Training

- [ ] Trajectory directory exists: `ls trajectoryToLearn/world_json`
- [ ] Analysis complete: `trajectory_analysis_results.csv` exists
- [ ] Base movement validated: Ran visual test, base moves ✓
- [ ] Using `--use_all_trajectories` (not `--use_chassis_only`)
- [ ] Sufficient disk space for checkpoints (~2GB per 100M steps)
- [ ] Isaac Sim GPU drivers updated

## 🎯 Expected Results

**During Training:**
- FPS: 8000-9000
- Rewards: Gradually increasing
- Episode length: ~950-1000 steps (near max)
- Base diagnostics: Non-zero velocities

**After 100M Steps:**
- Success rate: 75%+ on diverse trajectories
- Base movement: Strategic (used when needed)
- Arm-only: Used for short-range movements
- Coordination: Smooth base + arm motion

## 📝 Key Files

- **Training script:** `scripts/reinforcement_learning/sb3/train.py`
- **Test script:** `scripts/test_chassis_trajectories.py`
- **Full guide:** `docs/TRAINING_WITH_RECORDED_TRAJECTORIES.md`
- **Analysis results:** `trajectory_analysis_results.csv`
- **Chassis indices:** `chassis_required_indices.txt`

## 🚨 Important Reminders

1. **For training:** Use `--use_all_trajectories` ✅
2. **For testing:** Use `scripts/test_chassis_trajectories.py` ✅
3. **DON'T mix them up!** Training on chassis-only = biased policy ❌

---

**One command to train them all:** The command at the top trains on all 1,038 trajectories for a robust, generalizable policy! 🎉
