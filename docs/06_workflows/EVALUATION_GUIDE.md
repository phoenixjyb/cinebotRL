# Evaluation Guide

Complete guide for evaluating your trained MobileMMTrackEE model and visualizing it against recorded trajectories.

## Quick Start

### 1. Basic Visualization (4 environments, GUI enabled)

```powershell
& "I:\isaaclab\isaaclab.bat" -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\evaluate.py `
    --checkpoint C:\Users\yanbo\wSpace\cinebotRL\logs\sb3\mobilemmtrackee_v0\20251018_001233\final_model.zip `
    --num_envs 4 `
    --num_episodes 10 `
    --deterministic
```

**What you'll see:**
- 🔴 **Red spheres** = Target trajectory points (from your recorded data)
- 🟢 **Green spheres** = Robot end-effector position
- Watch if the green sphere follows the red sphere accurately!

### 2. Test on ALL Training Trajectories (1,038 trajectories)

```powershell
& "I:\isaaclab\isaaclab.bat" -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\evaluate.py `
    --checkpoint C:\Users\yanbo\wSpace\cinebotRL\logs\sb3\mobilemmtrackee_v0\20251018_001233\final_model.zip `
    --num_envs 4 `
    --num_episodes 20 `
    --deterministic `
    --trajectory_type multi_recorded `
    --use_all_trajectories
```

**Why use this:**
- Tests generalization across all 1,038 recorded trajectories
- Randomly samples from the full training dataset
- See performance on diverse real-world motions

### 3. Test on Chassis-Required Trajectories Only (519 trajectories)

```powershell
& "I:\isaaclab\isaaclab.bat" -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\evaluate.py `
    --checkpoint C:\Users\yanbo\wSpace\cinebotRL\logs\sb3\mobilemmtrackee_v0\20251018_001233\final_model.zip `
    --num_envs 4 `
    --num_episodes 20 `
    --deterministic `
    --trajectory_type multi_recorded `
    --use_chassis_only
```

**Why use this:**
- Tests base movement capability specifically
- Only includes trajectories that require chassis motion
- Validates whole-body coordination

### 4. Headless Performance Testing (no GUI, faster)

```powershell
& "I:\isaaclab\isaaclab.bat" -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\evaluate.py `
    --checkpoint C:\Users\yanbo\wSpace\cinebotRL\logs\sb3\mobilemmtrackee_v0\20251018_001233\final_model.zip `
    --num_envs 64 `
    --num_episodes 100 `
    --deterministic `
    --trajectory_type multi_recorded `
    --use_all_trajectories `
    --headless
```

**Why use this:**
- Get quantitative metrics without visualization overhead
- Run many more parallel environments (64 instead of 4)
- Faster evaluation for large-scale testing
- Get statistical summary (mean/std/min/max rewards)

---

## Command Line Arguments Explained

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--checkpoint` | Path to trained model `.zip` file | `logs/sb3/.../final_model.zip` |

### Environment Settings

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | `MobileMMTrackEE-v0` | Task ID (usually no need to change) |
| `--num_envs` | `4` | Number of parallel environments (4 for visualization, 16-64 for headless) |
| `--headless` | `False` | Run without GUI (faster, metrics only) |

### Evaluation Settings

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_episodes` | `10` | Number of episodes to evaluate |
| `--deterministic` | `False` | Use deterministic actions (no exploration noise) - **recommended for evaluation** |
| `--device` | `auto` | Device to run on (`auto`, `cpu`, `cuda`) |

### Trajectory Configuration (NEW!)

| Argument | Default | Description |
|----------|---------|-------------|
| `--trajectory_type` | `circle` | Trajectory type: `line`, `circle`, `figure_eight`, `recorded`, `multi_recorded` |
| `--trajectory_dir` | `trajectoryToLearn/world_json` | Directory with recorded trajectories |
| `--use_all_trajectories` | `False` | Use ALL 1,038 trajectories (recommended) |
| `--use_chassis_only` | `False` | Use only 519 chassis-requiring trajectories |
| `--max_trajectories` | `None` | Limit number of trajectories (useful for quick tests) |

---

## Understanding the Output

### During Evaluation

```
Running 10 episodes...
  Episode 1/10: Reward=1542.35, Length=512
  Episode 2/10: Reward=1238.67, Length=512
  ...
```

- **Reward**: Higher is better (indicates good tracking)
- **Length**: Number of timesteps (controlled by `max_episode_length` in config)

### Summary Statistics

```
======================================================================
Evaluation Summary
======================================================================
Episodes completed: 10
Mean reward: 1456.23 ± 145.67
Min reward: 1123.45
Max reward: 1678.90
Mean episode length: 512.0 ± 0.0
======================================================================
```

- **Mean reward**: Average performance across episodes
- **±**: Standard deviation (lower = more consistent)
- **Min/Max**: Range of performance

---

## Visualization Tips

### Best Settings for Visualization

```powershell
# For best visualization experience:
--num_envs 1         # Watch a single environment closely
--num_episodes 5     # Don't run too many
--deterministic      # Consistent behavior
# (no --headless)    # GUI enabled by default
```

### What to Look For

1. **Tracking Accuracy**
   - Does the green sphere (end-effector) follow the red sphere (target)?
   - How close does it get?
   - Does it lag behind or overshoot?

2. **Smoothness**
   - Are movements smooth or jerky?
   - Does the robot oscillate around the target?

3. **Base Movement**
   - Does the chassis move when needed?
   - Does it stay still when arm can reach?

4. **Self-Collision**
   - Does the robot collide with itself?
   - Are movements safe?

5. **Failure Modes**
   - When does tracking fail?
   - What trajectories are harder?

---

## Performance Benchmarking

### Quick Test (5 minutes)

```powershell
& "I:\isaaclab\isaaclab.bat" -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\evaluate.py `
    --checkpoint <YOUR_CHECKPOINT> `
    --num_envs 16 `
    --num_episodes 50 `
    --deterministic `
    --trajectory_type multi_recorded `
    --use_all_trajectories `
    --headless
```

### Full Benchmark (30 minutes)

```powershell
& "I:\isaaclab\isaaclab.bat" -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\evaluate.py `
    --checkpoint <YOUR_CHECKPOINT> `
    --num_envs 64 `
    --num_episodes 500 `
    --deterministic `
    --trajectory_type multi_recorded `
    --use_all_trajectories `
    --headless
```

---

## Comparing Different Checkpoints

### Compare Against Intermediate Checkpoints

Your training saves checkpoints during training:

```
logs/sb3/mobilemmtrackee_v0/20251018_001233/
├── final_model.zip              # After 100M timesteps
├── checkpoints/
│   ├── rl_model_25000000_steps.zip   # After 25M timesteps
│   ├── rl_model_50000000_steps.zip   # After 50M timesteps
│   └── rl_model_75000000_steps.zip   # After 75M timesteps
```

Evaluate each checkpoint to see learning progress:

```powershell
# Test checkpoint at 25M
& "I:\isaaclab\isaaclab.bat" -p scripts\reinforcement_learning\sb3\evaluate.py `
    --checkpoint logs\sb3\mobilemmtrackee_v0\20251018_001233\checkpoints\rl_model_25000000_steps.zip `
    --num_envs 16 --num_episodes 50 --deterministic --headless

# Test final model at 100M
& "I:\isaaclab\isaaclab.bat" -p scripts\reinforcement_learning\sb3\evaluate.py `
    --checkpoint logs\sb3\mobilemmtrackee_v0\20251018_001233\final_model.zip `
    --num_envs 16 --num_episodes 50 --deterministic --headless
```

---

## Troubleshooting

### Issue: "Checkpoint not found"

**Solution:** Use absolute path or check file exists:

```powershell
Test-Path "C:\Users\yanbo\wSpace\cinebotRL\logs\sb3\mobilemmtrackee_v0\20251018_001233\final_model.zip"
```

### Issue: "Out of memory" with GUI

**Solution:** Reduce `num_envs`:

```powershell
--num_envs 1  # Minimal memory usage
```

### Issue: Evaluation is slow

**Solutions:**
1. Use `--headless` for faster evaluation
2. Increase `num_envs` (16-64) in headless mode
3. Reduce `num_episodes` for quick tests

### Issue: Robot behavior looks wrong

**Check:**
1. Are you using `--deterministic`? (Recommended for evaluation)
2. Is the checkpoint from the correct training run?
3. Does the trajectory type match training? (Use `multi_recorded` if trained with it)

### Issue: Visualization is choppy

**Solutions:**
1. Reduce `num_envs` to 1-4 for smooth rendering
2. Close other GPU-intensive applications
3. Check GPU utilization with `nvidia-smi`

---

## Advanced Usage

### Custom Trajectory Testing

To test on a specific subset of trajectories:

```powershell
# Limit to first 100 trajectories for quick testing
--max_trajectories 100
```

### Non-Deterministic Evaluation

To see exploration behavior (not recommended for benchmarking):

```powershell
# Remove --deterministic flag
& "I:\isaaclab\isaaclab.bat" -p scripts\reinforcement_learning\sb3\evaluate.py `
    --checkpoint <YOUR_CHECKPOINT> `
    --num_envs 4 `
    --num_episodes 10
```

### CPU-Only Evaluation

If GPU is unavailable or for debugging:

```powershell
--device cpu
```

---

## Next Steps After Evaluation

1. **If performance is good:**
   - Deploy to real robot
   - Create deployment documentation
   - Share results

2. **If performance needs improvement:**
   - Check Tensorboard logs to identify issues
   - Adjust reward weights
   - Continue training with `--checkpoint` resume
   - Try different hyperparameters

3. **For analysis:**
   - Record evaluation videos
   - Export trajectory data
   - Analyze failure cases
   - Compare different checkpoints

---

## Related Documentation

- [Training Guide](./TRAIN_ON_WINDOWS.md) - How to train models
- [Multi-Trajectory Training](./workflows/multi_trajectory_training.md) - Training with recorded trajectories
- [Visualization Options](./workflows/visualization_options.md) - Different ways to visualize
- [Quick Reference](./QUICK_REFERENCE.md) - Command cheatsheet
