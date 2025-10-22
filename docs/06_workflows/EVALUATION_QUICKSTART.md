# 🎯 Evaluation Quick Start

Your 100M timestep training is complete! Here's how to visualize and test your trained robot.

## ⚡ Fastest Way to Visualize

```powershell
# Run the convenient launcher script
.\scripts\evaluate_model.ps1 -Mode visualize
```

This will show you 4 parallel robots following trajectories from your training dataset with a nice GUI.

---

## 📋 All Evaluation Modes

### 1. **Visual Inspection** (Default)
```powershell
.\scripts\evaluate_model.ps1 -Mode visualize -NumEpisodes 10
```
- 🔴 Red spheres = Target trajectory
- 🟢 Green spheres = Robot end-effector
- Watch if green follows red!

### 2. **Test on ALL 1,038 Trajectories**
```powershell
.\scripts\evaluate_model.ps1 -Mode test-all -NumEpisodes 20
```
- Samples from your entire training dataset
- Validates generalization

### 3. **Test Chassis Movement** (519 trajectories)
```powershell
.\scripts\evaluate_model.ps1 -Mode test-chassis -NumEpisodes 20
```
- Only trajectories that require base movement
- Validates whole-body coordination

### 4. **Quick Performance Benchmark** (~5 min)
```powershell
.\scripts\evaluate_model.ps1 -Mode benchmark-quick
```
- Headless (no GUI), 16 parallel environments
- Get quantitative metrics fast

### 5. **Full Statistical Benchmark** (~30 min)
```powershell
.\scripts\evaluate_model.ps1 -Mode benchmark-full
```
- Headless, 64 parallel environments
- Comprehensive statistics over 500 episodes

---

## 🎮 Manual Control (Advanced)

If you want full control over evaluation parameters:

```powershell
& "I:\isaaclab\isaaclab.bat" -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\evaluate.py `
    --checkpoint C:\Users\yanbo\wSpace\cinebotRL\logs\sb3\mobilemmtrackee_v0\20251018_001233\final_model.zip `
    --num_envs 4 `
    --num_episodes 10 `
    --deterministic `
    --trajectory_type multi_recorded `
    --use_all_trajectories
```

### Key Arguments

| Argument | Options | Description |
|----------|---------|-------------|
| `--checkpoint` | Path to `.zip` | Your trained model |
| `--num_envs` | 1-64 | Parallel environments (4 for GUI, 16-64 headless) |
| `--num_episodes` | Any number | How many episodes to run |
| `--deterministic` | Flag | No exploration noise (recommended) |
| `--trajectory_type` | `multi_recorded` | Use your training trajectories |
| `--use_all_trajectories` | Flag | All 1,038 trajectories |
| `--use_chassis_only` | Flag | Only 519 chassis-requiring ones |
| `--headless` | Flag | No GUI (faster) |

---

## 📊 Understanding the Output

### During Evaluation
```
Episode 1/10: Reward=1542.35, Length=512
Episode 2/10: Reward=1238.67, Length=512
```

### Summary Statistics
```
Mean reward: 1456.23 ± 145.67
Min reward: 1123.45
Max reward: 1678.90
```

**Higher reward = Better tracking performance**

---

## 🔍 What to Look For

### Visual Inspection ✅
1. **Tracking accuracy** - Does green follow red closely?
2. **Smoothness** - Are movements smooth or jerky?
3. **Base usage** - Does chassis move when needed?
4. **Self-collision** - Any unsafe movements?

### Metrics ✅
1. **Mean reward** - Overall performance
2. **Std deviation** - Consistency (lower is better)
3. **Min/Max** - Performance range

---

## 🔧 Common Issues

### "Checkpoint not found"
```powershell
# List available checkpoints
Get-ChildItem C:\Users\yanbo\wSpace\cinebotRL\logs\sb3\mobilemmtrackee_v0\* -Recurse -Filter "*.zip"
```

### Out of memory with GUI
```powershell
# Reduce environments
.\scripts\evaluate_model.ps1 -Mode visualize  # Uses 4 by default
```

### Want to see exploration behavior
```powershell
# Add -Stochastic flag
.\scripts\evaluate_model.ps1 -Mode visualize -Stochastic
```

---

## 📖 Detailed Documentation

- **[EVALUATION_GUIDE.md](./EVALUATION_GUIDE.md)** - Complete evaluation guide
- **[workflows/multi_trajectory_training.md](./workflows/multi_trajectory_training.md)** - Training details
- **[workflows/visualization_options.md](./workflows/visualization_options.md)** - More visualization options

---

## 🚀 Your Latest Training Run

**Checkpoint Location:**
```
C:\Users\yanbo\wSpace\cinebotRL\logs\sb3\mobilemmtrackee_v0\20251018_001233\final_model.zip
```

**Training Stats:**
- ✅ 100.1M timesteps completed
- ✅ 1,038 trajectories used
- ✅ 191 iterations (4.86 hours)
- ✅ 5,716 FPS throughput
- ✅ Explained variance: 0.994 (excellent!)

**Start Evaluating:**
```powershell
.\scripts\evaluate_model.ps1
```

---

## 🎯 Next Steps

1. **Visualize immediately** - `.\scripts\evaluate_model.ps1`
2. **Run quick benchmark** - `.\scripts\evaluate_model.ps1 -Mode benchmark-quick`
3. **Compare checkpoints** - Test 25M, 50M, 75M, 100M checkpoints
4. **Analyze failures** - Watch which trajectories are difficult
5. **Document results** - Record videos and metrics
6. **Deploy or retrain** - Based on performance

---

**Questions?** Check [EVALUATION_GUIDE.md](./EVALUATION_GUIDE.md) for comprehensive documentation.
