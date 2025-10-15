# Visualization During Training

## Overview

While training runs in **headless mode** (no GUI), you can launch a **separate Isaac Sim instance** with GUI enabled to visualize and inspect the environment. This allows you to:

- ✅ See your robot in action
- ✅ Inspect the USD asset and scene setup
- ✅ Verify joint articulation and movement
- ✅ Check trajectory targets and obstacles
- ✅ Debug environment behavior

Both instances (training + visualization) can run simultaneously on your dual-GPU setup.

---

## Quick Start

### Option 1: Environment Inspector (Recommended First)

**Best for:** Checking robot setup, scene layout, asset validation

```powershell
# Inspect single robot
.\scripts\inspect_environment.ps1

# Inspect multiple robots in grid
.\scripts\inspect_environment.ps1 -NumEnvs 4
```

**What you'll see:**
- Robot USD asset loaded in the scene
- Random policy (untrained) - robot will move randomly
- End-effector trajectory visualization
- Ground plane, lighting, scene elements

---

### Option 2: Live Training Visualization

**Best for:** Watching the environment during training (random policy, no checkpoint loaded yet)

```powershell
# Visualize with 16 environments
.\scripts\visualize_training.ps1

# Visualize with fewer envs for better performance
.\scripts\visualize_training.ps1 -NumEnvs 4

# Single robot view
.\scripts\visualize_training.ps1 -NumEnvs 1
```

**Note:** This launches a NEW Isaac Sim instance separate from your headless training. The robots will show random behavior since no checkpoint is loaded.

---

### Option 3: Policy Visualization (After Training)

**Best for:** Seeing your trained policy in action

```powershell
# Use latest checkpoint automatically
.\scripts\visualize_policy.ps1 -Latest

# Or specify checkpoint manually
.\scripts\visualize_policy.ps1 -Checkpoint "I:\isaaclab\logs\sb3\MobileMMTrackEE-v0\2025-10-15_14-30-45\checkpoints\model_1000000_steps.zip"
```

**Note:** Requires checkpoints to be saved (training must run for a bit first). Currently launches environment visualization - full policy loading requires extending `train.py` with test mode.

---

## How It Works

### Dual-GPU Setup

Your system has two GPUs that can be utilized simultaneously:

```
RTX 3090 (Device 0)         Quadro P2000 (Device 1)
└── Headless Training       └── GUI Visualization
    (Terminal 1)                (Terminal 2)
    - 64-1024 environments      - 1-16 environments
    - No rendering              - Full rendering
    - Fast training             - Interactive inspection
```

### Separate Isaac Sim Instances

- **Training (Headless):** Runs via `launch_training_windows.ps1` with `--headless true`
- **Visualization (GUI):** Runs via new scripts with `--headless false`
- Both can run at the same time using different GPU resources

---

## Usage Guide

### 1. Start Training (First Terminal - Already Running)

```powershell
# Your current headless training
.\scripts\launch_training_windows.ps1 -Task MobileMMTrackEE-v0 -NumEnvs 64 -Headless
```

### 2. Open Visualization (New Terminal)

```powershell
# Open a NEW PowerShell terminal
# Navigate to project
cd C:\Users\yanbo\wSpace\cinebotRL

# Launch inspector
.\scripts\inspect_environment.ps1
```

### 3. What You'll See

**Isaac Sim GUI Window Opens:**
- Scene with your mobile manipulator robot(s)
- End-effector moving along trajectory
- Robot joints articulating
- Random policy (since no checkpoint loaded)

**Controls:**
- **Mouse:** Rotate/pan camera view
- **Scroll:** Zoom in/out
- **Click robot:** Select and view properties in side panel
- **ESC:** Close window

---

## Performance Tips

### Environment Count Recommendations

| Purpose | NumEnvs | Performance | Best For |
|---------|---------|-------------|----------|
| Detailed inspection | 1 | Excellent | Close-up robot viewing |
| Grid layout | 4 | Very Good | Multiple angle observation |
| Training observation | 16 | Good | Seeing population behavior |
| Heavy visualization | 64+ | Poor | Not recommended for GUI |

### GPU Memory Management

If you run into GPU memory issues:

1. **Reduce visualization envs:** Use `-NumEnvs 1` or `-NumEnvs 4`
2. **Lower training envs temporarily:** Reduce training from 1024 to 512 or 256
3. **Use inspector only:** `inspect_environment.ps1` uses minimal resources

---

## Checkpoint Locations

Training checkpoints are saved here:

```
I:\isaaclab\logs\sb3\MobileMMTrackEE-v0\
└── <timestamp>/
    └── checkpoints/
        ├── model_50000_steps.zip
        ├── model_100000_steps.zip
        ├── model_150000_steps.zip
        └── ...
```

**Checkpoint saving frequency:** Depends on SB3 configuration (typically every 50K-100K steps)

---

## Current Training Status

Check your current training:

```powershell
# View logs
.\scripts\monitor_training.ps1 -Mode logs

# Check GPU usage
.\scripts\monitor_training.ps1 -Mode gpu
```

---

## Troubleshooting

### "Cannot launch Isaac Sim - already running"

**Solution:** Isaac Sim can run multiple instances if configured correctly. If you get this error:
1. Make sure training is using `CUDA_VISIBLE_DEVICES=0` (RTX 3090)
2. Visualization can use Device 1 (Quadro P2000) or share Device 0

### "Out of memory"

**Solutions:**
- Reduce visualization envs: Use `-NumEnvs 1`
- Close other GPU applications
- Temporarily lower training env count

### "Slow GUI performance"

**Solutions:**
- Use fewer environments: `-NumEnvs 1` or `-NumEnvs 4`
- The Quadro P2000 is less powerful - keep env count low for GUI
- Close unnecessary applications

### "Robot behaving randomly"

**Expected!** The visualization scripts don't load trained checkpoints yet. To implement:
- Extend `train.py` with `--test` mode and `--checkpoint` loading
- Modify SB3 wrapper to support evaluation mode
- See [TRAINING_SUCCESS.md](TRAINING_SUCCESS.md) for architecture details

---

## Next Steps: Implementing Policy Loading

To visualize trained policies (not just random behavior), you'll need to:

1. **Add test mode to train.py:**
```python
parser.add_argument("--test", action="store_true", help="Test mode (no training)")
parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file")
```

2. **Load checkpoint:**
```python
if args.test and args.checkpoint:
    model = PPO.load(args.checkpoint, env=vec_env)
```

3. **Run evaluation loop:**
```python
obs = vec_env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
```

4. **Use the visualization:**
```powershell
.\scripts\visualize_policy.ps1 -Latest
```

---

## Summary

**Right now, while training runs:**

```powershell
# Open NEW PowerShell terminal
cd C:\Users\yanbo\wSpace\cinebotRL

# Quick inspection (1 robot)
.\scripts\inspect_environment.ps1

# Or view multiple robots
.\scripts\inspect_environment.ps1 -NumEnvs 4
```

**The GUI will show:**
- ✅ Your robot asset and scene
- ✅ Joint articulation and movement  
- ✅ End-effector trajectory
- ⚠️ Random policy (not trained) - this is normal for inspection

**Your training continues uninterrupted in the original terminal!**

---

**Created:** 2025-10-15  
**For:** Visualization during headless training
