# Training on Windows with Isaac Sim

## Overview 🪟

This guide covers setting up and running RL training natively on Windows using Isaac Sim GUI or headless mode.

## Why Windows Training?

**Advantages**:
- ✅ Full GPU support (no WSL limitations)
- ✅ Isaac Sim GUI available for debugging
- ✅ No WSL CUDA driver issues
- ✅ Native Windows performance
- ✅ Can visualize training live

**When to use**:
- WSL has GPU/CUDA issues
- Need visual debugging
- Want to see robot in action
- Recording demo videos

---

## Prerequisites

### 1. Isaac Sim Installation
```
Location: I:\isaacsim\
Version: 5.0.0 or newer
```

### 2. Python Environment
```
Location: I:\isaaclab\
Python: 3.11
Isaac Lab: 2.2.0
```

### 3. Project Location
```
Location: I:\wSpace\cinebotRL\
(or C:\Users\yanbo\wSpace\cinebotRL\)
```

---

## Setup Steps

### Step 1: Activate Isaac Lab Environment

```powershell
# PowerShell
cd I:\isaaclab
.\isaaclab.bat

# This activates the conda/venv and sets up paths
```

### Step 2: Verify CUDA

```powershell
# In Isaac Lab environment
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"
```

Expected output:
```
CUDA available: True
Device: NVIDIA GeForce RTX 3090
```

### Step 3: Install Additional Dependencies

```powershell
# In Isaac Lab environment
cd I:\wSpace\cinebotRL

# Install project in editable mode
pip install -e .

# Install Stable Baselines3
pip install stable-baselines3[extra]
```

### Step 4: Test Environment

```powershell
# Headless test (no GUI)
python scripts\test_mobile_mm_env.py --num_envs 1 --steps 5 --headless

# With GUI (slower but visual)
python scripts\test_mobile_mm_env.py --num_envs 1 --steps 5
```

---

## Training Commands

### Headless Training (Recommended)

```powershell
# Fast headless training
python scripts\reinforcement_learning\sb3\train.py `
    --task MobileMMTrackEE-v0 `
    --num_envs 1024 `
    --headless `
    --total_timesteps 5000000 `
    --save_freq 100000
```

### With Periodic Visualization

```powershell
# Train mostly headless, render occasionally
python scripts\reinforcement_learning\sb3\train.py `
    --task MobileMMTrackEE-v0 `
    --num_envs 512 `
    --headless `
    --render_every_n_steps 10000 `
    --total_timesteps 5000000
```

### Multi-Trajectory Training

```powershell
# Train on 1000+ diverse trajectories
python scripts\reinforcement_learning\sb3\train.py `
    --task MobileMMTrackEE-v0 `
    --num_envs 1024 `
    --headless `
    --trajectory_type multi_recorded `
    --trajectory_dir trajectoryToLearn\world_json\cinematic_db `
    --total_timesteps 10000000
```

---

## Monitoring Training

### TensorBoard

```powershell
# Terminal 1: Train
python scripts\reinforcement_learning\sb3\train.py ...

# Terminal 2: Monitor
tensorboard --logdir logs\sb3\MobileMMTrackEE-v0
# Open: http://localhost:6006
```

### Key Metrics to Watch

1. **rollout/ep_rew_mean**: Average episode reward
   - Target: >10 per step (>2000 per episode)
   
2. **reward_components/self_collision_penalty**: Should decrease to ~0
   
3. **reward_components/position_tracking**: Should increase to ~10

4. **train/explained_variance**: Should be >0.7 (good value function)

5. **train/loss**: Should decrease steadily

---

## GPU Configuration

### Single GPU (RTX 3090)

Isaac Lab will automatically use the available GPU. No configuration needed!

### Multiple GPUs

If you have multiple GPUs and want to select a specific one:

```python
# In your training script
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(
    headless=True,
    device="cuda:0",  # or "cuda:1", etc.
)
```

---

## Troubleshooting

### Issue: "No CUDA devices found"

**Solution**:
```powershell
# Check CUDA installation
nvidia-smi

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

If False, reinstall PyTorch with CUDA:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Issue: "Out of memory"

**Solution**: Reduce `num_envs`:
```powershell
python scripts\reinforcement_learning\sb3\train.py --num_envs 512  # Instead of 1024
```

### Issue: GUI crashes during training

**Solution**: Use headless mode:
```powershell
python scripts\reinforcement_learning\sb3\train.py --headless
```

### Issue: Slow training

**Check**:
1. Are you in headless mode? (--headless flag)
2. Is GPU being used? (check nvidia-smi during training)
3. Are you using too many environments? (reduce --num_envs)

**Expected speed** (RTX 3090, headless, 1024 envs):
- ~100,000 steps/hour
- ~2.8 steps/second per environment

---

## Recording Demo Videos

After training, record a video of the trained policy:

```powershell
# Load checkpoint and record
python scripts\record_policy.py `
    --checkpoint logs\sb3\MobileMMTrackEE-v0\best_model.zip `
    --output videos\demo.mp4 `
    --num_episodes 5
```

---

## Checkpoint Management

### Auto-saving

Checkpoints are saved automatically during training:
```
logs/sb3/MobileMMTrackEE-v0/
├── best_model.zip        # Best performing model
├── model_100000_steps.zip
├── model_200000_steps.zip
└── ...
```

### Manual saving

```python
# In your training script
model.save("checkpoints/my_model.zip")
```

### Loading checkpoint

```python
from stable_baselines3 import PPO

model = PPO.load("checkpoints/my_model.zip")
```

---

## Comparing Windows vs WSL

| Aspect | Windows | WSL |
|--------|---------|-----|
| **Setup** | Simpler | More complex |
| **GPU Support** | Native | Requires driver setup |
| **Performance** | ~100K steps/hr | ~80K steps/hr (if working) |
| **GUI** | Available | No (headless only) |
| **Debugging** | Easier | Harder |
| **Recommendation** | ✅ Use for training | Use for Linux workflows |

---

## Next Steps

1. ✅ Test environment loads: `python scripts\test_mobile_mm_env.py`
2. ✅ Start training: Use headless mode with 1024 envs
3. ✅ Monitor TensorBoard: Watch reward components
4. ✅ Tune hyperparameters: Adjust reward weights if needed
5. ✅ Train diverse policy: Use multi-trajectory mode

---

## Quick Reference

```powershell
# Activate environment
cd I:\isaaclab && .\isaaclab.bat

# Quick test
cd I:\wSpace\cinebotRL
python scripts\test_mobile_mm_env.py --num_envs 1 --steps 2

# Start training
python scripts\reinforcement_learning\sb3\train.py `
    --task MobileMMTrackEE-v0 `
    --num_envs 1024 `
    --headless

# Monitor
tensorboard --logdir logs\sb3
```

**Happy training!** 🚀
