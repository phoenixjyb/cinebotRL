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

### 2. Isaac Lab Installation
```
Location: I:\isaaclab\
Python: 3.11 (included)
Isaac Lab: 2.2.0
```

### 3. Project Location
```
Location: I:\wSpace\cinebotRL\
(or C:\Users\yanbo\wSpace\cinebotRL\)
```

---

## Python Environment Setup

You have **two options** for the Python environment:

### Option A: Use Isaac Lab's Environment (Recommended) ✅

Isaac Lab comes with a complete Python environment. **No separate venv needed!**

```powershell
cd I:\isaaclab
.\isaaclab.bat

# This automatically:
# - Activates Isaac Lab's conda/venv environment
# - Sets up all Isaac Sim paths
# - Includes PyTorch, CUDA, and all dependencies
```

**Advantages**:
- ✅ Pre-configured for Isaac Sim
- ✅ All dependencies included
- ✅ No version conflicts
- ✅ Just works!

**Then install your project**:
```powershell
cd I:\wSpace\cinebotRL

# Install project in editable mode
pip install -e .

# Install Stable Baselines3
pip install stable-baselines3[extra]
```

### Option B: Create Dedicated venv (Advanced)

Only if you need full control over packages:

```powershell
# Create new venv
cd I:\wSpace\cinebotRL
python -m venv .venv_windows

# Activate it
.\.venv_windows\Scripts\activate

# Install Isaac Sim packages
pip install isaacsim==5.0.0.0
pip install isaaclab==2.2.0

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install your project
pip install -e .
pip install stable-baselines3[extra]
```

**Note**: This guide assumes **Option A** (using Isaac Lab's environment).

---

## Setup Steps

### Step 1: Activate Isaac Lab Environment

```powershell
# PowerShell
cd I:\isaaclab
.\isaaclab.bat

# This activates the environment and sets up paths
# You should see something like: (isaac-lab) PS I:\isaaclab>
```

### Step 2: Install Your Project

```powershell
# Still in Isaac Lab environment
cd I:\wSpace\cinebotRL

# Install project in editable mode
pip install -e .

# Install Stable Baselines3
pip install stable-baselines3[extra]
```

### Step 3: Verify CUDA

```powershell
# In Isaac Lab environment
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"
```

Expected output:
```
CUDA available: True
Device: NVIDIA GeForce RTX 3090
```

### Step 4: Test Environment

```powershell
# Still in Isaac Lab environment
cd I:\wSpace\cinebotRL

# Headless test (no GUI)
python scripts\test_mobile_mm_env.py --num_envs 1 --steps 5 --headless

# With GUI (slower but visual)
python scripts\test_mobile_mm_env.py --num_envs 1 --steps 5
```

**Expected output**:
```
✓ Environment created
✓ Robot loaded
✓ EE link 'left_gripper_link' found
✓ Step 0: reward=X.XXXX
✓ Step 1: reward=X.XXXX
...
✓ Test completed successfully
```

---

## Important Notes 📝

### Environment Isolation

Your WSL and Windows environments are **completely separate**:

| Environment | Location | Python | Purpose |
|-------------|----------|--------|---------|
| **WSL** | `/mnt/c/Users/yanbo/wSpace/cinebotRL/.venv_rl311/` | 3.11 | WSL development |
| **Windows** | `I:\isaaclab\` (built-in) | 3.11 | Windows training |

**No conflicts!** You can switch between them freely.

### When to Use Each

**Use Isaac Lab's environment** (`.\isaaclab.bat`) when:
- ✅ Training with Isaac Sim
- ✅ Running environment tests
- ✅ Recording demos
- ✅ Debugging with GUI

**Use separate venv** only if:
- ⚠️ You need specific package versions
- ⚠️ You want isolation from Isaac Lab
- ⚠️ You're doing development without Isaac Sim

### Activating the Environment

**Every time** you start a new PowerShell session:

```powershell
# ALWAYS do this first!
cd I:\isaaclab
.\isaaclab.bat

# Then navigate to your project
cd I:\wSpace\cinebotRL

# Now you can run training scripts
python scripts\reinforcement_learning\sb3\train.py ...
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
# === EVERY TIME YOU START ===
# 1. Activate Isaac Lab environment
cd I:\isaaclab
.\isaaclab.bat

# 2. Navigate to project
cd I:\wSpace\cinebotRL

# 3. Quick test
python scripts\test_mobile_mm_env.py --num_envs 1 --steps 2 --headless

# 4. Start training
python scripts\reinforcement_learning\sb3\train.py `
    --task MobileMMTrackEE-v0 `
    --num_envs 1024 `
    --headless

# 5. Monitor (in separate PowerShell)
cd I:\isaaclab
.\isaaclab.bat
cd I:\wSpace\cinebotRL
tensorboard --logdir logs\sb3
```

---

## Environment Setup Summary

**Recommended setup** (Option A):
1. ✅ Use Isaac Lab's built-in environment (`.\isaaclab.bat`)
2. ✅ Install your project: `pip install -e .`
3. ✅ Install SB3: `pip install stable-baselines3[extra]`
4. ✅ Done! Ready to train.

**No separate venv needed** - Isaac Lab's environment has everything! 🚀

**Happy training!** 🚀
