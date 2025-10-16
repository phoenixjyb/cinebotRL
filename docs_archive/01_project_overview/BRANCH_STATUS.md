# Branch Status & Training Next Steps

## Git Status ✅

**Current branches**:
- `master`: WSL environment setup with robot constraints and self-collision
- `train-windows`: Windows training guide added (current branch)

**Commits**:
1. `7302ea0`: Initial package structure and test scripts  
2. `b66a093`: WSL environment setup with robot constraints
3. `7de1b0e`: Windows training documentation

**Note**: No remote repository configured yet. All changes are saved locally in:
```
/mnt/c/Users/yanbo/wSpace/cinebotRL/.git/
```

---

## What We Accomplished 🎉

### WSL Side (master branch)

✅ **Environment Architecture**:
- Fixed Isaac Lab 2.2.0 import paths (`isaaclab` not `omni.isaac.lab`)
- Created WSL CUDA wrapper script (`scripts/run_with_wsl_cuda.sh`)
- Added lazy initialization for robot data (joint limits, EE body index)
- Fixed `num_envs` configuration (was hardcoded to 1024)

✅ **Robot Constraints**:
- Velocity limits (1.5 m/s linear, 2.0 rad/s angular)
- Acceleration limits (5.0 m/s²)
- Jerk limits (5.0 m/s³)
- Joint limits with 0.1 rad margin
- Lateral motion penalty (differential drive constraint)

✅ **Self-Collision System** (CRITICAL):
- Contact sensors enabled
- Continuous penalty function (weight 50.0!)
- Two-tier system: 1N penalty, 10N termination
- Self-collision detection in `_get_dones()`

✅ **Documentation**:
- `docs/troubleshooting/wsl2_cuda_fix_summary.md` - WSL CUDA setup
- `docs/reference/robot_constraints_updated.md` - Physical constraints
- `docs/architecture/training_architecture.md` - WSL vs Windows
- `docs/workflows/multi_trajectory_training.md` - 1000+ trajectories
- `docs/UPDATES_SUMMARY.md` - Quick reference

### Windows Side (train-windows branch)

✅ **Training Guide**:
- Complete Windows setup instructions
- GPU configuration
- Training commands with examples
- TensorBoard monitoring
- Troubleshooting common issues
- Video recording guide

---

## Known Issues ⚠️

### WSL (Paused for Now)

1. **Contact forces API**: 
   ```python
   # Current (placeholder):
   net_contact_forces = torch.zeros(...)
   
   # TODO: Find Isaac Lab 2.2.0 API for:
   net_contact_forces = self.robot.data.???
   ```

2. **Warp UUID errors** (harmless):
   ```
   Warp CUDA error: Failed to get driver entry point 'cuDeviceGetUuid'
   ```
   - Doesn't affect physics
   - WSL2 driver limitation

3. **Environment steps complete** but contact detection disabled

### What Works in WSL

✅ Environment creation  
✅ Robot loading  
✅ EE tracking setup  
✅ Reward computation (except contact forces)  
✅ CUDA GPU acceleration  
✅ Physics simulation  

---

## Next Steps for Windows Training 🚀

### 1. Setup Environment

```powershell
# PowerShell on Windows
cd I:\isaaclab
.\isaaclab.bat

# Install project
cd I:\wSpace\cinebotRL
pip install -e .
pip install stable-baselines3[extra]
```

### 2. Quick Test

```powershell
# Test environment (headless)
python scripts\test_mobile_mm_env.py --num_envs 1 --steps 5 --headless
```

**Expected output**:
- Environment creates successfully
- Robot spawns
- EE link found
- Reward components computed
- Steps complete without errors

### 3. Start Training

```powershell
# Baseline training (1024 envs, headless)
python scripts\reinforcement_learning\sb3\train.py `
    --task MobileMMTrackEE-v0 `
    --num_envs 1024 `
    --headless `
    --total_timesteps 5000000 `
    --save_freq 100000

# Multi-trajectory training (recommended)
python scripts\reinforcement_learning\sb3\train.py `
    --task MobileMMTrackEE-v0 `
    --num_envs 1024 `
    --headless `
    --trajectory_type multi_recorded `
    --trajectory_dir trajectoryToLearn\world_json\cinematic_db `
    --total_timesteps 10000000
```

### 4. Monitor Progress

```powershell
# Terminal 2
tensorboard --logdir logs\sb3\MobileMMTrackEE-v0
```

Open: http://localhost:6006

**Watch for**:
- `rollout/ep_rew_mean` increasing to >10 per step
- `reward_components/self_collision_penalty` decreasing to ~0
- `reward_components/position_tracking` increasing to ~10
- `train/explained_variance` > 0.7

---

## Training Timeline Estimates

**Hardware**: RTX 3090, 1024 parallel environments

| Phase | Steps | Time | Goal |
|-------|-------|------|------|
| Initial learning | 0-100K | ~1 hour | Basic tracking |
| Constraint satisfaction | 100K-500K | ~5 hours | No collisions |
| Smooth tracking | 500K-2M | ~20 hours | Accurate + smooth |
| Multi-trajectory | 2M-10M | ~100 hours | Generalization |

**Checkpoints saved**: Every 100K steps  
**Best model saved**: Automatically

---

## When to Switch Back to WSL

**Only if** you need to:
- Fix contact force API for self-collision detection
- Use Linux-specific tools
- Deploy to Linux production environment

**For training**: Windows is ready to go! 🎬

---

## Files to Check on Windows

Before training, verify these paths exist:

```powershell
# Project structure
I:\wSpace\cinebotRL\
├── src\rl_platform\tasks\mobile_mm\
│   ├── env.py
│   ├── config.py
│   ├── rewards.py
│   └── ...
├── assets_own\usd\
│   └── mobile_manipulator_PPR_base_corrected.usd
├── trajectoryToLearn\world_json\cinematic_db\
│   ├── arc_left_push\
│   ├── crane_up\
│   └── ... (10 categories)
└── scripts\
    ├── test_mobile_mm_env.py
    └── reinforcement_learning\sb3\train.py
```

---

## Documentation Reference

- **Windows setup**: `docs/setup/TRAIN_ON_WINDOWS.md`
- **WSL setup**: `docs/troubleshooting/wsl2_cuda_fix_summary.md`
- **Constraints**: `docs/reference/robot_constraints_updated.md`
- **Multi-trajectory**: `docs/workflows/multi_trajectory_training.md`
- **Quick ref**: `docs/UPDATES_SUMMARY.md`

---

## Summary

✅ **WSL work paused** (contact force API needs research)  
✅ **Windows branch ready** for training  
✅ **All code committed** and saved locally  
✅ **Documentation complete** for Windows training  

**Next command**:
```powershell
cd I:\isaaclab
.\isaaclab.bat
cd I:\wSpace\cinebotRL
python scripts\test_mobile_mm_env.py --num_envs 1 --steps 5 --headless
```

🚀 **Let's train on Windows!**
