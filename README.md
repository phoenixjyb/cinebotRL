# Cinebot RL Environment

## Overview

This project implements reinforcement learning for a mobile manipulator robot using Isaac Sim and Isaac Lab on **Windows native**. WSL2 provides optional ROS 2 integration, data analysis, and monitoring capabilities.

**Status (2025-10-15):** ✅ **Windows Training Fully Operational** - All compatibility issues resolved, Stable Baselines3 PPO training verified working

## Architecture

```
Windows (Primary Training Platform) ✅
├── Isaac Sim 5.0.0-rc.45
├── Isaac Lab 2.2.0 (Python 3.11.13, torch 2.7.0+cu128)
├── Stable Baselines3 PPO Training
├── Custom IsaacLabToSB3VecEnvWrapper (Isaac Lab ↔ SB3 bridge)
├── ROS 2 Humble (optional, for topic bridging)
└── RTX 3090 (auto-detected for training)

WSL2 (Optional Support - Not Required)
├── System Python 3.10
│   └── ROS 2 Humble (communication & monitoring)
└── .venv_rl311 (Python 3.11)
    └── PyTorch, SB3 (data analysis only)
```

## Quick Start

### Windows Training (Recommended)

**Start Training in 3 Commands:**
```powershell
# 1. Navigate to project
cd C:\Users\yanbo\wSpace\cinebotRL

# 2. Launch training (default: 64 envs, headless, 5M steps)
.\scripts\launch_training_windows.ps1 -Task MobileMMTrackEE-v0 -NumEnvs 64 -Headless

# 3. Monitor in separate window
.\scripts\monitor_training.ps1 -Mode all
```

**Alternative: Direct Isaac Lab Launcher**
```powershell
I:\isaaclab\isaaclab.bat -p scripts/reinforcement_learning/sb3/train.py `
  --task MobileMMTrackEE-v0 --num_envs 64 --headless true
```

**📚 Documentation**: 
- **Quick Reference**: [docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)
- **Directory Structure**: [DIRECTORY_STRUCTURE.md](DIRECTORY_STRUCTURE.md) ⭐ NEW
- **Documentation Index**: [docs/README.md](docs/README.md)

### Optional: WSL Setup (For Monitoring/Analysis Only)

**WSL is NOT required for training.** Use only if you need ROS 2 monitoring or data analysis tools.

```bash
cd /mnt/c/Users/yanbo/wSpace/cinebotRL
bash scripts/wsl/check_wsl_setup.sh
```

**For ROS 2 Monitoring:**
```bash
source scripts/wsl/setup_ros2_only.sh
ros2 topic list
```

**For Data Analysis:**
```bash
source scripts/wsl/activate_rl_env_wsl.sh
python experiments/analyze_logs.py
```

## Key Scripts

### Windows (Primary)
- `scripts\launch_training_windows.ps1` - ✅ **Primary training launcher** (auto GPU detection, parameters)
- `scripts\monitor_training.ps1` - Monitor training progress (logs/gpu/tensorboard)
- `scripts\commit_and_start_training.ps1` - Commit changes and start training in one command
- `scripts\networking\setup_ros2_humble_windows.ps1` - ROS 2 setup (optional)
- `scripts\networking\configure_fastdds_firewall.ps1` - Firewall configuration (optional)

### WSL (Optional - Not Required for Training)
- `scripts/wsl/check_wsl_setup.sh` - Environment verification
- `scripts/wsl/setup_ros2_only.sh` - ROS 2 environment setup  
- `scripts/wsl/activate_rl_env_wsl.sh` - RL analysis environment
- `scripts/networking/configure_fastdds_wsl.sh` - Fast DDS network configuration

## Documentation

📖 **[Complete Documentation Index](docs/README.md)** - Start here for organized documentation

### Essential Guides

- ⚡ **[START_TRAINING_NOW.md](START_TRAINING_NOW.md)** - ✅ **Quick start guide** - Get training running in 3 commands!
- 📊 **[TRAINING_SUCCESS.md](TRAINING_SUCCESS.md)** - Technical details on all 12+ compatibility fixes
- ⚡ **[Quick Reference Card](docs/QUICK_REFERENCE.md)** - One-page cheat sheet
- 🗺️ **[Roadmap](ROADMAP.md)** - Project phases and implementation plan (updated for Windows training)

### Setup & Architecture
- 🪟 **[Windows Setup Guide](docs/setup/windows_setup_guide.md)** - Configure Windows environment
- 🔧 **[WSL Setup Guide](docs/setup/wsl_setup_guide.md)** - Optional WSL configuration  
- 🏗️ **[Architecture Overview](docs/architecture/overview.md)** - How everything fits together
- 🌉 **[ROS 2 Communication](docs/architecture/ros2_communication.md)** - How WSL ↔ Windows works (optional)
- 🐍 **[Python Environments](docs/architecture/python_environments.md)** - Why multiple Python versions

### Workflows
- ⚡ **[Daily Workflow](docs/workflows/daily_workflow.md)** - Common tasks and commands
- 🎯 **[Multi-Trajectory Training](docs/workflows/multi_trajectory_training.md)** - Advanced training workflows

## Environment Details

### Windows Side (Primary Training Platform)
- **Isaac Sim:** `I:\isaacsim` (5.0.0-rc.45, Python 3.11.13)
- **Isaac Lab:** `I:\isaaclab` (2.2.0, torch 2.7.0+cu128, editable install with SB3)
- **Training Framework:** Stable Baselines3 PPO with custom `IsaacLabToSB3VecEnvWrapper`
- **Training GPU:** RTX 3090 (CUDA device 0, auto-detected)
- **Display GPU:** Quadro P2000 (CUDA device 1)
- **ROS 2 Humble:** `I:\ros2humble\ros2-windows` (optional, Python 3.10)
- **Status:** ✅ All compatibility issues resolved, training verified working

### WSL Side (Optional - Not Required for Training)
- **OS:** Ubuntu 22.04 (WSL2)
- **ROS 2 Humble:** System Python 3.10 (`/opt/ros/humble`) - for monitoring only
- **RL Environment:** `.venv_rl311` Python 3.11 (PyTorch 2.6.0+cu124, SB3 2.7.0) - for analysis only
- **CUDA:** 12.6.85 (GPU passthrough available but not used)

## Common Commands

### Training & Monitoring (Windows)

```powershell
# Start training (basic)
.\scripts\launch_training_windows.ps1 -Task MobileMMTrackEE-v0 -NumEnvs 64 -Headless

# Start training with custom settings
.\scripts\launch_training_windows.ps1 -Task MobileMMTrackEE-v0 -NumEnvs 128 -TotalTimesteps 10000000

# Monitor training logs
.\scripts\monitor_training.ps1 -Mode logs

# Watch GPU usage
.\scripts\monitor_training.ps1 -Mode gpu

# Launch TensorBoard
.\scripts\monitor_training.ps1 -Mode tensorboard

# All-in-one: commit and train
.\scripts\commit_and_start_training.ps1 -Task MobileMMTrackEE-v0 -NumEnvs 64 -Headless
```

### Test ROS 2 Communication (Optional)

```bash
# WSL: Publish messages
source scripts/wsl/setup_ros2_only.sh
ros2 run demo_nodes_cpp talker
```

```powershell
# Windows: Receive messages
.\scripts\networking\setup_ros2_humble_windows.ps1
ros2 run demo_nodes_cpp listener
```

### Data Analysis (WSL - Optional)

```bash
# Start TensorBoard for analysis
source scripts/wsl/activate_rl_env_wsl.sh
tensorboard --logdir /mnt/i/isaaclab/logs

# Monitor ROS topics
source scripts/wsl/setup_ros2_only.sh
ros2 topic list
```

## Python Version Guidelines

- **Windows Python 3.11 (Isaac Lab):** ✅ **Primary for training** - bundled with Isaac Sim, use for all RL training
- **Windows Python 3.10 (ROS 2):** Optional - only if using ROS 2 on Windows for topic bridging
- **WSL Python 3.11 (venv):** Optional - only for data analysis, visualization, not training
- **WSL System Python 3.10:** Optional - only for ROS 2 monitoring and automation scripts

## Training Performance

**Current Task:** `MobileMMTrackEE-v0` (Mobile Manipulator End-Effector Tracking)
- **Robot:** 9 DOF (6 arm joints + 3 chassis: vx, vy, wz)
- **Action Space:** 8D (6 arm positions + vx + wz, differential drive excludes vy)
- **Observation Space:** 76D (dynamically discovered)
- **Algorithm:** Stable Baselines3 PPO with VecNormalize

**Expected Timeline:**
- 64 envs: ~60 minutes for 100K steps, ~8-10 hours for 5M steps  
- 128 envs: ~30 minutes for 100K steps, ~4-5 hours for 5M steps

**Architecture:** Custom `IsaacLabToSB3VecEnvWrapper` bridges Isaac Lab (dict observations, torch tensors, GPU) to Stable Baselines3 (numpy arrays, CPU). See [TRAINING_SUCCESS.md](TRAINING_SUCCESS.md) for technical details.

## Troubleshooting

**Training not starting?**
- Verify Isaac Lab installation: `I:\isaaclab\isaaclab.bat -h`
- Check GPU detection: `nvidia-smi`
- Review logs in latest directory under `I:\isaaclab\logs\sb3\`
- See [docs_archive/03_training_guides/TRAINING_SUCCESS.md](docs_archive/03_training_guides/TRAINING_SUCCESS.md) for known issues and solutions

**Import errors or compatibility issues?**
- All 12+ compatibility issues have been resolved as of 2025-10-15
- Gymnasium ale_py issue: Already patched in Isaac Lab Python environment
- See [docs_archive/02_bug_fixes/ALL_FIXES_COMPLETE.md](docs_archive/02_bug_fixes/ALL_FIXES_COMPLETE.md) for complete list of fixes

**ROS 2 topics not visible? (Optional)**
- Check `ROS_DOMAIN_ID=55` on both sides
- Verify firewall: `netsh advfirewall firewall show rule name=all | findstr 7410`
- Reconfigure Fast DDS: `bash scripts/networking/configure_fastdds_wsl.sh`

**PyTorch CUDA not available in WSL?**
- Not needed for training! Use Windows side for training
- For analysis only: Activate venv `source scripts/wsl/activate_rl_env_wsl.sh`

See [docs/reference/troubleshooting.md](docs/reference/troubleshooting.md) for detailed troubleshooting.

## Documentation

📚 **Complete documentation index**: [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)

**Quick links**:
- 🚀 [Quick Start Training](docs_archive/04_gpu_optimization/RTX3090_REFERENCE_CARD.md) - START HERE
- 🐛 [Bug Fixes Summary](docs_archive/02_bug_fixes/ALL_FIXES_COMPLETE.md) - All resolved ✅
- ⚡ [GPU Optimization](docs_archive/04_gpu_optimization/RTX3090_QUICK_START.md) - 6x speedup guide
- 📖 [Training Guides](docs_archive/03_training_guides/) - Workflows and tuning
- 🎯 [Project Overview](docs_archive/01_project_overview/) - Roadmap and status

## Next Steps

1. ✅ Environment setup complete
2. ✅ ROS 2 communication tested
3. ✅ **Windows training verified working** (2025-10-15)
4. ✅ `MobileMMTrackEE-v0` task implemented and training
5. ✅ Robot USD asset created and validated
6. ✅ **5 critical bugs fixed and verified**
7. ✅ **GPU optimization implemented** (6-8x speedup potential)
8. ⏭️ Scale to Phase 2 training (4096 envs)
9. ⏭️ Evaluate trained policy performance
10. ⏭️ Deploy to physical robot

---

**Last Updated:** 2025-10-16  
**Training Status:** ✅ Fully Operational + Optimized  
**GPU Utilization:** Scaling from 10% → 60-80% (6-8x faster)

