# Cinebot RL Environment

## Overview

This project implements reinforcement learning for a mobile manipulator robot using Isaac Sim and Isaac Lab on Windows, with WSL2 providing ROS 2 integration, data processing, and monitoring capabilities.

**Status (2025-10-13):** ✅ Environment fully configured, ROS 2 communication verified

## Architecture

```
Windows (Training)          WSL2 (Support & Monitoring)
├── Isaac Sim 5.0.0-rc      ├── System Python 3.10
├── Isaac Lab               │   └── ROS 2 Humble (communication)
├── ROS 2 Humble            └── .venv_rl311 (Python 3.11)
└── RTX 3090 (Training)         └── PyTorch, SB3, Gymnasium
         │                              (analysis & dev)
         └──── Fast DDS Bridge ────┘
               (Domain ID: 55)
```

## Quick Start

### 1. Verify WSL Setup

```bash
cd /mnt/c/Users/yanbo/wSpace/cinebotRL
bash scripts/wsl/check_wsl_setup.sh
```

### 2. Choose Your Workflow

**For ROS 2 Communication:**
```bash
source scripts/wsl/setup_ros2_only.sh
ros2 run demo_nodes_cpp talker
```

**For RL Development (analysis, utilities):**
```bash
source scripts/wsl/activate_rl_env_wsl.sh
python experiments/analyze_logs.py
```

**For Training (Windows):**
```powershell
I:\isaaclab\isaaclab-3090.bat -p scripts/reinforcement_learning/sb3/train.py --task YourTask-v0 --headless
```

## Key Scripts

### WSL
- `scripts/wsl/check_wsl_setup.sh` - Comprehensive environment verification
- `scripts/wsl/setup_ros2_only.sh` - ROS 2 environment setup
- `scripts/wsl/activate_rl_env_wsl.sh` - RL development environment
- `scripts/networking/configure_fastdds_wsl.sh` - Fast DDS network configuration

### Windows
- `scripts\networking\setup_ros2_humble_windows.ps1` - ROS 2 setup
- `scripts\networking\configure_fastdds_firewall.ps1` - Firewall configuration
- `I:\isaaclab\isaaclab-3090.bat` - Launch Isaac Lab

## Documentation

📖 **[Complete Documentation Index](docs/README.md)** - Start here for organized documentation

### Quick Links

- ⚡ **[Quick Reference Card](docs/QUICK_REFERENCE.md)** - One-page cheat sheet (print this!)
- 🔧 **[WSL Setup Guide](docs/setup/wsl_setup_guide.md)** - Configure WSL environment  
- 🪟 **[Windows Setup Guide](docs/setup/windows_setup_guide.md)** - Configure Windows side
- 🏗️ **[Architecture Overview](docs/architecture/overview.md)** - How everything fits together
- 🌉 **[ROS 2 Communication](docs/architecture/ros2_communication.md)** - How WSL ↔ Windows works
- 🐍 **[Python Environments](docs/architecture/python_environments.md)** - Why multiple Python versions
- ⚡ **[Daily Workflow](docs/workflows/daily_workflow.md)** - Common tasks and commands
- 🗺️ **[Roadmap](ROADMAP.md)** - Project phases and implementation plan

## Environment Details

### Windows Side
- **Isaac Sim:** `I:\isaacsim` (5.0.0-rc.45, Python 3.11)
- **Isaac Lab:** `I:\isaaclab` (editable install with SB3, RL-Games)
- **ROS 2 Humble:** `I:\ros2humble\ros2-windows` (Python 3.10)
- **Training GPU:** RTX 3090 (CUDA device 0)

### WSL Side
- **OS:** Ubuntu 22.04 (WSL2)
- **ROS 2 Humble:** System Python 3.10 (`/opt/ros/humble`)
- **RL Environment:** `.venv_rl311` Python 3.11 (PyTorch 2.6.0+cu124, SB3 2.7.0)
- **CUDA:** 12.6.85 (GPU passthrough enabled)

## Common Commands

### Test ROS 2 Communication

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

### Monitor Training

```bash
# Start TensorBoard
source scripts/wsl/activate_rl_env_wsl.sh
tensorboard --logdir /mnt/i/isaaclab/logs

# Monitor ROS topics
source scripts/wsl/setup_ros2_only.sh
ros2 topic list
```

## Python Version Guidelines

- **System Python 3.10:** Use for ROS 2 nodes, `ros2` CLI commands
- **Venv Python 3.11:** Use for RL development, PyTorch, data analysis
- **Windows Python 3.11:** Isaac Lab training (bundled with Isaac Sim)
- **Windows Python 3.10:** ROS 2 on Windows (separate installation)

## Troubleshooting

**ROS 2 topics not visible?**
- Check `ROS_DOMAIN_ID=55` on both sides
- Verify firewall: `netsh advfirewall firewall show rule name=all | findstr 7410`
- Reconfigure Fast DDS: `bash scripts/networking/configure_fastdds_wsl.sh`

**PyTorch CUDA not available?**
- Activate venv: `source scripts/wsl/activate_rl_env_wsl.sh`
- Check paths: `echo $LD_LIBRARY_PATH`

**`rclpy` import error in venv?**
- Expected! Deactivate venv and use system Python for ROS 2

See [WSL Workflow Guide](docs/wsl_workflow_guide.md) for detailed troubleshooting.

## Next Steps

1. ✅ Environment setup complete
2. ✅ ROS 2 communication tested
3. ⏭️ Implement custom RL task (`MobileMMTrackEE-v0`)
4. ⏭️ Convert robot URDF to USD format
5. ⏭️ Run baseline training experiments

---

**Last Updated:** 2025-10-13

