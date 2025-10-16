# WSL-Side RL Platform Workflow Guide

## Overview

This document explains the **actual** working setup for the WSL side of the CinebotRL project. Due to Python version constraints, we maintain two separate environments that work together:

- **System Python 3.10 + ROS 2 Humble**: For ROS 2 communication with Windows
- **Venv Python 3.11 + PyTorch + RL libs**: For RL training experiments and utilities

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Windows Side                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Isaac Sim 5.0.0-rc.45 (Python 3.11)                    │ │
│  │ Isaac Lab (RL training loop)                           │ │
│  │ ROS 2 Humble (Python 3.10)                             │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  │ Fast DDS (Domain ID: 55)
                  │ UDP ports: 7400-7410, 7420, 8800
                  │
┌─────────────────┴───────────────────────────────────────────┐
│                    WSL2 (Ubuntu 22.04)                       │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ System Python 3.10 + ROS 2 Humble                      │ │
│  │ - ros2 CLI tools                                       │ │
│  │ - demo nodes (talker/listener)                         │ │
│  │ - Custom ROS 2 publishers/subscribers                  │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ .venv_rl311 (Python 3.11)                              │ │
│  │ - PyTorch 2.6.0+cu124                                  │ │
│  │ - stable-baselines3 2.7.0                              │ │
│  │ - gymnasium 1.2.1                                      │ │
│  │ - pandas, numpy, matplotlib                            │ │
│  │ - tensorboard, wandb                                   │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## Use Cases & Environment Selection

### When to Use System Python 3.10 (ROS 2)

✅ **Use system Python for:**
- Running ROS 2 nodes (publishers, subscribers, services)
- Testing ROS 2 communication with Windows
- Running `ros2` CLI commands
- Any code that imports `rclpy`, `rclcpp`, or ROS 2 message types

```bash
# Don't activate venv, just source ROS 2
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=55
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_DEFAULT_PROFILES_FILE=$HOME/fastdds_windows.xml

# Run ROS 2 commands
ros2 topic list
ros2 run demo_nodes_cpp talker
python3 my_ros_node.py  # Uses system Python 3.10
```

### When to Use .venv_rl311 (Python 3.11)

✅ **Use venv for:**
- Data preprocessing and analysis
- Training standalone RL agents (not in Isaac Lab)
- Running TensorBoard
- Jupyter notebooks
- Asset inspection and validation
- Post-processing training logs
- Creating visualizations

```bash
# Activate venv
source /mnt/c/Users/yanbo/wSpace/cinebotRL/.venv_rl311/bin/activate

# Set CUDA paths
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/cuda-12.6/bin:${PATH}"

# Run RL utilities
python scripts/process_logs.py
tensorboard --logdir /mnt/i/isaaclab/logs
jupyter notebook
```

### Hybrid Workflow (Both Environments)

Some workflows need both environments **in sequence**:

1. **RL Training on Windows** → generates logs
2. **Data processing in WSL venv** → analyzes logs
3. **ROS 2 publishing (system Python)** → sends results back to Windows

Example:
```bash
# Step 1: Process training logs (use venv)
source /mnt/c/Users/yanbo/wSpace/cinebotRL/.venv_rl311/bin/activate
python experiments/analyze_episode_rewards.py --logdir /mnt/i/isaaclab/logs/latest
deactivate

# Step 2: Publish results via ROS 2 (use system Python)
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=55
python3 experiments/publish_metrics_to_ros.py --topic /training/metrics
```

---

## Quick Start Scripts

### Option 1: ROS 2 Communication Only

```bash
# Use the provided helper (sets up Fast DDS + ROS 2)
cd /mnt/c/Users/yanbo/wSpace/cinebotRL
source scripts/wsl/setup_ros2_only.sh

# Test communication
ros2 run demo_nodes_cpp talker
```

### Option 2: RL Development Environment Only

```bash
# Use the RL venv activation helper
cd /mnt/c/Users/yanbo/wSpace/cinebotRL
source scripts/wsl/activate_rl_env_wsl.sh

# Check PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Option 3: Full Environment (Both)

```bash
# This is rare but possible - sets up both in correct order
cd /mnt/c/Users/yanbo/wSpace/cinebotRL
source scripts/wsl/setup_wsl_environment.sh
```

**Note:** When both are sourced, the venv Python 3.11 will be active, but ROS 2 commands will still work (they use `/opt/ros/humble/bin/python3` directly).

---

## Status Check

To verify your WSL setup:

```bash
cd /mnt/c/Users/yanbo/wSpace/cinebotRL
bash scripts/wsl/check_wsl_setup.sh
```

This script checks:
- ✓ System info (Ubuntu 22.04)
- ✓ GPU accessibility (nvidia-smi)
- ✓ CUDA toolkit (nvcc)
- ✓ Python venv (.venv_rl311)
- ✓ PyTorch CUDA availability
- ✓ ROS 2 Humble installation
- ✓ Fast DDS configuration
- ✓ Network connectivity to Windows

---

## Common Workflows

### Workflow 1: Monitor Windows Training from WSL

**Goal:** Watch ROS 2 topics published by Isaac Sim while training runs on Windows

```bash
# Terminal 1: Setup ROS 2 environment
cd /mnt/c/Users/yanbo/wSpace/cinebotRL
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=55
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_DEFAULT_PROFILES_FILE=$HOME/fastdds_windows.xml

# Monitor all topics
ros2 topic list

# Echo specific topic (e.g., robot joint states)
ros2 topic echo /robot/joint_states

# Record data to bag file
ros2 bag record -a -o ./experiments/data/run_$(date +%Y%m%d_%H%M%S)
```

### Workflow 2: Process Training Logs

**Goal:** Analyze training results from Isaac Lab

```bash
# Activate RL environment
cd /mnt/c/Users/yanbo/wSpace/cinebotRL
source .venv_rl311/bin/activate

# Navigate to logs (accessible via /mnt/i)
cd /mnt/i/isaaclab/logs/sb3/mobile_mm_track_ee/

# Run analysis
python /mnt/c/Users/yanbo/wSpace/cinebotRL/experiments/plot_training_curves.py \
  --logdir . \
  --output /mnt/c/Users/yanbo/wSpace/cinebotRL/experiments/results/
```

### Workflow 3: Test Communication Before Training

**Goal:** Verify WSL ↔ Windows ROS 2 bridge is working

```bash
# WSL Terminal
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=55
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_DEFAULT_PROFILES_FILE=$HOME/fastdds_windows.xml
ros2 run demo_nodes_cpp talker
```

```powershell
# Windows PowerShell (run simultaneously)
cd C:\Users\yanbo\wSpace\cinebotRL
.\scripts\networking\setup_ros2_humble_windows.ps1
ros2 run demo_nodes_cpp listener
```

Expected output on Windows:
```
[INFO] [listener]: I heard: [Hello World: 1]
[INFO] [listener]: I heard: [Hello World: 2]
...
```

### Workflow 4: Validate Robot Assets

**Goal:** Check URDF/USD files before training

```bash
# Activate RL venv (has asset inspector tools)
source /mnt/c/Users/yanbo/wSpace/cinebotRL/.venv_rl311/bin/activate

# Run asset inspector
cd /mnt/c/Users/yanbo/wSpace/cinebotRL/src/asset_inspector
python -m asset_inspector validate \
  --urdf-path ../../assets_own/mobile_manipulator_PPR_base_corrected.urdf \
  --output-dir ../../assets/processed/mobile_arm_whole_body/

# Check the generated report
cat ../../assets/processed/mobile_arm_whole_body/inspection_report.json | jq .
```

---

## Environment Variables Reference

### ROS 2 (System Python 3.10)

```bash
export ROS_DOMAIN_ID=55                                    # Must match Windows
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp                 # Use Fast DDS
export FASTDDS_DEFAULT_PROFILES_FILE=$HOME/fastdds_windows.xml  # Network config
```

### CUDA (for venv Python 3.11)

```bash
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/cuda-12.6/bin:${PATH}"
```

### Optional: For accessing Windows Isaac Lab from WSL

```bash
export ISAACLAB_WIN_ROOT="/mnt/i/isaaclab"
export ISAACSIM_WIN_ROOT="/mnt/i/isaacsim"
```

---

## Troubleshooting

### Issue: `ros2: command not found`

**Solution:**
```bash
source /opt/ros/humble/setup.bash
```

Add to `~/.bashrc` for persistence:
```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

### Issue: `ModuleNotFoundError: No module named 'rclpy'` in venv

**Expected behavior!** This is correct - `rclpy` only works with system Python 3.10.

**Solution:** Deactivate venv and use system Python for ROS 2:
```bash
deactivate  # Exit venv
python3 --version  # Should show 3.10.x
python3 my_ros_script.py
```

### Issue: PyTorch CUDA not available

**Solution:**
```bash
# Ensure CUDA paths are exported
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH:-}"

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

### Issue: Cannot see Windows ROS 2 topics

**Checklist:**
1. ✓ Both sides use `ROS_DOMAIN_ID=55`
2. ✓ Both sides use `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`
3. ✓ Fast DDS profile configured: `export FASTDDS_DEFAULT_PROFILES_FILE=$HOME/fastdds_windows.xml`
4. ✓ Windows firewall allows UDP 7400-7410, 7420, 8800
5. ✓ Clash/Mihomo proxy bypasses ROS traffic
6. ✓ Can ping Windows from WSL: `ping $(grep nameserver /etc/resolv.conf | awk '{print $2}')`

---

## File Organization

```
/mnt/c/Users/yanbo/wSpace/cinebotRL/
├── scripts/
│   ├── wsl/
│   │   ├── check_wsl_setup.sh              # Run this to verify setup
│   │   ├── setup_wsl_environment.sh        # Full environment (venv + ROS 2)
│   │   ├── activate_rl_env_wsl.sh          # RL venv only
│   │   └── setup_ros2_only.sh              # ROS 2 only (to be created)
│   └── networking/
│       └── configure_fastdds_wsl.sh        # Generate Fast DDS profile
├── src/
│   ├── asset_inspector/                    # Use with venv
│   └── rl_platform/                        # Use with venv
├── experiments/                            # Analysis scripts (use venv)
└── .venv_rl311/                           # Python 3.11 RL environment
```

---

## Next Steps

1. **Test Current Setup:**
   ```bash
   bash scripts/wsl/check_wsl_setup.sh
   ```

2. **Test ROS 2 Communication:**
   ```bash
   source /opt/ros/humble/setup.bash
   export ROS_DOMAIN_ID=55
   ros2 run demo_nodes_cpp talker
   ```

3. **Start Isaac Lab Training on Windows** (see `docs/windows_side_requirements.md`)

4. **Monitor from WSL:**
   ```bash
   source /opt/ros/humble/setup.bash
   export ROS_DOMAIN_ID=55
   ros2 topic list
   ```

---

**Last Updated:** 2025-10-13  
**Status:** WSL environment configured and documented ✓
