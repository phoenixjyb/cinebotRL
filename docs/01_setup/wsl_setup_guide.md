# WSL Environment Setup Guide

**Goal:** Verify and configure your WSL2 environment to work with Windows Isaac Sim/Lab.

---

## Prerequisites

- WSL2 with Ubuntu 22.04
- NVIDIA GPU passthrough enabled
- Windows with Isaac Sim and Isaac Lab installed

---

## Quick Setup (5 Minutes)

### 1. Verify Your Environment

Run the comprehensive status check:

```bash
cd /mnt/c/Users/yanbo/wSpace/cinebotRL
bash scripts/wsl/check_wsl_setup.sh
```

This checks:
- ✓ Ubuntu 22.04 installation
- ✓ GPU access (nvidia-smi)
- ✓ CUDA toolkit (12.6)
- ✓ Python virtual environment (.venv_rl311)
- ✓ PyTorch CUDA availability
- ✓ ROS 2 Humble installation
- ✓ Fast DDS network configuration
- ✓ Windows connectivity

**Expected Result:** All checks pass with green ✓ marks.

### 2. Choose Your Environment

Based on what you need to do:

#### For ROS 2 Communication (Most Common)

```bash
cd /mnt/c/Users/yanbo/wSpace/cinebotRL
source scripts/wsl/setup_ros2_only.sh
```

**Use this when:**
- Testing ROS 2 topics with Windows
- Running ROS 2 nodes
- Monitoring Isaac Sim

#### For RL Development & Analysis

```bash
cd /mnt/c/Users/yanbo/wSpace/cinebotRL
source scripts/wsl/activate_rl_env_wsl.sh
```

**Use this when:**
- Processing training logs
- Running data analysis
- Using PyTorch/Gymnasium
- Running TensorBoard

---

## Detailed Setup

### System Python 3.10 (ROS 2)

**What it's for:** ROS 2 communication with Windows

**How to activate:**
```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=55
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_DEFAULT_PROFILES_FILE=$HOME/fastdds_windows.xml
```

**Or use the helper script:**
```bash
source scripts/wsl/setup_ros2_only.sh
```

**Test it works:**
```bash
ros2 topic list
python3 --version  # Should show 3.10.x
```

### Python 3.11 Virtual Environment (RL Development)

**What it's for:** RL experiments, data processing, PyTorch

**Location:** `.venv_rl311`

**How to activate:**
```bash
source scripts/wsl/activate_rl_env_wsl.sh
```

**What's included:**
- PyTorch 2.6.0+cu124 (CUDA enabled)
- stable-baselines3 2.7.0
- gymnasium 1.2.1
- pandas, numpy, matplotlib
- tensorboard, wandb

**Test it works:**
```bash
python --version  # Should show 3.11.x
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"  # Should show True
```

---

## Network Configuration

### Fast DDS Setup

This enables WSL ↔ Windows ROS 2 communication.

**One-time setup:**
```bash
cd /mnt/c/Users/yanbo/wSpace/cinebotRL
bash scripts/networking/configure_fastdds_wsl.sh
```

This creates `~/fastdds_windows.xml` with your Windows host IP.

**Every session:**
```bash
export ROS_DOMAIN_ID=55
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_DEFAULT_PROFILES_FILE=$HOME/fastdds_windows.xml
```

Or just use:
```bash
source scripts/wsl/setup_ros2_only.sh
```

---

## Verification Tests

### Test 1: GPU Access

```bash
nvidia-smi
```

**Expected:** Should show RTX 3090 and Quadro P2000

### Test 2: CUDA

```bash
nvcc --version
```

**Expected:** Should show CUDA 12.6.85

### Test 3: PyTorch CUDA

```bash
source scripts/wsl/activate_rl_env_wsl.sh
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
```

**Expected:** `CUDA: True, Devices: 2`

### Test 4: ROS 2 Installation

```bash
source /opt/ros/humble/setup.bash
ros2 --version
```

**Expected:** Should show ROS 2 Humble

### Test 5: ROS 2 Communication with Windows

**WSL Terminal:**
```bash
source scripts/wsl/setup_ros2_only.sh
ros2 run demo_nodes_cpp talker
```

**Windows Terminal (simultaneously):**
```powershell
cd C:\Users\yanbo\wSpace\cinebotRL
.\scripts\networking\setup_ros2_humble_windows.ps1
ros2 run demo_nodes_cpp listener
```

**Expected:** Windows listener shows "I heard: [Hello World: N]"

---

## Common Issues

### Issue: `nvidia-smi` not found

**Cause:** GPU passthrough not enabled or NVIDIA drivers not installed

**Fix:**
1. Update Windows to latest version
2. Update NVIDIA drivers on Windows
3. Restart WSL: `wsl --shutdown` in Windows PowerShell

### Issue: `nvcc: command not found`

**Cause:** CUDA toolkit not installed

**Fix:**
```bash
# Install CUDA toolkit 12.6
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
```

### Issue: PyTorch CUDA not available

**Cause:** CUDA paths not exported or PyTorch not installed with CUDA support

**Fix:**
```bash
source scripts/wsl/activate_rl_env_wsl.sh
# Check paths
echo $LD_LIBRARY_PATH
# Should include /usr/lib/wsl/lib and /usr/local/cuda-12.6/lib64
```

### Issue: ROS 2 topics not visible from Windows

**Cause:** Network configuration issue

**Fix:**
1. Verify domain ID matches: `echo $ROS_DOMAIN_ID` (should be 55)
2. Reconfigure Fast DDS: `bash scripts/networking/configure_fastdds_wsl.sh`
3. Check Windows firewall allows UDP 7400-7410
4. Verify can ping Windows: `ping $(grep nameserver /etc/resolv.conf | awk '{print $2}')`

### Issue: `ModuleNotFoundError: rclpy` in venv

**This is expected!** ROS 2's `rclpy` only works with system Python 3.10, not the venv Python 3.11.

**Fix:** Deactivate venv for ROS 2 work:
```bash
deactivate
source scripts/wsl/setup_ros2_only.sh
```

---

## Environment Variables Reference

### ROS 2 Mode (System Python 3.10)

```bash
ROS_DOMAIN_ID=55                           # Match Windows
RMW_IMPLEMENTATION=rmw_fastrtps_cpp        # Use Fast DDS
FASTDDS_DEFAULT_PROFILES_FILE=$HOME/fastdds_windows.xml
```

### RL Mode (Venv Python 3.11)

```bash
LD_LIBRARY_PATH="/usr/lib/wsl/lib:/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH"
PATH="/usr/local/cuda-12.6/bin:$PATH"
```

---

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `scripts/wsl/check_wsl_setup.sh` | Comprehensive environment check |
| `scripts/wsl/setup_ros2_only.sh` | Setup ROS 2 environment |
| `scripts/wsl/activate_rl_env_wsl.sh` | Setup RL venv |
| `scripts/wsl/setup_wsl_environment.sh` | Setup both (rarely needed) |
| `scripts/networking/configure_fastdds_wsl.sh` | Configure Fast DDS |

---

## Next Steps

After verifying your WSL setup:

1. **Test Communication:** [ROS 2 Communication Setup](ros2_communication_setup.md)
2. **Understand Architecture:** [Architecture Overview](../architecture/overview.md)
3. **Daily Usage:** [Daily Workflow Guide](../workflows/daily_workflow.md)

---

## Quick Reference

**Most common commands:**

```bash
# Check setup
bash scripts/wsl/check_wsl_setup.sh

# ROS 2 work
source scripts/wsl/setup_ros2_only.sh
ros2 topic list

# RL development
source scripts/wsl/activate_rl_env_wsl.sh
python my_script.py

# View logs from Windows
tensorboard --logdir /mnt/i/isaaclab/logs
```

---

**Last Updated:** 2025-10-13  
**See Also:** 
- [Windows Setup Guide](windows_setup_guide.md)
- [Quick Reference Card](../QUICK_REFERENCE.md)
- [Troubleshooting](../reference/troubleshooting.md)
