# WSL + Windows Integration Summary

## ✅ Current Status (2025-10-13)

### What's Working
- ✅ **WSL Environment**: Ubuntu 22.04 with CUDA 12.6, nvidia-smi shows both GPUs
- ✅ **ROS 2 Humble**: Installed and functional on both WSL and Windows
- ✅ **Fast DDS Communication**: WSL talker ↔ Windows listener verified
- ✅ **Python Environments**: 
  - System Python 3.10 for ROS 2
  - `.venv_rl311` Python 3.11 for RL/PyTorch work
- ✅ **PyTorch CUDA**: Available in venv with 2 GPUs detected
- ✅ **Isaac Lab on Windows**: Installed at `I:\isaaclab` with editable packages
- ✅ **Isaac Sim on Windows**: 5.0.0-rc.45 at `I:\isaacsim`

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Windows Host                                 │
│                                                                      │
│  ┌──────────────────────┐      ┌──────────────────────────────┐   │
│  │ Isaac Sim 5.0.0-rc   │      │ ROS 2 Humble (Python 3.8)    │   │
│  │ - Python 3.11        │      │ - Monitoring & CLI tools     │   │
│  │ - Physics engine     │      │ - ros2 topic echo, etc.      │   │
│  │ - Built-in ROS 2     │      └──────────────┬───────────────┘   │
│  └──────────┬───────────┘                     │                    │
│             │                                  │                    │
│             └──────────────┬───────────────────┘                    │
│                            │                                        │
│                   ┌────────▼────────┐                               │
│                   │   Fast DDS      │                               │
│                   │   Domain ID: 55 │                               │
│                   │   (UDP Network) │                               │
│                   └────────┬────────┘                               │
│                            │                                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Isaac Lab (I:\isaaclab)                                      │   │
│  │ - RL training loop (SB3, RL-Games)                           │   │
│  │ - RTX 3090 (CUDA device 0)                                   │   │
│  │ - Logs: I:\isaaclab\logs\                                    │   │
│  └──────────────────────────────────────────────────────────────┘   │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             │ Fast DDS Network
                             │ (Peer-to-peer, no central broker!)
                             │ UDP: 7400-7410, 7420, 8800
                             │ ROS_DOMAIN_ID=55
                             │
┌────────────────────────────┴─────────────────────────────────────────┐
│                       WSL2 (Ubuntu 22.04)                            │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ System Python 3.10 Environment                               │   │
│  │ ✓ ROS 2 Humble                                               │   │
│  │ ✓ Fast DDS configured                                        │   │
│  │ ✓ ros2 CLI tools                                             │   │
│  │ ✓ demo_nodes_cpp (talker/listener)                           │   │
│  │                                                               │   │
│  │ Use: source scripts/wsl/setup_ros2_only.sh                   │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ .venv_rl311 (Python 3.11)                                    │   │
│  │ ✓ PyTorch 2.6.0+cu124 (CUDA working)                         │   │
│  │ ✓ stable-baselines3 2.7.0                                    │   │
│  │ ✓ gymnasium 1.2.1                                            │   │
│  │ ✓ pandas, numpy, matplotlib                                  │   │
│  │ ✓ tensorboard, wandb                                         │   │
│  │                                                               │   │
│  │ Use: source scripts/wsl/activate_rl_env_wsl.sh               │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  GPU Access: RTX 3090 + Quadro P2000 (via nvidia-smi)              │
│  CUDA: 12.6.85                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start Guide

### 1. Verify WSL Setup

```bash
cd /mnt/c/Users/yanbo/wSpace/cinebotRL
bash scripts/wsl/check_wsl_setup.sh
```

### 2. Choose Your Workflow

#### For ROS 2 Communication (WSL → Windows)

```bash
# Setup ROS 2 environment
source scripts/wsl/setup_ros2_only.sh

# Test communication
ros2 run demo_nodes_cpp talker  # WSL publishes
```

```powershell
# Windows Terminal (run simultaneously)
cd C:\Users\yanbo\wSpace\cinebotRL
.\scripts\networking\setup_ros2_humble_windows.ps1
ros2 run demo_nodes_cpp listener  # Windows receives
```

#### For RL Development (Data Processing, Analysis)

```bash
# Activate RL environment
source scripts/wsl/activate_rl_env_wsl.sh

# Use PyTorch, SB3, etc.
python experiments/analyze_logs.py
tensorboard --logdir /mnt/i/isaaclab/logs
```

#### For Training (Windows Isaac Lab)

```powershell
# Windows PowerShell
cd C:\Users\yanbo\wSpace\cinebotRL
I:\isaaclab\isaaclab-3090.bat -p scripts/reinforcement_learning/sb3/train.py `
  --task MobileMMTrackEE-v0 `
  --num_envs 1024 `
  --headless
```

---

## Key Scripts Reference

### WSL Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/wsl/check_wsl_setup.sh` | Verify all WSL components | `bash scripts/wsl/check_wsl_setup.sh` |
| `scripts/wsl/setup_ros2_only.sh` | ROS 2 environment only | `source scripts/wsl/setup_ros2_only.sh` |
| `scripts/wsl/activate_rl_env_wsl.sh` | RL venv only | `source scripts/wsl/activate_rl_env_wsl.sh` |
| `scripts/wsl/setup_wsl_environment.sh` | Both ROS 2 + venv | `source scripts/wsl/setup_wsl_environment.sh` |
| `scripts/networking/configure_fastdds_wsl.sh` | Generate Fast DDS profile | `bash scripts/networking/configure_fastdds_wsl.sh` |

### Windows Scripts

| Script | Purpose |
|--------|---------|
| `scripts\networking\setup_ros2_humble_windows.ps1` | Setup ROS 2 environment |
| `scripts\networking\configure_fastdds_firewall.ps1` | Configure firewall rules |
| `I:\isaaclab\isaaclab-3090.bat` | Launch Isaac Lab |
| `I:\isaacsim\isaac-sim.bat` | Launch Isaac Sim |

---

## Environment Variables Cheat Sheet

### WSL - ROS 2 Environment

```bash
export ROS_DOMAIN_ID=55
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_DEFAULT_PROFILES_FILE=$HOME/fastdds_windows.xml
```

### WSL - RL Environment

```bash
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/cuda-12.6/bin:${PATH}"
```

### Windows - ROS 2

```powershell
$env:ROS_DOMAIN_ID = "55"
$env:RMW_IMPLEMENTATION = "rmw_fastrtps_cpp"
```

### Windows - Isaac Lab

```powershell
$env:CUDA_VISIBLE_DEVICES = "0"  # RTX 3090
$env:ISAACSIM_PATH = "I:\isaacsim"
$env:OV_ASSETS_ROOT = "I:\isaacsim_assets"
```

---

## Common Tasks

### Task 1: Test ROS 2 Communication

**Goal:** Verify WSL and Windows can exchange ROS 2 messages

```bash
# WSL Terminal
source scripts/wsl/setup_ros2_only.sh
ros2 run demo_nodes_cpp talker
```

```powershell
# Windows Terminal
.\scripts\networking\setup_ros2_humble_windows.ps1
ros2 run demo_nodes_cpp listener
```

**Expected:** Windows listener shows "I heard: [Hello World: N]"

### Task 2: Start Training on Windows

```powershell
cd C:\Users\yanbo\wSpace\cinebotRL
I:\isaaclab\isaaclab-3090.bat -p scripts/reinforcement_learning/sb3/train.py `
  --task Isaac-Cartpole-Direct-v0 `
  --num_envs 512 `
  --headless
```

**Outputs:** Checkpoints in `I:\isaaclab\logs\sb3\cartpole_direct\`

### Task 3: Monitor Training from WSL

```bash
# Option A: TensorBoard
source scripts/wsl/activate_rl_env_wsl.sh
tensorboard --logdir /mnt/i/isaaclab/logs --port 6006

# Option B: ROS 2 topics (if Isaac Sim publishes)
source scripts/wsl/setup_ros2_only.sh
ros2 topic list
ros2 topic echo /robot/joint_states
```

### Task 4: Analyze Training Results

```bash
source scripts/wsl/activate_rl_env_wsl.sh
cd experiments
python analyze_episode_rewards.py --logdir /mnt/i/isaaclab/logs/sb3/latest
```

### Task 5: Convert Robot Assets

**Windows (Isaac Sim Asset Converter):**
1. Open Isaac Sim
2. Load URDF from `C:\Users\yanbo\wSpace\cinebotRL\assets_own\`
3. Apply mesh scale: 0.001
4. Export to `assets_own\usd\`

**WSL (Validation):**
```bash
source scripts/wsl/activate_rl_env_wsl.sh
cd src/asset_inspector
python -m asset_inspector validate \
  --usd-path ../../assets_own/usd/mobile_manipulator_PPR_base_corrected.usd
```

---

## Troubleshooting

### Cannot see ROS 2 topics across WSL/Windows

**Checklist:**
1. Both sides use `ROS_DOMAIN_ID=55`
2. Both sides use `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`
3. Fast DDS profile configured on WSL
4. Windows firewall allows UDP 7400-7410, 7420, 8800
5. Clash/Mihomo bypasses ROS traffic
6. Can ping: `ping $(grep nameserver /etc/resolv.conf | awk '{print $2}')`

**Fix:**
```bash
# WSL: Reconfigure Fast DDS
bash scripts/networking/configure_fastdds_wsl.sh
source scripts/wsl/setup_ros2_only.sh
```

```powershell
# Windows: Reconfigure firewall
.\scripts\networking\configure_fastdds_firewall.ps1
# Restart Clash
.\scripts\networking\setup_ros2_humble_windows.ps1
```

### PyTorch CUDA not available in WSL

```bash
# Check CUDA paths
source scripts/wsl/activate_rl_env_wsl.sh
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# If false, manually export:
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH:-}"
```

### `ModuleNotFoundError: rclpy` in venv

**This is expected!** `rclpy` only works with system Python 3.10.

**Solution:** Deactivate venv for ROS 2 work:
```bash
deactivate
source scripts/wsl/setup_ros2_only.sh
```

---

## Documentation Index

- **Windows Setup:** [`docs/windows_side_requirements.md`](windows_side_requirements.md)
- **WSL Workflows:** [`docs/wsl_workflow_guide.md`](wsl_workflow_guide.md)
- **Phase 0 Log:** [`docs/tracking/phase0_environment.md`](tracking/phase0_environment.md)
- **Roadmap:** [`ROADMAP.md`](../ROADMAP.md)
- **This Summary:** [`docs/wsl_windows_integration.md`](wsl_windows_integration.md)

---

## Next Steps

1. ✅ WSL environment verified
2. ✅ ROS 2 communication tested
3. ✅ Documentation complete
4. ⏭️ Implement `MobileMMTrackEE-v0` task
5. ⏭️ Convert robot assets to USD
6. ⏭️ Run baseline training
7. ⏭️ Setup continuous monitoring

---

**Last Updated:** 2025-10-13  
**Status:** Integration complete and tested ✅
