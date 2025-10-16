# Quick Reference Card

## 🚀 Quick Start (Copy & Paste)

### WSL - Check Everything
```bash
cd /mnt/c/Users/yanbo/wSpace/cinebotRL && bash scripts/wsl/check_wsl_setup.sh
```

### WSL - ROS 2 Only
```bash
cd /mnt/c/Users/yanbo/wSpace/cinebotRL && source scripts/wsl/setup_ros2_only.sh
```

### WSL - RL Development
```bash
cd /mnt/c/Users/yanbo/wSpace/cinebotRL && source scripts/wsl/activate_rl_env_wsl.sh
```

### Windows - ROS 2 Setup
```powershell
cd C:\Users\yanbo\wSpace\cinebotRL
.\scripts\networking\setup_ros2_humble_windows.ps1
```

---

## 📊 Environment Inventory

### ✅ What's on WSL
| Component | Status | Path/Version |
|-----------|--------|--------------|
| OS | ✅ Ready | Ubuntu 22.04 WSL2 |
| GPU Access | ✅ Ready | RTX 3090 + P2000 (nvidia-smi) |
| CUDA | ✅ Ready | 12.6.85 |
| ROS 2 Humble | ✅ Ready | Python 3.10 (system) |
| RL Venv | ✅ Ready | `.venv_rl311` Python 3.11 |
| PyTorch | ✅ Ready | 2.6.0+cu124 (CUDA enabled) |
| Stable Baselines3 | ✅ Ready | 2.7.0 |
| Gymnasium | ✅ Ready | 1.2.1 |
| Fast DDS | ✅ Ready | Configured for Windows |

### ✅ What's on Windows
| Component | Status | Location |
|-----------|--------|----------|
| Isaac Sim | ✅ Ready | I:\isaacsim (5.0.0-rc.45) |
| Isaac Lab | ✅ Ready | I:\isaaclab |
| ROS 2 Humble | ✅ Ready | I:\ros2humble\ros2-windows |
| Training GPU | ✅ Ready | RTX 3090 (CUDA device 0) |
| Fast DDS | ✅ Ready | Firewall configured |
| Python 3.11 | ✅ Ready | Bundled with Isaac Sim |
| Python 3.10 | ✅ Ready | For ROS 2 |

---

## 🔌 Communication Test

### Understanding the "Bridge"

**There is no separate bridge process!** All ROS 2 nodes communicate via **Fast DDS** (peer-to-peer network). Isaac Sim (Py 3.11), Windows ROS 2 (Py 3.8), and WSL ROS 2 (Py 3.10) all talk to each other automatically via UDP on Domain ID 55.

See [`docs/ros2_bridge_explained.md`](ros2_bridge_explained.md) for detailed architecture.

### Test 1: WSL → Windows (Talker/Listener)

**WSL Terminal:**
```bash
source scripts/wsl/setup_ros2_only.sh
ros2 run demo_nodes_cpp talker
```

**Windows Terminal:**
```powershell
.\scripts\networking\setup_ros2_humble_windows.ps1
ros2 run demo_nodes_cpp listener
```

**✅ Success:** Windows shows "I heard: [Hello World: N]"

---

## 🎯 Use Case Decision Tree

```
Need to...

├─ Communicate with Windows via ROS 2?
│  └─ Use: System Python 3.10 + ROS 2
│     Command: source scripts/wsl/setup_ros2_only.sh
│
├─ Run PyTorch/RL experiments?
│  └─ Use: .venv_rl311 Python 3.11
│     Command: source scripts/wsl/activate_rl_env_wsl.sh
│
├─ Train in Isaac Lab?
│  └─ Use: Windows Isaac Lab
│     Command: I:\isaaclab\isaaclab-3090.bat -p scripts/...
│
└─ Analyze training logs?
   └─ Use: .venv_rl311 Python 3.11
      Command: source scripts/wsl/activate_rl_env_wsl.sh
```

---

## 🔧 Essential Environment Variables

### WSL - ROS 2 Mode
```bash
export ROS_DOMAIN_ID=55
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_DEFAULT_PROFILES_FILE=$HOME/fastdds_windows.xml
```

### WSL - RL Mode
```bash
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/cuda-12.6/bin:${PATH}"
```

### Windows
```powershell
$env:ROS_DOMAIN_ID = "55"
$env:RMW_IMPLEMENTATION = "rmw_fastrtps_cpp"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:ISAACSIM_PATH = "I:\isaacsim"
```

---

## 🐛 Troubleshooting One-Liners

### Can't see ROS topics?
```bash
# WSL: Check domain ID
echo $ROS_DOMAIN_ID  # Should be 55

# WSL: Check RMW
echo $RMW_IMPLEMENTATION  # Should be rmw_fastrtps_cpp

# WSL: Reconfigure
bash scripts/networking/configure_fastdds_wsl.sh
```

### PyTorch CUDA not working?
```bash
source scripts/wsl/activate_rl_env_wsl.sh
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
```

### rclpy import error in venv?
```bash
# This is expected! Use system Python for ROS 2:
deactivate
source scripts/wsl/setup_ros2_only.sh
```

---

## 📁 Key Files Locations

### WSL
```
/mnt/c/Users/yanbo/wSpace/cinebotRL/
├── scripts/wsl/
│   ├── check_wsl_setup.sh          ← Status check
│   ├── setup_ros2_only.sh          ← ROS 2 environment
│   ├── activate_rl_env_wsl.sh      ← RL environment
│   └── setup_wsl_environment.sh    ← Both environments
├── docs/
│   ├── wsl_workflow_guide.md       ← Detailed workflows
│   ├── windows_side_requirements.md ← Windows setup
│   └── wsl_windows_integration.md  ← Architecture overview
└── .venv_rl311/                    ← Python 3.11 RL venv
```

### Windows
```
I:\isaaclab\                        ← Isaac Lab installation
I:\isaacsim\                        ← Isaac Sim installation
I:\ros2humble\ros2-windows\         ← ROS 2 Humble
C:\Users\yanbo\wSpace\cinebotRL\    ← Project root
```

---

## 📚 Documentation Quick Links

- **[README.md](../README.md)** - Main project overview
- **[ROADMAP.md](../ROADMAP.md)** - Implementation plan
- **[WSL Workflow Guide](wsl_workflow_guide.md)** - Detailed WSL usage
- **[Windows Requirements](windows_side_requirements.md)** - Windows setup
- **[Integration Summary](wsl_windows_integration.md)** - Complete architecture
- **[Phase 0 Log](tracking/phase0_environment.md)** - Setup history

---

## ⚡ Most Common Commands

### Daily Morning Check
```bash
cd /mnt/c/Users/yanbo/wSpace/cinebotRL
bash scripts/wsl/check_wsl_setup.sh
```

### Start ROS 2 Monitoring
```bash
source scripts/wsl/setup_ros2_only.sh
ros2 topic list
```

### Start RL Development
```bash
source scripts/wsl/activate_rl_env_wsl.sh
python my_script.py
```

### Monitor Training Logs
```bash
source scripts/wsl/activate_rl_env_wsl.sh
tensorboard --logdir /mnt/i/isaaclab/logs
```

### Windows: Start Training
```powershell
I:\isaaclab\isaaclab-3090.bat -p scripts/reinforcement_learning/sb3/train.py --task YourTask-v0 --headless
```

---

**Print this card for quick reference!**  
Last Updated: 2025-10-13
