# Cinebot RL Environment

## Overview

This project implements **reinforcement learning** for a **mobile manipulator robot** (6-DOF arm + differential drive base) using **Isaac Sim** and **Isaac Lab** on **Windows native**. The system trains a PPO agent to perform precise end-effector trajectory tracking using 8,192 parallel environments.

**Current Status (2026-06-26):** Proto2 baseline active. The repo now loads
`recomoProto2-1190` by default and exposes `RecomoProto2TrackEE-v0` for the
first 8-action SB3 baseline.

**Key Achievements:**
- ✅ **Windows Training Operational** - Isaac Lab + Stable-Baselines3 PPO verified
- ✅ **Session 8i v3 Success** - Sigmoid transition eliminates emergency pauses; KL 0.0222, Explained Variance 0.342
- ✅ **Comprehensive Documentation** - Reward system, model architecture, training guides
- ✅ **8,192 Parallel Envs** - High-throughput training (~12M interactions/sec)
- ⚠️ **Reachability 0.3%** - Task performance needs evaluation and next-session tuning

## Architecture

```
Windows (Primary Training Platform) ✅
├── Isaac Sim 5.0.0
├── Isaac Lab 2.x / pip:isaaclab==0.46.2 (Python 3.11.13, torch 2.7.0+cu128)
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

# 2. Launch training (default task: RecomoProto2TrackEE-v0)
.\scripts\launch_training_windows.ps1 -NumEnvs 64 -Headless

# 3. Monitor in separate window
.\scripts\monitor_training.ps1 -Mode all
```

**Alternative: Direct Isaac Lab Launcher**
```powershell
# Set ISAAC_LAB_ROOT once (or add to your profile):
$env:ISAAC_LAB_ROOT = "I:\isaaclab"   # adjust to your installation

& "$env:ISAAC_LAB_ROOT\isaaclab.bat" -p scripts/reinforcement_learning/sb3/train.py `
  --task RecomoProto2TrackEE-v0 --num_envs 64 --headless
```

> **Tip:** Set `ISAAC_LAB_ROOT` as a system environment variable to avoid editing scripts when
> moving your Isaac Lab installation. The launcher script reads it automatically.

**📚 Documentation**: 
- **⚡ Quick Start**: [START_TRAINING_NOW.md](START_TRAINING_NOW.md) - Get training running in 3 commands!
- **Proto2 Baseline**: [docs/03_training/PROTO2_BASELINE.md](docs/03_training/PROTO2_BASELINE.md) - Active USD, task alias, action contract, and validation gates
- **📖 Complete Documentation**: [docs/README.md](docs/README.md) - Organized learning path
- **🔧 Quick Reference**: [docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md) - One-page command cheat sheet
- **📊 Reward System**: [docs/reference/REWARD_SYSTEM_DESIGN.md](docs/reference/REWARD_SYSTEM_DESIGN.md) - How rewards are designed
- **🧠 Model Architecture**: [docs/reference/MODEL_ARCHITECTURE.md](docs/reference/MODEL_ARCHITECTURE.md) - PPO network, obs/action spaces, hyperparameters
- **📈 Training Sessions**: [docs/training_sessions/TRAINING_SESSIONS_MASTER_LOG.md](docs/training_sessions/TRAINING_SESSIONS_MASTER_LOG.md) - All training runs documented

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

📖 **[Complete Documentation Index](docs/README.md)** - Organized learning path with 9 categorized sections

### Essential Documents (Start Here!)

**Getting Started:**
- ⚡ **[START_TRAINING_NOW.md](START_TRAINING_NOW.md)** - Quick start guide (3 commands to train!)
- 🔧 **[Quick Reference Card](docs/QUICK_REFERENCE.md)** - One-page command cheat sheet
- 🗺️ **[ROADMAP.md](ROADMAP.md)** - Project phases and implementation plan

**Core Documentation:**
- 📊 **[Reward System Design](docs/reference/REWARD_SYSTEM_DESIGN.md)** - 9 reward components explained (800+ lines)
- 🧠 **[Model Architecture](docs/reference/MODEL_ARCHITECTURE.md)** - PPO network, obs/action spaces, hyperparameters (1000+ lines)
- 📈 **[Training Sessions Master Log](docs/training_sessions/TRAINING_SESSIONS_MASTER_LOG.md)** - All training runs documented
- 🔧 **[Session 5b Fix Summary](docs/training_sessions/SESSION_5B_FIX_SUMMARY.md)** - Base mobility fixes explained

### Setup & Architecture

- 🪟 **[Windows Setup Guide](docs/01_setup/windows_setup_guide.md)** - Configure Windows environment
- 🔧 **[WSL Setup Guide](docs/01_setup/wsl_setup_guide.md)** - Optional WSL configuration  
- 🏗️ **[Architecture Overview](docs/02_architecture/overview.md)** - How everything fits together
- 🚂 **[Training Architecture](docs/02_architecture/training_architecture.md)** - IsaacLab → SB3 pipeline
- 🏗️ **[PPR Control Architecture](docs/02_architecture/PPR_CONTROL_ARCHITECTURE.md)** - Base mobility control flow

### Training Guides

- 📚 **[Multi-Trajectory Training](docs/03_training/multi_trajectory_training.md)** - Advanced training workflows
- 📋 **[Training Command Reference](docs/03_training/TRAINING_COMMAND_REFERENCE.md)** - All CLI flags explained
- 🎯 **[Training Readiness Checklist](docs/03_training/TRAINING_READINESS_CHECKLIST.md)** - Pre-flight checks
- � **[Entropy & KL Explained](docs/03_training/ENTROPY_AND_KL_EXPLAINED.md)** - PPO hyperparameters demystified

### Bug Fixes & Optimizations

- 🐛 **[Base Movement Bug Analysis](docs/05_bug_fixes/BASE_MOVEMENT_BUG_ANALYSIS.md)** - "Lazy base" problem solved
- 🚀 **[Policy Divergence at 200M](docs/04_optimization/POLICY_DIVERGENCE_200M.md)** - Long training stability
- ⚡ **[Trajectory Tracking Improvements](docs/04_optimization/TRAJECTORY_TRACKING_IMPROVEMENTS.md)** - Performance tuning

### Workflows & Visualization

- ⚡ **[Daily Workflow](docs/06_workflows/daily_workflow.md)** - Common tasks and commands
- 📊 **[Visualization Guide](docs/06_workflows/VISUALIZATION_GUIDE.md)** - TensorBoard, RViz, Isaac Sim
- 🧪 **[Evaluation Guide](docs/06_workflows/EVALUATION_GUIDE.md)** - Test trained policies

### Reference Materials

- 🔧 **[Troubleshooting](docs/07_reference/troubleshooting.md)** - Common issues & solutions
- 🏠 **[Robot Home Position](docs/07_reference/ROBOT_HOME_POSITION.md)** - Joint configurations
- 📊 **[Reward Cheatsheet](docs/07_reference/reward_cheatsheet.md)** - Quick reward formula reference
- 📝 **[Trajectory Info](docs/07_reference/TRAJECTORY_INFO.md)** - Training trajectory details

## Environment Details

### Windows Side (Primary Training Platform)
- **Isaac Sim:** `I:\isaacsim` (5.0.0, Python 3.11.13)
- **Isaac Lab:** `I:\isaaclab` (pip:isaaclab==0.46.2 / GitHub 2.x, torch 2.7.0+cu128, editable install with SB3)
- **Training Framework:** Stable Baselines3 PPO with custom `IsaacLabToSB3VecEnvWrapper`
- **Training GPU:** RTX 3090 (CUDA device 0, auto-detected)
- **Display GPU:** Quadro P2000 (CUDA device 1)
- **ROS 2 Humble:** two installs — `I:\ros2\ros2-windows` **(Python 3.8, verified ✅)** and `I:\ros2humble\ros2-windows` (Python 3.10, not yet tested). Use `scripts\networking\setup_ros2_humble_windows.ps1` (defaults to 3.8 install).
- **Status:** ✅ All compatibility issues resolved, training verified working

### WSL Side (Optional - Not Required for Training)
- **OS:** Ubuntu 22.04 (WSL2)
- **ROS 2 Humble:** System Python 3.10 (`/opt/ros/humble`) - for monitoring only
- **RL Environment:** `.venv_rl311` Python 3.11 (PyTorch 2.7.0+cu128, SB3 2.5.0) - for analysis only; run `setup_rl_venv.sh` to update
- **CUDA:** 12.6.x installed; 12.8 recommended (`install_cuda_wsl.sh` default updated)

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

## Training Performance (Session 5b Baseline)

**Current Task:** `MobileMMTrackEE-v0` (Mobile Manipulator End-Effector Tracking)

**Robot Specifications:**
- **DOF:** 9 total (6 arm joints + 3 base: PPR = prismatic X, prismatic Y, revolute Z)
- **Action Space:** 9D continuous (6 arm joint positions + body-frame base v_x, v_y, and ω_z)
- **Observation Space:** 46D (base state + arm joints + EE state + tracking errors + base-to-target signals)
- **Algorithm:** Stable Baselines3 PPO with enhanced 3-layer MLP ([256, 256, 128] → ~235K params)

**Session 5b Results (100M+ Steps):**
- **Training Time:** ~18 hours (RTX 3090, 8,192 parallel environments)
- **Throughput:** ~1,500 steps/sec wall-clock (~12.3M env interactions/sec)
- **Final Reward:** ~0.85 (normalized, see [Reward System Design](docs/reference/REWARD_SYSTEM_DESIGN.md))
- **GPU Utilization:** ~75% (18GB/24GB memory)
- **Base Mobility:** ✅ Fixed - base actively repositions when targets are out of reach

**Key Fixes Applied in Session 5b:**
- Capped `base_mobilization_reward` to prevent reward hacking
- Added `excessive_base_movement_penalty` for large movements (>0.1m)
- Increased `target_distance_penalty` from 3.0 → 5.0
- Added 4D base-to-target observations (dx, dy, distance, out_of_reach flag)

See [Session 5b Fix Summary](docs/training_sessions/SESSION_5B_FIX_SUMMARY.md) for complete details.

**Scaling Recommendations:**

| Num Envs | GPU Memory | Training Time (100M) | Recommended GPU |
|----------|------------|----------------------|-----------------|
| 512 | ~4GB | ~7 days | RTX 3060 (12GB) |
| 2048 | ~8GB | ~2 days | RTX 3070 (8GB) |
| 4096 | ~12GB | ~1 day | RTX 3080 (10GB) |
| 8192 | ~18GB | ~18 hours | RTX 3090 (24GB) |
| 16384 | ~32GB | ~10 hours | RTX A6000 (48GB) |

**Architecture Details:**  
Custom `IsaacLabToSB3VecEnvWrapper` bridges Isaac Lab (dict observations, torch tensors, GPU) to Stable Baselines3 (numpy arrays, CPU). See [Model Architecture](docs/reference/MODEL_ARCHITECTURE.md) for technical details.

## Troubleshooting

**Training not starting?**
- Verify Isaac Lab installation: `I:\isaaclab\isaaclab.bat -h`
- Check GPU detection: `nvidia-smi`
- Review logs in latest directory under `I:\isaaclab\logs\sb3\`
- See [Troubleshooting Guide](docs/07_reference/troubleshooting.md) for known issues and solutions

**Import errors or compatibility issues?**
- All 12+ compatibility issues have been resolved as of 2025-10-15
- Gymnasium ale_py issue: Already patched in Isaac Lab Python environment
- See [TRAINING_SUCCESS.md](TRAINING_SUCCESS.md) for complete list of fixes

**Base not moving during training?**
- Verify URDF fixes are applied (PPR helper masses = 1.0kg, joint_theta limits = ±6.28 rad)
- Check observation space includes `base_to_target` signals (4 dims)
- Review reward components include `base_mobilization_reward`
- See [Base Movement Bug Analysis](docs/05_bug_fixes/BASE_MOVEMENT_BUG_ANALYSIS.md) for details

**Policy diverging after 50M+ steps?**
- Enable entropy decay: `--enable_entropy_decay` (exponential decay, τ=10M)
- Monitor KL divergence: should stay below 0.03
- Check `excessive_base_movement_penalty` is enabled
- See [Policy Divergence at 200M](docs/04_optimization/POLICY_DIVERGENCE_200M.md) for long training stability

**ROS 2 topics not visible? (Optional)**
- Check `ROS_DOMAIN_ID=55` on both sides
- Verify firewall: `netsh advfirewall firewall show rule name=all | findstr 7410`
- Reconfigure Fast DDS: `bash scripts/networking/configure_fastdds_wsl.sh`

**PyTorch CUDA not available in WSL?**
- Not needed for training! Use Windows side for training
- For analysis only: Activate venv `source scripts/wsl/activate_rl_env_wsl.sh`

**Complete troubleshooting guide:** [docs/07_reference/troubleshooting.md](docs/07_reference/troubleshooting.md)

## Quick Links

📚 **Complete documentation index**: [docs/README.md](docs/README.md)

**Essential Documents:**
- 🚀 [Quick Start Training](START_TRAINING_NOW.md) - Get training running in 3 commands
- 📊 [Reward System Design](docs/reference/REWARD_SYSTEM_DESIGN.md) - How rewards guide learning (800+ lines)
- 🧠 [Model Architecture](docs/reference/MODEL_ARCHITECTURE.md) - PPO network, obs/action spaces (1000+ lines)
- 📈 [Training Sessions Log](docs/training_sessions/TRAINING_SESSIONS_MASTER_LOG.md) - All training runs documented
- � [Session 5b Fix Summary](docs/training_sessions/SESSION_5B_FIX_SUMMARY.md) - Base mobility fixes explained
- 🏗️ [PPR Control Architecture](docs/02_architecture/PPR_CONTROL_ARCHITECTURE.md) - Base control flow explained
- � [Training Command Reference](docs/03_training/TRAINING_COMMAND_REFERENCE.md) - All CLI flags
- 🔧 [Troubleshooting Guide](docs/07_reference/troubleshooting.md) - Common issues & solutions

## Next Steps

**Current Phase (Phase 2 - Scaling & Deployment):**
1. ✅ Environment setup complete
2. ✅ ROS 2 communication tested (optional)
3. ✅ Windows training verified working
4. ✅ `MobileMMTrackEE-v0` task implemented and training
5. ✅ Robot USD asset created and validated
6. ✅ **5 critical URDF physics bugs fixed** (base mobility, joint limits, inertia)
7. ✅ **Session 5b completed** (100M+ steps, base mobility validated)
8. ✅ **Comprehensive documentation published** (Reward System, Model Architecture)
9. ✅ **Documentation reorganized** (37 files into 8 categorized directories)
10. ✅ **Session 8i v3 completed** (41.94M steps, sigmoid transition, training stable)

**Next Steps:**
11. ⏭️ Evaluate Session 8i v3 policy on diverse test trajectories (reachability currently 0.3%)
12. ⏭️ Tune reward shaping to improve reachability rate
13. ⏭️ Scale to 16,384 envs for faster convergence (requires RTX A6000 or 2× RTX 3090)
14. ⏭️ Implement real-time policy inference pipeline
15. ⏭️ Deploy to physical robot hardware

**See [ROADMAP.md](ROADMAP.md) for complete project timeline.**

---

**Last Updated:** 2025-11-08  
**Training Status:** ✅ Session 8i v3 Complete (41.94M steps, sigmoid transition)  
**Documentation:** ✅ Comprehensive (Reward System + Model Architecture)  
**GPU Utilization:** 75% (18GB/24GB on RTX 3090)
