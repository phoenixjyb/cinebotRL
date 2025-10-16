# Windows-Side Setup and Requirements

## Overview
This document outlines all the components that must be running on the Windows side to enable RL training coordination with WSL. The Windows machine hosts Isaac Sim, Isaac Lab, and the actual RL training loops, while WSL provides ROS 2 nodes, data preprocessing, and monitoring capabilities.

---

## 1. Hardware Configuration

### GPUs
- **RTX 3090** (CUDA device 0): Primary training GPU for Isaac Lab
- **Quadro P2000**: Display/secondary GPU

### Environment Variable
```powershell
$env:CUDA_VISIBLE_DEVICES = "0"  # Pin Isaac Lab to RTX 3090
```

---

## 2. Isaac Sim Installation

### Location
```
I:\isaacsim\
```

### Version
Isaac Sim 5.0.0-rc.45

### Key Launchers
- `I:\isaacsim\isaac-sim.bat` - GUI mode
- `I:\isaacsim\python.bat` - Bundled Python 3.11.13

### Asset Libraries
- `I:\isaacsim_assets` - Shared Omniverse assets
- `I:\OmniAssets` - Additional asset cache

### Environment Variables
```powershell
$env:ISAACSIM_PATH = "I:\isaacsim"
$env:OV_ASSETS_ROOT = "I:\isaacsim_assets"
```

### Launch with ROS 2 Bridge
```powershell
I:\isaacsim\isaac-sim.bat --/exts/ros2_bridge/useDomainID=55
```

---

## 3. Isaac Lab Installation

### Location
```
I:\isaaclab\
```

### Launcher Script
```
I:\isaaclab\isaaclab-3090.bat
```
This script:
- Activates bundled Python 3.11 environment
- Sets `CUDA_VISIBLE_DEVICES=0`
- Sources Isaac Sim paths

### Installed Packages
Editable installs in Isaac Sim's bundled Python:
- `isaaclab`
- `isaaclab_tasks`
- `isaaclab_assets`
- `isaaclab_rl[rl-games]`

### Key Dependencies
- torch 2.7.0+cu128
- ray 2.49.2
- rl-games (python3.11 branch)
- stable-baselines3
- gymnasium 0.23.1

### Verification Command
```powershell
I:\isaaclab\isaaclab-3090.bat -p scripts\reinforcement_learning\rl_games\train.py --task Isaac-Cartpole-Direct-v0 --max_iterations 10 --headless
```

---

## 4. ROS 2 Humble Installation

### Location
```
I:\ros2\ros2-windows\
```

### Python Version
Python 3.8 (use `py -3.8` for ROS commands)

**Note:** A newer Python 3.10 installation exists at `I:\ros2humble\ros2-windows` but the Python 3.8 version has been verified for WSL communication. See [`docs/ros2_python_versions_explained.md`](ros2_python_versions_explained.md) for details.

### Setup Script
```powershell
.\scripts\networking\setup_ros2_humble_windows.ps1
```

This script:
- Sources `local_setup.bat` from `I:\ros2\ros2-windows`
- Sets `ROS_DOMAIN_ID=55`
- Sets `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`
- Configures Fast DDS profile if present

**To use the Python 3.10 version instead:**
```powershell
.\scripts\networking\setup_ros2_humble_windows.ps1 -RosInstall I:\ros2humble\ros2-windows
```

### Test Commands
```powershell
# Listener (receives from WSL talker)
ros2 run demo_nodes_cpp listener

# Talker (sends to WSL listener)
ros2 run demo_nodes_cpp talker
```

---

## 5. Fast DDS Network Configuration

### Firewall Rules
Run the configuration script with elevated privileges:
```powershell
.\scripts\networking\configure_fastdds_firewall.ps1
```

This opens:
- UDP ports 7400-7410 (Fast DDS discovery)
- UDP port 7420 (Fast DDS data)
- UDP port 8800 (additional DDS traffic)

### Clash Proxy Bypass
The script also adds bypass entries to Clash for:
- WSL subnet: `172.16.0.0/12`
- ROS 2 ports: UDP 7400-7500, 8800

**Important:** Restart Clash after running the script.

### Environment Variables
```powershell
$env:ROS_DOMAIN_ID = "55"
$env:RMW_IMPLEMENTATION = "rmw_fastrtps_cpp"
```

### Verification
With WSL running `ros2 run demo_nodes_cpp talker`, the Windows listener should receive messages:
```
[INFO] [1697212345.123456789] [listener]: I heard: [Hello World: 427]
[INFO] [1697212346.123456789] [listener]: I heard: [Hello World: 428]
...
```

---

## 6. Training Workflow

### Stable Baselines 3 Example
```powershell
Set-Location C:\Users\yanbo\wSpace\cinebotRL
I:\isaaclab\isaaclab-3090.bat -p scripts/reinforcement_learning/sb3/train.py `
  --task MobileMMTrackEE-v0 `
  --num_envs 1024 `
  --headless
```

### RL-Games Example
```powershell
I:\isaaclab\isaaclab-3090.bat -p scripts/reinforcement_learning/rl_games/train.py `
  --task MobileMMTrackEE-v0 `
  --num_envs 2048 `
  --headless `
  --max_iterations 5000
```

### Outputs
- Checkpoints: `I:\isaaclab\logs\`
- TensorBoard logs: `I:\isaaclab\logs\<algorithm>\<task>\<timestamp>\`

### TensorBoard Monitoring
```powershell
# From Windows
I:\isaaclab\isaaclab-3090.bat -p -m tensorboard --logdir I:\isaaclab\logs

# Or from WSL (if path is accessible)
source scripts/wsl/setup_wsl_environment.sh
tensorboard --logdir /mnt/i/isaaclab/logs
```

---

## 7. Asset Management

### Source URDF Location
```
assets_own/mobile_manipulator_PPR_base_corrected.urdf
```

### USD Export Location
```
assets_own/usd/mobile_manipulator_PPR_base_corrected.usd
```

### Conversion Process
1. Open Isaac Sim Asset Converter (GUI or script)
2. Load URDF from `assets_own/`
3. Apply uniform mesh scale: `0.001` (millimeters to meters)
4. Export to `assets_own/usd/`
5. Run asset inspector from WSL:
   ```bash
   source scripts/wsl/setup_wsl_environment.sh
   cd src/asset_inspector
   python -m asset_inspector validate --usd-path ../../assets_own/usd/mobile_manipulator_PPR_base_corrected.usd
   ```

---

## 8. Status Checklist

Use this checklist before starting an RL training session:

### Prerequisites
- [ ] RTX 3090 is visible and idle
- [ ] Isaac Sim launches without errors
- [ ] Isaac Lab editable packages are installed
- [ ] ROS 2 Humble is sourced
- [ ] Fast DDS firewall rules are active
- [ ] Clash proxy bypass is configured

### Network Connectivity
- [ ] Can ping WSL from Windows: `ping <WSL_IP>`
- [ ] ROS 2 topics visible: `ros2 topic list --spin-time 5`
- [ ] Demo talker/listener exchange works

### Assets
- [ ] URDF exists in `assets_own/`
- [ ] USD exported to `assets_own/usd/`
- [ ] Asset validation report generated

### Training Environment
- [ ] Task registered in Isaac Lab
- [ ] Config files validated
- [ ] Log directory writable

---

## 9. Troubleshooting

### Issue: ROS 2 topics not visible across WSL/Windows
**Solution:**
1. Verify `ROS_DOMAIN_ID=55` on both sides
2. Check firewall rules: `netsh advfirewall firewall show rule name=all | findstr 7410`
3. Restart Clash proxy if enabled
4. Verify Fast DDS profile loaded: `echo $env:FASTDDS_DEFAULT_PROFILES_FILE`

### Issue: Isaac Sim crashes on launch
**Solution:**
1. Check GPU driver version (should be 580.97+)
2. Verify `ISAACSIM_PATH` is set correctly
3. Clear cache: Delete `%LOCALAPPDATA%\ov\cache`
4. Run with `--verbose` flag for detailed logs

### Issue: Training runs on wrong GPU
**Solution:**
1. Verify `CUDA_VISIBLE_DEVICES=0` before launching
2. Check with `nvidia-smi` that RTX 3090 is device 0
3. Confirm in Isaac Lab logs which device is being used

### Issue: Asset conversion fails
**Solution:**
1. Verify mesh files exist in `assets_own/meshes/`
2. Check mesh scale (STL files are in millimeters)
3. Use Asset Converter with `--uniform-scale 0.001`
4. Validate with asset inspector after conversion

---

## 10. Daily Workflow

### Morning Setup (5 minutes)
```powershell
# 1. Navigate to project
Set-Location C:\Users\yanbo\wSpace\cinebotRL

# 2. Setup ROS 2 environment
.\scripts\networking\setup_ros2_humble_windows.ps1

# 3. Start ROS 2 listener (for testing WSL connection)
Start-Job -ScriptBlock { ros2 run demo_nodes_cpp listener }

# 4. Verify Isaac Lab
I:\isaaclab\isaaclab-3090.bat -p -c "import isaaclab; print(f'Isaac Lab {isaaclab.__version__} ready')"
```

### Start Training Session
```powershell
# Launch training with monitoring
I:\isaaclab\isaaclab-3090.bat -p scripts/reinforcement_learning/sb3/train.py `
  --task MobileMMTrackEE-v0 `
  --num_envs 1024 `
  --headless `
  --enable_cameras False

# In another terminal, start TensorBoard
I:\isaaclab\isaaclab-3090.bat -p -m tensorboard --logdir I:\isaaclab\logs --port 6006
```

### End of Day
1. Check training logs in `I:\isaaclab\logs\`
2. Backup checkpoints if needed
3. Review TensorBoard metrics
4. Update experiment notes in `experiments/`

---

## Next Steps

1. **Complete Asset Pipeline**: Convert all robot URDFs to USD format
2. **Task Implementation**: Code the `MobileMMTrackEE-v0` environment
3. **Baseline Training**: Run initial experiments with default hyperparameters
4. **Cross-Platform Testing**: Verify WSL can monitor Windows training in real-time
5. **Documentation**: Update this file with actual training results and learnings

---

**Last Updated:** 2025-10-13  
**Status:** Network communication verified ✓
