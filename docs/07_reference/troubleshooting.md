# Troubleshooting Guide

Common issues and their solutions for the CinebotRL project.

---

## Quick Diagnosis

Run the status check script first:
```bash
bash scripts/wsl/check_wsl_setup.sh
```

This will identify most issues automatically.

---

## WSL Issues

### GPU Not Detected

**Symptoms:**
```bash
nvidia-smi
# bash: nvidia-smi: command not found
```

**Causes:**
- GPU passthrough not enabled
- NVIDIA drivers not installed on Windows
- WSL2 not using latest kernel

**Solutions:**

1. Update Windows to latest version
2. Update NVIDIA drivers on Windows (580.97 or newer)
3. Restart WSL:
   ```powershell
   # In Windows PowerShell
   wsl --shutdown
   wsl
   ```
4. Verify WSL2 kernel version:
   ```bash
   uname -r
   # Should be 5.15.x or newer
   ```

---

### CUDA Toolkit Not Found

**Symptoms:**
```bash
nvcc --version
# bash: nvcc: command not found
```

**Causes:**
- CUDA toolkit not installed
- CUDA not in PATH

**Solutions:**

1. Install CUDA toolkit 12.6:
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
   sudo dpkg -i cuda-keyring_1.0-1_all.deb
   sudo apt-get update
   sudo apt-get -y install cuda-toolkit-12-6
   ```

2. Add to PATH (or use activation script):
   ```bash
   export PATH="/usr/local/cuda-12.6/bin:$PATH"
   export LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH"
   ```

---

### PyTorch CUDA Not Available

**Symptoms:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
# False
```

**Causes:**
- CUDA paths not set
- PyTorch installed without CUDA support
- Wrong PyTorch version

**Solutions:**

1. Ensure you're in the venv:
   ```bash
   source scripts/wsl/activate_rl_env_wsl.sh
   ```

2. Check CUDA paths are set:
   ```bash
   echo $LD_LIBRARY_PATH
   # Should include /usr/lib/wsl/lib and /usr/local/cuda-12.6/lib64
   ```

3. Verify PyTorch version:
   ```bash
   pip list | grep torch
   # Should show torch 2.6.0+cu124 or similar
   ```

4. Reinstall PyTorch with CUDA if needed:
   ```bash
   pip install --force-reinstall torch torchvision torchaudio \
     --index-url https://download.pytorch.org/whl/cu121
   ```

---

### ROS 2 Not Found

**Symptoms:**
```bash
ros2 --version
# bash: ros2: command not found
```

**Causes:**
- ROS 2 not installed
- ROS 2 not sourced

**Solutions:**

1. Source ROS 2:
   ```bash
   source /opt/ros/humble/setup.bash
   ```

2. If still not found, install ROS 2 Humble:
   ```bash
   sudo apt update
   sudo apt install -y ros-humble-desktop
   ```

---

### `ModuleNotFoundError: rclpy` in Virtual Environment

**Symptoms:**
```bash
source .venv_rl311/bin/activate
python -c "import rclpy"
# ModuleNotFoundError: No module named 'rclpy._rclpy_pybind11'
```

**This is EXPECTED behavior!**

**Cause:**
ROS 2 Humble's `rclpy` is compiled for Python 3.10, but your venv uses Python 3.11.

**Solution:**
Use system Python for ROS 2 work:
```bash
deactivate  # Exit venv
source /opt/ros/humble/setup.bash
python3 --version  # Should show 3.10.x
python3 -c "import rclpy; print('Works!')"
```

Or use the helper:
```bash
source scripts/wsl/setup_ros2_only.sh
```

---

## ROS 2 Communication Issues

### Cannot See Topics from Windows

**Symptoms:**
```bash
ros2 topic list
# Only shows /parameter_events and /rosout
# Missing topics from Windows Isaac Sim
```

**Causes:**
- Domain ID mismatch
- RMW implementation mismatch
- Fast DDS not configured
- Firewall blocking traffic
- Network issues

**Solutions:**

1. **Verify Domain ID (must be 55 on both sides):**
   ```bash
   # WSL
   echo $ROS_DOMAIN_ID
   # Should be 55
   
   # Windows (PowerShell)
   echo $env:ROS_DOMAIN_ID
   # Should be 55
   ```

2. **Verify RMW implementation:**
   ```bash
   # WSL
   echo $RMW_IMPLEMENTATION
   # Should be rmw_fastrtps_cpp
   ```

3. **Reconfigure Fast DDS:**
   ```bash
   bash scripts/networking/configure_fastdds_wsl.sh
   source scripts/wsl/setup_ros2_only.sh
   ```

4. **Test network connectivity:**
   ```bash
   # Get Windows IP
   WIN_IP=$(grep nameserver /etc/resolv.conf | awk '{print $2}')
   echo "Windows IP: $WIN_IP"
   
   # Test ping
   ping -c 3 $WIN_IP
   ```

5. **Check Windows firewall (on Windows):**
   ```powershell
   # Re-run firewall configuration
   .\scripts\networking\configure_fastdds_firewall.ps1
   
   # Verify rules exist
   netsh advfirewall firewall show rule name=all | findstr "7410"
   ```

6. **Restart proxy if using Clash/Mihomo:**
   - Windows: Restart Clash
   - WSL: Check `/etc/mihomo/no_proxy.list` includes Windows IP

---

### Topics Visible But No Data

**Symptoms:**
```bash
ros2 topic list
# Shows /robot/joint_states
ros2 topic echo /robot/joint_states
# No output, just waits...
```

**Causes:**
- Publisher not running
- QoS policy mismatch
- Message type incompatibility

**Solutions:**

1. **Check if publisher is active:**
   ```bash
   ros2 topic info /robot/joint_states
   # Should show publisher count > 0
   ```

2. **Check topic rate:**
   ```bash
   ros2 topic hz /robot/joint_states
   # Should show messages per second
   ```

3. **Try with --no-qos:**
   ```bash
   ros2 topic echo /robot/joint_states --no-qos
   ```

---

## Windows Issues

### ROS 2 Python Version Mismatch

**Symptoms:**
```powershell
ros2 run demo_nodes_cpp listener
# ModuleNotFoundError: _rclpy_pybind11
```

**Cause:**
Using wrong Python version for your ROS 2 installation.

**Solutions:**

1. Check which Python your ROS 2 installation needs:
   ```powershell
   dir I:\ros2\ros2-windows\Lib\site-packages\rclpy\_rclpy*.pyd
   # Look for cp38 (Python 3.8) or cp310 (Python 3.10)
   ```

2. Use correct Python:
   ```powershell
   # If cp38:
   py -3.8 I:\ros2\ros2-windows\Scripts\ros2-script.py run demo_nodes_cpp listener
   
   # If cp310:
   py -3.10 I:\ros2\ros2-windows\Scripts\ros2-script.py run demo_nodes_cpp listener
   ```

3. Or use the setup script (defaults to Python 3.8 installation):
   ```powershell
   .\scripts\networking\setup_ros2_humble_windows.ps1
   ros2 run demo_nodes_cpp listener
   ```

---

### Isaac Lab Training Crashes

**Symptoms:**
```powershell
I:\isaaclab\isaaclab-3090.bat -p scripts/train.py
# CUDA out of memory error
```

**Causes:**
- Too many parallel environments
- GPU memory exhausted
- Wrong GPU selected

**Solutions:**

1. **Reduce number of environments:**
   ```powershell
   # Instead of --num_envs 2048
   I:\isaaclab\isaaclab-3090.bat -p scripts/train.py --num_envs 512
   ```

2. **Verify correct GPU:**
   ```powershell
   echo $env:CUDA_VISIBLE_DEVICES
   # Should be 0 (RTX 3090)
   ```

3. **Check GPU usage:**
   ```powershell
   nvidia-smi
   # Look for memory usage on RTX 3090
   ```

---

## Asset Issues

### URDF Conversion Fails

**Symptoms:**
- Isaac Sim Asset Converter crashes
- USD file not generated

**Causes:**
- Missing mesh files
- Invalid URDF syntax
- Incorrect file paths

**Solutions:**

1. **Verify URDF is valid:**
   ```bash
   check_urdf assets_own/mobile_manipulator_PPR_base_corrected.urdf
   ```

2. **Check mesh files exist:**
   ```bash
   ls -la assets_own/meshes/
   # All referenced STL files should be present
   ```

3. **Use correct mesh scale:**
   - If meshes are in millimeters, use scale 0.001
   - If meshes are in meters, use scale 1.0

---

### Asset Inspector Errors

**Symptoms:**
```bash
python -m asset_inspector validate --usd-path file.usd
# Various validation errors
```

**Solutions:**

1. **Ensure using correct Python environment:**
   ```bash
   source scripts/wsl/activate_rl_env_wsl.sh
   python -m asset_inspector validate --usd-path file.usd
   ```

2. **Check USD file exists and is readable:**
   ```bash
   file assets_own/usd/mobile_manipulator_PPR_base_corrected.usd
   # Should show "data" or similar
   ```

---

## Performance Issues

### Training Very Slow

**Symptoms:**
- Low FPS during training
- GPU not fully utilized

**Causes:**
- Wrong GPU selected
- Too many environments
- CPU bottleneck
- Headless mode not enabled

**Solutions:**

1. **Verify GPU selection:**
   ```powershell
   nvidia-smi
   # Check which GPU is being used
   ```

2. **Use headless mode:**
   ```powershell
   I:\isaaclab\isaaclab-3090.bat -p scripts/train.py --headless
   ```

3. **Adjust number of environments:**
   ```powershell
   # Find optimal number (start with 512, increase until GPU maxed)
   I:\isaaclab\isaaclab-3090.bat -p scripts/train.py --num_envs 1024
   ```

---

## Environment Variable Issues

### Variables Not Persisting

**Symptoms:**
- Need to export variables every terminal session
- Scripts don't find expected paths

**Solutions:**

1. **Use the activation scripts:**
   ```bash
   # Instead of manually exporting
   source scripts/wsl/setup_ros2_only.sh  # For ROS 2
   source scripts/wsl/activate_rl_env_wsl.sh  # For RL
   ```

2. **Add to shell profile (optional):**
   ```bash
   # Add to ~/.bashrc for persistence
   echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc
   echo 'export ROS_DOMAIN_ID=55' >> ~/.bashrc
   ```

---

## Getting More Help

### Enable Verbose Logging

**ROS 2:**
```bash
export RCUTILS_CONSOLE_OUTPUT_FORMAT="[{severity}] [{name}]: {message}"
export RCUTILS_LOGGING_USE_STDOUT=1
export RCUTILS_LOGGING_BUFFERED_STREAM=1
```

**Isaac Lab:**
```powershell
I:\isaaclab\isaaclab-3090.bat -p scripts/train.py --verbose
```

### Collect Diagnostic Information

Run the status check and save output:
```bash
bash scripts/wsl/check_wsl_setup.sh > wsl_diagnostics.txt 2>&1
```

### Check Logs

**ROS 2 logs:**
```bash
cat ~/.ros/log/latest/rosout.log
```

**Isaac Lab logs:**
```powershell
# Check most recent log directory
dir I:\isaaclab\logs\ | sort -Descending | select -First 1
```

---

## Common Error Messages

### `DLL load failed` (Windows)

**Error:**
```
ImportError: DLL load failed while importing _rclpy_pybind11
```

**Solution:**
Missing DLL dependencies. Ensure Fast DDS DLLs are on PATH:
```powershell
.\scripts\networking\setup_ros2_humble_windows.ps1
```

### `Connection refused` (ROS 2)

**Error:**
```
[ERROR] [rclpy]: Failed to contact master at [...]
```

**Note:** This is a legacy ROS 1 error message. ROS 2 doesn't use a master.

**Solution:**
- Ignore if using ROS 2
- Verify `ROS_DOMAIN_ID` is set
- Check Fast DDS configuration

### `CUDA initialization failure`

**Error:**
```
RuntimeError: CUDA initialization failure
```

**Solution:**
```bash
# Check GPU visibility
nvidia-smi

# Check CUDA paths
echo $LD_LIBRARY_PATH

# Reactivate environment
source scripts/wsl/activate_rl_env_wsl.sh
```

---

## Still Stuck?

1. Check [Phase 0 Environment Log](../tracking/phase0_environment.md) for setup history
2. Review [Architecture Documentation](../architecture/) to understand system design
3. Compare your setup with [WSL Setup Guide](../setup/wsl_setup_guide.md)

---

**Last Updated:** 2025-10-13
