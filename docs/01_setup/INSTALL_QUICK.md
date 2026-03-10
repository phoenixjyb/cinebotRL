# Isaac Lab Installation Quick Reference

## Your Current Setup ✓

- **WSL2**: Ubuntu 22.04
- **Python**: 3.11.10 (in .venv_rl311)
- **GPU**: NVIDIA RTX 3090 (24 GB) + Quadro P2000
- **Driver**: 580.97
- **PyTorch**: Upgrading to 2.7.0+cu128 (NVIDIA recommended)
- **Existing packages**: gymnasium, stable-baselines3

## Install Isaac Lab (Choose One)

### Option 1: Lightweight (Recommended for WSL)
```bash
cd /mnt/c/Users/yanbo/wSpace/cinebotRL
./scripts/install_isaaclab_wsl.sh
# Select option 1 when prompted
```

### Option 2: Full Stack (If you need full Isaac Sim)
```bash
./scripts/install_isaaclab_wsl.sh
# Select option 2 when prompted
# WARNING: 5-10 GB download, takes 15-30 min
```

### Option 3: From Source (For development)
```bash
./scripts/install_isaaclab_wsl.sh
# Select option 3 when prompted
# Clones to ~/IsaacLab and installs in editable mode
```

## After Installation

### 1. Reactivate environment
```bash
deactivate
source .venv_rl311/bin/activate
```

### 2. Verify installation
```bash
python scripts/verify_isaaclab_wsl.py
```

### 3. Quick import test
```bash
python -c "import isaaclab; print('✓ Isaac Lab ready')"
```

### 4. Test custom task
```bash
python -c "
from src.task_spec import register_isaac_lab_tasks
import gymnasium as gym

register_isaac_lab_tasks()
print('✓ Task registered:', 'MobileMMTrackEE-v0' in gym.envs.registry)
"
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'omni'"
Isaac Lab not installed yet. Run `./scripts/install_isaaclab_wsl.sh`

### "CUDA out of memory"
Reduce num_envs: `--num_envs 512` instead of 1024

### "libcuda.so.1: cannot open"
```bash
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH}"
```

### "Vulkan not available"
```bash
sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-tools
```

### Isaac Lab pip package not found
Isaac Lab 2.2.0 is available via pip on the NVIDIA index:
```bash
pip install --extra-index-url https://pypi.nvidia.com isaaclab==2.2.0
```
Use Option 3 (source install) only if you need a development/editable install.

## Environment Variables (Auto-configured)

The install script creates `~/.isaaclab_env` with:
```bash
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH}"
export VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/nvidia_icd.json"
export PHYSX_GPU_MAX_RIGID_CONTACT_COUNT=524288
export DISPLAY=""
```

This is automatically sourced when you activate the venv.

## Next Steps After Successful Install

1. **Test environment loading**
   ```bash
   python scripts/verify_isaaclab_wsl.py
   ```

2. **Run short training test**
   ```bash
   python scripts/reinforcement_learning/sb3/train.py \
       --task MobileMMTrackEE-v0 \
       --num_envs 4 \
       --total_timesteps 10000 \
       --headless
   ```

3. **Full training run**
   ```bash
   python scripts/reinforcement_learning/sb3/train.py \
       --task MobileMMTrackEE-v0 \
       --num_envs 1024 \
       --total_timesteps 10000000 \
       --headless
   ```

4. **Monitor with TensorBoard**
   ```bash
   tensorboard --logdir logs/sb3/MobileMMTrackEE-v0
   ```

## GPU Utilization Targets

With RTX 3090 (24 GB):
- **Training**: Aim for 1024-2048 parallel environments
- **GPU usage**: Should see >90% utilization in `nvidia-smi`
- **Memory**: ~20-22 GB used during training
- **Expected throughput**: 50K-100K steps/second

## Installation Locations

- **Venv**: `/mnt/c/Users/yanbo/wSpace/cinebotRL/.venv_rl311`
- **Isaac Lab (source)**: `~/IsaacLab` (if Option 3)
- **Environment file**: `~/.isaaclab_env`
- **USD Assets**: `/mnt/c/Users/yanbo/wSpace/cinebotRL/assets_own/usd/`

## When to Use Windows vs WSL

| Task | Location | Why |
|------|----------|-----|
| Asset conversion (URDF→USD) | Windows Isaac Sim GUI | Visual feedback, easier debugging |
| Visual debugging | Windows Isaac Sim GUI | See robot, trajectories in 3D |
| ROS 2 visualization | Windows | Better ROS 2 integration |
| RL Training | **Windows** Isaac Lab | Primary path: faster, CUDA 12.8, fully tested |
| Batch experiments | **Windows** Isaac Lab | PowerShell launchers in scripts/ |
| RL Analysis (offline) | WSL venv | PyTorch available without full Isaac Sim |

Both access same assets via `/mnt/c/` mount!
