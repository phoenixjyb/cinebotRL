# Isaac Lab on WSL2 (pip installation)

This guide covers installing Isaac Lab via pip in WSL2 for headless RL training. This approach is cleaner and faster than the full Isaac Sim install.

## Prerequisites

- **WSL2 Ubuntu 22.04** with NVIDIA GPU support
- **NVIDIA Driver** on Windows (545.84+ recommended)
- **CUDA toolkit** in WSL (verify with `nvidia-smi` and `nvcc --version`)
- **Python 3.10 or 3.11** (Isaac Lab officially supports these versions)

Verify GPU access:
```bash
nvidia-smi
# Should show GPU and driver version
```

## Architecture: Windows GUI + WSL Training

```
Windows (Isaac Sim GUI)          WSL2 (Isaac Lab Headless)
├─ Asset conversion (URDF→USD)   ├─ High-throughput RL training
├─ Visual debugging               ├─ Batch environment simulation
├─ ROS 2 visualization            ├─ Headless rendering
└─ Interactive testing            └─ Python 3.11 + CUDA acceleration

        Shared via /mnt/c/
        ├─ USD robot assets
        ├─ Training scripts
        ├─ Trajectory data
        └─ Checkpoints
```

## Step 1: Set Python version

Isaac Lab requires Python 3.10 or 3.11. Use pyenv to set the correct version:

```bash
# Check available versions
pyenv versions

# Set Python 3.11.10 for this project
cd /mnt/c/Users/yanbo/wSpace/cinebotRL
pyenv local 3.11.10

# Verify
python --version
# Should output: Python 3.11.10
```

## Step 2: Create Isaac Lab virtual environment

Use the project's venv setup script:

```bash
cd /mnt/c/Users/yanbo/wSpace/cinebotRL
./scripts/setup_rl_venv.sh --python python3.11

# This creates .venv_rl311 with PyTorch + CUDA 12.1
```

Activate the environment:
```bash
source .venv_rl311/bin/activate
```

## Step 3: Install Isaac Lab via pip

Isaac Lab can now be installed directly via pip (as of Isaac Sim 4.0+):

```bash
# First, ensure you have the recommended PyTorch version (NVIDIA official)
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# Verify CUDA is available
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# Install Isaac Lab with Isaac Sim runtime
pip install isaacsim-rl isaacsim-replicator isaacsim-extscache-physics isaacsim-extscache-kit-sdk isaacsim-extscache-kit isaacsim-app --extra-index-url https://pypi.nvidia.com

# Install Isaac Lab (lightweight, no full Isaac Sim)
pip install --upgrade pip
pip install isaaclab

# Or install from source for development:
# git clone https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
# cd ~/IsaacLab
# pip install -e .
```

**Alternative (lighter install)**: If the full `isaacsim-*` packages are too heavy, try:
```bash
# Minimal Isaac Lab install (may require additional dependencies)
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install isaaclab
```

**Note**: As of 2025, NVIDIA recommends PyTorch 2.7.0 with CUDA 12.8. Check the [official docs](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html) for the latest package names.

## Step 4: Install training dependencies

```bash
pip install stable-baselines3[extra]
pip install wandb tensorboard hydra-core
pip install rsl-rl  # If using RSL_RL instead of SB3
```

## Step 5: Set environment variables

Create `~/.isaaclab_env` for persistent settings:

```bash
cat > ~/.isaaclab_env <<'ENV'
# Isaac Lab paths
export ISAAC_LAB_PATH="${HOME}/IsaacLab"  # If installed from source
export ISAAC_SIM_ASSETS="/mnt/c/Users/yanbo/isaac_assets"  # Optional

# CUDA settings
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64"

# Performance tuning
export PHYSX_GPU_MAX_RIGID_CONTACT_COUNT=524288
export PHYSX_GPU_MAX_RIGID_PATCH_COUNT=163840
export PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY=2097152
export PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY=2097152

# Headless rendering
export DISPLAY=""
export VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/nvidia_icd.json"
ENV

# Load in bashrc
echo "source ~/.isaaclab_env" >> ~/.bashrc
source ~/.isaaclab_env
```

## Step 6: Update activation script

Add Isaac Lab environment to the venv activation:

```bash
cat >> /mnt/c/Users/yanbo/wSpace/cinebotRL/.venv_rl311/bin/activate <<'ACT'

# Isaac Lab environment
if [ -f ~/.isaaclab_env ]; then
    source ~/.isaaclab_env
fi
ACT
```

Or use the project's WSL activation script:
```bash
source scripts/wsl/activate_rl_env_wsl.sh
```

## Step 7: Verify installation

Test Isaac Lab import:
```bash
python -c "import isaaclab; print(f'Isaac Lab version: {isaaclab.__version__}')"
```

Test GPU access:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

Check Isaac Sim packages (if installed):
```bash
python -c "import omni; print('Omniverse runtime OK')"
```

## Step 8: Test with our custom task

Register and test the MobileMMTrackEE task:

```bash
cd /mnt/c/Users/yanbo/wSpace/cinebotRL

# Verify task registration
python -c "
import gymnasium as gym
from src.task_spec import register_isaac_lab_tasks
register_isaac_lab_tasks()
print('Available tasks:', [k for k in gym.envs.registry.keys() if 'MobileMM' in k])
"

# Quick environment test (single env, 10 steps)
python -c "
import torch
import gymnasium as gym
from src.task_spec import register_isaac_lab_tasks

register_isaac_lab_tasks()
env = gym.make('MobileMMTrackEE-v0', num_envs=1, headless=True)
env.reset()

for i in range(10):
    actions = torch.zeros(env.action_space.shape)
    obs, reward, done, truncated, info = env.step(actions)
    print(f'Step {i}: reward={reward.item():.3f}')

print('✓ Environment test passed')
env.close()
"
```

## Step 9: Run training

Start baseline training:
```bash
cd /mnt/c/Users/yanbo/wSpace/cinebotRL
source .venv_rl311/bin/activate

python scripts/reinforcement_learning/sb3/train.py \
    --task MobileMMTrackEE-v0 \
    --num_envs 1024 \
    --headless \
    --total_timesteps 10000000 \
    --checkpoint_freq 100000

# Monitor with TensorBoard
tensorboard --logdir logs/sb3/MobileMMTrackEE-v0
```

## Troubleshooting

### "ImportError: libcuda.so.1: cannot open shared object file"
```bash
# Add NVIDIA libraries to LD_LIBRARY_PATH
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH}"
```

### "Vulkan not available"
```bash
# Install Vulkan loader
sudo apt update
sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-tools

# Verify
vulkaninfo | grep "GPU id"
```

### "RuntimeError: CUDA out of memory"
Reduce number of environments:
```bash
python train.py --num_envs 512  # Instead of 1024
```

### "Cannot find end-effector link"
The environment will print a warning and fall back to the last body. Check USD structure:
```bash
python scripts/inspect_usd.py assets_own/usd/mobile_manipulator_PPR_base_corrected.usd
```

### Performance issues
- Enable PhysX GPU buffers (already in `~/.isaaclab_env`)
- Use headless mode: `--headless`
- Reduce render interval in task config
- Check GPU utilization: `watch -n 0.5 nvidia-smi`

## Performance Tips

1. **Maximize environments**: Start with 1024, increase to 2048-4096 if memory allows
2. **Headless mode**: Always use `--headless` for training
3. **Fast reset**: Ensure `reset_on_done=True` in env config
4. **GPU monitoring**: Watch `nvidia-smi` to ensure >90% GPU utilization
5. **Batch size**: Match PPO batch size to num_envs for efficiency

## Next Steps

- [ ] Test environment loading with 1 env
- [ ] Run short training (1M steps) to validate setup
- [ ] Load recorded trajectory: `--trajectory_file trajectoryToLearn/1_pull_world_scaled.json`
- [ ] Scale to 1024+ environments
- [ ] Set up wandb logging: `--wandb_project cinebotRL`
- [ ] Document baseline results

## References

- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [pip Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html)
- [Troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/setup/troubleshooting.html)
