# WSL2 + Isaac Sim CUDA Detection Issue

## Problem Summary

Isaac Sim cannot detect CUDA GPUs in WSL2 despite nvidia-smi working correctly.

### Symptoms
```
[Error] [omni.physx.tensors.plugin] CUDA error: initialization error
Warp CUDA error 100: no CUDA-capable device is detected
[Error] [carb.graphics-vulkan.plugin] No physical device is found
[Error] [omni.physx.plugin] No CUDA devices found
```

### System Configuration
- **GPUs**: 
  - GPU 0: Quadro P2000 (CUDA capability 6.1 - NOT supported by PyTorch 2.7+)
  - GPU 1: RTX 3090 (CUDA capability 8.6 - SUPPORTED)
- **OS**: WSL2 (Ubuntu)
- **CUDA**: 12.8 visible to nvidia-smi
- **PyTorch**: 2.7.0+cu128 (can detect RTX 3090 with `CUDA_DEVICE_ORDER=PCI_BUS_ID`)
- **Isaac Sim**: 5.0.0.0 (via Isaac Lab 2.2.0 pip)

## Root Cause

Isaac Sim's underlying libraries (**Warp** and **PhysX tensors**) fail to initialize CUDA in WSL2 environment. This is a known limitation of running Isaac Sim headless in WSL2.

The issue occurs at multiple levels:
1. **Warp CUDA initialization** fails to detect any CUDA devices
2. **Vulkan graphics backend** cannot find physical devices (WSLg limitation)
3. **PhysX GPU manager** cannot create CUDA context

## Attempted Solutions (All Failed)

### 1. ❌ Setting `CUDA_VISIBLE_DEVICES=1`
**Result**: Omniverse warns this causes issues with device enumeration

### 2. ❌ Setting `PHYSX_GPU=1`
**Result**: No effect, Warp fails before PhysX initialization

### 3. ❌ Setting `CUDA_DEVICE_ORDER=PCI_BUS_ID`
**Result**: Helps PyTorch find RTX 3090, but Isaac Sim still fails

### 4. ❌ Headless rendering flags
```bash
export CARB_GRAPHICS_HEADLESS=1
export OMNI_KIT_FORCE_HEADLESS=1
```
**Result**: Vulkan still attempts initialization and fails

## Current Status

**Isaac Sim headless training in WSL2 is NOT working** with the current system configuration.

## Recommended Solutions

### Option A: Use Windows Isaac Sim (RECOMMENDED)
Isaac Sim runs natively on Windows with full GPU support.

**Pros**:
- ✅ Full GPU acceleration (RTX 3090)
- ✅ Vulkan graphics works properly
- ✅ No WSL2 limitations
- ✅ Can use GUI for debugging

**Cons**:
- ❌ Windows environment may have different Python dependencies
- ❌ Cannot use WSL workflows directly

**Setup**:
1. Install Isaac Sim on Windows (already installed)
2. Copy project to Windows filesystem
3. Set up Python environment on Windows
4. Run training from Windows terminal

### Option B: Use Native Linux (if available)
If you have access to a native Linux machine with RTX 3090:

**Pros**:
- ✅ Full Isaac Sim support
- ✅ Linux workflows
- ✅ Better performance than WSL2

**Cons**:
- ❌ Requires dedicated Linux machine

### Option C: Docker Container with NVIDIA Runtime (Advanced)
Use official Isaac Sim Docker containers with `--gpus all`.

**Pros**:
- ✅ Isolated environment
- ✅ Official NVIDIA support
- ✅ May work better than WSL2 direct install

**Cons**:
- ❌ Complex setup
- ❌ Still may have WSL2 GPU passthrough issues

### Option D: Remote Linux Server (Best for production)
Use a cloud instance or remote Linux server with GPU.

**Pros**:
- ✅ Dedicated GPU resources
- ✅ Full Linux support
- ✅ Scalable for large-scale training

**Cons**:
- ❌ Cost
- ❌ Network latency

## Technical Details

### Why PyTorch Works but Isaac Sim Doesn't

**PyTorch CUDA**: Uses CUDA driver API directly, which WSL2 supports well
```python
import torch
torch.cuda.is_available()  # Returns True with RTX 3090
```

**Isaac Sim (Warp/PhysX)**: 
- Requires full CUDA runtime initialization
- Attempts Vulkan graphics initialization
- Needs proper GPU device enumeration through multiple layers
- WSL2's GPU passthrough may not expose all required CUDA/Vulkan features

### Warp Library Issue

The Warp library (NVIDIA's framework for GPU-accelerated simulation) is more sensitive to CUDA environment than PyTorch. It may require:
- Full CUDA Toolkit (not just runtime)
- Proper OpenGL/Vulkan context
- Specific CUDA device capabilities

### Device Number Confusion

The Quadro P2000 (device 0) being unsupported by modern PyTorch adds complexity:
- PyTorch ignores it (too old CUDA capability)
- Isaac Sim may try to use it first
- Device enumeration order varies by framework

## Next Steps

**For immediate testing**: Use Windows Isaac Sim

```powershell
# On Windows PowerShell
cd C:\Users\yanbo\wSpace\cinebotRL
# Activate Windows Python environment
python scripts\test_mobile_mm_env.py --num_envs 1 --steps 5
```

**For long-term**: Consider cloud GPU instance or native Linux machine for production training.

## References

- [Isaac Sim Docker Containers](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_container.html)
- [WSL2 CUDA Support](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- [Warp Documentation](https://nvidia.github.io/warp/)

## Update Log

- **2025-10-15**: Identified root cause as Warp/PhysX CUDA initialization failure in WSL2
- **2025-10-15**: Confirmed PyTorch can access RTX 3090 with `CUDA_DEVICE_ORDER=PCI_BUS_ID`
- **2025-10-15**: Multiple attempts to configure Isaac Sim for WSL2 failed
- **2025-10-15**: Recommended switching to Windows for immediate testing
