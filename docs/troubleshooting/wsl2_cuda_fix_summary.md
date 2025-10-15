# WSL2 Isaac Sim CUDA Fix - SOLVED! ✅

## Date: October 15, 2025

## Problem Summary
Isaac Sim could not detect CUDA GPUs in WSL2, despite nvidia-smi showing RTX 3090.

## Root Causes Identified

### 1. Missing WSL CUDA Library Path
**Symptom:**
```
Warp CUDA error 100: no CUDA-capable device is detected
```

**Cause:** Warp (Isaac Sim's GPU tensor library) couldn't find CUDA driver stub in WSL2.

**Solution:**
```bash
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
```

**Why:** WSL2's CUDA driver stub is in `/usr/lib/wsl/lib`, not the standard CUDA paths.

### 2. Wrong GPU Device Selection
**Symptom:**
```
[Warning] [omni.physx.plugin] PhysX warning: Minimum GPU compute capability 7.0 is required
[Error] [omni.physx.plugin] Failed to create Cuda Context Manager
```

**Cause:** Isaac Sim defaulted to device 0 (Quadro P2000 with compute capability 6.1 - unsupported).

**Solution:**
```python
AppLauncher(device="cuda:1")  # RTX 3090
```

**Why:** System has 2 GPUs:
- Device 0: Quadro P2000 (sm_61 - not supported by PyTorch 2.7+)
- Device 1: RTX 3090 (sm_86 - supported ✅)

### 3. Wrong Import Paths
**Symptom:**
```
ModuleNotFoundError: No module named 'omni.isaac.lab'
TypeError: object.__init__() takes exactly one argument
```

**Cause:** Isaac Lab 2.2.0 pip package doesn't use `omni.isaac.lab` namespace.

**Solution:**
```python
# WRONG (old Isaac Lab):
from omni.isaac.lab.envs import DirectRLEnv

# CORRECT (Isaac Lab 2.2.0 pip):
from isaaclab.envs import DirectRLEnv
from isaaclab.actuators import ImplicitActuatorCfg
```

## Complete Solution

### Step 1: Create Wrapper Script
File: `scripts/run_with_wsl_cuda.sh`

```bash
#!/bin/bash
# Wrapper script for Isaac Sim in WSL2

# Activate venv if not active
if [[ -z "$VIRTUAL_ENV" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
    source "$PROJECT_ROOT/.venv_rl311/bin/activate"
fi

# CRITICAL: Add WSL CUDA library path BEFORE Python starts
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"

# Set device ordering for consistent numbering
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

# Accept EULA
export OMNI_KIT_ACCEPT_EULA="yes"
export ACCEPT_EULA="YES"

# Run command
exec "$@"
```

### Step 2: Use RTX 3090 in Code
File: `scripts/test_mobile_mm_env.py`

```python
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(
    headless=True,
    enable_cameras=False,
    device="cuda:1",  # RTX 3090!
)
```

### Step 3: Fix Imports
File: `src/rl_platform/tasks/mobile_mm/env.py`

```python
# Isaac Lab 2.2.0 pip package imports
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
import isaaclab.sim as sim_utils
```

### Step 4: Run Everything Through Wrapper
```bash
# ALWAYS use the wrapper script!
./scripts/run_with_wsl_cuda.sh python scripts/test_mobile_mm_env.py --num_envs 1
```

## Verification

### CUDA Detection ✅
```
Warp 1.9.1 initialized:
   CUDA Toolkit 12.8, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA GeForce RTX 3090"  ← Found!
     "cuda:1"   : "Quadro P2000"
```

### PhysX GPU Selection ✅
```
[Info] [omni.physx.plugin] Using CUDA device ordinal 1.  ← RTX 3090!
```

### Isaac Lab Imports ✅
```python
from isaaclab.envs import DirectRLEnv  # Works!
```

## Critical Device ID Warning ⚠️

**THREE different device numbering systems exist:**

| System | Device 0 | Device 1 |
|--------|----------|----------|
| **nvidia-smi** | Quadro P2000 | RTX 3090 |
| **Warp (with PCI_BUS_ID)** | RTX 3090 | Quadro P2000 |
| **Isaac Sim/PhysX** | Quadro P2000 | RTX 3090 |

**ALWAYS use Isaac Sim's numbering:**
- `device="cuda:1"` → RTX 3090 ✅
- `device="cuda:0"` → Quadro P2000 ❌

## Remaining Issues

### 1. Vulkan Errors (HARMLESS) ⚠️
```
[Error] [carb.graphics-vulkan.plugin] No physical device is found
```
**Status:** Expected in WSL2 headless mode. Can be ignored - we don't need rendering!

### 2. Warp UUID Errors (HARMLESS) ⚠️
```
Warp CUDA error: Failed to get driver entry point 'cuDeviceGetUuid'
Warp CUDA error 36: API call is not supported in the installed CUDA driver
```
**Status:** WSL2 CUDA stub doesn't support all APIs. Doesn't affect physics simulation.

### 3. Configuration Validation (TO FIX) ❌
```
TypeError: Missing values detected in object MobileMMTrackEEEnvCfg for the following fields:
  - scene.robot.prim_path
  - scene.ground.prim_path
  - observation_space
  - action_space
```
**Status:** Need to complete environment configuration (next step).

## Success Criteria Met

- ✅ CUDA detection working
- ✅ RTX 3090 selected correctly
- ✅ PhysX initializing on correct GPU
- ✅ Isaac Lab imports working
- ✅ Environment class loading
- ⏳ Configuration validation (in progress)

## Usage for All Future Commands

**MANDATORY:** Always use the wrapper script:

```bash
# Training
./scripts/run_with_wsl_cuda.sh python scripts/train.py

# Testing
./scripts/run_with_wsl_cuda.sh python scripts/test_mobile_mm_env.py

# Any Isaac Sim script
./scripts/run_with_wsl_cuda.sh python your_script.py
```

**DO NOT run Isaac Sim scripts directly without the wrapper!**

## Files Modified

1. `scripts/run_with_wsl_cuda.sh` - Created (WSL CUDA setup)
2. `scripts/test_mobile_mm_env.py` - Updated (device="cuda:1")
3. `src/rl_platform/tasks/mobile_mm/env.py` - Updated (import paths)
4. `docs/troubleshooting/wsl2_cuda_isaac_sim.md` - Updated

## Lessons Learned

1. **LD_LIBRARY_PATH matters:** Must be set BEFORE Python starts (hence wrapper script)
2. **Device numbering is confusing:** Always verify which device ID system you're using
3. **Import paths changed:** Isaac Lab 2.2.0 pip uses `isaaclab` not `omni.isaac.lab`
4. **Vulkan errors are OK:** Headless physics doesn't need graphics
5. **Test incrementally:** Fix one error at a time, verify each fix

## Next Steps

1. Complete environment configuration (prim paths, spaces)
2. Test robot spawning and EE tracking
3. Verify all 14 reward components
4. Run baseline training

---

**Document created:** 2025-10-15  
**Last updated:** 2025-10-15  
**Status:** CUDA/GPU issues RESOLVED ✅
