# Windows Training Setup - Progress Summary

**Date**: October 15, 2025  
**Status**: ⚠️ In Progress - Isaac Sim initialization issue

---

## ✅ Completed

### 1. GPU Configuration
- ✅ Confirmed GPU setup:
  - Device 0: RTX 3090 (compute 8.6) - Training GPU
  - Device 1: Quadro P2000 (compute 6.1) - Display
- ✅ Implemented auto-detection for best GPU
- ✅ Removed WSL-specific workarounds (LD_LIBRARY_PATH, cuda:1 hardcoding)

### 2. Code Updates for Windows
- ✅ **test_mobile_mm_env.py**: Removed WSL CUDA library paths, added auto GPU detection
- ✅ **env.py**: Removed hardcoded `device="cuda:1"`, let AppLauncher handle device
- ✅ **train.py**: Added project path handling, updated for Windows
- ✅ **launch_training_windows.ps1**: Created convenient launcher script with GPU detection

### 3. Documentation
- ✅ **WINDOWS_TRAINING_QUICKSTART.md**: Complete quick start guide
- ✅ Updated comments to clarify Windows-native operation (no WSL hacks needed)

---

## ⚠️ Current Issue

### Isaac Sim Initialization Crash

**Symptom**:
```
Windows fatal exception: access violation
...
File "I:\isaaclab\_isaac_sim\kit\python\Lib\site-packages\ale_py\__init__.py", line 34 in <module>
```

**Root Cause**:
The `ale_py` (Atari Learning Environment) package crashes during initialization when Gymnasium tries to register Atari envs. This is a known issue with ale_py on Windows.

**Impact**:
- Isaac Lab AppLauncher fails to complete initialization
- Cannot create Isaac Lab environments
- Training cannot proceed

---

## 🔧 Solutions to Try

### Option 1: Disable Atari Environment Registration (Recommended)

Set environment variable to skip loading problematic Atari envs:

```powershell
# Add to launch script before Isaac Lab initialization
$env:GYMNASIUM_DISABLE_PLUGIN_ENTRYPOINTS = "1"
```

This tells Gymnasium not to auto-register plugin environments, which includes the problematic Atari envs.

### Option 2: Uninstall ale_py

If not needed for other tasks:

```powershell
cd I:\isaaclab
.\isaaclab.bat
pip uninstall ale_py shimmy
```

**Risk**: May break other Isaac Lab features that depend on these packages.

### Option 3: Use Isaac Lab's Own Examples First

Test with Isaac Lab's built-in environments to confirm setup works:

```powershell
cd I:\isaaclab
.\isaaclab.bat -p scripts/rl_games/train.py task=Isaac-Cartpole-Direct-v0 headless=true max_iterations=10
```

If this works, the issue is specific to Gymnasium initialization in our scripts.

### Option 4: Update Gymnasium/ale_py

Try updating to latest versions:

```powershell
cd I:\isaaclab
.\isaaclab.bat
pip install --upgrade gymnasium ale-py
```

---

## 📋 Next Steps

1. **Try Option 1** (Disable plugin entrypoints):
   - Update `launch_training_windows.ps1` to set `GYMNASIUM_DISABLE_PLUGIN_ENTRYPOINTS`
   - Test again

2. **If that fails, try Option 3**:
   - Verify Isaac Lab's own examples work
   - This confirms the base installation is functional

3. **Alternative approach**: Import Gymnasium differently
   - Delay gymnasium import until after Isaac Sim is fully initialized
   - Use direct environment creation instead of `gym.make()`

---

## 🎯 Goal

Get to this point:
```
[1/8] Initializing Isaac Lab... ✓
[2/8] Importing dependencies... ✓
[3/8] Registering MobileMMTrackEE-v0 task... ✓
[4/8] Creating environment... ✓
[5/8] Resetting environment... ✓
[6/8] Checking robot structure... ✓
[7/8] Running test steps... ✓
[8/8] Test completed successfully! ✓
```

---

## 📁 Files Modified

1. `scripts/test_mobile_mm_env.py` - Windows-native GPU detection
2. `scripts/reinforcement_learning/sb3/train.py` - Path handling for Windows
3. `src/rl_platform/tasks/mobile_mm/env.py` - Removed hardcoded device
4. `scripts/launch_training_windows.ps1` - NEW convenient launcher
5. `scripts/setup_project_in_isaaclab.ps1` - NEW setup script (has encoding issues)
6. `docs/setup/WINDOWS_TRAINING_QUICKSTART.md` - NEW documentation

---

## 💡 Key Insights

1. **Windows is simpler than WSL**: No special CUDA library paths, no device ordering workarounds
2. **Isaac Lab uses native Windows CUDA**: Vulkan and Warp work out-of-the-box
3. **ale_py is problematic on Windows**: Known issue, not related to our code
4. **Project path handling**: Need to add both PROJECT_ROOT and src/ to sys.path

---

## 🚀 When Working...

Training commands will be:

```powershell
# Quick test
.\scripts\launch_training_windows.ps1 -Test -Headless

# Small training run
.\scripts\launch_training_windows.ps1 -Headless -NumEnvs 64 -TotalTimesteps 100000

# Full training
.\scripts\launch_training_windows.ps1 -Headless -NumEnvs 1024 -TotalTimesteps 5000000
```

Expected performance on RTX 3090:
- ~100,000 steps/hour with 1024 parallel environments
- ~2-3 FPS per environment in headless mode
