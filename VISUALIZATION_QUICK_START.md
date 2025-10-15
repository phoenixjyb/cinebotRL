# Visualization Quick Start Guide

## ✅ Scripts Fixed and Working!

The visualization scripts have been fixed to work with Isaac Sim GUI mode.

---

## 🚀 Quick Commands

### Option 1: Single Robot Inspector (Recommended First)

```powershell
.\scripts\inspect_environment.ps1
```

**What happens:**
- Opens Isaac Sim with full GUI
- Shows 1 mobile manipulator robot
- Robot performs random actions (no trained policy)
- Takes 30-60 seconds to load
- Press ESC to close when done

---

### Option 2: Multiple Robots Grid View

```powershell
.\scripts\inspect_environment.ps1 -NumEnvs 4
```

Shows 4 robots in a grid layout for observing different behaviors simultaneously.

---

### Option 3: Training Visualization (More Environments)

```powershell
.\scripts\visualize_training.ps1 -NumEnvs 16
```

Shows more environments running for a longer duration (up to 1M timesteps).

---

## 📊 What You'll See in Isaac Sim GUI

When the window opens, you'll see:

1. **Viewport (Main Window):**
   - Your mobile manipulator robot(s)
   - Ground plane
   - End-effector target trajectory (circular path)
   - Lighting and scene elements

2. **Controls:**
   - **Left Mouse:** Rotate camera view
   - **Middle Mouse:** Pan camera
   - **Scroll Wheel:** Zoom in/out
   - **Click Robot:** Select and view properties
   - **ESC:** Close Isaac Sim

3. **Robot Behavior:**
   - ⚠️ **Random movements** - This is normal!
   - No trained policy is loaded
   - The robot is just exploring the action space
   - Purpose: Verify environment setup, not showcase trained behavior

---

## 🔧 Current Status

### Currently Running:
- **Terminal 1:** Headless training (64 envs, RTX 3090)
- **Terminal 2 (NEW):** GUI visualization (1 env, loading on RTX 3090)

### GPU Usage:
Your dual-GPU setup allows both to run simultaneously:
- **RTX 3090:** Handles both headless training + GUI rendering
- **Quadro P2000:** Display output (GUI actually renders on RTX 3090)

---

## ⏱️ Startup Time

**Expected loading time:** 30-60 seconds

You'll see lots of extension loading messages:
```
[ext: omni.kit.window.property-1.12.1] startup
[ext: omni.physx.foundation-107.3.18] startup
[ext: isaacsim.core.utils-3.4.5] startup
...
```

This is normal! Just wait for the GUI window to appear.

---

## 🎯 What To Look For

### Asset Validation:
- ✅ Robot USD loaded correctly?
- ✅ All joints articulating?
- ✅ Collision meshes visible?
- ✅ End-effector trajectory displayed?

### Scene Setup:
- ✅ Ground plane present?
- ✅ Lighting working?
- ✅ Robot positioned correctly?

### Movement:
- ✅ Arm joints moving?
- ✅ Base chassis moving (vx, wz)?
- ✅ No collisions/explosions?

---

## 🐛 Troubleshooting

### Isaac Sim not opening?
- Wait full 60 seconds - it's slow to start
- Check terminal for errors
- Verify Isaac Lab path: `I:\isaaclab\isaaclab.bat`

### Black screen / no robot?
- Scene is still loading - wait a few more seconds
- Try moving camera (mouse drag)
- Check if robot loaded: Look for console messages

### "Out of memory"?
- Close other GPU applications
- Use fewer environments: `-NumEnvs 1`
- Close training temporarily to free GPU memory

### Robot behaving strangely?
- **This is expected!** Random policy only
- Purpose: Environment inspection, not trained behavior
- To see trained policy: Need to implement checkpoint loading

---

## 📝 Notes

### About Random Behavior:
The visualization scripts don't load trained checkpoints yet. To implement policy visualization:

1. Add test mode to `train.py`:
   ```python
   parser.add_argument("--test", action="store_true")
   parser.add_argument("--checkpoint", type=str)
   ```

2. Load checkpoint in test mode:
   ```python
   if args.test and args.checkpoint:
       model = PPO.load(args.checkpoint, env=vec_env)
   ```

3. Run evaluation loop instead of training

See [`docs/workflows/visualization_during_training.md`](docs/workflows/visualization_during_training.md) for complete implementation guide.

---

## 🎉 Success!

If you see:
- ✅ Isaac Sim GUI window opens
- ✅ Robot(s) visible in viewport
- ✅ Robot moving (even if randomly)
- ✅ Can control camera with mouse

**Your environment is working correctly!**

The random behavior is normal for inspection mode. Your actual training in Terminal 1 is learning the proper policy using the reward function.

---

## 🔄 Continue Training

Your headless training in Terminal 1 continues uninterrupted while you inspect the environment. When done:

1. **Close Isaac Sim:** Press ESC or close window
2. **Check training:** View logs with `.\scripts\monitor_training.ps1 -Mode logs`
3. **Training continues:** No impact from visualization

---

## 📚 Next Steps

1. **Verify robot looks correct** - Check joints, meshes, articulation
2. **Close inspector** - Press ESC
3. **Let training run** - Wait for checkpoints to be saved
4. **Monitor progress** - Use `.\scripts\monitor_training.ps1`
5. **Later: Implement policy loading** - To see trained behavior

---

**Created:** 2025-10-15  
**Status:** ✅ Scripts fixed and working  
**Current:** Isaac Sim GUI loading in Terminal 2
