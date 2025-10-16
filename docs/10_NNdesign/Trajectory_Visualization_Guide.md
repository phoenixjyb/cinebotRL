# Trajectory Visualization in Isaac Sim

## 🎨 Visual Markers

The environment now includes real-time visual markers to help you see what the robot is trying to track!

### What You'll See:

**🔴 Red Spheres** - Target positions (where the robot should go)
- Updated every timestep
- Follows the reference trajectory (circle, line, figure-8, or recorded)
- Radius: 5cm

**🟢 Green Spheres** - End-effector positions (where the robot actually is)
- Shows actual gripper/tool position
- Updated every timestep
- Radius: 4cm

**Tracking Quality:**
- Green and red spheres **close together** = Good tracking ✓
- Green and red spheres **far apart** = Poor tracking, large error ✗

---

## 🚀 How to Use

### Method 1: Visualize During Training (Live)

```powershell
# Watch the robot learn in real-time (NO --headless flag!)
cd I:\isaaclab

.\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py `
  --task MobileMMTrackEE-v0 `
  --num_envs 4 `
  --total_timesteps 50000
  
# Note: Use fewer environments (1-4) for better observation
```

### Method 2: Visualize Trained Policy

```powershell
# Load a trained checkpoint and watch it perform
cd I:\isaaclab

.\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py `
  --task MobileMMTrackEE-v0 `
  --num_envs 1 `
  --checkpoint "C:\Users\yanbo\wSpace\cinebotRL\logs\sb3\mobilemmtrackee_v0\20251016_154914\final_model.zip" `
  --total_timesteps 5000
```

### Method 3: Using Convenience Script

```powershell
cd C:\Users\yanbo\wSpace\cinebotRL

# Automatically find latest checkpoint
.\scripts\visualize_policy.ps1 -Latest -NumEnvs 1

# Or specify checkpoint
.\scripts\visualize_policy.ps1 `
  -Checkpoint "logs\sb3\mobilemmtrackee_v0\20251016_154914\final_model.zip" `
  -NumEnvs 1
```

---

## 🎥 What to Look For

### Good Tracking (Trained Policy)
- 🟢 Green sphere follows 🔴 red sphere closely
- Smooth motion, no jerky movements
- Robot reaches targets before they move away
- Small gap between markers (~1-5cm)

### Poor Tracking (Untrained/Early Training)
- 🟢 Green sphere lags behind 🔴 red sphere
- Large gap between markers (>10cm)
- Erratic, jerky robot movements
- Robot overshoots or undershoots targets

### Different Trajectory Types

**Circle (Default):**
- Red sphere moves in a circular path at height 1.0m
- Good for testing smooth continuous tracking

**Line:**
- Red sphere moves back and forth along a line
- Tests direction reversal and stopping

**Figure-8:**
- Red sphere traces a figure-8 pattern
- More complex, tests acceleration changes

**Recorded:**
- Red sphere follows a pre-recorded cinematic camera path
- Most realistic for actual use cases

---

## 🔧 Technical Details

### Marker Configuration

Located in: `src/rl_platform/tasks/mobile_mm/env.py`

```python
# Target marker (red)
- Radius: 0.05m (5cm)
- Color: RGB(1.0, 0.0, 0.0) - Pure red
- Updated: Every environment step

# End-effector marker (green)
- Radius: 0.04m (4cm)  
- Color: RGB(0.0, 1.0, 0.0) - Pure green
- Updated: Every environment step
```

### Automatic Enable/Disable

- **GUI Mode**: Markers automatically enabled
- **Headless Mode**: Markers automatically disabled (no overhead)
- **Error Handling**: Silently disables if visualization fails

### Performance Impact

- **Negligible in GUI mode** (rendering already active)
- **Zero in headless mode** (visualization skipped)
- Safe for training with many environments

---

## 🎮 Camera Controls in Isaac Sim

When visualizing, you can control the camera:

- **Mouse Drag**: Rotate view
- **Mouse Scroll**: Zoom in/out
- **Middle Mouse + Drag**: Pan view
- **F**: Focus on selected object
- **ESC**: Exit simulation

**Tip:** Focus on the robot (press F) to keep it centered while it moves!

---

## 🐛 Troubleshooting

### No markers visible?

**Check 1:** Is Isaac Sim in GUI mode?
```powershell
# BAD - No GUI
--headless

# GOOD - Has GUI
# (just don't include --headless flag)
```

**Check 2:** Check terminal for visualization status
```
✓ Trajectory visualization markers enabled  # Good!
ℹ Visualization markers disabled             # Headless mode
```

**Check 3:** Camera might be far away
- Use mouse scroll to zoom in
- Press F to focus on robot

### Markers not updating?

- Markers update every step (~50 Hz)
- If frozen, simulation may be paused
- Check Isaac Sim play/pause button

### Want to hide markers?

Edit `env.py` line 342 and change:
```python
self._visualization_enabled = False  # Force disable
```

---

## 📊 Example Use Cases

### 1. Debugging Reward Function
- Watch if robot tries correct behavior
- See if target positions make sense
- Verify trajectory generation is correct

### 2. Evaluating Training Progress
- Compare early vs late training runs
- See tracking error visually
- Identify failure modes (overshoots, oscillations)

### 3. Recording Videos
- Use Isaac Sim's video recorder
- Document training improvements
- Create demos for presentations

### 4. Tuning Hyperparameters
- Visual feedback faster than TensorBoard
- Immediately see if changes help/hurt
- Spot unintended behaviors

---

## 🎬 Next Steps

After adding visualizations, you can:

1. **Test untrained policy** - See random behavior with markers
2. **Watch training live** - See gradual improvement over iterations
3. **Evaluate trained model** - Verify final performance visually
4. **Compare different trajectories** - Test circle vs figure-8 vs recorded

**Recommended first test:**
```powershell
# Simple circle tracking with 1 robot
cd I:\isaaclab
.\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py `
  --task MobileMMTrackEE-v0 --num_envs 1 --total_timesteps 5000
```

Look for:
- 🔴 Red sphere moving in circle
- 🟢 Green sphere following behind
- Gap slowly closing as policy learns

---

**Added:** 2025-10-16  
**Location:** `src/rl_platform/tasks/mobile_mm/env.py`  
**Lines:** 291-375 (visualization setup and update)  
**Enabled:** Automatically in GUI mode, disabled in headless
