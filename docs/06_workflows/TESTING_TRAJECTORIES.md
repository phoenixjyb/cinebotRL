# Testing Recorded Trajectories - Visual Validation

This guide explains how to visually test your robot's ability to follow recorded trajectories, specifically those requiring significant chassis (base) movement.

## 📋 Background

From the trajectory analysis, we identified **519 out of 1,038 trajectories (50%)** that require chassis movement (X-direction change ≥ 2.0m). These are critical test cases for validating that your base movement fixes are working.

## 🎯 Quick Start

### Option 1: Simple Test (Recommended)

Test the top 10 most challenging trajectories with 4 parallel environments:

```bash
cd I:\isaaclab
.\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\test_chassis_trajectories.py
```

### Option 2: Custom Number of Trajectories

```bash
# Test top 20 trajectories with 8 environments
.\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\test_chassis_trajectories.py --num 20 --envs 8
```

### Option 3: Test with Trained Model

```bash
# Use your trained model checkpoint
.\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\test_chassis_trajectories.py \
    --checkpoint logs/sb3/MobileMMTrackEE-v0/final_model.zip \
    --num 10 --envs 4
```

## 📊 What to Look For

### ✅ **Good Signs (Base Movement Working)**:

1. **Base Velocity Changes**: See non-zero `vx`, `vy`, `ω` (yaw rate) in diagnostics
2. **Visual Movement**: Robot platform visibly moves forward/backward and rotates
3. **Positive Rewards**: Episodes ending with positive rewards (tracking successful)
4. **Distance Traveled**: `base traveled` showing > 1.0m per episode

Example good output:
```
Step    50 | Reward:   +45.23 | Base: vx=+0.35 vy=-0.12 ω=+0.08
  ✓ Episode 1 complete (env 0) | reward: +6451.23, base traveled: 2.85m
```

### ❌ **Bad Signs (Base Still Frozen)**:

1. **Zero Velocities**: `vx=0.00 vy=0.00 ω=0.00` consistently
2. **No Visual Movement**: Robot platform stationary, only arm moving
3. **Negative Rewards**: Large negative rewards indicating tracking failure
4. **Zero Distance**: `base traveled: 0.00m` or very small (< 0.1m)

Example bad output:
```
Step    50 | Reward:  -755.61 | Base: vx=+0.00 vy=+0.00 ω=+0.00
  ✓ Episode 1 complete (env 0) | reward: -755061.00, base traveled: 0.02m
```

## 🎬 Trajectory Types Being Tested

The test uses these challenging trajectory types (all require chassis movement):

| Type | Count | Avg X Change | Description |
|------|-------|--------------|-------------|
| **arc_left_push** | 100 | 2.893m | Arc trajectory with left push motion |
| **push** | 200 | 2.910m | Forward pushing motion |
| **orbit_left** | 100 | 2.688m | Orbiting motion to the left |
| **orbit_right** | 100 | 2.630m | Orbiting motion to the right |
| **approach** (scene_4) | 11 | 2.990m | Smooth approach trajectories |

## 🔧 Script Parameters

### `test_chassis_trajectories.py`

```bash
python scripts/test_chassis_trajectories.py [OPTIONS]

Options:
  --num INT           Number of trajectories to test (default: 10)
  --envs INT          Number of parallel environments (default: 4)
  --checkpoint PATH   Path to trained model .zip file (optional)
  --headless          Run without GUI (for CI/CD)
```

### Using Specific Trajectory Indices

If you want to test specific trajectories, edit the `CHASSIS_REQUIRED_INDICES` list in the script (first 100 shown, but 519 total available).

## 📈 Performance Expectations

### Random Policy (Baseline):
- **Rewards**: Highly variable, often negative
- **Base Movement**: Should still occur if action scaling works
- **Purpose**: Validate that base CAN move (not policy quality)

### Trained Policy:
- **Rewards**: Should be positive for most episodes (75%+ success rate)
- **Base Movement**: Coordinated with arm for smooth tracking
- **Purpose**: Validate both base movement AND policy quality

## 🐛 Troubleshooting

### Issue: "No trajectory files found"

**Solution**: Check that your trajectory directory exists:
```bash
ls trajectoryToLearn/world_json
```

Should show folders like `scene_1/`, `scene_2/`, etc.

### Issue: "Module import errors"

**Solution**: Make sure you're running through Isaac Lab:
```bash
# ❌ Wrong
python scripts/test_chassis_trajectories.py

# ✅ Correct  
cd I:\isaaclab
.\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\test_chassis_trajectories.py
```

### Issue: Simulation very slow

**Solution**: Reduce number of environments:
```bash
.\isaaclab.bat -p ... --envs 2  # Instead of 4 or 8
```

### Issue: Can't see trajectories clearly

**Solution**: The red spheres (targets) and green spheres (end-effector) should be visible. If not:
1. Zoom in using mouse wheel
2. Adjust camera angle by dragging
3. Check that visualization markers are enabled in env config

## 📝 Analyzing Results

### Collecting Data

Redirect output to file for analysis:
```bash
.\isaaclab.bat -p ... > test_results.txt 2>&1
```

### Key Metrics to Track

1. **Average Reward per Episode**: Should improve with trained model
2. **Base Distance Traveled**: Should be > 2.0m for chassis-required trajectories
3. **Episode Completion Rate**: % of episodes running to max steps (good) vs early termination (bad)
4. **Base Velocity Statistics**: Mean, max, variance of vx, vy, ω

### Comparison Table

| Metric | Before Fix | After Fix | Target |
|--------|-----------|-----------|--------|
| Avg Base Distance | < 0.1m | > 2.0m | 2.5m+ |
| Avg vx (forward) | ~0.0 m/s | 0.2-0.5 m/s | 0.3 m/s |
| Training FPS | < 1000 | 8000-9000 | 8000+ |
| Episode Reward | < -100K | > +1K | +5K+ |

## 🎓 Advanced Usage

### Test Only Specific Trajectory Type

Edit the indices list to focus on one type (e.g., only `arc_left_push`):

```python
# In test_chassis_trajectories.py
CHASSIS_REQUIRED_INDICES = list(range(0, 100))  # First 100 = arc_left_push
```

### Record Video

```bash
# Enable video recording (if implemented)
.\isaaclab.bat -p ... --record_video --video_dir videos/chassis_test
```

### Batch Testing

Create a batch script to test multiple configurations:

```bash
# test_all_configs.sh
for num in 10 20 50 100; do
    echo "Testing with $num trajectories..."
    .\isaaclab.bat -p ... --num $num --envs 4 > results_$num.txt
done
```

## 📚 Related Documentation

- [Trajectory Analysis Summary](../docs/TRAJECTORY_ANALYSIS_SUMMARY.md) - Full analysis results
- [Base Movement Analysis](../docs/BASE_MOVEMENT_COMPREHENSIVE_ANALYSIS.md) - Original problem diagnosis
- [Visualization Guide](../docs/VISUALIZATION_GUIDE.md) - Understanding what you see

## 🎯 Success Criteria

Your base movement fix is successful if:

✅ Base velocities are non-zero and coordinated with arm motion
✅ Visual observation shows platform moving (not just arm)
✅ Base distance traveled > 2.0m for chassis-required trajectories
✅ Training FPS remains high (8000+ FPS)
✅ Episode rewards are positive for trained policy (75%+ success rate)
✅ No "early stopping at step 0" during training

Good luck testing! 🚀
