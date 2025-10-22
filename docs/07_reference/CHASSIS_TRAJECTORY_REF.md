# Quick Reference: Testing Chassis-Required Trajectories

## 🚀 One-Line Command

```bash
cd I:\isaaclab && .\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\test_chassis_trajectories.py
```

## 📊 What You'll See

### Good (Base Working) ✅
```
Step    50 | Reward:   +45.23 | Base: vx=+0.35 vy=-0.12 ω=+0.08
  ✓ Episode 1 complete (env 0) | reward: +6451.23, base traveled: 2.85m
```

### Bad (Base Frozen) ❌
```
Step    50 | Reward:  -755.61 | Base: vx=+0.00 vy=+0.00 ω=+0.00
  ✓ Episode 1 complete (env 0) | reward: -755061.00, base traveled: 0.02m
```

## 🎯 Success Criteria

| Metric | Target |
|--------|--------|
| Base vx/vy | Non-zero (0.2-0.5 m/s) |
| Distance traveled | > 2.0m per episode |
| Episode reward | Positive (+1K to +10K) |
| Visual movement | Platform moves, not just arm |

## 📝 Files Created

1. **`scripts/test_chassis_trajectories.py`** - Main test script
2. **`scripts/analyze_trajectories.py`** - Analysis tool
3. **`docs/TRAJECTORY_ANALYSIS_SUMMARY.md`** - Full analysis
4. **`docs/TESTING_RECORDED_TRAJECTORIES.md`** - Complete guide
5. **`trajectory_analysis_results.csv`** - All 1,038 trajectories data
6. **`chassis_required_indices.txt`** - 519 chassis-required indices

## 🔧 Quick Options

```bash
# More trajectories
.\isaaclab.bat -p ... --num 20 --envs 8

# With trained model
.\isaaclab.bat -p ... --checkpoint logs/final_model.zip

# Headless (no GUI)
.\isaaclab.bat -p ... --headless
```

## 📈 Key Findings from Analysis

- **519 / 1,038 trajectories (50%)** require chassis movement
- **X change threshold**: 2.0m (longitudinal direction)
- **Top types**: arc_left_push, push, orbit_left, orbit_right, approach
- **Average X change**: 2.089m (median: 2.002m)
- **Max X change**: 3.000m (237 trajectories)

## 🎬 What's Happening

The test:
1. Loads top 10 most challenging trajectories (3.0m X change)
2. Runs 4 parallel environments for efficiency  
3. Shows real-time base velocity diagnostics
4. Reports distance traveled and rewards
5. Proves visually that base CAN move

This is your **proof that the base movement fix works**! 🎉
