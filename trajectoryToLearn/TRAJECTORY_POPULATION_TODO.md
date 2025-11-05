# Trajectory Curriculum - Population Guide

## Current Status

**Directory structure**: ✅ Created
**README documentation**: ✅ Complete
**Actual trajectories**: ⏳ TODO

## Quick Start Options

### Option 1: Use Existing Trajectories (Fastest)
For initial testing, can use symbolic links or copies from `world_json/`:

```powershell
# Stage 0: Copy a subset of chassis-friendly trajectories
# (Manually select ~20 easy trajectories based on criteria)

# Stage 1-3: Gradually add more until stage3 has all
```

### Option 2: Use Chassis-Only Filter (Recommended for 20M test)
```powershell
# Use existing filtering mechanism
--trajectory_dir trajectoryToLearn/world_json
--use_chassis_only  # Already filters to easier trajectories
```

### Option 3: Generate Staged Trajectories (Future work)
Use MATLAB to generate trajectories meeting each stage's criteria:
- Stage 0: Short, 0.4-0.6m reach, static
- Stage 1: Recovery drills (far start → near)
- Stage 2: Medium length, some behind-base
- Stage 3: All difficulties

## For Session 8h Initial Run

**Recommendation**: Use Option 2 (chassis-only) for 20M validation run
- Already tested and working
- Meets stage0_easy criteria reasonably well
- Can evaluate approach before investing in trajectory generation

**If 20M succeeds**: Generate proper staged trajectories for full 100M run

## Stage Selection Logic (Future)

In launcher, can add logic to select trajectory directory based on training step:
```powershell
if ($TrainingStep -lt 20000000) {
    $TrajectoryDir = "stage0_easy"
} elseif ($TrainingStep -lt 40000000) {
    $TrajectoryDir = "stage1_recovery"
} elseif ($TrainingStep -lt 70000000) {
    $TrajectoryDir = "stage2_moderate"
} else {
    $TrajectoryDir = "stage3_full"
}
```

For now, can manually specify per phase in launcher.
