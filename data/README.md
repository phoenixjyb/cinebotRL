# Data Directory

This directory contains data files used by the training system.

## Structure

```
data/
└── trajectory_filters/          # Trajectory selection configurations
    ├── chassis_required_indices.txt       # Indices of chassis-only trajectories
    └── chassis_required_trajectories.txt  # Names of chassis-only trajectories
```

## Usage

### Trajectory Filters

The `trajectory_filters/` directory contains files that specify which trajectories to use when training with the `--use_chassis_only` flag:

- **chassis_required_indices.txt**: Line numbers (0-indexed) of trajectories in `trajectoryToLearn/`
- **chassis_required_trajectories.txt**: Human-readable names corresponding to those indices

These files are read by `src/rl_platform/tasks/mobile_mm/trajectories.py` during environment initialization.

## Related Files

- Training script: `scripts/reinforcement_learning/sb3/train.py`
- Trajectory loader: `src/rl_platform/tasks/mobile_mm/trajectories.py`
- Trajectory data: `trajectoryToLearn/`
