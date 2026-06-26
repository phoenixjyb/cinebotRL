# Mobile Manipulator End-Effector Tracking Task

## Overview

The `MobileMMTrackEE-v0` task trains a mobile manipulator to track a reference trajectory with its end-effector while maintaining stability and avoiding obstacles. The active robot asset is `recomoProto2-1190`; use `RecomoProto2TrackEE-v0` as the explicit Proto2 task alias.

## Task Structure

```
src/rl_platform/tasks/mobile_mm/
├── __init__.py          # Package exports
├── config.py            # Configuration dataclasses
├── env.py               # Main environment (DirectRLEnv)
├── trajectories.py      # Reference trajectory generators
├── observations.py      # Observation composition
└── rewards.py           # Reward and penalty terms
```

## Features

### Trajectory Types
- **Circle**: Circular trajectory in the horizontal plane
- **Line**: Linear back-and-forth motion
- **Figure-8**: Lissajous curve (figure-eight pattern)
- **Recorded**: Playback from waypoint file (TODO)

### Observation Space

The observation includes:
- Base state: position, orientation, velocities (13 dims)
- Joint state: positions and velocities (12 dims for 6 joints)
- End-effector state: position, orientation, velocities (13 dims)
- Tracking error: position, quaternion, and axis-angle orientation errors (10 dims)
- Optional: Lookahead targets (configurable)
- Optional: Action history (configurable)
- Optional: Contact forces (TODO)
- Optional: Obstacle distances (TODO)

**Default observation dimension**: computed from config at startup.

### Action Space

8-dimensional continuous actions:
- 6 arm joint position targets
- 2 base commands (`base_vx`, `base_wz`)

The Proto2 USD also contains `base_joint_vy` and virtual gimbal joints
(`ee1_rot_z`, `ee1_rot_y`, `ee1_rot_x`). These joints are locked/passive in the
v1 policy and must not be added to the action space without revisiting
observations, rewards, and checkpoint compatibility.

### Reward Function

Weighted combination of:
- **Position tracking**: Exponential reward for EE position accuracy
- **Orientation tracking**: Exponential reward for EE orientation accuracy
- **Progress bonus**: Reward for reducing tracking error
- **Action magnitude penalty**: Discourages large control efforts
- **Action rate penalty**: Encourages smooth control
- **Collision penalty**: Penalizes contact forces (when sensors enabled)
- **Stability penalty**: Penalizes excessive base motion
- **Obstacle distance**: Rewards safe distances (when obstacles enabled)

## Configuration

### Example Configuration

```python
from rl_platform.tasks.mobile_mm import MobileMMTrackEEEnvCfg
from rl_platform.tasks.mobile_mm.config import MobileMMTrackConfig, TrajectoryConfig, RewardWeights

# Create custom configuration
task_config = MobileMMTrackConfig(
    trajectory=TrajectoryConfig(
        type="circle",
        amplitude=0.5,
        speed=0.2,
        height=1.0,
    ),
    rewards=RewardWeights(
        position_tracking=10.0,
        orientation_tracking=2.0,
        action_magnitude=0.01,
    ),
    episode_length_s=20.0,
    use_lookahead=True,
    lookahead_steps=3,
)

cfg = MobileMMTrackEEEnvCfg(task_config=task_config)
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `trajectory.type` | `"circle"` | Trajectory type |
| `trajectory.amplitude` | `0.5` | Trajectory size (meters) |
| `trajectory.speed` | `0.2` | Trajectory speed (m/s) |
| `rewards.position_tracking` | `10.0` | Position tracking weight |
| `rewards.orientation_tracking` | `2.0` | Orientation tracking weight |
| `episode_length_s` | `20.0` | Episode duration (seconds) |
| `decimation` | `4` | Control frequency divider |

## Training

### Basic Training Command

```bash
# Windows with Isaac Lab
I:\isaaclab\isaaclab-3090.bat -p scripts/reinforcement_learning/sb3/train.py \
    --task RecomoProto2TrackEE-v0 \
    --num_envs 1024 \
    --headless
```

### Training with Custom Parameters

```bash
I:\isaaclab\isaaclab-3090.bat -p scripts/reinforcement_learning/sb3/train.py \
    --task RecomoProto2TrackEE-v0 \
    --num_envs 2048 \
    --headless \
    --total_timesteps 5000000 \
    --learning_rate 3e-4 \
    --n_steps 2048 \
    --batch_size 512 \
    --wandb \
    --wandb_project cinebotrl
```

### Resume from Checkpoint

```bash
I:\isaaclab\isaaclab-3090.bat -p scripts/reinforcement_learning/sb3/train.py \
    --task RecomoProto2TrackEE-v0 \
    --num_envs 1024 \
    --headless \
    --checkpoint logs/sb3/mobile_mm_track_ee/20251013_143022/checkpoints/ppo_mobile_mm_1000000_steps
```

## Monitoring

### TensorBoard

```bash
# Windows
I:\isaaclab\isaaclab-3090.bat -p -m tensorboard --logdir logs/sb3

# WSL (if logs accessible)
source scripts/wsl/activate_rl_env_wsl.sh
tensorboard --logdir /mnt/c/Users/yanbo/wSpace/cinebotRL/logs/sb3
```

### W&B Integration

Enable Weights & Biases logging with `--wandb` flag:

```bash
--wandb --wandb_project cinebotrl
```

## TODO / Future Enhancements

### High Priority
- [ ] Integrate mobile base control (currently arm-only)
- [ ] Add contact sensors for collision detection
- [ ] Implement end-effector frame detection (currently using last body)
- [ ] Add recorded trajectory playback

### Medium Priority
- [ ] Obstacle spawning and randomization
- [ ] Domain randomization (mass, friction, torque limits)
- [ ] Curriculum learning (progressive difficulty)
- [ ] Multi-trajectory evaluation

### Low Priority
- [ ] ROS 2 topic publishing for monitoring
- [ ] Custom network architectures
- [ ] Recurrent policies (LSTM/GRU)
- [ ] Vision-based observations

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'rl_platform'`
- **Solution**: Make sure you're running from the project root and Isaac Lab can find the package

**Issue**: `FileNotFoundError: Missing USD file`
- **Solution**: Convert URDF to USD using Isaac Sim Asset Converter (see ROADMAP.md)

**Issue**: Environment crashes immediately
- **Solution**: Check that robot USD path is correct and assets are validated

**Issue**: Training is unstable
- **Solution**: Try reducing learning rate, increasing batch size, or adjusting reward weights

## Performance Baselines

(To be filled after initial training runs)

| Configuration | Success Rate | Avg Tracking Error | Training Time |
|---------------|--------------|-------------------|---------------|
| Circle (easy) | TBD | TBD | TBD |
| Figure-8 (hard) | TBD | TBD | TBD |

## Related Documentation

- [ROADMAP.md](../../../../ROADMAP.md) - Overall project plan
- [Windows Setup Guide](../../../../docs/setup/windows_setup_guide.md) - Windows configuration
- [WSL Setup Guide](../../../../docs/setup/wsl_setup_guide.md) - WSL configuration
- [Architecture Overview](../../../../docs/architecture/overview.md) - System design

---

**Last Updated**: 2025-10-13
**Status**: Initial implementation complete, ready for testing
