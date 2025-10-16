# GUI Visualization Options for WSL Headless Training

## Overview

You're training **headless in WSL** (Option 1 - fast, simple), but want **GUI visualization on Windows** to confirm behavior and debug issues.

**Three approaches ranked by complexity:**

---

## Option A: Live ROS2 Streaming (Recommended) 🚀

**Best for**: Real-time monitoring during training

### How It Works

```
┌─────────────────────────────────┐
│        WSL (Training)           │
│  ┌───────────────────────┐     │
│  │ Isaac Lab Headless    │     │
│  │ - Physics simulation  │     │
│  │ - RL training         │     │
│  └──────────┬────────────┘     │
│             │                   │
│  ┌──────────▼────────────┐     │
│  │ ROS2 Publisher        │     │
│  │ - /robot/joint_states │     │
│  │ - /robot/ee_pose      │     │
│  │ - /trajectory/target  │     │
│  │ - /reward_components  │     │
│  └──────────┬────────────┘     │
└─────────────┼─────────────────┘
              │ Fast DDS Network
              │ (Domain ID: 55)
┌─────────────┼─────────────────┐
│      Windows (Visualization)  │
│  ┌──────────▼────────────┐    │
│  │ ROS2 Subscriber       │    │
│  └──────────┬────────────┘    │
│             │                 │
│  ┌──────────▼────────────┐    │
│  │ Isaac Sim GUI         │    │
│  │ - Subscribes to ROS2  │    │
│  │ - Displays robot      │    │
│  │ - Shows trajectory    │    │
│  └───────────────────────┘    │
└───────────────────────────────┘
```

### Advantages ✅
- **Real-time**: See robot move as it trains
- **Low latency**: Fast DDS is efficient
- **Selective**: Choose which env to visualize (env_id=0 only)
- **Non-intrusive**: Doesn't slow down training
- **Debug-friendly**: See failures live

### Disadvantages ⚠️
- Requires ROS2 setup (already done!)
- Network setup (already configured!)
- Needs ROS2 bridge code in environment

### Implementation

#### 1. Add ROS2 Publisher to Environment

```python
# src/rl_platform/tasks/mobile_mm/ros2_bridge.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped

class TrainingVisualizer(Node):
    def __init__(self):
        super().__init__('training_visualizer')
        self.joint_pub = self.create_publisher(JointState, '/robot/joint_states', 10)
        self.ee_pub = self.create_publisher(PoseStamped, '/robot/ee_pose', 10)
        self.target_pub = self.create_publisher(PoseStamped, '/trajectory/target', 10)
        
    def publish_state(self, joint_pos, ee_pose, target_pose):
        # Only publish env 0 to avoid spam
        joint_msg = JointState()
        joint_msg.position = joint_pos[0].cpu().tolist()
        self.joint_pub.publish(joint_msg)
        
        # ... publish ee_pose and target_pose
```

#### 2. Enable in Training Script

```python
# scripts/reinforcement_learning/sb3/train.py
parser.add_argument('--enable_ros2_viz', action='store_true', 
                    help='Enable ROS2 publishing for Windows visualization')

if args.enable_ros2_viz:
    from rl_platform.tasks.mobile_mm.ros2_bridge import TrainingVisualizer
    viz = TrainingVisualizer()
    
# In training loop
if args.enable_ros2_viz:
    viz.publish_state(joint_pos, ee_pose, target_pose)
```

#### 3. Visualize on Windows

```powershell
# Windows Terminal
cd I:\isaaclab
isaaclab.bat

# In Isaac Sim Python console
import omni.isaac.ros2_bridge
# Subscribe to /robot/joint_states
# Apply to articulation in scene
```

---

## Option B: Checkpoint Replay (Simpler) 📦

**Best for**: Periodic inspection, debugging specific behaviors

### How It Works

```
┌─────────────────────────────────┐
│        WSL (Training)           │
│  ┌───────────────────────┐     │
│  │ Train headlessly      │     │
│  │ Save checkpoints      │     │
│  │   model_100k.zip      │     │
│  │   model_500k.zip      │     │
│  │   best_model.zip      │     │
│  └───────────────────────┘     │
└─────────────────────────────────┘
              │
              │ Shared filesystem
              │ /mnt/c/Users/.../checkpoints/
              │
┌─────────────▼─────────────────┐
│      Windows (Replay)         │
│  ┌───────────────────────┐   │
│  │ Isaac Sim GUI         │   │
│  │ Load checkpoint       │   │
│  │ Run trained policy    │   │
│  │ Record video          │   │
│  └───────────────────────┘   │
└───────────────────────────────┘
```

### Advantages ✅
- **Simple**: No ROS2 needed
- **Stable**: Load checkpoint anytime
- **Repeatable**: Test same policy multiple times
- **Video recording**: Easy to capture demos
- **Offline**: No network dependency

### Disadvantages ⚠️
- **Not real-time**: Can't see training live
- **Manual**: Have to stop, load, visualize
- **Delayed**: Only see behavior after training

### Implementation

#### 1. Save Checkpoints Regularly

```python
# scripts/reinforcement_learning/sb3/train.py
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path='checkpoints/',
    name_prefix='model'
)

model.learn(callback=[checkpoint_callback])
```

#### 2. Create Replay Script

```python
# scripts/replay_checkpoint.py (Windows)
"""Replay trained policy in Isaac Sim GUI."""
import argparse
from stable_baselines3 import PPO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to .zip checkpoint')
    parser.add_argument('--num_envs', type=int, default=1)
    parser.add_argument('--gui', action='store_true', default=True)
    args = parser.parse_args()
    
    # Load policy
    model = PPO.load(args.checkpoint)
    
    # Create environment with GUI
    env = gym.make('MobileMMTrackEE-v0', num_envs=args.num_envs, headless=False)
    
    # Run policy
    obs = env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done.any():
            obs = env.reset()
    
    env.close()

if __name__ == '__main__':
    main()
```

#### 3. Visualize on Windows

```powershell
# Windows Terminal
cd I:\isaaclab
isaaclab.bat

# Run replay
python I:\wSpace\cinebotRL\scripts\replay_checkpoint.py `
    --checkpoint I:\wSpace\cinebotRL\checkpoints\model_100000_steps.zip `
    --gui
```

---

## Option C: ROS2 Bag Recording (Hybrid) 💾

**Best for**: Post-training analysis, sharing results

### How It Works

```
┌─────────────────────────────────┐
│        WSL (Training)           │
│  ┌───────────────────────┐     │
│  │ Train headlessly      │     │
│  │ Publish ROS2 topics   │     │
│  └──────────┬────────────┘     │
│             │                   │
│  ┌──────────▼────────────┐     │
│  │ ros2 bag record       │     │
│  │ Save to bag file      │     │
│  │   training_run.db3    │     │
│  └───────────────────────┘     │
└─────────────────────────────────┘
              │
              │ Copy bag file to Windows
              │
┌─────────────▼─────────────────┐
│      Windows (Playback)       │
│  ┌───────────────────────┐   │
│  │ ros2 bag play         │   │
│  └──────────┬────────────┘   │
│             │                 │
│  ┌──────────▼────────────┐   │
│  │ Isaac Sim GUI         │   │
│  │ Subscribes to topics  │   │
│  │ Replays recording     │   │
│  └───────────────────────┘   │
└───────────────────────────────┘
```

### Advantages ✅
- **Complete data**: Record everything
- **Portable**: Share bag files
- **Analyzable**: Use `ros2 bag info`, `rqt_bag`
- **Time control**: Pause, slow-motion, rewind
- **No training overhead**: Record separately

### Disadvantages ⚠️
- **Large files**: Can be gigabytes for long training
- **Post-hoc**: Not real-time
- **ROS2 dependency**: Need ROS2 on both sides

### Implementation

#### 1. Record During Training (WSL)

```bash
# WSL Terminal 1: Train
source .venv_rl311/bin/activate
export OMNI_KIT_ACCEPT_EULA=yes
python scripts/reinforcement_learning/sb3/train.py --enable_ros2_viz

# WSL Terminal 2: Record
source scripts/wsl/setup_ros2_only.sh
ros2 bag record \
    /robot/joint_states \
    /robot/ee_pose \
    /trajectory/target \
    /reward_components \
    -o training_run_$(date +%Y%m%d_%H%M%S)
```

#### 2. Copy to Windows

```bash
# WSL
cp training_run_*.db3 /mnt/i/wSpace/cinebotRL/recordings/
```

#### 3. Replay on Windows

```powershell
# Windows Terminal 1: Play bag
cd I:\ros2
.\setup.ps1
ros2 bag play I:\wSpace\cinebotRL\recordings\training_run_*.db3

# Windows Terminal 2: Visualize
cd I:\isaaclab
isaaclab.bat
# Subscribe to topics in Isaac Sim
```

---

## Comparison Table

| Feature | Live ROS2 | Checkpoint Replay | Bag Recording |
|---------|-----------|-------------------|---------------|
| **Real-time** | ✅ Yes | ❌ No | ❌ No |
| **Complexity** | Medium | Low | Medium |
| **ROS2 needed** | Yes | No | Yes |
| **Training overhead** | Low | None | Low |
| **Disk usage** | None | Low (checkpoints) | High (bags) |
| **Reproducibility** | ❌ Live only | ✅ High | ✅ High |
| **Analysis tools** | ROS2 tools | TensorBoard | ROS2 tools |
| **Best for** | Debugging | Demos | Analysis |

---

## Recommendation 🎯

### Start with: **Option B (Checkpoint Replay)** ✅

**Why**:
- Simplest to implement
- No additional dependencies
- Works with current setup
- Easy to share demos

**Workflow**:
1. Train headless in WSL (fast!)
2. Save checkpoints every 100K steps
3. When you want to see behavior:
   - Open Isaac Sim GUI on Windows
   - Load checkpoint
   - Run replay script
   - Watch and record video

### Add later: **Option A (Live ROS2)** if needed

**When**:
- You need to debug specific behaviors
- Want to see training progress live
- Need real-time monitoring

**Already configured**:
- ✅ Fast DDS on both sides
- ✅ ROS2 Humble installed
- ✅ Network working (verified 2025-10-13)

Just need to add:
- ROS2 publisher in environment
- ROS2 subscriber in Windows Isaac Sim

---

## Implementation Priority

### Phase 1: Basic (Now)
1. ✅ Train headless in WSL
2. ✅ Save checkpoints regularly
3. ✅ TensorBoard for metrics

### Phase 2: Visualization (When needed)
4. Create checkpoint replay script
5. Test on Windows Isaac Sim GUI
6. Record demo videos

### Phase 3: Advanced (Optional)
7. Add ROS2 publisher to environment
8. Set up Windows Isaac Sim subscriber
9. Enable live visualization

---

## Code to Create

### Minimal for Option B (Checkpoint Replay)

```python
# scripts/replay_checkpoint.py
# ~50 lines - load policy and run in GUI
```

### Full for Option A (Live ROS2)

```python
# src/rl_platform/tasks/mobile_mm/ros2_bridge.py
# ~200 lines - ROS2 publisher node

# scripts/windows/isaac_sim_subscriber.py
# ~150 lines - Windows subscriber
```

---

## Summary

**Your setup**:
- Primary: WSL headless training (fast, simple)
- Secondary: Windows GUI visualization (when needed)

**Best approach**:
1. Start with checkpoint replay (Option B)
2. Add live ROS2 if debugging needs it (Option A)
3. Keep bag recording as analysis tool (Option C)

**All options compatible** - can use all three!

---

**Next step**: Finish fixing the environment test, then implement Option B replay script! 🚀
