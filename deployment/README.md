# Deployment Package: CinebotRL → NVIDIA Orin + ROS2 Humble

This directory contains everything needed to deploy your trained PPO policy from Windows/Isaac Lab to a real robot running on NVIDIA Orin with ROS2 Humble.

## 📁 Files in This Directory

- **`DEPLOYMENT_GUIDE.md`** - Step-by-step deployment instructions
- **`ARCHITECTURE.md`** - System architecture and data flow diagrams  
- **`ros2_policy_node.py`** - ROS2 node for policy inference
- **`policy_inference.launch.py`** - ROS2 launch file
- **`policy_session_7d.onnx`** - Exported trained policy (created after training)
- **`normalization_stats.npz`** - Observation normalization (created after training)

## 🚀 Quick Start

### 1. Export Policy (After Training Completes)

```bash
# On Windows training PC
cd C:\Users\yanbo\wSpace\cinebotRL

python scripts\export_policy_for_deployment.py \
    --checkpoint logs\sb3\mobilemmtrackee_v0\20251028_XXXXXX\checkpoints\rl_model_200000000_steps.zip \
    --output deployment\policy_session_7d.onnx \
    --format onnx
```

Output:
- `deployment/policy_session_7d.onnx` (~235 KB)
- `deployment/normalization_stats.npz` (observation stats)

### 2. Transfer to NVIDIA Orin

```powershell
# From Windows
scp deployment\policy_session_7d.onnx orin@YOUR_ORIN_IP:~/cinebot_ws/
scp deployment\normalization_stats.npz orin@YOUR_ORIN_IP:~/cinebot_ws/
scp deployment\ros2_policy_node.py orin@YOUR_ORIN_IP:~/cinebot_ws/src/cinebot_control/scripts/
scp deployment\policy_inference.launch.py orin@YOUR_ORIN_IP:~/cinebot_ws/src/cinebot_control/launch/
```

### 3. Setup on Orin

```bash
# SSH into Orin
ssh orin@YOUR_ORIN_IP

# Install dependencies
pip3 install onnxruntime-gpu numpy
sudo apt install ros-humble-tf-transformations

# Build workspace
cd ~/cinebot_ws
colcon build --packages-select cinebot_control
source install/setup.bash
```

### 4. Launch Policy

```bash
ros2 launch cinebot_control policy_inference.launch.py \
    model_path:=/home/orin/cinebot_ws/policy_session_7d.onnx \
    stats_path:=/home/orin/cinebot_ws/normalization_stats.npz
```

Expected output:
```
[policy_inference]: Policy inference node initialized
[policy_inference]:   Model: policy_session_7d.onnx
[policy_inference]:   Provider: CUDAExecutionProvider  ← GPU inference!
[policy_inference]:   Control freq: 20.0 Hz
```

## 📊 System Requirements

### NVIDIA Orin Specifications
- **OS**: Ubuntu 20.04/22.04 (ARM64)
- **ROS2**: Humble Hawksbill
- **RAM**: 8GB+ recommended
- **GPU**: 1024-2048 CUDA cores (Orin NX/AGX)
- **Storage**: 1GB free space

### Python Dependencies
```
onnxruntime-gpu >= 1.12.0  # GPU-accelerated inference
numpy >= 1.20.0
tf-transformations          # Quaternion math
```

### ROS2 Packages
```
ros-humble-geometry-msgs
ros-humble-sensor-msgs
ros-humble-trajectory-msgs
ros-humble-tf-transformations
```

## 🔌 ROS2 Topic Interface

### Subscribed Topics (Inputs)

| Topic | Type | Rate | Description |
|-------|------|------|-------------|
| `/joint_states` | `sensor_msgs/JointState` | 50+ Hz | Current joint positions/velocities |
| `/base_pose` | `geometry_msgs/PoseStamped` | 20+ Hz | Base localization (x, y, yaw) |
| `/camera_target_pose` | `geometry_msgs/PoseStamped` | 10 Hz | Target EE pose from trajectory |

### Published Topics (Outputs)

| Topic | Type | Rate | Description |
|-------|------|------|-------------|
| `/joint_commands` | `trajectory_msgs/JointTrajectory` | 20 Hz | Joint position commands |

## ⚙️ Configuration

Key parameters in `policy_inference.launch.py`:

```python
{
    'model_path': 'policy_session_7d.onnx',      # Path to ONNX model
    'stats_path': 'normalization_stats.npz',      # Normalization stats
    'control_frequency': 20.0,                    # Must match training!
    'base_joints': ['joint_x', 'joint_y', 'joint_theta'],
    'arm_joints': ['left_arm_joint1', ..., 'left_arm_joint6']
}
```

## 📈 Performance Metrics

Expected performance on NVIDIA Orin AGX:

| Metric | Value | Notes |
|--------|-------|-------|
| **Inference time** | 2-5 ms | GPU (CUDA) |
| **Control loop** | 20 Hz | 50 ms period |
| **E2E latency** | <10 ms | Observation → Command |
| **GPU utilization** | 5-10% | Plenty of headroom |
| **CPU utilization** | 10-15% | Mostly ROS overhead |
| **Memory** | ~500 MB | Including ROS nodes |

## 🛠️ Troubleshooting

### "CUDAExecutionProvider not available"
```bash
# Install GPU-enabled ONNX Runtime
pip3 uninstall onnxruntime
pip3 install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```

### Robot motion is jerky
1. Check control frequency: `ros2 topic hz /joint_commands` (should be ~20 Hz)
2. Verify observation topics publishing at high rate
3. Reduce action scaling in `ros2_policy_node.py` (multiply velocities by 0.5)

### Poor tracking accuracy
1. Verify normalization stats match training
2. Check base localization quality (odometry drift?)
3. Test with simpler trajectories first
4. Consider domain adaptation fine-tuning

## 📚 Documentation

- **`DEPLOYMENT_GUIDE.md`** - Complete deployment walkthrough
- **`ARCHITECTURE.md`** - System architecture diagrams
- **`../docs/TRAIN_ON_WINDOWS.md`** - Training documentation
- **`../README.md`** - Project overview

## 🔐 Safety Checklist

Before running on real hardware:

- [ ] Emergency stop button accessible
- [ ] Joint limits configured correctly
- [ ] Workspace clear of obstacles
- [ ] Start with 0.3x action scaling
- [ ] Test with simple trajectories
- [ ] Monitor for oscillations
- [ ] Have manual override ready

## 🎯 Next Steps

1. **Test in simulation** - Verify deployment setup in Gazebo/Isaac Sim
2. **Gradual scaling** - Start with reduced velocities, increase gradually
3. **Data collection** - Log real-world performance for analysis
4. **Fine-tuning** - Use collected data for domain adaptation if needed
5. **Integration** - Connect to trajectory library (1,038 cinematic moves)

## 📞 Support

For issues:
1. Check `DEPLOYMENT_GUIDE.md` troubleshooting section
2. Verify ROS2 topic connectivity: `ros2 topic list`
3. Monitor node output: `ros2 node info /policy_inference`
4. See training logs for hyperparameter reference

---

**Training Status**: Session 7d ongoing (28.3M / 200M timesteps)  
**Expected completion**: October 29, 2025 ~1:00 AM  
**Deployment ready**: After checkpoint export (~235 KB model)
