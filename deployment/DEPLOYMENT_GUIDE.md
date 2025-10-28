# Deployment Guide: Trained Policy on NVIDIA Orin + ROS2 Humble

This guide covers deploying your trained PPO policy from Windows training to NVIDIA Orin with ROS2 Humble.

## Prerequisites

### On Training PC (Windows)
- Trained model checkpoint: `logs/sb3/mobilemmtrackee_v0/TIMESTAMP/checkpoints/rl_model_XXXXX_steps.zip`
- Python 3.8+ with PyTorch, stable-baselines3, onnx, onnxruntime

### On NVIDIA Orin (Linux ARM64)
- ROS2 Humble installed
- Python 3.10+
- ONNX Runtime (GPU or CPU)

---

## Step 1: Export Trained Policy (Windows)

After training completes (~200M timesteps), export the best checkpoint:

```bash
# From cinebotRL directory
python scripts/export_policy_for_deployment.py \
    --checkpoint logs/sb3/mobilemmtrackee_v0/20251028_XXXXXX/checkpoints/rl_model_200000000_steps.zip \
    --output deployment/policy_session_7d.onnx \
    --format onnx
```

This creates:
- `deployment/policy_session_7d.onnx` - Optimized ONNX model (~235KB)
- `deployment/normalization_stats.npz` - Observation normalization stats

---

## Step 2: Transfer Files to NVIDIA Orin

```bash
# From Windows (PowerShell)
scp deployment/policy_session_7d.onnx orin@orin-hostname:/home/orin/cinebot_ws/
scp deployment/normalization_stats.npz orin@orin-hostname:/home/orin/cinebot_ws/
scp deployment/ros2_policy_node.py orin@orin-hostname:/home/orin/cinebot_ws/src/cinebot_control/scripts/
scp deployment/policy_inference.launch.py orin@orin-hostname:/home/orin/cinebot_ws/src/cinebot_control/launch/
```

---

## Step 3: Install Dependencies on Orin

```bash
# SSH into Orin
ssh orin@orin-hostname

# Install ONNX Runtime (GPU-accelerated for Orin)
sudo apt update
sudo apt install python3-pip

# For GPU inference (recommended)
pip3 install onnxruntime-gpu

# Or for CPU-only inference
# pip3 install onnxruntime

# Install other dependencies
pip3 install numpy
sudo apt install ros-humble-tf-transformations
```

---

## Step 4: Setup ROS2 Workspace

```bash
cd ~/cinebot_ws

# Make policy node executable
chmod +x src/cinebot_control/scripts/ros2_policy_node.py

# Build workspace
colcon build --packages-select cinebot_control

# Source workspace
source install/setup.bash
```

---

## Step 5: Configure ROS2 Topics

Ensure your robot hardware drivers publish to these topics:

### Required Subscribers:
- `/joint_states` (sensor_msgs/JointState)
  - Joint names: `['joint_x', 'joint_y', 'joint_theta', 'left_arm_joint1', ..., 'left_arm_joint6']`
  - Published by robot state publisher / hardware drivers

- `/camera_target_pose` (geometry_msgs/PoseStamped)
  - Target EE pose from trajectory playback node
  - [x, y, z, qx, qy, qz, qw]

- `/base_pose` (geometry_msgs/PoseStamped)
  - Current base pose from localization (e.g., AMCL, wheel odometry)
  - [x, y, yaw]

### Published Commands:
- `/joint_commands` (trajectory_msgs/JointTrajectory)
  - Joint position commands for robot controllers
  - 20 Hz update rate (matches training)

---

## Step 6: Launch Policy Inference

```bash
# Test with default paths
ros2 launch cinebot_control policy_inference.launch.py

# Or specify paths explicitly
ros2 launch cinebot_control policy_inference.launch.py \
    model_path:=/home/orin/cinebot_ws/policy_session_7d.onnx \
    stats_path:=/home/orin/cinebot_ws/normalization_stats.npz \
    control_frequency:=20.0
```

Expected output:
```
[policy_inference]: Policy inference node initialized
[policy_inference]:   Model: policy_session_7d.onnx
[policy_inference]:   Provider: CUDAExecutionProvider
[policy_inference]:   Control freq: 20.0 Hz
```

---

## Step 7: Monitor Performance

### Check topics:
```bash
# Monitor published commands
ros2 topic echo /joint_commands

# Monitor node status
ros2 node info /policy_inference

# Check inference latency
ros2 topic hz /joint_commands  # Should be ~20 Hz
```

### Expected performance on Orin:
- **Inference time**: 2-5ms per step (GPU)
- **Control frequency**: 20 Hz (50ms loop)
- **Latency**: <10ms end-to-end

---

## Step 8: Integration with Robot Hardware

### Option A: Direct Joint Control
If your robot supports trajectory controllers:

```bash
# In your robot's launch file, add controller manager
ros2 run controller_manager spawner joint_trajectory_controller

# Remap policy output to controller input
ros2 topic remap /joint_commands /joint_trajectory_controller/joint_trajectory
```

### Option B: Velocity Control
If using velocity controllers, modify `ros2_policy_node.py`:

```python
# In control_loop(), publish velocities directly:
msg = Float64MultiArray()
msg.data = [base_vx, base_vy, base_wz] + arm_velocities.tolist()
self.cmd_pub.publish(msg)
```

---

## Troubleshooting

### Issue: "CUDA execution provider not available"
**Solution**: Install ONNX Runtime GPU build:
```bash
pip3 uninstall onnxruntime
pip3 install onnxruntime-gpu
```

### Issue: "Observation dimension mismatch"
**Solution**: Verify observation construction in `ros2_policy_node.py` matches training:
```bash
# Check expected obs dim from ONNX model
python3 -c "import onnxruntime as ort; sess = ort.InferenceSession('policy_session_7d.onnx'); print(sess.get_inputs()[0].shape)"
# Should output: [1, 74]
```

### Issue: Robot jerky/unstable motion
**Possible causes:**
1. **Control frequency mismatch** - Ensure 20 Hz matches training
2. **Normalization stats incorrect** - Re-export with correct stats
3. **Action scaling wrong** - Adjust velocity limits in `control_loop()`

### Issue: Poor tracking performance
**Diagnostics:**
1. Check if target poses match training distribution
2. Verify base odometry quality (localization drift)
3. Test with simpler trajectories first (straight lines)

---

## Performance Optimization

### 1. Enable TensorRT (NVIDIA Orin)
For 2-3x faster inference:

```bash
# Install TensorRT
sudo apt install tensorrt

# Convert ONNX to TensorRT engine
trtexec --onnx=policy_session_7d.onnx --saveEngine=policy_session_7d.trt --fp16
```

Then modify node to load `.trt` file instead of `.onnx`.

### 2. Reduce Control Frequency
If inference is slow:

```bash
ros2 launch cinebot_control policy_inference.launch.py control_frequency:=10.0
```

### 3. Batch Processing
For multiple robots, modify node to batch observations and run inference once.

---

## Testing Checklist

- [ ] ONNX model loads successfully on Orin
- [ ] All ROS2 topics publishing at expected rates
- [ ] Joint commands within safe limits
- [ ] Robot responds to commands smoothly
- [ ] Tracking error acceptable for application
- [ ] No collisions during operation
- [ ] Emergency stop working properly

---

## Next Steps

1. **Sim-to-Real Gap**: Initial performance may differ from simulation
   - Tune action scaling factors
   - Adjust velocity limits
   - Consider domain randomization fine-tuning

2. **Safety Monitoring**: Add safety checks in ROS2 node
   - Joint limit checking
   - Collision detection
   - Velocity limiting

3. **Performance Logging**: Record tracking metrics
   - Position error
   - Velocity smoothness
   - Inference latency

4. **Trajectory Playback**: Integrate with cinematic trajectory library
   - Load trajectories from JSON files
   - Publish target poses at 10 Hz

---

## Additional Resources

- **ONNX Runtime Docs**: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
- **ROS2 Humble Tutorials**: https://docs.ros.org/en/humble/Tutorials.html
- **NVIDIA Orin Setup**: https://developer.nvidia.com/embedded/jetson-agx-orin-developer-kit

For issues, see: `docs/troubleshooting/deployment_issues.md`
