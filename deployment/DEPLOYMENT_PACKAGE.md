# Deployment Package - Session 7d Demo Model

**Created:** January 29, 2025  
**Training Checkpoint:** 28.6M steps (October 28, 2025)  
**Status:** ✅ ONNX export successful, ready for Orin deployment

---

## 📦 Package Contents

### 1. Exported Model Files
- **`policy_demo.onnx`** (466.52 KB)
  - ONNX opset 14 (TensorRT compatible)
  - Input: `(batch_size, 74)` - observation vector
  - Output: `(batch_size, 8)` - action vector
  - Action range: `[-0.300, 0.568]` (normalized)

- **`normalization_stats.npz`** (1.2 KB)
  - `obs_mean`: observation normalization mean
  - `obs_var`: observation normalization variance
  - *Note: Identity normalization (mean=0, var=1) as no VecNormalize used*

### 2. ROS2 Integration
- **`ros2_policy_node.py`** - Inference node
- **`policy_inference.launch.py`** - Launch configuration
- **`test_onnx_inference.py`** - Local validation script

### 3. Documentation
- **`DEPLOYMENT_GUIDE.md`** - Step-by-step instructions
- **`ARCHITECTURE.md`** - System architecture details
- **`README.md`** - Quick start guide
- **`DEPLOYMENT_PACKAGE.md`** - This file

---

## 🎯 Training Details

**Session:** 7d (accelerated)  
**Timesteps:** 28,672,000 / 200,000,000 (14.3%)  
**Training Status:** In progress (~13 hours remaining)

**Dataset:**
- 1,038 cinematic trajectories
- Motion types: dolly, crane, orbit, arc, handheld
- Starting positions: 27 unique (1012 from common ready position)

**Performance at 28.6M steps:**
- Explained variance: ~0.92
- Value loss: ~0.008
- Policy loss: ~0.001
- Learning rate: 0.0003
- Entropy: Decreasing (decay schedule 100M-150M)

---

## 🚀 Quick Deployment (Orin)

### Step 1: Transfer Files
```bash
# From Windows machine
scp deployment/policy_demo.onnx orin@orin-hostname:~/cinebot_ws/models/
scp deployment/normalization_stats.npz orin@orin-hostname:~/cinebot_ws/models/
scp deployment/ros2_policy_node.py orin@orin-hostname:~/cinebot_ws/src/cinebot_control/scripts/
scp deployment/policy_inference.launch.py orin@orin-hostname:~/cinebot_ws/src/cinebot_control/launch/
```

### Step 2: Install Dependencies (on Orin)
```bash
sudo apt install python3-pip
pip3 install onnxruntime-gpu  # or onnxruntime for CPU
pip3 install numpy
```

### Step 3: Build ROS2 Workspace (on Orin)
```bash
cd ~/cinebot_ws
colcon build --packages-select cinebot_control
source install/setup.bash
```

### Step 4: Launch Inference Node (on Orin)
```bash
ros2 launch cinebot_control policy_inference.launch.py \
    model_path:=~/cinebot_ws/models/policy_demo.onnx \
    stats_path:=~/cinebot_ws/models/normalization_stats.npz
```

---

## 📊 Expected Performance

| Metric | Target | Notes |
|--------|--------|-------|
| Inference Latency | 2-5 ms | GPU-accelerated (TensorRT) |
| Control Frequency | 20 Hz | Matches training frequency |
| Model Size | ~467 KB | Lightweight for embedded deployment |
| Observation Dim | 74 | Joint states + base pose + target |
| Action Dim | 8 | 3 base velocities + 6 arm velocities |

---

## 🔧 Action Scaling

The exported model outputs normalized actions `[-1, 1]`. The ROS2 node scales them:

```python
# Base commands (mobile platform)
linear_x = action[0] * 1.5  # ±1.5 m/s
linear_y = action[1] * 1.5  # ±1.5 m/s
angular_z = action[2] * 2.0  # ±2.0 rad/s

# Arm commands (6 DOF manipulator)
joint_velocities = action[3:9] * 1.0  # ±1.0 rad/s
```

---

## ⚠️ Known Limitations

1. **Model Training Incomplete**
   - Only 14.3% of planned training (28.6M / 200M steps)
   - Performance may not be optimal for complex trajectories
   - Recommended to replace with final Session 7d model after training completes

2. **Local Testing Not Available**
   - ONNX Runtime has DLL compatibility issues with Isaac Sim environment on Windows
   - Validation script (`test_onnx_inference.py`) should be run on Orin or standard Python environment
   - ONNX export itself is verified (PyTorch test passed)

3. **Sim-to-Real Transfer**
   - Model trained in simulation may need tuning for real robot
   - Expect initial tracking errors and oscillations
   - Use conservative action scaling initially (e.g., 0.5x multiplier)

---

## 🔄 Updating to Final Model

When Session 7d completes (ETA: ~13 hours, January 29, 2025 ~1:00 AM):

### Step 1: Export Final Model
```powershell
# On Windows (in cinebotRL workspace)
I:\isaaclab\isaaclab.bat -p scripts/export_policy_simple.py `
    --checkpoint logs\sb3\mobilemmtrackee_v0\<SESSION_7D_DIR>\checkpoints\ppo_mobile_mm_200000000_steps.zip `
    --output deployment\policy_final.onnx
```

### Step 2: Replace Model on Orin
```bash
# On Orin
scp windows-pc:~/cinebotRL/deployment/policy_final.onnx ~/cinebot_ws/models/
scp windows-pc:~/cinebotRL/deployment/normalization_stats.npz ~/cinebot_ws/models/

# Restart inference node (it will automatically load new model)
ros2 launch cinebot_control policy_inference.launch.py \
    model_path:=~/cinebot_ws/models/policy_final.onnx
```

No code changes needed - just replace the ONNX file!

---

## 📝 ROS2 Topic Interface

### Subscribed Topics
| Topic | Type | Rate | Description |
|-------|------|------|-------------|
| `/joint_states` | `sensor_msgs/JointState` | 50 Hz | Current joint positions/velocities |
| `/base_pose` | `geometry_msgs/Pose2D` | 50 Hz | Base position (x, y, yaw) |
| `/camera_target_pose` | `geometry_msgs/PoseStamped` | 20 Hz | Target camera pose (7D) |

### Published Topics
| Topic | Type | Rate | Description |
|-------|------|------|-------------|
| `/joint_commands` | `trajectory_msgs/JointTrajectory` | 20 Hz | Velocity commands (3 base + 6 arm) |

---

## 🐛 Troubleshooting

### Issue: High inference latency (>10 ms)
**Solutions:**
1. Check GPU is being used: `nvidia-smi` during inference
2. Verify CUDA execution provider: Look for "CUDAExecutionProvider" in node startup logs
3. Install TensorRT for Orin: `sudo apt install libnvinfer8`
4. Use TensorRT provider in ROS2 node (edit `ros2_policy_node.py`)

### Issue: Robot moves erratically
**Solutions:**
1. Reduce action scaling (multiply by 0.5 in `ros2_policy_node.py`)
2. Check observation topics are publishing at correct rates
3. Verify joint order matches training configuration (see `src/task_spec.py`)
4. Enable smoothing/filtering on velocity commands

### Issue: Observations out of range
**Solutions:**
1. Check normalization stats are loaded correctly
2. Verify sensor data is in correct units (meters, radians)
3. Compare observation ranges with training data (see training logs)
4. Disable normalization if stats file is corrupted (use identity)

---

## 📞 Support

For issues or questions:
1. Check documentation: `docs/DEPLOYMENT_GUIDE.md`
2. Review architecture: `docs/ARCHITECTURE.md`
3. Check training logs: `logs/sb3/mobilemmtrackee_v0/<session_dir>/`
4. Examine training config: `scripts/launch_session_7d_accelerated.ps1`

---

## ✅ Deployment Checklist

Before deploying to real robot:

- [ ] ONNX model exported successfully (466 KB file size)
- [ ] Normalization stats file present
- [ ] ROS2 node and launch file copied to Orin
- [ ] Dependencies installed on Orin (onnxruntime-gpu, numpy)
- [ ] ROS2 workspace built (`colcon build`)
- [ ] Topic names match robot's sensor publishers
- [ ] Joint order verified against training configuration
- [ ] Action scaling tuned conservatively for initial tests
- [ ] Emergency stop mechanism tested
- [ ] Joint limits configured in controller
- [ ] Collision detection enabled
- [ ] Test with simple trajectory first (straight line or circle)
- [ ] Monitor for oscillations, overshoots, or instability
- [ ] Gradually increase complexity (multi-trajectory tracking)

---

## 📈 Performance Monitoring

During deployment, monitor:

1. **Inference metrics**
   - Latency (target: <5 ms)
   - GPU utilization (target: >80%)
   - CPU usage (target: <30%)

2. **Control quality**
   - Tracking error (end-effector to target)
   - Joint velocity magnitudes
   - Base velocity smoothness
   - Singularity avoidance

3. **System health**
   - Topic publish rates
   - Message queue sizes
   - Memory usage
   - Temperature (Orin can throttle under load)

---

## 🎬 Next Steps

1. **Immediate (0-2 hours)**
   - Transfer files to Orin
   - Set up ROS2 workspace
   - Test with demo model (28.6M steps)

2. **Short-term (13+ hours)**
   - Wait for Session 7d training to complete
   - Export final model (200M steps)
   - Hot-swap models and compare performance

3. **Long-term (days-weeks)**
   - Collect real-world tracking data
   - Fine-tune action scaling and control parameters
   - Train domain adaptation models if needed
   - Integrate with full cinematic camera system

---

**Status:** Ready for deployment testing ✅  
**Training:** In progress (Session 7d: 28.3M / 200M steps)  
**ETA Final Model:** ~13 hours (January 29, 2025 ~1:00 AM)
