# Deployment Checklist - CineBot RL to Orin

**Training Complete:** October 29, 2025  
**Final Model:** 200M timesteps (15.3 hours training)  
**Target Platform:** NVIDIA Orin + ROS2 Humble

---

## ✅ Pre-Deployment (Completed)

- [x] Training completed (200M timesteps)
- [x] Final model exported: `policy_final.onnx`
- [x] Demo model tested: `policy_demo.onnx` (28.6M checkpoint)
- [x] WSL validation passed (ONNX + ROS2 integration)
- [x] Robot interface documented (`ROBOT_INTERFACE.md`)
- [x] Deployment package created (all files ready)

---

## 📦 Files to Transfer to Orin

### Required Files:
```
deployment/
├── policy_final.onnx              # ← Final trained model (200M)
├── normalization_stats.npz        # ← Observation normalization
├── ros2_policy_node_robot.py      # ← ROS2 node (robot-specific)
├── policy_inference_robot.launch.py  # ← Launch file
├── ROBOT_INTERFACE.md             # ← Interface documentation
└── DEPLOYMENT_GUIDE.md            # ← Step-by-step instructions
```

### Transfer Command (from Windows to Orin):
```powershell
# Option 1: Via network (if Orin accessible)
scp deployment/policy_final.onnx deployment/normalization_stats.npz deployment/ros2_policy_node_robot.py deployment/policy_inference_robot.launch.py orin_user@orin_ip:/path/to/deployment/

# Option 2: Via USB drive
Copy-Item deployment\* -Destination E:\deployment\ -Recurse
```

---

## 🚀 Deployment Steps on Orin

### Step 1: Install Dependencies
```bash
# On Orin (Ubuntu 22.04 + ROS2 Humble)
pip3 install onnxruntime-gpu numpy
# OR for CPU-only testing:
pip3 install onnxruntime numpy
```

### Step 2: Verify ONNX Model
```bash
cd /path/to/deployment
python3 -c "
import onnxruntime as ort
import numpy as np

# Load model
session = ort.InferenceSession('policy_final.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
print(f'Model loaded: {session.get_providers()}')

# Test inference
obs = np.random.randn(1, 74).astype(np.float32)
actions = session.run(['action'], {'observation': obs})[0]
print(f'Test inference successful: actions shape = {actions.shape}')
"
```

### Step 3: Check Robot Topics
```bash
# Verify robot is publishing feedback
ros2 topic echo /hdas/feedback_arm_left --once
ros2 topic echo /odom_wheel --once

# Check topic types match
ros2 topic info /hdas/feedback_arm_left
ros2 topic info /mobile_base/commands/velocity
```

### Step 4: Verify Joint Names
```bash
# Check actual joint names from robot
ros2 topic echo /hdas/feedback_arm_left --once | grep "name:"

# Expected names (update ros2_policy_node_robot.py if different):
# - left_arm_joint1
# - left_arm_joint2
# - left_arm_joint3
# - left_arm_joint4
# - left_arm_joint5
# - left_arm_joint6
```

### Step 5: Create ROS2 Workspace
```bash
# Create workspace
mkdir -p ~/cinebot_ws/src
cd ~/cinebot_ws/src

# Create package
ros2 pkg create cinebot_policy --build-type ament_python --dependencies rclpy sensor_msgs geometry_msgs nav_msgs

# Copy files
cp /path/to/deployment/ros2_policy_node_robot.py cinebot_policy/cinebot_policy/
cp /path/to/deployment/policy_final.onnx cinebot_policy/cinebot_policy/
cp /path/to/deployment/normalization_stats.npz cinebot_policy/cinebot_policy/
cp /path/to/deployment/policy_inference_robot.launch.py cinebot_policy/launch/

# Build
cd ~/cinebot_ws
colcon build --packages-select cinebot_policy
source install/setup.bash
```

### Step 6: Test with Conservative Scaling (CRITICAL!)
```bash
# Launch with 0.5x scaling for safety
ros2 launch cinebot_policy policy_inference_robot.launch.py \
    model_path:=/path/to/policy_final.onnx \
    base_vel_scale:=0.5 \
    arm_vel_scale:=0.5 \
    control_frequency:=20.0
```

### Step 7: Monitor Performance
```bash
# Terminal 1: Watch policy node output
ros2 launch cinebot_policy policy_inference_robot.launch.py ...

# Terminal 2: Monitor command outputs
ros2 topic echo /motion_target/target_joint_state_arm_left
ros2 topic echo /mobile_base/commands/velocity

# Terminal 3: Check latency
ros2 topic hz /motion_target/target_joint_state_arm_left
# Should be ~20 Hz
```

---

## ⚠️ Safety Checklist

Before deploying to real robot:

- [ ] **Emergency Stop Ready**: E-stop button accessible
- [ ] **Conservative Scaling**: Start with 0.5x velocity scaling
- [ ] **Workspace Clear**: No obstacles in 15m workspace
- [ ] **Joint Limits**: Verify robot joint limits match training
- [ ] **Base Speed Limit**: Verify base can safely handle commanded velocities
- [ ] **Monitoring**: Person ready to E-stop if needed
- [ ] **Test Trajectory**: Start with simple, slow trajectory

---

## 🎯 Expected Performance

Based on training metrics:
- **Control Frequency:** 20 Hz (matches training)
- **Inference Latency:** 2-5ms on Orin GPU (tested 0.02ms on CPU in WSL)
- **Tracking Accuracy:** TBD (run evaluation first)
- **Convergence:** Policy fully converged at 200M timesteps

---

## 📊 Post-Deployment Validation

### Metrics to Monitor:
1. **Inference Latency** (log in node)
2. **Control Frequency** (should be 20 Hz)
3. **Tracking Error** (visual/quantitative)
4. **Joint Position Continuity** (no jumps)
5. **Base Velocity Smoothness**

### Tuning Parameters:
```python
# In policy_inference_robot.launch.py
base_vel_scale=0.5      # Increase if too slow (max 1.0)
arm_vel_scale=0.5       # Increase if too slow (max 1.0)
control_frequency=20.0  # Keep at 20 Hz (training frequency)
integration_dt=0.05     # Keep at 0.05 (matches control frequency)
```

---

## 🔧 Troubleshooting

### Issue: High Latency (>10ms)
**Solution:**
- Use GPU: `providers=['CUDAExecutionProvider']`
- Reduce batch size (already 1)
- Check Orin CPU/GPU utilization

### Issue: Robot Doesn't Move
**Solution:**
- Check topic remapping in launch file
- Verify joint names match robot
- Check action scaling (might be too conservative)
- Verify normalization stats loaded correctly

### Issue: Jerky Motion
**Solution:**
- Increase `integration_dt` smoothing
- Add low-pass filter to velocity commands
- Check control frequency stability

### Issue: Robot Moves Wrong Direction
**Solution:**
- Check joint order in `ros2_policy_node_robot.py`
- Verify base coordinate frame (x=forward, y=left)
- Check velocity sign conventions

---

## 📝 Deployment Log Template

```
Deployment Date: ___________
Orin Device ID: ___________
ROS2 Version: Humble
Model: policy_final.onnx (200M timesteps)

Test 1: Static Position
- Arm Position: [___, ___, ___, ___, ___, ___]
- Base Position: (x=___, y=___, yaw=___)
- Result: PASS / FAIL
- Notes: ___________

Test 2: Simple Trajectory
- Trajectory: ___________
- Scaling: base=___, arm=___
- Tracking Quality: ___/10
- Issues: ___________

Test 3: Complex Trajectory
- Trajectory: ___________
- Scaling: base=___, arm=___
- Tracking Quality: ___/10
- Issues: ___________
```

---

## 🎬 Next Steps After Successful Deployment

1. **Collect Real-World Data**
   - Record actual tracking performance
   - Compare with simulation metrics

2. **Fine-Tune Scaling**
   - Gradually increase from 0.5x to 1.0x
   - Find optimal speed/accuracy trade-off

3. **Test All Trajectory Types**
   - Dolly (push/pull)
   - Crane (up/down)
   - Orbit
   - Arc
   - Handheld

4. **Integrate with Cinematic System**
   - Connect to shot planner
   - Add trajectory recording
   - Implement multi-shot sequences

---

## 📞 Support Resources

- **Training Documentation:** `docs/TRAINING_BUDGET_ANALYSIS.md`
- **Robot Interface:** `deployment/ROBOT_INTERFACE.md`
- **Deployment Guide:** `deployment/DEPLOYMENT_GUIDE.md`
- **WSL Test Results:** `deployment/WSL_TEST_RESULTS.md`
- **Architecture:** `deployment/ARCHITECTURE.md`

---

**Status:** Ready for deployment! All files prepared, model exported, documentation complete. 🚀
