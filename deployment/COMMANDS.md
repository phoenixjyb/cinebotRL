# Deployment Command Reference

Quick copy-paste commands for deploying the trained policy to NVIDIA Orin + ROS2 Humble.

---

## 📦 Files to Transfer

From Windows machine at `C:\Users\yanbo\wSpace\cinebotRL\deployment\`:
- `policy_demo.onnx` (466.52 KB)
- `normalization_stats.npz` (2.5 KB)
- `ros2_policy_node.py` (11.6 KB)
- `policy_inference.launch.py` (2.2 KB)

---

## 🖥️ Windows (Export - COMPLETED ✅)

### Export Existing Checkpoint
```powershell
I:\isaaclab\isaaclab.bat -p scripts/export_policy_simple.py `
    --checkpoint logs\sb3\mobilemmtrackee_v0\20251028_200923\checkpoints\ppo_mobile_mm_28672000_steps.zip `
    --output deployment\policy_demo.onnx
```

### Export Final Session 7d Model (After Training Completes)
```powershell
I:\isaaclab\isaaclab.bat -p scripts/export_policy_simple.py `
    --checkpoint logs\sb3\mobilemmtrackee_v0\<SESSION_DIR>\checkpoints\ppo_mobile_mm_200000000_steps.zip `
    --output deployment\policy_final.onnx
```

---

## 🔄 Transfer to Orin

### Method 1: SCP (Recommended)
```bash
# Set Orin hostname/IP
export ORIN_HOST="orin@192.168.1.100"  # Adjust IP address

# Create directories on Orin
ssh $ORIN_HOST "mkdir -p ~/cinebot_ws/models ~/cinebot_ws/src/cinebot_control/scripts ~/cinebot_ws/src/cinebot_control/launch"

# Transfer model files
scp deployment/policy_demo.onnx $ORIN_HOST:~/cinebot_ws/models/
scp deployment/normalization_stats.npz $ORIN_HOST:~/cinebot_ws/models/

# Transfer ROS2 files
scp deployment/ros2_policy_node.py $ORIN_HOST:~/cinebot_ws/src/cinebot_control/scripts/
scp deployment/policy_inference.launch.py $ORIN_HOST:~/cinebot_ws/src/cinebot_control/launch/

# Transfer documentation
scp deployment/README.md $ORIN_HOST:~/cinebot_ws/docs/
scp deployment/DEPLOYMENT_GUIDE.md $ORIN_HOST:~/cinebot_ws/docs/
```

### Method 2: USB Drive
```bash
# On Windows (PowerShell)
Copy-Item deployment\policy_demo.onnx E:\cinebot_deployment\
Copy-Item deployment\normalization_stats.npz E:\cinebot_deployment\
Copy-Item deployment\ros2_policy_node.py E:\cinebot_deployment\
Copy-Item deployment\policy_inference.launch.py E:\cinebot_deployment\

# On Orin (after mounting USB)
cp /media/usb/cinebot_deployment/* ~/cinebot_ws/models/
```

---

## 🤖 Orin Setup

### 1. Install Dependencies
```bash
# System dependencies
sudo apt update
sudo apt install -y python3-pip ros-humble-trajectory-msgs ros-humble-geometry-msgs

# Python dependencies
pip3 install onnxruntime-gpu numpy

# Optional: TensorRT for better performance
sudo apt install -y libnvinfer8 libnvinfer-plugin8

# Verify CUDA
nvidia-smi
```

### 2. Create ROS2 Package
```bash
# Navigate to workspace
cd ~/cinebot_ws/src

# Create package
ros2 pkg create cinebot_control --build-type ament_python --dependencies rclpy sensor_msgs geometry_msgs trajectory_msgs

# Create directories
mkdir -p cinebot_control/scripts cinebot_control/launch cinebot_control/models

# Copy files (if not done via SCP)
cp ~/cinebot_ws/models/*.{onnx,npz} cinebot_control/models/
cp <path>/ros2_policy_node.py cinebot_control/scripts/
cp <path>/policy_inference.launch.py cinebot_control/launch/

# Make scripts executable
chmod +x cinebot_control/scripts/ros2_policy_node.py
```

### 3. Update Package Configuration
```bash
# Edit setup.py
nano ~/cinebot_ws/src/cinebot_control/setup.py
```

Add to `setup.py`:
```python
from setuptools import setup
import os
from glob import glob

package_name = 'cinebot_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'models'), glob('models/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Cinebot Team',
    maintainer_email='team@cinebot.ai',
    description='Policy inference for cinematic mobile manipulator',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'policy_inference = cinebot_control.ros2_policy_node:main'
        ],
    },
)
```

### 4. Build Workspace
```bash
cd ~/cinebot_ws
colcon build --packages-select cinebot_control
source install/setup.bash

# Verify build
ros2 pkg list | grep cinebot_control
```

---

## 🚀 Launch Inference

### Basic Launch
```bash
# Source workspace
source ~/cinebot_ws/install/setup.bash

# Launch node
ros2 launch cinebot_control policy_inference.launch.py
```

### Launch with Custom Parameters
```bash
ros2 launch cinebot_control policy_inference.launch.py \
    model_path:=~/cinebot_ws/models/policy_demo.onnx \
    stats_path:=~/cinebot_ws/models/normalization_stats.npz \
    control_frequency:=20.0
```

### Launch as Background Service
```bash
# Launch in background
ros2 launch cinebot_control policy_inference.launch.py &

# Check if running
ros2 node list | grep policy_inference

# Kill background process
killall ros2
```

---

## 🔍 Monitoring & Debugging

### Check Topics
```bash
# List all topics
ros2 topic list

# Check topic rates
ros2 topic hz /joint_commands
ros2 topic hz /joint_states
ros2 topic hz /base_pose
ros2 topic hz /camera_target_pose

# Echo topic data
ros2 topic echo /joint_commands
```

### Monitor Node
```bash
# Check node info
ros2 node info /policy_inference_node

# View logs
ros2 run rqt_console rqt_console

# Or direct logs
ros2 topic echo /rosout | grep policy_inference
```

### Check GPU Usage
```bash
# While inference is running
watch -n 0.5 nvidia-smi

# Or continuous monitoring
nvidia-smi dmon -s um
```

### Benchmark Latency
```bash
# Enable verbose logging (edit launch file or node)
ros2 launch cinebot_control policy_inference.launch.py --log-level debug

# Or check latency programmatically
ros2 topic echo /joint_commands --field time
```

---

## 🔧 Tuning & Adjustment

### Reduce Action Scaling (If Robot Moves Too Fast)
```bash
# Edit node file
nano ~/cinebot_ws/src/cinebot_control/scripts/ros2_policy_node.py

# Find action scaling section (around line 220):
# Change:
#   base_linear_scale = 1.5
#   base_angular_scale = 2.0
#   arm_velocity_scale = 1.0
# To:
#   base_linear_scale = 0.75   # Reduce by 50%
#   base_angular_scale = 1.0   # Reduce by 50%
#   arm_velocity_scale = 0.5   # Reduce by 50%

# Rebuild
cd ~/cinebot_ws && colcon build --packages-select cinebot_control
source install/setup.bash
```

### Change Control Frequency
```bash
# Lower frequency (more stable, slower response)
ros2 launch cinebot_control policy_inference.launch.py control_frequency:=10.0

# Higher frequency (faster response, may oscillate)
ros2 launch cinebot_control policy_inference.launch.py control_frequency:=30.0

# Default: 20.0 Hz (matches training)
```

### Disable Normalization (If Observations Look Wrong)
```bash
# Edit node file
nano ~/cinebot_ws/src/cinebot_control/scripts/ros2_policy_node.py

# Find normalization section (around line 210):
# Comment out:
#   obs_std = np.sqrt(self.obs_var + 1e-8)
#   normalized_obs = (obs - self.obs_mean) / obs_std
# Add:
#   normalized_obs = obs  # Use raw observations

# Rebuild
cd ~/cinebot_ws && colcon build --packages-select cinebot_control
source install/setup.bash
```

---

## 🔄 Update to Final Model

### After Session 7d Completes (200M Steps)

**On Windows:**
```powershell
# Export final model
I:\isaaclab\isaaclab.bat -p scripts/export_policy_simple.py `
    --checkpoint logs\sb3\mobilemmtrackee_v0\<SESSION_DIR>\checkpoints\ppo_mobile_mm_200000000_steps.zip `
    --output deployment\policy_final.onnx

# Transfer to Orin
scp deployment\policy_final.onnx $ORIN_HOST:~/cinebot_ws/models/
```

**On Orin:**
```bash
# Stop inference node
killall ros2

# Replace model
mv ~/cinebot_ws/models/policy_demo.onnx ~/cinebot_ws/models/policy_demo_backup.onnx
mv ~/cinebot_ws/models/policy_final.onnx ~/cinebot_ws/models/policy_demo.onnx

# Restart node (no rebuild needed!)
ros2 launch cinebot_control policy_inference.launch.py
```

---

## 📊 Performance Verification

### Test Inference Latency
```bash
# Install Python dependencies for testing
pip3 install matplotlib pandas

# Run latency test (create test_latency.py)
cat > ~/cinebot_ws/test_latency.py << 'EOF'
import time
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession(
    "models/policy_demo.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

obs = np.random.randn(1, 74).astype(np.float32)

# Warmup
for _ in range(10):
    session.run(None, {"observation": obs})

# Benchmark
latencies = []
for _ in range(1000):
    start = time.perf_counter()
    session.run(None, {"observation": obs})
    latencies.append((time.perf_counter() - start) * 1000)

latencies = np.array(latencies)
print(f"Mean: {latencies.mean():.2f} ms")
print(f"Median: {np.median(latencies):.2f} ms")
print(f"P95: {np.percentile(latencies, 95):.2f} ms")
print(f"Min: {latencies.min():.2f} ms")
print(f"Max: {latencies.max():.2f} ms")
EOF

python3 ~/cinebot_ws/test_latency.py
```

### Test Full Pipeline (with ROS2)
```bash
# Terminal 1: Launch inference node
ros2 launch cinebot_control policy_inference.launch.py

# Terminal 2: Publish test joint states
ros2 topic pub /joint_states sensor_msgs/JointState "{
    name: ['wheel_left_joint', 'wheel_right_joint', 'base_x_joint', 
           'shoulder_joint', 'elbow_joint', 'wrist_pitch_joint', 
           'wrist_roll_joint', 'wrist_yaw_joint', 'gripper_joint'],
    position: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    velocity: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}" --rate 50

# Terminal 3: Publish test base pose
ros2 topic pub /base_pose geometry_msgs/Pose2D "{x: 1.0, y: 0.0, theta: 0.0}" --rate 50

# Terminal 4: Publish test target
ros2 topic pub /camera_target_pose geometry_msgs/PoseStamped "{
    header: {frame_id: 'world'},
    pose: {
        position: {x: 2.0, y: 0.0, z: 1.0},
        orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
    }
}" --rate 20

# Terminal 5: Monitor output
ros2 topic echo /joint_commands
ros2 topic hz /joint_commands  # Should be ~20 Hz
```

---

## 🛡️ Safety Checks

### Pre-Deployment Checklist
```bash
# 1. Verify joint limits configured
ros2 param list /policy_inference_node | grep limit

# 2. Test emergency stop
# (Implement E-stop handler in robot controller)

# 3. Check collision detection active
# (Verify in robot's motion planning stack)

# 4. Test with conservative scaling first
ros2 launch cinebot_control policy_inference.launch.py  # Edit node for 0.5x scaling

# 5. Monitor for oscillations
ros2 topic echo /joint_commands | tee commands.log
# Analyze commands.log for rapid direction changes

# 6. Verify communication rates
ros2 topic hz /joint_states      # Should be 50 Hz
ros2 topic hz /base_pose         # Should be 50 Hz
ros2 topic hz /camera_target_pose # Should be 20 Hz
ros2 topic hz /joint_commands     # Should be 20 Hz
```

---

## 📝 Troubleshooting Quick Fixes

### Cannot import onnxruntime
```bash
pip3 install --upgrade onnxruntime-gpu
# Or for CPU:
pip3 install --upgrade onnxruntime
```

### CUDA not found
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Add to ~/.bashrc
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

source ~/.bashrc
```

### Node doesn't start
```bash
# Check ROS2 environment
source /opt/ros/humble/setup.bash
source ~/cinebot_ws/install/setup.bash

# Rebuild with verbose output
cd ~/cinebot_ws
colcon build --packages-select cinebot_control --event-handlers console_direct+
```

### Robot moves erratically
```bash
# Emergency: Reduce action scaling
nano ~/cinebot_ws/src/cinebot_control/scripts/ros2_policy_node.py
# Set all scales to 0.25x
colcon build --packages-select cinebot_control
```

### High latency (>10 ms)
```bash
# Check GPU usage
nvidia-smi

# Try CPU provider (slower but stable)
# Edit ros2_policy_node.py:
# providers = ["CPUExecutionProvider"]

# Or install TensorRT
sudo apt install libnvinfer8
```

---

## 📚 Documentation Links

- **Main Deployment Guide:** `deployment/DEPLOYMENT_GUIDE.md`
- **Architecture Details:** `deployment/ARCHITECTURE.md`
- **Quick Start:** `deployment/README.md`
- **Full Package Info:** `deployment/DEPLOYMENT_PACKAGE.md`
- **Training Config:** `scripts/launch_session_7d_accelerated.ps1`

---

**Quick Start Summary:**
1. Transfer files: `scp deployment/* $ORIN_HOST:~/cinebot_ws/`
2. Install deps: `sudo apt install ros-humble-trajectory-msgs && pip3 install onnxruntime-gpu`
3. Build: `cd ~/cinebot_ws && colcon build && source install/setup.bash`
4. Launch: `ros2 launch cinebot_control policy_inference.launch.py`

**Status:** Ready for deployment! ✅
