# WSL Pre-Deployment Verification Guide

Quick validation steps to test the deployment package in WSL before deploying to Orin.

---

## 🎯 Why Test in WSL First?

- ✅ Same Linux environment as Orin (Ubuntu + ROS2 Humble)
- ✅ Fast iteration (no file transfer, direct filesystem access)
- ✅ Risk-free debugging (not on actual robot)
- ✅ GPU testing if NVIDIA GPU available (CUDA via WSL2)
- ✅ Catch dependency/compatibility issues early

**Time Investment:** 30-60 minutes  
**Time Saved:** Hours of remote debugging on Orin

---

## 📋 Prerequisites

### Check WSL2 Installation
```powershell
# On Windows
wsl --version
wsl --list --verbose
# Should show WSL 2 and Ubuntu distribution
```

### Install/Update Ubuntu in WSL
```powershell
# If not installed
wsl --install -d Ubuntu-22.04

# If already installed, ensure it's WSL2
wsl --set-version Ubuntu-22.04 2
```

---

## 🚀 Step-by-Step Verification

### Step 1: Access Deployment Files in WSL
```bash
# Launch WSL
wsl

# Navigate to project (Windows drives mounted at /mnt/)
cd /mnt/c/Users/yanbo/wSpace/cinebotRL/deployment

# Verify files present
ls -lh
# Should see: policy_demo.onnx, normalization_stats.npz, ros2_policy_node.py, etc.
```

### Step 2: Install ROS2 Humble (If Not Already Installed)
```bash
# Check if ROS2 installed
which ros2

# If not installed, follow quick install:
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install ros-humble-desktop ros-humble-trajectory-msgs -y

# Source ROS2
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### Step 3: Install Python Dependencies
```bash
# Basic dependencies
sudo apt install python3-pip python3-numpy -y

# ONNX Runtime (CPU version for quick test)
pip3 install onnxruntime numpy

# Optional: GPU version if you have NVIDIA GPU
# pip3 install onnxruntime-gpu

# Verify installation
python3 -c "import onnxruntime as ort; print(f'ONNX Runtime {ort.__version__}'); print(f'Providers: {ort.get_available_providers()}')"
```

### Step 4: Test ONNX Model Loading
```bash
# Navigate to deployment directory
cd /mnt/c/Users/yanbo/wSpace/cinebotRL/deployment

# Create quick test script
cat > test_model_load.py << 'EOF'
import onnxruntime as ort
import numpy as np

print("Testing ONNX model loading...")

# Load model
session = ort.InferenceSession(
    "policy_demo.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

print(f"✅ Model loaded successfully!")
print(f"   Providers: {session.get_providers()}")

# Get input/output info
input_info = session.get_inputs()[0]
output_info = session.get_outputs()[0]
print(f"   Input:  {input_info.name}, shape={input_info.shape}")
print(f"   Output: {output_info.name}, shape={output_info.shape}")

# Test inference
obs = np.random.randn(1, 74).astype(np.float32)
actions = session.run(None, {input_info.name: obs})[0]
print(f"✅ Inference successful!")
print(f"   Action shape: {actions.shape}")
print(f"   Action range: [{actions.min():.3f}, {actions.max():.3f}]")

# Load normalization stats
stats = np.load("normalization_stats.npz")
print(f"✅ Normalization stats loaded!")
print(f"   Obs mean shape: {stats['obs_mean'].shape}")
print(f"   Obs var shape: {stats['obs_var'].shape}")
EOF

python3 test_model_load.py
```

### Step 5: Test ROS2 Integration (Minimal)
```bash
# Create minimal ROS2 workspace
mkdir -p ~/cinebot_ws_test/src
cd ~/cinebot_ws_test/src

# Create package
ros2 pkg create cinebot_control_test --build-type ament_python \
    --dependencies rclpy sensor_msgs geometry_msgs trajectory_msgs

# Copy files
mkdir -p cinebot_control_test/scripts cinebot_control_test/launch
cp /mnt/c/Users/yanbo/wSpace/cinebotRL/deployment/ros2_policy_node.py cinebot_control_test/scripts/
cp /mnt/c/Users/yanbo/wSpace/cinebotRL/deployment/policy_inference.launch.py cinebot_control_test/launch/
chmod +x cinebot_control_test/scripts/ros2_policy_node.py

# Update setup.py
cat > cinebot_control_test/setup.py << 'EOF'
from setuptools import setup
import os
from glob import glob

package_name = 'cinebot_control_test'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Test User',
    maintainer_email='test@test.com',
    description='Test deployment',
    license='MIT',
    entry_points={
        'console_scripts': [
            'policy_inference = cinebot_control_test.ros2_policy_node:main'
        ],
    },
)
EOF

# Build
cd ~/cinebot_ws_test
colcon build --packages-select cinebot_control_test
source install/setup.bash

# Check if package is recognized
ros2 pkg list | grep cinebot_control_test
```

### Step 6: Test Node Launch (Dry Run)
```bash
# This will fail because topics aren't publishing, but verifies node loads
source ~/cinebot_ws_test/install/setup.bash

# Try launching (Ctrl+C after you see it start)
ros2 launch cinebot_control_test policy_inference.launch.py \
    model_path:=/mnt/c/Users/yanbo/wSpace/cinebotRL/deployment/policy_demo.onnx \
    stats_path:=/mnt/c/Users/yanbo/wSpace/cinebotRL/deployment/normalization_stats.npz

# Expected: Node starts, waits for topics (this is SUCCESS)
# If errors, debug before deploying to Orin
```

### Step 7: Benchmark Inference Latency
```bash
cd /mnt/c/Users/yanbo/wSpace/cinebotRL/deployment

cat > benchmark_latency.py << 'EOF'
import time
import onnxruntime as ort
import numpy as np

print("Benchmarking inference latency...")

# Load model
session = ort.InferenceSession(
    "policy_demo.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

print(f"Provider: {session.get_providers()[0]}")

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
print(f"\nLatency Statistics (1000 iterations):")
print(f"  Mean:   {latencies.mean():.2f} ms")
print(f"  Median: {np.median(latencies):.2f} ms")
print(f"  P95:    {np.percentile(latencies, 95):.2f} ms")
print(f"  Min:    {latencies.min():.2f} ms")
print(f"  Max:    {latencies.max():.2f} ms")

if session.get_providers()[0] == "CPUExecutionProvider":
    print(f"\n⚠️  Using CPU (WSL). Orin with GPU will be faster!")
else:
    print(f"\n✅ Using GPU! Orin should have similar latency.")
EOF

python3 benchmark_latency.py
```

---

## ✅ Verification Checklist

After completing the steps above, verify:

- [ ] ONNX Runtime imports successfully
- [ ] Model loads without errors (policy_demo.onnx)
- [ ] Normalization stats load (normalization_stats.npz)
- [ ] Test inference runs (random observation → action output)
- [ ] Action shape is (1, 8)
- [ ] Action range is reasonable (approx [-1, 1])
- [ ] ROS2 package builds without errors
- [ ] Inference node launches (even if waiting for topics)
- [ ] No Python import errors
- [ ] Latency benchmark completes

**If all checks pass:** ✅ Ready to deploy to Orin!  
**If any fail:** 🐛 Debug in WSL, much easier than on Orin

---

## 🎯 Expected Results

### CPU Inference (WSL without GPU)
- **Latency:** 5-20 ms (acceptable for 20 Hz control)
- **Provider:** CPUExecutionProvider
- **Orin Improvement:** 2-5x faster with GPU

### GPU Inference (WSL with NVIDIA GPU)
- **Latency:** 2-8 ms (similar to Orin expectations)
- **Provider:** CUDAExecutionProvider
- **Orin Similar:** ARM64 vs x86_64 shouldn't matter much for inference

---

## 🐛 Common Issues & Fixes

### Issue: "No module named 'onnxruntime'"
```bash
pip3 install onnxruntime
# If that fails:
pip3 install --user onnxruntime
```

### Issue: ROS2 commands not found
```bash
source /opt/ros/humble/setup.bash
# Or if using different ROS2 version:
source /opt/ros/<distro>/setup.bash
```

### Issue: colcon build fails
```bash
# Install colcon
sudo apt install python3-colcon-common-extensions

# Check for missing dependencies
rosdep install --from-paths src --ignore-src -r -y
```

### Issue: "CUDA not available" in WSL
```bash
# Check NVIDIA drivers on Windows
nvidia-smi  # (in PowerShell)

# In WSL, check CUDA access
nvidia-smi  # Should show GPU

# If not working, update WSL kernel
wsl --update
```

### Issue: Model file not found
```bash
# Check Windows path is correctly mapped
ls /mnt/c/Users/yanbo/wSpace/cinebotRL/deployment/

# If not found, check drive letter and path
pwd
df -h
```

---

## 🚀 After Successful WSL Verification

Once all tests pass in WSL:

1. **Package for Orin:**
   ```bash
   # Create deployment tarball
   cd /mnt/c/Users/yanbo/wSpace/cinebotRL
   tar -czf cinebot_deployment.tar.gz deployment/
   ```

2. **Transfer to Orin:**
   ```bash
   # From WSL or Windows
   scp cinebot_deployment.tar.gz orin@orin-ip:~/
   ```

3. **Deploy on Orin:**
   ```bash
   # On Orin
   tar -xzf cinebot_deployment.tar.gz
   # Follow same installation steps as WSL
   # Key difference: Use onnxruntime-gpu for CUDA acceleration
   ```

4. **Confidence Level:** 🚀🚀🚀
   - You know the model works
   - You know the ROS2 integration builds
   - You've debugged all import/dependency issues
   - Only variable is Orin hardware (which should perform better)

---

## 📊 Time Investment vs. Savings

| Activity | Time | Value |
|----------|------|-------|
| **WSL Setup** | 30-60 min | One-time investment |
| **Verification** | 15-30 min | Catch 90% of issues |
| **Total WSL Time** | 45-90 min | Controlled environment |
| **vs.** | | |
| **Direct Orin Deploy** | Variable | Unknown issues |
| **Orin Remote Debug** | Hours | Slower iteration |
| **Network Latency** | Annoying | SCP, SSH, rebuild cycles |

**Recommendation:** Spend 1 hour in WSL, save 3-5 hours of Orin debugging.

---

## 💡 Pro Tips

1. **Keep WSL Environment:** Even after Orin deployment, WSL is useful for:
   - Quick model exports and tests
   - Offline development when Orin unavailable
   - Rapid prototyping without robot risk

2. **Use VS Code Remote-WSL:**
   - Install "Remote - WSL" extension
   - Edit files with full IDE features
   - Integrated terminal for commands
   - Best of both worlds (Windows + Linux)

3. **GPU Pass-through:**
   - If you have NVIDIA GPU, use it in WSL2
   - Windows 11 supports native GPU access in WSL
   - Test CUDA path thoroughly before Orin

4. **Automated Testing:**
   - Save verification scripts for future updates
   - Create `test_deployment.sh` with all checks
   - Run before each new model export

---

**Bottom Line:** Yes, absolutely use WSL for verification! It's the smart way to de-risk your Orin deployment.
