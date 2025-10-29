#!/bin/bash
# Quick WSL verification script for deployment testing
# Run this in WSL: bash deployment/test_in_wsl.sh

set -e  # Exit on error

echo "=================================================="
echo "WSL Deployment Verification - Quick Test"
echo "=================================================="
echo ""

# Navigate to deployment directory (Windows path accessed from WSL)
cd /mnt/c/Users/yanbo/wSpace/cinebotRL/deployment

echo "[1/6] Checking Python..."
python3 --version
echo "✅ Python available"
echo ""

echo "[2/6] Checking deployment files..."
if [ -f "policy_demo.onnx" ]; then
    echo "✅ policy_demo.onnx found ($(du -h policy_demo.onnx | cut -f1))"
else
    echo "❌ policy_demo.onnx not found!"
    exit 1
fi

if [ -f "normalization_stats.npz" ]; then
    echo "✅ normalization_stats.npz found ($(du -h normalization_stats.npz | cut -f1))"
else
    echo "❌ normalization_stats.npz not found!"
    exit 1
fi
echo ""

echo "[3/6] Installing Python dependencies (if needed)..."
pip3 install --quiet --upgrade pip 2>/dev/null || true
if ! python3 -c "import numpy" 2>/dev/null; then
    echo "Installing numpy..."
    pip3 install --user numpy
fi
if ! python3 -c "import onnxruntime" 2>/dev/null; then
    echo "Installing onnxruntime..."
    pip3 install --user onnxruntime
fi
echo "✅ Dependencies ready"
echo ""

echo "[4/6] Testing ONNX model loading..."
python3 << 'EOF'
import numpy as np
import onnxruntime as ort

try:
    # Load model
    session = ort.InferenceSession(
        "policy_demo.onnx",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    
    provider = session.get_providers()[0]
    print(f"✅ Model loaded successfully!")
    print(f"   Provider: {provider}")
    
    # Get input/output info
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    print(f"   Input:  {input_info.name}, shape={input_info.shape}")
    print(f"   Output: {output_info.name}, shape={output_info.shape}")
    
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    exit(1)
EOF
echo ""

echo "[5/6] Testing inference..."
python3 << 'EOF'
import numpy as np
import onnxruntime as ort
import time

# Load model
session = ort.InferenceSession(
    "policy_demo.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

# Test observation
obs = np.random.randn(1, 74).astype(np.float32)

# Warmup
for _ in range(5):
    _ = session.run(None, {input_name: obs})

# Timed inference
latencies = []
for _ in range(100):
    start = time.perf_counter()
    actions = session.run(None, {input_name: obs})[0]
    latencies.append((time.perf_counter() - start) * 1000)

latencies = np.array(latencies)

print(f"✅ Inference successful!")
print(f"   Action shape: {actions.shape}")
print(f"   Action range: [{actions.min():.3f}, {actions.max():.3f}]")
print(f"   Mean latency: {latencies.mean():.2f} ms")
print(f"   P95 latency:  {np.percentile(latencies, 95):.2f} ms")

if latencies.mean() < 20:
    print(f"   ✅ Latency acceptable for 20 Hz control")
else:
    print(f"   ⚠️  Latency high (CPU inference)")
EOF
echo ""

echo "[6/6] Loading normalization stats..."
python3 << 'EOF'
import numpy as np

try:
    stats = np.load("normalization_stats.npz")
    print(f"✅ Normalization stats loaded!")
    print(f"   obs_mean shape: {stats['obs_mean'].shape}")
    print(f"   obs_var shape: {stats['obs_var'].shape}")
    
    obs_mean = stats['obs_mean']
    obs_var = stats['obs_var']
    print(f"   Mean range: [{obs_mean.min():.3f}, {obs_mean.max():.3f}]")
    print(f"   Var range:  [{obs_var.min():.3f}, {obs_var.max():.3f}]")
    
except Exception as e:
    print(f"❌ Stats loading failed: {e}")
    exit(1)
EOF
echo ""

echo "=================================================="
echo "VERIFICATION COMPLETE ✅"
echo "=================================================="
echo ""
echo "All checks passed! The deployment package is ready."
echo ""
echo "Next steps:"
echo "  1. Check ROS2 installation: which ros2"
echo "  2. Test ROS2 integration: See WSL_VERIFICATION_GUIDE.md"
echo "  3. Deploy to Orin: See COMMANDS.md"
echo ""
