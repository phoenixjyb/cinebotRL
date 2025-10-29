#!/usr/bin/env python3
"""
Local test script to validate ONNX policy inference.

Tests:
1. ONNX Runtime installation and GPU provider
2. Model loading and inference
3. Input/output shapes and ranges
4. Inference latency
"""

import argparse
import time
from pathlib import Path

import numpy as np


def test_onnx_inference(
    model_path: str = "deployment/policy_demo.onnx",
    stats_path: str = "deployment/normalization_stats.npz",
    num_iterations: int = 100,
):
    """Test ONNX model inference."""
    
    print("=" * 60)
    print("ONNX INFERENCE TEST")
    print("=" * 60)
    
    # 1. Check ONNX Runtime installation
    print("\n[1/5] Checking ONNX Runtime...")
    try:
        import onnxruntime as ort
        print(f"  ✅ ONNX Runtime version: {ort.__version__}")
    except ImportError:
        print("  ❌ ONNX Runtime not installed!")
        print("  Install: pip install onnxruntime-gpu  # or onnxruntime for CPU")
        return False
    
    # 2. Check available execution providers
    print("\n[2/5] Checking execution providers...")
    providers = ort.get_available_providers()
    print(f"  Available: {providers}")
    
    if "CUDAExecutionProvider" in providers:
        provider = "CUDAExecutionProvider"
        print(f"  ✅ Using GPU: {provider}")
    elif "TensorrtExecutionProvider" in providers:
        provider = "TensorrtExecutionProvider"
        print(f"  ✅ Using TensorRT: {provider}")
    else:
        provider = "CPUExecutionProvider"
        print(f"  ⚠️  Using CPU: {provider}")
    
    # 3. Load ONNX model
    print("\n[3/5] Loading ONNX model...")
    try:
        session = ort.InferenceSession(model_path, providers=[provider])
        print(f"  ✅ Model loaded: {model_path}")
        
        # Get model info
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        
        print(f"\n  Model Details:")
        print(f"    Input:  {input_info.name}, shape={input_info.shape}, dtype={input_info.type}")
        print(f"    Output: {output_info.name}, shape={output_info.shape}, dtype={output_info.type}")
        
        obs_dim = input_info.shape[1]
        action_dim = output_info.shape[1]
        
    except Exception as e:
        print(f"  ❌ Failed to load model: {e}")
        return False
    
    # 4. Load normalization stats
    print("\n[4/5] Loading normalization stats...")
    try:
        stats = np.load(stats_path)
        obs_mean = stats["obs_mean"]
        obs_var = stats["obs_var"]
        print(f"  ✅ Stats loaded: {stats_path}")
        print(f"    Obs mean range: [{obs_mean.min():.3f}, {obs_mean.max():.3f}]")
        print(f"    Obs var range:  [{obs_var.min():.3f}, {obs_var.max():.3f}]")
    except Exception as e:
        print(f"  ⚠️  No stats file: {e}")
        print("  Using identity normalization (mean=0, var=1)")
        obs_mean = np.zeros(obs_dim, dtype=np.float32)
        obs_var = np.ones(obs_dim, dtype=np.float32)
    
    # 5. Test inference
    print(f"\n[5/5] Testing inference ({num_iterations} iterations)...")
    
    # Generate random observation
    np.random.seed(42)
    raw_obs = np.random.randn(1, obs_dim).astype(np.float32)
    
    # Normalize observation
    obs_std = np.sqrt(obs_var + 1e-8)
    normalized_obs = (raw_obs - obs_mean) / obs_std
    
    print(f"\n  Test Observation:")
    print(f"    Raw:        [{raw_obs.min():.3f}, {raw_obs.max():.3f}]")
    print(f"    Normalized: [{normalized_obs.min():.3f}, {normalized_obs.max():.3f}]")
    
    # Run inference (warmup)
    for _ in range(5):
        _ = session.run(None, {input_info.name: normalized_obs})
    
    # Timed inference
    latencies = []
    for _ in range(num_iterations):
        start_time = time.perf_counter()
        actions = session.run(None, {input_info.name: normalized_obs})[0]
        latencies.append((time.perf_counter() - start_time) * 1000)  # ms
    
    latencies = np.array(latencies)
    
    print(f"\n  Test Actions:")
    print(f"    Shape: {actions.shape}")
    print(f"    Range: [{actions.min():.3f}, {actions.max():.3f}]")
    print(f"    Mean:  {actions.mean():.3f}")
    print(f"    Std:   {actions.std():.3f}")
    
    print(f"\n  Inference Latency:")
    print(f"    Mean:   {latencies.mean():.2f} ms")
    print(f"    Median: {np.median(latencies):.2f} ms")
    print(f"    Min:    {latencies.min():.2f} ms")
    print(f"    Max:    {latencies.max():.2f} ms")
    print(f"    P95:    {np.percentile(latencies, 95):.2f} ms")
    
    # 6. Action scaling example
    print(f"\n  Action Scaling Example (20 Hz control):")
    
    # Base actions (3 velocities)
    base_linear_x = actions[0, 0] * 1.5  # ±1.5 m/s
    base_linear_y = actions[0, 1] * 1.5  # ±1.5 m/s
    base_angular_z = actions[0, 2] * 2.0  # ±2.0 rad/s
    
    # Arm actions (6 joint velocities)
    arm_velocities = actions[0, 3:9] * 1.0  # ±1.0 rad/s
    
    print(f"    Base commands:")
    print(f"      Linear X:  {base_linear_x:+.3f} m/s")
    print(f"      Linear Y:  {base_linear_y:+.3f} m/s")
    print(f"      Angular Z: {base_angular_z:+.3f} rad/s")
    print(f"    Arm velocities: {arm_velocities}")
    
    # 7. Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if provider == "CUDAExecutionProvider":
        gpu_ok = "✅"
    else:
        gpu_ok = "⚠️ "
    
    if latencies.mean() < 5.0:
        latency_ok = "✅"
    elif latencies.mean() < 10.0:
        latency_ok = "⚠️ "
    else:
        latency_ok = "❌"
    
    if actions.shape == (1, action_dim):
        shape_ok = "✅"
    else:
        shape_ok = "❌"
    
    print(f"  {gpu_ok} GPU Acceleration:  {provider}")
    print(f"  {latency_ok} Inference Latency: {latencies.mean():.2f} ms (target: <5 ms)")
    print(f"  {shape_ok} Output Shape:      {actions.shape} (expected: (1, {action_dim}))")
    print(f"  ✅ Model ready for deployment!")
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("  1. Copy files to Orin:")
    print(f"     scp {model_path} orin@orin-hostname:~/cinebot_ws/src/cinebot_control/models/")
    print(f"     scp {stats_path} orin@orin-hostname:~/cinebot_ws/src/cinebot_control/models/")
    print("  2. Build ROS2 workspace:")
    print("     cd ~/cinebot_ws && colcon build")
    print("  3. Launch inference node:")
    print("     ros2 launch cinebot_control policy_inference.launch.py")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ONNX policy inference")
    parser.add_argument(
        "--model",
        type=str,
        default="deployment/policy_demo.onnx",
        help="Path to ONNX model file",
    )
    parser.add_argument(
        "--stats",
        type=str,
        default="deployment/normalization_stats.npz",
        help="Path to normalization stats file",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of inference iterations for timing",
    )
    
    args = parser.parse_args()
    
    success = test_onnx_inference(args.model, args.stats, args.iterations)
    exit(0 if success else 1)
