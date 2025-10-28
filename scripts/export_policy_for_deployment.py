#!/usr/bin/env python3
"""
Export trained PPO policy for deployment on NVIDIA Orin with ROS2.

Usage:
    python scripts/export_policy_for_deployment.py \
        --checkpoint logs/sb3/mobilemmtrackee_v0/TIMESTAMP/checkpoints/rl_model_XXXXX_steps.zip \
        --output deployment/policy_session_7d.onnx \
        --format onnx

Supports ONNX (recommended for Orin) or TorchScript export.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
import onnx
import onnxruntime as ort


def export_to_onnx(model: PPO, output_path: Path, obs_dim: int = 74):
    """Export policy to ONNX format for cross-platform deployment."""
    
    print(f"\n[1/5] Extracting policy network from PPO model...")
    policy = model.policy
    policy.eval()
    
    # Create dummy input matching observation space
    dummy_obs = torch.randn(1, obs_dim, dtype=torch.float32)
    
    print(f"[2/5] Tracing policy network...")
    print(f"  Input shape: {dummy_obs.shape}")
    
    # Export only the deterministic action (mean), not stochastic sampling
    class DeterministicPolicy(torch.nn.Module):
        def __init__(self, policy_net):
            super().__init__()
            self.policy_net = policy_net
            
        def forward(self, obs):
            # Get mean action (deterministic policy for deployment)
            with torch.no_grad():
                latent_pi = self.policy_net.mlp_extractor.policy_net(
                    self.policy_net.mlp_extractor.shared_net(obs)
                )
                mean_actions = self.policy_net.action_net(latent_pi)
            return mean_actions
    
    deterministic_policy = DeterministicPolicy(policy)
    
    print(f"[3/5] Exporting to ONNX...")
    torch.onnx.export(
        deterministic_policy,
        dummy_obs,
        str(output_path),
        export_params=True,
        opset_version=14,  # Compatible with NVIDIA TensorRT
        do_constant_folding=True,
        input_names=['observation'],
        output_names=['action'],
        dynamic_axes={
            'observation': {0: 'batch_size'},
            'action': {0: 'batch_size'}
        }
    )
    
    print(f"[4/5] Validating ONNX model...")
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    
    print(f"[5/5] Testing inference...")
    ort_session = ort.InferenceSession(str(output_path))
    test_obs = np.random.randn(1, obs_dim).astype(np.float32)
    ort_inputs = {ort_session.get_inputs()[0].name: test_obs}
    ort_outs = ort_session.run(None, ort_inputs)
    
    print(f"\n✅ ONNX export successful!")
    print(f"  Output: {output_path}")
    print(f"  Input shape: (batch, {obs_dim})")
    print(f"  Output shape: {ort_outs[0].shape}")
    print(f"  Model size: {output_path.stat().st_size / 1024:.2f} KB")
    
    return output_path


def export_to_torchscript(model: PPO, output_path: Path, obs_dim: int = 74):
    """Export policy to TorchScript (alternative, requires PyTorch on Orin)."""
    
    print(f"\n[1/3] Extracting policy network...")
    policy = model.policy
    policy.eval()
    
    dummy_obs = torch.randn(1, obs_dim, dtype=torch.float32)
    
    print(f"[2/3] Tracing with TorchScript...")
    
    class DeterministicPolicy(torch.nn.Module):
        def __init__(self, policy_net):
            super().__init__()
            self.policy_net = policy_net
            
        def forward(self, obs):
            with torch.no_grad():
                latent_pi = self.policy_net.mlp_extractor.policy_net(
                    self.policy_net.mlp_extractor.shared_net(obs)
                )
                mean_actions = self.policy_net.action_net(latent_pi)
            return mean_actions
    
    deterministic_policy = DeterministicPolicy(policy)
    traced_script = torch.jit.trace(deterministic_policy, dummy_obs)
    
    print(f"[3/3] Saving TorchScript model...")
    traced_script.save(str(output_path))
    
    print(f"\n✅ TorchScript export successful!")
    print(f"  Output: {output_path}")
    print(f"  Model size: {output_path.stat().st_size / 1024:.2f} KB")
    
    return output_path


def save_normalization_stats(model: PPO, output_dir: Path):
    """Save observation normalization statistics for deployment."""
    
    stats_file = output_dir / "normalization_stats.npz"
    
    # Extract VecNormalize statistics if present
    if hasattr(model, 'get_vec_normalize_env'):
        vec_norm = model.get_vec_normalize_env()
        if vec_norm is not None:
            obs_mean = vec_norm.obs_rms.mean
            obs_var = vec_norm.obs_rms.var
            print(f"\n✅ Saving VecNormalize statistics...")
        else:
            print(f"\n⚠️  No VecNormalize found, using zero normalization")
            obs_mean = np.zeros(74)
            obs_var = np.ones(74)
    else:
        print(f"\n⚠️  No VecNormalize found, using zero normalization")
        obs_mean = np.zeros(74)
        obs_var = np.ones(74)
    
    np.savez(
        stats_file,
        obs_mean=obs_mean,
        obs_var=obs_var,
        obs_std=np.sqrt(obs_var + 1e-8)
    )
    
    print(f"  Saved to: {stats_file}")
    return stats_file


def main():
    parser = argparse.ArgumentParser(description="Export trained policy for deployment")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to trained model checkpoint (.zip)")
    parser.add_argument("--output", type=str, required=True,
                       help="Output path for exported model")
    parser.add_argument("--format", type=str, choices=["onnx", "torchscript"], 
                       default="onnx",
                       help="Export format (onnx recommended for Orin)")
    parser.add_argument("--obs-dim", type=int, default=74,
                       help="Observation dimension")
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"POLICY EXPORT FOR DEPLOYMENT")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_path}")
    print(f"Format: {args.format}")
    
    # Load trained model
    print(f"\nLoading trained PPO model...")
    model = PPO.load(str(checkpoint_path))
    
    # Export based on format
    if args.format == "onnx":
        exported_path = export_to_onnx(model, output_path, args.obs_dim)
    else:
        exported_path = export_to_torchscript(model, output_path, args.obs_dim)
    
    # Save normalization stats
    stats_path = save_normalization_stats(model, output_path.parent)
    
    print(f"\n{'='*60}")
    print(f"EXPORT COMPLETE!")
    print(f"{'='*60}")
    print(f"\nFiles ready for deployment:")
    print(f"  1. Model: {exported_path}")
    print(f"  2. Stats: {stats_path}")
    print(f"\nNext steps:")
    print(f"  1. Copy files to NVIDIA Orin")
    print(f"  2. Install ROS2 policy inference node")
    print(f"  3. Configure ROS2 topics in launch file")
    print(f"  4. Test with: ros2 launch cinebot_control policy_inference.launch.py")


if __name__ == "__main__":
    main()
