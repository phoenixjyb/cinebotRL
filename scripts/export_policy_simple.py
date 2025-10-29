#!/usr/bin/env python3
"""
Simplified policy export for deployment (no onnxruntime validation needed).

Usage:
    isaaclab.bat -p scripts/export_policy_simple.py \
        --checkpoint logs/.../ppo_mobile_mm_XXXXX_steps.zip \
        --output deployment/policy_demo.onnx
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO


def export_to_onnx(model: PPO, output_path: Path, obs_dim: int = 74):
    """Export policy to ONNX format."""
    
    print(f"\n{'='*60}")
    print(f"POLICY EXPORT (SIMPLIFIED)")
    print(f"{'='*60}")
    
    print(f"\n[1/4] Extracting policy network from PPO model...")
    policy = model.policy
    
    # Move policy to CPU to avoid device mismatch during export
    device = torch.device("cpu")
    policy.to(device)
    policy.eval()
    
    # Create dummy input on CPU
    dummy_obs = torch.randn(1, obs_dim, dtype=torch.float32, device=device)
    print(f"  Input shape: {dummy_obs.shape}")
    
    print(f"\n[2/4] Creating deterministic policy wrapper...")
    
    class DeterministicPolicy(torch.nn.Module):
        """Wrapper to extract mean actions only (no stochastic sampling)."""
        def __init__(self, policy_net):
            super().__init__()
            self.policy_net = policy_net
            
        def forward(self, obs):
            # Get mean action (deterministic policy for deployment)
            with torch.no_grad():
                features = self.policy_net.extract_features(obs)
                if hasattr(self.policy_net, 'mlp_extractor'):
                    latent_pi = self.policy_net.mlp_extractor.forward_actor(features)
                else:
                    latent_pi = features
                mean_actions = self.policy_net.action_net(latent_pi)
            return mean_actions
    
    deterministic_policy = DeterministicPolicy(policy)
    
    print(f"\n[3/4] Exporting to ONNX (opset 14)...")
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
        },
        verbose=False
    )
    
    print(f"  ✅ ONNX export successful!")
    print(f"  File: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024:.2f} KB")
    
    print(f"\n[4/4] Testing with PyTorch...")
    with torch.no_grad():
        test_obs = torch.randn(1, obs_dim, dtype=torch.float32)
        test_action = deterministic_policy(test_obs)
        print(f"  Test action shape: {test_action.shape}")
        print(f"  Test action range: [{test_action.min():.3f}, {test_action.max():.3f}]")
    
    return output_path


def save_normalization_stats(model: PPO, output_dir: Path):
    """Save observation normalization statistics."""
    
    stats_file = output_dir / "normalization_stats.npz"
    
    print(f"\nSaving normalization statistics...")
    
    # Note: SB3 training without VecNormalize means no normalization needed
    # But we save zeros for consistency with deployment pipeline
    obs_dim = 74
    obs_mean = np.zeros(obs_dim)
    obs_var = np.ones(obs_dim)
    
    np.savez(
        stats_file,
        obs_mean=obs_mean,
        obs_var=obs_var,
        obs_std=np.sqrt(obs_var + 1e-8)
    )
    
    print(f"  ✅ Stats saved: {stats_file}")
    print(f"  (No VecNormalize detected - using identity normalization)")
    
    return stats_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to trained model checkpoint (.zip)")
    parser.add_argument("--output", type=str, required=True,
                       help="Output path for ONNX model")
    parser.add_argument("--obs-dim", type=int, default=74,
                       help="Observation dimension")
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Output: {output_path}")
    
    # Load trained model
    print(f"\nLoading trained PPO model...")
    model = PPO.load(str(checkpoint_path))
    print(f"  ✅ Model loaded")
    
    # Export to ONNX
    exported_path = export_to_onnx(model, output_path, args.obs_dim)
    
    # Save normalization stats
    stats_path = save_normalization_stats(model, output_path.parent)
    
    print(f"\n{'='*60}")
    print(f"EXPORT COMPLETE!")
    print(f"{'='*60}")
    print(f"\nFiles ready for deployment:")
    print(f"  1. Model:  {exported_path}")
    print(f"  2. Stats:  {stats_path}")
    print(f"\nNext steps:")
    print(f"  1. Test locally: python deployment/test_onnx_inference.py")
    print(f"  2. Copy to Orin: scp deployment/* orin@orin-hostname:~/cinebot_ws/")
    print(f"  3. Launch on Orin: ros2 launch cinebot_control policy_inference.launch.py")
    print(f"\nTo use the latest Session 7d model later:")
    print(f"  - Re-run this script with the final checkpoint")
    print(f"  - Overwrite policy_demo.onnx with new export")


if __name__ == "__main__":
    main()
