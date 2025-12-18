#!/usr/bin/env python3
"""
Export the PyBullet mobile manipulator PPO policy (linux_env_dev) to ONNX for
NVIDIA Orin / TensorRT deployment.

Example:
    python pybulletDeploy/export_pybullet_policy.py \
        --checkpoint linux_env_dev/models/logs_20251126_112856/ppo_mobile_mm_final.zip \
        --output pybulletDeploy/policy_mobile_mm.onnx \
        --obs-dim 53 --action-dim 8
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO

# Alias for checkpoints saved with numpy>=2 (uses numpy._core.* modules)
try:
    import numpy._core as _np_core  # type: ignore[attr-defined]
except Exception:
    try:
        import numpy.core as _np_core  # type: ignore
    except Exception:
        _np_core = None  # type: ignore

if _np_core is not None:
    try:
        sys.modules.setdefault("numpy._core", _np_core)
        sys.modules.setdefault("numpy._core.numeric", _np_core.numeric)
    except Exception:
        pass


# Ensure custom env/modules from linux_env_dev are importable when loading the checkpoint
REPO_ROOT = Path(__file__).resolve().parents[1]
PYBULLET_ENV_PATH = REPO_ROOT / "linux_env_dev"
if PYBULLET_ENV_PATH.exists():
    # Prepend so our local pybullet_envs shadows the pip package of the same name.
    sys.path.insert(0, str(PYBULLET_ENV_PATH))


def build_deterministic_policy(policy: torch.nn.Module) -> torch.nn.Module:
    """Wrap SB3 policy to output mean actions only (deployment-friendly)."""

    class DeterministicPolicy(torch.nn.Module):
        def __init__(self, policy_net: torch.nn.Module):
            super().__init__()
            self.policy_net = policy_net

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            # SB3 takes flattened obs; we only need the actor path
            with torch.no_grad():
                features = self.policy_net.extract_features(obs)
                if hasattr(self.policy_net, "mlp_extractor"):
                    latent_pi = self.policy_net.mlp_extractor.forward_actor(features)
                else:
                    latent_pi = features
                mean_actions = self.policy_net.action_net(latent_pi)
            # SB3 clips actions to the Box bounds before returning them to the env.
            # Our env action spaces are [-1, 1], so clamp here for deployment safety.
            return torch.clamp(mean_actions, -1.0, 1.0)

    return DeterministicPolicy(policy)


def export_to_onnx(
    model: PPO,
    output_path: Path,
    obs_dim: int = 53,
    action_dim: int | None = 8,
    opset: int = 14,
    device: str = "cpu",
) -> Path:
    """Export PPO policy to ONNX (dynamic batch axis)."""
    policy = model.policy
    policy.to(device)
    policy.eval()

    dummy_obs = torch.randn(1, obs_dim, dtype=torch.float32, device=device)
    det_policy = build_deterministic_policy(policy)

    print(f"\n[1/3] Exporting to ONNX (opset {opset}, dynamic batch)...")
    torch.onnx.export(
        det_policy,
        dummy_obs,
        str(output_path),
        export_params=True,
        opset_version=int(opset),
        do_constant_folding=True,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch_size"}, "action": {0: "batch_size"}},
        verbose=False,
    )

    # Lightweight validation (optional if onnx is installed)
    try:
        import onnx

        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("  ✅ ONNX graph check passed")
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"  ⚠️  Skipping ONNX checker or validation failed: {exc}")

    with torch.no_grad():
        torch_out = det_policy(dummy_obs)
    print(f"  Output tensor shape: {tuple(torch_out.shape)}")
    if action_dim is not None and torch_out.shape[1] != action_dim:
        print(
            f"  ⚠️  Action dim mismatch: expected {action_dim}, got {torch_out.shape[1]}"
        )

    print(f"[2/3] Saved ONNX to: {output_path} ({output_path.stat().st_size/1024:.1f} KB)")
    return output_path


def save_normalization_stats(model: PPO, stats_file: Path, obs_dim: int) -> Path:
    """Save observation normalization stats (VecNormalize if present, else identity)."""
    stats_file = Path(stats_file)
    stats_file.parent.mkdir(parents=True, exist_ok=True)

    obs_mean = np.zeros(obs_dim, dtype=np.float32)
    obs_var = np.ones(obs_dim, dtype=np.float32)

    if hasattr(model, "get_vec_normalize_env"):
        vec_norm = model.get_vec_normalize_env()
        if vec_norm is not None and hasattr(vec_norm, "obs_rms"):
            obs_mean = vec_norm.obs_rms.mean
            obs_var = vec_norm.obs_rms.var
            print("  ✅ VecNormalize stats detected")
        else:
            print("  ℹ️  No VecNormalize stats; using identity normalization")
    else:
        print("  ℹ️  No VecNormalize stats; using identity normalization")

    np.savez(
        stats_file,
        obs_mean=obs_mean,
        obs_var=obs_var,
        obs_std=np.sqrt(obs_var + 1e-8),
    )
    print(f"[3/3] Saved normalization stats to: {stats_file}")
    return stats_file


def maybe_test_with_ort(model_path: Path, obs_dim: int) -> None:
    """Try a single inference with onnxruntime if available."""
    try:
        import onnxruntime as ort
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"  ℹ️  onnxruntime not installed; skip runtime test ({exc})")
        return

    providers = ort.get_available_providers()
    provider_pref = (
        "CUDAExecutionProvider" if "CUDAExecutionProvider" in providers else "CPUExecutionProvider"
    )
    try:
        session = ort.InferenceSession(str(model_path), providers=[provider_pref])
    except Exception as exc:
        print(f"  ⚠️  onnxruntime session failed on {provider_pref}: {exc}")
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        provider_pref = "CPUExecutionProvider"

    dummy = np.random.randn(1, obs_dim).astype(np.float32)
    outputs = session.run(None, {session.get_inputs()[0].name: dummy})[0]
    actual = session.get_providers()[0] if session.get_providers() else provider_pref
    print(
        f"  ✅ onnxruntime inference ok on {actual}, output shape={outputs.shape}, "
        f"range=({outputs.min():.3f},{outputs.max():.3f})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PyBullet PPO policy to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to SB3 checkpoint (.zip)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output ONNX path",
    )
    parser.add_argument(
        "--obs-dim",
        type=int,
        default=-1,
        help="Observation dimension. Use -1 to infer from checkpoint (recommended).",
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=-1,
        help="Action dimension. Use -1 to infer from checkpoint (recommended).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=None,
        help="Output path for normalization stats (.npz). Default: <output_stem>_stats.npz",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for export (cpu or cuda)",
    )
    parser.add_argument(
        "--skip-ort",
        action="store_true",
        help="Skip onnxruntime sanity test after export",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PYBULLET POLICY EXPORT → ONNX")
    print("=" * 60)
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Output     : {output_path}")
    print(f"Opset      : {args.opset}")
    print(f"Device     : {args.device}")

    model = PPO.load(str(args.checkpoint), device=args.device)
    print("  ✅ SB3 checkpoint loaded")

    def _infer_dim(space, label: str) -> int:
        shape = getattr(space, "shape", None)
        if not shape or len(shape) != 1:
            raise ValueError(f"Cannot infer {label} dim from space={space!r}")
        return int(shape[0])

    obs_dim = int(args.obs_dim)
    if obs_dim <= 0:
        obs_dim = _infer_dim(getattr(model, "observation_space", None), "obs")
    action_dim = int(args.action_dim)
    if action_dim <= 0:
        action_dim = _infer_dim(getattr(model, "action_space", None), "action")

    stats_output = args.stats_output
    if stats_output is None:
        stats_output = output_path.parent / f"{output_path.stem}_stats.npz"

    print(f"Obs dim    : {obs_dim}")
    print(f"Action dim : {action_dim}")
    print(f"Stats out  : {stats_output}")

    exported = export_to_onnx(
        model,
        output_path,
        obs_dim=obs_dim,
        action_dim=action_dim,
        opset=int(args.opset),
        device=args.device,
    )
    stats = save_normalization_stats(model, stats_output, obs_dim)

    if not args.skip_ort:
        maybe_test_with_ort(exported, obs_dim)

    print("\nDONE. Files ready for transfer:")
    print(f" - ONNX: {exported}")
    print(f" - Stats: {stats}")


if __name__ == "__main__":
    main()
