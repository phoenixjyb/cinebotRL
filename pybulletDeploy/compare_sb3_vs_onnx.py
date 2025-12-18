#!/usr/bin/env python3
"""
Compare SB3 deterministic actions vs exported ONNX actions for the same observations.

Examples:
  # recomo (obs_dim=51, action_dim=5)
  python pybulletDeploy/compare_sb3_vs_onnx.py \\
    --checkpoint linux_env_dev/models/logs_20251217_184738_recomo/best_model/best_model.zip \\
    --onnx pybulletDeploy/policy_recomo.onnx \\
    --n_samples 200

  # mobile_mm (obs_dim=53, action_dim=8)
  python pybulletDeploy/compare_sb3_vs_onnx.py \\
    --checkpoint linux_env_dev/models/logs_20251126_112856/ppo_mobile_mm_final.zip \\
    --onnx pybulletDeploy/policy_mobile_mm.onnx \\
    --n_samples 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _ensure_linux_env_dev_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    linux_env_dev = repo_root / "linux_env_dev"
    if linux_env_dev.exists():
        sys.path.insert(0, str(linux_env_dev))


def _infer_1d_space_dim(space, label: str) -> int:
    shape = getattr(space, "shape", None)
    if not shape or len(shape) != 1:
        raise ValueError(f"Cannot infer {label} dim from space={space!r}")
    return int(shape[0])


def compare(
    checkpoint: Path,
    onnx_path: Path,
    n_samples: int = 256,
    seed: int = 0,
    tol: float = 1e-3,
) -> bool:
    _ensure_linux_env_dev_on_path()

    from stable_baselines3 import PPO

    try:
        import onnxruntime as ort
    except Exception as exc:
        raise RuntimeError("onnxruntime is required. Install onnxruntime-gpu or onnxruntime.") from exc

    model = PPO.load(str(checkpoint), device="cpu")
    obs_dim = _infer_1d_space_dim(getattr(model, "observation_space", None), "obs")
    act_dim = _infer_1d_space_dim(getattr(model, "action_space", None), "action")

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    rng = np.random.default_rng(int(seed))
    obs_batch = rng.standard_normal((int(n_samples), obs_dim), dtype=np.float32)

    sb3_actions = np.zeros((int(n_samples), act_dim), dtype=np.float32)
    onnx_actions = np.zeros((int(n_samples), act_dim), dtype=np.float32)

    for i in range(int(n_samples)):
        obs = obs_batch[i]
        a_sb3, _ = model.predict(obs, deterministic=True)
        sb3_actions[i] = np.asarray(a_sb3, dtype=np.float32).reshape(-1)

        out = sess.run(None, {input_name: obs.reshape(1, -1)})[0]
        onnx_actions[i] = np.asarray(out, dtype=np.float32).reshape(-1)

    abs_err = np.abs(sb3_actions - onnx_actions)
    max_abs = float(np.max(abs_err))
    mean_abs = float(np.mean(abs_err))
    p99 = float(np.percentile(abs_err, 99))

    print("=" * 60)
    print("SB3 vs ONNX ACTION PARITY")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint}")
    print(f"ONNX      : {onnx_path}")
    print(f"obs_dim={obs_dim}, act_dim={act_dim}, n_samples={n_samples}, seed={seed}")
    print(f"abs_err: mean={mean_abs:.6g}, p99={p99:.6g}, max={max_abs:.6g}, tol={tol}")

    ok = bool(max_abs <= float(tol))
    print("OK" if ok else "FAIL")
    return ok


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare SB3 deterministic actions vs ONNX exported actions.")
    p.add_argument("--checkpoint", type=Path, required=True, help="SB3 .zip checkpoint")
    p.add_argument("--onnx", type=Path, required=True, help="Exported ONNX file")
    p.add_argument("--n_samples", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tol", type=float, default=1e-3, help="Max-abs tolerance for parity")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ok = compare(
        checkpoint=args.checkpoint,
        onnx_path=args.onnx,
        n_samples=int(args.n_samples),
        seed=int(args.seed),
        tol=float(args.tol),
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

