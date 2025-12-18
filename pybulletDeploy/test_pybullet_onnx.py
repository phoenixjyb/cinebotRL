#!/usr/bin/env python3
"""
Sanity-check ONNX export of the PyBullet PPO policy (mobile_mm or recomo).

Examples:
    python pybulletDeploy/test_pybullet_onnx.py \
        --model pybulletDeploy/policy_mobile_mm.onnx \
        --stats pybulletDeploy/policy_mobile_mm_stats.npz

    python pybulletDeploy/test_pybullet_onnx.py \
        --model pybulletDeploy/policy_recomo.onnx \
        --stats pybulletDeploy/policy_recomo_stats.npz --robot recomo
"""

import argparse
import time
from pathlib import Path
import os
import sys

import numpy as np


def _venv_cuda_lib_dirs(venv_dir: Path) -> list[str]:
    lib_dirs = list((venv_dir / "lib").glob("python*/site-packages/nvidia/*/lib"))
    return [str(p) for p in lib_dirs if p.is_dir()]


def ensure_venv_cuda_ld_library_path():
    """Ensure CUDA/cuDNN libs from repo-local `.venv` are on LD_LIBRARY_PATH.

    Notes:
    - ORT loads provider shared libs via dlopen, and on some systems the dynamic
      loader won't pick up LD_LIBRARY_PATH changes made *after* process start.
      We detect and self-reexec once with an updated LD_LIBRARY_PATH so CUDA EP
      can find libcudnn.so.9, etc.
    """
    repo_root = Path(__file__).resolve().parents[1]
    venv_dir = repo_root / ".venv"
    if not venv_dir.exists():
        return
    cuda_libs = _venv_cuda_lib_dirs(venv_dir)
    if not cuda_libs:
        return

    current = os.environ.get("LD_LIBRARY_PATH", "")
    current_parts = [p for p in current.split(":") if p]
    if all(p in current_parts for p in cuda_libs):
        return

    # Avoid infinite recursion.
    if os.environ.get("_CINEBOTRL_LD_REEXEC", "") == "1":
        return

    os.environ["_CINEBOTRL_LD_REEXEC"] = "1"
    os.environ["LD_LIBRARY_PATH"] = ":".join(cuda_libs + current_parts)
    os.execv(sys.executable, [sys.executable] + sys.argv)


def test_onnx(
    model_path: Path,
    stats_path: Path | None,
    provider: str = "auto",
    robot: str = "auto",
    num_iterations: int = 100,
    frame_skip: int = 24,
) -> bool:
    print("=" * 60)
    print("PYBULLET ONNX INFERENCE TEST")
    print("=" * 60)

    # 1) load onnxruntime
    ensure_venv_cuda_ld_library_path()
    try:
        import onnxruntime as ort
    except ImportError:
        print("❌ onnxruntime not installed. pip install onnxruntime-gpu (or onnxruntime)")
        return False

    providers_available = ort.get_available_providers()
    print(f"Providers available: {providers_available}")

    provider_pref = (provider or "auto").strip().lower()
    if provider_pref == "tensorrt":
        ort_provider = "TensorrtExecutionProvider"
    elif provider_pref == "cuda":
        ort_provider = "CUDAExecutionProvider"
    elif provider_pref == "cpu":
        ort_provider = "CPUExecutionProvider"
    else:
        ort_provider = "CUDAExecutionProvider" if "CUDAExecutionProvider" in providers_available else "CPUExecutionProvider"

    # Always include CPU as a fallback provider.
    provider_chain = [ort_provider]
    if ort_provider != "CPUExecutionProvider":
        provider_chain.append("CPUExecutionProvider")
    provider_chain = [p for p in provider_chain if p in providers_available]
    if not provider_chain:
        provider_chain = ["CPUExecutionProvider"]

    print(f"Provider request: {provider_pref} -> {provider_chain}")

    # 2) load model
    session = ort.InferenceSession(str(model_path), providers=provider_chain)
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]

    obs_dim = int(input_info.shape[1])
    action_dim = int(output_info.shape[1])
    print(f"Model loaded: {model_path}")
    print(f"  Input : {input_info.name}, shape={input_info.shape}")
    print(f"  Output: {output_info.name}, shape={output_info.shape}")
    print(f"Session providers: {session.get_providers()}")

    # 3) load stats (identity if missing or shape mismatch)
    if stats_path is None:
        auto_stats = model_path.parent / f"{model_path.stem}_stats.npz"
        if auto_stats.exists():
            stats_path = auto_stats

    obs_mean = np.zeros(obs_dim, dtype=np.float32)
    obs_var = np.ones(obs_dim, dtype=np.float32)
    if stats_path is not None and stats_path.exists():
        try:
            stats = np.load(stats_path)
            obs_mean = stats["obs_mean"].astype(np.float32)
            obs_var = stats["obs_var"].astype(np.float32)
            if obs_mean.shape != (obs_dim,) or obs_var.shape != (obs_dim,):
                raise ValueError(
                    f"Stats shape mismatch: expected ({obs_dim},), got mean={obs_mean.shape}, var={obs_var.shape}"
                )
            print(f"Stats loaded: {stats_path}")
        except Exception as exc:
            print(f"Stats invalid -> using identity normalization ({exc})")
            obs_mean = np.zeros(obs_dim, dtype=np.float32)
            obs_var = np.ones(obs_dim, dtype=np.float32)
    else:
        print("Stats missing -> using identity normalization (mean=0, var=1)")

    obs_std = np.sqrt(obs_var + 1e-8).astype(np.float32)

    # 4) build test obs
    np.random.seed(0)
    raw_obs = np.random.randn(1, obs_dim).astype(np.float32)
    normalized_obs = (raw_obs - obs_mean) / obs_std
    print(f"Obs stats: raw range[{raw_obs.min():.3f}, {raw_obs.max():.3f}] "
          f"-> norm range[{normalized_obs.min():.3f}, {normalized_obs.max():.3f}]")

    # warmup
    for _ in range(5):
        session.run(None, {input_info.name: normalized_obs})

    # timed runs
    latencies = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        actions = session.run(None, {input_info.name: normalized_obs})[0]
        latencies.append((time.perf_counter() - t0) * 1000.0)
    latencies = np.asarray(latencies)

    actions = actions.astype(np.float32)
    print(f"Action shape: {actions.shape} (expected (1, {action_dim}))")
    print(f"Action range: [{actions.min():.3f}, {actions.max():.3f}]")

    print("Latency (ms): "
          f"mean {latencies.mean():.3f}, p50 {np.percentile(latencies, 50):.3f}, "
          f"p95 {np.percentile(latencies, 95):.3f}, min {latencies.min():.3f}")

    # 5) map actions to physical deltas per training code
    coeff = actions[0]
    robot_pref = (robot or "auto").strip().lower()
    if robot_pref == "auto":
        if action_dim == 5:
            robot_pref = "recomo"
        elif action_dim == 8:
            robot_pref = "mobile_mm"

    dt = float(int(frame_skip)) / 240.0
    print(f"\nExample action mapping (robot={robot_pref}, dt={dt:.3f}s):")
    if robot_pref == "recomo":
        base_speed_limit = 0.5  # m/s (see MobileMMTrajEnv.step)
        yaw_rate_limit = 0.1  # rad/s
        arm_delta_limit = 0.02  # rad/step
        vx = float(coeff[0]) * base_speed_limit
        vy = float(coeff[1]) * base_speed_limit
        wz = float(coeff[2]) * yaw_rate_limit
        dq = coeff[3:] * arm_delta_limit
        print(f"  vx (m/s):    {vx:+.4f}  -> dx (m/step): {vx*dt:+.5f}")
        print(f"  vy (m/s):    {vy:+.4f}  -> dy (m/step): {vy*dt:+.5f}")
        print(f"  wz (rad/s):  {wz:+.4f}  -> dtheta (rad/step): {wz*dt:+.5f}")
        print(f"  dq_arm (rad/step): {dq}")
    else:
        base_acc_limit = 1.5  # m/s^2
        yaw_rate_limit = 0.1  # rad/s
        arm_delta_limit = 0.02  # rad/step
        last_forward_vel = 0.0  # demo assumption
        ds = dt * last_forward_vel + 0.5 * (base_acc_limit * float(coeff[0])) * dt * dt
        dtheta = float(coeff[1]) * yaw_rate_limit * dt
        dq = coeff[2:] * arm_delta_limit
        print(f"  ds (m):        {ds:+.5f}")
        print(f"  dtheta (rad):  {dtheta:+.5f}")
        print(f"  dq_arm (rad):  {dq}")

    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test PyBullet ONNX inference")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("pybulletDeploy/policy_mobile_mm.onnx"),
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=None,
        help="Path to normalization stats (.npz). Default: <model_stem>_stats.npz if present.",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="auto",
        choices=["auto", "mobile_mm", "recomo"],
        help="Robot type for action mapping demo (auto infers from action_dim)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "tensorrt"],
        help="onnxruntime execution provider preference (auto prefers CUDA if available)",
    )
    parser.add_argument(
        "--frame_skip",
        type=int,
        default=24,
        help="frame_skip used during training (dt = frame_skip/240)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of timed inference iterations",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ok = test_onnx(
        model_path=args.model,
        stats_path=args.stats,
        provider=args.provider,
        robot=args.robot,
        num_iterations=args.iterations,
        frame_skip=int(args.frame_skip),
    )
    raise SystemExit(0 if ok else 1)
