#!/usr/bin/env python3
"""Record an Isaac-rendered PD balance sanity rollout."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("ACCEPT_EULA", "YES")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")
os.environ.setdefault("GYMNASIUM_DISABLE_PLUGIN_ENTRYPOINTS", "1")

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--fps", type=int, default=50)
parser.add_argument("--reset-pitch-deg", type=float, default=2.0)
parser.add_argument("--pd-kp", type=float, default=-1.0)
parser.add_argument("--pd-kd", type=float, default=-0.2)
parser.add_argument("--output-dir", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch

from rl_platform.tasks.two_wheel_balance import RecomoTwoWheelBalanceEnvCfg
from task_spec import register_isaac_lab_tasks


def main() -> int:
    register_isaac_lab_tasks()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = RecomoTwoWheelBalanceEnvCfg()
    cfg.scene.num_envs = 1
    cfg.reset_pitch_rad = float(np.deg2rad(args.reset_pitch_deg))
    cfg.viewer.eye = (2.1, -2.6, 1.25)
    cfg.viewer.lookat = (0.0, 0.0, 0.45)
    raw_env = gym.make("RecomoTwoWheelBalance-v0", cfg=cfg, render_mode="rgb_array")
    raw_env.unwrapped.sim.set_camera_view(eye=cfg.viewer.eye, target=cfg.viewer.lookat)
    env = gym.wrappers.RecordVideo(
        raw_env,
        video_folder=str(args.output_dir),
        step_trigger=lambda step: step == 0,
        video_length=args.steps,
        fps=args.fps,
        name_prefix="two-wheel-pd-sanity",
        disable_logger=True,
    )
    obs, _ = env.reset()
    policy_obs = obs["policy"]
    for _ in range(args.steps):
        actions = torch.zeros((1, 2), device=raw_env.unwrapped.device)
        actions[:, 0] = torch.clamp(
            args.pd_kp * policy_obs[:, 0] + args.pd_kd * policy_obs[:, 1],
            -0.5,
            0.5,
        )
        obs, _, _, _, _ = env.step(actions)
        policy_obs = obs["policy"]
    env.close()
    videos = sorted(args.output_dir.glob("*.mp4"))
    if not videos:
        raise RuntimeError(f"RecordVideo did not produce an mp4 in {args.output_dir}")
    print(f"video={videos[-1]}", flush=True)
    return 0


exit_code = 1
try:
    exit_code = main()
except Exception:
    import traceback

    traceback.print_exc()
finally:
    import threading

    watchdog = threading.Timer(10.0, lambda: os._exit(exit_code))
    watchdog.daemon = True
    watchdog.start()
    app.close()
    watchdog.cancel()
raise SystemExit(exit_code)
