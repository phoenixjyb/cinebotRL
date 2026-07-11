#!/usr/bin/env python3
"""Deterministically evaluate a two-wheel SB3 checkpoint without training."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("ACCEPT_EULA", "YES")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")
os.environ.setdefault("GYMNASIUM_DISABLE_PLUGIN_ENTRYPOINTS", "1")

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--episodes", type=int, default=128)
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--seed", type=int, default=20260711)
parser.add_argument("--reset-pitch-deg", type=float, default=2.0)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
from stable_baselines3 import PPO

from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from rl_platform.tasks.two_wheel_balance import RecomoTwoWheelBalanceEnvCfg
from task_spec import register_isaac_lab_tasks


def main() -> int:
    register_isaac_lab_tasks()
    cfg = RecomoTwoWheelBalanceEnvCfg()
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.reset_pitch_rad = float(np.deg2rad(args.reset_pitch_deg))
    env = Sb3VecEnvWrapper(
        gym.make("RecomoTwoWheelBalance-v0", cfg=cfg, render_mode=None),
        fast_variant=True,
    )
    model = PPO.load(args.checkpoint, env=env, device="cuda")
    obs = env.reset()
    lengths = np.zeros(args.num_envs, dtype=np.int64)
    completed: list[int] = []
    pitch_samples: list[np.ndarray] = []
    falls = 0
    timeouts = 0
    max_steps = max(10_000, int(np.ceil(args.episodes / args.num_envs)) * 4_000)
    for _ in range(max_steps):
        actions, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = env.step(actions)
        lengths += 1
        pitch_samples.append(np.abs(obs[:, 0]).copy())
        for index in np.flatnonzero(dones):
            completed.append(int(lengths[index]))
            lengths[index] = 0
            if infos[index].get("TimeLimit.truncated", False):
                timeouts += 1
            else:
                falls += 1
            if len(completed) >= args.episodes:
                break
        if len(completed) >= args.episodes:
            break
    if len(completed) < args.episodes:
        raise RuntimeError(f"evaluation produced {len(completed)}/{args.episodes} episodes")

    episode_lengths = np.asarray(completed[: args.episodes], dtype=np.float64)
    pitch = np.concatenate(pitch_samples)
    result = {
        "schema": "recomo_two_wheel_policy_evaluation_v1",
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "episodes": args.episodes,
        "episode_length_mean": float(episode_lengths.mean()),
        "episode_length_p50": float(np.percentile(episode_lengths, 50)),
        "episode_length_p95": float(np.percentile(episode_lengths, 95)),
        "fall_rate": falls / args.episodes,
        "timeout_rate": timeouts / args.episodes,
        "abs_pitch_mean_deg": float(np.rad2deg(pitch.mean())),
        "abs_pitch_p95_deg": float(np.rad2deg(np.percentile(pitch, 95))),
        "abs_pitch_max_deg": float(np.rad2deg(pitch.max())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    env.close()
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
