#!/usr/bin/env python3
"""Compare failed PPO behavior with passive, random, and PD controllers."""

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
parser.add_argument("--steps-per-mode", type=int, default=1000)
parser.add_argument("--seed", type=int, default=20260711)
parser.add_argument("--reset-pitch-deg", type=float, default=2.0)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
from stable_baselines3 import PPO

from rl_platform.tasks.two_wheel_balance import RecomoTwoWheelBalanceEnvCfg
from rl_platform.tasks.two_wheel_balance.metrics import OBSERVATION_NAMES
from task_spec import register_isaac_lab_tasks


def safe_correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def summarize_mode(
    env,
    model: PPO,
    mode: str,
    steps: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    obs, _ = env.reset(seed=args.seed)
    policy_obs = obs["policy"]
    observations: list[np.ndarray] = []
    actions_seen: list[np.ndarray] = []
    pd_actions_seen: list[np.ndarray] = []
    reward_terms: dict[str, list[float]] = {}
    episode_lengths: list[int] = []
    current_length = 0
    terminations = 0
    timeouts = 0

    for _ in range(steps):
        obs_np = policy_obs.detach().cpu().numpy()
        pd_common = np.clip(
            env.unwrapped.cfg.pd_common_kp * obs_np[:, 0]
            + env.unwrapped.cfg.pd_common_kd * obs_np[:, 1],
            -env.unwrapped.cfg.pd_common_action_limit,
            env.unwrapped.cfg.pd_common_action_limit,
        )
        if mode == "zero":
            action_np = np.zeros((1, 2), dtype=np.float32)
        elif mode == "random":
            action_np = rng.uniform(-0.15, 0.15, size=(1, 2)).astype(np.float32)
        elif mode == "pd":
            action_np = np.zeros((1, 2), dtype=np.float32)
            action_np[:, 0] = pd_common
        else:
            action_np, _ = model.predict(obs_np, deterministic=True)
            action_np = np.asarray(action_np, dtype=np.float32)

        observations.append(obs_np[0].copy())
        actions_seen.append(action_np[0].copy())
        pd_actions_seen.append(np.array([pd_common[0], 0.0], dtype=np.float32))
        action_tensor = torch.as_tensor(action_np, device=env.unwrapped.device)
        obs, _, terminated, truncated, _ = env.step(action_tensor)
        policy_obs = obs["policy"]
        for key, value in env.unwrapped.last_reward_terms.items():
            reward_terms.setdefault(key, []).append(float(value[0].item()))
        current_length += 1
        if bool(terminated[0]) or bool(truncated[0]):
            episode_lengths.append(current_length)
            current_length = 0
            terminations += int(bool(terminated[0]))
            timeouts += int(bool(truncated[0]))

    obs_values = np.asarray(observations)
    action_values = np.asarray(actions_seen)
    pd_values = np.asarray(pd_actions_seen)
    valid_pd = np.abs(pd_values[:, 0]) > 1e-3
    sign_match = (
        float(np.mean(np.sign(action_values[valid_pd, 0]) == np.sign(pd_values[valid_pd, 0])))
        if np.any(valid_pd)
        else None
    )
    return {
        "steps": steps,
        "terminations": terminations,
        "timeouts": timeouts,
        "episode_length_mean": float(np.mean(episode_lengths)) if episode_lengths else None,
        "episode_lengths": episode_lengths,
        "action_common_mean": float(action_values[:, 0].mean()),
        "action_common_std": float(action_values[:, 0].std()),
        "action_yaw_mean": float(action_values[:, 1].mean()),
        "action_vs_pd_common_rmse": float(
            np.sqrt(np.mean(np.square(action_values[:, 0] - pd_values[:, 0])))
        ),
        "action_vs_pd_sign_match": sign_match,
        "action_common_pitch_correlation": safe_correlation(action_values[:, 0], obs_values[:, 0]),
        "action_common_pitch_rate_correlation": safe_correlation(action_values[:, 0], obs_values[:, 1]),
        "observations": {
            name: {
                "mean": float(obs_values[:, index].mean()),
                "std": float(obs_values[:, index].std()),
                "min": float(obs_values[:, index].min()),
                "max": float(obs_values[:, index].max()),
                "abs_p95": float(np.percentile(np.abs(obs_values[:, index]), 95)),
            }
            for index, name in enumerate(OBSERVATION_NAMES)
        },
        "reward_terms": {
            key: {
                "mean_per_step": float(np.mean(values)),
                "sum": float(np.sum(values)),
            }
            for key, values in reward_terms.items()
        },
    }


def main() -> int:
    register_isaac_lab_tasks()
    cfg = RecomoTwoWheelBalanceEnvCfg()
    cfg.seed = args.seed
    cfg.scene.num_envs = 1
    cfg.reset_pitch_rad = float(np.deg2rad(args.reset_pitch_deg))
    cfg.enable_reward_term_telemetry = True
    env = gym.make(
        "RecomoTwoWheelBalance-v0",
        cfg=cfg,
        render_mode=None,
        disable_env_checker=True,
    )
    model = PPO.load(args.checkpoint, device="cuda")
    rng = np.random.default_rng(args.seed + 1)
    modes = {
        mode: summarize_mode(env, model, mode, args.steps_per_mode, rng)
        for mode in ("zero", "random", "pd", "ppo")
    }

    pitch_grid_deg = np.linspace(-10.0, 10.0, 41)
    canonical_obs = np.zeros((len(pitch_grid_deg), len(OBSERVATION_NAMES)), dtype=np.float32)
    canonical_obs[:, 0] = np.deg2rad(pitch_grid_deg)
    canonical_actions, _ = model.predict(canonical_obs, deterministic=True)
    result = {
        "schema": "recomo_two_wheel_learning_diagnosis_v1",
        "checkpoint": str(args.checkpoint),
        "reset_pitch_deg": args.reset_pitch_deg,
        "modes": modes,
        "canonical_pitch_sweep": {
            "pitch_deg": pitch_grid_deg.tolist(),
            "ppo_common_action": np.asarray(canonical_actions)[:, 0].tolist(),
            "pd_common_action": (
                cfg.pd_common_kp * canonical_obs[:, 0]
                + cfg.pd_common_kd * canonical_obs[:, 1]
            ).tolist(),
        },
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
