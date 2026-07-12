#!/usr/bin/env python3
"""Bounded PPO learning-signal gate for stand-only two-wheel balance."""

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
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--total-timesteps", type=int, default=65_536)
parser.add_argument("--eval-episodes", type=int, default=128)
parser.add_argument("--seed", type=int, default=20260711)
parser.add_argument("--reset-pitch-deg", type=float, default=2.0)
parser.add_argument("--control-mode", choices=("direct", "pd_residual"), default="direct")
parser.add_argument("--policy-residual-scale", type=float, default=0.15)
parser.add_argument("--reference-direct-episode-length", type=float, default=125.1640625)
parser.add_argument("--reference-direct-pitch-p95-deg", type=float, default=13.874094446891013)
parser.add_argument("--output-dir", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from rl_platform.tasks.two_wheel_balance import RecomoTwoWheelBalanceEnvCfg
from task_spec import register_isaac_lab_tasks


class RolloutMetricsCallback(BaseCallback):
    def __init__(self) -> None:
        super().__init__()
        self.history: list[dict[str, float | int]] = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        lengths = [float(item["l"]) for item in self.model.ep_info_buffer]
        rewards = [float(item["r"]) for item in self.model.ep_info_buffer]
        values = self.model.logger.name_to_value
        self.history.append(
            {
                "timesteps": int(self.num_timesteps),
                "episode_length_mean": float(np.mean(lengths)) if lengths else 0.0,
                "episode_reward_mean": float(np.mean(rewards)) if rewards else 0.0,
                "approx_kl": float(values.get("train/approx_kl", 0.0)),
                "clip_fraction": float(values.get("train/clip_fraction", 0.0)),
                "entropy_loss": float(values.get("train/entropy_loss", 0.0)),
                "explained_variance": float(values.get("train/explained_variance", 0.0)),
            }
        )


def make_env() -> Sb3VecEnvWrapper:
    cfg = RecomoTwoWheelBalanceEnvCfg()
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.reset_pitch_rad = float(np.deg2rad(args.reset_pitch_deg))
    cfg.control_mode = args.control_mode
    cfg.policy_residual_scale = args.policy_residual_scale
    env = gym.make("RecomoTwoWheelBalance-v0", cfg=cfg, render_mode=None)
    return Sb3VecEnvWrapper(env, fast_variant=True)


def evaluate_actions(
    env: Sb3VecEnvWrapper,
    *,
    episodes: int,
    model: PPO | None,
    seed: int,
    random_actions: bool = True,
) -> dict[str, float | int]:
    obs = env.reset()
    rng = np.random.default_rng(seed)
    current_lengths = np.zeros(env.num_envs, dtype=np.int64)
    completed_lengths: list[int] = []
    pitch_values: list[np.ndarray] = []
    timeout_count = 0
    terminated_count = 0
    max_steps = max(10_000, int(np.ceil(episodes / env.num_envs)) * 4_000)
    for _ in range(max_steps):
        if model is None and random_actions:
            actions = rng.uniform(-0.15, 0.15, size=(env.num_envs, 2)).astype(np.float32)
        elif model is None:
            actions = np.zeros((env.num_envs, 2), dtype=np.float32)
        else:
            actions, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = env.step(actions)
        current_lengths += 1
        pitch_values.append(np.abs(obs[:, 0]).copy())
        for index in np.flatnonzero(dones):
            completed_lengths.append(int(current_lengths[index]))
            current_lengths[index] = 0
            if infos[index].get("TimeLimit.truncated", False):
                timeout_count += 1
            else:
                terminated_count += 1
            if len(completed_lengths) >= episodes:
                break
        if len(completed_lengths) >= episodes:
            break
    if len(completed_lengths) < episodes:
        raise RuntimeError(f"evaluation produced {len(completed_lengths)}/{episodes} episodes")
    lengths = np.asarray(completed_lengths[:episodes], dtype=np.float64)
    pitch = np.concatenate(pitch_values)
    return {
        "episodes": int(episodes),
        "episode_length_mean": float(lengths.mean()),
        "episode_length_p50": float(np.percentile(lengths, 50)),
        "episode_length_p95": float(np.percentile(lengths, 95)),
        "fall_rate": float(terminated_count / episodes),
        "timeout_rate": float(timeout_count / episodes),
        "abs_pitch_mean_deg": float(np.rad2deg(pitch.mean())),
        "abs_pitch_p95_deg": float(np.rad2deg(np.percentile(pitch, 95))),
        "abs_pitch_max_deg": float(np.rad2deg(pitch.max())),
    }


def main() -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    register_isaac_lab_tasks()
    env = make_env()

    baseline = evaluate_actions(
        env,
        episodes=args.eval_episodes,
        model=None,
        seed=args.seed + 1,
    )
    (args.output_dir / "random_baseline.json").write_text(
        json.dumps(baseline, indent=2), encoding="utf-8"
    )
    print(f"random baseline: {baseline}", flush=True)
    pd_prior = None
    if args.control_mode == "pd_residual":
        pd_prior = evaluate_actions(
            env,
            episodes=args.eval_episodes,
            model=None,
            seed=args.seed + 2,
            random_actions=False,
        )
        (args.output_dir / "pd_prior_baseline.json").write_text(
            json.dumps(pd_prior, indent=2), encoding="utf-8"
        )
        print(f"PD prior baseline: {pd_prior}", flush=True)

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs={
            "net_arch": {"pi": [64, 64], "vf": [64, 64]},
            "activation_fn": torch.nn.ELU,
            "ortho_init": True,
            "log_std_init": -1.5,
        },
        learning_rate=3e-4,
        n_steps=256,
        batch_size=512,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.05,
        seed=args.seed,
        device="cuda",
        verbose=1,
    )
    callback = RolloutMetricsCallback()
    model.learn(total_timesteps=args.total_timesteps, callback=callback, progress_bar=False)
    checkpoint = args.output_dir / "ppo_balance_gate"
    model.save(checkpoint)
    (args.output_dir / "rollout_history.json").write_text(
        json.dumps(callback.history, indent=2), encoding="utf-8"
    )

    evaluation = evaluate_actions(
        env,
        episodes=args.eval_episodes,
        model=model,
        seed=args.seed + 2,
    )
    finite_training = all(
        np.isfinite(float(row[key]))
        for row in callback.history
        for key in ("approx_kl", "entropy_loss", "explained_variance")
    )
    learning_signal = (
        evaluation["episode_length_mean"] > baseline["episode_length_mean"] * 1.10
        and evaluation["fall_rate"] < baseline["fall_rate"]
        and evaluation["abs_pitch_p95_deg"] < baseline["abs_pitch_p95_deg"]
    )
    prior_preserved = (
        pd_prior is None
        or (
            evaluation["episode_length_mean"] >= pd_prior["episode_length_mean"] * 0.90
            and evaluation["abs_pitch_p95_deg"] <= pd_prior["abs_pitch_p95_deg"] * 1.10
        )
    )
    direct_baseline_improved = (
        evaluation["episode_length_mean"] > args.reference_direct_episode_length * 1.50
        and evaluation["abs_pitch_p95_deg"] < args.reference_direct_pitch_p95_deg
    )
    result = {
        "schema": "recomo_two_wheel_ppo_gate_v1",
        "seed": args.seed,
        "num_envs": args.num_envs,
        "total_timesteps": args.total_timesteps,
        "reset_pitch_deg": args.reset_pitch_deg,
        "control_mode": args.control_mode,
        "policy_residual_scale": args.policy_residual_scale,
        "reference_direct_baseline": {
            "episode_length_mean": args.reference_direct_episode_length,
            "abs_pitch_p95_deg": args.reference_direct_pitch_p95_deg,
        },
        "random_baseline": baseline,
        "pd_prior_baseline": pd_prior,
        "deterministic_policy": evaluation,
        "finite_training_metrics": finite_training,
        "learning_signal": learning_signal,
        "prior_preserved": prior_preserved,
        "direct_baseline_improved": direct_baseline_improved,
        "passed": finite_training and learning_signal and prior_preserved and direct_baseline_improved,
        "checkpoint": str(checkpoint.with_suffix(".zip")),
    }
    (args.output_dir / "ppo_gate_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    env.close()
    return 0 if result["passed"] else 2


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
