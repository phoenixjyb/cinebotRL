#!/usr/bin/env python3
"""Deterministic vectorized smoke for the two-wheel DirectRLEnv."""

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
parser.add_argument("--steps", type=int, default=2048)
parser.add_argument("--seed", type=int, default=20260711)
parser.add_argument("--zero-steps", type=int, default=256)
parser.add_argument(
    "--action-mode",
    choices=("zero_then_random", "zero", "common", "yaw", "pd"),
    default="zero_then_random",
)
parser.add_argument("--action-value", type=float, default=0.1)
parser.add_argument("--pd-kp", type=float, default=1.0)
parser.add_argument("--pd-kd", type=float, default=0.2)
parser.add_argument("--pd-action-limit", type=float, default=0.5)
parser.add_argument("--progress-every", type=int, default=0)
parser.add_argument("--reset-pitch-deg", type=float, default=0.0)
parser.add_argument("--control-mode", choices=("direct", "pd_residual"), default="direct")
parser.add_argument("--policy-residual-scale", type=float, default=0.15)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

from task_spec import register_isaac_lab_tasks
from rl_platform.tasks.two_wheel_balance import RecomoTwoWheelBalanceEnvCfg


def main() -> int:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    register_isaac_lab_tasks()
    cfg = RecomoTwoWheelBalanceEnvCfg()
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.reset_pitch_rad = np.deg2rad(args.reset_pitch_deg)
    cfg.control_mode = args.control_mode
    cfg.policy_residual_scale = args.policy_residual_scale
    env = gym.make("RecomoTwoWheelBalance-v0", cfg=cfg, render_mode=None)
    obs, _ = env.reset(seed=args.seed)
    policy_obs = obs["policy"]
    if policy_obs.shape != (args.num_envs, 10):
        raise AssertionError(f"unexpected observation shape: {policy_obs.shape}")

    generator = torch.Generator(device=env.unwrapped.device)
    generator.manual_seed(args.seed)
    terminated_total = 0
    timeout_total = 0
    reward_sum = 0.0
    episode_lengths = []
    current_lengths = torch.zeros(args.num_envs, dtype=torch.long, device=env.unwrapped.device)
    pitch_samples = []
    pitch_rate_samples = []
    action_rate_samples = []
    previous_actions = torch.zeros((args.num_envs, 2), device=env.unwrapped.device)
    vx_error_squared_sum = 0.0
    wz_error_squared_sum = 0.0
    tracking_sample_count = 0
    saturated_effort_count = 0
    effort_sample_count = 0
    for step in range(args.steps):
        if args.action_mode == "zero" or step < args.zero_steps:
            actions = torch.zeros((args.num_envs, 2), device=env.unwrapped.device)
        elif args.action_mode == "zero_then_random":
            actions = 0.15 * (2.0 * torch.rand((args.num_envs, 2), generator=generator, device=env.unwrapped.device) - 1.0)
        elif args.action_mode in {"common", "yaw"}:
            actions = torch.zeros((args.num_envs, 2), device=env.unwrapped.device)
            actions[:, 0 if args.action_mode == "common" else 1] = args.action_value
        else:
            actions = torch.zeros((args.num_envs, 2), device=env.unwrapped.device)
            actions[:, 0] = torch.clamp(
                args.pd_kp * policy_obs[:, 0] + args.pd_kd * policy_obs[:, 1],
                -args.pd_action_limit,
                args.pd_action_limit,
            )
        obs, reward, terminated, truncated, _ = env.step(actions)
        policy_obs = obs["policy"]
        if not torch.isfinite(policy_obs).all():
            raise RuntimeError(f"non-finite observation at step {step}")
        terminated_total += int(torch.count_nonzero(terminated).item())
        timeout_total += int(torch.count_nonzero(truncated).item())
        reward_sum += float(reward.mean().item())
        current_lengths += 1
        done = terminated | truncated
        if torch.any(done):
            episode_lengths.extend(current_lengths[done].cpu().tolist())
            current_lengths[done] = 0
        pitch_samples.append(policy_obs[:, 0].abs().detach().cpu())
        pitch_rate_samples.append(policy_obs[:, 1].abs().detach().cpu())
        action_rate_samples.append(torch.linalg.norm(actions - previous_actions, dim=1).detach().cpu())
        previous_actions.copy_(actions)
        state = env.unwrapped._state_terms()
        vx_error_squared_sum += torch.square(state["vx"] - env.unwrapped.vx_ref).sum().item()
        wz_error_squared_sum += torch.square(state["yaw_rate"] - env.unwrapped.wz_ref).sum().item()
        tracking_sample_count += args.num_envs
        saturated_effort_count += int(
            torch.count_nonzero(
                env.unwrapped.wheel_efforts.abs() >= env.unwrapped.cfg.torque_limit_nm
            ).item()
        )
        effort_sample_count += args.num_envs * 2
        if args.progress_every and (step + 1) % args.progress_every == 0:
            checkpoint = env.unwrapped.diagnostic_snapshot()
            print(
                f"step={step + 1} terminated={terminated_total} "
                f"pitch_max_deg={checkpoint['abs_pitch_max_deg']:.3f}",
                flush=True,
            )

    snapshot = env.unwrapped.diagnostic_snapshot()
    pitch_values = torch.cat(pitch_samples)
    pitch_rate_values = torch.cat(pitch_rate_samples)
    action_rate_values = torch.cat(action_rate_samples)
    completed_episodes = len(episode_lengths)
    reason_total = sum(snapshot["reset_reason_counts"].values()) - snapshot["reset_reason_counts"]["timeout"]
    result = {
        "schema": "recomo_two_wheel_env_smoke_v1",
        "seed": args.seed,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "zero_steps": args.zero_steps,
        "action_mode": args.action_mode,
        "action_value": args.action_value,
        "control_mode": args.control_mode,
        "policy_residual_scale": args.policy_residual_scale,
        "scripted_pd_kp": args.pd_kp,
        "scripted_pd_kd": args.pd_kd,
        "scripted_pd_action_limit": args.pd_action_limit,
        "control_pd_kp": env.unwrapped.cfg.pd_common_kp,
        "control_pd_kd": env.unwrapped.cfg.pd_common_kd,
        "control_pd_action_limit": env.unwrapped.cfg.pd_common_action_limit,
        "observation_shape": list(policy_obs.shape),
        "action_shape": [args.num_envs, 2],
        "terminated_total": terminated_total,
        "timeout_total": timeout_total,
        "completed_episodes": completed_episodes,
        "fall_rate": terminated_total / max(completed_episodes, 1),
        "episode_length_mean": float(np.mean(episode_lengths)) if episode_lengths else None,
        "episode_length_p50": float(np.percentile(episode_lengths, 50)) if episode_lengths else None,
        "episode_length_p95": float(np.percentile(episode_lengths, 95)) if episode_lengths else None,
        "abs_pitch_mean_deg_all": float(torch.rad2deg(pitch_values.mean()).item()),
        "abs_pitch_p95_deg_all": float(torch.rad2deg(torch.quantile(pitch_values, 0.95)).item()),
        "abs_pitch_max_deg_all": float(torch.rad2deg(pitch_values.max()).item()),
        "abs_pitch_rate_p95_all": float(torch.quantile(pitch_rate_values, 0.95).item()),
        "abs_pitch_rate_max_all": float(pitch_rate_values.max().item()),
        "vx_rmse": float(np.sqrt(vx_error_squared_sum / tracking_sample_count)),
        "wz_rmse": float(np.sqrt(wz_error_squared_sum / tracking_sample_count)),
        "effort_saturation_ratio_all": saturated_effort_count / effort_sample_count,
        "action_rate_p95": float(torch.quantile(action_rate_values, 0.95).item()),
        "reset_reasons_accounted": reason_total == terminated_total,
        "reward_mean_per_step": reward_sum / args.steps,
        **snapshot,
    }
    result["passed"] = result["nonfinite_count"] == 0 and result["reset_reasons_accounted"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    env.close()
    return 0 if result["passed"] else 2


exit_code = 1
try:
    exit_code = main()
except Exception:
    import traceback

    traceback.print_exc()
finally:
    # Isaac Sim 5.1 can hang in shutdown after layered-USD/contact-sensor runs.
    # Keep cleanup graceful, but do not leave orphaned GPU processes forever.
    import threading

    shutdown_watchdog = threading.Timer(10.0, lambda: os._exit(exit_code))
    shutdown_watchdog.daemon = True
    shutdown_watchdog.start()
    app.close()
    shutdown_watchdog.cancel()
raise SystemExit(exit_code)
