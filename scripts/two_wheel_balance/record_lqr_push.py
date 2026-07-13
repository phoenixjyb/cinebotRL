#!/usr/bin/env python3
"""Record rendered bidirectional push recovery with the frozen nominal LQR."""

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
parser.add_argument("--gains", type=Path, required=True)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--fps", type=int, default=50)
parser.add_argument("--push-force-n", type=float, default=60.0)
parser.add_argument("--push-height-m", type=float, default=0.5)
parser.add_argument("--push-duration-steps", type=int, default=20)
parser.add_argument("--first-push-step", type=int, default=120)
parser.add_argument("--second-push-step", type=int, default=360)
parser.add_argument("--seed", type=int, default=20260713)
parser.add_argument("--output-dir", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

from rl_platform.tasks.two_wheel_balance import RecomoTwoWheelBalanceEnvCfg
from rl_platform.tasks.two_wheel_balance.metrics import LQR_STATE_NAMES, lqr_action
from task_spec import register_isaac_lab_tasks


POLICY_HZ = 200.0


def set_push_wrench(env, force_x_n: float) -> None:
    force = torch.zeros((1, 1, 3), device=env.device)
    torque = torch.zeros_like(force)
    force[:, 0, 0] = force_x_n
    torque[:, 0, 1] = force_x_n * args.push_height_m
    env.robot.set_external_force_and_torque(
        forces=force,
        torques=torque,
        body_ids=env._base_body_idx,
        is_global=True,
    )


def scheduled_force(step: int) -> float:
    if args.first_push_step <= step < args.first_push_step + args.push_duration_steps:
        return args.push_force_n
    if args.second_push_step <= step < args.second_push_step + args.push_duration_steps:
        return -args.push_force_n
    return 0.0


def main() -> int:
    if args.steps < 1 or args.fps < 1 or args.push_duration_steps < 1:
        raise ValueError("--steps, --fps, and --push-duration-steps must be positive")
    if args.push_height_m < 0.0:
        raise ValueError("--push-height-m must be non-negative")
    if not 0 <= args.first_push_step < args.second_push_step < args.steps:
        raise ValueError("push steps must be ordered inside the rollout")
    if args.second_push_step + args.push_duration_steps >= args.steps:
        raise ValueError("the second push must end before the rollout")

    gain_data = json.loads(args.gains.resolve().read_text(encoding="utf-8"))
    gain = np.asarray(gain_data["selected_gain"], dtype=np.float64)
    expected_shape = (2, len(LQR_STATE_NAMES))
    if gain.shape != expected_shape:
        raise ValueError(f"expected gain shape {expected_shape}, got {gain.shape}")
    control_interval = int(gain_data["control_interval_steps"])
    action_limit = float(gain_data["action_limit"])

    register_isaac_lab_tasks()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = RecomoTwoWheelBalanceEnvCfg()
    cfg.seed = args.seed
    cfg.scene.num_envs = 1
    cfg.reset_pitch_rad = 0.0
    cfg.control_mode = "direct"
    cfg.viewer.eye = (3.2, -4.0, 1.65)
    cfg.viewer.lookat = (0.0, 0.0, 0.5)
    raw_env = gym.make(
        "RecomoTwoWheelBalance-v0",
        cfg=cfg,
        render_mode="rgb_array",
        disable_env_checker=True,
    )
    raw_env.unwrapped.sim.set_camera_view(eye=cfg.viewer.eye, target=cfg.viewer.lookat)
    env = gym.wrappers.RecordVideo(
        raw_env,
        video_folder=str(args.output_dir),
        step_trigger=lambda step: step == 0,
        video_length=args.steps,
        fps=args.fps,
        name_prefix="two-wheel-lqr-bidirectional-push",
        disable_logger=True,
    )

    obs, _ = env.reset(seed=args.seed)
    set_push_wrench(raw_env.unwrapped, 0.0)
    policy_obs = obs["policy"]
    action_np = np.zeros((1, 2), dtype=np.float32)
    peak_pitch_deg = 0.0
    peak_pitch_rate = 0.0
    peak_wheel_speed = 0.0
    terminated = False
    truncated = False
    previous_force = 0.0
    for step in range(args.steps):
        force = scheduled_force(step)
        if force != previous_force:
            set_push_wrench(raw_env.unwrapped, force)
            previous_force = force
        if step % control_interval == 0:
            states = policy_obs[:, : len(LQR_STATE_NAMES)].detach().cpu().numpy()
            action_np = lqr_action(states, gain, action_limit=action_limit).astype(np.float32)
        actions = torch.as_tensor(action_np, device=raw_env.unwrapped.device)
        obs, _, terminated_tensor, truncated_tensor, _ = env.step(actions)
        policy_obs = obs["policy"]
        state = raw_env.unwrapped._state_terms()
        peak_pitch_deg = max(peak_pitch_deg, abs(float(np.degrees(policy_obs[0, 0].item()))))
        peak_pitch_rate = max(peak_pitch_rate, abs(float(policy_obs[0, 1].item())))
        peak_wheel_speed = max(
            peak_wheel_speed, float(state["max_abs_wheel_velocity"][0].item())
        )
        terminated = bool(terminated_tensor[0])
        truncated = bool(truncated_tensor[0])
        if terminated or truncated:
            break

    set_push_wrench(raw_env.unwrapped, 0.0)
    env.close()
    videos = sorted(args.output_dir.glob("*.mp4"), key=lambda path: path.stat().st_mtime)
    if not videos:
        raise RuntimeError(f"RecordVideo did not produce an mp4 in {args.output_dir}")
    playback_scale = POLICY_HZ / args.fps
    result = {
        "video": str(videos[-1].resolve()),
        "steps_requested": args.steps,
        "steps_recorded": step + 1,
        "fps": args.fps,
        "playback_slowdown": playback_scale,
        "push_force_n": args.push_force_n,
        "push_height_m": args.push_height_m,
        "push_duration_s_sim": args.push_duration_steps / POLICY_HZ,
        "push_events_video_s": [
            args.first_push_step / args.fps,
            args.second_push_step / args.fps,
        ],
        "peak_pitch_deg": peak_pitch_deg,
        "peak_pitch_rate_rad_s": peak_pitch_rate,
        "peak_wheel_speed_rad_s": peak_wheel_speed,
        "terminated": terminated,
        "truncated": truncated,
        "selected_gain_scale": gain_data["selected_gain_scale"],
        "control_interval_steps": control_interval,
        "training_started": False,
    }
    (args.output_dir / "recording.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0 if not terminated else 2


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
