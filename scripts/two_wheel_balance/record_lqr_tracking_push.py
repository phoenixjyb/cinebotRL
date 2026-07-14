#!/usr/bin/env python3
"""Record rendered chassis tracking through an upper-body push."""

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
parser.add_argument("--vx-ref", type=float, default=0.2)
parser.add_argument("--wz-ref", type=float, default=-0.4)
parser.add_argument("--command-start-step", type=int, default=100)
parser.add_argument("--push-start-step", type=int, default=300)
parser.add_argument("--push-duration-steps", type=int, default=20)
parser.add_argument("--push-force-n", type=float, default=-60.0)
parser.add_argument("--push-height-m", type=float, default=0.5)
parser.add_argument("--com-offset-x-m", type=float, default=0.0)
parser.add_argument("--path-progress-governor", action="store_true")
parser.add_argument("--seed", type=int, default=20260713)
parser.add_argument("--output-dir", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

from rl_platform.tasks.two_wheel_balance import RecomoTwoWheelBalanceEnvCfg
from rl_platform.tasks.two_wheel_balance.metrics import (
    LQR_STATE_NAMES,
    CascadedLQRConfig,
    cascaded_lqr_action,
)
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


def apply_base_com_offset_x(env, offset_x_m: float) -> None:
    if offset_x_m == 0.0:
        return
    env_ids = torch.tensor([0], dtype=torch.long, device="cpu")
    base_body_id = int(env._base_body_idx[0])
    coms = env.robot.root_physx_view.get_coms().clone()
    coms[0, base_body_id, 0] += offset_x_m
    env.robot.root_physx_view.set_coms(coms, env_ids)


def main() -> int:
    push_end_step = args.push_start_step + args.push_duration_steps
    if args.steps < 1 or args.fps < 1 or args.push_duration_steps < 1:
        raise ValueError("--steps, --fps, and --push-duration-steps must be positive")
    if not 0 <= args.command_start_step < args.push_start_step < push_end_step < args.steps:
        raise ValueError("command and push windows must be ordered inside the rollout")
    if args.push_height_m < 0.0:
        raise ValueError("--push-height-m must be non-negative")

    gain_data = json.loads(args.gains.resolve().read_text(encoding="utf-8"))
    gain = np.asarray(gain_data["selected_gain"], dtype=np.float64)
    control_interval = int(gain_data["control_interval_steps"])
    action_limit = float(gain_data["action_limit"])
    config = CascadedLQRConfig(
        action_limit=action_limit,
        path_progress_governor_enabled=args.path_progress_governor,
    )

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
        name_prefix="two-wheel-lqr-tracking-push",
        disable_logger=True,
    )

    obs, _ = env.reset(seed=args.seed)
    unwrapped = raw_env.unwrapped
    apply_base_com_offset_x(unwrapped, args.com_offset_x_m)
    set_push_wrench(unwrapped, 0.0)
    current_states = obs["policy"][:, : len(LQR_STATE_NAMES)].detach().cpu().numpy()
    integrals = np.zeros((1, 6), dtype=np.float64)
    action_np = np.zeros((1, 2), dtype=np.float32)
    admitted_vx_ref = 0.0
    admitted_wz_ref = 0.0
    path_progress_scale_min = 1.0
    path_progress_scale_sum = 0.0
    path_progress_scale_samples = 0
    requested_vx_squared_error = 0.0
    requested_wz_squared_error = 0.0
    admitted_vx_squared_error = 0.0
    admitted_wz_squared_error = 0.0
    tracking_samples = 0
    initial_xy = unwrapped.robot.data.root_pos_w[0, :2].detach().cpu().numpy().copy()
    peak_pitch_deg = 0.0
    peak_roll_deg = 0.0
    peak_wheel_speed = 0.0
    terminated = False
    truncated = False

    for step in range(args.steps):
        command_active = step >= args.command_start_step
        vx_ref = np.array([args.vx_ref if command_active else 0.0])
        wz_ref = np.array([args.wz_ref if command_active else 0.0])
        unwrapped.vx_ref.fill_(float(vx_ref[0]))
        unwrapped.wz_ref.fill_(float(wz_ref[0]))
        if step == args.push_start_step:
            set_push_wrench(unwrapped, args.push_force_n)
        elif step == push_end_step:
            set_push_wrench(unwrapped, 0.0)
        if step % control_interval == 0:
            action_np, integrals, diagnostics = cascaded_lqr_action(
                current_states,
                vx_ref,
                wz_ref,
                gain,
                integrals,
                control_dt=control_interval / POLICY_HZ,
                config=config,
            )
            admitted_vx_ref = float(diagnostics["governed_vx_ref"][0])
            admitted_wz_ref = float(diagnostics["governed_wz_ref"][0])
            if command_active:
                path_progress_scale = float(diagnostics["path_progress_scale"][0])
                path_progress_scale_min = min(
                    path_progress_scale_min, path_progress_scale
                )
                path_progress_scale_sum += path_progress_scale
                path_progress_scale_samples += 1
            action_np = action_np.astype(np.float32)
        obs, _, terminated_tensor, truncated_tensor, _ = env.step(
            torch.as_tensor(action_np, device=unwrapped.device)
        )
        current_states = obs["policy"][:, : len(LQR_STATE_NAMES)].detach().cpu().numpy()
        state = unwrapped._state_terms()
        peak_pitch_deg = max(
            peak_pitch_deg, abs(float(np.degrees(current_states[0, 0])))
        )
        peak_roll_deg = max(
            peak_roll_deg,
            abs(float(np.degrees(state["roll"][0].detach().cpu().item()))),
        )
        peak_wheel_speed = max(
            peak_wheel_speed,
            float(state["max_abs_wheel_velocity"][0].detach().cpu().item()),
        )
        if command_active:
            vx_truth = float(state["vx"][0].detach().cpu().item())
            wz_truth = float(state["yaw_rate"][0].detach().cpu().item())
            requested_vx_squared_error += (vx_truth - args.vx_ref) ** 2
            requested_wz_squared_error += (wz_truth - args.wz_ref) ** 2
            admitted_vx_squared_error += (vx_truth - admitted_vx_ref) ** 2
            admitted_wz_squared_error += (wz_truth - admitted_wz_ref) ** 2
            tracking_samples += 1
        terminated = bool(terminated_tensor[0])
        truncated = bool(truncated_tensor[0])
        if terminated or truncated:
            break

    set_push_wrench(unwrapped, 0.0)
    final_xy = unwrapped.robot.data.root_pos_w[0, :2].detach().cpu().numpy().copy()
    env.close()
    videos = sorted(args.output_dir.glob("*.mp4"), key=lambda path: path.stat().st_mtime)
    if not videos:
        raise RuntimeError(f"RecordVideo did not produce an mp4 in {args.output_dir}")
    result = {
        "video": str(videos[-1].resolve()),
        "seed": args.seed,
        "steps_requested": args.steps,
        "steps_recorded": step + 1,
        "fps": args.fps,
        "playback_slowdown": POLICY_HZ / args.fps,
        "vx_ref_m_s": args.vx_ref,
        "wz_ref_rad_s": args.wz_ref,
        "com_offset_x_m": args.com_offset_x_m,
        "command_start_video_s": args.command_start_step / args.fps,
        "push_force_n": args.push_force_n,
        "push_impulse_ns": args.push_force_n * args.push_duration_steps / POLICY_HZ,
        "push_height_m": args.push_height_m,
        "push_start_video_s": args.push_start_step / args.fps,
        "push_end_video_s": push_end_step / args.fps,
        "peak_pitch_deg": peak_pitch_deg,
        "peak_roll_deg": peak_roll_deg,
        "peak_wheel_speed_rad_s": peak_wheel_speed,
        "planar_displacement_m": float(np.linalg.norm(final_xy - initial_xy)),
        "requested_vx_rmse": float(
            np.sqrt(requested_vx_squared_error / max(tracking_samples, 1))
        ),
        "requested_wz_rmse": float(
            np.sqrt(requested_wz_squared_error / max(tracking_samples, 1))
        ),
        "admitted_vx_rmse": float(
            np.sqrt(admitted_vx_squared_error / max(tracking_samples, 1))
        ),
        "admitted_wz_rmse": float(
            np.sqrt(admitted_wz_squared_error / max(tracking_samples, 1))
        ),
        "path_progress_scale_min": path_progress_scale_min,
        "path_progress_scale_mean": (
            path_progress_scale_sum / max(path_progress_scale_samples, 1)
        ),
        "terminated": terminated,
        "truncated": truncated,
        "selected_gain_scale": gain_data["selected_gain_scale"],
        "controller": {
            "vx_kp": config.vx_kp,
            "vx_ki": config.vx_ki,
            "wz_kp": config.wz_kp,
            "wz_ki": config.wz_ki,
            "wz_feedforward": config.wz_feedforward,
            "pitch_bias_adaptation_rate": config.pitch_bias_adaptation_rate,
            "pitch_bias_limit_deg": float(np.degrees(config.pitch_bias_limit_rad)),
            "path_progress_governor_enabled": config.path_progress_governor_enabled,
            "governor_bias_start_deg": float(
                np.degrees(config.governor_bias_start_rad)
            ),
            "governor_bias_full_deg": float(
                np.degrees(config.governor_bias_full_rad)
            ),
            "governor_minimum_progress_scale": (
                config.governor_minimum_progress_scale
            ),
        },
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
