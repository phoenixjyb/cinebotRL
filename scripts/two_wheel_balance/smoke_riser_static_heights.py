#!/usr/bin/env python3
"""Balance the riser robot at its lower, home, and upper camera heights."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("ACCEPT_EULA", "YES")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")
os.environ.setdefault("GYMNASIUM_DISABLE_PLUGIN_ENTRYPOINTS", "1")

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--gains", type=Path, required=True)
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--maximum-pitch-deg", type=float, default=12.0)
parser.add_argument("--pitch-p95-deg", type=float, default=6.0)
parser.add_argument("--maximum-height-error-m", type=float, default=0.03)
parser.add_argument("--maximum-saturation-ratio", type=float, default=0.20)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import torch

from rl_platform.robots.two_wheel_balance import TWO_WHEEL_RISER_CFG
from rl_platform.tasks.two_wheel_balance import RecomoTwoWheelBalanceEnvCfg
from rl_platform.tasks.two_wheel_balance.metrics import (
    ACTION_NAMES,
    LQR_STATE_NAMES,
    cascaded_lqr_action,
    cascaded_lqr_config,
)
from task_spec import register_isaac_lab_tasks


CAMERA_HEIGHTS_M = np.array([0.6, 0.9, 1.8], dtype=np.float64)
RISER_TARGETS_M = CAMERA_HEIGHTS_M - 0.6
GIMBAL_JOINTS = (
    "joint3_gimbal_yaw",
    "joint2_gimbal_roll",
    "joint1_gimbal_pitch",
)
REMOVED_ARM_JOINTS = {
    "joint6_arm_yaw",
    "joint5_arm_pitch",
    "joint4_elbow_pitch",
}


def _single_joint_id(robot, name: str) -> int:
    ids = robot.find_joints(name)[0]
    if len(ids) != 1:
        raise RuntimeError(f"expected one joint named {name}, got {ids}")
    return ids[0]


def main() -> int:
    gain_data = json.loads(args.gains.resolve().read_text(encoding="utf-8"))
    gain = np.asarray(gain_data["selected_gain"], dtype=np.float64)
    if gain.shape != (len(ACTION_NAMES), len(LQR_STATE_NAMES)):
        raise ValueError(f"invalid gain shape: {gain.shape}")
    control_interval = int(gain_data["control_interval_steps"])

    register_isaac_lab_tasks()
    cfg = RecomoTwoWheelBalanceEnvCfg()
    cfg.seed = 20260715
    cfg.scene.num_envs = len(CAMERA_HEIGHTS_M)
    cfg.robot_cfg = TWO_WHEEL_RISER_CFG
    cfg.episode_length_s = (args.steps + 100) * cfg.decimation * cfg.sim.dt
    cfg.reset_pitch_rad = 0.0
    cfg.control_mode = "direct"
    env = gym.make(
        "RecomoTwoWheelBalance-v0",
        cfg=cfg,
        render_mode=None,
        disable_env_checker=True,
    )
    obs, _ = env.reset(seed=20260715)
    unwrapped = env.unwrapped
    robot = unwrapped.robot
    env_ids = torch.arange(unwrapped.num_envs, device=unwrapped.device, dtype=torch.long)
    riser_id = _single_joint_id(robot, "riser_joint")
    gimbal_ids = [_single_joint_id(robot, name) for name in GIMBAL_JOINTS]
    cam_ids = robot.find_bodies("cam_link")[0]
    semantic_ids = robot.find_bodies("ee1_tool")[0]
    if len(cam_ids) != 1 or len(semantic_ids) != 1:
        raise RuntimeError(f"invalid camera frame inventory: cam={cam_ids}, dfr={semantic_ids}")

    riser_targets = torch.as_tensor(
        RISER_TARGETS_M[:, None], dtype=torch.float32, device=unwrapped.device
    )
    gimbal_targets = torch.zeros(
        (unwrapped.num_envs, len(gimbal_ids)),
        dtype=torch.float32,
        device=unwrapped.device,
    )
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = robot.data.default_joint_vel[env_ids].clone()
    joint_pos[:, riser_id] = riser_targets[:, 0]
    joint_pos[:, gimbal_ids] = 0.0
    joint_vel.zero_()
    robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
    robot.set_joint_position_target(riser_targets, joint_ids=[riser_id])
    robot.set_joint_velocity_target(torch.zeros_like(riser_targets), joint_ids=[riser_id])
    robot.set_joint_position_target(gimbal_targets, joint_ids=gimbal_ids)

    controller_state = np.zeros((unwrapped.num_envs, 6), dtype=np.float64)
    current_states = obs["policy"][:, : len(LQR_STATE_NAMES)].detach().cpu().numpy()
    action = np.zeros((unwrapped.num_envs, len(ACTION_NAMES)), dtype=np.float32)
    controller_cfg = cascaded_lqr_config("structural_robust_v1")

    pitch_samples = [[] for _ in CAMERA_HEIGHTS_M]
    height_error_samples = [[] for _ in CAMERA_HEIGHTS_M]
    wheel_saturated = np.zeros(unwrapped.num_envs, dtype=np.int64)
    riser_saturated = np.zeros(unwrapped.num_envs, dtype=np.int64)
    nonfinite_count = 0
    completed_steps = 0
    termination = None

    for step in range(args.steps):
        robot.set_joint_position_target(riser_targets, joint_ids=[riser_id])
        robot.set_joint_velocity_target(
            torch.zeros_like(riser_targets), joint_ids=[riser_id]
        )
        robot.set_joint_position_target(gimbal_targets, joint_ids=gimbal_ids)
        if step % control_interval == 0:
            action, controller_state, _ = cascaded_lqr_action(
                current_states,
                np.zeros(unwrapped.num_envs),
                np.zeros(unwrapped.num_envs),
                gain,
                controller_state,
                control_dt=control_interval / 200.0,
                config=controller_cfg,
            )
            action = action.astype(np.float32)
        obs, _, terminated, truncated, _ = env.step(
            torch.as_tensor(action, device=unwrapped.device)
        )
        current_states = obs["policy"][:, : len(LQR_STATE_NAMES)].detach().cpu().numpy()
        state = unwrapped._state_terms()
        camera_z = robot.data.body_pos_w[:, cam_ids[0], 2]
        expected_z = torch.as_tensor(
            CAMERA_HEIGHTS_M + float(cfg.robot_cfg.init_state.pos[2]),
            dtype=torch.float32,
            device=unwrapped.device,
        )
        height_error = (camera_z - expected_z).abs()
        finite = torch.cat(
            (
                obs["policy"],
                robot.data.joint_pos[:, [riser_id, *gimbal_ids]],
                robot.data.joint_vel[:, [riser_id, *gimbal_ids]],
                camera_z[:, None],
            ),
            dim=1,
        ).isfinite().all(dim=1)
        nonfinite_count += int(torch.count_nonzero(~finite).item())
        pitch_deg = torch.rad2deg(state["pitch"].abs()).detach().cpu().numpy()
        height_error_np = height_error.detach().cpu().numpy()
        wheel_effort = robot.data.applied_torque[:, unwrapped._wheel_joint_idx].abs()
        riser_effort = robot.data.applied_torque[:, riser_id].abs()
        wheel_saturated += (
            wheel_effort.max(dim=1).values >= 0.99 * cfg.torque_limit_nm
        ).detach().cpu().numpy()
        riser_saturated += (riser_effort >= 0.99 * 300.0).detach().cpu().numpy()
        for index in range(unwrapped.num_envs):
            pitch_samples[index].append(float(pitch_deg[index]))
            height_error_samples[index].append(float(height_error_np[index]))
        completed_steps = step + 1
        done = terminated | truncated
        if torch.any(done):
            termination = {
                "step": step + 1,
                "done_env_ids": torch.nonzero(done).flatten().detach().cpu().tolist(),
                "reset_reason_counts": dict(unwrapped.reset_reason_counts),
            }
            break

    rows = []
    for index, height in enumerate(CAMERA_HEIGHTS_M):
        pitches = np.asarray(pitch_samples[index], dtype=np.float64)
        errors = np.asarray(height_error_samples[index], dtype=np.float64)
        rows.append(
            {
                "camera_height_target_m": float(height),
                "riser_target_m": float(RISER_TARGETS_M[index]),
                "pitch_p95_deg": float(np.percentile(pitches, 95)) if pitches.size else math.inf,
                "pitch_max_deg": float(np.max(pitches)) if pitches.size else math.inf,
                "camera_height_error_p95_m": float(np.percentile(errors, 95)) if errors.size else math.inf,
                "camera_height_error_max_m": float(np.max(errors)) if errors.size else math.inf,
                "wheel_saturation_ratio": float(wheel_saturated[index] / max(completed_steps, 1)),
                "riser_saturation_ratio": float(riser_saturated[index] / max(completed_steps, 1)),
            }
        )

    total_mass_kg = float(robot.data.default_mass[0].sum().item())
    joint_names = list(robot.joint_names)
    body_names = list(robot.body_names)
    env.close()
    checks = {
        "completed_horizon": completed_steps == args.steps,
        "no_termination": termination is None,
        "no_nonfinite": nonfinite_count == 0,
        "total_mass_28kg": abs(total_mass_kg - 28.0) < 0.01,
        "riser_and_gimbal_dofs_present": {"riser_joint", *GIMBAL_JOINTS}
        <= set(joint_names),
        "arm_dofs_absent": not REMOVED_ARM_JOINTS & set(joint_names),
        "camera_frames_present": {"cam_link", "ee1_tool"} <= set(body_names),
        "all_pitch_max_below_limit": all(
            row["pitch_max_deg"] <= args.maximum_pitch_deg for row in rows
        ),
        "all_pitch_p95_below_limit": all(
            row["pitch_p95_deg"] <= args.pitch_p95_deg for row in rows
        ),
        "all_height_errors_below_limit": all(
            row["camera_height_error_p95_m"] <= args.maximum_height_error_m
            for row in rows
        ),
        "all_saturation_ratios_below_limit": all(
            row["wheel_saturation_ratio"] <= args.maximum_saturation_ratio
            and row["riser_saturation_ratio"] <= args.maximum_saturation_ratio
            for row in rows
        ),
    }
    result = {
        "schema": "recomo_two_wheel_riser_static_height_gate_v1",
        "controller_profile": "structural_robust_v1",
        "requested_steps": args.steps,
        "completed_steps": completed_steps,
        "total_mass_kg": total_mass_kg,
        "rows": rows,
        "termination": termination,
        "nonfinite_count": nonfinite_count,
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


exit_code = 1
try:
    exit_code = main()
except Exception:
    import traceback

    traceback.print_exc()
finally:
    app.close()
raise SystemExit(exit_code)
