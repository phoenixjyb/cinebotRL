#!/usr/bin/env python3
"""Balance and hold the complete upper body with physical gimbal DOFs enabled."""

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
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--maximum-pitch-deg", type=float, default=12.0)
parser.add_argument("--maximum-arm-error-deg", type=float, default=5.0)
parser.add_argument("--maximum-gimbal-error-deg", type=float, default=5.0)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import torch

from rl_platform.robots.two_wheel_balance import TWO_WHEEL_WHOLE_BODY_ATTITUDE_CFG
from rl_platform.tasks.two_wheel_balance import RecomoTwoWheelBalanceEnvCfg
from rl_platform.tasks.two_wheel_balance.metrics import (
    ACTION_NAMES,
    LQR_STATE_NAMES,
    cascaded_lqr_action,
    cascaded_lqr_config,
)
from task_spec import register_isaac_lab_tasks


ARM_JOINTS = (
    "joint6_arm_yaw",
    "joint5_arm_pitch",
    "joint4_elbow_pitch",
)
GIMBAL_JOINTS = (
    "joint3_gimbal_yaw",
    "joint2_gimbal_roll",
    "joint1_gimbal_pitch",
)
VIRTUAL_JOINTS = (
    "ee1_level_pitch",
    "ee1_rot_z",
    "ee1_rot_y",
    "ee1_rot_x",
)
ARM_HOME = np.array([0.0, math.pi / 2.0, 3.0 * math.pi / 4.0])
GIMBAL_HOME = np.zeros(3)


def _single_joint_ids(robot, names: tuple[str, ...]) -> list[int]:
    ids = []
    for name in names:
        matches = robot.find_joints(name)[0]
        if len(matches) != 1:
            raise RuntimeError(f"expected one joint named {name}, got {matches}")
        ids.append(matches[0])
    return ids


def main() -> int:
    gain_data = json.loads(args.gains.resolve().read_text(encoding="utf-8"))
    gain = np.asarray(gain_data["selected_gain"], dtype=np.float64)
    if gain.shape != (len(ACTION_NAMES), len(LQR_STATE_NAMES)):
        raise ValueError(f"invalid gain shape: {gain.shape}")
    control_interval = int(gain_data["control_interval_steps"])

    register_isaac_lab_tasks()
    cfg = RecomoTwoWheelBalanceEnvCfg()
    cfg.seed = 20260715
    cfg.scene.num_envs = 1
    cfg.robot_cfg = TWO_WHEEL_WHOLE_BODY_ATTITUDE_CFG
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
    robot = env.unwrapped.robot
    arm_ids = _single_joint_ids(robot, ARM_JOINTS)
    gimbal_ids = _single_joint_ids(robot, GIMBAL_JOINTS)
    cam_ids = robot.find_bodies("cam_link")[0]
    semantic_ids = robot.find_bodies("ee1_tool")[0]
    if len(cam_ids) != 1 or len(semantic_ids) != 1:
        raise RuntimeError(f"invalid camera bodies: cam={cam_ids}, ee1={semantic_ids}")

    arm_target = torch.as_tensor(
        ARM_HOME[None, :], dtype=torch.float32, device=env.unwrapped.device
    )
    gimbal_target = torch.as_tensor(
        GIMBAL_HOME[None, :], dtype=torch.float32, device=env.unwrapped.device
    )
    controller_state = np.zeros((1, 6), dtype=np.float64)
    current_states = obs["policy"][:, : len(LQR_STATE_NAMES)].detach().cpu().numpy()
    action = np.zeros((1, len(ACTION_NAMES)), dtype=np.float32)
    controller_cfg = cascaded_lqr_config("structural_robust_v1")

    peak_pitch_deg = 0.0
    peak_arm_error_deg = 0.0
    peak_gimbal_error_deg = 0.0
    peak_arm_effort_nm = 0.0
    peak_gimbal_effort_nm = 0.0
    nonfinite_count = 0
    completed_steps = 0
    termination = None
    initial_cam_quat = robot.data.body_quat_w[0, cam_ids[0]].clone()
    initial_gimbal_position_rad = {
        name: float(robot.data.joint_pos[0, joint_id].item())
        for name, joint_id in zip(GIMBAL_JOINTS, gimbal_ids, strict=True)
    }
    startup_trace = []

    for step in range(args.steps):
        robot.set_joint_position_target(arm_target, joint_ids=arm_ids)
        robot.set_joint_position_target(gimbal_target, joint_ids=gimbal_ids)
        if step % control_interval == 0:
            action, controller_state, _ = cascaded_lqr_action(
                current_states,
                np.zeros(1),
                np.zeros(1),
                gain,
                controller_state,
                control_dt=control_interval / 200.0,
                config=controller_cfg,
            )
            action = action.astype(np.float32)
        obs, _, terminated, truncated, _ = env.step(
            torch.as_tensor(action, device=env.unwrapped.device)
        )
        current_states = obs["policy"][:, : len(LQR_STATE_NAMES)].detach().cpu().numpy()
        state = env.unwrapped._state_terms()
        arm_error = robot.data.joint_pos[:, arm_ids] - arm_target
        gimbal_error = robot.data.joint_pos[:, gimbal_ids] - gimbal_target
        finite_values = torch.cat(
            (
                obs["policy"],
                robot.data.joint_pos[:, arm_ids + gimbal_ids],
                robot.data.joint_vel[:, arm_ids + gimbal_ids],
                robot.data.body_quat_w[:, cam_ids[0], :],
            ),
            dim=1,
        )
        nonfinite_count += int(torch.count_nonzero(~torch.isfinite(finite_values)).item())
        peak_pitch_deg = max(peak_pitch_deg, math.degrees(state["pitch"].abs().max().item()))
        peak_arm_error_deg = max(
            peak_arm_error_deg, math.degrees(arm_error.abs().max().item())
        )
        peak_gimbal_error_deg = max(
            peak_gimbal_error_deg, math.degrees(gimbal_error.abs().max().item())
        )
        if hasattr(robot.data, "applied_torque"):
            peak_arm_effort_nm = max(
                peak_arm_effort_nm, robot.data.applied_torque[:, arm_ids].abs().max().item()
            )
            peak_gimbal_effort_nm = max(
                peak_gimbal_effort_nm,
                robot.data.applied_torque[:, gimbal_ids].abs().max().item(),
            )
        if step in (0, 1, 2, 4, 9, 19, 49, 99, 199):
            startup_trace.append(
                {
                    "step": step + 1,
                    "gimbal_position_rad": {
                        name: float(robot.data.joint_pos[0, joint_id].item())
                        for name, joint_id in zip(GIMBAL_JOINTS, gimbal_ids, strict=True)
                    },
                    "gimbal_velocity_rad_s": {
                        name: float(robot.data.joint_vel[0, joint_id].item())
                        for name, joint_id in zip(GIMBAL_JOINTS, gimbal_ids, strict=True)
                    },
                    "gimbal_applied_effort_nm": {
                        name: float(robot.data.applied_torque[0, joint_id].item())
                        for name, joint_id in zip(GIMBAL_JOINTS, gimbal_ids, strict=True)
                    },
                }
            )
        completed_steps = step + 1
        done = terminated | truncated
        if torch.any(done):
            termination = {
                "step": step + 1,
                "terminated": bool(torch.any(terminated).item()),
                "truncated": bool(torch.any(truncated).item()),
                "reset_reason_counts": dict(env.unwrapped.reset_reason_counts),
            }
            break

    final_cam_quat = robot.data.body_quat_w[0, cam_ids[0]]
    cam_dot = torch.abs(torch.dot(initial_cam_quat, final_cam_quat)).clamp(max=1.0)
    cam_attitude_drift_deg = math.degrees(2.0 * math.acos(float(cam_dot.item())))
    joint_names = list(robot.joint_names)
    body_names = list(robot.body_names)
    total_mass_kg = float(robot.data.default_mass[0].sum().item())
    final_gimbal_position_rad = {
        name: float(robot.data.joint_pos[0, joint_id].item())
        for name, joint_id in zip(GIMBAL_JOINTS, gimbal_ids, strict=True)
    }
    env.close()

    checks = {
        "completed_horizon": completed_steps == args.steps,
        "no_termination": termination is None,
        "no_nonfinite": nonfinite_count == 0,
        "total_mass_28kg": abs(total_mass_kg - 28.0) < 0.01,
        "physical_camera_and_semantic_tool_present": {"cam_link", "ee1_tool"} <= set(body_names),
        "physical_gimbal_dofs_present": set(GIMBAL_JOINTS) <= set(joint_names),
        "virtual_frame_dofs_absent": not set(VIRTUAL_JOINTS) & set(joint_names),
        "policy_action_dimension_unchanged": len(ACTION_NAMES) == 2,
        "peak_pitch_below_limit": peak_pitch_deg < args.maximum_pitch_deg,
        "peak_arm_error_below_limit": peak_arm_error_deg < args.maximum_arm_error_deg,
        "peak_gimbal_error_below_limit": peak_gimbal_error_deg < args.maximum_gimbal_error_deg,
    }
    result = {
        "schema": "two_wheel_complete_upper_body_attitude_home_smoke_v1",
        "semantic_contract": {
            "position_target_link": "ee1_tool",
            "attitude_target": "semantic_DFR_world_quaternion_or_optical_axis",
            "physical_camera_observation_link": "cam_link",
            "frame_conversion": "R_world_cam = R_world_DFR * Rz(+pi/2)",
            "physical_gimbal_joint_mode": "internal_sim_attitude_adapter_only",
            "learned_physical_gimbal_joint_action": False,
        },
        "steps": completed_steps,
        "total_mass_kg": total_mass_kg,
        "joint_names": joint_names,
        "peak_pitch_deg": peak_pitch_deg,
        "peak_arm_error_deg": peak_arm_error_deg,
        "peak_gimbal_error_deg": peak_gimbal_error_deg,
        "peak_arm_effort_nm": peak_arm_effort_nm,
        "peak_gimbal_effort_nm": peak_gimbal_effort_nm,
        "initial_gimbal_position_rad": initial_gimbal_position_rad,
        "final_gimbal_position_rad": final_gimbal_position_rad,
        "startup_trace": startup_trace,
        "cam_attitude_drift_deg": cam_attitude_drift_deg,
        "nonfinite_count": nonfinite_count,
        "termination": termination,
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
