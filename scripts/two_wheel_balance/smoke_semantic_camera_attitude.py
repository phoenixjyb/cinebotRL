#!/usr/bin/env python3
"""Track a small semantic DFR attitude motion with the physical camera chain."""

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
parser.add_argument("--urdf", type=Path, required=True)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--amplitude-deg", type=float, default=5.0)
parser.add_argument("--frequency-hz", type=float, default=0.25)
parser.add_argument("--maximum-pitch-deg", type=float, default=12.0)
parser.add_argument("--maximum-attitude-p95-deg", type=float, default=5.0)
parser.add_argument("--maximum-attitude-error-deg", type=float, default=10.0)
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
from rl_platform.tasks.two_wheel_balance.camera_attitude import (
    ARM_JOINTS,
    PHYSICAL_GIMBAL_JOINTS,
    UrdfPhysicalCameraKinematics,
    matrix_quaternion_wxyz,
    physical_cam_to_semantic_dfr_quat_wxyz,
    quaternion_matrix_wxyz,
    rotation_error_vector,
    semantic_dfr_to_physical_cam_quat_wxyz,
)
from rl_platform.tasks.two_wheel_balance.metrics import (
    ACTION_NAMES,
    LQR_STATE_NAMES,
    cascaded_lqr_action,
    cascaded_lqr_config,
)
from task_spec import register_isaac_lab_tasks


POLICY_HZ = 200.0
ARM_HOME = np.array([0.0, math.pi / 2.0, 3.0 * math.pi / 4.0])


def _ids(robot, names: tuple[str, ...]) -> list[int]:
    result = []
    for name in names:
        found = robot.find_joints(name)[0]
        if len(found) != 1:
            raise RuntimeError(f"expected one joint named {name}, got {found}")
        result.append(found[0])
    return result


def main() -> int:
    gain_data = json.loads(args.gains.resolve().read_text(encoding="utf-8"))
    gain = np.asarray(gain_data["selected_gain"], dtype=np.float64)
    control_interval = int(gain_data["control_interval_steps"])
    kinematics = UrdfPhysicalCameraKinematics(args.urdf.resolve())

    register_isaac_lab_tasks()
    cfg = RecomoTwoWheelBalanceEnvCfg()
    cfg.seed = 20260715
    cfg.scene.num_envs = 1
    cfg.robot_cfg = TWO_WHEEL_WHOLE_BODY_ATTITUDE_CFG
    cfg.episode_length_s = (args.steps + 100) * cfg.decimation * cfg.sim.dt
    cfg.reset_pitch_rad = 0.0
    cfg.control_mode = "direct"
    env = gym.make(
        "RecomoTwoWheelBalance-v0", cfg=cfg, render_mode=None, disable_env_checker=True
    )
    obs, _ = env.reset(seed=20260715)
    unwrapped = env.unwrapped
    robot = unwrapped.robot
    arm_ids = _ids(robot, ARM_JOINTS)
    gimbal_ids = _ids(robot, PHYSICAL_GIMBAL_JOINTS)
    cam_ids = robot.find_bodies("cam_link")[0]
    if len(cam_ids) != 1:
        raise RuntimeError(f"expected physical cam_link, got {cam_ids}")

    arm_target = torch.as_tensor(
        ARM_HOME[None, :], dtype=torch.float32, device=unwrapped.device
    )
    controller_state = np.zeros((1, 6), dtype=np.float64)
    current_states = obs["policy"][:, : len(LQR_STATE_NAMES)].detach().cpu().numpy()
    action = np.zeros((1, len(ACTION_NAMES)), dtype=np.float32)
    controller_cfg = cascaded_lqr_config("structural_robust_v1")
    previous_gimbal_target = robot.data.joint_pos[0, gimbal_ids].detach().cpu().numpy().copy()
    home_physical_rotation = quaternion_matrix_wxyz(
        robot.data.body_quat_w[0, cam_ids[0]].detach().cpu().numpy()
    )

    attitude_errors_deg = []
    ik_errors_deg = []
    peak_pitch_deg = 0.0
    peak_gimbal_effort_nm = 0.0
    ik_nonconverged_steps = 0
    gimbal_target_saturation_steps = 0
    nonfinite_count = 0
    termination = None
    completed_steps = 0
    target_amplitude_observed_deg = 0.0
    settle_steps = int(0.5 * POLICY_HZ)

    for step in range(args.steps):
        elapsed_s = step / POLICY_HZ
        phase_s = max(0.0, elapsed_s - 0.5)
        target_angle = math.radians(args.amplitude_deg) * math.sin(
            2.0 * math.pi * args.frequency_hz * phase_s
        )
        target_amplitude_observed_deg = max(
            target_amplitude_observed_deg, abs(math.degrees(target_angle))
        )
        local_motion = np.array(
            [
                [math.cos(target_angle), -math.sin(target_angle), 0.0],
                [math.sin(target_angle), math.cos(target_angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        desired_physical_rotation = home_physical_rotation @ local_motion
        semantic_dfr_target = physical_cam_to_semantic_dfr_quat_wxyz(
            matrix_quaternion_wxyz(desired_physical_rotation)
        )

        root_quat = robot.data.root_quat_w[0].detach().cpu().numpy()
        actual_arm = robot.data.joint_pos[0, arm_ids].detach().cpu().numpy()
        ik = kinematics.solve_semantic_attitude(
            root_quat,
            actual_arm,
            semantic_dfr_target,
            previous_gimbal_target,
        )
        ik_errors_deg.append(math.degrees(ik.orientation_error_rad))
        ik_nonconverged_steps += int(not ik.converged)
        maximum_delta = 0.5 / POLICY_HZ
        requested_delta = ik.gimbal_q - previous_gimbal_target
        limited_delta = np.clip(requested_delta, -maximum_delta, maximum_delta)
        gimbal_target_saturation_steps += int(
            np.any(np.abs(requested_delta) > maximum_delta + 1e-12)
        )
        previous_gimbal_target = previous_gimbal_target + limited_delta

        robot.set_joint_position_target(arm_target, joint_ids=arm_ids)
        robot.set_joint_position_target(
            torch.as_tensor(
                previous_gimbal_target[None, :],
                dtype=torch.float32,
                device=unwrapped.device,
            ),
            joint_ids=gimbal_ids,
        )
        if step % control_interval == 0:
            action, controller_state, _ = cascaded_lqr_action(
                current_states,
                np.zeros(1),
                np.zeros(1),
                gain,
                controller_state,
                control_dt=control_interval / POLICY_HZ,
                config=controller_cfg,
            )
            action = action.astype(np.float32)
        obs, _, terminated, truncated, _ = env.step(
            torch.as_tensor(action, device=unwrapped.device)
        )
        current_states = obs["policy"][:, : len(LQR_STATE_NAMES)].detach().cpu().numpy()
        actual_physical_quat = robot.data.body_quat_w[0, cam_ids[0]].detach().cpu().numpy()
        expected_physical_quat = semantic_dfr_to_physical_cam_quat_wxyz(
            semantic_dfr_target
        )
        attitude_error_deg = math.degrees(
            np.linalg.norm(
                rotation_error_vector(
                    quaternion_matrix_wxyz(actual_physical_quat),
                    quaternion_matrix_wxyz(expected_physical_quat),
                )
            )
        )
        if step >= settle_steps:
            attitude_errors_deg.append(attitude_error_deg)
        state = unwrapped._state_terms()
        peak_pitch_deg = max(
            peak_pitch_deg, math.degrees(state["pitch"].abs().max().item())
        )
        if hasattr(robot.data, "applied_torque"):
            peak_gimbal_effort_nm = max(
                peak_gimbal_effort_nm,
                robot.data.applied_torque[:, gimbal_ids].abs().max().item(),
            )
        finite = torch.isfinite(
            torch.cat(
                (
                    obs["policy"],
                    robot.data.joint_pos[:, arm_ids + gimbal_ids],
                    robot.data.body_quat_w[:, cam_ids[0], :],
                ),
                dim=1,
            )
        ).all()
        nonfinite_count += int(not bool(finite.item()))
        completed_steps = step + 1
        if torch.any(terminated | truncated):
            termination = {
                "step": step + 1,
                "terminated": bool(torch.any(terminated).item()),
                "truncated": bool(torch.any(truncated).item()),
                "reset_reason_counts": dict(unwrapped.reset_reason_counts),
            }
            break

    error_array = np.asarray(attitude_errors_deg)
    p95_error_deg = float(np.percentile(error_array, 95)) if error_array.size else math.inf
    maximum_error_deg = float(np.max(error_array)) if error_array.size else math.inf
    total_mass_kg = float(robot.data.default_mass[0].sum().item())
    env.close()
    checks = {
        "completed_horizon": completed_steps == args.steps,
        "no_termination": termination is None,
        "no_nonfinite": nonfinite_count == 0,
        "total_mass_28kg": abs(total_mass_kg - 28.0) < 0.01,
        "policy_action_dimension_unchanged": len(ACTION_NAMES) == 2,
        "commanded_motion_observed": target_amplitude_observed_deg > 0.9 * args.amplitude_deg,
        "all_ik_steps_converged": ik_nonconverged_steps == 0,
        "peak_pitch_below_limit": peak_pitch_deg < args.maximum_pitch_deg,
        "attitude_p95_below_limit": p95_error_deg < args.maximum_attitude_p95_deg,
        "attitude_max_below_limit": maximum_error_deg < args.maximum_attitude_error_deg,
    }
    result = {
        "schema": "two_wheel_semantic_dfr_attitude_smoke_v1",
        "semantic_contract": {
            "input": "semantic_DFR_world_quaternion",
            "conversion": "R_world_cam = R_world_DFR * Rz(+pi/2)",
            "observation_and_reward_link": "cam_link",
            "internal_actuation": list(PHYSICAL_GIMBAL_JOINTS),
            "learned_physical_gimbal_joint_action": False,
        },
        "steps": completed_steps,
        "command_amplitude_deg": args.amplitude_deg,
        "command_frequency_hz": args.frequency_hz,
        "observed_target_amplitude_deg": target_amplitude_observed_deg,
        "attitude_p95_error_deg": p95_error_deg,
        "attitude_max_error_deg": maximum_error_deg,
        "ik_max_residual_deg": float(np.max(ik_errors_deg)),
        "ik_nonconverged_steps": ik_nonconverged_steps,
        "gimbal_target_saturation_ratio": gimbal_target_saturation_steps / max(completed_steps, 1),
        "peak_gimbal_effort_nm": peak_gimbal_effort_nm,
        "peak_pitch_deg": peak_pitch_deg,
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
