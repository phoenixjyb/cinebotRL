#!/usr/bin/env python3
"""Verify signed arm-joint motion while the whole-body balance LQR is active."""

from __future__ import annotations

import argparse
import copy
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
parser.add_argument("--joint", required=True)
parser.add_argument("--delta-rad", type=float, default=-0.2)
parser.add_argument("--hold-steps", type=int, default=300)
parser.add_argument("--motion-steps", type=int, default=600)
parser.add_argument("--settle-steps", type=int, default=300)
parser.add_argument("--minimum-motion-ratio", type=float, default=0.8)
parser.add_argument("--maximum-pitch-deg", type=float, default=12.0)
parser.add_argument("--arm-stiffness", type=float, default=200.0)
parser.add_argument("--arm-damping", type=float, default=20.0)
parser.add_argument("--urdf", type=Path)
parser.add_argument("--gravity-compensation", action="store_true")
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import torch

from rl_platform.robots.two_wheel_balance import TWO_WHEEL_WHOLE_BODY_CFG
from rl_platform.tasks.two_wheel_balance import RecomoTwoWheelBalanceEnvCfg
from rl_platform.tasks.two_wheel_balance.metrics import (
    ACTION_NAMES,
    LQR_STATE_NAMES,
    cascaded_lqr_action,
    cascaded_lqr_config,
)
from rl_platform.tasks.two_wheel_balance.whole_body_kinematics import (
    UrdfPositionKinematics,
)
from task_spec import register_isaac_lab_tasks


POLICY_HZ = 200.0
ARM_HOME = {
    "joint6_arm_yaw": 0.0,
    "joint5_arm_pitch": math.pi / 2.0,
    "joint4_elbow_pitch": 3.0 * math.pi / 4.0,
}


def main() -> int:
    if args.joint not in ARM_HOME:
        raise ValueError(f"unsupported arm joint {args.joint!r}")
    if args.gravity_compensation and args.urdf is None:
        raise ValueError("--urdf is required with --gravity-compensation")
    kinematics = (
        UrdfPositionKinematics(args.urdf) if args.gravity_compensation else None
    )
    gain_data = json.loads(args.gains.read_text(encoding="utf-8"))
    gain = np.asarray(gain_data["selected_gain"], dtype=np.float64)
    control_interval = int(gain_data["control_interval_steps"])
    if gain.shape != (len(ACTION_NAMES), len(LQR_STATE_NAMES)):
        raise ValueError(f"invalid gain shape: {gain.shape}")

    register_isaac_lab_tasks()
    cfg = RecomoTwoWheelBalanceEnvCfg()
    cfg.seed = 20260714
    cfg.scene.num_envs = 1
    cfg.robot_cfg = copy.deepcopy(TWO_WHEEL_WHOLE_BODY_CFG)
    cfg.robot_cfg.actuators["arm_home_hold"].stiffness = args.arm_stiffness
    cfg.robot_cfg.actuators["arm_home_hold"].damping = args.arm_damping
    total_steps = args.hold_steps + args.motion_steps + args.settle_steps
    cfg.episode_length_s = (total_steps + 100) / POLICY_HZ
    cfg.reset_pitch_rad = 0.0
    cfg.control_mode = "direct"
    env = gym.make(
        "RecomoTwoWheelBalance-v0",
        cfg=cfg,
        render_mode=None,
        disable_env_checker=True,
    )
    obs, _ = env.reset(seed=20260714)
    unwrapped = env.unwrapped
    arm_names = tuple(ARM_HOME)
    arm_ids = []
    for name in arm_names:
        ids = unwrapped.robot.find_joints(name)[0]
        if len(ids) != 1:
            raise RuntimeError(f"expected one joint named {name}, got {ids}")
        arm_ids.append(ids[0])
    tested_index = arm_names.index(args.joint)

    home = np.asarray([ARM_HOME[name] for name in arm_names], dtype=np.float64)
    commanded = home.copy()
    commanded[tested_index] += args.delta_rad
    controller_state = np.zeros((1, 6), dtype=np.float64)
    current_states = obs["policy"][:, : len(LQR_STATE_NAMES)].detach().cpu().numpy()
    action = np.zeros((1, len(ACTION_NAMES)), dtype=np.float32)
    config = cascaded_lqr_config("structural_robust_v1")
    initial_position = float(unwrapped.robot.data.joint_pos[0, arm_ids[tested_index]])
    peak_pitch_deg = 0.0
    termination = None
    trace = []

    for step in range(total_steps):
        target = home if step < args.hold_steps else commanded
        target_tensor = torch.as_tensor(
            target[None, :], dtype=torch.float32, device=unwrapped.device
        )
        unwrapped.robot.set_joint_position_target(target_tensor, joint_ids=arm_ids)
        gravity_effort = np.zeros(3, dtype=np.float64)
        if kinematics is not None:
            actual_arm = (
                unwrapped.robot.data.joint_pos[0, arm_ids].detach().cpu().numpy()
            )
            gravity_effort = kinematics.gravitational_effort_nm(
                np.concatenate((np.zeros(3), actual_arm))
            )
            unwrapped.robot.set_joint_effort_target(
                torch.as_tensor(
                    gravity_effort[None, :],
                    dtype=torch.float32,
                    device=unwrapped.device,
                ),
                joint_ids=arm_ids,
            )
        if step % control_interval == 0:
            action, controller_state, _ = cascaded_lqr_action(
                current_states,
                np.zeros(1),
                np.zeros(1),
                gain,
                controller_state,
                control_dt=control_interval / POLICY_HZ,
                config=config,
            )
            action = action.astype(np.float32)
        obs, _, terminated, truncated, _ = env.step(
            torch.as_tensor(action, device=unwrapped.device)
        )
        current_states = obs["policy"][:, : len(LQR_STATE_NAMES)].detach().cpu().numpy()
        state = unwrapped._state_terms()
        peak_pitch_deg = max(
            peak_pitch_deg, math.degrees(abs(float(state["pitch"][0].item())))
        )
        if step % 100 == 0 or step + 1 == total_steps:
            trace.append(
                {
                    "step": step + 1,
                    "target_rad": float(target[tested_index]),
                    "position_rad": float(
                        unwrapped.robot.data.joint_pos[0, arm_ids[tested_index]]
                    ),
                    "velocity_radps": float(
                        unwrapped.robot.data.joint_vel[0, arm_ids[tested_index]]
                    ),
                    "applied_effort_nm": float(
                        unwrapped.robot.data.applied_torque[0, arm_ids[tested_index]]
                    ),
                    "gravity_feedforward_nm": float(
                        gravity_effort[tested_index]
                    ),
                    "pitch_deg": math.degrees(abs(float(state["pitch"][0].item()))),
                }
            )
        if bool((terminated | truncated)[0].item()):
            termination = {
                "step": step + 1,
                "terminated": bool(terminated[0].item()),
                "truncated": bool(truncated[0].item()),
                "reset_reason_counts": dict(unwrapped.reset_reason_counts),
            }
            break

    final_position = float(unwrapped.robot.data.joint_pos[0, arm_ids[tested_index]])
    env.close()
    requested_motion = float(commanded[tested_index] - initial_position)
    achieved_motion = final_position - initial_position
    motion_ratio = achieved_motion / requested_motion if requested_motion != 0.0 else 0.0
    checks = {
        "no_termination": termination is None,
        "correct_direction": achieved_motion * requested_motion > 0.0,
        "motion_ratio_sufficient": motion_ratio >= args.minimum_motion_ratio,
        "pitch_below_limit": peak_pitch_deg <= args.maximum_pitch_deg,
    }
    result = {
        "schema": "recomo_two_wheel_whole_body_arm_motion_smoke_v1",
        "training_started": False,
        "joint": args.joint,
        "arm_stiffness": args.arm_stiffness,
        "arm_damping": args.arm_damping,
        "gravity_compensation_enabled": args.gravity_compensation,
        "initial_position_rad": initial_position,
        "target_position_rad": float(commanded[tested_index]),
        "final_position_rad": final_position,
        "requested_motion_rad": requested_motion,
        "achieved_motion_rad": achieved_motion,
        "motion_ratio": motion_ratio,
        "peak_pitch_deg": peak_pitch_deg,
        "termination": termination,
        "trace": trace,
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
