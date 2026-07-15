#!/usr/bin/env python3
"""Run parallel jerk-limited riser round trips while balancing."""

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
parser.add_argument("--steps", type=int, default=10000)
parser.add_argument("--dwell-s", type=float, default=0.5)
parser.add_argument("--maximum-pitch-deg", type=float, default=12.0)
parser.add_argument("--pitch-p95-deg", type=float, default=6.0)
parser.add_argument("--maximum-height-error-m", type=float, default=0.03)
parser.add_argument("--maximum-travel-overshoot-m", type=float, default=0.01)
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
from rl_platform.tasks.two_wheel_balance.riser_control import (
    QuinticRiserMove,
    RiserLimits,
    balance_progress_scale,
)
from task_spec import register_isaac_lab_tasks


STAGE_SPEEDS_MPS = np.array([0.1, 0.25, 0.5, 1.0], dtype=np.float64)
GIMBAL_JOINTS = (
    "joint3_gimbal_yaw",
    "joint2_gimbal_roll",
    "joint1_gimbal_pitch",
)


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
    cfg.scene.num_envs = len(STAGE_SPEEDS_MPS)
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
    if len(cam_ids) != 1:
        raise RuntimeError(f"expected one cam_link, got {cam_ids}")

    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = robot.data.default_joint_vel[env_ids].clone()
    joint_pos[:, riser_id] = 0.0
    joint_pos[:, gimbal_ids] = 0.0
    joint_vel.zero_()
    robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
    gimbal_targets = torch.zeros(
        (unwrapped.num_envs, len(gimbal_ids)),
        dtype=torch.float32,
        device=unwrapped.device,
    )

    policy_dt = cfg.sim.dt * cfg.decimation
    limits = RiserLimits()
    ascent_moves = [
        QuinticRiserMove.for_peak_velocity(0.0, 1.2, float(speed), limits)
        for speed in STAGE_SPEEDS_MPS
    ]
    descent_moves = [
        QuinticRiserMove.for_peak_velocity(1.2, 0.0, float(speed), limits)
        for speed in STAGE_SPEEDS_MPS
    ]
    references = [move.sample(0.0) for move in ascent_moves]
    phases = ["ascending" for _ in STAGE_SPEEDS_MPS]
    phase_elapsed_s = np.zeros(unwrapped.num_envs, dtype=np.float64)
    completed_roundtrip = np.zeros(unwrapped.num_envs, dtype=bool)

    controller_state = np.zeros((unwrapped.num_envs, 6), dtype=np.float64)
    current_states = obs["policy"][:, : len(LQR_STATE_NAMES)].detach().cpu().numpy()
    action = np.zeros((unwrapped.num_envs, len(ACTION_NAMES)), dtype=np.float32)
    controller_cfg = cascaded_lqr_config("structural_robust_v1")

    pitch_samples = [[] for _ in STAGE_SPEEDS_MPS]
    height_error_samples = [[] for _ in STAGE_SPEEDS_MPS]
    measured_speed_peak = np.zeros(unwrapped.num_envs)
    reference_speed_peak = np.zeros(unwrapped.num_envs)
    lower_undershoot = np.zeros(unwrapped.num_envs)
    upper_overshoot = np.zeros(unwrapped.num_envs)
    wheel_saturated = np.zeros(unwrapped.num_envs, dtype=np.int64)
    riser_saturated = np.zeros(unwrapped.num_envs, dtype=np.int64)
    completion_steps = [None for _ in STAGE_SPEEDS_MPS]
    nonfinite_count = 0
    completed_steps = 0
    termination = None

    for step in range(args.steps):
        pitch_now = unwrapped._state_terms()["pitch"].detach().cpu().numpy()
        for index, speed in enumerate(STAGE_SPEEDS_MPS):
            if completed_roundtrip[index]:
                continue
            scale = balance_progress_scale(float(abs(pitch_now[index])))
            if phases[index] == "ascending":
                move = ascent_moves[index]
                phase_elapsed_s[index] = min(
                    move.duration_s, phase_elapsed_s[index] + policy_dt * scale
                )
                references[index] = move.sample(phase_elapsed_s[index])
                if phase_elapsed_s[index] >= move.duration_s:
                    phases[index] = "upper_dwell"
                    phase_elapsed_s[index] = 0.0
            elif phases[index] == "upper_dwell":
                references[index] = ascent_moves[index].sample(
                    ascent_moves[index].duration_s
                )
                phase_elapsed_s[index] += policy_dt
                if phase_elapsed_s[index] >= args.dwell_s:
                    phases[index] = "descending"
                    phase_elapsed_s[index] = 0.0
            elif phases[index] == "descending":
                move = descent_moves[index]
                phase_elapsed_s[index] = min(
                    move.duration_s, phase_elapsed_s[index] + policy_dt * scale
                )
                references[index] = move.sample(phase_elapsed_s[index])
                if phase_elapsed_s[index] >= move.duration_s:
                    phases[index] = "complete"
                    completed_roundtrip[index] = True
                    completion_steps[index] = step + 1
            reference_speed_peak[index] = max(
                reference_speed_peak[index], abs(references[index].velocity_mps)
            )

        riser_targets = torch.as_tensor(
            [[reference.position_m] for reference in references],
            dtype=torch.float32,
            device=unwrapped.device,
        )
        riser_velocity_targets = torch.as_tensor(
            [[reference.velocity_mps] for reference in references],
            dtype=torch.float32,
            device=unwrapped.device,
        )
        robot.set_joint_position_target(riser_targets, joint_ids=[riser_id])
        robot.set_joint_velocity_target(
            riser_velocity_targets, joint_ids=[riser_id]
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
        expected_z = riser_targets[:, 0] + 0.6 + float(cfg.robot_cfg.init_state.pos[2])
        height_error = (camera_z - expected_z).abs()
        riser_position = robot.data.joint_pos[:, riser_id]
        riser_velocity = robot.data.joint_vel[:, riser_id]
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
        riser_position_np = riser_position.detach().cpu().numpy()
        riser_velocity_np = riser_velocity.abs().detach().cpu().numpy()
        wheel_effort = robot.data.applied_torque[:, unwrapped._wheel_joint_idx].abs()
        riser_effort = robot.data.applied_torque[:, riser_id].abs()
        wheel_saturated += (
            wheel_effort.max(dim=1).values >= 0.99 * cfg.torque_limit_nm
        ).detach().cpu().numpy()
        riser_saturated += (riser_effort >= 0.99 * 300.0).detach().cpu().numpy()
        for index in range(unwrapped.num_envs):
            pitch_samples[index].append(float(pitch_deg[index]))
            height_error_samples[index].append(float(height_error_np[index]))
            measured_speed_peak[index] = max(
                measured_speed_peak[index], float(riser_velocity_np[index])
            )
            lower_undershoot[index] = max(
                lower_undershoot[index], max(0.0, -float(riser_position_np[index]))
            )
            upper_overshoot[index] = max(
                upper_overshoot[index],
                max(0.0, float(riser_position_np[index]) - limits.upper_m),
            )
        completed_steps = step + 1
        done = terminated | truncated
        if torch.any(done):
            termination = {
                "step": step + 1,
                "done_env_ids": torch.nonzero(done).flatten().detach().cpu().tolist(),
                "reset_reason_counts": dict(unwrapped.reset_reason_counts),
            }
            break
        if np.all(completed_roundtrip):
            break

    rows = []
    for index, speed in enumerate(STAGE_SPEEDS_MPS):
        pitches = np.asarray(pitch_samples[index], dtype=np.float64)
        errors = np.asarray(height_error_samples[index], dtype=np.float64)
        rows.append(
            {
                "requested_speed_mps": float(speed),
                "completed_roundtrip": bool(completed_roundtrip[index]),
                "completion_step": completion_steps[index],
                "reference_speed_peak_mps": float(reference_speed_peak[index]),
                "measured_speed_peak_mps": float(measured_speed_peak[index]),
                "pitch_p95_deg": float(np.percentile(pitches, 95)) if pitches.size else math.inf,
                "pitch_max_deg": float(np.max(pitches)) if pitches.size else math.inf,
                "camera_height_error_p95_m": float(np.percentile(errors, 95)) if errors.size else math.inf,
                "camera_height_error_max_m": float(np.max(errors)) if errors.size else math.inf,
                "travel_overshoot_m": float(max(lower_undershoot[index], upper_overshoot[index])),
                "wheel_saturation_ratio": float(wheel_saturated[index] / max(completed_steps, 1)),
                "riser_saturation_ratio": float(riser_saturated[index] / max(completed_steps, 1)),
            }
        )

    env.close()
    checks = {
        "all_roundtrips_completed": all(row["completed_roundtrip"] for row in rows),
        "no_termination": termination is None,
        "no_nonfinite": nonfinite_count == 0,
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
        "all_travel_overshoot_below_limit": all(
            row["travel_overshoot_m"] <= args.maximum_travel_overshoot_m
            for row in rows
        ),
        "all_saturation_ratios_below_limit": all(
            row["wheel_saturation_ratio"] <= args.maximum_saturation_ratio
            and row["riser_saturation_ratio"] <= args.maximum_saturation_ratio
            for row in rows
        ),
        "measured_speed_within_hard_limit": all(
            row["measured_speed_peak_mps"] <= 1.02 for row in rows
        ),
    }
    result = {
        "schema": "recomo_two_wheel_riser_dynamic_gate_v1",
        "controller_profile": "structural_robust_v1",
        "riser_drive_contract": "position_plus_velocity_feedforward",
        "reference_limits": {
            "velocity_mps": limits.maximum_velocity_mps,
            "acceleration_mps2": limits.maximum_acceleration_mps2,
            "jerk_mps3": limits.maximum_jerk_mps3,
        },
        "requested_steps": args.steps,
        "completed_steps": completed_steps,
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
