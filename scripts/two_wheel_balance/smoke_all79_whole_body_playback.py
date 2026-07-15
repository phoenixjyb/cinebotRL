#!/usr/bin/env python3
"""Replay retargeted all-79 candidates with the whole-body balance controller."""

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
parser.add_argument("--retarget-dir", type=Path, required=True)
parser.add_argument("--urdf", type=Path, required=True)
parser.add_argument("--cases", default="1,20,28,50,79")
parser.add_argument("--maximum-pitch-deg", type=float, default=12.0)
parser.add_argument("--maximum-arm-error-deg", type=float, default=10.0)
parser.add_argument("--maximum-position-p95-m", type=float, default=0.15)
parser.add_argument("--maximum-position-error-m", type=float, default=0.25)
parser.add_argument("--maximum-action-saturation-ratio", type=float, default=0.20)
parser.add_argument("--maximum-arm-effort-saturation-ratio", type=float, default=0.20)
parser.add_argument("--arm-effort-limit-nm", type=float, default=30.0)
parser.add_argument("--arm-stiffness", type=float, default=400.0)
parser.add_argument("--arm-damping", type=float, default=40.0)
parser.add_argument("--open-loop", action="store_true")
parser.add_argument("--maximum-duration-scale", type=float, default=2.0)
parser.add_argument("--enable-phase-governor", action="store_true")
parser.add_argument("--disable-com-pitch-feedforward", action="store_true")
parser.add_argument("--disable-arm-gravity-feedforward", action="store_true")
parser.add_argument("--video-dir", type=Path)
parser.add_argument("--video-fps", type=int, default=50)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import torch

from isaaclab import sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
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
from rl_platform.tasks.two_wheel_balance.whole_body_tracking import (
    WholeBodyTrackingConfig,
    bounded_base_references,
    bounded_dls_arm_target,
    bounded_progress_scale,
    equilibrium_pitch_from_world_com,
    slew_limited_arm_target,
    yaw_from_quaternion_wxyz,
)
from task_spec import register_isaac_lab_tasks


POLICY_HZ = 200.0
WHEEL_RADIUS_M = 0.1016
ARM_JOINTS = (
    "joint6_arm_yaw",
    "joint5_arm_pitch",
    "joint4_elbow_pitch",
)


def load_candidate(case: int) -> dict[str, np.ndarray]:
    path = args.retarget_dir / f"case_{case:04d}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        candidate = {name: np.asarray(data[name]) for name in data.files}
    expected = {
        "time_s",
        "target_position_world_m",
        "base_arm_q",
        "control_v_wz_darm",
    }
    if not expected <= set(candidate):
        raise ValueError(f"candidate {path} is missing {sorted(expected - set(candidate))}")
    time_s = candidate["time_s"]
    if (
        time_s.ndim != 1
        or len(time_s) < 2
        or time_s[0] != 0.0
        or np.any(np.diff(time_s) <= 0.0)
    ):
        raise ValueError(f"invalid candidate time in {path}")
    if candidate["base_arm_q"].shape != (len(time_s), 6):
        raise ValueError(f"invalid candidate state shape in {path}")
    if candidate["target_position_world_m"].shape != (len(time_s), 3):
        raise ValueError(f"invalid candidate target shape in {path}")
    if candidate["control_v_wz_darm"].shape != (len(time_s) - 1, 5):
        raise ValueError(f"invalid candidate control shape in {path}")
    return candidate


def interpolate(
    candidate: dict[str, np.ndarray], elapsed_s: float
) -> tuple[np.ndarray, np.ndarray, float, float]:
    time_s = candidate["time_s"]
    upper = int(np.searchsorted(time_s, elapsed_s, side="right"))
    upper = min(max(upper, 1), len(time_s) - 1)
    lower = upper - 1
    dt = float(time_s[upper] - time_s[lower])
    alpha = np.clip((elapsed_s - time_s[lower]) / dt, 0.0, 1.0)
    state = (1.0 - alpha) * candidate["base_arm_q"][lower] + alpha * candidate[
        "base_arm_q"
    ][upper]
    target = (1.0 - alpha) * candidate["target_position_world_m"][lower] + alpha * candidate[
        "target_position_world_m"
    ][upper]
    control = candidate["control_v_wz_darm"][lower]
    return state, target, float(control[0]), float(control[1])


def evaluate_case(
    env,
    case: int,
    candidate: dict[str, np.ndarray],
    gain: np.ndarray,
    control_interval: int,
    kinematics: UrdfPositionKinematics,
    target_marker: VisualizationMarkers | None = None,
    path_marker: VisualizationMarkers | None = None,
) -> dict[str, object]:
    obs, _ = env.reset(seed=20260714 + case)
    unwrapped = env.unwrapped
    arm_ids = []
    for name in ARM_JOINTS:
        ids = unwrapped.robot.find_joints(name)[0]
        if len(ids) != 1:
            raise RuntimeError(f"expected one joint named {name}, got {ids}")
        arm_ids.append(ids[0])
    tool_ids = unwrapped.robot.find_bodies("ee1_tool")[0]
    if len(tool_ids) != 1:
        raise RuntimeError(f"expected semantic ee1_tool body, got {tool_ids}")

    controller_state = np.zeros((1, 6), dtype=np.float64)
    current_states = obs["policy"][:, : len(LQR_STATE_NAMES)].detach().cpu().numpy()
    action = np.zeros((1, len(ACTION_NAMES)), dtype=np.float32)
    config = cascaded_lqr_config("structural_robust_v1")
    tracking_config = WholeBodyTrackingConfig()
    source_duration_s = float(candidate["time_s"][-1])
    maximum_steps = int(
        math.ceil(source_duration_s * args.maximum_duration_scale * POLICY_HZ)
    ) + 1
    position_errors = []
    arm_servo_errors = []
    nominal_arm_errors = []
    vx_errors = []
    wz_errors = []
    peak_pitch_deg = 0.0
    peak_arm_error_deg = 0.0
    peak_nominal_arm_error_deg = 0.0
    peak_arm_effort_nm = 0.0
    peak_arm_gravity_feedforward_nm = 0.0
    peak_base_xy_error_m = 0.0
    peak_base_yaw_error_deg = 0.0
    peak_ik_correction_deg = 0.0
    peak_com_pitch_bias_deg = 0.0
    peak_position_error_step = 0
    peak_arm_error_step = 0
    saturated_actions = 0
    action_count = 0
    saturated_arm_efforts = 0
    arm_effort_count = 0
    termination = None
    completed_steps = 0
    trace = []
    phase_time_s = 0.0
    progress_scale = 1.0
    progress_scale_min = 1.0
    progress_scale_sum = 0.0
    previous_arm_target = (
        unwrapped.robot.data.joint_pos[0, arm_ids].detach().cpu().numpy().copy()
    )
    body_masses = unwrapped.robot.data.default_mass[0].to(unwrapped.device)
    if not hasattr(unwrapped.robot.data, "body_com_pos_w"):
        raise RuntimeError("Isaac articulation data does not expose body_com_pos_w")
    if path_marker is not None:
        path_marker.visualize(candidate["target_position_world_m"][::2])

    for step in range(maximum_steps):
        elapsed_s = step / POLICY_HZ
        desired_state, position_target, vx_feedforward, wz_feedforward = interpolate(
            candidate, phase_time_s
        )
        root_position = unwrapped.robot.data.root_pos_w[0].detach().cpu().numpy()
        root_quaternion = unwrapped.robot.data.root_quat_w[0].detach().cpu().numpy()
        actual_base = np.array(
            [root_position[0], root_position[1], yaw_from_quaternion_wxyz(root_quaternion)]
        )
        actual_arm = (
            unwrapped.robot.data.joint_pos[0, arm_ids].detach().cpu().numpy()
        )
        actual_tool_position = (
            unwrapped.robot.data.body_pos_w[0, tool_ids[0]].detach().cpu().numpy()
        )
        body_com_positions = unwrapped.robot.data.body_com_pos_w[0]
        center_of_mass_world = (
            torch.sum(body_masses[:, None] * body_com_positions, dim=0)
            / torch.sum(body_masses)
        ).detach().cpu().numpy()
        com_pitch_bias, _ = equilibrium_pitch_from_world_com(
            root_position,
            root_quaternion,
            center_of_mass_world,
            WHEEL_RADIUS_M,
        )
        peak_com_pitch_bias_deg = max(
            peak_com_pitch_bias_deg, math.degrees(abs(com_pitch_bias))
        )
        if args.open_loop:
            vx_ref = vx_feedforward
            wz_ref = wz_feedforward
            arm_target = desired_state[3:]
            ik_correction = np.zeros(3)
        else:
            vx_ref, wz_ref, _ = bounded_base_references(
                desired_state[:3],
                actual_base,
                vx_feedforward,
                wz_feedforward,
                tracking_config,
            )
            requested_arm_target, ik_diagnostics = bounded_dls_arm_target(
                kinematics,
                actual_base,
                actual_arm,
                desired_state[3:],
                position_target,
                actual_tool_position,
                tracking_config,
            )
            arm_target = slew_limited_arm_target(
                requested_arm_target,
                previous_arm_target,
                1.0 / POLICY_HZ,
                kinematics,
                tracking_config,
            )
            ik_correction = arm_target - desired_state[3:]
        previous_arm_target = arm_target.copy()
        if target_marker is not None:
            target_marker.visualize(position_target[None, :])
        arm_target_tensor = torch.as_tensor(
            arm_target[None, :], dtype=torch.float32, device=unwrapped.device
        )
        unwrapped.robot.set_joint_position_target(arm_target_tensor, joint_ids=arm_ids)
        arm_gravity_feedforward = np.zeros(3, dtype=np.float64)
        if not args.disable_arm_gravity_feedforward:
            arm_gravity_feedforward = kinematics.gravitational_effort_nm(
                np.concatenate((actual_base, actual_arm))
            )
            unwrapped.robot.set_joint_effort_target(
                torch.as_tensor(
                    arm_gravity_feedforward[None, :],
                    dtype=torch.float32,
                    device=unwrapped.device,
                ),
                joint_ids=arm_ids,
            )
        peak_arm_gravity_feedforward_nm = max(
            peak_arm_gravity_feedforward_nm,
            float(np.max(np.abs(arm_gravity_feedforward))),
        )
        unwrapped.vx_ref.fill_(vx_ref)
        unwrapped.wz_ref.fill_(wz_ref)
        if step % control_interval == 0:
            action, controller_state, _ = cascaded_lqr_action(
                current_states,
                np.array([vx_ref]),
                np.array([wz_ref]),
                gain,
                controller_state,
                control_dt=control_interval / POLICY_HZ,
                config=config,
                pitch_bias_override_rad=(
                    None
                    if args.disable_com_pitch_feedforward
                    else np.array([com_pitch_bias])
                ),
            )
            action = action.astype(np.float32)
        saturated_actions += int(np.count_nonzero(np.abs(action) >= config.action_limit - 1e-6))
        action_count += action.size
        obs, _, terminated, truncated, _ = env.step(
            torch.as_tensor(action, device=unwrapped.device)
        )
        current_states = obs["policy"][:, : len(LQR_STATE_NAMES)].detach().cpu().numpy()
        state = unwrapped._state_terms()
        actual_position = (
            unwrapped.robot.data.body_pos_w[0, tool_ids[0]].detach().cpu().numpy()
        )
        position_error = float(np.linalg.norm(actual_position - position_target))
        position_errors.append(position_error)
        actual_arm = unwrapped.robot.data.joint_pos[0, arm_ids]
        arm_error_by_joint = torch.abs(actual_arm - arm_target_tensor[0])
        arm_error = arm_error_by_joint.max().item()
        nominal_arm = torch.as_tensor(
            desired_state[3:], dtype=torch.float32, device=unwrapped.device
        )
        nominal_arm_error_by_joint = torch.abs(actual_arm - nominal_arm)
        nominal_arm_error = nominal_arm_error_by_joint.max().item()
        arm_servo_errors.append(arm_error_by_joint.detach().cpu().numpy())
        nominal_arm_errors.append(
            nominal_arm_error_by_joint.detach().cpu().numpy()
        )
        if math.degrees(arm_error) > peak_arm_error_deg:
            peak_arm_error_deg = math.degrees(arm_error)
            peak_arm_error_step = step + 1
        peak_nominal_arm_error_deg = max(
            peak_nominal_arm_error_deg, math.degrees(nominal_arm_error)
        )
        peak_ik_correction_deg = max(
            peak_ik_correction_deg,
            math.degrees(float(np.max(np.abs(ik_correction)))),
        )
        current_root_position = (
            unwrapped.robot.data.root_pos_w[0, :2].detach().cpu().numpy()
        )
        current_root_quaternion = (
            unwrapped.robot.data.root_quat_w[0].detach().cpu().numpy()
        )
        base_xy_error = float(
            np.linalg.norm(current_root_position - desired_state[:2])
        )
        base_yaw_error = abs(
            math.atan2(
                math.sin(
                    yaw_from_quaternion_wxyz(current_root_quaternion)
                    - desired_state[2]
                ),
                math.cos(
                    yaw_from_quaternion_wxyz(current_root_quaternion)
                    - desired_state[2]
                ),
            )
        )
        peak_base_xy_error_m = max(peak_base_xy_error_m, base_xy_error)
        peak_base_yaw_error_deg = max(
            peak_base_yaw_error_deg, math.degrees(base_yaw_error)
        )
        arm_effort_values = np.zeros(3, dtype=np.float64)
        if hasattr(unwrapped.robot.data, "applied_torque"):
            arm_effort = unwrapped.robot.data.applied_torque[0, arm_ids].abs()
            arm_effort_values = arm_effort.detach().cpu().numpy()
            peak_arm_effort_nm = max(peak_arm_effort_nm, arm_effort.max().item())
            saturated_arm_efforts += int(
                torch.count_nonzero(
                    arm_effort >= args.arm_effort_limit_nm - 1e-3
                ).item()
            )
            arm_effort_count += len(arm_ids)
        if position_error >= max(position_errors):
            peak_position_error_step = step + 1
        progress_scale = (
            bounded_progress_scale(
                base_xy_error, position_error, tracking_config
            )
            if args.enable_phase_governor
            else 1.0
        )
        progress_scale_min = min(progress_scale_min, progress_scale)
        progress_scale_sum += progress_scale
        if (
            step % 200 == 0
            or step + 1 == maximum_steps
            or phase_time_s >= source_duration_s
        ):
            trace.append(
                {
                    "step": step + 1,
                    "elapsed_s": elapsed_s,
                    "phase_time_s": phase_time_s,
                    "progress_scale": progress_scale,
                    "com_pitch_bias_deg": math.degrees(com_pitch_bias),
                    "position_error_m": position_error,
                    "pitch_deg": math.degrees(abs(float(state["pitch"][0].item()))),
                    "desired_base_q": desired_state[:3].tolist(),
                    "actual_base_q": [
                        float(current_root_position[0]),
                        float(current_root_position[1]),
                        yaw_from_quaternion_wxyz(current_root_quaternion),
                    ],
                    "nominal_arm_q": desired_state[3:].tolist(),
                    "commanded_arm_q": arm_target.tolist(),
                    "actual_arm_q": actual_arm.detach().cpu().numpy().tolist(),
                    "arm_effort_nm": arm_effort_values.tolist(),
                    "arm_gravity_feedforward_nm": arm_gravity_feedforward.tolist(),
                    "vx_reference_mps": vx_ref,
                    "vx_actual_mps": float(state["vx"][0].item()),
                    "wz_reference_radps": wz_ref,
                    "wz_actual_radps": float(state["yaw_rate"][0].item()),
                }
            )
        peak_pitch_deg = max(
            peak_pitch_deg, math.degrees(abs(float(state["pitch"][0].item())))
        )
        vx_errors.append(float(state["vx"][0].item()) - vx_ref)
        wz_errors.append(float(state["yaw_rate"][0].item()) - wz_ref)
        completed_steps = step + 1
        if bool((terminated | truncated)[0].item()):
            termination = {
                "step": completed_steps,
                "elapsed_s": elapsed_s,
                "terminated": bool(terminated[0].item()),
                "truncated": bool(truncated[0].item()),
                "reset_reason_counts": dict(unwrapped.reset_reason_counts),
            }
            break
        if phase_time_s >= source_duration_s:
            break
        phase_time_s = min(
            source_duration_s,
            phase_time_s + progress_scale / POLICY_HZ,
        )

    errors = np.asarray(position_errors)
    arm_servo_errors = np.asarray(arm_servo_errors)
    nominal_arm_errors = np.asarray(nominal_arm_errors)
    saturation_ratio = saturated_actions / max(action_count, 1)
    arm_effort_saturation_ratio = saturated_arm_efforts / max(arm_effort_count, 1)
    checks = {
        "completed_reference": phase_time_s >= source_duration_s,
        "no_termination": termination is None,
        "peak_pitch_below_limit": peak_pitch_deg <= args.maximum_pitch_deg,
        "peak_arm_error_below_limit": peak_arm_error_deg <= args.maximum_arm_error_deg,
        "position_p95_below_limit": float(np.percentile(errors, 95))
        <= args.maximum_position_p95_m,
        "position_max_below_limit": float(np.max(errors))
        <= args.maximum_position_error_m,
        "action_saturation_below_limit": saturation_ratio
        <= args.maximum_action_saturation_ratio,
        "arm_effort_saturation_below_limit": arm_effort_saturation_ratio
        <= args.maximum_arm_effort_saturation_ratio,
    }
    return {
        "case": case,
        "source_duration_s": source_duration_s,
        "maximum_steps": maximum_steps,
        "completed_steps": completed_steps,
        "wall_duration_s": completed_steps / POLICY_HZ,
        "completed_phase_time_s": phase_time_s,
        "progress_scale_min": progress_scale_min,
        "progress_scale_mean": progress_scale_sum / max(completed_steps, 1),
        "peak_pitch_deg": peak_pitch_deg,
        "peak_arm_error_deg": peak_arm_error_deg,
        "peak_arm_error_elapsed_s": (peak_arm_error_step - 1) / POLICY_HZ,
        "arm_error_p95_by_joint_deg": {
            name: math.degrees(float(np.percentile(arm_servo_errors[:, index], 95)))
            for index, name in enumerate(ARM_JOINTS)
        },
        "peak_nominal_arm_error_deg": peak_nominal_arm_error_deg,
        "nominal_arm_error_p95_by_joint_deg": {
            name: math.degrees(float(np.percentile(nominal_arm_errors[:, index], 95)))
            for index, name in enumerate(ARM_JOINTS)
        },
        "peak_arm_effort_nm": peak_arm_effort_nm,
        "peak_arm_gravity_feedforward_nm": peak_arm_gravity_feedforward_nm,
        "peak_base_xy_error_m": peak_base_xy_error_m,
        "peak_base_yaw_error_deg": peak_base_yaw_error_deg,
        "peak_ik_correction_deg": peak_ik_correction_deg,
        "peak_com_pitch_bias_deg": peak_com_pitch_bias_deg,
        "position_error_mean_m": float(np.mean(errors)),
        "position_error_p95_m": float(np.percentile(errors, 95)),
        "position_error_max_m": float(np.max(errors)),
        "position_error_max_elapsed_s": (peak_position_error_step - 1) / POLICY_HZ,
        "vx_rmse_mps": float(np.sqrt(np.mean(np.square(vx_errors)))),
        "wz_rmse_radps": float(np.sqrt(np.mean(np.square(wz_errors)))),
        "action_saturation_ratio": saturation_ratio,
        "arm_effort_saturation_ratio": arm_effort_saturation_ratio,
        "termination": termination,
        "trace_1hz": trace,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    cases = [int(item) for item in args.cases.split(",") if item.strip()]
    if not cases or len(set(cases)) != len(cases):
        raise ValueError(f"invalid cases: {cases}")
    if args.video_dir is not None and len(cases) != 1:
        raise ValueError("video recording requires exactly one case")
    if args.video_fps < 1:
        raise ValueError("--video-fps must be positive")
    candidates = {case: load_candidate(case) for case in cases}
    kinematics = UrdfPositionKinematics(args.urdf)
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
    cfg.episode_length_s = (
        max(float(item["time_s"][-1]) for item in candidates.values())
        * args.maximum_duration_scale
        + 2.0
    )
    cfg.reset_pitch_rad = 0.0
    cfg.control_mode = "direct"
    if args.video_dir is not None:
        cfg.viewer.eye = (3.2, -4.5, 2.2)
        cfg.viewer.lookat = (0.0, 0.0, 1.0)
    raw_env = gym.make(
        "RecomoTwoWheelBalance-v0",
        cfg=cfg,
        render_mode="rgb_array" if args.video_dir is not None else None,
        disable_env_checker=True,
    )
    target_marker = None
    path_marker = None
    if args.video_dir is not None:
        raw_env.unwrapped.sim.set_camera_view(
            eye=cfg.viewer.eye, target=cfg.viewer.lookat
        )
        target_marker = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/All79PlaybackCurrentTarget",
                markers={
                    "target": sim_utils.SphereCfg(
                        radius=0.05,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 0.15, 0.05)
                        ),
                    )
                },
            )
        )
        path_marker = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/All79PlaybackTargetPath",
                markers={
                    "path": sim_utils.SphereCfg(
                        radius=0.012,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.1, 0.65, 1.0)
                        ),
                    )
                },
            )
        )
        args.video_dir.mkdir(parents=True, exist_ok=True)
        video_length = int(
            math.ceil(
                max(float(item["time_s"][-1]) for item in candidates.values())
                * args.maximum_duration_scale
                * POLICY_HZ
            )
        ) + 1
        env = gym.wrappers.RecordVideo(
            raw_env,
            video_folder=str(args.video_dir),
            step_trigger=lambda step: step == 0,
            video_length=video_length,
            fps=args.video_fps,
            name_prefix=f"two-wheel-all79-case-{cases[0]:04d}",
            disable_logger=True,
        )
    else:
        env = raw_env
    results = [
        evaluate_case(
            env,
            case,
            candidates[case],
            gain,
            control_interval,
            kinematics,
            target_marker,
            path_marker,
        )
        for case in cases
    ]
    env.close()
    result = {
        "schema": "recomo_two_wheel_all79_whole_body_playback_smoke_v2",
        "training_started": False,
        "controller_profile": "structural_robust_v1",
        "task_space_feedback_enabled": not args.open_loop,
        "phase_governor_enabled": args.enable_phase_governor,
        "com_pitch_feedforward_enabled": not args.disable_com_pitch_feedforward,
        "arm_gravity_feedforward_enabled": not args.disable_arm_gravity_feedforward,
        "arm_stiffness": args.arm_stiffness,
        "arm_damping": args.arm_damping,
        "position_target_link": "ee1_tool",
        "camera_attitude_tracking_enabled": False,
        "physical_gimbal_control_enabled": False,
        "video_fps": args.video_fps if args.video_dir is not None else None,
        "video_playback_slowdown": (
            POLICY_HZ / args.video_fps if args.video_dir is not None else None
        ),
        "cases": cases,
        "passed_case_count": sum(item["passed"] for item in results),
        "results": results,
        "passed": all(item["passed"] for item in results),
    }
    if args.video_dir is not None:
        videos = sorted(
            args.video_dir.glob("*.mp4"), key=lambda path: path.stat().st_mtime
        )
        if not videos:
            raise RuntimeError(f"RecordVideo did not produce an mp4 in {args.video_dir}")
        result["video"] = str(videos[-1].resolve())
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
