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
parser.add_argument("--maximum-com-pitch-bias-deg", type=float, default=10.5)
parser.add_argument("--maximum-controller-pitch-target-deg", type=float, default=11.5)
parser.add_argument("--maximum-arm-error-deg", type=float, default=10.0)
parser.add_argument("--maximum-task-space-base-offset-m", type=float, default=0.40)
parser.add_argument(
    "--maximum-task-space-base-offset-rate-mps", type=float, default=0.20
)
parser.add_argument("--maximum-position-p95-m", type=float, default=0.15)
parser.add_argument("--maximum-position-error-m", type=float, default=0.25)
parser.add_argument("--maximum-attitude-p95-deg", type=float, default=8.0)
parser.add_argument("--maximum-attitude-error-deg", type=float, default=15.0)
parser.add_argument("--maximum-gimbal-error-deg", type=float, default=10.0)
parser.add_argument("--maximum-gimbal-target-saturation-ratio", type=float, default=0.20)
parser.add_argument("--camera-attitude-feedback-gain", type=float, default=0.7)
parser.add_argument("--camera-attitude-feedback-damping", type=float, default=0.05)
parser.add_argument(
    "--camera-attitude-feedback-time-constant-s", type=float, default=0.10
)
parser.add_argument(
    "--maximum-gimbal-feedback-joint-offset-deg", type=float, default=15.0
)
parser.add_argument("--maximum-action-saturation-ratio", type=float, default=0.20)
parser.add_argument("--maximum-arm-effort-saturation-ratio", type=float, default=0.20)
parser.add_argument("--arm-effort-limit-nm", type=float, default=30.0)
parser.add_argument("--maximum-gimbal-effort-saturation-ratio", type=float, default=0.20)
parser.add_argument("--gimbal-effort-limit-nm", type=float, default=10.0)
parser.add_argument("--arm-stiffness", type=float, default=400.0)
parser.add_argument("--arm-damping", type=float, default=40.0)
parser.add_argument("--open-loop", action="store_true")
parser.add_argument(
    "--enable-acquisition-task-space-arm-feedback", action="store_true"
)
parser.add_argument("--maximum-duration-scale", type=float, default=2.0)
parser.add_argument("--enable-phase-governor", action="store_true")
parser.add_argument("--phase-governor-pitch-start-deg", type=float, default=10.5)
parser.add_argument("--phase-governor-pitch-stop-deg", type=float, default=11.5)
parser.add_argument("--phase-governor-attitude-start-deg", type=float, default=4.0)
parser.add_argument("--disable-com-pitch-feedforward", action="store_true")
parser.add_argument("--disable-arm-gravity-feedforward", action="store_true")
parser.add_argument("--disable-gimbal-gravity-feedforward", action="store_true")
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
from rl_platform.robots.two_wheel_balance import TWO_WHEEL_WHOLE_BODY_ATTITUDE_CFG
from rl_platform.tasks.two_wheel_balance import RecomoTwoWheelBalanceEnvCfg
from rl_platform.tasks.two_wheel_balance.all79_reference import quaternion_slerp_wxyz
from rl_platform.tasks.two_wheel_balance.exact_source_reference import (
    validate_exact_source_candidate,
)
from rl_platform.tasks.two_wheel_balance.camera_attitude import (
    PHYSICAL_GIMBAL_JOINTS,
    UrdfPhysicalCameraKinematics,
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
from rl_platform.tasks.two_wheel_balance.whole_body_kinematics import (
    UrdfPositionKinematics,
)
from rl_platform.tasks.two_wheel_balance.whole_body_tracking import (
    WholeBodyTrackingConfig,
    bounded_attitude_progress_scale,
    bounded_balance_progress_scale,
    bounded_base_references,
    bounded_phase_progress_scale,
    bounded_semantic_arm_target,
    bounded_task_space_base_target,
    equilibrium_pitch_from_world_com,
    phase_scaled_feedforward,
    quaternion_from_pitch_yaw_wxyz,
    roll_pitch_from_quaternion_wxyz,
    slew_limited_planar_offset,
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
    candidate = validate_exact_source_candidate(path)
    expected = {
        "schema",
        "trajectory_integrity_contract",
        "source_trajectory_integrity_passed",
        "offline_executable_quality_passed",
        "valid_for_dynamic_evaluation",
        "physical_gimbal_joint_labels_included",
        "time_s",
        "semantic_start_index",
        "target_position_world_m",
        "base_arm_q",
        "control_v_wz_darm",
        "target_attitude_world_dfr_quat_wxyz",
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
    if candidate["target_attitude_world_dfr_quat_wxyz"].shape != (len(time_s), 4):
        raise ValueError(f"invalid candidate attitude shape in {path}")
    semantic_start_index = int(candidate["semantic_start_index"].item())
    if not 1 <= semantic_start_index < len(time_s) - 1:
        raise ValueError(f"invalid semantic start index in {path}")
    if str(candidate.get("schema", "").item()) != "cinebotrl_two_wheel_exact_source_retarget_v1":
        raise ValueError(f"candidate {path} is not exact-source schema v1")
    if str(candidate["trajectory_integrity_contract"].item()) != "exact_source_v1":
        raise ValueError(f"candidate {path} does not preserve exact_source_v1")
    if not bool(candidate["source_trajectory_integrity_passed"].item()):
        raise ValueError(f"candidate {path} failed source trajectory integrity")
    if not bool(candidate["offline_executable_quality_passed"].item()):
        raise ValueError(f"candidate {path} failed offline executable quality")
    if not bool(candidate["valid_for_dynamic_evaluation"].item()):
        raise ValueError(f"candidate {path} is not admitted for dynamic evaluation")
    if bool(candidate.get("physical_gimbal_joint_labels_included", True).item()):
        raise ValueError(f"candidate {path} contains physical gimbal joint labels")
    return candidate


def interpolate(
    candidate: dict[str, np.ndarray], elapsed_s: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
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
    attitude = quaternion_slerp_wxyz(
        candidate["target_attitude_world_dfr_quat_wxyz"][lower],
        candidate["target_attitude_world_dfr_quat_wxyz"][upper],
        np.array([alpha]),
    )[0]
    control = candidate["control_v_wz_darm"][lower]
    return state, target, attitude, float(control[0]), float(control[1])


def evaluate_case(
    env,
    case: int,
    candidate: dict[str, np.ndarray],
    gain: np.ndarray,
    control_interval: int,
    kinematics: UrdfPositionKinematics,
    camera_kinematics: UrdfPhysicalCameraKinematics,
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
    gimbal_ids = []
    for name in PHYSICAL_GIMBAL_JOINTS:
        ids = unwrapped.robot.find_joints(name)[0]
        if len(ids) != 1:
            raise RuntimeError(f"expected one joint named {name}, got {ids}")
        gimbal_ids.append(ids[0])
    tool_ids = unwrapped.robot.find_bodies("ee1_tool")[0]
    if len(tool_ids) != 1:
        raise RuntimeError(f"expected semantic ee1_tool body, got {tool_ids}")
    cam_ids = unwrapped.robot.find_bodies("cam_link")[0]
    if len(cam_ids) != 1:
        raise RuntimeError(f"expected physical cam_link body, got {cam_ids}")

    controller_state = np.zeros((1, 6), dtype=np.float64)
    current_states = obs["policy"][:, : len(LQR_STATE_NAMES)].detach().cpu().numpy()
    action = np.zeros((1, len(ACTION_NAMES)), dtype=np.float32)
    config = cascaded_lqr_config(
        "structural_robust_v1",
        pitch_bias_limit_rad=math.radians(args.maximum_com_pitch_bias_deg),
        total_pitch_target_limit_rad=math.radians(
            args.maximum_controller_pitch_target_deg
        ),
    )
    tracking_config = WholeBodyTrackingConfig(
        progress_error_full_m=args.maximum_position_p95_m,
        minimum_progress_scale=0.0,
    )
    source_duration_s = float(candidate["time_s"][-1])
    semantic_start_index = int(candidate["semantic_start_index"].item())
    semantic_start_time_s = float(candidate["time_s"][semantic_start_index])
    maximum_steps = int(
        math.ceil(source_duration_s * args.maximum_duration_scale * POLICY_HZ)
    ) + 1
    position_errors = []
    attitude_errors_deg = []
    gimbal_servo_errors = []
    arm_servo_errors = []
    nominal_arm_errors = []
    vx_errors = []
    wz_errors = []
    peak_pitch_deg = 0.0
    peak_arm_error_deg = 0.0
    peak_nominal_arm_error_deg = 0.0
    peak_arm_effort_nm = 0.0
    peak_gimbal_error_deg = 0.0
    peak_gimbal_effort_nm = 0.0
    peak_gimbal_ik_residual_deg = 0.0
    peak_gimbal_feedback_joint_offset_deg = 0.0
    peak_arm_gravity_feedforward_nm = 0.0
    peak_gimbal_gravity_feedforward_nm = 0.0
    peak_base_xy_error_m = 0.0
    peak_base_yaw_error_deg = 0.0
    peak_ik_correction_deg = 0.0
    peak_root_tilt_displacement_m = 0.0
    peak_task_space_base_offset_m = 0.0
    peak_com_pitch_bias_deg = 0.0
    peak_controller_pitch_target_deg = 0.0
    peak_position_error_step = 0
    peak_arm_error_step = 0
    saturated_actions = 0
    action_count = 0
    saturated_arm_efforts = 0
    arm_effort_count = 0
    saturated_gimbal_efforts = 0
    gimbal_effort_count = 0
    gimbal_target_saturation_steps = 0
    gimbal_ik_nonconverged_steps = 0
    termination = None
    completed_steps = 0
    trace = []
    phase_time_s = 0.0
    progress_scale = 1.0
    tracking_progress_scale = 1.0
    balance_progress_scale = 1.0
    attitude_progress_scale = 1.0
    progress_scale_min = 1.0
    balance_progress_scale_min = 1.0
    attitude_progress_scale_min = 1.0
    progress_scale_sum = 0.0
    previous_arm_target = (
        unwrapped.robot.data.joint_pos[0, arm_ids].detach().cpu().numpy().copy()
    )
    initial_gimbal_target = (
        unwrapped.robot.data.joint_pos[0, gimbal_ids]
        .detach()
        .cpu()
        .numpy()
        .copy()
    )
    previous_gimbal_target = initial_gimbal_target.copy()
    previous_nominal_gimbal_target = initial_gimbal_target.copy()
    previous_camera_feedback_correction = np.zeros(3, dtype=np.float64)
    previous_task_space_base_offset = np.zeros(2, dtype=np.float64)
    body_masses = unwrapped.robot.data.default_mass[0].to(unwrapped.device)
    if not hasattr(unwrapped.robot.data, "body_com_pos_w"):
        raise RuntimeError("Isaac articulation data does not expose body_com_pos_w")
    if path_marker is not None:
        path_marker.visualize(candidate["target_position_world_m"][::2])

    for step in range(maximum_steps):
        elapsed_s = step / POLICY_HZ
        (
            desired_state,
            position_target,
            attitude_target,
            vx_feedforward,
            wz_feedforward,
        ) = interpolate(candidate, phase_time_s)
        root_position = unwrapped.robot.data.root_pos_w[0].detach().cpu().numpy()
        root_quaternion = unwrapped.robot.data.root_quat_w[0].detach().cpu().numpy()
        actual_base = np.array(
            [root_position[0], root_position[1], yaw_from_quaternion_wxyz(root_quaternion)]
        )
        actual_arm = (
            unwrapped.robot.data.joint_pos[0, arm_ids].detach().cpu().numpy()
        )
        actual_gimbal_position = (
            unwrapped.robot.data.joint_pos[0, gimbal_ids]
            .detach()
            .cpu()
            .numpy()
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
        root_tilt_displacement = (
            actual_tool_position
            - kinematics.position(np.concatenate((actual_base, actual_arm)))
        )
        peak_root_tilt_displacement_m = max(
            peak_root_tilt_displacement_m,
            float(np.linalg.norm(root_tilt_displacement)),
        )
        scaled_vx_feedforward, scaled_wz_feedforward = phase_scaled_feedforward(
            vx_feedforward, wz_feedforward, progress_scale
        )
        if args.open_loop:
            vx_ref = scaled_vx_feedforward
            wz_ref = scaled_wz_feedforward
            arm_target = desired_state[3:]
            ik_correction = np.zeros(3)
            task_space_base_target = desired_state[:3].copy()
            task_space_base_offset = np.zeros(2, dtype=np.float64)
            base_target_diagnostics = {
                "requested_offset_world_m": task_space_base_offset,
                "bounded_offset_world_m": task_space_base_offset,
                "retarget_residual_world_m": np.zeros(2, dtype=np.float64),
                "offset_saturated": 0.0,
            }
        else:
            requested_base_target, base_target_diagnostics = (
                bounded_task_space_base_target(
                    kinematics,
                    desired_state[:3],
                    desired_state[3:],
                    position_target,
                    root_tilt_displacement,
                    args.maximum_task_space_base_offset_m,
                )
            )
            requested_base_offset = requested_base_target[:2] - desired_state[:2]
            task_space_base_offset = slew_limited_planar_offset(
                requested_base_offset,
                previous_task_space_base_offset,
                1.0 / POLICY_HZ,
                args.maximum_task_space_base_offset_rate_mps,
            )
            task_space_base_target = desired_state[:3].copy()
            task_space_base_target[:2] += task_space_base_offset
            vx_ref, wz_ref, _ = bounded_base_references(
                task_space_base_target,
                actual_base,
                scaled_vx_feedforward,
                scaled_wz_feedforward,
                tracking_config,
            )
            arm_target, ik_correction = bounded_semantic_arm_target(
                kinematics,
                actual_base,
                actual_arm,
                desired_state[3:],
                previous_arm_target,
                position_target,
                actual_tool_position,
                dt=1.0 / POLICY_HZ,
                semantic_feedback_enabled=(
                    args.enable_acquisition_task_space_arm_feedback
                    or phase_time_s >= semantic_start_time_s
                ),
                config=tracking_config,
            )
        previous_task_space_base_offset = task_space_base_offset.copy()
        peak_task_space_base_offset_m = max(
            peak_task_space_base_offset_m,
            float(np.linalg.norm(task_space_base_offset)),
        )
        previous_arm_target = arm_target.copy()
        predicted_root_pitch = kinematics.equilibrium_pitch_rad(
            desired_state, WHEEL_RADIUS_M
        )
        nominal_gimbal_root_quaternion = quaternion_from_pitch_yaw_wxyz(
            predicted_root_pitch, float(desired_state[2])
        )
        nominal_gimbal_ik = camera_kinematics.solve_semantic_attitude_continuous(
            nominal_gimbal_root_quaternion,
            desired_state[3:],
            attitude_target,
            previous_nominal_gimbal_target,
        )
        previous_nominal_gimbal_target = nominal_gimbal_ik.gimbal_q.copy()
        camera_feedback = camera_kinematics.bounded_attitude_feedback_target(
            root_quaternion,
            actual_arm,
            actual_gimbal_position,
            attitude_target,
            nominal_gimbal_ik.gimbal_q,
            previous_camera_feedback_correction,
            1.0 / POLICY_HZ,
            gain=args.camera_attitude_feedback_gain,
            damping=args.camera_attitude_feedback_damping,
            maximum_correction_rad=math.radians(
                args.maximum_gimbal_feedback_joint_offset_deg
            ),
            time_constant_s=args.camera_attitude_feedback_time_constant_s,
        )
        previous_camera_feedback_correction = camera_feedback.correction_q.copy()
        peak_gimbal_ik_residual_deg = max(
            peak_gimbal_ik_residual_deg,
            math.degrees(nominal_gimbal_ik.orientation_error_rad),
        )
        gimbal_ik_nonconverged_steps += int(not nominal_gimbal_ik.converged)
        gimbal_feedback_joint_offset_deg = math.degrees(
            float(np.max(np.abs(camera_feedback.correction_q)))
        )
        peak_gimbal_feedback_joint_offset_deg = max(
            peak_gimbal_feedback_joint_offset_deg,
            gimbal_feedback_joint_offset_deg,
        )
        maximum_gimbal_delta = 0.5 / POLICY_HZ
        requested_gimbal_delta = (
            camera_feedback.gimbal_q - previous_gimbal_target
        )
        limited_gimbal_delta = np.clip(
            requested_gimbal_delta,
            -maximum_gimbal_delta,
            maximum_gimbal_delta,
        )
        gimbal_target_saturation_steps += int(
            np.any(
                np.abs(requested_gimbal_delta)
                > maximum_gimbal_delta + 1e-12
            )
        )
        previous_gimbal_target = previous_gimbal_target + limited_gimbal_delta
        if target_marker is not None:
            target_marker.visualize(position_target[None, :])
        arm_target_tensor = torch.as_tensor(
            arm_target[None, :], dtype=torch.float32, device=unwrapped.device
        )
        unwrapped.robot.set_joint_position_target(arm_target_tensor, joint_ids=arm_ids)
        gimbal_target_tensor = torch.as_tensor(
            previous_gimbal_target[None, :],
            dtype=torch.float32,
            device=unwrapped.device,
        )
        unwrapped.robot.set_joint_position_target(
            gimbal_target_tensor, joint_ids=gimbal_ids
        )
        arm_gravity_feedforward = np.zeros(3, dtype=np.float64)
        if not args.disable_arm_gravity_feedforward:
            arm_gravity_feedforward = kinematics.gravitational_effort_nm(
                np.concatenate((actual_base, actual_arm)),
                root_roll_pitch_rad=roll_pitch_from_quaternion_wxyz(
                    root_quaternion
                ),
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
        gimbal_gravity_feedforward = np.zeros(3, dtype=np.float64)
        if not args.disable_gimbal_gravity_feedforward:
            gimbal_gravity_feedforward = (
                camera_kinematics.gimbal_gravitational_effort_nm(
                    root_quaternion,
                    actual_arm,
                    actual_gimbal_position,
                )
            )
            unwrapped.robot.set_joint_effort_target(
                torch.as_tensor(
                    gimbal_gravity_feedforward[None, :],
                    dtype=torch.float32,
                    device=unwrapped.device,
                ),
                joint_ids=gimbal_ids,
            )
        peak_gimbal_gravity_feedforward_nm = max(
            peak_gimbal_gravity_feedforward_nm,
            float(np.max(np.abs(gimbal_gravity_feedforward))),
        )
        unwrapped.vx_ref.fill_(vx_ref)
        unwrapped.wz_ref.fill_(wz_ref)
        if step % control_interval == 0:
            action, controller_state, controller_diagnostics = cascaded_lqr_action(
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
        peak_controller_pitch_target_deg = max(
            peak_controller_pitch_target_deg,
            math.degrees(
                abs(float(controller_diagnostics["pitch_target"][0]))
            ),
        )
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
        actual_cam_quaternion = (
            unwrapped.robot.data.body_quat_w[0, cam_ids[0]]
            .detach()
            .cpu()
            .numpy()
        )
        expected_cam_quaternion = semantic_dfr_to_physical_cam_quat_wxyz(
            attitude_target
        )
        attitude_error_deg = math.degrees(
            float(
                np.linalg.norm(
                    rotation_error_vector(
                        quaternion_matrix_wxyz(actual_cam_quaternion),
                        quaternion_matrix_wxyz(expected_cam_quaternion),
                    )
                )
            )
        )
        attitude_errors_deg.append(attitude_error_deg)
        actual_arm = unwrapped.robot.data.joint_pos[0, arm_ids]
        arm_error_by_joint = torch.abs(actual_arm - arm_target_tensor[0])
        arm_error = arm_error_by_joint.max().item()
        nominal_arm = torch.as_tensor(
            desired_state[3:], dtype=torch.float32, device=unwrapped.device
        )
        nominal_arm_error_by_joint = torch.abs(actual_arm - nominal_arm)
        nominal_arm_error = nominal_arm_error_by_joint.max().item()
        actual_gimbal = unwrapped.robot.data.joint_pos[0, gimbal_ids]
        gimbal_error_by_joint = torch.abs(
            actual_gimbal - gimbal_target_tensor[0]
        )
        gimbal_error = gimbal_error_by_joint.max().item()
        gimbal_servo_errors.append(
            gimbal_error_by_joint.detach().cpu().numpy()
        )
        peak_gimbal_error_deg = max(
            peak_gimbal_error_deg, math.degrees(gimbal_error)
        )
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
            np.linalg.norm(current_root_position - task_space_base_target[:2])
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
            gimbal_effort = unwrapped.robot.data.applied_torque[0, gimbal_ids].abs()
            peak_gimbal_effort_nm = max(
                peak_gimbal_effort_nm, gimbal_effort.max().item()
            )
            saturated_arm_efforts += int(
                torch.count_nonzero(
                    arm_effort >= args.arm_effort_limit_nm - 1e-3
                ).item()
            )
            arm_effort_count += len(arm_ids)
            saturated_gimbal_efforts += int(
                torch.count_nonzero(
                    gimbal_effort >= args.gimbal_effort_limit_nm - 1e-3
                ).item()
            )
            gimbal_effort_count += len(gimbal_ids)
        if position_error >= max(position_errors):
            peak_position_error_step = step + 1
        if args.enable_phase_governor:
            acquisition_phase = phase_time_s < semantic_start_time_s
            tracking_progress_scale = bounded_phase_progress_scale(
                base_xy_error,
                position_error,
                acquisition_phase,
                tracking_config,
            )
            balance_progress_scale = bounded_balance_progress_scale(
                float(state["pitch"][0].item()),
                math.radians(args.phase_governor_pitch_start_deg),
                math.radians(args.phase_governor_pitch_stop_deg),
            )
            attitude_progress_scale = bounded_attitude_progress_scale(
                math.radians(attitude_error_deg),
                math.radians(args.phase_governor_attitude_start_deg),
                math.radians(args.maximum_attitude_p95_deg),
            )
            progress_scale = min(
                tracking_progress_scale,
                balance_progress_scale,
                attitude_progress_scale,
            )
        else:
            tracking_progress_scale = 1.0
            balance_progress_scale = 1.0
            attitude_progress_scale = 1.0
            progress_scale = 1.0
        progress_scale_min = min(progress_scale_min, progress_scale)
        balance_progress_scale_min = min(
            balance_progress_scale_min, balance_progress_scale
        )
        attitude_progress_scale_min = min(
            attitude_progress_scale_min, attitude_progress_scale
        )
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
                    "reference_phase": (
                        "acquisition"
                        if phase_time_s < semantic_start_time_s
                        else "semantic"
                    ),
                    "progress_scale": progress_scale,
                    "tracking_progress_scale": tracking_progress_scale,
                    "base_error_in_progress_governor": (
                        phase_time_s >= semantic_start_time_s
                    ),
                    "balance_progress_scale": balance_progress_scale,
                    "attitude_progress_scale": attitude_progress_scale,
                    "com_pitch_bias_deg": math.degrees(com_pitch_bias),
                    "applied_com_pitch_bias_deg": math.degrees(
                        float(controller_diagnostics["applied_pitch_bias"][0])
                    ),
                    "velocity_pitch_reference_deg": math.degrees(
                        float(controller_diagnostics["pitch_reference"][0])
                    ),
                    "controller_pitch_target_deg": math.degrees(
                        float(controller_diagnostics["pitch_target"][0])
                    ),
                    "root_tilt_displacement_m": root_tilt_displacement.tolist(),
                    "root_roll_pitch_deg": np.degrees(
                        roll_pitch_from_quaternion_wxyz(current_root_quaternion)
                    ).tolist(),
                    "predicted_equilibrium_pitch_deg": math.degrees(
                        predicted_root_pitch
                    ),
                    "attitude_error_deg": attitude_error_deg,
                    "gimbal_ik_residual_deg": math.degrees(
                        nominal_gimbal_ik.orientation_error_rad
                    ),
                    "camera_feedback_model_error_deg": math.degrees(
                        camera_feedback.orientation_error_rad
                    ),
                    "nominal_gimbal_target_q": (
                        nominal_gimbal_ik.gimbal_q.tolist()
                    ),
                    "camera_feedback_correction_q": (
                        camera_feedback.correction_q.tolist()
                    ),
                    "feedback_gimbal_target_q": (
                        camera_feedback.gimbal_q.tolist()
                    ),
                    "commanded_gimbal_target_q": previous_gimbal_target.tolist(),
                    "actual_gimbal_q": actual_gimbal.detach().cpu().numpy().tolist(),
                    "gimbal_feedback_joint_offset_deg": (
                        gimbal_feedback_joint_offset_deg
                    ),
                    "task_space_base_offset_m": task_space_base_offset.tolist(),
                    "task_space_base_offset_requested_m": base_target_diagnostics[
                        "requested_offset_world_m"
                    ].tolist(),
                    "position_error_m": position_error,
                    "pitch_deg": math.degrees(abs(float(state["pitch"][0].item()))),
                    "desired_base_q": desired_state[:3].tolist(),
                    "control_base_q": task_space_base_target.tolist(),
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
                    "gimbal_gravity_feedforward_nm": (
                        gimbal_gravity_feedforward.tolist()
                    ),
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
    attitude_errors_deg = np.asarray(attitude_errors_deg)
    arm_servo_errors = np.asarray(arm_servo_errors)
    nominal_arm_errors = np.asarray(nominal_arm_errors)
    gimbal_servo_errors = np.asarray(gimbal_servo_errors)
    saturation_ratio = saturated_actions / max(action_count, 1)
    arm_effort_saturation_ratio = saturated_arm_efforts / max(arm_effort_count, 1)
    gimbal_effort_saturation_ratio = saturated_gimbal_efforts / max(
        gimbal_effort_count, 1
    )
    gimbal_target_saturation_ratio = (
        gimbal_target_saturation_steps / max(completed_steps, 1)
    )
    checks = {
        "completed_acquisition": phase_time_s >= semantic_start_time_s,
        "completed_reference": phase_time_s >= source_duration_s,
        "no_termination": termination is None,
        "peak_pitch_below_limit": peak_pitch_deg <= args.maximum_pitch_deg,
        "peak_arm_error_below_limit": peak_arm_error_deg <= args.maximum_arm_error_deg,
        "position_p95_below_limit": float(np.percentile(errors, 95))
        <= args.maximum_position_p95_m,
        "position_max_below_limit": float(np.max(errors))
        <= args.maximum_position_error_m,
        "attitude_p95_below_limit": float(np.percentile(attitude_errors_deg, 95))
        <= args.maximum_attitude_p95_deg,
        "attitude_max_below_limit": float(np.max(attitude_errors_deg))
        <= args.maximum_attitude_error_deg,
        "all_gimbal_ik_steps_converged": gimbal_ik_nonconverged_steps == 0,
        "peak_gimbal_error_below_limit": peak_gimbal_error_deg
        <= args.maximum_gimbal_error_deg,
        "gimbal_target_saturation_below_limit": gimbal_target_saturation_ratio
        <= args.maximum_gimbal_target_saturation_ratio,
        "action_saturation_below_limit": saturation_ratio
        <= args.maximum_action_saturation_ratio,
        "arm_effort_saturation_below_limit": arm_effort_saturation_ratio
        <= args.maximum_arm_effort_saturation_ratio,
        "gimbal_effort_saturation_below_limit": gimbal_effort_saturation_ratio
        <= args.maximum_gimbal_effort_saturation_ratio,
    }
    return {
        "case": case,
        "source_duration_s": source_duration_s,
        "semantic_start_time_s": semantic_start_time_s,
        "maximum_steps": maximum_steps,
        "completed_steps": completed_steps,
        "wall_duration_s": completed_steps / POLICY_HZ,
        "completed_phase_time_s": phase_time_s,
        "progress_scale_min": progress_scale_min,
        "balance_progress_scale_min": balance_progress_scale_min,
        "attitude_progress_scale_min": attitude_progress_scale_min,
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
        "peak_gimbal_error_deg": peak_gimbal_error_deg,
        "gimbal_error_p95_by_joint_deg": {
            name: math.degrees(
                float(np.percentile(gimbal_servo_errors[:, index], 95))
            )
            for index, name in enumerate(PHYSICAL_GIMBAL_JOINTS)
        },
        "peak_gimbal_effort_nm": peak_gimbal_effort_nm,
        "peak_gimbal_ik_residual_deg": peak_gimbal_ik_residual_deg,
        "peak_gimbal_feedback_joint_offset_deg": (
            peak_gimbal_feedback_joint_offset_deg
        ),
        "gimbal_ik_nonconverged_steps": gimbal_ik_nonconverged_steps,
        "gimbal_target_saturation_ratio": gimbal_target_saturation_ratio,
        "peak_arm_gravity_feedforward_nm": peak_arm_gravity_feedforward_nm,
        "peak_gimbal_gravity_feedforward_nm": (
            peak_gimbal_gravity_feedforward_nm
        ),
        "peak_base_xy_error_m": peak_base_xy_error_m,
        "peak_base_yaw_error_deg": peak_base_yaw_error_deg,
        "peak_ik_correction_deg": peak_ik_correction_deg,
        "peak_root_tilt_displacement_m": peak_root_tilt_displacement_m,
        "peak_task_space_base_offset_m": peak_task_space_base_offset_m,
        "peak_com_pitch_bias_deg": peak_com_pitch_bias_deg,
        "peak_controller_pitch_target_deg": peak_controller_pitch_target_deg,
        "position_error_mean_m": float(np.mean(errors)),
        "position_error_p95_m": float(np.percentile(errors, 95)),
        "position_error_max_m": float(np.max(errors)),
        "attitude_error_mean_deg": float(np.mean(attitude_errors_deg)),
        "attitude_error_p95_deg": float(np.percentile(attitude_errors_deg, 95)),
        "attitude_error_max_deg": float(np.max(attitude_errors_deg)),
        "position_error_max_elapsed_s": (peak_position_error_step - 1) / POLICY_HZ,
        "vx_rmse_mps": float(np.sqrt(np.mean(np.square(vx_errors)))),
        "wz_rmse_radps": float(np.sqrt(np.mean(np.square(wz_errors)))),
        "action_saturation_ratio": saturation_ratio,
        "arm_effort_saturation_ratio": arm_effort_saturation_ratio,
        "gimbal_effort_saturation_ratio": gimbal_effort_saturation_ratio,
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
    if not (
        0.0
        < args.maximum_com_pitch_bias_deg
        <= args.maximum_controller_pitch_target_deg
        <= args.maximum_pitch_deg
    ):
        raise ValueError(
            "pitch limits must satisfy 0 < COM bias <= controller target "
            "<= physical gate"
        )
    if not (
        0.0 <= args.phase_governor_pitch_start_deg
        < args.phase_governor_pitch_stop_deg
        < args.maximum_pitch_deg
    ):
        raise ValueError(
            "phase-governor pitch thresholds must satisfy "
            "0 <= start < stop < physical gate"
        )
    if not (
        0.0 <= args.phase_governor_attitude_start_deg
        < args.maximum_attitude_p95_deg
    ):
        raise ValueError(
            "phase-governor attitude start must be below the attitude p95 gate"
        )
    if not (
        0.0 < WholeBodyTrackingConfig().progress_error_start_m
        < args.maximum_position_p95_m
    ):
        raise ValueError(
            "position p95 gate must exceed the phase-governor slowdown start"
        )
    if args.maximum_task_space_base_offset_m <= 0.0:
        raise ValueError("maximum task-space base offset must be positive")
    if args.maximum_task_space_base_offset_rate_mps <= 0.0:
        raise ValueError("maximum task-space base offset rate must be positive")
    if not 0.0 <= args.camera_attitude_feedback_gain <= 1.0:
        raise ValueError("camera attitude feedback gain must be in [0, 1]")
    if args.camera_attitude_feedback_damping <= 0.0:
        raise ValueError("camera attitude feedback damping must be positive")
    if args.camera_attitude_feedback_time_constant_s < 0.0:
        raise ValueError("camera attitude feedback time constant must be non-negative")
    if args.maximum_gimbal_feedback_joint_offset_deg <= 0.0:
        raise ValueError("maximum gimbal feedback joint offset must be positive")
    if args.arm_effort_limit_nm <= 0.0 or args.gimbal_effort_limit_nm <= 0.0:
        raise ValueError("actuator effort limits must be positive")
    for name, value in (
        ("arm effort saturation", args.maximum_arm_effort_saturation_ratio),
        ("gimbal effort saturation", args.maximum_gimbal_effort_saturation_ratio),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"maximum {name} ratio must be in [0, 1]")
    candidates = {case: load_candidate(case) for case in cases}
    kinematics = UrdfPositionKinematics(args.urdf)
    camera_kinematics = UrdfPhysicalCameraKinematics(args.urdf)
    gain_data = json.loads(args.gains.read_text(encoding="utf-8"))
    gain = np.asarray(gain_data["selected_gain"], dtype=np.float64)
    control_interval = int(gain_data["control_interval_steps"])
    if gain.shape != (len(ACTION_NAMES), len(LQR_STATE_NAMES)):
        raise ValueError(f"invalid gain shape: {gain.shape}")

    register_isaac_lab_tasks()
    cfg = RecomoTwoWheelBalanceEnvCfg()
    cfg.seed = 20260714
    cfg.scene.num_envs = 1
    cfg.robot_cfg = copy.deepcopy(TWO_WHEEL_WHOLE_BODY_ATTITUDE_CFG)
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
            camera_kinematics,
            target_marker,
            path_marker,
        )
        for case in cases
    ]
    env.close()
    result = {
        "schema": "recomo_two_wheel_corrected_full_pose_playback_smoke_v3",
        "training_started": False,
        "controller_profile": "structural_robust_v1",
        "task_space_feedback_enabled": not args.open_loop,
        "acquisition_task_space_feedback_enabled": not args.open_loop,
        "acquisition_chassis_lag_feedback_enabled": False,
        "acquisition_root_tilt_compensation_enabled": False,
        "task_space_base_compensation_enabled": not args.open_loop,
        "task_space_arm_feedback_enabled": not args.open_loop,
        "task_space_arm_feedback_phase": (
            "acquisition_and_semantic"
            if args.enable_acquisition_task_space_arm_feedback
            else "semantic_only"
        ),
        "acquisition_task_space_arm_feedback_enabled": (
            args.enable_acquisition_task_space_arm_feedback
        ),
        "semantic_task_space_feedback_enabled": not args.open_loop,
        "phase_governor_enabled": args.enable_phase_governor,
        "acquisition_progress_contract": (
            "ee1_tool_position_plus_cam_link_attitude_plus_balance"
        ),
        "semantic_progress_includes_base_error": True,
        "phase_governor_pitch_start_deg": args.phase_governor_pitch_start_deg,
        "phase_governor_pitch_stop_deg": args.phase_governor_pitch_stop_deg,
        "phase_governor_position_stop_m": args.maximum_position_p95_m,
        "phase_governor_attitude_start_deg": (
            args.phase_governor_attitude_start_deg
        ),
        "phase_governor_attitude_stop_deg": args.maximum_attitude_p95_deg,
        "phase_feedforward_scaled_by_progress": True,
        "com_pitch_feedforward_enabled": not args.disable_com_pitch_feedforward,
        "arm_gravity_feedforward_enabled": not args.disable_arm_gravity_feedforward,
        "gimbal_gravity_feedforward_enabled": (
            not args.disable_gimbal_gravity_feedforward
        ),
        "arm_stiffness": args.arm_stiffness,
        "arm_damping": args.arm_damping,
        "arm_effort_limit_nm": args.arm_effort_limit_nm,
        "gimbal_effort_limit_nm": args.gimbal_effort_limit_nm,
        "maximum_task_space_base_offset_m": args.maximum_task_space_base_offset_m,
        "maximum_task_space_base_offset_rate_mps": (
            args.maximum_task_space_base_offset_rate_mps
        ),
        "maximum_com_pitch_bias_deg": args.maximum_com_pitch_bias_deg,
        "maximum_controller_pitch_target_deg": (
            args.maximum_controller_pitch_target_deg
        ),
        "position_target_link": "ee1_tool",
        "camera_attitude_tracking_enabled": True,
        "camera_observation_and_reward_link": "cam_link",
        "camera_frame_conversion": "R_world_cam = R_world_DFR * Rz(+pi/2)",
        "physical_gimbal_control_enabled": True,
        "physical_gimbal_command_state_model": (
            "nominal_predicted_root_branch_plus_bounded_differential_cam_feedback"
        ),
        "camera_attitude_feedback_gain": args.camera_attitude_feedback_gain,
        "camera_attitude_feedback_damping": (
            args.camera_attitude_feedback_damping
        ),
        "camera_attitude_feedback_time_constant_s": (
            args.camera_attitude_feedback_time_constant_s
        ),
        "maximum_gimbal_feedback_joint_offset_deg": (
            args.maximum_gimbal_feedback_joint_offset_deg
        ),
        "physical_gimbal_joint_labels_learned": False,
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
