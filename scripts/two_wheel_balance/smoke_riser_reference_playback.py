#!/usr/bin/env python3
"""Replay corrected riser plans with balance control and optional RTX video."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import re
import sys
import threading

import numpy as np

os.environ.setdefault("ACCEPT_EULA", "YES")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")
os.environ.setdefault("GYMNASIUM_DISABLE_PLUGIN_ENTRYPOINTS", "1")

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--gains", type=Path, required=True)
parser.add_argument("--plan-dir", type=Path, required=True)
parser.add_argument(
    "--plan-filename-template",
    default="case_{case:04d}_riser_playback_v1.npz",
    help="Plan basename template; must contain exactly one integer {case} field.",
)
parser.add_argument("--cases", default="1,31,73")
parser.add_argument("--maximum-duration-scale", type=float, default=2.0)
parser.add_argument("--maximum-pitch-deg", type=float, default=12.0)
parser.add_argument("--maximum-position-p95-m", type=float, default=0.15)
parser.add_argument("--maximum-position-error-m", type=float, default=0.25)
parser.add_argument("--maximum-attitude-p95-deg", type=float, default=5.0)
parser.add_argument("--maximum-attitude-error-deg", type=float, default=10.0)
parser.add_argument("--maximum-riser-servo-error-m", type=float, default=0.03)
parser.add_argument("--maximum-proxy-servo-error-deg", type=float, default=5.0)
parser.add_argument("--maximum-internal-proxy-rate-deg-s", type=float, default=360.0)
parser.add_argument("--maximum-saturation-ratio", type=float, default=0.20)
parser.add_argument("--disable-phase-governor", action="store_true")
parser.add_argument("--disable-com-pitch-feedforward", action="store_true")
parser.add_argument("--disable-semantic-proxy-state-adapter", action="store_true")
parser.add_argument("--controller-wz-kp", type=float)
parser.add_argument("--controller-wz-ki", type=float)
parser.add_argument("--controller-wz-feedforward", type=float)
parser.add_argument("--controller-wheel-difference-kp", type=float)
parser.add_argument("--tracking-along-kp", type=float)
parser.add_argument("--tracking-cross-kp", type=float)
parser.add_argument("--tracking-yaw-kp", type=float)
parser.add_argument("--video-dir", type=Path)
parser.add_argument(
    "--dataset-dir",
    type=Path,
    help="Write dense pre-action executed-state residual datasets per case.",
)
parser.add_argument(
    "--residual-policy",
    type=Path,
    help="Optional gated TorchScript high-level residual policy.",
)
parser.add_argument(
    "--residual-policy-device",
    choices=("cpu", "cuda"),
    default="cuda",
)
parser.add_argument(
    "--zero-policy-action",
    action="store_true",
    help="Evaluate the null high-level policy action above the frozen balance LQR.",
)
# RecordVideo receives one frame per 200 Hz policy step.  Encoding at the same
# rate preserves simulation time; viewing derivatives may be transcoded to 50.
parser.add_argument("--video-fps", type=int, default=200)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
for name in ("tracking_along_kp", "tracking_cross_kp", "tracking_yaw_kp"):
    value = getattr(args, name)
    if value is not None and value < 0.0:
        parser.error(f"--{name.replace('_', '-')} must be non-negative")
app = AppLauncher(args).app

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import torch

from isaaclab import sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from rl_platform.robots.two_wheel_balance import TWO_WHEEL_RISER_CFG
from rl_platform.tasks.two_wheel_balance import RecomoTwoWheelBalanceEnvCfg
from rl_platform.tasks.two_wheel_balance.camera_attitude import (
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
from rl_platform.tasks.two_wheel_balance.riser_playback import (
    RiserPlaybackPlan,
    interpolate_riser_playback_plan,
    load_riser_playback_plan,
    phase_scaled_feedforward,
)
from rl_platform.tasks.two_wheel_balance.riser_kinematics import (
    UrdfRiserCameraKinematics,
)
from rl_platform.tasks.two_wheel_balance.riser_control import balance_progress_scale
from rl_platform.tasks.two_wheel_balance.riser_residual_dataset import (
    LOOKAHEAD_HORIZONS_S,
    apply_residual_action,
    build_raw_residual_command,
    build_executed_observation,
    build_residual_action,
    normalize_residual_command,
    residual_action_envelope_passed,
    save_case_dataset,
)
from rl_platform.tasks.two_wheel_balance.whole_body_tracking import (
    bounded_base_references,
    bounded_progress_scale,
    continuous_joint_error,
    equilibrium_pitch_from_world_com,
    nearest_equivalent_angle,
    riser_tracking_config,
    yaw_from_quaternion_wxyz,
)
from task_spec import register_isaac_lab_tasks


POLICY_HZ = 200.0
WHEEL_RADIUS_M = 0.1016
PROXY_JOINTS = (
    "joint3_gimbal_yaw",
    "joint2_gimbal_roll",
    "joint1_gimbal_pitch",
)


def parse_cases(value: str) -> list[int]:
    cases = [int(item) for item in value.split(",") if item.strip()]
    if not cases or len(cases) != len(set(cases)):
        raise ValueError("cases must be a non-empty unique list")
    return cases


def plan_path(plan_dir: Path, template: str, case: int) -> Path:
    try:
        name = template.format(case=case)
    except (KeyError, ValueError) as exc:
        raise ValueError("invalid plan filename template") from exc
    if Path(name).name != name or "{case" not in template:
        raise ValueError("plan filename template must be a basename with a {case} field")
    return plan_dir / name


def single_joint_id(robot, name: str) -> int:
    ids = robot.find_joints(name)[0]
    if len(ids) != 1:
        raise RuntimeError(f"expected one joint named {name}, got {ids}")
    return ids[0]


def current_lqr_state(unwrapped) -> np.ndarray:
    state = unwrapped._state_terms()
    return np.column_stack(
        [state[name].detach().cpu().numpy() for name in LQR_STATE_NAMES]
    )


def initialize_case(env, plan: RiserPlaybackPlan) -> tuple[dict[str, torch.Tensor], list[int], int]:
    obs, _ = env.reset(seed=20260716 + plan.case)
    unwrapped = env.unwrapped
    robot = unwrapped.robot
    env_ids = torch.zeros(1, dtype=torch.long, device=unwrapped.device)
    riser_id = single_joint_id(robot, "riser_joint")
    proxy_ids = [single_joint_id(robot, name) for name in PROXY_JOINTS]

    root_state = robot.data.default_root_state[env_ids].clone()
    root_state[:, :3] += unwrapped.scene.env_origins[env_ids]
    root_state[0, 0] = plan.base_xy_yaw[0, 0]
    root_state[0, 1] = plan.base_xy_yaw[0, 1]
    half_yaw = 0.5 * plan.base_xy_yaw[0, 2]
    root_state[0, 3] = math.cos(half_yaw)
    root_state[0, 4] = 0.0
    root_state[0, 5] = 0.0
    root_state[0, 6] = math.sin(half_yaw)
    root_state[0, 7:] = 0.0
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = robot.data.default_joint_vel[env_ids].clone()
    joint_pos[0, riser_id] = plan.riser_q[0]
    joint_pos[0, proxy_ids] = torch.as_tensor(
        plan.proxy_gimbal_q[0], dtype=torch.float32, device=unwrapped.device
    )
    joint_vel.zero_()
    robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
    robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
    unwrapped.actions.zero_()
    unwrapped.policy_actions.zero_()
    unwrapped.previous_actions.zero_()
    unwrapped.wheel_efforts.zero_()
    unwrapped.episode_length_buf.zero_()
    return obs, proxy_ids, riser_id


def evaluate_case(
    env,
    plan: RiserPlaybackPlan,
    gain: np.ndarray,
    control_interval: int,
    target_marker: VisualizationMarkers | None,
    path_marker: VisualizationMarkers | None,
    dataset_dir: Path | None,
    residual_policy,
    residual_policy_device: torch.device,
    zero_policy_action: bool,
) -> dict[str, object]:
    _, proxy_ids, riser_id = initialize_case(env, plan)
    unwrapped = env.unwrapped
    robot = unwrapped.robot
    env_ids = torch.zeros(1, dtype=torch.long, device=unwrapped.device)
    cam_ids = robot.find_bodies("cam_link")[0]
    if len(cam_ids) != 1:
        raise RuntimeError(f"expected one physical cam_link, got {cam_ids}")
    cam_id = cam_ids[0]
    if path_marker is not None:
        path_marker.visualize(plan.target_position_world_m[::2])

    controller_state = np.zeros((1, 6), dtype=np.float64)
    current_states = current_lqr_state(unwrapped)
    action = np.zeros((1, len(ACTION_NAMES)), dtype=np.float32)
    controller_overrides = {
        name: value
        for name, value in {
            "wz_kp": args.controller_wz_kp,
            "wz_ki": args.controller_wz_ki,
            "wz_feedforward": args.controller_wz_feedforward,
            "wheel_difference_kp": args.controller_wheel_difference_kp,
        }.items()
        if value is not None
    }
    controller_cfg = cascaded_lqr_config(
        "structural_robust_v1", **controller_overrides
    )
    tracking_overrides = {
        name: value
        for name, value in {
            "along_track_kp": args.tracking_along_kp,
            "cross_track_kp": args.tracking_cross_kp,
            "yaw_kp": args.tracking_yaw_kp,
        }.items()
        if value is not None
    }
    tracking_cfg = riser_tracking_config(**tracking_overrides)
    execution_duration_s = float(plan.time_s[-1])
    source_duration_s = float(plan.source_time_s[-1])
    maximum_steps = int(
        math.ceil(execution_duration_s * args.maximum_duration_scale * POLICY_HZ)
    ) + 1
    phase_time_s = 0.0
    progress_scale = 1.0
    progress_samples = []
    position_errors = []
    attitude_errors_deg = []
    riser_errors = []
    proxy_errors_deg = []
    pitch_samples_deg = []
    saturated_actions = 0
    action_count = 0
    saturated_riser = 0
    riser_effort_count = 0
    saturated_proxy = 0
    proxy_effort_count = 0
    proxy_axis_saturated = np.zeros(len(PROXY_JOINTS), dtype=np.int64)
    proxy_velocity_samples_deg_s = []
    proxy_target_velocity_samples_deg_s = []
    proxy_effort_samples_nm = []
    latest_proxy_effort_nm = np.full(len(PROXY_JOINTS), np.nan, dtype=np.float64)
    peak_base_xy_error_m = 0.0
    peak_base_yaw_error_deg = 0.0
    peak_com_pitch_bias_deg = 0.0
    internal_attitude_ik_failures = 0
    internal_attitude_ik_error_max_deg = 0.0
    internal_proxy_rate_max_deg_s = 0.0
    termination = None
    trace = []
    dataset_observations = []
    dataset_actions = []
    dataset_elapsed_time = []
    dataset_phase_time = []
    dataset_baseline_actions = []
    dataset_teacher_commands = []
    applied_residual_actions = []
    raw_residual_commands = []
    normalized_residual_labels = []
    previous_residual_action = np.zeros(3, dtype=np.float32)
    completed_steps = 0
    body_masses = robot.data.default_mass[0].to(unwrapped.device)
    kinematics = UrdfRiserCameraKinematics(
        PROJECT_ROOT
        / "assets_own/recomoProto2_two_wheel_riser/recomoProto2_two_wheel_riser.urdf"
    )
    previous_proxy_command = plan.proxy_gimbal_q[0].copy()
    if not hasattr(robot.data, "body_com_pos_w"):
        raise RuntimeError("Isaac articulation data does not expose body_com_pos_w")

    for step in range(maximum_steps):
        elapsed_s = step / POLICY_HZ
        sample = interpolate_riser_playback_plan(plan, phase_time_s)
        (
            phase_feedforward_v_mps,
            phase_feedforward_wz_rad_s,
            phase_feedforward_riser_velocity_mps,
            phase_feedforward_proxy_velocity_rad_s,
        ) = phase_scaled_feedforward(sample, progress_scale)
        root_position = robot.data.root_pos_w[0].detach().cpu().numpy()
        root_quaternion = robot.data.root_quat_w[0].detach().cpu().numpy()
        actual_base = np.array(
            [
                root_position[0],
                root_position[1],
                yaw_from_quaternion_wxyz(root_quaternion),
            ]
        )
        actual_camera_position_pre = (
            robot.data.body_pos_w[0, cam_id].detach().cpu().numpy()
        )
        actual_camera_quaternion_pre = (
            robot.data.body_quat_w[0, cam_id].detach().cpu().numpy()
        )
        vx_ref, wz_ref, base_tracking_diagnostics = bounded_base_references(
            sample.base_xy_yaw,
            actual_base,
            phase_feedforward_v_mps,
            phase_feedforward_wz_rad_s,
            tracking_cfg,
        )
        actual_riser_pre = float(robot.data.joint_pos[0, riser_id].item())
        actual_riser_velocity_pre = float(
            robot.data.joint_vel[0, riser_id].item()
        )
        target_physical_camera_quaternion = (
            semantic_dfr_to_physical_cam_quat_wxyz(
                sample.target_semantic_dfr_quat_wxyz
            )
        )
        raw_residual_command = build_raw_residual_command(
            feedforward_vx_m_s=phase_feedforward_v_mps,
            feedforward_wz_rad_s=phase_feedforward_wz_rad_s,
            commanded_vx_m_s=vx_ref,
            commanded_wz_rad_s=wz_ref,
            actual_riser_position_m=actual_riser_pre,
            target_riser_position_m=sample.riser_q,
        )
        normalized_residual_label = normalize_residual_command(raw_residual_command)
        raw_residual_commands.append(raw_residual_command.copy())
        normalized_residual_labels.append(normalized_residual_label.copy())
        if dataset_dir is not None:
            teacher_residual_action = build_residual_action(
                feedforward_vx_m_s=phase_feedforward_v_mps,
                feedforward_wz_rad_s=phase_feedforward_wz_rad_s,
                commanded_vx_m_s=vx_ref,
                commanded_wz_rad_s=wz_ref,
                actual_riser_position_m=actual_riser_pre,
                target_riser_position_m=sample.riser_q,
            )
        else:
            teacher_residual_action = normalized_residual_label.astype(np.float32)
        lookahead_samples = [
            interpolate_riser_playback_plan(
                plan,
                min(execution_duration_s, phase_time_s + horizon_s),
            )
            for horizon_s in LOOKAHEAD_HORIZONS_S
        ]
        lookahead_feedforward = np.asarray(
            [
                phase_scaled_feedforward(future, progress_scale)[:3]
                for future in lookahead_samples
            ],
            dtype=np.float64,
        )
        executed_observation = build_executed_observation(
            lqr_state=current_states[0],
            actual_base_xy_yaw=actual_base,
            target_base_xy_yaw=sample.base_xy_yaw,
            actual_camera_position_world_m=actual_camera_position_pre,
            target_camera_position_world_m=sample.target_position_world_m,
            actual_camera_quat_wxyz=actual_camera_quaternion_pre,
            target_camera_quat_wxyz=target_physical_camera_quaternion,
            riser_position_m=actual_riser_pre,
            riser_velocity_m_s=actual_riser_velocity_pre,
            riser_target_m=sample.riser_q,
            feedforward_vx_m_s=phase_feedforward_v_mps,
            feedforward_wz_rad_s=phase_feedforward_wz_rad_s,
            feedforward_riser_velocity_m_s=phase_feedforward_riser_velocity_mps,
            phase_fraction=phase_time_s / execution_duration_s,
            progress_scale=progress_scale,
            previous_residual_action=previous_residual_action,
            lookahead_base_xy_yaw=np.asarray(
                [future.base_xy_yaw for future in lookahead_samples]
            ),
            lookahead_camera_position_world_m=np.asarray(
                [
                    future.target_position_world_m
                    for future in lookahead_samples
                ]
            ),
            lookahead_camera_quat_wxyz=np.asarray(
                [
                    semantic_dfr_to_physical_cam_quat_wxyz(
                        future.target_semantic_dfr_quat_wxyz
                    )
                    for future in lookahead_samples
                ]
            ),
            lookahead_riser_target_m=np.asarray(
                [future.riser_q for future in lookahead_samples]
            ),
            lookahead_feedforward_v_wz_riser=lookahead_feedforward,
        )
        if residual_policy is None and not zero_policy_action:
            applied_residual_action = (
                teacher_residual_action
                if dataset_dir is not None
                else np.zeros(3, dtype=np.float32)
            )
            commanded_riser_target = sample.riser_q
        else:
            if zero_policy_action:
                applied_residual_action = np.zeros(3, dtype=np.float32)
            else:
                with torch.inference_mode():
                    policy_output = residual_policy(
                        torch.as_tensor(
                            executed_observation[None, :],
                            dtype=torch.float32,
                            device=residual_policy_device,
                        )
                    )
                applied_residual_action = (
                    policy_output.detach().cpu().numpy().reshape(-1)
                )
            if (
                applied_residual_action.shape != (3,)
                or not np.isfinite(applied_residual_action).all()
                or np.max(np.abs(applied_residual_action)) > 1.0 + 1e-6
            ):
                raise ValueError("residual policy produced an invalid action")
            vx_ref, wz_ref, commanded_riser_target = apply_residual_action(
                phase_feedforward_v_mps,
                phase_feedforward_wz_rad_s,
                actual_riser_pre,
                applied_residual_action,
                maximum_linear_velocity_m_s=tracking_cfg.maximum_linear_velocity_mps,
                maximum_yaw_rate_rad_s=tracking_cfg.maximum_yaw_rate_radps,
                riser_bounds_m=(kinematics.riser_lower, kinematics.riser_upper),
            )
        applied_residual_actions.append(applied_residual_action.copy())
        if dataset_dir is not None:
            dataset_observations.append(executed_observation)
            dataset_actions.append(teacher_residual_action)
            dataset_elapsed_time.append(elapsed_s)
            dataset_phase_time.append(phase_time_s)
            dataset_teacher_commands.append([vx_ref, wz_ref, sample.riser_q])
        previous_residual_action = applied_residual_action
        riser_target = torch.tensor(
            [[commanded_riser_target]], dtype=torch.float32, device=unwrapped.device
        )
        riser_velocity_target = torch.tensor(
            [[phase_feedforward_riser_velocity_mps]],
            dtype=torch.float32,
            device=unwrapped.device,
        )
        proxy_command = sample.proxy_gimbal_q.copy()
        proxy_command_rate = phase_feedforward_proxy_velocity_rad_s.copy()
        if not args.disable_semantic_proxy_state_adapter:
            attitude_ik = kinematics.solve_semantic_attitude_robust(
                root_quaternion,
                sample.riser_q,
                sample.target_semantic_dfr_quat_wxyz,
                previous_proxy_command,
            )
            internal_attitude_ik_failures += int(not attitude_ik.converged)
            internal_attitude_ik_error_max_deg = max(
                internal_attitude_ik_error_max_deg,
                math.degrees(attitude_ik.orientation_error_rad),
            )
            desired_proxy = attitude_ik.gimbal_q.copy()
            desired_proxy[2] = previous_proxy_command[2] + math.atan2(
                math.sin(desired_proxy[2] - previous_proxy_command[2]),
                math.cos(desired_proxy[2] - previous_proxy_command[2]),
            )
            maximum_delta = math.radians(
                args.maximum_internal_proxy_rate_deg_s
            ) / POLICY_HZ
            proxy_delta = np.clip(
                desired_proxy - previous_proxy_command,
                -maximum_delta,
                maximum_delta,
            )
            proxy_command = previous_proxy_command + proxy_delta
            proxy_command_rate = proxy_delta * POLICY_HZ
            internal_proxy_rate_max_deg_s = max(
                internal_proxy_rate_max_deg_s,
                float(np.max(np.abs(np.rad2deg(proxy_command_rate)))),
            )
            previous_proxy_command = proxy_command.copy()
        actual_proxy_pre = robot.data.joint_pos[0, proxy_ids].detach().cpu().numpy()
        proxy_sim_command = proxy_command.copy()
        proxy_sim_command[2] = nearest_equivalent_angle(
            proxy_command[2], actual_proxy_pre[2]
        )
        proxy_target = torch.as_tensor(
            proxy_sim_command[None, :],
            dtype=torch.float32,
            device=unwrapped.device,
        )
        robot.set_joint_position_target(riser_target, joint_ids=[riser_id])
        robot.set_joint_velocity_target(
            riser_velocity_target, joint_ids=[riser_id]
        )
        robot.set_joint_position_target(proxy_target, joint_ids=proxy_ids)
        # The proxy represents a DJI attitude setpoint, not a motor shaft.  The
        # Ronin controller owns motor velocity, so injecting differentiated
        # teacher labels into the PhysX velocity drive makes the proxy lead its
        # semantic command.  Keep the differentiated rate for contract audits.
        if not args.disable_semantic_proxy_state_adapter:
            proxy_velocity_state = torch.as_tensor(
                proxy_command_rate[None, :],
                dtype=torch.float32,
                device=unwrapped.device,
            )
            robot.write_joint_state_to_sim(
                proxy_target,
                proxy_velocity_state,
                joint_ids=proxy_ids,
                env_ids=env_ids,
            )

        body_com_positions = robot.data.body_com_pos_w[0]
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
                config=controller_cfg,
                pitch_bias_override_rad=(
                    None
                    if args.disable_com_pitch_feedforward
                    else np.array([com_pitch_bias])
                ),
            )
            action = action.astype(np.float32)
        dataset_baseline_actions.append(action[0].copy())
        saturated_actions += int(
            np.count_nonzero(np.abs(action) >= controller_cfg.action_limit - 1e-6)
        )
        action_count += action.size
        obs, _, terminated, truncated, _ = env.step(
            torch.as_tensor(action, device=unwrapped.device)
        )
        current_states = (
            obs["policy"][:, : len(LQR_STATE_NAMES)].detach().cpu().numpy()
        )
        state = unwrapped._state_terms()
        root_position_post = robot.data.root_pos_w[0].detach().cpu().numpy()
        root_quaternion_post = robot.data.root_quat_w[0].detach().cpu().numpy()
        actual_base_post = np.array(
            [
                root_position_post[0],
                root_position_post[1],
                yaw_from_quaternion_wxyz(root_quaternion_post),
            ]
        )
        actual_cam_position = robot.data.body_pos_w[0, cam_id].detach().cpu().numpy()
        actual_cam_quaternion = (
            robot.data.body_quat_w[0, cam_id].detach().cpu().numpy()
        )
        position_error = float(
            np.linalg.norm(actual_cam_position - sample.target_position_world_m)
        )
        physical_target = quaternion_matrix_wxyz(
            semantic_dfr_to_physical_cam_quat_wxyz(
                sample.target_semantic_dfr_quat_wxyz
            )
        )
        attitude_error_deg = math.degrees(
            float(
                np.linalg.norm(
                    rotation_error_vector(
                        quaternion_matrix_wxyz(actual_cam_quaternion), physical_target
                    )
                )
            )
        )
        actual_riser = float(robot.data.joint_pos[0, riser_id].item())
        actual_proxy = robot.data.joint_pos[0, proxy_ids].detach().cpu().numpy()
        signed_proxy_error = proxy_sim_command - actual_proxy
        signed_proxy_error[2] = continuous_joint_error(
            proxy_sim_command[2], actual_proxy[2]
        )
        signed_proxy_error_deg = np.rad2deg(signed_proxy_error)
        actual_proxy_velocity_deg_s = np.rad2deg(
            robot.data.joint_vel[0, proxy_ids].detach().cpu().numpy()
        )
        target_proxy_velocity_deg_s = np.rad2deg(
            proxy_command_rate
        )
        riser_error = abs(actual_riser - sample.riser_q)
        proxy_error_deg = np.abs(signed_proxy_error_deg)
        pitch_deg = math.degrees(abs(float(state["pitch"][0].item())))
        camera_position_error_vector = (
            actual_cam_position - sample.target_position_world_m
        )
        base_xy_error = float(
            np.linalg.norm(actual_base_post[:2] - sample.base_xy_yaw[:2])
        )
        base_yaw_error = abs(
            math.atan2(
                math.sin(actual_base_post[2] - sample.base_xy_yaw[2]),
                math.cos(actual_base_post[2] - sample.base_xy_yaw[2]),
            )
        )
        peak_base_xy_error_m = max(peak_base_xy_error_m, base_xy_error)
        peak_base_yaw_error_deg = max(
            peak_base_yaw_error_deg, math.degrees(base_yaw_error)
        )
        position_errors.append(position_error)
        attitude_errors_deg.append(attitude_error_deg)
        riser_errors.append(riser_error)
        proxy_errors_deg.append(proxy_error_deg)
        proxy_velocity_samples_deg_s.append(actual_proxy_velocity_deg_s)
        proxy_target_velocity_samples_deg_s.append(target_proxy_velocity_deg_s)
        pitch_samples_deg.append(pitch_deg)
        if target_marker is not None:
            target_marker.visualize(sample.target_position_world_m[None, :])

        if hasattr(robot.data, "applied_torque"):
            riser_effort = abs(float(robot.data.applied_torque[0, riser_id].item()))
            proxy_effort = robot.data.applied_torque[0, proxy_ids].abs()
            proxy_effort_np = proxy_effort.detach().cpu().numpy()
            latest_proxy_effort_nm = (
                robot.data.applied_torque[0, proxy_ids].detach().cpu().numpy()
            )
            proxy_effort_samples_nm.append(proxy_effort_np)
            saturated_riser += int(riser_effort >= 300.0 - 1e-3)
            riser_effort_count += 1
            saturated_proxy += int(torch.count_nonzero(proxy_effort >= 10.0 - 1e-3))
            proxy_axis_saturated += proxy_effort_np >= 10.0 - 1e-3
            proxy_effort_count += len(proxy_ids)

        progress_scale = 1.0
        if not args.disable_phase_governor:
            progress_scale = min(
                bounded_progress_scale(base_xy_error, position_error, tracking_cfg),
                balance_progress_scale(abs(float(state["pitch"][0].item()))),
            )
        progress_samples.append(progress_scale)
        if step % 200 == 0 or phase_time_s >= execution_duration_s:
            trace.append(
                {
                    "step": step + 1,
                    "elapsed_s": elapsed_s,
                    "phase_time_s": phase_time_s,
                    "progress_scale": progress_scale,
                    "position_error_m": position_error,
                    "attitude_error_deg": attitude_error_deg,
                    "pitch_deg": pitch_deg,
                    "riser_error_m": riser_error,
                    "proxy_error_deg": proxy_error_deg.tolist(),
                    "proxy_signed_error_deg": signed_proxy_error_deg.tolist(),
                    "proxy_target_deg": np.rad2deg(proxy_sim_command).tolist(),
                    "proxy_unwrapped_semantic_target_deg": np.rad2deg(
                        proxy_command
                    ).tolist(),
                    "proxy_actual_deg": np.rad2deg(actual_proxy).tolist(),
                    "proxy_velocity_deg_s": actual_proxy_velocity_deg_s.tolist(),
                    "proxy_target_velocity_deg_s": target_proxy_velocity_deg_s.tolist(),
                    "proxy_applied_effort_nm": latest_proxy_effort_nm.tolist(),
                    "base_xy_error_m": base_xy_error,
                    "base_yaw_error_deg": math.degrees(base_yaw_error),
                    "actual_base_xy_yaw": actual_base_post.tolist(),
                    "target_base_xy_yaw": sample.base_xy_yaw.tolist(),
                    "camera_position_error_xyz_m": (
                        camera_position_error_vector.tolist()
                    ),
                    "actual_camera_position_world_m": actual_cam_position.tolist(),
                    "target_camera_position_world_m": (
                        sample.target_position_world_m.tolist()
                    ),
                    "actual_yaw_rate_rad_s": float(state["yaw_rate"][0].item()),
                    "phase_feedforward_v_mps": phase_feedforward_v_mps,
                    "phase_feedforward_wz_rad_s": phase_feedforward_wz_rad_s,
                    "vx_reference_mps": vx_ref,
                    "wz_reference_rad_s": wz_ref,
                    "along_track_error_m": base_tracking_diagnostics[
                        "along_track_error_m"
                    ],
                    "cross_track_error_m": base_tracking_diagnostics[
                        "cross_track_error_m"
                    ],
                    "base_yaw_error_rad": base_tracking_diagnostics[
                        "yaw_error_rad"
                    ],
                    "raw_velocity_reference_mps": base_tracking_diagnostics[
                        "raw_velocity_reference_mps"
                    ],
                    "base_position_error_m": base_tracking_diagnostics[
                        "base_position_error_m"
                    ],
                    "feedforward_direction": base_tracking_diagnostics[
                        "feedforward_direction"
                    ],
                    "feedback_motion_direction": base_tracking_diagnostics[
                        "feedback_motion_direction"
                    ],
                    "direction_recovery_blend": base_tracking_diagnostics[
                        "direction_recovery_blend"
                    ],
                    "motion_direction": base_tracking_diagnostics[
                        "motion_direction"
                    ],
                }
            )
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
        if phase_time_s >= execution_duration_s:
            break
        phase_time_s = min(
            execution_duration_s,
            phase_time_s + progress_scale / POLICY_HZ,
        )

    position = np.asarray(position_errors, dtype=np.float64)
    attitude = np.asarray(attitude_errors_deg, dtype=np.float64)
    riser_error_values = np.asarray(riser_errors, dtype=np.float64)
    proxy_error_values = np.asarray(proxy_errors_deg, dtype=np.float64)
    proxy_velocity_values = np.asarray(proxy_velocity_samples_deg_s, dtype=np.float64)
    proxy_target_velocity_values = np.asarray(
        proxy_target_velocity_samples_deg_s, dtype=np.float64
    )
    proxy_effort_values = np.asarray(proxy_effort_samples_nm, dtype=np.float64)
    applied_residual_values = np.asarray(
        applied_residual_actions, dtype=np.float64
    )
    raw_residual_values = np.asarray(raw_residual_commands, dtype=np.float64)
    normalized_residual_values = np.asarray(
        normalized_residual_labels, dtype=np.float64
    )
    teacher_residual_values = np.asarray(dataset_actions, dtype=np.float64)
    pitches = np.asarray(pitch_samples_deg, dtype=np.float64)
    action_saturation_ratio = saturated_actions / max(action_count, 1)
    riser_saturation_ratio = saturated_riser / max(riser_effort_count, 1)
    proxy_saturation_ratio = saturated_proxy / max(proxy_effort_count, 1)
    checks = {
        "completed_reference": phase_time_s >= execution_duration_s,
        "no_termination": termination is None,
        "pitch_bounded": float(np.max(pitches)) <= args.maximum_pitch_deg,
        "position_p95_bounded": float(np.percentile(position, 95))
        <= args.maximum_position_p95_m,
        "position_max_bounded": float(np.max(position))
        <= args.maximum_position_error_m,
        "attitude_p95_bounded": float(np.percentile(attitude, 95))
        <= args.maximum_attitude_p95_deg,
        "attitude_max_bounded": float(np.max(attitude))
        <= args.maximum_attitude_error_deg,
        "riser_servo_error_bounded": float(np.percentile(riser_error_values, 95))
        <= args.maximum_riser_servo_error_m,
        "proxy_servo_error_bounded": float(np.percentile(proxy_error_values, 95))
        <= args.maximum_proxy_servo_error_deg,
        "action_saturation_bounded": action_saturation_ratio
        <= args.maximum_saturation_ratio,
        "riser_saturation_bounded": riser_saturation_ratio
        <= args.maximum_saturation_ratio,
        "proxy_saturation_bounded": proxy_saturation_ratio
        <= args.maximum_saturation_ratio,
        "internal_attitude_ik_converged": internal_attitude_ik_failures == 0,
        "internal_proxy_rate_bounded": internal_proxy_rate_max_deg_s
        <= args.maximum_internal_proxy_rate_deg_s + 1e-6,
        "residual_teacher_unclipped": dataset_dir is None
        or float(np.max(np.abs(teacher_residual_values))) < 1.0 - 1e-6,
    }
    dynamic_quality_passed = all(checks.values())
    residual_label_envelope_ok = all(
        residual_action_envelope_passed(value)
        for value in normalized_residual_values
    )
    dataset_path = None
    if dataset_dir is not None and dynamic_quality_passed:
        dataset_path = dataset_dir / f"case_{plan.case:04d}_executed_residual_v2.npz"
        count = len(dataset_observations)
        save_case_dataset(
            dataset_path,
            plan.case,
            {
                "observations": np.asarray(dataset_observations, dtype=np.float32),
                "actions": np.asarray(dataset_actions, dtype=np.float32),
                "case_ids": np.full(count, plan.case, dtype=np.int16),
                "elapsed_time_s": np.asarray(dataset_elapsed_time, dtype=np.float64),
                "phase_time_s": np.asarray(dataset_phase_time, dtype=np.float64),
                "baseline_wheel_actions": np.asarray(
                    dataset_baseline_actions, dtype=np.float32
                ),
                "teacher_commands": np.asarray(
                    dataset_teacher_commands, dtype=np.float32
                ),
            },
        )
    return {
        "case": plan.case,
        "planning_strategy": plan.planning_strategy,
        "source_duration_s": source_duration_s,
        "execution_duration_s": execution_duration_s,
        "completed_phase_time_s": phase_time_s,
        "completed_steps": completed_steps,
        "wall_duration_s": completed_steps / POLICY_HZ,
        "progress_scale_min": float(np.min(progress_samples)),
        "progress_scale_mean": float(np.mean(progress_samples)),
        "position_error_p95_m": float(np.percentile(position, 95)),
        "position_error_max_m": float(np.max(position)),
        "attitude_error_p95_deg": float(np.percentile(attitude, 95)),
        "attitude_error_max_deg": float(np.max(attitude)),
        "pitch_p95_deg": float(np.percentile(pitches, 95)),
        "pitch_max_deg": float(np.max(pitches)),
        "riser_servo_error_p95_m": float(np.percentile(riser_error_values, 95)),
        "riser_servo_error_max_m": float(np.max(riser_error_values)),
        "proxy_servo_error_p95_deg": float(np.percentile(proxy_error_values, 95)),
        "proxy_servo_error_max_deg": float(np.max(proxy_error_values)),
        "proxy_axis_error_p95_deg": np.percentile(
            proxy_error_values, 95, axis=0
        ).tolist(),
        "proxy_axis_velocity_max_abs_deg_s": np.max(
            np.abs(proxy_velocity_values), axis=0
        ).tolist(),
        "proxy_axis_target_velocity_max_abs_deg_s": np.max(
            np.abs(proxy_target_velocity_values), axis=0
        ).tolist(),
        "proxy_axis_effort_p95_nm": np.percentile(
            proxy_effort_values, 95, axis=0
        ).tolist(),
        "proxy_axis_saturation_ratio": (
            proxy_axis_saturated / max(len(proxy_effort_values), 1)
        ).tolist(),
        "action_saturation_ratio": action_saturation_ratio,
        "riser_saturation_ratio": riser_saturation_ratio,
        "proxy_saturation_ratio": proxy_saturation_ratio,
        "residual_action_abs_max": np.max(
            np.abs(applied_residual_values), axis=0
        ).tolist(),
        "raw_residual_command_abs_max": np.max(
            np.abs(raw_residual_values), axis=0
        ).tolist(),
        "normalized_residual_label_abs_max": np.max(
            np.abs(normalized_residual_values), axis=0
        ).tolist(),
        "raw_residual_label_applied_to_commands": False,
        "dynamic_quality_passed": dynamic_quality_passed,
        "residual_label_envelope_passed": residual_label_envelope_ok,
        "residual_label_admission_passed": (
            dynamic_quality_passed and residual_label_envelope_ok
        ),
        "peak_base_xy_error_m": peak_base_xy_error_m,
        "peak_base_yaw_error_deg": peak_base_yaw_error_deg,
        "peak_com_pitch_bias_deg": peak_com_pitch_bias_deg,
        "internal_attitude_ik_failures": internal_attitude_ik_failures,
        "internal_attitude_ik_error_max_deg": internal_attitude_ik_error_max_deg,
        "internal_proxy_rate_max_deg_s": internal_proxy_rate_max_deg_s,
        "termination": termination,
        "trace": trace,
        "checks": checks,
        "executed_residual_dataset": (
            None if dataset_path is None else str(dataset_path.resolve())
        ),
        "passed": dynamic_quality_passed,
    }


def main() -> int:
    cases = parse_cases(args.cases)
    if args.maximum_duration_scale < 1.0:
        raise ValueError("maximum duration scale must be at least one")
    if args.video_dir is not None and len(cases) != 1:
        raise ValueError("video recording requires exactly one case")
    if args.dataset_dir is not None and (
        args.residual_policy is not None or args.zero_policy_action
    ):
        raise ValueError("teacher dataset collection and residual rollout are exclusive")
    if args.residual_policy is not None and args.zero_policy_action:
        raise ValueError("learned and zero-action policy modes are exclusive")
    if args.video_fps <= 0:
        raise ValueError("video fps must be positive")
    plans = {
        case: load_riser_playback_plan(
            plan_path(args.plan_dir, args.plan_filename_template, case)
        )
        for case in cases
    }
    if any(plan.case != case for case, plan in plans.items()):
        raise ValueError("playback case metadata mismatch")
    residual_policy_device = torch.device(args.residual_policy_device)
    residual_policy = None
    if args.residual_policy is not None:
        if residual_policy_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA residual policy requested but CUDA is unavailable")
        residual_policy = torch.jit.load(
            str(args.residual_policy), map_location=residual_policy_device
        ).eval()
    gain_data = json.loads(args.gains.read_text(encoding="utf-8"))
    gain = np.asarray(gain_data["selected_gain"], dtype=np.float64)
    control_interval = int(gain_data["control_interval_steps"])
    if gain.shape != (len(ACTION_NAMES), len(LQR_STATE_NAMES)):
        raise ValueError(f"invalid LQR gain shape: {gain.shape}")

    register_isaac_lab_tasks()
    cfg = RecomoTwoWheelBalanceEnvCfg()
    cfg.seed = 20260716
    cfg.scene.num_envs = 1
    cfg.robot_cfg = copy.deepcopy(TWO_WHEEL_RISER_CFG)
    cfg.episode_length_s = (
        max(float(plan.time_s[-1]) for plan in plans.values())
        * args.maximum_duration_scale
        + 2.0
    )
    cfg.reset_pitch_rad = 0.0
    cfg.control_mode = "direct"
    if args.video_dir is not None:
        cfg.viewer.eye = (3.2, -4.5, 2.4)
        cfg.viewer.lookat = (0.0, 0.0, 1.1)
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
                prim_path="/Visuals/RiserPlaybackCurrentTarget",
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
                prim_path="/Visuals/RiserPlaybackTargetPath",
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
                plans[cases[0]].time_s[-1]
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
            name_prefix=f"two-wheel-riser-case-{cases[0]:04d}",
            disable_logger=True,
        )
    else:
        env = raw_env
    results = [
        evaluate_case(
            env,
            plans[case],
            gain,
            control_interval,
            target_marker,
            path_marker,
            args.dataset_dir,
            residual_policy,
            residual_policy_device,
            args.zero_policy_action,
        )
        for case in cases
    ]
    env.close()
    result = {
        "schema": "recomo_two_wheel_riser_reference_playback_v1",
        "training_started": False,
        "ppo_authorized": False,
        "controller_profile": "structural_robust_v1",
        "tracking_profile": "riser_recovery_direction_v4",
        "tracking_direction_blend_speed_mps": (
            riser_tracking_config().direction_blend_speed_mps
        ),
        "tracking_direction_recovery_error_range_m": [
            riser_tracking_config().direction_recovery_error_start_m,
            riser_tracking_config().direction_recovery_error_full_m,
        ],
        "phase_feedforward_contract": "derivatives_scaled_by_progress_v1",
        "controller_overrides": {
            name: value
            for name, value in {
                "wz_kp": args.controller_wz_kp,
                "wz_ki": args.controller_wz_ki,
                "wz_feedforward": args.controller_wz_feedforward,
                "wheel_difference_kp": args.controller_wheel_difference_kp,
            }.items()
            if value is not None
        },
        "tracking_overrides": {
            name: value
            for name, value in {
                "along_track_kp": args.tracking_along_kp,
                "cross_track_kp": args.tracking_cross_kp,
                "yaw_kp": args.tracking_yaw_kp,
            }.items()
            if value is not None
        },
        "position_observation_link": "physical_cam_link_fk",
        "target_attitude_contract": "semantic_dfr_to_physical_cam_v1",
        "hardware_proxy_command_contract": "semantic_attitude_position_only",
        "simulation_proxy_realization": (
            "rate_audited_ideal_state_adapter"
            if not args.disable_semantic_proxy_state_adapter
            else "implicit_position_drive_diagnostic"
        ),
        "phase_governor_enabled": not args.disable_phase_governor,
        "com_pitch_feedforward_enabled": not args.disable_com_pitch_feedforward,
        "trajectory_command_source": (
            "deterministic_teacher"
            if residual_policy is None and not args.zero_policy_action
            else (
                "zero_policy_action_baseline"
                if args.zero_policy_action
                else "torchscript_residual_policy"
            )
        ),
        "residual_policy": (
            None if args.residual_policy is None else str(args.residual_policy.resolve())
        ),
        "cases": cases,
        "passed_case_count": sum(item["passed"] for item in results),
        "dynamic_quality_passed": all(
            item["dynamic_quality_passed"] for item in results
        ),
        "residual_label_envelope_passed": all(
            item["residual_label_envelope_passed"] for item in results
        ),
        "residual_label_admission_passed": all(
            item["residual_label_admission_passed"] for item in results
        ),
        "results": results,
        "passed": all(item["passed"] for item in results),
    }
    if args.video_dir is not None:
        videos = sorted(
            args.video_dir.glob("*.mp4"), key=lambda path: path.stat().st_mtime
        )
        if not videos:
            raise RuntimeError(f"RecordVideo produced no MP4 in {args.video_dir}")
        result["video"] = str(videos[-1].resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


def write_runtime_failure(exc: Exception) -> None:
    message = str(exc)
    match = re.search(r"residual action scale is too small: \[([^]]+)\]", message)
    normalized = [float(value) for value in match.group(1).split()] if match else None
    cases = parse_cases(args.cases)
    failure_plan = None
    if len(cases) == 1:
        try:
            failure_plan = load_riser_playback_plan(
                plan_path(args.plan_dir, args.plan_filename_template, cases[0])
            )
        except Exception:
            failure_plan = None
    classification = (
        "residual_label_envelope_rejection"
        if normalized is not None
        else "runtime_exception"
    )
    result = {
        "schema": "recomo_two_wheel_riser_reference_playback_failure_v1",
        "training_started": False,
        "ppo_authorized": False,
        "trajectory_command_source": "deterministic_teacher",
        "residual_policy": None,
        "cases": cases,
        "passed_case_count": 0,
        "results": [
            {
                "case": cases[0] if len(cases) == 1 else None,
                "passed": False,
                "stage": (
                    "runtime_label_capture_envelope"
                    if normalized is not None
                    else "runtime"
                ),
                "classification": classification,
                "exception_type": type(exc).__name__,
                "exception_message": message,
                "normalized_action": normalized,
                "offline_admission_failure": False,
                "failing_step_action_applied": False,
                "source_duration_s": (
                    None
                    if failure_plan is None
                    else float(failure_plan.source_time_s[-1])
                ),
                "execution_duration_s": (
                    None if failure_plan is None else float(failure_plan.time_s[-1])
                ),
                "executed_residual_dataset": None,
            }
        ],
        "passed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


exit_code = 1
try:
    exit_code = main()
except Exception as exc:
    import traceback

    write_runtime_failure(exc)
    traceback.print_exc()
    sys.stderr.flush()
    shutdown_guard = threading.Timer(60.0, lambda: os._exit(1))
    shutdown_guard.daemon = True
    shutdown_guard.start()
    try:
        app.close()
    finally:
        shutdown_guard.cancel()
else:
    app.close()
raise SystemExit(exit_code)
