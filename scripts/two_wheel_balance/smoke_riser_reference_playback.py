#!/usr/bin/env python3
"""Replay corrected riser plans with balance control and optional RTX video."""

from __future__ import annotations

import argparse
import copy
import hashlib
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from rl_platform.tasks.two_wheel_balance.runtime_heartbeat import (
    write_runtime_heartbeat,
)
from rl_platform.tasks.two_wheel_balance.riser_perturbation import (
    DeterministicWrenchPulse,
    DeterministicWrenchPulseRuntime,
    load_deterministic_wrench_profile,
)
from rl_platform.tasks.two_wheel_balance.riser_corrective_teacher import (
    CORRECTIVE_TEACHER_CONTRACT,
    CorrectiveTeacherTelemetry,
    build_corrective_teacher_action,
    load_corrective_teacher_profile,
)
from rl_platform.tasks.two_wheel_balance.riser_corrective_capture import (
    load_capture_admission,
    save_corrective_capture,
)

from isaaclab.app import AppLauncher


def parse_action_scales(value: str) -> np.ndarray:
    scales = np.asarray([float(item) for item in value.split(",")], dtype=np.float64)
    if scales.shape != (3,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise argparse.ArgumentTypeError("expected three positive residual action scales")
    return scales


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
parser.add_argument("--controller-vx-kp", type=float)
parser.add_argument("--controller-wz-kp", type=float)
parser.add_argument("--controller-wz-ki", type=float)
parser.add_argument("--controller-wz-feedforward", type=float)
parser.add_argument("--controller-wheel-difference-kp", type=float)
parser.add_argument(
    "--limit-total-pitch-reference",
    action="store_true",
    help=(
        "Apply the pitch limit to equilibrium bias plus velocity correction, "
        "rather than limiting the velocity correction before adding the bias."
    ),
)
parser.add_argument(
    "--reset-opposing-vx-integral-on-directional-deficit",
    action="store_true",
    help=(
        "Reset only opposing longitudinal PI memory when an active, deadbanded "
        "velocity reference is under-tracked; the frozen inner LQR remains "
        "unchanged."
    ),
)
parser.add_argument(
    "--vx-integral-reset-reference-deadband-mps",
    type=float,
    default=0.05,
)
parser.add_argument(
    "--use-root-velocity-outer-feedback",
    action="store_true",
    help="Use measured root vx for the outer PI only; keep wheel state in the frozen LQR.",
)
parser.add_argument("--tracking-along-kp", type=float)
parser.add_argument("--tracking-cross-kp", type=float)
parser.add_argument("--tracking-yaw-kp", type=float)
parser.add_argument(
    "--tracking-maximum-linear-velocity-mps",
    type=float,
    help=(
        "Optional symmetric base-command cap. It may reduce, but never expand, "
        "the default 0.4 m/s tracking envelope."
    ),
)
parser.add_argument(
    "--tracking-minimum-progress-scale",
    type=float,
    help=(
        "Optional position-governor floor; 0 holds immutable trajectory phase "
        "at the full-error boundary."
    ),
)
parser.add_argument(
    "--enable-camera-lever-arm-compensation",
    action="store_true",
    help="Apply bounded camera-to-base lever displacement feedback to base XY.",
)
parser.add_argument(
    "--use-commanded-base-progress-error",
    action="store_true",
    help=(
        "Govern trajectory phase using error to the lever-compensated base "
        "command instead of the nominal plan allocation."
    ),
)
parser.add_argument("--camera-lever-arm-compensation-gain", type=float, default=1.0)
parser.add_argument("--maximum-camera-lever-arm-correction-m", type=float, default=0.05)
parser.add_argument(
    "--enable-camera-error-recovery-governor",
    action="store_true",
    help="Reduce phase progress near the position gate while camera correction saturates.",
)
parser.add_argument("--camera-recovery-error-start-m", type=float, default=0.13)
parser.add_argument("--camera-recovery-error-full-m", type=float, default=0.155)
parser.add_argument("--minimum-camera-recovery-scale", type=float, default=0.20)
parser.add_argument("--video-dir", type=Path)
parser.add_argument(
    "--dataset-dir",
    type=Path,
    help="Write dense pre-action executed-state residual datasets per case.",
)
parser.add_argument(
    "--raw-teacher-dir",
    type=Path,
    help=(
        "Write scale-independent raw teacher captures without applying residuals; "
        "these artifacts are not directly trainable."
    ),
)
parser.add_argument(
    "--policy-trace-dir",
    type=Path,
    help=(
        "Write non-trainable policy-rate observations, applied actions, commands, "
        "and post-step outcomes for diagnosis only."
    ),
)
parser.add_argument(
    "--shadow-teacher-trace-dir",
    type=Path,
    help=(
        "Record deterministic teacher labels on learned-policy or deterministic-"
        "controller visited states without applying or admitting the labels."
    ),
)
parser.add_argument(
    "--deterministic-wrench-profile",
    type=Path,
    help=(
        "Hash-bound, measurement-only body-longitudinal wrench pulse; requires "
        "one learned-policy shadow-teacher case and cannot create a dataset."
    ),
)
parser.add_argument(
    "--corrective-teacher-profile",
    type=Path,
    help=(
        "Hash-bound, case-30-only corrective teacher above the complete model "
        "planner; diagnostic paired rollout only and never a capture mode."
    ),
)
parser.add_argument(
    "--corrective-teacher-capture-dir",
    type=Path,
    help=(
        "Write one admitted corrective-label archive. Requires a "
        "separate hash-bound runtime admission and never creates a normalized "
        "training dataset."
    ),
)
parser.add_argument(
    "--corrective-teacher-capture-admission",
    type=Path,
    help="Runtime admission paired with --corrective-teacher-capture-dir.",
)
parser.add_argument(
    "--corrective-teacher-capture-split",
    choices=("train", "validation"),
    help=(
        "Expected split pinned independently of the admission payload; required "
        "with the corrective capture directory and admission."
    ),
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
parser.add_argument(
    "--residual-action-scales",
    type=parse_action_scales,
    default=parse_action_scales("0.30,0.40,0.10"),
    help="Physical [vx,wz,riser] scales for normalized residual policy actions.",
)
parser.add_argument(
    "--policy-command-base",
    choices=("phase_feedforward", "model_based_planner"),
    default="phase_feedforward",
    help=(
        "Apply policy output over historical phase feedforward or over the complete "
        "model-based planner command."
    ),
)
# Video frame stride changes rendering only; physics and control remain at 200 Hz.
parser.add_argument("--video-frame-stride", type=int, default=1)
parser.add_argument("--video-fps", type=int, default=200)
parser.add_argument(
    "--runtime-heartbeat",
    type=Path,
    help=(
        "Atomically overwrite a non-training progress snapshot so an external "
        "timeout preserves the last completed policy-step evidence."
    ),
)
parser.add_argument(
    "--runtime-heartbeat-interval-steps",
    type=int,
    default=2000,
)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.runtime_heartbeat_interval_steps <= 0:
    parser.error("--runtime-heartbeat-interval-steps must be positive")
corrective_teacher_case = None
corrective_teacher_config = None
corrective_teacher_profile_identity = None
if args.corrective_teacher_profile is not None:
    requested_cases = [int(item) for item in args.cases.split(",") if item.strip()]
    if len(requested_cases) != 1:
        parser.error("corrective teacher profile requires one pinned case only")
    try:
        (
            corrective_teacher_case,
            corrective_teacher_config,
            corrective_teacher_profile_identity,
        ) = load_corrective_teacher_profile(
            args.corrective_teacher_profile,
            expected_case=requested_cases[0],
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if requested_cases != [corrective_teacher_case]:
        parser.error("corrective teacher profile requires its one pinned case only")
    if (
        args.policy_command_base != "model_based_planner"
        or not args.zero_policy_action
        or args.residual_policy is not None
    ):
        parser.error(
            "corrective teacher requires model-based zero-policy mode without "
            "a TorchScript policy"
        )
corrective_capture_admission = None
if (
    args.corrective_teacher_capture_dir is not None
    or args.corrective_teacher_capture_admission is not None
    or args.corrective_teacher_capture_split is not None
):
    if (
        args.corrective_teacher_capture_dir is None
        or args.corrective_teacher_capture_admission is None
        or args.corrective_teacher_capture_split is None
    ):
        parser.error(
            "corrective capture directory, admission, and split are required together"
        )
    if corrective_teacher_config is None or corrective_teacher_profile_identity is None:
        parser.error("corrective capture requires the reviewed corrective profile")
    try:
        corrective_capture_admission = load_capture_admission(
            args.corrective_teacher_capture_admission,
            expected_case=corrective_teacher_case,
            expected_split=args.corrective_teacher_capture_split,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if corrective_capture_admission["case"] != corrective_teacher_case:
        parser.error("corrective capture admission case mismatch")
    if (
        corrective_capture_admission["corrective_profile_sha256"]
        != corrective_teacher_profile_identity["sha256"]
    ):
        parser.error("corrective capture profile identity mismatch")
    if any(
        path is not None
        for path in (
            args.dataset_dir,
            args.raw_teacher_dir,
            args.policy_trace_dir,
            args.shadow_teacher_trace_dir,
        )
    ):
        parser.error("corrective capture is exclusive with every legacy capture path")
if args.policy_command_base == "model_based_planner":
    if args.residual_policy is None and not args.zero_policy_action:
        parser.error("model-based policy command base requires learned or zero policy mode")
    if not np.array_equal(
        args.residual_action_scales, np.array([0.05, 0.05, 0.02])
    ):
        parser.error("model-based policy command base requires scales 0.05,0.05,0.02")
    if any(
        path is not None
        for path in (
            args.dataset_dir,
            args.raw_teacher_dir,
            args.policy_trace_dir,
            args.shadow_teacher_trace_dir,
        )
    ):
        parser.error("model-based residual canary mode forbids every capture path")
deterministic_wrench_profile = None
deterministic_wrench_profile_identity = None
if args.deterministic_wrench_profile is not None:
    requested_cases = [
        int(item) for item in args.cases.split(",") if item.strip()
    ]
    if len(requested_cases) != 1:
        parser.error("deterministic wrench profile requires one pinned case only")
    try:
        (
            deterministic_wrench_profile,
            deterministic_wrench_profile_identity,
        ) = load_deterministic_wrench_profile(
            args.deterministic_wrench_profile,
            expected_case=requested_cases[0],
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if requested_cases != [deterministic_wrench_profile.case]:
        parser.error(
            "deterministic wrench profile requires its one pinned case only"
        )
    legacy_shadow_measurement = (
        args.residual_policy is not None
        and args.shadow_teacher_trace_dir is not None
        and not args.zero_policy_action
        and args.dataset_dir is None
        and args.raw_teacher_dir is None
        and args.policy_trace_dir is None
    )
    corrective_teacher_measurement = (
        corrective_teacher_config is not None
        and args.zero_policy_action
        and args.residual_policy is None
        and args.shadow_teacher_trace_dir is None
        and args.dataset_dir is None
        and args.raw_teacher_dir is None
        and args.policy_trace_dir is None
    )
    corrective_teacher_capture = (
        corrective_teacher_measurement
        and corrective_capture_admission is not None
        and args.corrective_teacher_capture_dir is not None
    )
    model_based_zero_measurement = (
        corrective_teacher_config is None
        and args.policy_command_base == "model_based_planner"
        and args.zero_policy_action
        and args.residual_policy is None
        and args.shadow_teacher_trace_dir is None
        and args.dataset_dir is None
        and args.raw_teacher_dir is None
        and args.policy_trace_dir is None
    )
    if not (
        legacy_shadow_measurement
        or corrective_teacher_measurement
        or corrective_teacher_capture
        or model_based_zero_measurement
    ):
        parser.error(
            "deterministic wrench profile requires learned-policy shadow-teacher "
            "or model-based paired measurement mode and forbids capture"
        )
for name in ("tracking_along_kp", "tracking_cross_kp", "tracking_yaw_kp"):
    value = getattr(args, name)
    if value is not None and value < 0.0:
        parser.error(f"--{name.replace('_', '-')} must be non-negative")
if args.tracking_minimum_progress_scale is not None and not (
    math.isfinite(args.tracking_minimum_progress_scale)
    and 0.0 <= args.tracking_minimum_progress_scale <= 1.0
):
    parser.error("--tracking-minimum-progress-scale must be in [0, 1]")
if args.tracking_maximum_linear_velocity_mps is not None and not (
    math.isfinite(args.tracking_maximum_linear_velocity_mps)
    and 0.0 < args.tracking_maximum_linear_velocity_mps <= 0.4
):
    parser.error(
        "--tracking-maximum-linear-velocity-mps must be in (0, 0.4]"
    )
if (
    args.tracking_minimum_progress_scale is not None
    and args.disable_phase_governor
):
    parser.error(
        "--tracking-minimum-progress-scale requires the phase governor"
    )
if (
    args.tracking_minimum_progress_scale == 0.0
    and args.enable_camera_error_recovery_governor
):
    parser.error(
        "zero-progress hold cannot be combined with the camera error governor"
    )
if not (
    math.isfinite(args.camera_lever_arm_compensation_gain)
    and 0.0 <= args.camera_lever_arm_compensation_gain <= 1.0
):
    parser.error("--camera-lever-arm-compensation-gain must be in [0, 1]")
if not (
    math.isfinite(args.maximum_camera_lever_arm_correction_m)
    and args.maximum_camera_lever_arm_correction_m > 0.0
):
    parser.error("--maximum-camera-lever-arm-correction-m must be positive")
if not (
    math.isfinite(args.camera_recovery_error_start_m)
    and math.isfinite(args.camera_recovery_error_full_m)
    and 0.0
    < args.camera_recovery_error_start_m
    < args.camera_recovery_error_full_m
):
    parser.error("camera recovery error bounds must be finite and increasing")
if not (
    math.isfinite(args.minimum_camera_recovery_scale)
    and 0.0 < args.minimum_camera_recovery_scale <= 1.0
):
    parser.error("--minimum-camera-recovery-scale must be in (0, 1]")
if args.enable_camera_error_recovery_governor and (
    args.disable_phase_governor or not args.enable_camera_lever_arm_compensation
):
    parser.error(
        "camera error recovery requires the phase governor and camera lever-arm compensation"
    )
if args.use_commanded_base_progress_error and (
    args.disable_phase_governor or not args.enable_camera_lever_arm_compensation
):
    parser.error(
        "commanded-base progress error requires the phase governor and camera lever-arm compensation"
    )
if not (
    math.isfinite(args.vx_integral_reset_reference_deadband_mps)
    and args.vx_integral_reset_reference_deadband_mps > 0.0
):
    parser.error("--vx-integral-reset-reference-deadband-mps must be positive")
if args.controller_vx_kp is not None and not (
    math.isfinite(args.controller_vx_kp) and 0.0 < args.controller_vx_kp <= 1.0
):
    parser.error("--controller-vx-kp must be in (0, 1]")
app = AppLauncher(args).app

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
    interpolate_riser_initialization,
    interpolate_riser_playback_plan,
    load_riser_playback_plan,
    phase_scaled_feedforward,
)
from rl_platform.tasks.two_wheel_balance.riser_kinematics import (
    UrdfRiserCameraKinematics,
)
from rl_platform.tasks.two_wheel_balance.riser_control import (
    RISER_THERMAL_FORCE_CONTRACT,
    RiserMotorThermalMonitor,
    balance_progress_scale,
)
from rl_platform.tasks.two_wheel_balance.riser_residual_dataset import (
    LOOKAHEAD_HORIZONS_S,
    MODEL_BASED_POLICY_RESIDUAL_CONTRACT,
    OBSERVATION_INDEX,
    apply_residual_action,
    apply_model_based_policy_residual,
    build_raw_residual_command,
    build_executed_observation,
    build_residual_action,
    normalize_residual_command,
    residual_action_envelope_passed,
    save_case_dataset,
    save_policy_trace,
    save_raw_teacher_case,
    save_shadow_teacher_trace,
)
from rl_platform.tasks.two_wheel_balance.riser_recovery_evidence import (
    LONGITUDINAL_AUTHORITY_TELEMETRY_SCHEMA,
    RECOVERY_TELEMETRY_SCHEMA,
    VELOCITY_FEEDBACK_TELEMETRY_SCHEMA,
    LongitudinalAuthorityTelemetryAccumulator,
    RecoveryTelemetryAccumulator,
    VelocityFeedbackTelemetryAccumulator,
)
from rl_platform.tasks.two_wheel_balance.whole_body_tracking import (
    bounded_base_references,
    bounded_camera_recovery_progress_scale,
    bounded_camera_lever_arm_base_target,
    bounded_progress_scale,
    continuous_joint_error,
    equilibrium_pitch_from_world_com,
    nearest_equivalent_angle,
    riser_tracking_config,
    select_progress_governor_base_error,
    summarize_progress_governor_base_error,
    summarize_progress_hold,
    yaw_from_quaternion_wxyz,
)
from task_spec import register_isaac_lab_tasks


POLICY_HZ = 200.0
WHEEL_RADIUS_M = 0.1016
TOTAL_PITCH_REFERENCE_LIMIT_RAD = cascaded_lqr_config(
    "structural_robust_v1"
).pitch_reference_limit_rad
PROXY_JOINTS = (
    "joint3_gimbal_yaw",
    "joint2_gimbal_roll",
    "joint1_gimbal_pitch",
)


class StridedRecordVideo(gym.wrappers.RecordVideo):
    """Record every Nth control step without changing environment stepping."""

    def __init__(self, *args, frame_stride: int, **kwargs):
        self.frame_stride = frame_stride
        self.capture_call_count = 0
        super().__init__(*args, **kwargs)

    def _capture_frame(self) -> None:
        should_capture = self.capture_call_count % self.frame_stride == 0
        self.capture_call_count += 1
        if should_capture:
            super()._capture_frame()


def tracking_profile_name() -> str:
    if args.tracking_minimum_progress_scale == 0.0:
        if args.tracking_maximum_linear_velocity_mps is not None:
            if args.limit_total_pitch_reference:
                return (
                    "riser_recovery_direction_v4_camera_lever_arm_"
                    "zero_progress_hold_velocity_cap_total_pitch_limit_v1"
                )
            return (
                "riser_recovery_direction_v4_camera_lever_arm_"
                "zero_progress_hold_velocity_cap_v1"
            )
        return "riser_recovery_direction_v4_camera_lever_arm_zero_progress_hold_v1"
    if args.enable_camera_error_recovery_governor:
        return "riser_recovery_direction_v4_camera_lever_arm_error_governor_v1"
    if args.enable_camera_lever_arm_compensation:
        return "riser_recovery_direction_v4_camera_lever_arm_v1"
    return "riser_recovery_direction_v4"


def progress_governor_base_error_source() -> str:
    return (
        "lever_compensated_commanded_base_target"
        if args.use_commanded_base_progress_error
        else "nominal_plan_base_target"
    )


def phase_governor_contract() -> str:
    return (
        "commanded_base_and_camera_error_continuous_phase_scale_v1"
        if args.use_commanded_base_progress_error
        else "position_error_continuous_phase_scale_v1"
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


def set_base_longitudinal_wrench(
    unwrapped,
    force_body_x_n: float,
    application_height_m: float,
) -> None:
    """Apply a body-frame force and its equivalent pitch moment."""

    force = torch.zeros((1, 1, 3), device=unwrapped.device)
    torque = torch.zeros_like(force)
    force[0, 0, 0] = force_body_x_n
    torque[0, 0, 1] = force_body_x_n * application_height_m
    unwrapped.robot.set_external_force_and_torque(
        forces=force,
        torques=torque,
        body_ids=unwrapped._base_body_idx,
        is_global=False,
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
    has_initialization = (
        plan.initialization_time_s is not None
        and plan.initialization_state is not None
        and len(plan.initialization_time_s) > 0
    )
    initial_state = (
        plan.initialization_state[0]
        if has_initialization
        else np.concatenate(
            (plan.base_xy_yaw[0], [plan.riser_q[0]], plan.proxy_gimbal_q[0])
        )
    )
    root_state[0, 0] = initial_state[0]
    root_state[0, 1] = initial_state[1]
    half_yaw = 0.5 * initial_state[2]
    root_state[0, 3] = math.cos(half_yaw)
    root_state[0, 4] = 0.0
    root_state[0, 5] = 0.0
    root_state[0, 6] = math.sin(half_yaw)
    root_state[0, 7:] = 0.0
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = robot.data.default_joint_vel[env_ids].clone()
    joint_pos[0, riser_id] = initial_state[3]
    joint_pos[0, proxy_ids] = torch.as_tensor(
        initial_state[4:], dtype=torch.float32, device=unwrapped.device
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
    raw_teacher_dir: Path | None,
    policy_trace_dir: Path | None,
    shadow_teacher_trace_dir: Path | None,
    residual_policy,
    residual_policy_device: torch.device,
    zero_policy_action: bool,
    wrench_profile: DeterministicWrenchPulse | None,
    runtime_heartbeat: Path | None,
    runtime_heartbeat_interval_steps: int,
    corrective_capture_dir: Path | None,
    corrective_capture_admission: dict[str, object] | None,
    plan_sha256: str,
) -> dict[str, object]:
    _, proxy_ids, riser_id = initialize_case(env, plan)
    unwrapped = env.unwrapped
    robot = unwrapped.robot
    env_ids = torch.zeros(1, dtype=torch.long, device=unwrapped.device)
    perturbation_runtime = DeterministicWrenchPulseRuntime(wrench_profile)
    if wrench_profile is not None:
        set_base_longitudinal_wrench(
            unwrapped, 0.0, wrench_profile.application_height_m
        )
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
            "vx_kp": args.controller_vx_kp,
            "wz_kp": args.controller_wz_kp,
            "wz_ki": args.controller_wz_ki,
            "wz_feedforward": args.controller_wz_feedforward,
            "wheel_difference_kp": args.controller_wheel_difference_kp,
            "limit_total_pitch_reference": (
                True if args.limit_total_pitch_reference else None
            ),
            "reset_opposing_vx_integral_on_directional_deficit": (
                True
                if args.reset_opposing_vx_integral_on_directional_deficit
                else None
            ),
            "vx_integral_reset_reference_deadband_mps": (
                args.vx_integral_reset_reference_deadband_mps
                if args.reset_opposing_vx_integral_on_directional_deficit
                else None
            ),
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
            "maximum_linear_velocity_mps": (
                args.tracking_maximum_linear_velocity_mps
            ),
            "minimum_progress_scale": args.tracking_minimum_progress_scale,
            "camera_recovery_error_start_m": (
                args.camera_recovery_error_start_m
            ),
            "camera_recovery_error_full_m": args.camera_recovery_error_full_m,
            "minimum_camera_recovery_scale": (
                args.minimum_camera_recovery_scale
            ),
        }.items()
        if value is not None
        and (
            name
            not in {
                "camera_recovery_error_start_m",
                "camera_recovery_error_full_m",
                "minimum_camera_recovery_scale",
            }
            or args.enable_camera_error_recovery_governor
        )
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
    nominal_base_progress_error_samples = []
    commanded_base_progress_error_samples = []
    selected_base_progress_error_samples = []
    camera_recovery_progress_samples = []
    camera_lever_arm_correction_norm_samples = []
    camera_lever_arm_raw_correction_norm_samples = []
    camera_lever_arm_saturated_samples = []
    position_errors = []
    attitude_errors_deg = []
    riser_errors = []
    proxy_errors_deg = []
    pitch_samples_deg = []
    saturated_actions = 0
    action_count = 0
    saturated_riser = 0
    riser_effort_count = 0
    riser_thermal_monitor = RiserMotorThermalMonitor()
    recovery_telemetry = RecoveryTelemetryAccumulator()
    velocity_feedback_telemetry = VelocityFeedbackTelemetryAccumulator()
    longitudinal_authority_telemetry = (
        LongitudinalAuthorityTelemetryAccumulator(
            reference_deadband_mps=(
                args.vx_integral_reset_reference_deadband_mps
            )
        )
    )
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
    policy_trace_observations = []
    policy_trace_actions = []
    policy_trace_commands = []
    policy_trace_wheel_actions = []
    policy_trace_elapsed_time = []
    policy_trace_phase_time = []
    policy_trace_position_errors = []
    policy_trace_attitude_errors = []
    policy_trace_base_states = []
    policy_trace_camera_positions = []
    policy_trace_pitch = []
    policy_trace_riser_positions = []
    policy_trace_proxy_positions = []
    shadow_teacher_raw_commands = []
    shadow_teacher_normalized_actions = []
    shadow_teacher_high_level_commands = []
    corrective_capture_observations = []
    corrective_capture_model_commands = []
    corrective_capture_requested_residual_commands = []
    corrective_capture_requested_normalized_actions = []
    corrective_capture_effective_residual_commands = []
    corrective_capture_effective_normalized_actions = []
    corrective_capture_residual_deltas = []
    corrective_capture_command_clipped = []
    corrective_capture_final_commands = []
    corrective_capture_elapsed_time = []
    corrective_capture_execution_time = []
    corrective_capture_source_time = []
    corrective_capture_amplitude_limited = []
    corrective_capture_slew_limited = []
    corrective_capture_perturbation_active = []
    previous_residual_action = np.zeros(3, dtype=np.float32)
    previous_corrective_teacher_residual = np.zeros(3, dtype=np.float64)
    corrective_teacher_telemetry = CorrectiveTeacherTelemetry()
    completed_steps = 0
    if runtime_heartbeat is not None:
        write_runtime_heartbeat(
            runtime_heartbeat,
            {
                "case": plan.case,
                "stage": "execution_pending",
                "completed_steps": 0,
                "maximum_steps": maximum_steps,
                "policy_hz": POLICY_HZ,
                "elapsed_s": 0.0,
                "phase_time_s": 0.0,
                "source_duration_s": source_duration_s,
                "execution_duration_s": execution_duration_s,
                "maximum_simulated_horizon_s": (
                    execution_duration_s * args.maximum_duration_scale
                ),
                "gate_result_written": False,
                "capture_outputs_enabled": any(
                    path is not None
                    for path in (
                        dataset_dir,
                        raw_teacher_dir,
                        policy_trace_dir,
                        shadow_teacher_trace_dir,
                        corrective_capture_dir,
                    )
                ),
            },
        )
    body_masses = robot.data.default_mass[0].to(unwrapped.device)
    kinematics = UrdfRiserCameraKinematics(
        PROJECT_ROOT
        / "assets_own/recomoProto2_two_wheel_riser/recomoProto2_two_wheel_riser.urdf"
    )
    previous_proxy_command = plan.proxy_gimbal_q[0].copy()
    if not hasattr(robot.data, "body_com_pos_w"):
        raise RuntimeError("Isaac articulation data does not expose body_com_pos_w")
    initialization_steps = 0
    initialization_completed = plan.initialization_time_s is None or len(
        plan.initialization_time_s
    ) == 0
    initialization_duration_s = (
        0.0
        if initialization_completed
        else float(plan.initialization_time_s[-1])
    )
    initialization_terminal_base_error_m = 0.0
    initialization_terminal_base_yaw_error_deg = 0.0
    initialization_terminal_riser_error_m = 0.0
    initialization_terminal_proxy_error_deg = 0.0
    initialization_thermal_monitor = RiserMotorThermalMonitor()
    initialization_saturated_actions = 0
    initialization_action_count = 0
    if not initialization_completed:
        initialization_step_limit = int(
            math.ceil(initialization_duration_s * POLICY_HZ)
        )
        for initialization_step in range(initialization_step_limit):
            initialization_elapsed_s = min(
                initialization_duration_s,
                (initialization_step + 1) / POLICY_HZ,
            )
            initialization_sample = interpolate_riser_initialization(
                plan, initialization_elapsed_s
            )
            root_position = robot.data.root_pos_w[0].detach().cpu().numpy()
            root_quaternion = robot.data.root_quat_w[0].detach().cpu().numpy()
            actual_base = np.array(
                [
                    root_position[0],
                    root_position[1],
                    yaw_from_quaternion_wxyz(root_quaternion),
                ]
            )
            vx_ref, wz_ref, _ = bounded_base_references(
                initialization_sample.base_xy_yaw,
                actual_base,
                initialization_sample.feedforward_v_mps,
                initialization_sample.feedforward_wz_rad_s,
                tracking_cfg,
            )
            riser_target = torch.tensor(
                [[initialization_sample.riser_q]],
                dtype=torch.float32,
                device=unwrapped.device,
            )
            riser_velocity_target = torch.tensor(
                [[initialization_sample.feedforward_riser_velocity_mps]],
                dtype=torch.float32,
                device=unwrapped.device,
            )
            actual_proxy = robot.data.joint_pos[0, proxy_ids].detach().cpu().numpy()
            proxy_command = initialization_sample.proxy_gimbal_q.copy()
            proxy_command[2] = nearest_equivalent_angle(
                proxy_command[2], actual_proxy[2]
            )
            proxy_target = torch.as_tensor(
                proxy_command[None, :],
                dtype=torch.float32,
                device=unwrapped.device,
            )
            robot.set_joint_position_target(riser_target, joint_ids=[riser_id])
            robot.set_joint_velocity_target(
                riser_velocity_target, joint_ids=[riser_id]
            )
            robot.set_joint_position_target(proxy_target, joint_ids=proxy_ids)
            if not args.disable_semantic_proxy_state_adapter:
                proxy_velocity_state = torch.as_tensor(
                    initialization_sample.feedforward_proxy_velocity_rad_s[None, :],
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
            unwrapped.vx_ref.fill_(vx_ref)
            unwrapped.wz_ref.fill_(wz_ref)
            if initialization_step % control_interval == 0:
                initialization_outer_vx = (
                    np.array(
                        [float(unwrapped._state_terms()["vx"][0].item())]
                    )
                    if args.use_root_velocity_outer_feedback
                    else None
                )
                action, controller_state, _ = cascaded_lqr_action(
                    current_states,
                    np.array([vx_ref]),
                    np.array([wz_ref]),
                    gain,
                    controller_state,
                    control_dt=control_interval / POLICY_HZ,
                    config=controller_cfg,
                    pitch_bias_override_rad=np.array([com_pitch_bias]),
                    outer_vx_feedback_m_s=initialization_outer_vx,
                )
                action = action.astype(np.float32)
            obs, _, terminated, truncated, _ = env.step(
                torch.as_tensor(action, device=unwrapped.device)
            )
            initialization_saturated_actions += int(
                np.count_nonzero(
                    np.abs(action) >= controller_cfg.action_limit - 1e-6
                )
            )
            initialization_action_count += action.size
            if hasattr(robot.data, "applied_torque"):
                initialization_riser_effort = abs(
                    float(robot.data.applied_torque[0, riser_id].item())
                )
                initialization_thermal_monitor.step(
                    initialization_riser_effort, 1.0 / POLICY_HZ
                )
            current_states = (
                obs["policy"][:, : len(LQR_STATE_NAMES)].detach().cpu().numpy()
            )
            initialization_steps = initialization_step + 1
            if bool((terminated | truncated)[0].item()):
                raise RuntimeError(
                    "initialization pre-roll terminated before source phase zero"
                )
            if initialization_elapsed_s >= initialization_duration_s:
                initialization_completed = True
                break

        root_position = robot.data.root_pos_w[0].detach().cpu().numpy()
        root_quaternion = robot.data.root_quat_w[0].detach().cpu().numpy()
        terminal_base = np.array(
            [
                root_position[0],
                root_position[1],
                yaw_from_quaternion_wxyz(root_quaternion),
            ]
        )
        initialization_terminal_base_error_m = float(
            np.linalg.norm(terminal_base[:2] - plan.base_xy_yaw[0, :2])
        )
        initialization_terminal_base_yaw_error_deg = math.degrees(
            abs(
                math.atan2(
                    math.sin(terminal_base[2] - plan.base_xy_yaw[0, 2]),
                    math.cos(terminal_base[2] - plan.base_xy_yaw[0, 2]),
                )
            )
        )
        initialization_terminal_riser_error_m = abs(
            float(robot.data.joint_pos[0, riser_id].item()) - plan.riser_q[0]
        )
        terminal_proxy = robot.data.joint_pos[0, proxy_ids].detach().cpu().numpy()
        terminal_proxy_error = plan.proxy_gimbal_q[0] - terminal_proxy
        terminal_proxy_error[2] = continuous_joint_error(
            plan.proxy_gimbal_q[0, 2], terminal_proxy[2]
        )
        initialization_terminal_proxy_error_deg = float(
            np.max(np.abs(np.rad2deg(terminal_proxy_error)))
        )
        previous_proxy_command = plan.proxy_gimbal_q[0].copy()

    initialization_source_metrics_clean = not any(
        (
            position_errors,
            attitude_errors_deg,
            riser_errors,
            proxy_errors_deg,
            raw_residual_commands,
            normalized_residual_labels,
            dataset_observations,
        )
    )
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
        commanded_base_xy_yaw, camera_lever_arm_diagnostics = (
            bounded_camera_lever_arm_base_target(
                sample.base_xy_yaw,
                actual_base,
                sample.target_position_world_m,
                actual_camera_position_pre,
                gain=(
                    args.camera_lever_arm_compensation_gain
                    if args.enable_camera_lever_arm_compensation
                    else 0.0
                ),
                maximum_correction_m=(
                    args.maximum_camera_lever_arm_correction_m
                ),
            )
        )
        camera_lever_arm_correction_norm_samples.append(
            camera_lever_arm_diagnostics["correction_norm_m"]
        )
        camera_lever_arm_raw_correction_norm_samples.append(
            camera_lever_arm_diagnostics["raw_correction_norm_m"]
        )
        camera_lever_arm_saturated_samples.append(
            camera_lever_arm_diagnostics["saturated"]
        )
        vx_ref, wz_ref, base_tracking_diagnostics = bounded_base_references(
            commanded_base_xy_yaw,
            actual_base,
            phase_feedforward_v_mps,
            phase_feedforward_wz_rad_s,
            tracking_cfg,
        )
        legacy_wz_ref = float(
            np.clip(
                phase_feedforward_wz_rad_s
                + tracking_cfg.yaw_kp
                * base_tracking_diagnostics["yaw_error_rad"]
                + tracking_cfg.cross_track_kp
                * base_tracking_diagnostics["feedforward_direction"]
                * base_tracking_diagnostics["cross_track_error_m"],
                -tracking_cfg.maximum_yaw_rate_radps,
                tracking_cfg.maximum_yaw_rate_radps,
            )
        )
        recovery_telemetry.step(
            recovery_blend=base_tracking_diagnostics["direction_recovery_blend"],
            motion_direction=base_tracking_diagnostics["motion_direction"],
            feedback_motion_direction=base_tracking_diagnostics[
                "feedback_motion_direction"
            ],
            candidate_yaw_rate_rad_s=wz_ref,
            legacy_yaw_rate_rad_s=legacy_wz_ref,
            maximum_yaw_rate_rad_s=tracking_cfg.maximum_yaw_rate_radps,
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
        model_based_vx_ref = vx_ref
        model_based_wz_ref = wz_ref
        model_based_riser_target = sample.riser_q
        normalized_residual_label = normalize_residual_command(raw_residual_command)
        shadow_teacher_normalized_action = (
            raw_residual_command / args.residual_action_scales
        )
        shadow_teacher_high_level_command = np.array(
            [vx_ref, wz_ref, sample.riser_q], dtype=np.float64
        )
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
        corrective_output = None
        if residual_policy is None and not zero_policy_action:
            applied_residual_action = (
                teacher_residual_action
                if dataset_dir is not None
                else np.zeros(3, dtype=np.float32)
            )
            commanded_riser_target = sample.riser_q
        else:
            if corrective_teacher_config is not None:
                corrective_output = build_corrective_teacher_action(
                    executed_observation[
                        [
                            OBSERVATION_INDEX["camera_target_longitudinal_error_m"],
                            OBSERVATION_INDEX["camera_target_lateral_error_m"],
                            OBSERVATION_INDEX["camera_target_vertical_error_m"],
                        ]
                    ],
                    previous_corrective_teacher_residual,
                    dt_s=1.0 / POLICY_HZ,
                    config=corrective_teacher_config,
                )
                applied_residual_action = corrective_output.normalized_action.astype(
                    np.float32
                )
                previous_corrective_teacher_residual = (
                    corrective_output.applied_residual.copy()
                )
                corrective_teacher_telemetry.step(corrective_output)
            elif zero_policy_action:
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
            if args.policy_command_base == "model_based_planner":
                vx_ref, wz_ref, commanded_riser_target = (
                    apply_model_based_policy_residual(
                        model_based_vx_ref,
                        model_based_wz_ref,
                        model_based_riser_target,
                        applied_residual_action,
                        action_scales=args.residual_action_scales,
                        maximum_linear_velocity_m_s=(
                            tracking_cfg.maximum_linear_velocity_mps
                        ),
                        maximum_yaw_rate_rad_s=tracking_cfg.maximum_yaw_rate_radps,
                        riser_bounds_m=(
                            kinematics.riser_lower,
                            kinematics.riser_upper,
                        ),
                    )
                )
            else:
                vx_ref, wz_ref, commanded_riser_target = apply_residual_action(
                    phase_feedforward_v_mps,
                    phase_feedforward_wz_rad_s,
                    actual_riser_pre,
                    applied_residual_action,
                    action_scales=args.residual_action_scales,
                    maximum_linear_velocity_m_s=(
                        tracking_cfg.maximum_linear_velocity_mps
                    ),
                    maximum_yaw_rate_rad_s=tracking_cfg.maximum_yaw_rate_radps,
                    riser_bounds_m=(kinematics.riser_lower, kinematics.riser_upper),
                )
        applied_residual_actions.append(applied_residual_action.copy())
        if dataset_dir is not None or raw_teacher_dir is not None:
            dataset_observations.append(executed_observation)
            if dataset_dir is not None:
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
        controller_updated = step % control_interval == 0
        if controller_updated:
            outer_vx_feedback = (
                np.array([float(unwrapped._state_terms()["vx"][0].item())])
                if args.use_root_velocity_outer_feedback
                else None
            )
            action, controller_state, controller_diagnostics = cascaded_lqr_action(
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
                outer_vx_feedback_m_s=outer_vx_feedback,
            )
            action = action.astype(np.float32)
        dataset_baseline_actions.append(action[0].copy())
        saturated_actions += int(
            np.count_nonzero(np.abs(action) >= controller_cfg.action_limit - 1e-6)
        )
        action_count += action.size
        perturbation_force_body_x_n = perturbation_runtime.command(
            step=step,
            phase_time_s=phase_time_s,
        )
        if corrective_capture_dir is not None:
            if corrective_output is None or corrective_capture_admission is None:
                raise RuntimeError("corrective capture lost its admitted teacher output")
            model_command = np.array(
                [
                    model_based_vx_ref,
                    model_based_wz_ref,
                    model_based_riser_target,
                ],
                dtype=np.float64,
            )
            corrective_capture_observations.append(executed_observation.copy())
            corrective_capture_model_commands.append(model_command)
            requested_residual = corrective_output.applied_residual.copy()
            requested_normalized = corrective_output.normalized_action.copy()
            final_command = np.array(
                [vx_ref, wz_ref, commanded_riser_target], dtype=np.float64
            )
            effective_residual = final_command - model_command
            effective_normalized = effective_residual / args.residual_action_scales
            residual_delta = effective_residual - requested_residual
            corrective_capture_requested_residual_commands.append(
                requested_residual
            )
            corrective_capture_requested_normalized_actions.append(
                requested_normalized
            )
            corrective_capture_effective_residual_commands.append(
                effective_residual
            )
            corrective_capture_effective_normalized_actions.append(
                effective_normalized
            )
            corrective_capture_residual_deltas.append(residual_delta)
            corrective_capture_command_clipped.append(
                np.abs(residual_delta) > 2e-7
            )
            corrective_capture_final_commands.append(final_command)
            corrective_capture_elapsed_time.append(elapsed_s)
            corrective_capture_execution_time.append(phase_time_s)
            corrective_capture_source_time.append(
                float(np.interp(phase_time_s, plan.time_s, plan.source_time_s))
            )
            corrective_capture_amplitude_limited.append(
                corrective_output.amplitude_limited.copy()
            )
            corrective_capture_slew_limited.append(
                corrective_output.slew_limited.copy()
            )
            corrective_capture_perturbation_active.append(
                abs(perturbation_force_body_x_n) > 1e-12
            )
        if wrench_profile is not None:
            set_base_longitudinal_wrench(
                unwrapped,
                perturbation_force_body_x_n,
                wrench_profile.application_height_m,
            )
        obs, _, terminated, truncated, _ = env.step(
            torch.as_tensor(action, device=unwrapped.device)
        )
        current_states = (
            obs["policy"][:, : len(LQR_STATE_NAMES)].detach().cpu().numpy()
        )
        state = unwrapped._state_terms()
        root_velocity_mps = float(state["vx"][0].item())
        wheel_velocity_mps = WHEEL_RADIUS_M * float(
            state["mean_wheel_velocity"][0].item()
        )
        effective_velocity_reference_mps = float(
            controller_diagnostics["effective_vx_ref"][0]
        )
        pitch_reference_rad = float(
            controller_diagnostics["pitch_reference"][0]
        )
        total_pitch_reference_rad = float(
            controller_diagnostics["total_pitch_reference"][0]
        )
        applied_pitch_bias_rad = float(
            controller_diagnostics["applied_pitch_bias"][0]
        )
        common_action = float(action[0, 0])
        velocity_feedback_telemetry.step(
            root_velocity_mps=root_velocity_mps,
            wheel_velocity_mps=wheel_velocity_mps,
            effective_reference_mps=effective_velocity_reference_mps,
            pitch_reference_rad=pitch_reference_rad,
            total_pitch_reference_rad=total_pitch_reference_rad,
            applied_pitch_bias_rad=applied_pitch_bias_rad,
            common_action=common_action,
        )
        common_contributions = controller_diagnostics[
            "common_action_state_contributions"
        ][0]
        longitudinal_authority_telemetry.step(
            controller_updated=controller_updated,
            effective_reference_mps=effective_velocity_reference_mps,
            previous_effective_reference_mps=float(
                controller_diagnostics["previous_effective_vx_ref"][0]
            ),
            wheel_velocity_mps=wheel_velocity_mps,
            pitch_rad=float(current_states[0, 0]),
            pitch_rate_rad_s=float(current_states[0, 1]),
            total_pitch_reference_rad=total_pitch_reference_rad,
            total_pitch_limit_rad=controller_cfg.pitch_reference_limit_rad,
            common_action=common_action,
            vx_integral_before=float(
                controller_diagnostics["vx_integral_before"][0]
            ),
            vx_integral_after=float(
                controller_diagnostics["vx_integral_after"][0]
            ),
            integral_reset=bool(
                controller_updated
                and controller_diagnostics[
                    "opposing_vx_integral_deficit_reset"
                ][0]
            ),
            pitch_contribution=float(common_contributions[0]),
            pitch_rate_contribution=float(common_contributions[1]),
            wheel_velocity_contribution=float(common_contributions[3]),
        )
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
        commanded_base_xy_error = float(
            np.linalg.norm(actual_base_post[:2] - commanded_base_xy_yaw[:2])
        )
        progress_base_error = select_progress_governor_base_error(
            base_xy_error,
            commanded_base_xy_error,
            use_commanded_base_target=args.use_commanded_base_progress_error,
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
        if policy_trace_dir is not None or shadow_teacher_trace_dir is not None:
            policy_trace_observations.append(executed_observation.copy())
            policy_trace_actions.append(applied_residual_action.copy())
            policy_trace_commands.append(
                [vx_ref, wz_ref, commanded_riser_target]
            )
            policy_trace_wheel_actions.append(action[0].copy())
            policy_trace_elapsed_time.append(elapsed_s)
            policy_trace_phase_time.append(phase_time_s)
            policy_trace_position_errors.append(position_error)
            policy_trace_attitude_errors.append(attitude_error_deg)
            policy_trace_base_states.append(actual_base_post.copy())
            policy_trace_camera_positions.append(actual_cam_position.copy())
            policy_trace_pitch.append(pitch_deg)
            policy_trace_riser_positions.append(actual_riser)
            policy_trace_proxy_positions.append(actual_proxy.copy())
        if shadow_teacher_trace_dir is not None:
            shadow_teacher_raw_commands.append(raw_residual_command.copy())
            shadow_teacher_normalized_actions.append(
                shadow_teacher_normalized_action.copy()
            )
            shadow_teacher_high_level_commands.append(
                shadow_teacher_high_level_command.copy()
            )
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
            riser_thermal_monitor.step(riser_effort, 1.0 / POLICY_HZ)
            saturated_proxy += int(torch.count_nonzero(proxy_effort >= 10.0 - 1e-3))
            proxy_axis_saturated += proxy_effort_np >= 10.0 - 1e-3
            proxy_effort_count += len(proxy_ids)

        progress_scale = 1.0
        camera_recovery_progress_scale = 1.0
        if not args.disable_phase_governor:
            if args.enable_camera_error_recovery_governor:
                camera_recovery_progress_scale = (
                    bounded_camera_recovery_progress_scale(
                        position_error,
                        bool(camera_lever_arm_diagnostics["saturated"]),
                        tracking_cfg,
                    )
                )
            progress_scale = min(
                bounded_progress_scale(
                    progress_base_error,
                    position_error,
                    tracking_cfg,
                ),
                balance_progress_scale(abs(float(state["pitch"][0].item()))),
                camera_recovery_progress_scale,
            )
        progress_samples.append(progress_scale)
        nominal_base_progress_error_samples.append(base_xy_error)
        commanded_base_progress_error_samples.append(commanded_base_xy_error)
        selected_base_progress_error_samples.append(progress_base_error)
        camera_recovery_progress_samples.append(camera_recovery_progress_scale)
        if step % 200 == 0 or phase_time_s >= execution_duration_s:
            trace.append(
                {
                    "step": step + 1,
                    "elapsed_s": elapsed_s,
                    "phase_time_s": phase_time_s,
                    "progress_scale": progress_scale,
                    "camera_recovery_progress_scale": (
                        camera_recovery_progress_scale
                    ),
                    "camera_recovery_active": (
                        camera_recovery_progress_scale < 1.0 - 1e-12
                    ),
                    "position_error_m": position_error,
                    "attitude_error_deg": attitude_error_deg,
                    "pitch_deg": pitch_deg,
                    "riser_error_m": riser_error,
                    "riser_thermal_load": riser_thermal_monitor.thermal_load,
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
                    "commanded_base_xy_error_m": commanded_base_xy_error,
                    "progress_base_error_m": progress_base_error,
                    "progress_base_error_source": (
                        progress_governor_base_error_source()
                    ),
                    "base_yaw_error_deg": math.degrees(base_yaw_error),
                    "actual_base_xy_yaw": actual_base_post.tolist(),
                    "target_base_xy_yaw": sample.base_xy_yaw.tolist(),
                    "commanded_base_xy_yaw": commanded_base_xy_yaw.tolist(),
                    "camera_lever_arm_target_xy_m": (
                        camera_lever_arm_diagnostics["target_lever_xy_m"].tolist()
                    ),
                    "camera_lever_arm_actual_xy_m": (
                        camera_lever_arm_diagnostics["actual_lever_xy_m"].tolist()
                    ),
                    "camera_lever_arm_error_xy_m": (
                        camera_lever_arm_diagnostics["lever_error_xy_m"].tolist()
                    ),
                    "camera_lever_arm_correction_xy_m": (
                        camera_lever_arm_diagnostics["correction_xy_m"].tolist()
                    ),
                    "camera_lever_arm_correction_saturated": (
                        camera_lever_arm_diagnostics["saturated"]
                    ),
                    "camera_position_error_xyz_m": (
                        camera_position_error_vector.tolist()
                    ),
                    "actual_camera_position_world_m": actual_cam_position.tolist(),
                    "target_camera_position_world_m": (
                        sample.target_position_world_m.tolist()
                    ),
                    "actual_yaw_rate_rad_s": float(state["yaw_rate"][0].item()),
                    "actual_root_velocity_mps": root_velocity_mps,
                    "wheel_derived_velocity_mps": wheel_velocity_mps,
                    "root_wheel_velocity_mismatch_mps": (
                        root_velocity_mps - wheel_velocity_mps
                    ),
                    "effective_velocity_reference_mps": (
                        effective_velocity_reference_mps
                    ),
                    "pitch_reference_rad": pitch_reference_rad,
                    "total_pitch_reference_rad": total_pitch_reference_rad,
                    "applied_pitch_bias_rad": applied_pitch_bias_rad,
                    "common_wheel_action": common_action,
                    "phase_feedforward_v_mps": phase_feedforward_v_mps,
                    "phase_feedforward_wz_rad_s": phase_feedforward_wz_rad_s,
                    "model_based_vx_reference_mps": model_based_vx_ref,
                    "model_based_wz_reference_rad_s": model_based_wz_ref,
                    "model_based_riser_target_m": model_based_riser_target,
                    "applied_policy_residual_action": (
                        applied_residual_action.tolist()
                    ),
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
                    "perturbation_active": (
                        abs(perturbation_force_body_x_n) > 0.0
                    ),
                    "perturbation_force_body_x_n": (
                        perturbation_force_body_x_n
                    ),
                }
            )
        completed_steps = step + 1
        if runtime_heartbeat is not None and (
            completed_steps == 1
            or completed_steps % runtime_heartbeat_interval_steps == 0
            or bool((terminated | truncated)[0].item())
            or phase_time_s >= execution_duration_s
        ):
            write_runtime_heartbeat(
                runtime_heartbeat,
                {
                    "case": plan.case,
                    "stage": "execution",
                    "completed_steps": completed_steps,
                    "maximum_steps": maximum_steps,
                    "policy_hz": POLICY_HZ,
                    "elapsed_s": elapsed_s,
                    "phase_time_s": phase_time_s,
                    "source_duration_s": source_duration_s,
                    "execution_duration_s": execution_duration_s,
                    "maximum_simulated_horizon_s": (
                        execution_duration_s * args.maximum_duration_scale
                    ),
                    "progress_scale": progress_scale,
                    "tracking_state": {
                        "position_error_m": position_error,
                        "attitude_error_deg": attitude_error_deg,
                        "base_xy_error_m": base_xy_error,
                        "commanded_base_xy_error_m": commanded_base_xy_error,
                        "base_yaw_error_deg": math.degrees(base_yaw_error),
                        "riser_error_m": riser_error,
                        "proxy_error_max_deg": float(np.max(proxy_error_deg)),
                    },
                    "safety_state": {
                        "pitch_deg": pitch_deg,
                        "peak_position_error_m": float(np.max(position_errors)),
                        "peak_attitude_error_deg": float(
                            np.max(attitude_errors_deg)
                        ),
                        "peak_pitch_deg": float(np.max(np.abs(pitch_samples_deg))),
                        "action_saturation_ratio": (
                            saturated_actions / action_count
                            if action_count
                            else 0.0
                        ),
                        "riser_saturation_ratio": (
                            saturated_riser / riser_effort_count
                            if riser_effort_count
                            else 0.0
                        ),
                        "proxy_saturation_ratio": (
                            saturated_proxy / proxy_effort_count
                            if proxy_effort_count
                            else 0.0
                        ),
                        "termination_pending": bool(
                            (terminated | truncated)[0].item()
                        ),
                    },
                    "gate_result_written": False,
                    "capture_outputs_enabled": any(
                        path is not None
                        for path in (
                            dataset_dir,
                            raw_teacher_dir,
                            policy_trace_dir,
                            shadow_teacher_trace_dir,
                            corrective_capture_dir,
                        )
                    ),
                },
            )
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

    if wrench_profile is not None:
        set_base_longitudinal_wrench(
            unwrapped, 0.0, wrench_profile.application_height_m
        )
        perturbation_runtime.mark_released()
    perturbation_telemetry = perturbation_runtime.summary()
    perturbation_contract_checks = perturbation_runtime.contract_checks()
    corrective_teacher_telemetry_summary = corrective_teacher_telemetry.summary(
        enabled=corrective_teacher_config is not None
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
    recovery_telemetry_summary = recovery_telemetry.summary()
    velocity_feedback_telemetry_summary = velocity_feedback_telemetry.summary()
    longitudinal_authority_telemetry_summary = (
        longitudinal_authority_telemetry.summary()
    )
    recovery_telemetry_observed = (
        recovery_telemetry_summary["schema"] == RECOVERY_TELEMETRY_SCHEMA
        and recovery_telemetry_summary["policy_rate_sample_count"]
        == completed_steps
    )
    velocity_feedback_telemetry_observed = (
        velocity_feedback_telemetry_summary["schema"]
        == VELOCITY_FEEDBACK_TELEMETRY_SCHEMA
        and velocity_feedback_telemetry_summary["policy_rate_sample_count"]
        == completed_steps
    )
    expected_controller_updates = (
        completed_steps + control_interval - 1
    ) // control_interval
    longitudinal_authority_telemetry_observed = (
        longitudinal_authority_telemetry_summary["schema"]
        == LONGITUDINAL_AUTHORITY_TELEMETRY_SCHEMA
        and longitudinal_authority_telemetry_summary[
            "policy_rate_sample_count"
        ]
        == completed_steps
        and longitudinal_authority_telemetry_summary[
            "controller_update_count"
        ]
        == expected_controller_updates
        and (
            args.reset_opposing_vx_integral_on_directional_deficit
            or longitudinal_authority_telemetry_summary[
                "integral_reset_count"
            ]
            == 0
        )
    )
    camera_lever_arm_telemetry_observed = (
        len(camera_lever_arm_correction_norm_samples) == completed_steps
        and len(camera_lever_arm_raw_correction_norm_samples) == completed_steps
        and len(camera_lever_arm_saturated_samples) == completed_steps
    )
    camera_recovery_telemetry_observed = (
        len(camera_recovery_progress_samples) == completed_steps
    )
    progress_base_error_summary = summarize_progress_governor_base_error(
        np.asarray(nominal_base_progress_error_samples),
        np.asarray(commanded_base_progress_error_samples),
        np.asarray(selected_base_progress_error_samples),
        use_commanded_base_target=args.use_commanded_base_progress_error,
        maximum_command_correction_m=(
            args.maximum_camera_lever_arm_correction_m
        ),
        expected_sample_count=completed_steps,
    )
    progress_base_error_telemetry_observed = progress_base_error_summary[
        "progress_base_error_telemetry_observed"
    ]
    dynamic_checks = {
        "initialization_completed": initialization_completed,
        "initialization_action_saturation_bounded": (
            initialization_saturated_actions
            / max(initialization_action_count, 1)
            <= args.maximum_saturation_ratio
        ),
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
    thermal_checks = {
        "initialization_riser_thermal_force_observed": (
            initialization_steps == 0
            or initialization_thermal_monitor.sample_count == initialization_steps
        ),
        "initialization_riser_thermal_load_bounded": (
            initialization_thermal_monitor.maximum_thermal_load <= 1.0 + 1e-9
        ),
        "initialization_riser_peak_force_bounded": (
            initialization_thermal_monitor.peak_force_violation_count == 0
        ),
        "riser_thermal_force_observed": (
            riser_thermal_monitor.sample_count == completed_steps
        ),
        "riser_thermal_load_bounded": (
            riser_thermal_monitor.maximum_thermal_load <= 1.0 + 1e-9
        ),
        "riser_peak_force_bounded": (
            riser_thermal_monitor.peak_force_violation_count == 0
        ),
    }
    controller_evidence_checks = {
        "initialization_source_metrics_clean": (
            initialization_source_metrics_clean
        ),
        "velocity_feedback_telemetry_observed": (
            velocity_feedback_telemetry_observed
        ),
        "longitudinal_authority_telemetry_observed": (
            longitudinal_authority_telemetry_observed
        ),
        "camera_lever_arm_telemetry_observed": (
            camera_lever_arm_telemetry_observed
        ),
        "camera_recovery_telemetry_observed": (
            camera_recovery_telemetry_observed
        ),
        "progress_base_error_telemetry_observed": (
            progress_base_error_telemetry_observed
        ),
        "progress_base_error_selected_source_matches": (
            progress_base_error_summary[
                "progress_base_error_selected_source_matches"
            ]
        ),
        "progress_base_error_command_delta_bounded": (
            progress_base_error_summary[
                "progress_base_error_command_delta_bounded"
            ]
        ),
    }
    checks = (
        dynamic_checks
        | thermal_checks
        | controller_evidence_checks
        | perturbation_contract_checks
    )
    dynamic_quality_passed = all(dynamic_checks.values())
    thermal_admission_passed = all(thermal_checks.values())
    controller_evidence_passed = all(controller_evidence_checks.values())
    perturbation_contract_passed = all(
        perturbation_contract_checks.values()
    )
    case_admission_passed = (
        dynamic_quality_passed
        and thermal_admission_passed
        and controller_evidence_passed
        and perturbation_contract_passed
    )
    residual_label_envelope_ok = all(
        residual_action_envelope_passed(value)
        for value in normalized_residual_values
    )
    dataset_path = None
    if dataset_dir is not None and case_admission_passed:
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
    raw_teacher_path = None
    if raw_teacher_dir is not None and case_admission_passed:
        raw_teacher_path = (
            raw_teacher_dir
            / f"case_{plan.case:04d}_executed_raw_teacher_v1.npz"
        )
        count = len(dataset_observations)
        save_raw_teacher_case(
            raw_teacher_path,
            plan.case,
            {
                "observations": np.asarray(dataset_observations, dtype=np.float32),
                "raw_residual_commands": np.asarray(
                    raw_residual_commands, dtype=np.float32
                ),
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
    corrective_capture_path = None
    if corrective_capture_dir is not None and case_admission_passed:
        if corrective_capture_admission is None:
            raise RuntimeError("corrective capture admission disappeared")
        corrective_capture_path = (
            corrective_capture_dir
            / f"case_{plan.case:04d}_corrective_teacher_capture_v2.npz"
        )
        count = len(corrective_capture_observations)
        save_corrective_capture(
            corrective_capture_path,
            {
                "observations": np.asarray(
                    corrective_capture_observations, dtype=np.float32
                ),
                "model_based_commands": np.asarray(
                    corrective_capture_model_commands, dtype=np.float32
                ),
                "requested_corrective_residual_commands": np.asarray(
                    corrective_capture_requested_residual_commands,
                    dtype=np.float32,
                ),
                "requested_corrective_normalized_actions": np.asarray(
                    corrective_capture_requested_normalized_actions,
                    dtype=np.float32,
                ),
                "effective_corrective_residual_commands": np.asarray(
                    corrective_capture_effective_residual_commands,
                    dtype=np.float32,
                ),
                "effective_corrective_normalized_actions": np.asarray(
                    corrective_capture_effective_normalized_actions,
                    dtype=np.float32,
                ),
                "requested_vs_effective_residual_delta": np.asarray(
                    corrective_capture_residual_deltas, dtype=np.float32
                ),
                "command_clipped": np.asarray(
                    corrective_capture_command_clipped, dtype=bool
                ),
                "final_high_level_commands": np.asarray(
                    corrective_capture_final_commands, dtype=np.float32
                ),
                "case_ids": np.full(count, plan.case, dtype=np.int16),
                "elapsed_time_s": np.asarray(
                    corrective_capture_elapsed_time, dtype=np.float64
                ),
                "execution_time_s": np.asarray(
                    corrective_capture_execution_time, dtype=np.float64
                ),
                "source_time_s": np.asarray(
                    corrective_capture_source_time, dtype=np.float64
                ),
                "initialization_mask": np.zeros(count, dtype=bool),
                "amplitude_limited": np.asarray(
                    corrective_capture_amplitude_limited, dtype=bool
                ),
                "slew_limited": np.asarray(
                    corrective_capture_slew_limited, dtype=bool
                ),
                "perturbation_active": np.asarray(
                    corrective_capture_perturbation_active, dtype=bool
                ),
                "sample_plan_sha256": np.full(count, plan_sha256),
                "sample_runtime_commit": np.full(
                    count, corrective_capture_admission["runtime_commit"]
                ),
            },
            source_duration_s=source_duration_s,
            execution_duration_s=execution_duration_s,
            plan_sha256=plan_sha256,
            runtime_commit=str(corrective_capture_admission["runtime_commit"]),
            corrective_profile_sha256=str(
                corrective_capture_admission["corrective_profile_sha256"]
            ),
            paired_final_status_sha256=str(
                corrective_capture_admission["paired_final_status_sha256"]
            ),
            case=int(corrective_capture_admission["case"]),
            split=args.corrective_teacher_capture_split,
        )
    policy_trace_path = None
    if policy_trace_dir is not None:
        policy_trace_path = (
            policy_trace_dir / f"case_{plan.case:04d}_policy_trace_v1.npz"
        )
        count = len(policy_trace_observations)
        save_policy_trace(
            policy_trace_path,
            plan.case,
            {
                "observations": np.asarray(
                    policy_trace_observations, dtype=np.float32
                ),
                "applied_residual_actions": np.asarray(
                    policy_trace_actions, dtype=np.float32
                ),
                "final_high_level_commands": np.asarray(
                    policy_trace_commands, dtype=np.float32
                ),
                "baseline_wheel_actions": np.asarray(
                    policy_trace_wheel_actions, dtype=np.float32
                ),
                "case_ids": np.full(count, plan.case, dtype=np.int16),
                "elapsed_time_s": np.asarray(
                    policy_trace_elapsed_time, dtype=np.float64
                ),
                "phase_time_s": np.asarray(
                    policy_trace_phase_time, dtype=np.float64
                ),
                "post_step_position_error_m": np.asarray(
                    policy_trace_position_errors, dtype=np.float32
                ),
                "post_step_attitude_error_deg": np.asarray(
                    policy_trace_attitude_errors, dtype=np.float32
                ),
                "post_step_base_xy_yaw": np.asarray(
                    policy_trace_base_states, dtype=np.float32
                ),
                "post_step_camera_position_world_m": np.asarray(
                    policy_trace_camera_positions, dtype=np.float32
                ),
                "post_step_pitch_deg": np.asarray(
                    policy_trace_pitch, dtype=np.float32
                ),
                "post_step_riser_position_m": np.asarray(
                    policy_trace_riser_positions, dtype=np.float32
                ),
                "post_step_proxy_position_rad": np.asarray(
                    policy_trace_proxy_positions, dtype=np.float32
                ),
            },
        )
    shadow_teacher_trace_path = None
    if shadow_teacher_trace_dir is not None:
        shadow_teacher_trace_path = (
            shadow_teacher_trace_dir
            / f"case_{plan.case:04d}_shadow_teacher_trace_v1.npz"
        )
        count = len(policy_trace_observations)
        save_shadow_teacher_trace(
            shadow_teacher_trace_path,
            plan.case,
            {
                "observations": np.asarray(
                    policy_trace_observations, dtype=np.float32
                ),
                "applied_residual_actions": np.asarray(
                    policy_trace_actions, dtype=np.float32
                ),
                "final_high_level_commands": np.asarray(
                    policy_trace_commands, dtype=np.float32
                ),
                "baseline_wheel_actions": np.asarray(
                    policy_trace_wheel_actions, dtype=np.float32
                ),
                "case_ids": np.full(count, plan.case, dtype=np.int16),
                "elapsed_time_s": np.asarray(
                    policy_trace_elapsed_time, dtype=np.float64
                ),
                "phase_time_s": np.asarray(
                    policy_trace_phase_time, dtype=np.float64
                ),
                "post_step_position_error_m": np.asarray(
                    policy_trace_position_errors, dtype=np.float32
                ),
                "post_step_attitude_error_deg": np.asarray(
                    policy_trace_attitude_errors, dtype=np.float32
                ),
                "post_step_base_xy_yaw": np.asarray(
                    policy_trace_base_states, dtype=np.float32
                ),
                "post_step_camera_position_world_m": np.asarray(
                    policy_trace_camera_positions, dtype=np.float32
                ),
                "post_step_pitch_deg": np.asarray(
                    policy_trace_pitch, dtype=np.float32
                ),
                "post_step_riser_position_m": np.asarray(
                    policy_trace_riser_positions, dtype=np.float32
                ),
                "post_step_proxy_position_rad": np.asarray(
                    policy_trace_proxy_positions, dtype=np.float32
                ),
                "shadow_teacher_raw_residual_commands": np.asarray(
                    shadow_teacher_raw_commands, dtype=np.float32
                ),
                "shadow_teacher_normalized_residual_actions": np.asarray(
                    shadow_teacher_normalized_actions, dtype=np.float32
                ),
                "shadow_teacher_high_level_commands": np.asarray(
                    shadow_teacher_high_level_commands, dtype=np.float32
                ),
            },
            action_scales=args.residual_action_scales,
            visited_state_source=(
                "deterministic_controller"
                if residual_policy is None
                else "learned_policy"
            ),
        )
    progress_hold_summary = summarize_progress_hold(
        np.asarray(progress_samples, dtype=np.float64)
    )
    return {
        "case": plan.case,
        "planning_strategy": plan.planning_strategy,
        "source_duration_s": source_duration_s,
        "execution_duration_s": execution_duration_s,
        "maximum_duration_scale": args.maximum_duration_scale,
        "outer_velocity_feedback_source": (
            "root_link_vx"
            if args.use_root_velocity_outer_feedback
            else "wheel_derived_vx"
        ),
        "maximum_runtime_s": execution_duration_s * args.maximum_duration_scale,
        "completed_phase_time_s": phase_time_s,
        "completed_steps": completed_steps,
        "initialization_duration_s": initialization_duration_s,
        "initialization_steps": initialization_steps,
        "initialization_completed": initialization_completed,
        "initialization_scored_as_source_tracking": False,
        "initialization_source_metric_samples": 0,
        "initialization_residual_label_samples": 0,
        "initialization_terminal_base_error_m": (
            initialization_terminal_base_error_m
        ),
        "initialization_terminal_base_yaw_error_deg": (
            initialization_terminal_base_yaw_error_deg
        ),
        "initialization_terminal_riser_error_m": (
            initialization_terminal_riser_error_m
        ),
        "initialization_terminal_proxy_error_deg": (
            initialization_terminal_proxy_error_deg
        ),
        "initialization_action_saturation_ratio": (
            initialization_saturated_actions
            / max(initialization_action_count, 1)
        ),
        "initialization_riser_effort_max_n": (
            initialization_thermal_monitor.maximum_abs_force_n
        ),
        "initialization_riser_thermal_load_max": (
            initialization_thermal_monitor.maximum_thermal_load
        ),
        "initialization_riser_thermal_sample_count": (
            initialization_thermal_monitor.sample_count
        ),
        "total_simulated_duration_s": (
            initialization_steps + completed_steps
        )
        / POLICY_HZ,
        "wall_duration_s": completed_steps / POLICY_HZ,
        "progress_scale_min": float(np.min(progress_samples)),
        "progress_scale_mean": float(np.mean(progress_samples)),
        "minimum_progress_scale": tracking_cfg.minimum_progress_scale,
        "commanded_base_progress_error_enabled": (
            args.use_commanded_base_progress_error
        ),
        "phase_governor_contract": phase_governor_contract(),
        **progress_base_error_summary,
        "maximum_linear_velocity_mps": tracking_cfg.maximum_linear_velocity_mps,
        "total_pitch_reference_limit_enabled": (
            controller_cfg.limit_total_pitch_reference
        ),
        "total_pitch_reference_limit_rad": controller_cfg.pitch_reference_limit_rad,
        **progress_hold_summary,
        "camera_recovery_governor_enabled": (
            args.enable_camera_error_recovery_governor
        ),
        "camera_recovery_governor_contract": (
            "saturated_camera_error_continuous_phase_cap_v1"
        ),
        "camera_recovery_error_range_m": [
            tracking_cfg.camera_recovery_error_start_m,
            tracking_cfg.camera_recovery_error_full_m,
        ],
        "minimum_camera_recovery_scale": (
            tracking_cfg.minimum_camera_recovery_scale
        ),
        "camera_recovery_telemetry_sample_count": len(
            camera_recovery_progress_samples
        ),
        "camera_recovery_telemetry_observed": (
            camera_recovery_telemetry_observed
        ),
        "camera_recovery_progress_scale_min": float(
            np.min(camera_recovery_progress_samples)
        ),
        "camera_recovery_progress_scale_mean": float(
            np.mean(camera_recovery_progress_samples)
        ),
        "camera_recovery_activation_ratio": float(
            np.mean(np.asarray(camera_recovery_progress_samples) < 1.0 - 1e-12)
        ),
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
        "riser_effort_max_n": riser_thermal_monitor.maximum_abs_force_n,
        "riser_thermal_load_final": riser_thermal_monitor.thermal_load,
        "riser_thermal_load_max": riser_thermal_monitor.maximum_thermal_load,
        "riser_thermal_sample_count": riser_thermal_monitor.sample_count,
        "riser_peak_force_violation_count": (
            riser_thermal_monitor.peak_force_violation_count
        ),
        "riser_continuous_force_n": riser_thermal_monitor.continuous_force_n,
        "riser_peak_force_n": riser_thermal_monitor.peak_force_n,
        "riser_thermal_time_constant_s": (
            riser_thermal_monitor.thermal_time_constant_s
        ),
        "recovery_telemetry": recovery_telemetry_summary,
        "velocity_feedback_telemetry": velocity_feedback_telemetry_summary,
        "velocity_feedback_telemetry_observed": (
            velocity_feedback_telemetry_observed
        ),
        "longitudinal_authority_telemetry": (
            longitudinal_authority_telemetry_summary
        ),
        "longitudinal_authority_telemetry_observed": (
            longitudinal_authority_telemetry_observed
        ),
        "controller_vx_kp": controller_cfg.vx_kp,
        "opposing_vx_integral_deficit_reset_enabled": (
            controller_cfg.reset_opposing_vx_integral_on_directional_deficit
        ),
        "vx_integral_reset_reference_deadband_mps": (
            controller_cfg.vx_integral_reset_reference_deadband_mps
        ),
        "recovery_telemetry_observed": recovery_telemetry_observed,
        "camera_lever_arm_compensation_enabled": (
            args.enable_camera_lever_arm_compensation
        ),
        "camera_lever_arm_compensation_gain": (
            args.camera_lever_arm_compensation_gain
        ),
        "maximum_camera_lever_arm_correction_m": (
            args.maximum_camera_lever_arm_correction_m
        ),
        "camera_lever_arm_telemetry_sample_count": len(
            camera_lever_arm_correction_norm_samples
        ),
        "camera_lever_arm_telemetry_observed": (
            camera_lever_arm_telemetry_observed
        ),
        "camera_lever_arm_correction_max_m": float(
            np.max(camera_lever_arm_correction_norm_samples)
        ),
        "camera_lever_arm_raw_correction_max_m": float(
            np.max(camera_lever_arm_raw_correction_norm_samples)
        ),
        "camera_lever_arm_correction_saturation_ratio": float(
            np.mean(camera_lever_arm_saturated_samples)
        ),
        "proxy_saturation_ratio": proxy_saturation_ratio,
        "residual_action_abs_max": np.max(
            np.abs(applied_residual_values), axis=0
        ).tolist(),
        "corrective_teacher_telemetry": corrective_teacher_telemetry_summary,
        "corrective_teacher_profile": corrective_teacher_profile_identity,
        "corrective_teacher_labels_captured": corrective_capture_path is not None,
        "raw_residual_command_abs_max": np.max(
            np.abs(raw_residual_values), axis=0
        ).tolist(),
        "normalized_residual_label_abs_max": np.max(
            np.abs(normalized_residual_values), axis=0
        ).tolist(),
        "raw_residual_label_applied_to_commands": False,
        "dynamic_quality_passed": dynamic_quality_passed,
        "thermal_admission_passed": thermal_admission_passed,
        "controller_evidence_passed": controller_evidence_passed,
        "perturbation_contract_passed": perturbation_contract_passed,
        "deterministic_wrench_perturbation": perturbation_telemetry,
        "perturbation_applied_to_planner_commands": False,
        "perturbation_applied_to_policy_actions": False,
        "residual_label_envelope_passed": residual_label_envelope_ok,
        "residual_label_admission_passed": (
            case_admission_passed and residual_label_envelope_ok
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
        "executed_raw_teacher_capture": (
            None
            if raw_teacher_path is None
            else str(raw_teacher_path.resolve())
        ),
        "executed_policy_trace": (
            None
            if policy_trace_path is None
            else str(policy_trace_path.resolve())
        ),
        "executed_shadow_teacher_trace": (
            None
            if shadow_teacher_trace_path is None
            else str(shadow_teacher_trace_path.resolve())
        ),
        "executed_corrective_teacher_capture": (
            None
            if corrective_capture_path is None
            else str(corrective_capture_path.resolve())
        ),
        "passed": case_admission_passed,
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
    if args.raw_teacher_dir is not None and (
        args.dataset_dir is not None
        or args.residual_policy is not None
        or args.zero_policy_action
    ):
        raise ValueError(
            "raw teacher capture, normalized dataset collection, and policy rollout "
            "are mutually exclusive"
        )
    if args.policy_trace_dir is not None and (
        args.dataset_dir is not None or args.raw_teacher_dir is not None
    ):
        raise ValueError(
            "policy trace, raw teacher capture, and normalized dataset collection "
            "are mutually exclusive"
        )
    if args.shadow_teacher_trace_dir is not None and (
        args.dataset_dir is not None
        or args.raw_teacher_dir is not None
        or args.policy_trace_dir is not None
        or args.zero_policy_action
    ):
        raise ValueError(
            "shadow teacher trace is exclusive with dataset, raw teacher, policy "
            "trace, and zero-action modes"
        )
    if args.residual_policy is not None and args.zero_policy_action:
        raise ValueError("learned and zero-action policy modes are exclusive")
    if args.video_fps <= 0:
        raise ValueError("video fps must be positive")
    if args.video_frame_stride <= 0:
        raise ValueError("video frame stride must be positive")
    plans = {
        case: load_riser_playback_plan(
            plan_path(args.plan_dir, args.plan_filename_template, case)
        )
        for case in cases
    }
    if any(plan.case != case for case, plan in plans.items()):
        raise ValueError("playback case metadata mismatch")
    plan_identities = {
        case: hashlib.sha256(
            plan_path(args.plan_dir, args.plan_filename_template, case).read_bytes()
        ).hexdigest()
        for case in cases
    }
    if corrective_capture_admission is not None:
        capture_case = int(corrective_capture_admission["case"])
        if plans[capture_case].source_time_s is None:
            raise ValueError("corrective capture requires authoritative source time")
        if (
            plan_identities[capture_case]
            != corrective_capture_admission["plan_sha256"]
        ):
            raise ValueError("corrective capture plan identity mismatch")
    if deterministic_wrench_profile is not None and not (
        deterministic_wrench_profile.start_phase_time_s
        < float(plans[deterministic_wrench_profile.case].time_s[-1])
    ):
        raise ValueError(
            "deterministic wrench start phase must precede plan completion"
        )
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
        max(
            float(plan.time_s[-1]) * args.maximum_duration_scale
            + (
                0.0
                if plan.initialization_time_s is None
                or len(plan.initialization_time_s) == 0
                else float(plan.initialization_time_s[-1])
            )
            for plan in plans.values()
        )
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
                / args.video_frame_stride
            )
        ) + 1
        env = StridedRecordVideo(
            raw_env,
            frame_stride=args.video_frame_stride,
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
            args.raw_teacher_dir,
            args.policy_trace_dir,
            args.shadow_teacher_trace_dir,
            residual_policy,
            residual_policy_device,
            args.zero_policy_action,
            deterministic_wrench_profile,
            args.runtime_heartbeat,
            args.runtime_heartbeat_interval_steps,
            args.corrective_teacher_capture_dir,
            corrective_capture_admission,
            plan_identities[case],
        )
        for case in cases
    ]
    env.close()
    result = {
        "schema": "recomo_two_wheel_riser_reference_playback_v1",
        "training_started": False,
        "ppo_authorized": False,
        "controller_profile": "structural_robust_v1",
        "tracking_profile": tracking_profile_name(),
        "controller_vx_kp": (
            args.controller_vx_kp
            if args.controller_vx_kp is not None
            else cascaded_lqr_config("structural_robust_v1").vx_kp
        ),
        "total_pitch_reference_limit_enabled": args.limit_total_pitch_reference,
        "total_pitch_reference_limit_rad": TOTAL_PITCH_REFERENCE_LIMIT_RAD,
        "opposing_vx_integral_deficit_reset_enabled": (
            args.reset_opposing_vx_integral_on_directional_deficit
        ),
        "vx_integral_reset_reference_deadband_mps": (
            args.vx_integral_reset_reference_deadband_mps
        ),
        "tracking_recovery_velocity_cap_enabled": (
            args.tracking_maximum_linear_velocity_mps is not None
        ),
        "maximum_linear_velocity_mps": (
            args.tracking_maximum_linear_velocity_mps
            if args.tracking_maximum_linear_velocity_mps is not None
            else riser_tracking_config().maximum_linear_velocity_mps
        ),
        "camera_lever_arm_compensation_contract": (
            "measured_camera_to_base_xy_offset_v1"
        ),
        "camera_lever_arm_compensation_enabled": (
            args.enable_camera_lever_arm_compensation
        ),
        "camera_lever_arm_compensation_gain": (
            args.camera_lever_arm_compensation_gain
        ),
        "maximum_camera_lever_arm_correction_m": (
            args.maximum_camera_lever_arm_correction_m
        ),
        "tracking_direction_blend_speed_mps": (
            riser_tracking_config().direction_blend_speed_mps
        ),
        "tracking_direction_recovery_error_range_m": [
            riser_tracking_config().direction_recovery_error_start_m,
            riser_tracking_config().direction_recovery_error_full_m,
        ],
        "phase_feedforward_contract": "derivatives_scaled_by_progress_v1",
        "riser_thermal_force_contract": RISER_THERMAL_FORCE_CONTRACT,
        "controller_overrides": {
            name: value
            for name, value in {
                "vx_kp": args.controller_vx_kp,
                "wz_kp": args.controller_wz_kp,
                "wz_ki": args.controller_wz_ki,
                "wz_feedforward": args.controller_wz_feedforward,
                "wheel_difference_kp": args.controller_wheel_difference_kp,
                "limit_total_pitch_reference": (
                    True if args.limit_total_pitch_reference else None
                ),
                "reset_opposing_vx_integral_on_directional_deficit": (
                    True
                    if args.reset_opposing_vx_integral_on_directional_deficit
                    else None
                ),
                "vx_integral_reset_reference_deadband_mps": (
                    args.vx_integral_reset_reference_deadband_mps
                    if args.reset_opposing_vx_integral_on_directional_deficit
                    else None
                ),
            }.items()
            if value is not None
        },
        "tracking_overrides": {
            name: value
            for name, value in {
                "along_track_kp": args.tracking_along_kp,
                "cross_track_kp": args.tracking_cross_kp,
                "yaw_kp": args.tracking_yaw_kp,
                "maximum_linear_velocity_mps": (
                    args.tracking_maximum_linear_velocity_mps
                ),
                "minimum_progress_scale": args.tracking_minimum_progress_scale,
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
        "phase_governor_contract": phase_governor_contract(),
        "commanded_base_progress_error_enabled": (
            args.use_commanded_base_progress_error
        ),
        "progress_base_error_source": progress_governor_base_error_source(),
        "minimum_progress_scale": (
            args.tracking_minimum_progress_scale
            if args.tracking_minimum_progress_scale is not None
            else riser_tracking_config().minimum_progress_scale
        ),
        "camera_recovery_governor_enabled": (
            args.enable_camera_error_recovery_governor
        ),
        "camera_recovery_governor_contract": (
            "saturated_camera_error_continuous_phase_cap_v1"
        ),
        "camera_recovery_error_range_m": [
            args.camera_recovery_error_start_m,
            args.camera_recovery_error_full_m,
        ],
        "minimum_camera_recovery_scale": args.minimum_camera_recovery_scale,
        "com_pitch_feedforward_enabled": not args.disable_com_pitch_feedforward,
        "maximum_duration_scale": args.maximum_duration_scale,
        "completion_horizon_contract": "bounded_execution_duration_scale_v1",
        "trajectory_command_source": (
            "model_based_planner_plus_corrective_teacher"
            if corrective_teacher_config is not None
            else (
                "deterministic_teacher"
                if residual_policy is None and not args.zero_policy_action
                else (
                    (
                        "model_based_planner_plus_zero_policy_residual"
                        if args.zero_policy_action
                        else "model_based_planner_plus_torchscript_residual"
                    )
                    if args.policy_command_base == "model_based_planner"
                    else (
                        "zero_policy_action_baseline"
                        if args.zero_policy_action
                        else "torchscript_residual_policy"
                    )
                )
            )
        ),
        "policy_command_base": args.policy_command_base,
        "policy_residual_contract": (
            CORRECTIVE_TEACHER_CONTRACT
            if corrective_teacher_config is not None
            else MODEL_BASED_POLICY_RESIDUAL_CONTRACT
            if args.policy_command_base == "model_based_planner"
            else "phase_feedforward_plus_planner_imitation_residual_v1"
        ),
        "corrective_teacher_enabled": corrective_teacher_config is not None,
        "corrective_teacher_profile": corrective_teacher_profile_identity,
        "corrective_teacher_label_capture_authorized": (
            corrective_capture_admission is not None
        ),
        "corrective_teacher_capture_started": (
            args.corrective_teacher_capture_dir is not None
        ),
        "corrective_teacher_capture_admission": (
            None
            if corrective_capture_admission is None
            else str(args.corrective_teacher_capture_admission.resolve())
        ),
        "corrective_teacher_capture_split": args.corrective_teacher_capture_split,
        "residual_policy": (
            None if args.residual_policy is None else str(args.residual_policy.resolve())
        ),
        "residual_action_scales": args.residual_action_scales.tolist(),
        "video_frame_stride": args.video_frame_stride,
        "video_fps": args.video_fps,
        "raw_teacher_capture_started": args.raw_teacher_dir is not None,
        "normalized_dataset_capture_started": args.dataset_dir is not None,
        "policy_trace_started": args.policy_trace_dir is not None,
        "policy_trace_valid_for_training": False,
        "shadow_teacher_trace_started": (
            args.shadow_teacher_trace_dir is not None
        ),
        "shadow_teacher_labels_applied": False,
        "shadow_teacher_labels_admitted_for_training": False,
        "runtime_heartbeat": (
            None
            if args.runtime_heartbeat is None
            else str(args.runtime_heartbeat.resolve())
        ),
        "runtime_heartbeat_interval_steps": (
            args.runtime_heartbeat_interval_steps
        ),
        "dagger_authorized": False,
        "deterministic_wrench_perturbation_enabled": (
            deterministic_wrench_profile is not None
        ),
        "deterministic_wrench_profile": (
            deterministic_wrench_profile_identity
        ),
        "deterministic_wrench_contract": (
            "body_longitudinal_single_phase_trigger_execution_step_bounded_v1"
        ),
        "perturbation_measurement_only": True,
        "perturbation_applied_to_planner_commands": False,
        "perturbation_applied_to_policy_actions": False,
        "cases": cases,
        "passed_case_count": sum(item["passed"] for item in results),
        "dynamic_quality_passed": all(
            item["dynamic_quality_passed"] for item in results
        ),
        "thermal_admission_passed": all(
            item["thermal_admission_passed"] for item in results
        ),
        "controller_evidence_passed": all(
            item["controller_evidence_passed"] for item in results
        ),
        "perturbation_contract_passed": all(
            item["perturbation_contract_passed"] for item in results
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
        "raw_teacher_capture_started": args.raw_teacher_dir is not None,
        "normalized_dataset_capture_started": args.dataset_dir is not None,
        "policy_trace_started": args.policy_trace_dir is not None,
        "policy_trace_valid_for_training": False,
        "shadow_teacher_trace_started": (
            args.shadow_teacher_trace_dir is not None
        ),
        "shadow_teacher_labels_applied": False,
        "shadow_teacher_labels_admitted_for_training": False,
        "corrective_teacher_capture_started": (
            args.corrective_teacher_capture_dir is not None
        ),
        "corrective_teacher_label_capture_authorized": (
            corrective_capture_admission is not None
        ),
        "dagger_authorized": False,
        "deterministic_wrench_perturbation_enabled": (
            deterministic_wrench_profile is not None
        ),
        "deterministic_wrench_profile": (
            deterministic_wrench_profile_identity
        ),
        "perturbation_measurement_only": True,
        "perturbation_applied_to_planner_commands": False,
        "perturbation_applied_to_policy_actions": False,
        "tracking_profile": tracking_profile_name(),
        "controller_vx_kp": (
            args.controller_vx_kp
            if args.controller_vx_kp is not None
            else cascaded_lqr_config("structural_robust_v1").vx_kp
        ),
        "total_pitch_reference_limit_enabled": args.limit_total_pitch_reference,
        "total_pitch_reference_limit_rad": TOTAL_PITCH_REFERENCE_LIMIT_RAD,
        "opposing_vx_integral_deficit_reset_enabled": (
            args.reset_opposing_vx_integral_on_directional_deficit
        ),
        "vx_integral_reset_reference_deadband_mps": (
            args.vx_integral_reset_reference_deadband_mps
        ),
        "tracking_recovery_velocity_cap_enabled": (
            args.tracking_maximum_linear_velocity_mps is not None
        ),
        "maximum_linear_velocity_mps": (
            args.tracking_maximum_linear_velocity_mps
            if args.tracking_maximum_linear_velocity_mps is not None
            else riser_tracking_config().maximum_linear_velocity_mps
        ),
        "trajectory_command_source": (
            "model_based_planner_plus_corrective_teacher"
            if corrective_teacher_config is not None
            else "deterministic_teacher"
        ),
        "residual_policy": None,
        "camera_lever_arm_compensation_contract": (
            "measured_camera_to_base_xy_offset_v1"
        ),
        "camera_lever_arm_compensation_enabled": (
            args.enable_camera_lever_arm_compensation
        ),
        "camera_lever_arm_compensation_gain": (
            args.camera_lever_arm_compensation_gain
        ),
        "maximum_camera_lever_arm_correction_m": (
            args.maximum_camera_lever_arm_correction_m
        ),
        "camera_recovery_governor_enabled": (
            args.enable_camera_error_recovery_governor
        ),
        "camera_recovery_governor_contract": (
            "saturated_camera_error_continuous_phase_cap_v1"
        ),
        "camera_recovery_error_range_m": [
            args.camera_recovery_error_start_m,
            args.camera_recovery_error_full_m,
        ],
        "minimum_camera_recovery_scale": args.minimum_camera_recovery_scale,
        "phase_governor_enabled": not args.disable_phase_governor,
        "phase_governor_contract": phase_governor_contract(),
        "commanded_base_progress_error_enabled": (
            args.use_commanded_base_progress_error
        ),
        "progress_base_error_source": progress_governor_base_error_source(),
        "cases": cases,
        "passed_case_count": 0,
        "dynamic_quality_passed": False,
        "thermal_admission_passed": False,
        "controller_evidence_passed": False,
        "perturbation_contract_passed": False,
        "residual_label_envelope_passed": False,
        "residual_label_admission_passed": False,
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
                "maximum_duration_scale": args.maximum_duration_scale,
                "maximum_runtime_s": (
                    None
                    if failure_plan is None
                    else float(failure_plan.time_s[-1])
                    * args.maximum_duration_scale
                ),
                "executed_residual_dataset": None,
                "executed_raw_teacher_capture": None,
                "executed_policy_trace": None,
                "executed_shadow_teacher_trace": None,
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
