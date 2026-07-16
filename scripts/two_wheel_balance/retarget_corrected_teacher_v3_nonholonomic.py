#!/usr/bin/env python3
"""Build full-pose two-wheel candidates from corrected semantic GIK teachers."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import dataclass
import hashlib
import heapq
from itertools import permutations
import json
import math
from pathlib import Path
import sys

import h5py
import numpy as np
from scipy.optimize import least_squares


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_platform.tasks.two_wheel_balance.all79_reference import (  # noqa: E402
    SparseTeacher,
    discover_v3_package,
    normalize_quaternions_wxyz,
    quaternion_slerp_wxyz,
)
from rl_platform.tasks.two_wheel_balance.camera_attitude import (  # noqa: E402
    UrdfPhysicalCameraKinematics,
    matrix_quaternion_wxyz,
    physical_cam_to_semantic_dfr_quat_wxyz,
    quaternion_matrix_wxyz,
    rotation_error_vector,
    semantic_dfr_to_physical_cam_quat_wxyz,
)
from rl_platform.tasks.two_wheel_balance.whole_body_kinematics import (  # noqa: E402
    UrdfPositionKinematics,
    integrate_unicycle,
)


HOME_ARM = np.array([0.0, np.pi / 2.0, 3.0 * np.pi / 4.0])
SOURCE_PASSIVE_DFR_JOINTS = {
    "ee1_level_pitch": 0.0,
    "ee1_rot_z": 0.0,
    "ee1_rot_y": 0.0,
    "ee1_rot_x": 0.0,
}
CANDIDATE_SCHEMA = "cinebotrl_two_wheel_corrected_semantic_retarget_v3"
BALANCE_PITCH_SOLVER_TOLERANCE_DEG = 1e-6
BALANCE_PITCH_OUTPUT_TOLERANCE_DEG = 0.001
BALANCE_PITCH_OPTIMIZATION_MARGIN_DEG = 0.01
UPRIGHT_PITCH_OPTIMIZATION_MARGIN_DEG = 0.0
FORBIDDEN_EXPORT_KEYS = {
    "physical_gimbal_q",
    "physical_gimbal_joint_labels",
    "physical_gimbal_diagnostic",
    "target_cam_link_quat_wxyz",
}


@dataclass(frozen=True)
class SemanticReference:
    case: int
    source_mat: Path
    time_s: np.ndarray
    positions_m: np.ndarray
    attitudes_wxyz: np.ndarray
    source_fk_max_error_m: float
    package_position_max_error_m: float
    package_q_max_error_rad: float
    package_time_max_error_s: float
    package_attitude_max_error_deg: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-package", type=Path, required=True)
    parser.add_argument("--source-batch", type=Path, required=True)
    parser.add_argument("--source-urdf", type=Path, required=True)
    parser.add_argument("--target-urdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", default="all")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--acquisition-dt-s", type=float, default=0.1)
    parser.add_argument("--minimum-acquisition-duration-s", type=float, default=3.0)
    parser.add_argument("--maximum-acquisition-position-rate-mps", type=float, default=0.2)
    parser.add_argument("--maximum-acquisition-attitude-rate-radps", type=float, default=0.35)
    parser.add_argument("--maximum-linear-velocity", type=float, default=0.4)
    parser.add_argument("--maximum-yaw-rate", type=float, default=0.4)
    parser.add_argument("--maximum-arm-rate", type=float, default=0.5)
    parser.add_argument("--maximum-gimbal-rate", type=float, default=0.25)
    parser.add_argument("--maximum-acquisition-linear-velocity", type=float, default=0.15)
    parser.add_argument("--maximum-acquisition-yaw-rate", type=float, default=0.2)
    parser.add_argument("--maximum-acquisition-arm-rate", type=float, default=0.2)
    parser.add_argument("--maximum-acquisition-gimbal-rate", type=float, default=0.2)
    parser.add_argument("--maximum-arm-gravity-effort-nm", type=float, default=29.5)
    parser.add_argument("--gravity-effort-tolerance-nm", type=float, default=0.01)
    parser.add_argument("--maximum-equilibrium-pitch-deg", type=float, default=10.0)
    parser.add_argument("--wheel-axle-height-m", type=float, default=0.1016)
    parser.add_argument(
        "--camera-solve-root-model",
        choices=("auto", "balanced", "upright"),
        default="auto",
    )
    parser.add_argument(
        "--minimum-anchor-gimbal-limit-margin-ratio", type=float, default=0.10
    )
    parser.add_argument("--position-scale-m", type=float, default=0.01)
    parser.add_argument("--control-regularization", type=float, default=0.01)
    parser.add_argument("--maximum-position-p95-m", type=float, default=0.10)
    parser.add_argument("--maximum-position-error-m", type=float, default=0.20)
    parser.add_argument("--maximum-ik-error-deg", type=float, default=0.1)
    parser.add_argument(
        "--maximum-gimbal-interpolation-error-deg", type=float, default=0.25
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def balance_pitch_optimization_margin_deg(args: argparse.Namespace) -> float:
    return (
        UPRIGHT_PITCH_OPTIMIZATION_MARGIN_DEG
        if getattr(args, "camera_solve_root_model", "balanced") == "upright"
        else BALANCE_PITCH_OPTIMIZATION_MARGIN_DEG
    )


def as_pose_samples(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"expected pose tensor, got {values.shape}")
    if values.shape[1:] == (4, 4):
        return np.swapaxes(values, 1, 2)
    if values.shape[:2] == (4, 4):
        return np.moveaxis(values, 2, 0)
    raise ValueError(f"cannot orient pose tensor {values.shape}")


def quaternion_angle_rad(first: np.ndarray, second: np.ndarray) -> float:
    dot = abs(float(np.dot(first, second)))
    return 2.0 * math.acos(float(np.clip(dot, -1.0, 1.0)))


def load_semantic_reference(
    teacher: SparseTeacher,
    source_mat: Path,
    source_kinematics: UrdfPositionKinematics,
) -> SemanticReference:
    with h5py.File(source_mat, "r") as handle:
        q_path = np.asarray(handle["qPath"], dtype=np.float64)
        time_s = np.asarray(handle["time"], dtype=np.float64).reshape(-1)
        poses = as_pose_samples(handle["semanticPoses"])
        attitudes = normalize_quaternions_wxyz(
            np.asarray(handle["semanticQuat"], dtype=np.float64).T
        )
    if q_path.shape[0] != len(time_s) or q_path.shape[1] < 6:
        raise ValueError(f"bad q/time shape in {source_mat}: {q_path.shape}")
    if poses.shape != (len(time_s), 4, 4) or attitudes.shape != (len(time_s), 4):
        raise ValueError(f"bad semantic pose shape in {source_mat}")
    if np.any(np.diff(time_s) <= 0.0) or abs(float(time_s[0])) > 1e-10:
        raise ValueError(f"bad source timing in {source_mat}")

    package_q_error = float(np.max(np.abs(q_path[:, :6] - teacher.base_arm_q)))
    package_time_error = float(np.max(np.abs(time_s - teacher.time_s)))
    dots = np.abs(np.sum(attitudes[1:] * teacher.dfr_attitudes_wxyz, axis=1))
    package_attitude_error = float(
        np.max(np.degrees(2.0 * np.arccos(np.clip(dots, -1.0, 1.0))))
    )
    source_positions = np.asarray(
        [source_kinematics.position(state) for state in q_path[:, :6]]
    )
    positions = poses[:, :3, 3]
    package_position_error = float(
        np.max(np.linalg.norm(positions - teacher.desired_positions_m, axis=1))
    )
    source_fk_error = float(
        np.max(np.linalg.norm(source_positions - positions, axis=1))
    )
    if package_q_error > 2e-6:
        raise ValueError(f"package/source q mismatch in case {teacher.case}: {package_q_error}")
    if package_time_error > 2e-8:
        raise ValueError(
            f"package/source time mismatch in case {teacher.case}: {package_time_error}"
        )
    if package_attitude_error > 1e-4:
        raise ValueError(
            f"package/source attitude mismatch in case {teacher.case}: "
            f"{package_attitude_error} deg"
        )
    if package_position_error > 1e-8:
        raise ValueError(
            f"package/source ordered position mismatch in case {teacher.case}: "
            f"{package_position_error} m"
        )
    if source_fk_error > 1e-6:
        raise ValueError(
            f"source ee1_tool FK mismatch in case {teacher.case}: {source_fk_error} m"
        )
    return SemanticReference(
        case=teacher.case,
        source_mat=source_mat,
        time_s=time_s,
        positions_m=teacher.desired_positions_m,
        attitudes_wxyz=teacher.desired_attitudes_wxyz,
        source_fk_max_error_m=source_fk_error,
        package_position_max_error_m=package_position_error,
        package_q_max_error_rad=package_q_error,
        package_time_max_error_s=package_time_error,
        package_attitude_max_error_deg=package_attitude_error,
    )


def build_target_path(
    reference: SemanticReference,
    position_kinematics: UrdfPositionKinematics,
    camera_kinematics: UrdfPhysicalCameraKinematics,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    home_state = np.concatenate((np.zeros(3), HOME_ARM))
    home_position = position_kinematics.position(home_state)
    home_physical_rotation = camera_kinematics.world_rotation(
        np.array([1.0, 0.0, 0.0, 0.0]), HOME_ARM, np.zeros(3)
    )
    home_attitude = physical_cam_to_semantic_dfr_quat_wxyz(
        matrix_quaternion_wxyz(home_physical_rotation)
    )
    acquisition_duration = max(
        args.minimum_acquisition_duration_s,
        float(np.linalg.norm(reference.positions_m[0] - home_position))
        / args.maximum_acquisition_position_rate_mps,
        quaternion_angle_rad(home_attitude, reference.attitudes_wxyz[0])
        / args.maximum_acquisition_attitude_rate_radps,
    )
    acquisition_steps = int(math.ceil(acquisition_duration / args.acquisition_dt_s))
    acquisition_duration = acquisition_steps * args.acquisition_dt_s
    acquisition_time = np.linspace(0.0, acquisition_duration, acquisition_steps + 1)
    phase = acquisition_time / acquisition_duration
    blend = 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5
    acquisition_positions = (
        home_position[None, :]
        + blend[:, None] * (reference.positions_m[0] - home_position)[None, :]
    )
    acquisition_attitudes = quaternion_slerp_wxyz(
        home_attitude, reference.attitudes_wxyz[0], blend
    )
    time_s = np.concatenate(
        (acquisition_time, acquisition_duration + reference.time_s[1:])
    )
    positions = np.vstack((acquisition_positions, reference.positions_m[1:]))
    attitudes = np.vstack((acquisition_attitudes, reference.attitudes_wxyz[1:]))
    return time_s, positions, attitudes, acquisition_steps, acquisition_duration


def retarget_positions(
    time_s: np.ndarray,
    targets: np.ndarray,
    kinematics: UrdfPositionKinematics,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    state = np.concatenate((np.zeros(3), HOME_ARM.copy()))
    states = np.empty((len(time_s), 6), dtype=np.float64)
    achieved = np.empty_like(targets)
    controls = np.zeros((len(time_s) - 1, 5), dtype=np.float64)
    states[0] = state
    achieved[0] = kinematics.position(state)
    previous_control = np.zeros(5, dtype=np.float64)
    for index in range(1, len(time_s)):
        dt = float(time_s[index] - time_s[index - 1])
        lower = np.concatenate(
            (
                [-args.maximum_linear_velocity, -args.maximum_yaw_rate],
                np.maximum(
                    -args.maximum_arm_rate * dt,
                    kinematics.arm_lower - state[3:],
                ),
            )
        )
        upper = np.concatenate(
            (
                [args.maximum_linear_velocity, args.maximum_yaw_rate],
                np.minimum(
                    args.maximum_arm_rate * dt,
                    kinematics.arm_upper - state[3:],
                ),
            )
        )

        def candidate(control: np.ndarray) -> np.ndarray:
            base = integrate_unicycle(state[:3], control[0], control[1], dt)
            return np.concatenate((base, state[3:] + control[2:]))

        def residual(control: np.ndarray) -> np.ndarray:
            next_state = candidate(control)
            position_error = (
                kinematics.position(next_state) - targets[index]
            ) / args.position_scale_m
            regularization = args.control_regularization * np.concatenate(
                (
                    control[:1] / args.maximum_linear_velocity,
                    control[1:2] / args.maximum_yaw_rate,
                    control[2:] / max(args.maximum_arm_rate * dt, 1e-9),
                )
            )
            return np.concatenate((position_error, regularization))

        solution = least_squares(
            residual,
            np.clip(previous_control, lower, upper),
            bounds=(lower, upper),
            max_nfev=80,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
        )
        previous_control = solution.x
        state = candidate(solution.x)
        controls[index - 1] = solution.x
        states[index] = state
        achieved[index] = kinematics.position(state)
    return states, achieved, controls, np.linalg.norm(achieved - targets, axis=1)


def validate_and_retime_gimbal(
    time_s: np.ndarray,
    states: np.ndarray,
    attitudes: np.ndarray,
    controls: np.ndarray,
    kinematics: UrdfPhysicalCameraKinematics,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    gimbal_path = np.empty((len(time_s), 3), dtype=np.float64)
    gimbal_path[0] = 0.0
    errors_deg = np.zeros(len(time_s), dtype=np.float64)
    nonconverged = []
    for index in range(1, len(time_s)):
        yaw = float(states[index, 2])
        root_quaternion = np.array(
            [math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)]
        )
        result = kinematics.solve_semantic_attitude_continuous(
            root_quaternion,
            states[index, 3:],
            attitudes[index],
            gimbal_path[index - 1],
        )
        gimbal_path[index] = result.gimbal_q
        errors_deg[index] = math.degrees(result.orientation_error_rad)
        if not result.converged:
            nonconverged.append(index)

    old_dt = np.diff(time_s)
    gimbal_delta = np.abs(np.diff(gimbal_path, axis=0))
    required_dt = np.max(gimbal_delta / args.maximum_gimbal_rate, axis=1)
    new_dt = np.maximum(old_dt, required_dt)
    retimed_time = np.concatenate(([0.0], np.cumsum(new_dt)))
    retimed_controls = controls.copy()
    retimed_controls[:, :2] *= (old_dt / new_dt)[:, None]
    achieved_rate = gimbal_delta / new_dt[:, None]
    diagnostics = {
        "physical_gimbal_ik_nonconverged_indices": nonconverged,
        "physical_gimbal_ik_max_error_deg": float(np.max(errors_deg)),
        "physical_gimbal_rate_max_radps": float(np.max(achieved_rate)),
        "gimbal_retimed_interval_count": int(np.count_nonzero(new_dt > old_dt + 1e-12)),
        "pre_gimbal_retime_duration_s": float(time_s[-1]),
        "retargeted_duration_s": float(retimed_time[-1]),
        "physical_gimbal_path_exported": False,
    }
    return retimed_time, retimed_controls, diagnostics


def wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def physical_camera_rotation(
    state: np.ndarray, kinematics: UrdfPhysicalCameraKinematics
) -> np.ndarray:
    yaw = float(state[2])
    root_quaternion = np.array(
        [math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)]
    )
    return kinematics.world_rotation(root_quaternion, state[3:6], state[6:9])


def root_quaternion_from_pitch_yaw(pitch: float, yaw: float) -> np.ndarray:
    half_pitch = 0.5 * pitch
    half_yaw = 0.5 * yaw
    cp, sp = math.cos(half_pitch), math.sin(half_pitch)
    cy, sy = math.cos(half_yaw), math.sin(half_yaw)
    return np.array([cp * cy, -sp * sy, sp * cy, cp * sy])


def equilibrium_pitch_deg(
    state: np.ndarray,
    kinematics: UrdfPositionKinematics,
    args: argparse.Namespace,
) -> float:
    return math.degrees(
        abs(kinematics.equilibrium_pitch_rad(state[:6], args.wheel_axle_height_m))
    )


def balanced_physical_camera_rotation(
    state: np.ndarray,
    position_kinematics: UrdfPositionKinematics,
    camera_kinematics: UrdfPhysicalCameraKinematics,
    args: argparse.Namespace,
) -> np.ndarray:
    pitch = position_kinematics.equilibrium_pitch_rad(
        state[:6], args.wheel_axle_height_m
    )
    return camera_kinematics.world_rotation(
        root_quaternion_from_pitch_yaw(pitch, float(state[2])),
        state[3:6],
        state[6:9],
    )


def retarget_solver_camera_rotation(
    state: np.ndarray,
    position_kinematics: UrdfPositionKinematics,
    camera_kinematics: UrdfPhysicalCameraKinematics,
    args: argparse.Namespace,
) -> np.ndarray:
    model = getattr(args, "camera_solve_root_model", "balanced")
    if model == "balanced":
        return balanced_physical_camera_rotation(
            state, position_kinematics, camera_kinematics, args
        )
    if model == "upright":
        return physical_camera_rotation(state, camera_kinematics)
    raise ValueError(f"unknown camera solve root model: {model}")


def physical_gimbal_interpolation_error(
    states: np.ndarray,
    attitudes: np.ndarray,
    position_kinematics: UrdfPositionKinematics,
    camera_kinematics: UrdfPhysicalCameraKinematics,
    args: argparse.Namespace,
    *,
    maximum_sample_joint_step_rad: float = math.radians(1.0),
) -> tuple[float, int]:
    """Audit the physical camera between IK endpoints, not only at them."""

    if (
        states.ndim != 2
        or states.shape[1] != 9
        or attitudes.shape != (len(states), 4)
        or len(states) < 2
        or maximum_sample_joint_step_rad <= 0.0
    ):
        raise ValueError("invalid physical-gimbal interpolation audit input")
    maximum_error_deg = 0.0
    maximum_error_interval = 0
    for index in range(len(states) - 1):
        gimbal_delta = float(
            np.max(np.abs(states[index + 1, 6:9] - states[index, 6:9]))
        )
        sample_count = max(
            1, int(math.ceil(gimbal_delta / maximum_sample_joint_step_rad))
        )
        for sample in range(sample_count + 1):
            fraction = sample / sample_count
            state = (1.0 - fraction) * states[index] + fraction * states[index + 1]
            attitude = quaternion_slerp_wxyz(
                attitudes[index], attitudes[index + 1], np.array([fraction])
            )[0]
            target_rotation = quaternion_matrix_wxyz(
                semantic_dfr_to_physical_cam_quat_wxyz(attitude)
            )
            error_deg = math.degrees(
                float(
                    np.linalg.norm(
                        rotation_error_vector(
                            balanced_physical_camera_rotation(
                                state,
                                position_kinematics,
                                camera_kinematics,
                                args,
                            ),
                            target_rotation,
                        )
                    )
                )
            )
            if error_deg > maximum_error_deg:
                maximum_error_deg = error_deg
                maximum_error_interval = index
    return maximum_error_deg, maximum_error_interval


def retime_gimbal_for_equilibrium_pitch(
    time_s: np.ndarray,
    states: np.ndarray,
    attitudes: np.ndarray,
    controls: np.ndarray,
    position_kinematics: UrdfPositionKinematics,
    camera_kinematics: UrdfPhysicalCameraKinematics,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    gimbal_path = np.empty((len(time_s), 3), dtype=np.float64)
    errors_deg = np.empty(len(time_s), dtype=np.float64)
    nonconverged = []
    previous = np.zeros(3, dtype=np.float64)
    for index, (state, attitude) in enumerate(zip(states, attitudes, strict=True)):
        pitch = position_kinematics.equilibrium_pitch_rad(
            state[:6], args.wheel_axle_height_m
        )
        root_quaternion = root_quaternion_from_pitch_yaw(pitch, float(state[2]))
        result = camera_kinematics.solve_semantic_attitude_continuous(
            root_quaternion,
            state[3:6],
            attitude,
            previous,
        )
        gimbal_path[index] = result.gimbal_q
        errors_deg[index] = math.degrees(result.orientation_error_rad)
        if not result.converged:
            nonconverged.append(index)
        previous = result.gimbal_q

    old_dt = np.diff(time_s)
    gimbal_delta = np.abs(np.diff(gimbal_path, axis=0))
    required_dt = np.max(gimbal_delta / args.maximum_gimbal_rate, axis=1)
    new_dt = np.maximum(old_dt, required_dt)
    retimed_time = np.concatenate(([0.0], np.cumsum(new_dt)))
    retimed_controls = controls.copy()
    retimed_controls[:, :2] *= (old_dt / new_dt)[:, None]
    retimed_states = states.copy()
    retimed_states[:, 6:9] = gimbal_path
    achieved_rate = gimbal_delta / new_dt[:, None]
    (
        interpolation_error_deg,
        interpolation_error_interval,
    ) = physical_gimbal_interpolation_error(
        retimed_states,
        attitudes,
        position_kinematics,
        camera_kinematics,
        args,
    )
    diagnostics = {
        "physical_gimbal_ik_nonconverged_indices": nonconverged,
        "physical_gimbal_ik_max_error_deg": float(np.max(errors_deg)),
        "physical_gimbal_rate_max_radps": float(np.max(achieved_rate)),
        "physical_gimbal_interpolation_max_error_deg": interpolation_error_deg,
        "physical_gimbal_interpolation_max_error_interval": (
            interpolation_error_interval
        ),
        "equilibrium_pitch_gimbal_retimed_interval_count": int(
            np.count_nonzero(new_dt > old_dt + 1e-12)
        ),
        "pre_equilibrium_pitch_gimbal_retime_duration_s": float(time_s[-1]),
        "retargeted_duration_s": float(retimed_time[-1]),
        "physical_gimbal_root_attitude_model": (
            "yaw_plus_predicted_equilibrium_pitch"
        ),
        "physical_gimbal_path_exported": False,
    }
    return retimed_time, retimed_states, retimed_controls, diagnostics


def solve_full_pose_anchor(
    source_base_arm_q: np.ndarray,
    target_position: np.ndarray,
    target_attitude: np.ndarray,
    position_kinematics: UrdfPositionKinematics,
    camera_kinematics: UrdfPhysicalCameraKinematics,
    args: argparse.Namespace,
) -> tuple[np.ndarray, float, float]:
    target_rotation = quaternion_matrix_wxyz(
        semantic_dfr_to_physical_cam_quat_wxyz(target_attitude)
    )
    lower = np.concatenate(
        (
            source_base_arm_q[:2] - 2.0,
            [-math.pi],
            position_kinematics.arm_lower,
            camera_kinematics.gimbal_lower,
        )
    )
    upper = np.concatenate(
        (
            source_base_arm_q[:2] + 2.0,
            [math.pi],
            position_kinematics.arm_upper,
            camera_kinematics.gimbal_upper,
        )
    )
    levels = tuple(
        (lower_value, 0.5 * (lower_value + upper_value), upper_value)
        for lower_value, upper_value in zip(
            camera_kinematics.gimbal_lower,
            camera_kinematics.gimbal_upper,
            strict=True,
        )
    )

    def residual(state: np.ndarray) -> np.ndarray:
        position_error = (
            position_kinematics.position(state[:6]) - target_position
        ) / 0.005
        attitude_error = rotation_error_vector(
            retarget_solver_camera_rotation(
                state, position_kinematics, camera_kinematics, args
            ),
            target_rotation,
        ) / math.radians(0.5)
        state_regularization = 0.005 * (
            state[:6] - source_base_arm_q
        ) / np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        return np.concatenate((position_error, attitude_error, state_regularization))

    gimbal_range = camera_kinematics.gimbal_upper - camera_kinematics.gimbal_lower
    gimbal_center = 0.5 * (
        camera_kinematics.gimbal_lower + camera_kinematics.gimbal_upper
    )

    def assess(
        state: np.ndarray,
    ) -> tuple[float, float, float, float, float, float, np.ndarray]:
        position_error = float(
            np.linalg.norm(position_kinematics.position(state[:6]) - target_position)
        )
        attitude_error = math.degrees(
            float(
                np.linalg.norm(
                    rotation_error_vector(
                        retarget_solver_camera_rotation(
                            state,
                            position_kinematics,
                            camera_kinematics,
                            args,
                        ),
                        target_rotation,
                    )
                )
            )
        )
        gravity_effort = float(
            np.max(
                np.abs(position_kinematics.gravitational_effort_nm(state[:6]))
            )
        )
        balance_pitch = equilibrium_pitch_deg(state, position_kinematics, args)
        gimbal_margin = float(
            np.min(
                np.minimum(
                    state[6:9] - camera_kinematics.gimbal_lower,
                    camera_kinematics.gimbal_upper - state[6:9],
                )
                / gimbal_range
            )
        )
        source_distance = float(np.linalg.norm(state[:6] - source_base_arm_q))
        return (
            position_error,
            attitude_error,
            gravity_effort,
            balance_pitch,
            gimbal_margin,
            source_distance,
            state,
        )

    def feasible_candidates(
        values: list[tuple[float, float, float, float, float, float, np.ndarray]],
    ) -> list[tuple[float, float, float, float, float, float, np.ndarray]]:
        return [
            item
            for item in values
            if item[0] <= 1e-4
            and item[1] <= 0.01
            and item[2]
            <= args.maximum_arm_gravity_effort_nm
            + args.gravity_effort_tolerance_nm
            and item[3]
            <= args.maximum_equilibrium_pitch_deg
            + BALANCE_PITCH_SOLVER_TOLERANCE_DEG
            and item[4] >= args.minimum_anchor_gimbal_limit_margin_ratio
        ]

    candidates = []
    for yaw in levels[0]:
        for roll in levels[1]:
            for pitch in levels[2]:
                seed = np.concatenate((source_base_arm_q, [yaw, roll, pitch]))
                solution = least_squares(
                    residual,
                    np.clip(seed, lower, upper),
                    bounds=(lower, upper),
                    max_nfev=300,
                    ftol=1e-10,
                    xtol=1e-10,
                    gtol=1e-10,
                )
                candidates.append(assess(solution.x))
    feasible = feasible_candidates(candidates)
    if not feasible:
        # Exact camera pose has redundant base/arm/gimbal solutions. The fast
        # seeds can land on either a gravity-safe arm or a well-centered gimbal;
        # refine both constraints together before rejecting the teacher.
        def constrained_residual(state: np.ndarray) -> np.ndarray:
            gravity = np.abs(
                position_kinematics.gravitational_effort_nm(state[:6])
            )
            balance_pitch = equilibrium_pitch_deg(
                state, position_kinematics, args
            )
            margin = np.minimum(
                state[6:9] - camera_kinematics.gimbal_lower,
                camera_kinematics.gimbal_upper - state[6:9],
            ) / gimbal_range
            return np.concatenate(
                (
                    residual(state),
                    np.maximum(
                        gravity - args.maximum_arm_gravity_effort_nm, 0.0
                    )
                    / 0.02,
                    [
                        max(
                            balance_pitch
                            - (
                                args.maximum_equilibrium_pitch_deg
                                - balance_pitch_optimization_margin_deg(args)
                            ),
                            0.0,
                        )
                        / 0.02
                    ],
                    np.maximum(
                        args.minimum_anchor_gimbal_limit_margin_ratio - margin,
                        0.0,
                    )
                    / 0.002,
                    0.01 * (state[6:9] - gimbal_center) / gimbal_range,
                )
            )

        refined = []
        for candidate in candidates:
            solution = least_squares(
                constrained_residual,
                candidate[6],
                bounds=(lower, upper),
                max_nfev=600,
                ftol=1e-11,
                xtol=1e-11,
                gtol=1e-11,
            )
            refined.append(assess(solution.x))
        candidates.extend(refined)
        feasible = feasible_candidates(candidates)
    if not feasible:
        best = min(
            candidates,
            key=lambda item: (
                (item[0] / 0.01)
                + item[1]
                + max(0.0, item[2] - args.maximum_arm_gravity_effort_nm)
                + max(0.0, item[3] - args.maximum_equilibrium_pitch_deg)
                + 10.0
                * max(
                    0.0,
                    args.minimum_anchor_gimbal_limit_margin_ratio - item[4],
                )
            ),
        )
        raise ValueError(
            f"cannot solve full-pose anchor: position={best[0]:.6f} m, "
            f"attitude={best[1]:.6f} deg, gravity={best[2]:.6f} Nm, "
            f"equilibrium_pitch={best[3]:.6f} deg, "
            f"gimbal_margin={best[4]:.6f}"
        )
    ranked = sorted(
        feasible,
        key=lambda item: item[5] + 0.02 * item[2] + 0.05 * item[3] - 0.25 * item[4],
    )
    acquisition_failures = []
    for position_error, attitude_error, _, _, _, _, state in ranked:
        try:
            build_gravity_aware_arm_acquisition(
                state[3:6], position_kinematics, args, allow_astar=False
            )
        except ValueError as error:
            acquisition_failures.append(str(error))
            continue
        return state, position_error, attitude_error
    for position_error, attitude_error, _, _, _, _, state in ranked:
        try:
            build_gravity_aware_arm_acquisition(
                state[3:6], position_kinematics, args
            )
        except ValueError as error:
            acquisition_failures.append(str(error))
            continue
        return state, position_error, attitude_error
    raise ValueError(
        "no pose-valid full-pose anchor has a gravity-safe home acquisition; "
        f"best failure: {acquisition_failures[0]}"
    )


def build_smooth_joint_segment(
    start: np.ndarray,
    end: np.ndarray,
    maximum_rate: float,
    dt: float,
) -> np.ndarray:
    maximum_delta = float(np.max(np.abs(end - start)))
    if maximum_delta <= 1e-12:
        return start[None, :]
    steps = int(math.ceil(1.875 * maximum_delta / (maximum_rate * dt)))
    phase = np.linspace(0.0, 1.0, steps + 1)
    blend = 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5
    return start[None, :] + blend[:, None] * (end - start)[None, :]


def build_gravity_aware_arm_acquisition(
    anchor_arm: np.ndarray,
    position_kinematics: UrdfPositionKinematics,
    args: argparse.Namespace,
    *,
    allow_astar: bool = True,
) -> tuple[np.ndarray, float, str]:
    gravity_cache: dict[tuple[float, float, float], float] = {}
    pitch_cache: dict[tuple[float, float, float], float] = {}

    def arm_gravity(arm: np.ndarray) -> float:
        key = tuple(np.round(np.asarray(arm, dtype=np.float64), 8))
        if key not in gravity_cache:
            gravity_cache[key] = float(
                np.max(
                    np.abs(
                        position_kinematics.gravitational_effort_nm(
                            np.concatenate((np.zeros(3), arm))
                        )
                    )
                )
            )
        return gravity_cache[key]

    def arm_balance_pitch(arm: np.ndarray) -> float:
        key = tuple(np.round(np.asarray(arm, dtype=np.float64), 8))
        if key not in pitch_cache:
            pitch_cache[key] = equilibrium_pitch_deg(
                np.concatenate((np.zeros(3), arm)),
                position_kinematics,
                args,
            )
        return pitch_cache[key]

    def maximum_gravity(path: np.ndarray) -> float:
        return max(arm_gravity(arm) for arm in path)

    def maximum_balance_pitch(path: np.ndarray) -> float:
        return max(arm_balance_pitch(arm) for arm in path)

    def path_is_safe(path: np.ndarray) -> bool:
        return (
            maximum_gravity(path) <= effort_limit
            and maximum_balance_pitch(path)
            <= args.maximum_equilibrium_pitch_deg
            + BALANCE_PITCH_SOLVER_TOLERANCE_DEG
        )

    effort_limit = (
        args.maximum_arm_gravity_effort_nm + args.gravity_effort_tolerance_nm
    )

    direct = build_smooth_joint_segment(
        HOME_ARM, anchor_arm, args.maximum_acquisition_arm_rate, args.acquisition_dt_s
    )
    direct_gravity = maximum_gravity(direct)
    if path_is_safe(direct):
        return direct, direct_gravity, "direct_quintic"

    candidates = []
    for order in permutations(range(3)):
        segments = []
        state = HOME_ARM.copy()
        for joint_index in order:
            target = state.copy()
            target[joint_index] = anchor_arm[joint_index]
            segment = build_smooth_joint_segment(
                state,
                target,
                args.maximum_acquisition_arm_rate,
                args.acquisition_dt_s,
            )
            segments.append(segment if not segments else segment[1:])
            state = target
        path = np.vstack(segments)
        gravity = maximum_gravity(path)
        balance_pitch = maximum_balance_pitch(path)
        candidates.append((gravity, balance_pitch, len(path), order, path))
    feasible = [
        item
        for item in candidates
        if item[0] <= effort_limit
        and item[1]
        <= args.maximum_equilibrium_pitch_deg
        + BALANCE_PITCH_SOLVER_TOLERANCE_DEG
    ]
    if feasible:
        gravity, _, _, order, path = min(
            feasible, key=lambda item: (item[2], item[0], item[1])
        )
        return path, gravity, "staged_" + "_".join(str(index) for index in order)
    if not allow_astar:
        raise ValueError("no direct or staged gravity/COM-safe arm acquisition")

    pitch_values = np.unique(
        np.concatenate(
            (
                np.linspace(
                    position_kinematics.arm_lower[1],
                    position_kinematics.arm_upper[1],
                    25,
                ),
                [HOME_ARM[1], anchor_arm[1]],
            )
        )
    )
    elbow_values = np.unique(
        np.concatenate(
            (
                np.linspace(
                    position_kinematics.arm_lower[2],
                    position_kinematics.arm_upper[2],
                    33,
                ),
                [HOME_ARM[2], anchor_arm[2]],
            )
        )
    )

    start = (
        int(np.argmin(np.abs(pitch_values - HOME_ARM[1]))),
        int(np.argmin(np.abs(elbow_values - HOME_ARM[2]))),
    )
    goal = (
        int(np.argmin(np.abs(pitch_values - anchor_arm[1]))),
        int(np.argmin(np.abs(elbow_values - anchor_arm[2]))),
    )

    def safe_segment(first: np.ndarray, second: np.ndarray) -> tuple[bool, np.ndarray]:
        segment = build_smooth_joint_segment(
            first,
            second,
            args.maximum_acquisition_arm_rate,
            args.acquisition_dt_s,
        )
        return path_is_safe(segment), segment

    directions = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )

    def search_pitch_elbow(fixed_yaw: float) -> tuple[np.ndarray, int] | None:
        def node_arm(node: tuple[int, int]) -> np.ndarray:
            return np.array(
                [fixed_yaw, pitch_values[node[0]], elbow_values[node[1]]]
            )

        safe = np.zeros((len(pitch_values), len(elbow_values)), dtype=bool)
        for pitch_index in range(len(pitch_values)):
            for elbow_index in range(len(elbow_values)):
                arm = node_arm((pitch_index, elbow_index))
                safe[pitch_index, elbow_index] = (
                    arm_gravity(arm) <= effort_limit
                    and arm_balance_pitch(arm)
                    <= args.maximum_equilibrium_pitch_deg
                    + BALANCE_PITCH_SOLVER_TOLERANCE_DEG
                )
        if not safe[start] or not safe[goal]:
            return None

        distances = {start: 0.0}
        parents: dict[tuple[int, int], tuple[int, int]] = {}
        queue = [(0.0, 0.0, start)]
        while queue:
            _, distance, node = heapq.heappop(queue)
            if distance > distances.get(node, math.inf) + 1e-12:
                continue
            if node == goal:
                break
            for pitch_delta, elbow_delta in directions:
                neighbor = (node[0] + pitch_delta, node[1] + elbow_delta)
                if not (
                    0 <= neighbor[0] < len(pitch_values)
                    and 0 <= neighbor[1] < len(elbow_values)
                    and safe[neighbor]
                ):
                    continue
                first = node_arm(node)
                second = node_arm(neighbor)
                edge_safe, _ = safe_segment(first, second)
                if not edge_safe:
                    continue
                edge_cost = float(np.linalg.norm(second[1:] - first[1:]))
                candidate_distance = distance + edge_cost
                if candidate_distance >= distances.get(neighbor, math.inf) - 1e-12:
                    continue
                distances[neighbor] = candidate_distance
                parents[neighbor] = node
                heuristic = float(
                    np.linalg.norm(node_arm(neighbor)[1:] - node_arm(goal)[1:])
                )
                heapq.heappush(
                    queue,
                    (candidate_distance + heuristic, candidate_distance, neighbor),
                )
        if goal not in distances:
            return None

        nodes = [goal]
        while nodes[-1] != start:
            nodes.append(parents[nodes[-1]])
        nodes.reverse()
        waypoints = [node_arm(node) for node in nodes]
        simplified = [waypoints[0]]
        index = 0
        while index < len(waypoints) - 1:
            for candidate_index in range(len(waypoints) - 1, index, -1):
                edge_safe, _ = safe_segment(
                    simplified[-1], waypoints[candidate_index]
                )
                if edge_safe:
                    simplified.append(waypoints[candidate_index])
                    index = candidate_index
                    break

        segments = []
        for first, second in zip(simplified[:-1], simplified[1:], strict=True):
            _, segment = safe_segment(first, second)
            segments.append(segment if not segments else segment[1:])
        path = (
            np.vstack(segments)
            if segments
            else np.asarray(simplified, dtype=np.float64)
        )
        return path, len(simplified)

    for yaw_order, fixed_yaw in (
        ("yaw_last", HOME_ARM[0]),
        ("yaw_first", anchor_arm[0]),
    ):
        pitch_elbow_result = search_pitch_elbow(float(fixed_yaw))
        if pitch_elbow_result is None:
            continue
        pitch_elbow_path, waypoint_count = pitch_elbow_result
        if yaw_order == "yaw_last":
            yaw_target = pitch_elbow_path[-1].copy()
            yaw_target[0] = anchor_arm[0]
            yaw_segment = build_smooth_joint_segment(
                pitch_elbow_path[-1],
                yaw_target,
                args.maximum_acquisition_arm_rate,
                args.acquisition_dt_s,
            )
            path = np.vstack((pitch_elbow_path, yaw_segment[1:]))
        else:
            yaw_start = HOME_ARM.copy()
            yaw_start[0] = anchor_arm[0]
            yaw_segment = build_smooth_joint_segment(
                HOME_ARM,
                yaw_start,
                args.maximum_acquisition_arm_rate,
                args.acquisition_dt_s,
            )
            path = np.vstack((yaw_segment, pitch_elbow_path[1:]))
        if path_is_safe(path):
            return (
                path,
                maximum_gravity(path),
                f"astar_pitch_elbow_{yaw_order}_{waypoint_count}_waypoints",
            )

    best = min(
        candidates,
        key=lambda item: (
            max(0.0, item[0] - effort_limit)
            + max(0.0, item[1] - args.maximum_equilibrium_pitch_deg)
        ),
    )
    raise ValueError(
        "cannot build gravity/COM-safe arm acquisition: "
        f"best_staged={best[0]:.6f} Nm, "
        f"best_pitch={best[1]:.6f} deg, "
        f"limit={args.maximum_arm_gravity_effort_nm:.6f} Nm, "
        f"pitch_limit={args.maximum_equilibrium_pitch_deg:.6f} deg, "
        "pitch_elbow_graph_disconnected_for_yaw_first_and_yaw_last"
    )


def build_feasible_acquisition(
    anchor: np.ndarray,
    anchor_semantic_attitude: np.ndarray,
    position_kinematics: UrdfPositionKinematics,
    camera_kinematics: UrdfPhysicalCameraKinematics,
    args: argparse.Namespace,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, object],
]:
    dt = args.acquisition_dt_s
    base = np.zeros(3, dtype=np.float64)
    base_states = [base.copy()]
    base_controls: list[tuple[float, float]] = []

    def append_segment(delta: float, maximum_rate: float, *, drive: bool) -> None:
        nonlocal base
        if abs(delta) <= 1e-12:
            return
        steps = int(math.ceil(abs(delta) / (maximum_rate * dt)))
        rate = delta / (steps * dt)
        for _ in range(steps):
            velocity, yaw_rate = (rate, 0.0) if drive else (0.0, rate)
            base = integrate_unicycle(base, velocity, yaw_rate, dt)
            base_states.append(base.copy())
            base_controls.append((velocity, yaw_rate))

    distance = float(np.linalg.norm(anchor[:2]))
    heading = math.atan2(float(anchor[1]), float(anchor[0])) if distance > 1e-12 else 0.0
    append_segment(
        wrap_angle(heading - base[2]),
        args.maximum_acquisition_yaw_rate,
        drive=False,
    )
    append_segment(distance, args.maximum_acquisition_linear_velocity, drive=True)
    append_segment(
        wrap_angle(float(anchor[2]) - base[2]),
        args.maximum_acquisition_yaw_rate,
        drive=False,
    )
    base_states[-1][:3] = anchor[:3]

    arm, arm_gravity, arm_plan = build_gravity_aware_arm_acquisition(
        anchor[3:6], position_kinematics, args
    )
    anchor_equilibrium_pitch = position_kinematics.equilibrium_pitch_rad(
        anchor[:6], args.wheel_axle_height_m
    )
    acquisition_gimbal_ik = camera_kinematics.solve_semantic_attitude_continuous(
        root_quaternion_from_pitch_yaw(
            anchor_equilibrium_pitch, float(anchor[2])
        ),
        anchor[3:6],
        anchor_semantic_attitude,
        anchor[6:9],
    )
    if not acquisition_gimbal_ik.converged:
        raise ValueError(
            "anchor camera attitude is infeasible at equilibrium pitch: "
            f"residual={math.degrees(acquisition_gimbal_ik.orientation_error_rad):.6f} deg"
        )
    acquisition_gimbal_target = acquisition_gimbal_ik.gimbal_q
    required_steps = max(
        int(math.ceil(args.minimum_acquisition_duration_s / dt)),
        len(arm) - 1,
        int(
            math.ceil(
                1.875
                * float(np.max(np.abs(acquisition_gimbal_target)))
                / (args.maximum_acquisition_gimbal_rate * dt)
            )
        ),
    )
    while len(base_controls) < required_steps:
        base_states.append(base_states[-1].copy())
        base_controls.append((0.0, 0.0))
    if len(arm) - 1 < required_steps:
        arm = np.vstack(
            (arm, np.repeat(arm[-1][None, :], required_steps - len(arm) + 1, axis=0))
        )

    base_array = np.asarray(base_states)
    steps = len(base_controls)
    phase = np.linspace(0.0, 1.0, steps + 1)
    blend = 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5
    gimbal = blend[:, None] * acquisition_gimbal_target[None, :]
    states = np.column_stack((base_array, arm, gimbal))
    controls = np.column_stack((np.asarray(base_controls), np.diff(arm, axis=0)))
    time_s = np.arange(steps + 1, dtype=np.float64) * dt
    positions = np.asarray(
        [position_kinematics.position(state[:6]) for state in states]
    )
    attitudes = np.asarray(
        [
            physical_cam_to_semantic_dfr_quat_wxyz(
                matrix_quaternion_wxyz(
                    camera_kinematics.world_rotation(
                        root_quaternion_from_pitch_yaw(
                            position_kinematics.equilibrium_pitch_rad(
                                state[:6], args.wheel_axle_height_m
                            ),
                            float(state[2]),
                        ),
                        state[3:6],
                        state[6:9],
                    )
                )
            )
            for state in states
        ]
    )
    diagnostics = {
        "arm_acquisition_plan": arm_plan,
        "arm_acquisition_predicted_gravity_max_nm": arm_gravity,
        "arm_acquisition_predicted_equilibrium_pitch_max_deg": float(
            max(
                equilibrium_pitch_deg(state, position_kinematics, args)
                for state in states
            )
        ),
        "acquisition_attitude_root_model": (
            "yaw_plus_predicted_equilibrium_pitch_physical_cam_fk"
        ),
        "acquisition_anchor_gimbal_ik_error_deg": math.degrees(
            acquisition_gimbal_ik.orientation_error_rad
        ),
    }
    return time_s, states, controls, positions, attitudes, gimbal, diagnostics


def build_com_safe_semantic_prior(
    reference: SemanticReference,
    anchor: np.ndarray,
    source_base_arm_q: np.ndarray,
    position_kinematics: UrdfPositionKinematics,
    args: argparse.Namespace,
) -> np.ndarray:
    """Relabel redundant base/arm poses while preserving semantic tool targets."""

    prior = np.empty_like(source_base_arm_q)
    prior[0] = anchor[:6]
    seeds = (
        np.radians([-45.0, 15.0, -80.0]),
        np.radians([45.0, 15.0, -80.0]),
        np.radians([0.0, -20.0, -100.0]),
    )
    for index in range(1, len(reference.time_s)):
        target_position = reference.positions_m[index]
        source_q = source_base_arm_q[index]
        previous_arm = prior[index - 1, 3:6]

        def candidate(arm: np.ndarray) -> np.ndarray:
            state = source_q.copy()
            state[3:6] = arm
            achieved = position_kinematics.position(state)
            state[:2] += target_position[:2] - achieved[:2]
            return state

        def residual(arm: np.ndarray) -> np.ndarray:
            state = candidate(arm)
            height_error = (
                position_kinematics.position(state)[2] - target_position[2]
            ) / 0.002
            balance_overrun = max(
                equilibrium_pitch_deg(state, position_kinematics, args)
                - (
                    args.maximum_equilibrium_pitch_deg
                    - balance_pitch_optimization_margin_deg(args)
                ),
                0.0,
            ) / 0.01
            gravity = np.abs(
                position_kinematics.gravitational_effort_nm(state)
            )
            gravity_overrun = np.maximum(
                gravity - args.maximum_arm_gravity_effort_nm, 0.0
            ) / 0.02
            return np.concatenate(
                (
                    [height_error, balance_overrun],
                    gravity_overrun,
                    0.02 * (arm - previous_arm),
                    0.005 * (arm - source_q[3:6]),
                )
            )

        solutions = [
            least_squares(
                residual,
                np.clip(seed, position_kinematics.arm_lower, position_kinematics.arm_upper),
                bounds=(position_kinematics.arm_lower, position_kinematics.arm_upper),
                max_nfev=300,
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
            )
            for seed in (previous_arm, source_q[3:6], *seeds)
        ]

        def rank(solution) -> tuple[bool, float]:
            state = candidate(solution.x)
            position_error = abs(
                float(position_kinematics.position(state)[2] - target_position[2])
            )
            balance_pitch = equilibrium_pitch_deg(
                state, position_kinematics, args
            )
            gravity = float(
                np.max(
                    np.abs(position_kinematics.gravitational_effort_nm(state))
                )
            )
            feasible = (
                position_error <= 1e-4
                and balance_pitch
                <= args.maximum_equilibrium_pitch_deg
                + 1e-9
                and gravity
                <= args.maximum_arm_gravity_effort_nm
                + args.gravity_effort_tolerance_nm
            )
            score = (
                position_error / 0.002
                + max(0.0, balance_pitch - args.maximum_equilibrium_pitch_deg)
                + max(0.0, gravity - args.maximum_arm_gravity_effort_nm)
                + 0.05 * float(np.linalg.norm(solution.x - previous_arm))
            )
            return feasible, score

        solution = min(
            solutions, key=lambda item: (not rank(item)[0], rank(item)[1])
        )
        if not rank(solution)[0]:
            state = candidate(solution.x)
            raise ValueError(
                "cannot build COM-safe semantic prior at source index "
                f"{index}: pitch="
                f"{equilibrium_pitch_deg(state, position_kinematics, args):.6f} deg"
            )
        prior[index] = candidate(solution.x)
    return prior


def retarget_semantic_full_pose(
    reference: SemanticReference,
    anchor: np.ndarray,
    source_base_arm_q: np.ndarray,
    position_kinematics: UrdfPositionKinematics,
    camera_kinematics: UrdfPhysicalCameraKinematics,
    args: argparse.Namespace,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
]:
    safe_source_base_arm_q = build_com_safe_semantic_prior(
        reference,
        anchor,
        source_base_arm_q,
        position_kinematics,
        args,
    )
    states = [anchor.copy()]
    controls: list[np.ndarray] = []
    target_positions = [reference.positions_m[0].copy()]
    target_attitudes = [reference.attitudes_wxyz[0].copy()]
    time_s = [0.0]
    position_errors = [
        float(
            np.linalg.norm(
                position_kinematics.position(anchor[:6]) - reference.positions_m[0]
            )
        )
    ]
    attitude_errors = [0.0]
    previous_control = np.zeros(8, dtype=np.float64)
    retimed_interval_count = 0

    for index in range(1, len(reference.time_s)):
        source_dt = float(reference.time_s[index] - reference.time_s[index - 1])
        segment_start_state = states[-1].copy()
        segment_start_control = previous_control.copy()
        attempts = []
        for time_scale in (1, 2, 4, 8, 12, 16, 24):
            trial_state = segment_start_state.copy()
            trial_control = segment_start_control.copy()
            trial_states = []
            trial_controls = []
            trial_positions = []
            trial_attitudes = []
            trial_position_errors = []
            trial_attitude_errors = []
            trial_gravity = []
            trial_balance_pitch = []
            trial_source_arm_errors = []
            for substep in range(1, time_scale + 1):
                fraction = substep / time_scale
                target_position = (
                    (1.0 - fraction) * reference.positions_m[index - 1]
                    + fraction * reference.positions_m[index]
                )
                target_attitude = quaternion_slerp_wxyz(
                    reference.attitudes_wxyz[index - 1],
                    reference.attitudes_wxyz[index],
                    np.array([fraction]),
                )[0]
                source_q = (
                    (1.0 - fraction) * safe_source_base_arm_q[index - 1]
                    + fraction * safe_source_base_arm_q[index]
                )
                target_rotation = quaternion_matrix_wxyz(
                    semantic_dfr_to_physical_cam_quat_wxyz(target_attitude)
                )
                lower = np.concatenate(
                    (
                        [-args.maximum_linear_velocity, -args.maximum_yaw_rate],
                        np.maximum(
                            -args.maximum_arm_rate * source_dt,
                            position_kinematics.arm_lower - trial_state[3:6],
                        ),
                        np.maximum(
                            -args.maximum_gimbal_rate * source_dt,
                            camera_kinematics.gimbal_lower - trial_state[6:9],
                        ),
                    )
                )
                upper = np.concatenate(
                    (
                        [args.maximum_linear_velocity, args.maximum_yaw_rate],
                        np.minimum(
                            args.maximum_arm_rate * source_dt,
                            position_kinematics.arm_upper - trial_state[3:6],
                        ),
                        np.minimum(
                            args.maximum_gimbal_rate * source_dt,
                            camera_kinematics.gimbal_upper - trial_state[6:9],
                        ),
                    )
                )

                def candidate(control: np.ndarray) -> np.ndarray:
                    return np.concatenate(
                        (
                            integrate_unicycle(
                                trial_state[:3], control[0], control[1], source_dt
                            ),
                            trial_state[3:6] + control[2:5],
                            trial_state[6:9] + control[5:8],
                        )
                    )

                def residual(control: np.ndarray) -> np.ndarray:
                    next_state = candidate(control)
                    position_error = (
                        position_kinematics.position(next_state[:6]) - target_position
                    ) / 0.01
                    attitude_error = rotation_error_vector(
                        retarget_solver_camera_rotation(
                            next_state,
                            position_kinematics,
                            camera_kinematics,
                            args,
                        ),
                        target_rotation,
                    ) / math.radians(args.maximum_ik_error_deg)
                    source_delta = next_state[:6] - source_q
                    source_regularization = np.concatenate(
                        (0.1 * source_delta[:3], source_delta[3:6] / 0.35)
                    )
                    gravity_effort = position_kinematics.gravitational_effort_nm(
                        next_state[:6]
                    )
                    gravity_overrun = np.maximum(
                        np.abs(gravity_effort)
                        - args.maximum_arm_gravity_effort_nm,
                        0.0,
                    ) / 0.02
                    balance_pitch_overrun = max(
                        equilibrium_pitch_deg(
                            next_state, position_kinematics, args
                        )
                        - (
                            args.maximum_equilibrium_pitch_deg
                            - balance_pitch_optimization_margin_deg(args)
                        ),
                        0.0,
                    ) / 0.02
                    gimbal_center = 0.5 * (
                        camera_kinematics.gimbal_lower
                        + camera_kinematics.gimbal_upper
                    )
                    gimbal_range = (
                        camera_kinematics.gimbal_upper
                        - camera_kinematics.gimbal_lower
                    )
                    return np.concatenate(
                        (
                            position_error,
                            attitude_error,
                            source_regularization,
                            gravity_overrun,
                            [balance_pitch_overrun],
                            0.002 * control,
                            0.01 * (next_state[6:9] - gimbal_center) / gimbal_range,
                        )
                    )

                def solve(seed: np.ndarray):
                    return least_squares(
                        residual,
                        np.clip(seed, lower, upper),
                        bounds=(lower, upper),
                        max_nfev=120,
                        ftol=1e-9,
                        xtol=1e-9,
                        gtol=1e-9,
                    )

                solutions = [solve(trial_control)]

                def solution_metrics(
                    solution,
                ) -> tuple[float, float, float, float]:
                    next_state = candidate(solution.x)
                    position_error = float(
                        np.linalg.norm(
                            position_kinematics.position(next_state[:6])
                            - target_position
                        )
                    )
                    attitude_error = math.degrees(
                        float(
                            np.linalg.norm(
                                rotation_error_vector(
                                    retarget_solver_camera_rotation(
                                        next_state,
                                        position_kinematics,
                                        camera_kinematics,
                                        args,
                                    ),
                                    target_rotation,
                                )
                            )
                        )
                    )
                    gravity = float(
                        np.max(
                            np.abs(
                                position_kinematics.gravitational_effort_nm(
                                    next_state[:6]
                                )
                            )
                        )
                    )
                    balance_pitch = equilibrium_pitch_deg(
                        next_state, position_kinematics, args
                    )
                    return position_error, attitude_error, gravity, balance_pitch

                initial_metrics = solution_metrics(solutions[0])
                if not (
                    initial_metrics[0] <= 0.05
                    and initial_metrics[1] <= args.maximum_ik_error_deg
                    and initial_metrics[2]
                    <= args.maximum_arm_gravity_effort_nm
                    + args.gravity_effort_tolerance_nm
                    and initial_metrics[3]
                    <= args.maximum_equilibrium_pitch_deg
                    + BALANCE_PITCH_SOLVER_TOLERANCE_DEG
                ):
                    source_seed = np.zeros(8, dtype=np.float64)
                    source_delta_xy = source_q[:2] - trial_state[:2]
                    heading = np.array(
                        [math.cos(trial_state[2]), math.sin(trial_state[2])]
                    )
                    source_seed[0] = np.dot(source_delta_xy, heading) / source_dt
                    source_seed[1] = (
                        wrap_angle(float(source_q[2] - trial_state[2])) / source_dt
                    )
                    source_seed[2:5] = source_q[3:6] - trial_state[3:6]
                    recovery_seeds = [np.zeros(8), source_seed]
                    tool_error = (
                        target_position
                        - position_kinematics.position(trial_state[:6])
                    )
                    if np.linalg.norm(tool_error[:2]) > 1e-8:
                        target_heading = math.atan2(tool_error[1], tool_error[0])
                        for direction, heading_offset in (
                            (1.0, 0.0),
                            (-1.0, math.pi),
                        ):
                            seed = source_seed.copy()
                            desired_heading = wrap_angle(
                                target_heading + heading_offset
                            )
                            seed[0] = direction * min(
                                args.maximum_linear_velocity,
                                np.linalg.norm(tool_error[:2]) / source_dt,
                            )
                            seed[1] = (
                                wrap_angle(desired_heading - trial_state[2])
                                / source_dt
                            )
                            recovery_seeds.append(seed)
                    for seed in recovery_seeds:
                        seed[5:8] = trial_control[5:8]
                        if not any(
                            np.allclose(seed, existing.x, atol=1e-8, rtol=0.0)
                            for existing in solutions
                        ):
                            solutions.append(solve(seed))

                def rank(solution) -> tuple[bool, float]:
                    (
                        position_error,
                        attitude_error,
                        gravity,
                        balance_pitch,
                    ) = solution_metrics(solution)
                    feasible = (
                        position_error <= 0.05
                        and attitude_error <= args.maximum_ik_error_deg
                        and gravity
                        <= args.maximum_arm_gravity_effort_nm
                        + args.gravity_effort_tolerance_nm
                        and balance_pitch
                        <= args.maximum_equilibrium_pitch_deg
                        + BALANCE_PITCH_SOLVER_TOLERANCE_DEG
                    )
                    score = (
                        position_error / 0.02
                        + attitude_error / args.maximum_ik_error_deg
                        + max(
                            0.0,
                            gravity - args.maximum_arm_gravity_effort_nm,
                        )
                        + max(
                            0.0,
                            balance_pitch - args.maximum_equilibrium_pitch_deg,
                        )
                    )
                    return feasible, score

                solution = min(
                    solutions,
                    key=lambda item: (not rank(item)[0], rank(item)[1]),
                )
                trial_control = solution.x
                trial_state = candidate(solution.x)
                position_error = float(
                    np.linalg.norm(
                        position_kinematics.position(trial_state[:6])
                        - target_position
                    )
                )
                attitude_error = math.degrees(
                    float(
                        np.linalg.norm(
                            rotation_error_vector(
                                retarget_solver_camera_rotation(
                                    trial_state,
                                    position_kinematics,
                                    camera_kinematics,
                                    args,
                                ),
                                target_rotation,
                            )
                        )
                    )
                )
                gravity_effort = float(
                    np.max(
                        np.abs(
                            position_kinematics.gravitational_effort_nm(
                                trial_state[:6]
                            )
                        )
                    )
                )
                balance_pitch = equilibrium_pitch_deg(
                    trial_state, position_kinematics, args
                )
                trial_states.append(trial_state.copy())
                trial_controls.append(
                    np.concatenate((solution.x[:2], solution.x[2:5]))
                )
                trial_positions.append(target_position)
                trial_attitudes.append(target_attitude)
                trial_position_errors.append(position_error)
                trial_attitude_errors.append(attitude_error)
                trial_gravity.append(gravity_effort)
                trial_balance_pitch.append(balance_pitch)
                trial_source_arm_errors.append(
                    float(np.linalg.norm(trial_state[3:6] - source_q[3:6]))
                )

            feasible = (
                max(trial_position_errors) <= 0.05
                and max(trial_attitude_errors) <= args.maximum_ik_error_deg
                and max(trial_gravity)
                <= args.maximum_arm_gravity_effort_nm
                + args.gravity_effort_tolerance_nm
                and max(trial_balance_pitch)
                <= args.maximum_equilibrium_pitch_deg
                + BALANCE_PITCH_SOLVER_TOLERANCE_DEG
            )
            score = (
                max(trial_position_errors) / 0.02
                + max(trial_attitude_errors) / args.maximum_ik_error_deg
                + max(trial_source_arm_errors) / 0.2
                + max(
                    0.0,
                    max(trial_gravity) - args.maximum_arm_gravity_effort_nm,
                )
                + max(
                    0.0,
                    max(trial_balance_pitch)
                    - args.maximum_equilibrium_pitch_deg,
                )
                + 0.01 * time_scale
            )
            attempts.append(
                (
                    feasible,
                    score,
                    trial_control,
                    trial_states,
                    trial_controls,
                    trial_positions,
                    trial_attitudes,
                    trial_position_errors,
                    trial_attitude_errors,
                )
            )
            if feasible:
                break

        feasible_attempts = [item for item in attempts if item[0]]
        if not feasible_attempts:
            best = min(attempts, key=lambda item: item[1])
            best_states = best[3]
            best_gravity = max(
                float(
                    np.max(
                        np.abs(
                            position_kinematics.gravitational_effort_nm(
                                state[:6]
                            )
                        )
                    )
                )
                for state in best_states
            )
            best_pitch = max(
                equilibrium_pitch_deg(state, position_kinematics, args)
                for state in best_states
            )
            raise ValueError(
                f"semantic interval {index} has no COM-safe nonholonomic solve: "
                f"position_max={max(best[7]):.6f} m, "
                f"attitude_max={max(best[8]):.6f} deg, "
                f"gravity_max={best_gravity:.6f} Nm, "
                f"equilibrium_pitch_max={best_pitch:.9f} deg"
            )
        selected = min(feasible_attempts, key=lambda item: item[1])
        (
            _,
            _,
            previous_control,
            selected_states,
            selected_controls,
            selected_positions,
            selected_attitudes,
            selected_position_errors,
            selected_attitude_errors,
        ) = selected
        if len(selected_states) > 1:
            retimed_interval_count += 1
        print(
            f"[case {reference.case}] semantic interval {index}/"
            f"{len(reference.time_s) - 1}: retime={len(selected_states)}, "
            f"position_max={max(selected_position_errors):.6f} m, "
            f"attitude_max={max(selected_attitude_errors):.6f} deg",
            flush=True,
        )
        for state, control, position, attitude, position_error, attitude_error in zip(
            selected_states,
            selected_controls,
            selected_positions,
            selected_attitudes,
            selected_position_errors,
            selected_attitude_errors,
            strict=True,
        ):
            states.append(state)
            controls.append(control)
            target_positions.append(position)
            target_attitudes.append(attitude)
            position_errors.append(position_error)
            attitude_errors.append(attitude_error)
            time_s.append(time_s[-1] + source_dt)

    state_array = np.asarray(states)
    return (
        np.asarray(time_s),
        state_array,
        np.asarray(controls),
        np.asarray(target_positions),
        np.asarray(target_attitudes),
        np.asarray(position_errors),
        np.asarray(attitude_errors),
        state_array[:, 6:9],
        retimed_interval_count,
    )


def retarget_case(
    teacher: SparseTeacher,
    reference: SemanticReference,
    position_kinematics: UrdfPositionKinematics,
    camera_kinematics: UrdfPhysicalCameraKinematics,
    args: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    anchor, anchor_position_error, anchor_attitude_error = solve_full_pose_anchor(
        teacher.base_arm_q[0],
        reference.positions_m[0],
        reference.attitudes_wxyz[0],
        position_kinematics,
        camera_kinematics,
        args,
    )
    (
        acquisition_time,
        acquisition_states,
        acquisition_controls,
        acquisition_positions,
        acquisition_attitudes,
        _,
        acquisition_diagnostics,
    ) = build_feasible_acquisition(
        anchor,
        reference.attitudes_wxyz[0],
        position_kinematics,
        camera_kinematics,
        args,
    )
    (
        semantic_time,
        semantic_states,
        semantic_controls,
        semantic_target_positions,
        semantic_target_attitudes,
        semantic_position_errors,
        semantic_attitude_errors,
        _,
        semantic_retimed_interval_count,
    ) = retarget_semantic_full_pose(
        reference,
        anchor,
        teacher.base_arm_q,
        position_kinematics,
        camera_kinematics,
        args,
    )
    time_s = np.concatenate(
        (acquisition_time, acquisition_time[-1] + semantic_time[1:])
    )
    internal_states = np.vstack((acquisition_states, semantic_states[1:]))
    controls = np.vstack((acquisition_controls, semantic_controls))
    targets = np.vstack((acquisition_positions, semantic_target_positions[1:]))
    attitudes = np.vstack((acquisition_attitudes, semantic_target_attitudes[1:]))
    (
        time_s,
        internal_states,
        controls,
        equilibrium_pitch_gimbal_diagnostics,
    ) = retime_gimbal_for_equilibrium_pitch(
        time_s,
        internal_states,
        attitudes,
        controls,
        position_kinematics,
        camera_kinematics,
        args,
    )
    states = internal_states[:, :6]
    achieved = np.asarray(
        [position_kinematics.position(state) for state in states]
    )
    errors = np.linalg.norm(achieved - targets, axis=1)
    gravity_efforts = np.asarray(
        [position_kinematics.gravitational_effort_nm(state) for state in states]
    )
    maximum_gravity_effort = float(np.max(np.abs(gravity_efforts)))
    equilibrium_pitches_deg = np.asarray(
        [equilibrium_pitch_deg(state, position_kinematics, args) for state in states]
    )
    maximum_equilibrium_pitch_deg = float(np.max(equilibrium_pitches_deg))
    acquisition_steps = len(acquisition_time) - 1
    gimbal = {
        "full_pose_anchor_position_error_m": anchor_position_error,
        "full_pose_anchor_attitude_error_deg": anchor_attitude_error,
        "semantic_retimed_interval_count": semantic_retimed_interval_count,
        **equilibrium_pitch_gimbal_diagnostics,
        **acquisition_diagnostics,
    }
    checks = {
        "exact_source_v1_contract_verified": True,
        "source_ee1_tool_fk_verified": reference.source_fk_max_error_m <= 1e-6,
        "position_p95_bounded": float(np.percentile(errors, 95))
        <= args.maximum_position_p95_m,
        "position_maximum_bounded": float(np.max(errors))
        <= args.maximum_position_error_m,
        "all_physical_gimbal_ik_converged": not gimbal[
            "physical_gimbal_ik_nonconverged_indices"
        ],
        "physical_gimbal_ik_error_bounded": gimbal[
            "physical_gimbal_ik_max_error_deg"
        ]
        <= args.maximum_ik_error_deg,
        "physical_gimbal_rate_bounded": gimbal[
            "physical_gimbal_rate_max_radps"
        ]
        <= args.maximum_gimbal_rate + 1e-9,
        "physical_gimbal_interpolation_error_bounded": gimbal[
            "physical_gimbal_interpolation_max_error_deg"
        ]
        <= args.maximum_gimbal_interpolation_error_deg,
        "arm_gravity_effort_bounded": maximum_gravity_effort
        <= args.maximum_arm_gravity_effort_nm
        + args.gravity_effort_tolerance_nm,
        "equilibrium_pitch_bounded": maximum_equilibrium_pitch_deg
        <= args.maximum_equilibrium_pitch_deg
        + BALANCE_PITCH_OUTPUT_TOLERANCE_DEG,
        "physical_gimbal_joint_labels_not_exported": True,
        "runtime_approval_remains_false": True,
        "training_not_started": True,
    }
    summary = {
        "case": teacher.case,
        "source_samples": len(reference.time_s),
        "candidate_samples": len(time_s),
        "source_duration_s": float(reference.time_s[-1]),
        "acquisition_duration_s": float(time_s[acquisition_steps]),
        "acquisition_steps": acquisition_steps,
        "position_error_mean_m": float(np.mean(errors)),
        "position_error_p95_m": float(np.percentile(errors, 95)),
        "position_error_max_m": float(np.max(errors)),
        "semantic_position_error_p95_m": float(
            np.percentile(errors[acquisition_steps:], 95)
        ),
        "semantic_position_error_max_m": float(
            np.max(errors[acquisition_steps:])
        ),
        "source_fk_max_error_m": reference.source_fk_max_error_m,
        "package_q_max_error_rad": reference.package_q_max_error_rad,
        "package_position_max_error_m": reference.package_position_max_error_m,
        "package_time_max_error_s": reference.package_time_max_error_s,
        "package_attitude_max_error_deg": reference.package_attitude_max_error_deg,
        "maximum_arm_gravity_effort_nm": maximum_gravity_effort,
        "gravity_effort_tolerance_nm": args.gravity_effort_tolerance_nm,
        "maximum_equilibrium_pitch_deg": maximum_equilibrium_pitch_deg,
        "equilibrium_pitch_limit_deg": args.maximum_equilibrium_pitch_deg,
        "equilibrium_pitch_numerical_tolerance_deg": (
            BALANCE_PITCH_OUTPUT_TOLERANCE_DEG
        ),
        "equilibrium_pitch_optimization_margin_deg": (
            balance_pitch_optimization_margin_deg(args)
        ),
        "camera_solve_root_model": getattr(
            args, "camera_solve_root_model", "balanced"
        ),
        "physical_camera_final_gate_root_model": (
            "yaw_plus_predicted_equilibrium_pitch"
        ),
        **gimbal,
        "checks": checks,
        "passed": all(checks.values()),
    }
    arrays = {
        "schema": np.asarray(CANDIDATE_SCHEMA),
        "trajectory_integrity_contract": np.asarray("exact_source_v1"),
        "source_trajectory_integrity_passed": np.bool_(True),
        "source_teacher_quality_passed": np.bool_(True),
        "valid_for_candidate_training": np.bool_(True),
        "case": np.int32(teacher.case),
        "runtime_approved": np.bool_(False),
        "training_started": np.bool_(False),
        "position_target_link": np.asarray("ee1_tool"),
        "attitude_target_contract": np.asarray(
            "world_semantic_DFR_quaternion_wxyz_option_B"
        ),
        "physical_gimbal_joint_labels_included": np.bool_(False),
        "camera_solve_root_model": np.asarray(
            getattr(args, "camera_solve_root_model", "balanced")
        ),
        "time_s": time_s,
        "source_time_s": reference.time_s,
        "semantic_start_index": np.int32(acquisition_steps),
        "target_position_world_m": targets,
        "target_attitude_world_dfr_quat_wxyz": attitudes,
        "achieved_position_world_m": achieved,
        "base_arm_q": states,
        "control_v_wz_darm": controls,
        "position_error_m": errors,
        "source_teacher_sha256": np.asarray(sha256(teacher.path)),
        "source_mat_sha256": np.asarray(sha256(reference.source_mat)),
    }
    forbidden = FORBIDDEN_EXPORT_KEYS & set(arrays)
    if forbidden:
        raise AssertionError(f"physical gimbal labels leaked into candidate: {forbidden}")
    return summary, arrays


def process_case(
    case: int,
    teacher: SparseTeacher,
    args: argparse.Namespace,
) -> dict[str, object]:
    try:
        source_kinematics = UrdfPositionKinematics(
            args.source_urdf.resolve(),
            passive_joint_positions=SOURCE_PASSIVE_DFR_JOINTS,
        )
        position_kinematics = UrdfPositionKinematics(args.target_urdf.resolve())
        camera_kinematics = UrdfPhysicalCameraKinematics(args.target_urdf.resolve())
        source_mat = args.source_batch / f"episode_{case:04d}" / "teacher_smoke.mat"
        reference = load_semantic_reference(
            teacher, source_mat.resolve(), source_kinematics
        )
        attempt_errors = []
        selected = None
        camera_solve_root_models = (
            ("balanced", "upright")
            if args.camera_solve_root_model == "auto"
            else (args.camera_solve_root_model,)
        )
        for camera_solve_root_model in camera_solve_root_models:
            attempt_args = argparse.Namespace(**vars(args))
            attempt_args.camera_solve_root_model = camera_solve_root_model
            try:
                summary, arrays = retarget_case(
                    teacher,
                    reference,
                    position_kinematics,
                    camera_kinematics,
                    attempt_args,
                )
            except Exception as error:
                attempt_errors.append(
                    {
                        "camera_solve_root_model": camera_solve_root_model,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                continue
            if summary["passed"]:
                selected = (summary, arrays, camera_solve_root_model)
                break
            attempt_errors.append(
                {
                    "camera_solve_root_model": camera_solve_root_model,
                    "error_type": "FailedChecks",
                    "error": ",".join(
                        name for name, passed in summary["checks"].items() if not passed
                    ),
                }
            )
        if selected is None:
            summary = {
                "case": case,
                "passed": False,
                "error_type": "RetargetAttemptsFailed",
                "error": "camera-root solve attempts failed",
                "attempts": attempt_errors,
            }
        else:
            summary, arrays, camera_solve_root_model = selected
            summary["camera_solve_root_model"] = camera_solve_root_model
            summary["prior_attempts"] = attempt_errors
            np.savez_compressed(args.output_dir / f"case_{case:04d}.npz", **arrays)
        (args.output_dir / f"case_{case:04d}.result.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return summary
    except Exception as error:
        summary = {
            "case": case,
            "passed": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        (args.output_dir / f"case_{case:04d}.result.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return summary


def main() -> int:
    args = parse_args()
    teachers = discover_v3_package(args.teacher_package)
    cases = (
        sorted(teachers)
        if args.cases.strip().lower() == "all"
        else [int(value) for value in args.cases.split(",") if value.strip()]
    )
    if not cases or any(case not in teachers for case in cases):
        raise ValueError(f"invalid cases: {cases}")
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    if args.jobs == 1:
        for case in cases:
            summary = process_case(case, teachers[case], args)
            rows.append(summary)
            print(json.dumps(summary, indent=2), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=min(args.jobs, len(cases))) as executor:
            futures = {
                executor.submit(process_case, case, teachers[case], args): case
                for case in cases
            }
            for future in as_completed(futures):
                summary = future.result()
                rows.append(summary)
                print(json.dumps(summary, indent=2), flush=True)
        rows.sort(key=lambda row: int(row["case"]))

    def maximum(key: str) -> float | None:
        values = [float(row[key]) for row in rows if key in row]
        return max(values) if values else None

    result = {
        "schema": "cinebotrl_two_wheel_corrected_semantic_retarget_batch_v3",
        "candidate_schema": CANDIDATE_SCHEMA,
        "training_started": False,
        "runtime_approved": False,
        "source_teacher_package": str(args.teacher_package.resolve()),
        "source_package_sha256": sha256(args.teacher_package.resolve() / "manifest.json"),
        "trajectory_integrity_contract": "exact_source_v1",
        "source_trajectory_integrity_passed": True,
        "source_teacher_quality_passed": True,
        "source_package_case_count": len(teachers),
        "cases": cases,
        "passed_case_count": sum(row["passed"] for row in rows),
        "physical_gimbal_joint_labels_exported": False,
        "physical_gimbal_path_use": "internal_feasibility_and_retiming_only",
        "orientation_target_contract": "semantic_DFR_world_quaternion_wxyz_option_B",
        "maximum_position_error_m": maximum("position_error_max_m"),
        "maximum_physical_gimbal_ik_error_deg": maximum(
            "physical_gimbal_ik_max_error_deg"
        ),
        "maximum_physical_gimbal_rate_radps": maximum(
            "physical_gimbal_rate_max_radps"
        ),
        "maximum_arm_gravity_effort_nm": maximum(
            "maximum_arm_gravity_effort_nm"
        ),
        "passed": all(row["passed"] for row in rows),
        "results": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    flat_rows = [
        {key: value for key, value in row.items() if key != "checks"}
        for row in rows
    ]
    fieldnames = list(
        dict.fromkeys(key for row in flat_rows for key in row)
    )
    with (args.output_dir / "cases.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(flat_rows)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
