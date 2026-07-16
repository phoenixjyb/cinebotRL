"""Full-pose reference decomposition through the riser RS4 attitude proxy."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .camera_attitude import (
    quaternion_matrix_wxyz,
    rotation_error_vector,
    semantic_dfr_to_physical_cam_quat_wxyz,
)
from .riser_kinematics import UrdfRiserCameraKinematics
from .riser_reference import CorrectedRiserReference, RiserKinematicPlan
from .riser_rs4_attitude import (
    RS4_FILMING_RATE_LIMIT_DEG_S,
    accepted62_body_basis_rotation,
    bounded_path_yaw_schedule,
    plan_rs4_attitude_commands,
    proxy_joint_rates_rad_s,
    resolve_rs4_position_command,
    rs4_command_to_proxy_joint_order,
    unwrap_proxy_joint_yaw,
)
from .whole_body_kinematics import integrate_unicycle


RS4_PROXY_RATE_PLANNING_MARGIN = 0.995


def _plan_joint_rs4_riser_reference(
    reference: CorrectedRiserReference,
    kinematics: UrdfRiserCameraKinematics,
    *,
    vertical_shift_m: float = 0.0,
    maximum_base_yaw_rate_rad_s: float = 0.4,
    maximum_base_linear_velocity_mps: float = 0.4,
    maximum_riser_rate_mps: float = 1.0,
    maximum_proxy_joint_rate_rad_s: float = math.radians(
        RS4_FILMING_RATE_LIMIT_DEG_S
    ),
    position_scale_m: float = 0.05,
    attitude_tolerance_rad: float = math.radians(2.0),
) -> RiserKinematicPlan:
    """Jointly resolve semantic pose into bounded unicycle and RS4 commands."""

    if vertical_shift_m < 0.0 or not math.isfinite(vertical_shift_m):
        raise ValueError("vertical shift must be finite and non-negative")
    if attitude_tolerance_rad <= 0.0 or not math.isfinite(attitude_tolerance_rad):
        raise ValueError("attitude tolerance must be finite and positive")
    if min(
        maximum_base_linear_velocity_mps,
        maximum_base_yaw_rate_rad_s,
        maximum_riser_rate_mps,
        maximum_proxy_joint_rate_rad_s,
        position_scale_m,
    ) <= 0.0:
        raise ValueError("pose planning limits must be positive")
    target_position = reference.positions_m.copy()
    target_position[:, 2] += vertical_shift_m

    count = len(reference.time_s)
    base = np.empty((count, 3), dtype=np.float64)
    riser = np.empty(count, dtype=np.float64)
    proxy_joint = np.empty((count, 3), dtype=np.float64)
    achieved = np.empty_like(target_position)
    attitude_error = np.empty(count, dtype=np.float64)
    attitude_converged = np.empty(count, dtype=bool)
    body_basis = accepted62_body_basis_rotation()
    initial_yaw = reference.initial_base_yaw_rad
    initial_world_basis = (
        Rotation.from_euler("z", initial_yaw).as_matrix() @ body_basis
    )
    previous_command, initial_feasible = resolve_rs4_position_command(
        initial_world_basis,
        quaternion_matrix_wxyz(reference.semantic_dfr_quat_wxyz[0]),
        np.zeros(3),
    )
    proxy_joint[0] = rs4_command_to_proxy_joint_order(previous_command)
    initial = kinematics.solve_position(
        target_position[0], initial_yaw, proxy_joint[0]
    )
    base[0] = initial.base_xy_yaw_riser[:3]
    riser[0] = initial.base_xy_yaw_riser[3]
    previous_control = np.zeros(3, dtype=np.float64)
    for index in range(count):
        command_feasible = initial_feasible
        if index > 0:
            dt = float(reference.time_s[index] - reference.time_s[index - 1])
            riser_delta_limit = maximum_riser_rate_mps * dt
            lower = np.array(
                [
                    -maximum_base_linear_velocity_mps,
                    -maximum_base_yaw_rate_rad_s,
                    max(riser_delta_limit * -1.0, kinematics.riser_lower - riser[index - 1]),
                ]
            )
            upper = np.array(
                [
                    maximum_base_linear_velocity_mps,
                    maximum_base_yaw_rate_rad_s,
                    min(riser_delta_limit, kinematics.riser_upper - riser[index - 1]),
                ]
            )

            def candidate(
                control: np.ndarray,
            ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, bool]:
                next_base = integrate_unicycle(
                    base[index - 1], float(control[0]), float(control[1]), dt
                )
                next_riser = float(riser[index - 1] + control[2])
                world_basis = (
                    Rotation.from_euler("z", next_base[2]).as_matrix()
                    @ body_basis
                )
                command, feasible = resolve_rs4_position_command(
                    world_basis,
                    quaternion_matrix_wxyz(
                        reference.semantic_dfr_quat_wxyz[index]
                    ),
                    previous_command,
                )
                next_proxy = rs4_command_to_proxy_joint_order(command)
                yaw_delta = (
                    next_proxy[2] - proxy_joint[index - 1, 2] + math.pi
                ) % (2.0 * math.pi) - math.pi
                next_proxy[2] = proxy_joint[index - 1, 2] + yaw_delta
                transform = kinematics.world_transform(
                    next_base, next_riser, next_proxy
                )
                return (
                    next_base,
                    next_riser,
                    command,
                    next_proxy,
                    transform,
                    feasible,
                )

            def residual(control: np.ndarray) -> np.ndarray:
                _, _, _, next_proxy, transform, _ = candidate(control)
                position_error = (
                    transform[:3, 3] - target_position[index]
                ) / position_scale_m
                proxy_delta = next_proxy - proxy_joint[index - 1]
                proxy_delta[2] = (
                    proxy_delta[2] + math.pi
                ) % (2.0 * math.pi) - math.pi
                proxy_rate_excess = np.maximum(
                    np.abs(proxy_delta) / dt - maximum_proxy_joint_rate_rad_s,
                    0.0,
                )
                regularization = 0.001 * np.array(
                    [
                        control[0] / maximum_base_linear_velocity_mps,
                        control[1] / maximum_base_yaw_rate_rad_s,
                        control[2] / max(riser_delta_limit, 1e-9),
                    ]
                )
                return np.r_[
                    position_error,
                    proxy_rate_excess / maximum_proxy_joint_rate_rad_s,
                    regularization,
                ]

            solution = least_squares(
                residual,
                np.clip(previous_control, lower, upper),
                bounds=(lower, upper),
                max_nfev=60,
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8,
            )
            previous_control = solution.x
            (
                base[index],
                riser[index],
                previous_command,
                proxy_joint[index],
                _,
                command_feasible,
            ) = candidate(solution.x)
        transform = kinematics.world_transform(
            base[index], riser[index], proxy_joint[index]
        )
        achieved[index] = transform[:3, 3]
        physical_target = quaternion_matrix_wxyz(
            semantic_dfr_to_physical_cam_quat_wxyz(
                reference.semantic_dfr_quat_wxyz[index]
            )
        )
        attitude_error[index] = float(
            np.linalg.norm(rotation_error_vector(transform[:3, :3], physical_target))
        )
        attitude_converged[index] = (
            command_feasible
            and attitude_error[index] <= attitude_tolerance_rad
        )

    return RiserKinematicPlan(
        time_s=reference.time_s.copy(),
        targets_m=target_position,
        base_xy_yaw=base,
        riser_q=riser,
        gimbal_q=proxy_joint,
        achieved_m=achieved,
        attitude_error_rad=attitude_error,
        attitude_converged=attitude_converged,
        vertical_shift_m=vertical_shift_m,
        planning_strategy="joint_adaptive",
    )


def _plan_fixed_path_rs4_riser_reference(
    reference: CorrectedRiserReference,
    kinematics: UrdfRiserCameraKinematics,
    *,
    vertical_shift_m: float,
    maximum_base_yaw_rate_rad_s: float,
    maximum_base_linear_velocity_mps: float,
    maximum_riser_rate_mps: float,
    position_scale_m: float,
    attitude_tolerance_rad: float,
) -> RiserKinematicPlan:
    """Track the bounded bidirectional path heading without local yaw search."""

    target_position = reference.positions_m.copy()
    target_position[:, 2] += vertical_shift_m
    base_yaw = bounded_path_yaw_schedule(
        reference, maximum_yaw_rate_rad_s=maximum_base_yaw_rate_rad_s
    )
    attitude = plan_rs4_attitude_commands(
        reference, accepted62_body_basis_rotation(), base_yaw
    )
    proxy_joint = rs4_command_to_proxy_joint_order(
        attitude.command_yaw_roll_pitch_rad
    )
    proxy_joint = unwrap_proxy_joint_yaw(proxy_joint)
    count = len(reference.time_s)
    base = np.empty((count, 3), dtype=np.float64)
    riser = np.empty(count, dtype=np.float64)
    achieved = np.empty_like(target_position)
    attitude_error = np.empty(count, dtype=np.float64)
    attitude_converged = np.empty(count, dtype=bool)
    initial = kinematics.solve_position(
        target_position[0], base_yaw[0], proxy_joint[0]
    )
    base[0] = initial.base_xy_yaw_riser[:3]
    riser[0] = initial.base_xy_yaw_riser[3]
    previous_control = np.zeros(2, dtype=np.float64)
    for index in range(count):
        if index > 0:
            dt = float(reference.time_s[index] - reference.time_s[index - 1])
            yaw_rate = (base_yaw[index] - base_yaw[index - 1]) / dt
            riser_delta_limit = maximum_riser_rate_mps * dt
            lower = np.array(
                [
                    -maximum_base_linear_velocity_mps,
                    max(
                        -riser_delta_limit,
                        kinematics.riser_lower - riser[index - 1],
                    ),
                ]
            )
            upper = np.array(
                [
                    maximum_base_linear_velocity_mps,
                    min(
                        riser_delta_limit,
                        kinematics.riser_upper - riser[index - 1],
                    ),
                ]
            )

            def candidate(control: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
                next_base = integrate_unicycle(
                    base[index - 1], float(control[0]), float(yaw_rate), dt
                )
                next_riser = float(riser[index - 1] + control[1])
                transform = kinematics.world_transform(
                    next_base, next_riser, proxy_joint[index]
                )
                return next_base, next_riser, transform

            def residual(control: np.ndarray) -> np.ndarray:
                _, _, transform = candidate(control)
                position_error = (
                    transform[:3, 3] - target_position[index]
                ) / position_scale_m
                regularization = 0.001 * np.array(
                    [
                        control[0] / maximum_base_linear_velocity_mps,
                        control[1] / max(riser_delta_limit, 1e-9),
                    ]
                )
                return np.r_[position_error, regularization]

            solution = least_squares(
                residual,
                np.clip(previous_control, lower, upper),
                bounds=(lower, upper),
                max_nfev=30,
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8,
            )
            previous_control = solution.x
            base[index], riser[index], _ = candidate(solution.x)
        transform = kinematics.world_transform(
            base[index], riser[index], proxy_joint[index]
        )
        achieved[index] = transform[:3, 3]
        physical_target = quaternion_matrix_wxyz(
            semantic_dfr_to_physical_cam_quat_wxyz(
                reference.semantic_dfr_quat_wxyz[index]
            )
        )
        attitude_error[index] = float(
            np.linalg.norm(
                rotation_error_vector(transform[:3, :3], physical_target)
            )
        )
        attitude_converged[index] = (
            attitude.command_feasible[index]
            and attitude_error[index] <= attitude_tolerance_rad
        )

    return RiserKinematicPlan(
        time_s=reference.time_s.copy(),
        targets_m=target_position,
        base_xy_yaw=base,
        riser_q=riser,
        gimbal_q=proxy_joint,
        achieved_m=achieved,
        attitude_error_rad=attitude_error,
        attitude_converged=attitude_converged,
        vertical_shift_m=vertical_shift_m,
        planning_strategy="fixed_path",
    )


def plan_rs4_riser_reference(
    reference: CorrectedRiserReference,
    kinematics: UrdfRiserCameraKinematics,
    *,
    vertical_shift_m: float = 0.0,
    maximum_base_yaw_rate_rad_s: float = 0.4,
    maximum_base_linear_velocity_mps: float = 0.4,
    maximum_riser_rate_mps: float = 1.0,
    maximum_proxy_joint_rate_rad_s: float = math.radians(
        RS4_FILMING_RATE_LIMIT_DEG_S
    ),
    position_scale_m: float = 0.05,
    attitude_tolerance_rad: float = math.radians(2.0),
) -> RiserKinematicPlan:
    """Select the safest deterministic fixed-path or joint-adaptive plan."""

    shared = {
        "vertical_shift_m": vertical_shift_m,
        "maximum_base_linear_velocity_mps": maximum_base_linear_velocity_mps,
        "maximum_riser_rate_mps": maximum_riser_rate_mps,
        "position_scale_m": position_scale_m,
        "attitude_tolerance_rad": attitude_tolerance_rad,
    }
    fixed = _plan_fixed_path_rs4_riser_reference(
        reference,
        kinematics,
        maximum_base_yaw_rate_rad_s=maximum_base_yaw_rate_rad_s,
        **shared,
    )
    joint = _plan_joint_rs4_riser_reference(
        reference,
        kinematics,
        maximum_base_yaw_rate_rad_s=maximum_base_yaw_rate_rad_s,
        maximum_proxy_joint_rate_rad_s=(
            maximum_proxy_joint_rate_rad_s * RS4_PROXY_RATE_PLANNING_MARGIN
        ),
        **shared,
    )

    def rank(plan: RiserKinematicPlan) -> tuple[bool, bool, float, float]:
        proxy_rate = proxy_joint_rates_rad_s(plan.gimbal_q, plan.time_s)
        position_error = np.linalg.norm(plan.achieved_m - plan.targets_m, axis=1)
        return (
            not bool(np.all(plan.attitude_converged)),
            float(np.max(np.abs(proxy_rate)))
            > maximum_proxy_joint_rate_rad_s + 1e-9,
            float(np.percentile(position_error, 95)),
            float(np.max(position_error)),
        )

    return min((fixed, joint), key=rank)
