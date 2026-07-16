"""Bounded task-space feedback for scripted two-wheel whole-body playback."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .whole_body_kinematics import UrdfPositionKinematics


def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def continuous_joint_error(target: float, measured: float) -> float:
    """Return the shortest signed error for a continuous revolute joint."""

    if not math.isfinite(target) or not math.isfinite(measured):
        raise ValueError("continuous joint angles must be finite")
    return wrap_to_pi(target - measured)


def nearest_equivalent_angle(target: float, reference: float) -> float:
    """Represent ``target`` on the nearest 2-pi branch to ``reference``."""

    return reference + continuous_joint_error(target, reference)


def yaw_from_quaternion_wxyz(quaternion: np.ndarray) -> float:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError(f"expected finite quaternion shape (4,), got {quaternion.shape}")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("quaternion norm is zero")
    w, x, y, z = quaternion / norm
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def rotation_matrix_from_quaternion_wxyz(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError(f"expected finite quaternion shape (4,), got {quaternion.shape}")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("quaternion norm is zero")
    w, x, y, z = quaternion / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def equilibrium_pitch_from_world_com(
    root_position_world_m: np.ndarray,
    root_quaternion_wxyz: np.ndarray,
    center_of_mass_world_m: np.ndarray,
    wheel_axle_height_m: float,
) -> tuple[float, np.ndarray]:
    root_position_world_m = np.asarray(root_position_world_m, dtype=np.float64)
    center_of_mass_world_m = np.asarray(center_of_mass_world_m, dtype=np.float64)
    if root_position_world_m.shape != (3,) or center_of_mass_world_m.shape != (3,):
        raise ValueError("root and COM positions must have shape (3,)")
    if wheel_axle_height_m <= 0.0:
        raise ValueError("wheel axle height must be positive")
    rotation_world_root = rotation_matrix_from_quaternion_wxyz(root_quaternion_wxyz)
    com_root = rotation_world_root.T @ (
        center_of_mass_world_m - root_position_world_m
    )
    com_from_axle = com_root - np.array([0.0, 0.0, wheel_axle_height_m])
    pitch = -math.atan2(float(com_from_axle[0]), float(com_from_axle[2]))
    return pitch, com_from_axle


@dataclass(frozen=True)
class WholeBodyTrackingConfig:
    along_track_kp: float = 0.8
    cross_track_kp: float = 1.2
    yaw_kp: float = 1.0
    maximum_linear_velocity_mps: float = 0.4
    maximum_yaw_rate_radps: float = 0.4
    ik_damping: float = 0.04
    ik_task_gain: float = 0.7
    nominal_arm_pull: float = 0.15
    maximum_arm_target_correction_rad: float = 0.35
    maximum_arm_target_rate_radps: float = 0.5
    progress_error_start_m: float = 0.05
    progress_error_full_m: float = 0.25
    minimum_progress_scale: float = 0.1


def riser_tracking_config(**overrides: float) -> WholeBodyTrackingConfig:
    """Use the accepted outer-loop profile for the arm-free riser platform."""

    values = {
        "along_track_kp": 1.6,
        "cross_track_kp": 1.5,
        "yaw_kp": 1.2,
    }
    values.update(overrides)
    return WholeBodyTrackingConfig(**values)


def bounded_progress_scale(
    base_position_error_m: float,
    tool_position_error_m: float,
    config: WholeBodyTrackingConfig,
) -> float:
    if base_position_error_m < 0.0 or tool_position_error_m < 0.0:
        raise ValueError("tracking errors must be non-negative")
    if not (
        0.0 < config.progress_error_start_m < config.progress_error_full_m
        and 0.0 < config.minimum_progress_scale <= 1.0
    ):
        raise ValueError("invalid progress governor configuration")
    worst_error = max(base_position_error_m, tool_position_error_m)
    severity = np.clip(
        (worst_error - config.progress_error_start_m)
        / (config.progress_error_full_m - config.progress_error_start_m),
        0.0,
        1.0,
    )
    return float(1.0 - severity * (1.0 - config.minimum_progress_scale))


def bounded_base_references(
    desired_base_q: np.ndarray,
    actual_base_q: np.ndarray,
    feedforward_v_mps: float,
    feedforward_wz_radps: float,
    config: WholeBodyTrackingConfig,
) -> tuple[float, float, dict[str, float]]:
    desired_base_q = np.asarray(desired_base_q, dtype=np.float64)
    actual_base_q = np.asarray(actual_base_q, dtype=np.float64)
    if desired_base_q.shape != (3,) or actual_base_q.shape != (3,):
        raise ValueError("base poses must have shape (3,)")
    if not np.isfinite(np.concatenate((desired_base_q, actual_base_q))).all():
        raise ValueError("base poses contain non-finite values")

    delta_world = desired_base_q[:2] - actual_base_q[:2]
    cosine = math.cos(actual_base_q[2])
    sine = math.sin(actual_base_q[2])
    along_error = cosine * delta_world[0] + sine * delta_world[1]
    cross_error = -sine * delta_world[0] + cosine * delta_world[1]
    yaw_error = wrap_to_pi(float(desired_base_q[2] - actual_base_q[2]))
    direction = 1.0 if feedforward_v_mps >= 0.0 else -1.0

    velocity = feedforward_v_mps + config.along_track_kp * along_error
    yaw_rate = (
        feedforward_wz_radps
        + config.yaw_kp * yaw_error
        + config.cross_track_kp * direction * cross_error
    )
    velocity = float(
        np.clip(
            velocity,
            -config.maximum_linear_velocity_mps,
            config.maximum_linear_velocity_mps,
        )
    )
    yaw_rate = float(
        np.clip(
            yaw_rate,
            -config.maximum_yaw_rate_radps,
            config.maximum_yaw_rate_radps,
        )
    )
    return velocity, yaw_rate, {
        "along_track_error_m": float(along_error),
        "cross_track_error_m": float(cross_error),
        "yaw_error_rad": yaw_error,
    }


def numerical_arm_position_jacobian(
    kinematics: UrdfPositionKinematics,
    base_arm_q: np.ndarray,
    epsilon: float = 1e-5,
) -> np.ndarray:
    base_arm_q = np.asarray(base_arm_q, dtype=np.float64)
    if base_arm_q.shape != (6,) or epsilon <= 0.0:
        raise ValueError("invalid state or finite-difference epsilon")
    jacobian = np.empty((3, 3), dtype=np.float64)
    for index in range(3):
        plus = base_arm_q.copy()
        minus = base_arm_q.copy()
        plus[index + 3] += epsilon
        minus[index + 3] -= epsilon
        jacobian[:, index] = (
            kinematics.position(plus) - kinematics.position(minus)
        ) / (2.0 * epsilon)
    return jacobian


def bounded_dls_arm_target(
    kinematics: UrdfPositionKinematics,
    actual_base_q: np.ndarray,
    actual_arm_q: np.ndarray,
    nominal_arm_q: np.ndarray,
    target_position_world_m: np.ndarray,
    actual_position_world_m: np.ndarray,
    config: WholeBodyTrackingConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
    actual_base_q = np.asarray(actual_base_q, dtype=np.float64)
    actual_arm_q = np.asarray(actual_arm_q, dtype=np.float64)
    nominal_arm_q = np.asarray(nominal_arm_q, dtype=np.float64)
    target_position_world_m = np.asarray(target_position_world_m, dtype=np.float64)
    actual_position_world_m = np.asarray(actual_position_world_m, dtype=np.float64)
    arrays = (
        actual_base_q,
        actual_arm_q,
        nominal_arm_q,
        target_position_world_m,
        actual_position_world_m,
    )
    if any(value.shape != (3,) for value in arrays):
        raise ValueError("tracking vectors must all have shape (3,)")
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("tracking vectors contain non-finite values")

    state = np.concatenate((actual_base_q, actual_arm_q))
    jacobian = numerical_arm_position_jacobian(kinematics, state)
    position_error = target_position_world_m - actual_position_world_m
    damped = jacobian @ jacobian.T + config.ik_damping**2 * np.eye(3)
    task_delta = jacobian.T @ np.linalg.solve(damped, position_error)
    raw_target = (
        actual_arm_q
        + config.ik_task_gain * task_delta
        + config.nominal_arm_pull * (nominal_arm_q - actual_arm_q)
    )
    correction = np.clip(
        raw_target - nominal_arm_q,
        -config.maximum_arm_target_correction_rad,
        config.maximum_arm_target_correction_rad,
    )
    target = np.clip(
        nominal_arm_q + correction,
        kinematics.arm_lower,
        kinematics.arm_upper,
    )
    return target, {
        "position_error_m": position_error,
        "task_delta_rad": task_delta,
        "target_correction_rad": target - nominal_arm_q,
        "jacobian_condition": float(np.linalg.cond(jacobian)),
    }


def slew_limited_arm_target(
    requested_target: np.ndarray,
    previous_target: np.ndarray,
    dt: float,
    kinematics: UrdfPositionKinematics,
    config: WholeBodyTrackingConfig,
) -> np.ndarray:
    requested_target = np.asarray(requested_target, dtype=np.float64)
    previous_target = np.asarray(previous_target, dtype=np.float64)
    if requested_target.shape != (3,) or previous_target.shape != (3,):
        raise ValueError("arm targets must have shape (3,)")
    if dt <= 0.0 or not np.isfinite(requested_target).all():
        raise ValueError("invalid arm target or timestep")
    maximum_delta = config.maximum_arm_target_rate_radps * dt
    target = previous_target + np.clip(
        requested_target - previous_target, -maximum_delta, maximum_delta
    )
    return np.clip(target, kinematics.arm_lower, kinematics.arm_upper)
