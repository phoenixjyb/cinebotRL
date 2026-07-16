"""Bounded task-space feedback for scripted two-wheel whole-body playback."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .whole_body_kinematics import UrdfPositionKinematics


def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quaternion_wxyz(quaternion: np.ndarray) -> float:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError(f"expected finite quaternion shape (4,), got {quaternion.shape}")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("quaternion norm is zero")
    w, x, y, z = quaternion / norm
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def roll_pitch_from_quaternion_wxyz(quaternion: np.ndarray) -> np.ndarray:
    rotation = rotation_matrix_from_quaternion_wxyz(quaternion)
    return np.array(
        [
            math.atan2(float(rotation[2, 1]), float(rotation[2, 2])),
            math.atan2(
                float(-rotation[2, 0]),
                math.hypot(float(rotation[2, 1]), float(rotation[2, 2])),
            ),
        ],
        dtype=np.float64,
    )


def quaternion_from_roll_pitch_yaw_wxyz(
    roll: float, pitch: float, yaw: float
) -> np.ndarray:
    if not all(math.isfinite(value) for value in (roll, pitch, yaw)):
        raise ValueError("roll, pitch, and yaw must be finite")
    half_roll = 0.5 * roll
    half_pitch = 0.5 * pitch
    half_yaw = 0.5 * yaw
    cr, sr = math.cos(half_roll), math.sin(half_roll)
    cp, sp = math.cos(half_pitch), math.sin(half_pitch)
    cy, sy = math.cos(half_yaw), math.sin(half_yaw)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )


def quaternion_from_pitch_yaw_wxyz(pitch: float, yaw: float) -> np.ndarray:
    return quaternion_from_roll_pitch_yaw_wxyz(0.0, pitch, yaw)


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


@dataclass(frozen=True)
class GimbalRootAttitudeFeedbackConfig:
    gain: float = 0.5
    time_constant_s: float = 0.15
    maximum_error_rad: float = math.radians(12.0)


def filtered_gimbal_root_attitude_command(
    actual_root_quaternion_wxyz: np.ndarray,
    nominal_pitch_rad: float,
    nominal_yaw_rad: float,
    previous_filtered_error_rpy_rad: np.ndarray,
    dt: float,
    config: GimbalRootAttitudeFeedbackConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Add smooth measured chassis-attitude feedback to gimbal feed-forward."""

    previous = np.asarray(previous_filtered_error_rpy_rad, dtype=np.float64)
    if previous.shape != (3,) or not np.isfinite(previous).all():
        raise ValueError("previous root-attitude error must be finite shape (3,)")
    if dt <= 0.0 or not math.isfinite(dt):
        raise ValueError("dt must be finite and positive")
    if not (
        0.0 <= config.gain <= 1.0
        and config.time_constant_s >= 0.0
        and config.maximum_error_rad > 0.0
    ):
        raise ValueError("invalid gimbal root-attitude feedback configuration")

    actual_roll, actual_pitch = roll_pitch_from_quaternion_wxyz(
        actual_root_quaternion_wxyz
    )
    actual_yaw = yaw_from_quaternion_wxyz(actual_root_quaternion_wxyz)
    bounded_error = np.clip(
        np.array(
            [
                actual_roll,
                actual_pitch - nominal_pitch_rad,
                wrap_to_pi(actual_yaw - nominal_yaw_rad),
            ],
            dtype=np.float64,
        ),
        -config.maximum_error_rad,
        config.maximum_error_rad,
    )
    alpha = (
        1.0
        if config.time_constant_s == 0.0
        else 1.0 - math.exp(-dt / config.time_constant_s)
    )
    filtered_error = previous + alpha * (bounded_error - previous)
    applied_correction = config.gain * filtered_error
    command_quaternion = quaternion_from_roll_pitch_yaw_wxyz(
        float(applied_correction[0]),
        nominal_pitch_rad + float(applied_correction[1]),
        nominal_yaw_rad + float(applied_correction[2]),
    )
    return command_quaternion, filtered_error, applied_correction


def bounded_progress_scale(
    base_position_error_m: float,
    tool_position_error_m: float,
    config: WholeBodyTrackingConfig,
) -> float:
    if base_position_error_m < 0.0 or tool_position_error_m < 0.0:
        raise ValueError("tracking errors must be non-negative")
    if not (
        0.0 < config.progress_error_start_m < config.progress_error_full_m
        and 0.0 <= config.minimum_progress_scale <= 1.0
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


def bounded_balance_progress_scale(
    pitch_rad: float,
    slowdown_start_rad: float,
    full_stop_rad: float,
    minimum_progress_scale: float = 0.0,
) -> float:
    """Reduce reference progress before chassis pitch reaches its hard gate."""

    if not all(
        math.isfinite(value)
        for value in (
            pitch_rad,
            slowdown_start_rad,
            full_stop_rad,
            minimum_progress_scale,
        )
    ):
        raise ValueError("balance progress governor inputs must be finite")
    if not (
        0.0 <= slowdown_start_rad < full_stop_rad
        and 0.0 <= minimum_progress_scale <= 1.0
    ):
        raise ValueError("invalid balance progress governor configuration")
    severity = np.clip(
        (abs(pitch_rad) - slowdown_start_rad)
        / (full_stop_rad - slowdown_start_rad),
        0.0,
        1.0,
    )
    return float(1.0 - severity * (1.0 - minimum_progress_scale))


def bounded_attitude_progress_scale(
    attitude_error_rad: float,
    slowdown_start_rad: float,
    full_stop_rad: float,
    minimum_progress_scale: float = 0.0,
) -> float:
    """Pause reference progress when camera attitude feedback cannot catch up."""

    if attitude_error_rad < 0.0:
        raise ValueError("attitude error must be non-negative")
    return bounded_balance_progress_scale(
        attitude_error_rad,
        slowdown_start_rad,
        full_stop_rad,
        minimum_progress_scale,
    )


def phase_scaled_feedforward(
    feedforward_v_mps: float,
    feedforward_wz_radps: float,
    progress_scale: float,
) -> tuple[float, float]:
    """Convert phase derivatives to wall-time feedforward commands."""

    if not all(
        math.isfinite(value)
        for value in (feedforward_v_mps, feedforward_wz_radps, progress_scale)
    ) or not 0.0 <= progress_scale <= 1.0:
        raise ValueError("progress scale must be finite and within [0, 1]")
    return (
        progress_scale * feedforward_v_mps,
        progress_scale * feedforward_wz_radps,
    )


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


def bounded_semantic_arm_target(
    kinematics: UrdfPositionKinematics,
    actual_base_q: np.ndarray,
    actual_arm_q: np.ndarray,
    nominal_arm_q: np.ndarray,
    previous_arm_target_q: np.ndarray,
    target_position_world_m: np.ndarray,
    actual_position_world_m: np.ndarray,
    dt: float,
    semantic_feedback_enabled: bool,
    config: WholeBodyTrackingConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply bounded DLS only after semantic playback has started."""

    requested = np.asarray(nominal_arm_q, dtype=np.float64)
    if semantic_feedback_enabled:
        requested, _ = bounded_dls_arm_target(
            kinematics,
            actual_base_q,
            actual_arm_q,
            nominal_arm_q,
            target_position_world_m,
            actual_position_world_m,
            config,
        )
    target = slew_limited_arm_target(
        requested,
        previous_arm_target_q,
        dt,
        kinematics,
        config,
    )
    return target, target - np.asarray(nominal_arm_q, dtype=np.float64)


def bounded_task_space_base_target(
    kinematics: UrdfPositionKinematics,
    desired_base_q: np.ndarray,
    desired_arm_q: np.ndarray,
    target_position_world_m: np.ndarray,
    root_tilt_displacement_world_m: np.ndarray,
    maximum_offset_m: float,
) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
    """Allocate horizontal physical-tool displacement to the wheel base.

    The retargeter uses planar FK, while the balancing root pitches in Isaac.
    The internal base target absorbs the resulting horizontal tool displacement
    plus any retained planar retarget residual.  The semantic tool target and
    teacher arm joints remain unchanged.
    """

    desired_base_q = np.asarray(desired_base_q, dtype=np.float64)
    desired_arm_q = np.asarray(desired_arm_q, dtype=np.float64)
    target_position_world_m = np.asarray(target_position_world_m, dtype=np.float64)
    root_tilt_displacement_world_m = np.asarray(
        root_tilt_displacement_world_m, dtype=np.float64
    )
    arrays = (
        desired_base_q,
        desired_arm_q,
        target_position_world_m,
        root_tilt_displacement_world_m,
    )
    if any(value.shape != (3,) for value in arrays):
        raise ValueError("base-compensation vectors must all have shape (3,)")
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("base-compensation vectors contain non-finite values")
    if maximum_offset_m <= 0.0:
        raise ValueError("maximum base offset must be positive")

    planar_nominal = kinematics.position(
        np.concatenate((desired_base_q, desired_arm_q))
    )
    retarget_residual = target_position_world_m[:2] - planar_nominal[:2]
    requested_offset = retarget_residual - root_tilt_displacement_world_m[:2]
    requested_norm = float(np.linalg.norm(requested_offset))
    scale = min(1.0, maximum_offset_m / max(requested_norm, 1e-12))
    bounded_offset = requested_offset * scale
    target = desired_base_q.copy()
    target[:2] += bounded_offset
    return target, {
        "requested_offset_world_m": requested_offset,
        "bounded_offset_world_m": bounded_offset,
        "retarget_residual_world_m": retarget_residual,
        "offset_saturated": float(requested_norm > maximum_offset_m),
    }


def slew_limited_planar_offset(
    requested_offset_world_m: np.ndarray,
    previous_offset_world_m: np.ndarray,
    dt: float,
    maximum_rate_mps: float,
) -> np.ndarray:
    requested_offset_world_m = np.asarray(
        requested_offset_world_m, dtype=np.float64
    )
    previous_offset_world_m = np.asarray(previous_offset_world_m, dtype=np.float64)
    if requested_offset_world_m.shape != (2,) or previous_offset_world_m.shape != (2,):
        raise ValueError("planar offsets must have shape (2,)")
    if not np.isfinite(
        np.concatenate((requested_offset_world_m, previous_offset_world_m))
    ).all():
        raise ValueError("planar offsets contain non-finite values")
    if dt <= 0.0 or maximum_rate_mps <= 0.0:
        raise ValueError("offset timestep and rate must be positive")

    delta = requested_offset_world_m - previous_offset_world_m
    delta_norm = float(np.linalg.norm(delta))
    maximum_delta = maximum_rate_mps * dt
    if delta_norm > maximum_delta:
        delta *= maximum_delta / delta_norm
    return previous_offset_world_m + delta


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
