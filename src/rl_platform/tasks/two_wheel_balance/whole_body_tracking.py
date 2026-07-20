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
    direction_blend_speed_mps: float = 0.05
    direction_recovery_error_start_m: float = 0.20
    direction_recovery_error_full_m: float = 0.40
    ik_damping: float = 0.04
    ik_task_gain: float = 0.7
    nominal_arm_pull: float = 0.15
    maximum_arm_target_correction_rad: float = 0.35
    maximum_arm_target_rate_radps: float = 0.5
    progress_error_start_m: float = 0.05
    progress_error_full_m: float = 0.25
    minimum_progress_scale: float = 0.1
    camera_recovery_error_start_m: float = 0.13
    camera_recovery_error_full_m: float = 0.155
    minimum_camera_recovery_scale: float = 0.2


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


def select_progress_governor_base_error(
    nominal_base_error_m: float,
    commanded_base_error_m: float,
    *,
    use_commanded_base_target: bool,
) -> float:
    """Select the base target that owns phase-governor error feedback."""

    values = (nominal_base_error_m, commanded_base_error_m)
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("progress-governor base errors must be finite and non-negative")
    return float(
        commanded_base_error_m
        if use_commanded_base_target
        else nominal_base_error_m
    )


def summarize_progress_governor_base_error(
    nominal_base_errors_m: np.ndarray,
    commanded_base_errors_m: np.ndarray,
    selected_base_errors_m: np.ndarray,
    *,
    use_commanded_base_target: bool,
    maximum_command_correction_m: float,
    expected_sample_count: int,
) -> dict[str, float | int | bool | str]:
    """Audit policy-rate phase-governor target selection and correction bounds."""

    nominal = np.asarray(nominal_base_errors_m, dtype=np.float64)
    commanded = np.asarray(commanded_base_errors_m, dtype=np.float64)
    selected = np.asarray(selected_base_errors_m, dtype=np.float64)
    arrays = (nominal, commanded, selected)
    if any(value.ndim != 1 or value.size == 0 for value in arrays):
        raise ValueError("progress-governor base-error evidence must be non-empty vectors")
    if not all(np.isfinite(value).all() and np.all(value >= 0.0) for value in arrays):
        raise ValueError("progress-governor base-error evidence must be finite and non-negative")
    if not math.isfinite(maximum_command_correction_m) or maximum_command_correction_m <= 0.0:
        raise ValueError("maximum command correction must be finite and positive")
    if expected_sample_count <= 0:
        raise ValueError("expected sample count must be positive")

    lengths_match = all(value.size == expected_sample_count for value in arrays)
    expected = commanded if use_commanded_base_target else nominal
    selected_matches_source = bool(
        selected.shape == expected.shape and np.array_equal(selected, expected)
    )
    if nominal.shape == commanded.shape:
        command_delta = commanded - nominal
        command_delta_abs_max = float(np.max(np.abs(command_delta)))
        command_delta_mean = float(np.mean(command_delta))
    else:
        command_delta_abs_max = math.inf
        command_delta_mean = math.nan
    command_delta_bounded = (
        command_delta_abs_max <= maximum_command_correction_m + 1e-9
    )
    telemetry_observed = bool(
        lengths_match and selected_matches_source and command_delta_bounded
    )
    return {
        "progress_base_error_source": (
            "lever_compensated_commanded_base_target"
            if use_commanded_base_target
            else "nominal_plan_base_target"
        ),
        "progress_base_error_telemetry_sample_count": int(selected.size),
        "progress_base_error_selected_source_matches": selected_matches_source,
        "progress_base_error_command_delta_bounded": command_delta_bounded,
        "progress_base_error_telemetry_observed": telemetry_observed,
        "nominal_base_progress_error_p95_m": float(np.percentile(nominal, 95)),
        "nominal_base_progress_error_max_m": float(np.max(nominal)),
        "commanded_base_progress_error_p95_m": float(
            np.percentile(commanded, 95)
        ),
        "commanded_base_progress_error_max_m": float(np.max(commanded)),
        "selected_base_progress_error_p95_m": float(np.percentile(selected, 95)),
        "selected_base_progress_error_max_m": float(np.max(selected)),
        "selected_vs_nominal_base_progress_error_mean_delta_m": (
            command_delta_mean if use_commanded_base_target else 0.0
        ),
        "selected_vs_nominal_base_progress_error_abs_max_delta_m": (
            command_delta_abs_max if use_commanded_base_target else 0.0
        ),
        "maximum_commanded_base_progress_error_delta_m": (
            maximum_command_correction_m
        ),
    }


def summarize_progress_hold(
    progress_scales: np.ndarray,
    *,
    hold_threshold: float = 1e-12,
) -> dict[str, float | int]:
    """Summarize exact phase holds without expanding policy-rate traces."""

    values = np.asarray(progress_scales, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("progress scales must be a non-empty vector")
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("progress scales must be finite and in [0, 1]")
    if not math.isfinite(hold_threshold) or not 0.0 <= hold_threshold < 1.0:
        raise ValueError("hold threshold must be finite and in [0, 1)")

    held = values <= hold_threshold
    starts = held & ~np.concatenate((np.asarray([False]), held[:-1]))
    return {
        "progress_hold_step_count": int(np.count_nonzero(held)),
        "progress_hold_ratio": float(np.mean(held)),
        "progress_hold_segment_count": int(np.count_nonzero(starts)),
    }


def bounded_camera_recovery_progress_scale(
    tool_position_error_m: float,
    correction_saturated: bool,
    config: WholeBodyTrackingConfig,
) -> float:
    """Slow phase only while a saturated camera correction is near its gate."""

    if not math.isfinite(tool_position_error_m) or tool_position_error_m < 0.0:
        raise ValueError("camera tracking error must be finite and non-negative")
    if not (
        0.0
        < config.camera_recovery_error_start_m
        < config.camera_recovery_error_full_m
        and 0.0 < config.minimum_camera_recovery_scale <= 1.0
    ):
        raise ValueError("invalid camera recovery governor configuration")
    if not correction_saturated:
        return 1.0
    severity = np.clip(
        (tool_position_error_m - config.camera_recovery_error_start_m)
        / (
            config.camera_recovery_error_full_m
            - config.camera_recovery_error_start_m
        ),
        0.0,
        1.0,
    )
    return float(
        1.0
        - severity * (1.0 - config.minimum_camera_recovery_scale)
    )


def bounded_camera_lever_arm_base_target(
    desired_base_q: np.ndarray,
    actual_base_q: np.ndarray,
    target_camera_position_world_m: np.ndarray,
    actual_camera_position_world_m: np.ndarray,
    *,
    gain: float,
    maximum_correction_m: float,
) -> tuple[np.ndarray, dict[str, np.ndarray | float | bool]]:
    """Offset the base target to cancel measured camera lever-arm displacement."""

    desired_base_q = np.asarray(desired_base_q, dtype=np.float64)
    actual_base_q = np.asarray(actual_base_q, dtype=np.float64)
    target_camera_position_world_m = np.asarray(
        target_camera_position_world_m, dtype=np.float64
    )
    actual_camera_position_world_m = np.asarray(
        actual_camera_position_world_m, dtype=np.float64
    )
    arrays = (
        desired_base_q,
        actual_base_q,
        target_camera_position_world_m,
        actual_camera_position_world_m,
    )
    if any(value.shape != (3,) for value in arrays):
        raise ValueError("base poses and camera positions must have shape (3,)")
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("base poses and camera positions must be finite")
    if not math.isfinite(gain) or not 0.0 <= gain <= 1.0:
        raise ValueError("camera lever-arm compensation gain must be in [0, 1]")
    if not math.isfinite(maximum_correction_m) or maximum_correction_m <= 0.0:
        raise ValueError("maximum camera lever-arm correction must be positive")

    target_lever_xy = (
        target_camera_position_world_m[:2] - desired_base_q[:2]
    )
    actual_lever_xy = (
        actual_camera_position_world_m[:2] - actual_base_q[:2]
    )
    lever_error_xy = actual_lever_xy - target_lever_xy
    raw_correction_xy = -gain * lever_error_xy
    raw_norm = float(np.linalg.norm(raw_correction_xy))
    correction_scale = (
        min(1.0, maximum_correction_m / raw_norm) if raw_norm > 0.0 else 1.0
    )
    correction_xy = raw_correction_xy * correction_scale
    compensated_base_q = desired_base_q.copy()
    compensated_base_q[:2] += correction_xy
    return compensated_base_q, {
        "target_lever_xy_m": target_lever_xy,
        "actual_lever_xy_m": actual_lever_xy,
        "lever_error_xy_m": lever_error_xy,
        "lever_error_norm_m": float(np.linalg.norm(lever_error_xy)),
        "raw_correction_xy_m": raw_correction_xy,
        "raw_correction_norm_m": raw_norm,
        "correction_xy_m": correction_xy,
        "correction_norm_m": float(np.linalg.norm(correction_xy)),
        "correction_scale": correction_scale,
        "saturated": raw_norm > maximum_correction_m + 1e-12,
    }


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
    if (
        not math.isfinite(config.direction_blend_speed_mps)
        or config.direction_blend_speed_mps <= 0.0
    ):
        raise ValueError("direction blend speed must be finite and positive")
    if not (
        math.isfinite(config.maximum_linear_velocity_mps)
        and config.maximum_linear_velocity_mps > 0.0
        and math.isfinite(config.maximum_yaw_rate_radps)
        and config.maximum_yaw_rate_radps > 0.0
    ):
        raise ValueError("base velocity limits must be finite and positive")
    if not (
        math.isfinite(config.direction_recovery_error_start_m)
        and math.isfinite(config.direction_recovery_error_full_m)
        and 0.0
        <= config.direction_recovery_error_start_m
        < config.direction_recovery_error_full_m
    ):
        raise ValueError("invalid direction recovery error bounds")

    delta_world = desired_base_q[:2] - actual_base_q[:2]
    cosine = math.cos(actual_base_q[2])
    sine = math.sin(actual_base_q[2])
    along_error = cosine * delta_world[0] + sine * delta_world[1]
    cross_error = -sine * delta_world[0] + cosine * delta_world[1]
    yaw_error = wrap_to_pi(float(desired_base_q[2] - actual_base_q[2]))
    feedforward_direction = 1.0 if feedforward_v_mps >= 0.0 else -1.0
    raw_velocity = feedforward_v_mps + config.along_track_kp * along_error
    velocity = float(
        np.clip(
            raw_velocity,
            -config.maximum_linear_velocity_mps,
            config.maximum_linear_velocity_mps,
        )
    )
    feedback_motion_direction = float(
        np.clip(velocity / config.direction_blend_speed_mps, -1.0, 1.0)
    )
    base_position_error = math.hypot(along_error, cross_error)
    direction_recovery_blend = float(
        np.clip(
            (
                base_position_error - config.direction_recovery_error_start_m
            )
            / (
                config.direction_recovery_error_full_m
                - config.direction_recovery_error_start_m
            ),
            0.0,
            1.0,
        )
    )
    motion_direction = (
        feedforward_direction
        + direction_recovery_blend
        * (feedback_motion_direction - feedforward_direction)
    )
    yaw_rate = (
        feedforward_wz_radps
        + config.yaw_kp * yaw_error
        + config.cross_track_kp * motion_direction * cross_error
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
        "raw_velocity_reference_mps": float(raw_velocity),
        "base_position_error_m": base_position_error,
        "feedforward_direction": feedforward_direction,
        "feedback_motion_direction": feedback_motion_direction,
        "direction_recovery_blend": direction_recovery_blend,
        "motion_direction": motion_direction,
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
