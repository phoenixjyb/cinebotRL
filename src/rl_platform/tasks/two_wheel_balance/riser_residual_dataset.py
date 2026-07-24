"""Versioned executed-state dataset contract for the riser residual student."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .camera_attitude import quaternion_matrix_wxyz, rotation_error_vector


DATASET_SCHEMA = "cinebotrl_two_wheel_riser_executed_residual_v2"
RAW_TEACHER_SCHEMA = "cinebotrl_two_wheel_riser_executed_raw_teacher_v1"
POLICY_TRACE_SCHEMA = "cinebotrl_two_wheel_riser_policy_trace_v1"
SHADOW_TEACHER_TRACE_SCHEMA = (
    "cinebotrl_two_wheel_riser_shadow_teacher_trace_v1"
)
LOOKAHEAD_HORIZONS_S = (0.25, 0.50, 1.00)
BASE_OBSERVATION_NAMES = (
    "pitch_rad",
    "pitch_rate_rad_s",
    "mean_wheel_position_rad",
    "mean_wheel_velocity_rad_s",
    "wheel_velocity_difference_rad_s",
    "yaw_rate_rad_s",
    "base_target_longitudinal_error_m",
    "base_target_lateral_error_m",
    "base_target_yaw_error_rad",
    "camera_target_longitudinal_error_m",
    "camera_target_lateral_error_m",
    "camera_target_vertical_error_m",
    "camera_attitude_error_x_rad",
    "camera_attitude_error_y_rad",
    "camera_attitude_error_z_rad",
    "riser_position_m",
    "riser_velocity_m_s",
    "riser_target_error_m",
    "feedforward_vx_m_s",
    "feedforward_wz_rad_s",
    "feedforward_riser_velocity_m_s",
    "phase_fraction",
    "progress_scale",
    "previous_residual_vx_normalized",
    "previous_residual_wz_normalized",
    "previous_residual_riser_target_normalized",
)
LOOKAHEAD_CHANNEL_NAMES = (
    "base_target_longitudinal_error_m",
    "base_target_lateral_error_m",
    "base_target_yaw_error_rad",
    "camera_target_longitudinal_error_m",
    "camera_target_lateral_error_m",
    "camera_target_vertical_error_m",
    "camera_attitude_error_x_rad",
    "camera_attitude_error_y_rad",
    "camera_attitude_error_z_rad",
    "riser_target_error_m",
    "feedforward_vx_m_s",
    "feedforward_wz_rad_s",
    "feedforward_riser_velocity_m_s",
)


def _lookahead_prefix(horizon_s: float) -> str:
    return f"lookahead_{horizon_s:.2f}s".replace(".", "p")


OBSERVATION_NAMES = BASE_OBSERVATION_NAMES + tuple(
    f"{_lookahead_prefix(horizon)}_{channel}"
    for horizon in LOOKAHEAD_HORIZONS_S
    for channel in LOOKAHEAD_CHANNEL_NAMES
)
OBSERVATION_INDEX = {
    name: index for index, name in enumerate(OBSERVATION_NAMES)
}
ACTION_NAMES = (
    "residual_vx_normalized",
    "residual_wz_normalized",
    "residual_riser_target_normalized",
)
ACTION_SCALES = np.array([0.30, 0.40, 0.10], dtype=np.float64)
MODEL_BASED_POLICY_RESIDUAL_SCALES = np.array(
    [0.05, 0.05, 0.02], dtype=np.float64
)
MODEL_BASED_POLICY_RESIDUAL_CONTRACT = (
    "model_based_planner_plus_bounded_policy_residual_v1"
)
MODEL_BASED_POLICY_CONTROL_OWNERSHIP_CONTRACT = (
    "frozen_lqr_high_level_residual_control_ownership_v1"
)
MODEL_BASED_POLICY_PREVIOUS_ACTION_CONTRACT = (
    "previous_effective_post_supervisor_action_v1"
)
SPLIT_NAMES = ("train", "validation", "holdout")
PREVIOUS_ACTION_NAMES = (
    "previous_residual_vx_normalized",
    "previous_residual_wz_normalized",
    "previous_residual_riser_target_normalized",
)
PREVIOUS_ACTION_INDICES = tuple(
    OBSERVATION_INDEX[name] for name in PREVIOUS_ACTION_NAMES
)


def _wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _world_xy_to_base(vector_xy: np.ndarray, yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array(
        [c * vector_xy[0] + s * vector_xy[1], -s * vector_xy[0] + c * vector_xy[1]],
        dtype=np.float64,
    )


def build_executed_observation(
    *,
    lqr_state: np.ndarray,
    actual_base_xy_yaw: np.ndarray,
    target_base_xy_yaw: np.ndarray,
    actual_camera_position_world_m: np.ndarray,
    target_camera_position_world_m: np.ndarray,
    actual_camera_quat_wxyz: np.ndarray,
    target_camera_quat_wxyz: np.ndarray,
    riser_position_m: float,
    riser_velocity_m_s: float,
    riser_target_m: float,
    feedforward_vx_m_s: float,
    feedforward_wz_rad_s: float,
    feedforward_riser_velocity_m_s: float,
    phase_fraction: float,
    progress_scale: float,
    previous_residual_action: np.ndarray,
    lookahead_base_xy_yaw: np.ndarray,
    lookahead_camera_position_world_m: np.ndarray,
    lookahead_camera_quat_wxyz: np.ndarray,
    lookahead_riser_target_m: np.ndarray,
    lookahead_feedforward_v_wz_riser: np.ndarray,
) -> np.ndarray:
    """Build one deployable pre-action observation from physical state."""

    lqr_state = np.asarray(lqr_state, dtype=np.float64)
    actual_base = np.asarray(actual_base_xy_yaw, dtype=np.float64)
    target_base = np.asarray(target_base_xy_yaw, dtype=np.float64)
    actual_camera = np.asarray(actual_camera_position_world_m, dtype=np.float64)
    target_camera = np.asarray(target_camera_position_world_m, dtype=np.float64)
    previous = np.asarray(previous_residual_action, dtype=np.float64)
    future_base = np.asarray(lookahead_base_xy_yaw, dtype=np.float64)
    future_camera = np.asarray(
        lookahead_camera_position_world_m, dtype=np.float64
    )
    future_camera_quat = np.asarray(
        lookahead_camera_quat_wxyz, dtype=np.float64
    )
    future_riser = np.asarray(lookahead_riser_target_m, dtype=np.float64)
    future_feedforward = np.asarray(
        lookahead_feedforward_v_wz_riser, dtype=np.float64
    )
    if lqr_state.shape != (6,) or actual_base.shape != (3,) or target_base.shape != (3,):
        raise ValueError("invalid LQR/base observation shape")
    if actual_camera.shape != (3,) or target_camera.shape != (3,) or previous.shape != (3,):
        raise ValueError("invalid camera/action-history observation shape")
    count = len(LOOKAHEAD_HORIZONS_S)
    if (
        future_base.shape != (count, 3)
        or future_camera.shape != (count, 3)
        or future_camera_quat.shape != (count, 4)
        or future_riser.shape != (count,)
        or future_feedforward.shape != (count, 3)
    ):
        raise ValueError("invalid lookahead observation shape")
    base_error = _world_xy_to_base(target_base[:2] - actual_base[:2], actual_base[2])
    camera_delta = target_camera - actual_camera
    camera_xy_error = _world_xy_to_base(camera_delta[:2], actual_base[2])
    attitude_error = rotation_error_vector(
        quaternion_matrix_wxyz(np.asarray(actual_camera_quat_wxyz, dtype=np.float64)),
        quaternion_matrix_wxyz(np.asarray(target_camera_quat_wxyz, dtype=np.float64)),
    )
    lookahead_features = []
    actual_camera_rotation = quaternion_matrix_wxyz(
        np.asarray(actual_camera_quat_wxyz, dtype=np.float64)
    )
    for index in range(count):
        future_base_error = _world_xy_to_base(
            future_base[index, :2] - actual_base[:2], actual_base[2]
        )
        future_camera_delta = future_camera[index] - actual_camera
        future_camera_xy_error = _world_xy_to_base(
            future_camera_delta[:2], actual_base[2]
        )
        future_attitude_error = rotation_error_vector(
            actual_camera_rotation,
            quaternion_matrix_wxyz(future_camera_quat[index]),
        )
        lookahead_features.extend(
            (
                *future_base_error,
                _wrap_angle(future_base[index, 2] - actual_base[2]),
                *future_camera_xy_error,
                future_camera_delta[2],
                *future_attitude_error,
                future_riser[index] - riser_position_m,
                *future_feedforward[index],
            )
        )
    observation = np.concatenate(
        (
            lqr_state,
            base_error,
            [_wrap_angle(target_base[2] - actual_base[2])],
            camera_xy_error,
            [camera_delta[2]],
            attitude_error,
            [riser_position_m, riser_velocity_m_s, riser_target_m - riser_position_m],
            [feedforward_vx_m_s, feedforward_wz_rad_s, feedforward_riser_velocity_m_s],
            [phase_fraction, progress_scale],
            previous,
            lookahead_features,
        )
    )
    if observation.shape != (len(OBSERVATION_NAMES),) or not np.isfinite(observation).all():
        raise ValueError("invalid executed observation")
    if not 0.0 <= phase_fraction <= 1.0 or not 0.0 <= progress_scale <= 1.0:
        raise ValueError("phase/progress values are outside [0, 1]")
    return observation.astype(np.float32)


def build_raw_residual_command(
    *,
    feedforward_vx_m_s: float,
    feedforward_wz_rad_s: float,
    commanded_vx_m_s: float,
    commanded_wz_rad_s: float,
    actual_riser_position_m: float,
    target_riser_position_m: float,
) -> np.ndarray:
    residual = np.array(
        [
            commanded_vx_m_s - feedforward_vx_m_s,
            commanded_wz_rad_s - feedforward_wz_rad_s,
            target_riser_position_m - actual_riser_position_m,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(residual).all():
        raise ValueError("residual action is non-finite")
    return residual


def normalize_residual_command(residual: np.ndarray) -> np.ndarray:
    residual = np.asarray(residual, dtype=np.float64)
    if residual.shape != (3,) or not np.isfinite(residual).all():
        raise ValueError("invalid raw residual command")
    return residual / ACTION_SCALES


def residual_action_envelope_passed(normalized: np.ndarray) -> bool:
    normalized = np.asarray(normalized, dtype=np.float64)
    return bool(
        normalized.shape == (3,)
        and np.isfinite(normalized).all()
        and np.max(np.abs(normalized)) < 1.0 - 1e-6
    )


def build_residual_action(
    *,
    feedforward_vx_m_s: float,
    feedforward_wz_rad_s: float,
    commanded_vx_m_s: float,
    commanded_wz_rad_s: float,
    actual_riser_position_m: float,
    target_riser_position_m: float,
) -> np.ndarray:
    residual = build_raw_residual_command(
        feedforward_vx_m_s=feedforward_vx_m_s,
        feedforward_wz_rad_s=feedforward_wz_rad_s,
        commanded_vx_m_s=commanded_vx_m_s,
        commanded_wz_rad_s=commanded_wz_rad_s,
        actual_riser_position_m=actual_riser_position_m,
        target_riser_position_m=target_riser_position_m,
    )
    normalized = normalize_residual_command(residual)
    if not residual_action_envelope_passed(normalized):
        raise ValueError(f"residual action scale is too small: {normalized}")
    return normalized.astype(np.float32)


def apply_residual_action(
    feedforward_vx_m_s: float,
    feedforward_wz_rad_s: float,
    actual_riser_position_m: float,
    action: np.ndarray,
    *,
    action_scales: np.ndarray = ACTION_SCALES,
    maximum_linear_velocity_m_s: float = 0.4,
    maximum_yaw_rate_rad_s: float = 0.4,
    riser_bounds_m: tuple[float, float] = (0.0, 1.2),
) -> np.ndarray:
    action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    scales = np.asarray(action_scales, dtype=np.float64)
    if action.shape != (3,) or not np.isfinite(action).all():
        raise ValueError("invalid residual action")
    if scales.shape != (3,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("invalid residual action scales")
    if maximum_linear_velocity_m_s <= 0.0 or maximum_yaw_rate_rad_s <= 0.0:
        raise ValueError("base command limits must be positive")
    command = np.array(
        [
            feedforward_vx_m_s + scales[0] * action[0],
            feedforward_wz_rad_s + scales[1] * action[1],
            actual_riser_position_m + scales[2] * action[2],
        ],
        dtype=np.float64,
    )
    command[0] = np.clip(
        command[0], -maximum_linear_velocity_m_s, maximum_linear_velocity_m_s
    )
    command[1] = np.clip(
        command[1], -maximum_yaw_rate_rad_s, maximum_yaw_rate_rad_s
    )
    command[2] = np.clip(command[2], *riser_bounds_m)
    return command


def apply_model_based_policy_residual(
    model_vx_m_s: float,
    model_wz_rad_s: float,
    model_riser_target_m: float,
    action: np.ndarray,
    *,
    action_scales: np.ndarray = MODEL_BASED_POLICY_RESIDUAL_SCALES,
    maximum_linear_velocity_m_s: float = 0.4,
    maximum_yaw_rate_rad_s: float = 0.4,
    riser_bounds_m: tuple[float, float] = (0.0, 1.2),
) -> np.ndarray:
    """Add a bounded learned correction above a complete model-based command."""

    base_command = np.asarray(
        [model_vx_m_s, model_wz_rad_s, model_riser_target_m], dtype=np.float64
    )
    normalized_action = np.asarray(action, dtype=np.float64)
    scales = np.asarray(action_scales, dtype=np.float64)
    if base_command.shape != (3,) or not np.isfinite(base_command).all():
        raise ValueError("invalid model-based command")
    if normalized_action.shape != (3,) or not np.isfinite(normalized_action).all():
        raise ValueError("invalid model-based residual action")
    if np.max(np.abs(normalized_action)) > 1.0 + 1e-6:
        raise ValueError("model-based residual action exceeds normalized bounds")
    if scales.shape != (3,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("invalid model-based residual action scales")
    if maximum_linear_velocity_m_s <= 0.0 or maximum_yaw_rate_rad_s <= 0.0:
        raise ValueError("base command limits must be positive")
    lower_riser, upper_riser = riser_bounds_m
    if not np.isfinite([lower_riser, upper_riser]).all() or lower_riser >= upper_riser:
        raise ValueError("invalid riser bounds")

    command = base_command + scales * np.clip(normalized_action, -1.0, 1.0)
    command[0] = np.clip(
        command[0], -maximum_linear_velocity_m_s, maximum_linear_velocity_m_s
    )
    command[1] = np.clip(
        command[1], -maximum_yaw_rate_rad_s, maximum_yaw_rate_rad_s
    )
    command[2] = np.clip(command[2], lower_riser, upper_riser)
    return command


def apply_model_based_policy_residual_with_effective_action(
    model_vx_m_s: float,
    model_wz_rad_s: float,
    model_riser_target_m: float,
    action: np.ndarray,
    *,
    action_scales: np.ndarray = MODEL_BASED_POLICY_RESIDUAL_SCALES,
    maximum_linear_velocity_m_s: float = 0.4,
    maximum_yaw_rate_rad_s: float = 0.4,
    riser_bounds_m: tuple[float, float] = (0.0, 1.2),
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a policy residual and recover the post-supervisor normalized action."""

    command = apply_model_based_policy_residual(
        model_vx_m_s,
        model_wz_rad_s,
        model_riser_target_m,
        action,
        action_scales=action_scales,
        maximum_linear_velocity_m_s=maximum_linear_velocity_m_s,
        maximum_yaw_rate_rad_s=maximum_yaw_rate_rad_s,
        riser_bounds_m=riser_bounds_m,
    )
    base_command = np.asarray(
        [model_vx_m_s, model_wz_rad_s, model_riser_target_m], dtype=np.float64
    )
    scales = np.asarray(action_scales, dtype=np.float64)
    effective_action = (command - base_command) / scales
    if not np.isfinite(effective_action).all():
        raise ValueError("effective model-based residual action is non-finite")
    if np.max(np.abs(effective_action)) > 1.0 + 1e-6:
        raise ValueError(
            "effective model-based residual action exceeds normalized bounds"
        )
    return command, np.clip(effective_action, -1.0, 1.0)


def validate_case_dataset(payload: dict[str, np.ndarray], *, expected_case: int | None = None) -> None:
    required = {
        "observations": 2,
        "actions": 2,
        "case_ids": 1,
        "elapsed_time_s": 1,
        "phase_time_s": 1,
        "baseline_wheel_actions": 2,
        "teacher_commands": 2,
    }
    for name, ndim in required.items():
        if name not in payload or np.asarray(payload[name]).ndim != ndim:
            raise ValueError(f"missing or invalid {name}")
    count = len(payload["observations"])
    if count < 2 or any(len(payload[name]) != count for name in required):
        raise ValueError("dataset row counts do not match")
    if payload["observations"].shape[1] != len(OBSERVATION_NAMES):
        raise ValueError("observation dimension mismatch")
    if payload["actions"].shape[1] != len(ACTION_NAMES):
        raise ValueError("action dimension mismatch")
    if payload["baseline_wheel_actions"].shape[1] != 2 or payload["teacher_commands"].shape[1] != 3:
        raise ValueError("auxiliary command dimension mismatch")
    if not all(np.isfinite(np.asarray(payload[name])).all() for name in required):
        raise ValueError("dataset contains non-finite values")
    if np.max(np.abs(payload["actions"])) > 1.0 + 1e-6:
        raise ValueError("residual action exceeds normalized bounds")
    cases = np.unique(payload["case_ids"])
    if len(cases) != 1 or (expected_case is not None and int(cases[0]) != expected_case):
        raise ValueError("case dataset mixes trajectories")
    if abs(float(payload["elapsed_time_s"][0])) > 1e-9 or not np.all(np.diff(payload["elapsed_time_s"]) > 0):
        raise ValueError("executed timestamps must start at zero and increase")
    if not np.all(np.diff(payload["phase_time_s"]) >= 0):
        raise ValueError("phase time must be monotonic")


def validate_raw_teacher_case(
    payload: dict[str, np.ndarray], *, expected_case: int | None = None
) -> None:
    required = {
        "observations": 2,
        "raw_residual_commands": 2,
        "case_ids": 1,
        "elapsed_time_s": 1,
        "phase_time_s": 1,
        "baseline_wheel_actions": 2,
        "teacher_commands": 2,
    }
    for name, ndim in required.items():
        if name not in payload or np.asarray(payload[name]).ndim != ndim:
            raise ValueError(f"missing or invalid {name}")
    count = len(payload["observations"])
    if count < 2 or any(len(payload[name]) != count for name in required):
        raise ValueError("raw teacher row counts do not match")
    if payload["observations"].shape[1] != len(OBSERVATION_NAMES):
        raise ValueError("raw teacher observation dimension mismatch")
    if payload["raw_residual_commands"].shape[1] != len(ACTION_NAMES):
        raise ValueError("raw residual command dimension mismatch")
    if (
        payload["baseline_wheel_actions"].shape[1] != 2
        or payload["teacher_commands"].shape[1] != 3
    ):
        raise ValueError("raw teacher auxiliary command dimension mismatch")
    if not all(np.isfinite(np.asarray(payload[name])).all() for name in required):
        raise ValueError("raw teacher capture contains non-finite values")
    cases = np.unique(payload["case_ids"])
    if len(cases) != 1 or (
        expected_case is not None and int(cases[0]) != expected_case
    ):
        raise ValueError("raw teacher capture mixes trajectories")
    if abs(float(payload["elapsed_time_s"][0])) > 1e-9 or not np.all(
        np.diff(payload["elapsed_time_s"]) > 0
    ):
        raise ValueError("raw teacher timestamps must start at zero and increase")
    if not np.all(np.diff(payload["phase_time_s"]) >= 0):
        raise ValueError("raw teacher phase time must be monotonic")
    previous = payload["observations"][:, PREVIOUS_ACTION_INDICES]
    if not np.allclose(previous, 0.0, atol=1e-12):
        raise ValueError("raw teacher previous-action placeholders must be zero")


def validate_policy_trace(
    payload: dict[str, np.ndarray], *, expected_case: int | None = None
) -> None:
    required = {
        "observations": 2,
        "applied_residual_actions": 2,
        "final_high_level_commands": 2,
        "baseline_wheel_actions": 2,
        "case_ids": 1,
        "elapsed_time_s": 1,
        "phase_time_s": 1,
        "post_step_position_error_m": 1,
        "post_step_attitude_error_deg": 1,
        "post_step_base_xy_yaw": 2,
        "post_step_camera_position_world_m": 2,
        "post_step_pitch_deg": 1,
        "post_step_riser_position_m": 1,
        "post_step_proxy_position_rad": 2,
    }
    for name, ndim in required.items():
        if name not in payload or np.asarray(payload[name]).ndim != ndim:
            raise ValueError(f"missing or invalid policy trace field {name}")
    count = len(payload["observations"])
    if count < 2 or any(len(payload[name]) != count for name in required):
        raise ValueError("policy trace row counts do not match")
    dimensions = {
        "observations": len(OBSERVATION_NAMES),
        "applied_residual_actions": len(ACTION_NAMES),
        "final_high_level_commands": 3,
        "baseline_wheel_actions": 2,
        "post_step_base_xy_yaw": 3,
        "post_step_camera_position_world_m": 3,
        "post_step_proxy_position_rad": 3,
    }
    for name, width in dimensions.items():
        if payload[name].shape[1] != width:
            raise ValueError(f"policy trace {name} dimension mismatch")
    if not all(np.isfinite(np.asarray(payload[name])).all() for name in required):
        raise ValueError("policy trace contains non-finite values")
    if np.max(np.abs(payload["applied_residual_actions"])) > 1.0 + 1e-6:
        raise ValueError("policy trace action exceeds normalized bounds")
    cases = np.unique(payload["case_ids"])
    if len(cases) != 1 or (
        expected_case is not None and int(cases[0]) != expected_case
    ):
        raise ValueError("policy trace mixes trajectories")
    if abs(float(payload["elapsed_time_s"][0])) > 1e-9 or not np.all(
        np.diff(payload["elapsed_time_s"]) > 0
    ):
        raise ValueError("policy trace timestamps must start at zero and increase")
    if not np.all(np.diff(payload["phase_time_s"]) >= 0):
        raise ValueError("policy trace phase time must be monotonic")


def save_policy_trace(
    path: Path, case: int, payload: dict[str, np.ndarray]
) -> None:
    validate_policy_trace(payload, expected_case=case)
    metadata = {
        "schema": POLICY_TRACE_SCHEMA,
        "case": case,
        "source": "executed_isaac_policy_rate_trace",
        "observation_names": list(OBSERVATION_NAMES),
        "action_names": list(ACTION_NAMES),
        "final_high_level_command_names": [
            "commanded_vx_m_s",
            "commanded_wz_rad_s",
            "commanded_riser_target_m",
        ],
        "observation_contract": "executed_state_with_execution_time_lookahead_v2",
        "sample_alignment_contract": "pre_action_observation_and_command_to_post_step_outcome_v1",
        "lookahead_horizons_s": list(LOOKAHEAD_HORIZONS_S),
        "camera_observation_frame": "physical_cam_link_fk",
        "target_attitude_contract": "semantic_dfr_to_physical_cam_v1",
        "teacher_labels_present": False,
        "residual_dataset_present": False,
        "trace_only": True,
        "training_started": False,
        "bc_authorized": False,
        "dagger_authorized": False,
        "ppo_authorized": False,
        "valid_for_training": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
        **payload,
    )


def load_policy_trace(
    path: Path,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        payload = {
            name: np.asarray(data[name])
            for name in data.files
            if name != "metadata_json"
        }
    if metadata.get("schema") != POLICY_TRACE_SCHEMA:
        raise ValueError(f"wrong policy trace schema in {path}")
    if metadata.get("observation_names") != list(OBSERVATION_NAMES):
        raise ValueError(f"policy trace observation contract mismatch in {path}")
    if metadata.get("trace_only") is not True:
        raise ValueError(f"policy trace artifact is not trace-only: {path}")
    if metadata.get("valid_for_training") is not False:
        raise ValueError(f"policy trace artifact is incorrectly training-enabled: {path}")
    case = int(metadata["case"])
    validate_policy_trace(payload, expected_case=case)
    return metadata, payload


def validate_shadow_teacher_trace(
    payload: dict[str, np.ndarray],
    *,
    action_scales: np.ndarray = ACTION_SCALES,
    expected_case: int | None = None,
) -> None:
    validate_policy_trace(payload, expected_case=expected_case)
    scales = np.asarray(action_scales, dtype=np.float64)
    if scales.shape != (3,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("invalid shadow teacher action scales")
    required = {
        "shadow_teacher_raw_residual_commands": 2,
        "shadow_teacher_normalized_residual_actions": 2,
        "shadow_teacher_high_level_commands": 2,
    }
    count = len(payload["observations"])
    for name, ndim in required.items():
        value = np.asarray(payload.get(name))
        if value.ndim != ndim or len(value) != count or value.shape[1] != 3:
            raise ValueError(f"missing or invalid shadow teacher field {name}")
        if not np.isfinite(value).all():
            raise ValueError(f"shadow teacher field {name} is non-finite")
    if not np.allclose(
        payload["shadow_teacher_raw_residual_commands"],
        payload["shadow_teacher_normalized_residual_actions"] * scales,
        atol=2e-7,
    ):
        raise ValueError("shadow teacher raw/normalized action mismatch")
    reconstructed = np.column_stack(
        (
            payload["observations"][:, OBSERVATION_INDEX["feedforward_vx_m_s"]]
            + scales[0]
            * payload["shadow_teacher_normalized_residual_actions"][:, 0],
            payload["observations"][:, OBSERVATION_INDEX["feedforward_wz_rad_s"]]
            + scales[1]
            * payload["shadow_teacher_normalized_residual_actions"][:, 1],
            payload["observations"][:, OBSERVATION_INDEX["riser_position_m"]]
            + scales[2]
            * payload["shadow_teacher_normalized_residual_actions"][:, 2],
        )
    )
    if not np.allclose(
        reconstructed,
        payload["shadow_teacher_high_level_commands"],
        atol=2e-6,
    ):
        raise ValueError("shadow teacher command reconstruction failed")


def save_shadow_teacher_trace(
    path: Path,
    case: int,
    payload: dict[str, np.ndarray],
    *,
    action_scales: np.ndarray = ACTION_SCALES,
    visited_state_source: str = "learned_policy",
) -> None:
    scales = np.asarray(action_scales, dtype=np.float64)
    if visited_state_source not in {"learned_policy", "deterministic_controller"}:
        raise ValueError("invalid shadow teacher visited-state source")
    validate_shadow_teacher_trace(
        payload, action_scales=scales, expected_case=case
    )
    metadata = {
        "schema": SHADOW_TEACHER_TRACE_SCHEMA,
        "case": case,
        "source": (
            "policy_visited_isaac_state_with_shadow_deterministic_teacher"
            if visited_state_source == "learned_policy"
            else "deterministic_controller_visited_isaac_state_with_shadow_labels"
        ),
        "visited_state_source": visited_state_source,
        "observation_names": list(OBSERVATION_NAMES),
        "applied_action_names": list(ACTION_NAMES),
        "shadow_teacher_raw_action_names": [
            "residual_vx_m_s",
            "residual_wz_rad_s",
            "residual_riser_target_m",
        ],
        "shadow_teacher_normalized_action_names": list(ACTION_NAMES),
        "action_scales": scales.tolist(),
        "sample_alignment_contract": (
            "pre_action_policy_visited_state_shadow_teacher_and_applied_policy_"
            "to_post_step_outcome_v1"
            if visited_state_source == "learned_policy"
            else "pre_action_deterministic_controller_state_shadow_label_to_post_"
            "step_outcome_v1"
        ),
        "shadow_teacher_computed_before_policy_overwrite": (
            visited_state_source == "learned_policy"
        ),
        "shadow_label_computed_before_command_application": True,
        "shadow_teacher_applied_to_commands": False,
        "shadow_teacher_labels_present": True,
        "shadow_teacher_labels_admitted_for_training": False,
        "residual_dataset_present": False,
        "trace_only": True,
        "training_started": False,
        "bc_authorized": False,
        "dagger_authorized": False,
        "ppo_authorized": False,
        "valid_for_training": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
        **payload,
    )


def load_shadow_teacher_trace(
    path: Path,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        payload = {
            name: np.asarray(data[name])
            for name in data.files
            if name != "metadata_json"
        }
    if metadata.get("schema") != SHADOW_TEACHER_TRACE_SCHEMA:
        raise ValueError(f"wrong shadow teacher trace schema in {path}")
    if (
        "visited_state_source" not in metadata
        and metadata.get("source")
        == "policy_visited_isaac_state_with_shadow_deterministic_teacher"
    ):
        metadata["visited_state_source"] = "learned_policy"
    checks = {
        "trace_only": metadata.get("trace_only") is True,
        "not_trainable": metadata.get("valid_for_training") is False,
        "teacher_unapplied": metadata.get("shadow_teacher_applied_to_commands")
        is False,
        "labels_unadmitted": metadata.get(
            "shadow_teacher_labels_admitted_for_training"
        )
        is False,
        "dagger_closed": metadata.get("dagger_authorized") is False,
        "visited_state_source": metadata.get("visited_state_source")
        in {"learned_policy", "deterministic_controller"},
    }
    if not all(checks.values()):
        raise ValueError(f"shadow teacher trace contract mismatch in {path}: {checks}")
    case = int(metadata["case"])
    validate_shadow_teacher_trace(
        payload,
        action_scales=np.asarray(metadata.get("action_scales"), dtype=np.float64),
        expected_case=case,
    )
    return metadata, payload


def save_raw_teacher_case(
    path: Path, case: int, payload: dict[str, np.ndarray]
) -> None:
    validate_raw_teacher_case(payload, expected_case=case)
    metadata = {
        "schema": RAW_TEACHER_SCHEMA,
        "case": case,
        "source": "executed_isaac_state_and_deterministic_controller",
        "observation_names": list(OBSERVATION_NAMES),
        "raw_action_names": [
            "residual_vx_m_s",
            "residual_wz_rad_s",
            "residual_riser_target_m",
        ],
        "observation_contract": "executed_state_with_execution_time_lookahead_v2",
        "lookahead_horizons_s": list(LOOKAHEAD_HORIZONS_S),
        "lookahead_reference_clock": "execution_time_s",
        "camera_observation_frame": "physical_cam_link_fk",
        "target_attitude_contract": "semantic_dfr_to_physical_cam_v1",
        "previous_action_contract": "zero_placeholder_relabel_after_scale_freeze_v1",
        "action_scale_frozen": False,
        "source_action_labels_used": False,
        "physical_gimbal_labels_used_as_actions": False,
        "raw_residual_applied_to_commands": False,
        "training_started": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "valid_for_training": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
        **payload,
    )


def load_raw_teacher_case(
    path: Path,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        payload = {
            name: np.asarray(data[name])
            for name in data.files
            if name != "metadata_json"
        }
    if metadata.get("schema") != RAW_TEACHER_SCHEMA:
        raise ValueError(f"wrong raw teacher schema in {path}")
    if metadata.get("observation_names") != list(OBSERVATION_NAMES):
        raise ValueError(f"raw teacher observation contract mismatch in {path}")
    if metadata.get("previous_action_contract") != (
        "zero_placeholder_relabel_after_scale_freeze_v1"
    ):
        raise ValueError(f"raw teacher previous-action contract mismatch in {path}")
    if metadata.get("valid_for_training") is not False:
        raise ValueError(f"raw teacher artifact is incorrectly training-enabled: {path}")
    case = int(metadata["case"])
    validate_raw_teacher_case(payload, expected_case=case)
    return metadata, payload


def normalize_raw_teacher_payload(
    payload: dict[str, np.ndarray], action_scales: np.ndarray
) -> dict[str, np.ndarray]:
    validate_raw_teacher_case(payload)
    scales = np.asarray(action_scales, dtype=np.float64)
    if scales.shape != (3,) or not np.isfinite(scales).all() or np.any(scales <= 0.0):
        raise ValueError("action scales must contain three positive finite values")
    actions = np.asarray(payload["raw_residual_commands"], dtype=np.float64) / scales
    if np.max(np.abs(actions)) >= 1.0 - 1e-6:
        raise ValueError("frozen action scales do not leave an unclipped margin")
    observations = np.asarray(payload["observations"], dtype=np.float32).copy()
    previous_actions = np.zeros_like(actions, dtype=np.float32)
    previous_actions[1:] = actions[:-1].astype(np.float32)
    observations[:, PREVIOUS_ACTION_INDICES] = previous_actions
    normalized = {
        "observations": observations,
        "actions": actions.astype(np.float32),
        "case_ids": np.asarray(payload["case_ids"]).copy(),
        "elapsed_time_s": np.asarray(payload["elapsed_time_s"]).copy(),
        "phase_time_s": np.asarray(payload["phase_time_s"]).copy(),
        "baseline_wheel_actions": np.asarray(
            payload["baseline_wheel_actions"]
        ).copy(),
        "teacher_commands": np.asarray(payload["teacher_commands"]).copy(),
    }
    validate_case_dataset(normalized)
    reconstructed = np.column_stack(
        (
            observations[:, OBSERVATION_INDEX["feedforward_vx_m_s"]]
            + scales[0] * normalized["actions"][:, 0],
            observations[:, OBSERVATION_INDEX["feedforward_wz_rad_s"]]
            + scales[1] * normalized["actions"][:, 1],
            observations[:, OBSERVATION_INDEX["riser_position_m"]]
            + scales[2] * normalized["actions"][:, 2],
        )
    )
    error = float(np.max(np.abs(reconstructed - normalized["teacher_commands"])))
    if error > 2e-6:
        raise ValueError(f"normalized teacher command reconstruction failed: {error}")
    return normalized


def save_case_dataset(path: Path, case: int, payload: dict[str, np.ndarray]) -> None:
    validate_case_dataset(payload, expected_case=case)
    metadata = {
        "schema": DATASET_SCHEMA,
        "case": case,
        "source": "executed_isaac_state_and_deterministic_controller",
        "observation_names": list(OBSERVATION_NAMES),
        "action_names": list(ACTION_NAMES),
        "action_scales": ACTION_SCALES.tolist(),
        "action_contract": "trajectory_command_residual_above_frozen_balance_lqr_v1",
        "observation_contract": "executed_state_with_execution_time_lookahead_v2",
        "lookahead_horizons_s": list(LOOKAHEAD_HORIZONS_S),
        "lookahead_reference_clock": "execution_time_s",
        "camera_observation_frame": "physical_cam_link_fk",
        "target_attitude_contract": "semantic_dfr_to_physical_cam_v1",
        "source_action_labels_used": False,
        "physical_gimbal_labels_used_as_actions": False,
        "training_started": False,
        "ppo_authorized": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, metadata_json=np.array(json.dumps(metadata, sort_keys=True)), **payload)


def load_case_dataset(path: Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        payload = {name: np.asarray(data[name]) for name in data.files if name != "metadata_json"}
    if metadata.get("schema") != DATASET_SCHEMA:
        raise ValueError(f"wrong residual dataset schema in {path}")
    if metadata.get("observation_names") != list(OBSERVATION_NAMES):
        raise ValueError(f"observation contract mismatch in {path}")
    if metadata.get("observation_contract") != (
        "executed_state_with_execution_time_lookahead_v2"
    ):
        raise ValueError(f"observation contract version mismatch in {path}")
    if metadata.get("lookahead_horizons_s") != list(LOOKAHEAD_HORIZONS_S):
        raise ValueError(f"lookahead horizon mismatch in {path}")
    if not np.allclose(metadata.get("action_scales"), ACTION_SCALES, atol=1e-12):
        raise ValueError(f"residual action scale mismatch in {path}")
    case = int(metadata["case"])
    validate_case_dataset(payload, expected_case=case)
    return metadata, payload
