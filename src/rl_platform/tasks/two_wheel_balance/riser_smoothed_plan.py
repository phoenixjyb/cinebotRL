"""Derived, duration-bounded riser plans with immutable source provenance."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .riser_exact_source import (
    EXACT_SOURCE_CONTRACT,
    ExactSourceRiserReference,
    camera_envelope_vertical_shift,
    sha256_file,
)
from .riser_kinematics import UrdfRiserCameraKinematics
from .riser_playback import (
    PLAYBACK_BASE_LATERAL_LIMIT_MPS,
    PLAYBACK_BASE_LINEAR_LIMIT_MPS,
    PLAYBACK_BASE_YAW_RATE_LIMIT_RAD_S,
    PLAYBACK_POSITION_MAX_LIMIT_M,
    PLAYBACK_POSITION_P95_LIMIT_M,
    PLAYBACK_PROXY_RATE_LIMIT_RAD_S,
    PLAYBACK_RISER_RATE_LIMIT_MPS,
    RiserPlaybackPlan,
    playback_plan_from_kinematic_plan,
    riser_playback_kinematic_gate,
    riser_playback_kinematic_metrics,
)
from .riser_reference import CorrectedRiserReference
from .riser_rs4_attitude import (
    accepted62_body_basis_rotation,
    plan_rs4_attitude_commands,
    rs4_command_to_proxy_joint_order,
    unwrap_proxy_joint_yaw,
)
from .riser_rs4_reference import _plan_preview_rs4_riser_reference


SMOOTHED_PLAN_SCHEMA = "cinebotrl_two_wheel_riser_smoothed_plan_v1"
SMOOTHED_TARGET_SCHEMA = "derived_smoothed_target_v1"
MAXIMUM_EXECUTION_SOURCE_DURATION_RATIO = 2.0
MAXIMUM_PATH_LENGTH_RELATIVE_DRIFT = 0.05
MAXIMUM_PRE_DENSIFICATION_BRANCH_STEP_RAD = 0.25
MINIMUM_EXECUTION_INTERVAL_S = 0.005
SMOOTHING_SIGMA_CANDIDATES = (0.0, 4.0, 8.0, 12.0, 16.0)
PREVIEW_CONFIGURATIONS = (
    (0.05, 2.75),
    (0.10, 2.75),
    (0.15, 2.75),
    (0.25, 2.75),
    (0.40, 1.00),
    (0.50, 1.00),
)
RESET_YAW_LOOKAHEAD_M = 0.50
RECOVERY_CONFIGURATIONS = (
    (16.0, 1.0, 0.25, 2.75, "reverse_path"),
    (0.0, 1.0, 0.25, 2.75, "forward_path"),
    (0.0, 1.0, 0.25, 2.75, "reverse_path"),
    (0.0, 1.0, 0.50, 1.00, "reverse_path"),
    (16.0, 0.45, 0.65, 1.00, "forward_path"),
    (64.0, 0.1276273593606172, 0.90, 1.00, "forward_path"),
)
BATCH_RECOVERY_ITERATIONS = 1000
BATCH_RECOVERY_RESIDUAL_BOUND = 0.05
BATCH_RECOVERY_REGULARIZATION = 0.002
BATCH_RECOVERY_LEARNING_RATE = 0.02
BATCH_RECOVERY_DURATION_WEIGHT = 250.0
BATCH_RECOVERY_DURATION_RATIO_TARGET = 1.965
BATCH_RECOVERY_SEED_POSITION_P95_LIMIT_M = 0.20
BATCH_RECOVERY_SEED_DURATION_RATIO_LIMIT = 2.01


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _path_length(position_m: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(position_m, axis=0), axis=1)))


def smooth_source_positions(
    source_position_m: np.ndarray,
    sigma_samples: float,
    blend_factor: float = 1.0,
) -> np.ndarray:
    """Smooth horizontal geometry while pinning both source endpoints exactly."""

    source = np.asarray(source_position_m, dtype=np.float64)
    _require(
        source.ndim == 2
        and source.shape[1] == 3
        and len(source) >= 2
        and np.isfinite(source).all(),
        "source positions must be finite shape (N,3)",
    )
    _require(
        math.isfinite(sigma_samples) and sigma_samples >= 0.0,
        "smoothing sigma must be finite and non-negative",
    )
    _require(
        math.isfinite(blend_factor) and 0.0 <= blend_factor <= 1.0,
        "smoothing blend factor must be finite and in [0,1]",
    )
    if sigma_samples == 0.0:
        return source.copy()
    result = source.copy()
    result[:, :2] = gaussian_filter1d(
        source[:, :2], sigma_samples, axis=0, mode="nearest"
    )
    progress = np.linspace(0.0, 1.0, len(source), dtype=np.float64)
    result += (1.0 - progress)[:, None] * (source[0] - result[0])
    result += progress[:, None] * (source[-1] - result[-1])
    result[:, 2] = source[:, 2]
    result = source + blend_factor * (result - source)
    result[0] = source[0]
    result[-1] = source[-1]
    return result


def derived_reset_yaw_rad(
    source_position_m: np.ndarray,
    source_initial_yaw_rad: float,
    mode: str,
) -> float:
    """Choose a reset yaw from the first 0.5 m of immutable source motion."""

    if mode == "source":
        return float(source_initial_yaw_rad)
    _require(mode in {"forward_path", "reverse_path"}, "invalid reset yaw mode")
    position = np.asarray(source_position_m, dtype=np.float64)
    arc_length = np.r_[
        0.0,
        np.cumsum(np.linalg.norm(np.diff(position[:, :2], axis=0), axis=1)),
    ]
    index = min(
        int(np.searchsorted(arc_length, RESET_YAW_LOOKAHEAD_M)), len(position) - 1
    )
    delta = position[index, :2] - position[0, :2]
    _require(float(np.linalg.norm(delta)) > 1e-8, "source has no reset-yaw motion")
    yaw = math.atan2(float(delta[1]), float(delta[0]))
    return yaw if mode == "forward_path" else yaw + math.pi


def _point_to_polyline_distances(
    points_m: np.ndarray, polyline_m: np.ndarray
) -> np.ndarray:
    points = np.asarray(points_m, dtype=np.float64)
    polyline = np.asarray(polyline_m, dtype=np.float64)
    _require(
        points.ndim == 2
        and points.shape[1] == 3
        and polyline.ndim == 2
        and polyline.shape[1] == 3
        and len(polyline) >= 2,
        "points and polyline must have shape (N,3)",
    )
    start = polyline[:-1]
    delta = np.diff(polyline, axis=0)
    denominator = np.sum(delta * delta, axis=1)
    distance = np.empty(len(points), dtype=np.float64)
    for index, point in enumerate(points):
        projection = np.zeros(len(delta), dtype=np.float64)
        nonzero = denominator > 1e-18
        projection[nonzero] = np.sum(
            (point - start[nonzero]) * delta[nonzero], axis=1
        ) / denominator[nonzero]
        projection = np.clip(projection, 0.0, 1.0)
        closest = start + projection[:, None] * delta
        distance[index] = float(np.min(np.linalg.norm(closest - point, axis=1)))
    return distance


def smoothed_path_metrics(
    source_position_m: np.ndarray, smoothed_position_m: np.ndarray
) -> dict[str, float]:
    source = np.asarray(source_position_m, dtype=np.float64)
    smoothed = np.asarray(smoothed_position_m, dtype=np.float64)
    _require(source.shape == smoothed.shape, "source and smoothed paths must match")
    source_length = _path_length(source)
    _require(source_length > 0.0, "source path length must be positive")
    smoothed_length = _path_length(smoothed)
    deviation = _point_to_polyline_distances(smoothed, source)
    source_delta = np.diff(source, axis=0)
    smoothed_delta = np.diff(smoothed, axis=0)
    source_norm = np.linalg.norm(source_delta, axis=1)
    smoothed_norm = np.linalg.norm(smoothed_delta, axis=1)
    active = (source_norm > 1e-7) & (smoothed_norm > 1e-7)
    direction_cosine = np.ones(len(source_delta), dtype=np.float64)
    direction_cosine[active] = np.sum(
        source_delta[active] * smoothed_delta[active], axis=1
    ) / (source_norm[active] * smoothed_norm[active])
    return {
        "source_path_length_m": source_length,
        "smoothed_path_length_m": smoothed_length,
        "path_length_relative_drift": smoothed_length / source_length - 1.0,
        "source_polyline_deviation_p95_m": float(np.percentile(deviation, 95)),
        "source_polyline_deviation_max_m": float(np.max(deviation)),
        "start_position_error_m": float(np.linalg.norm(smoothed[0] - source[0])),
        "final_position_error_m": float(np.linalg.norm(smoothed[-1] - source[-1])),
        "minimum_segment_direction_cosine": float(np.min(direction_cosine)),
        "opposed_segment_direction_count": int(np.sum(direction_cosine < -1e-12)),
    }


def _feedforward(
    plan: RiserPlaybackPlan, execution_time_s: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dt = np.diff(execution_time_s)
    yaw = np.unwrap(plan.base_xy_yaw[:, 2])
    midpoint = 0.5 * (yaw[:-1] + yaw[1:])
    delta_xy = np.diff(plan.base_xy_yaw[:, :2], axis=0)
    forward = (
        np.cos(midpoint) * delta_xy[:, 0]
        + np.sin(midpoint) * delta_xy[:, 1]
    ) / dt
    return (
        np.column_stack((forward, np.diff(yaw) / dt)),
        np.diff(plan.riser_q) / dt,
        np.diff(plan.proxy_gimbal_q, axis=0) / dt[:, None],
    )


def retime_smoothed_plan_from_demands(
    plan: RiserPlaybackPlan, source_duration_s: float
) -> RiserPlaybackPlan:
    """Allocate time from actual commands, without retaining obsolete dwell."""

    plan.validate()
    _require(
        math.isfinite(source_duration_s) and source_duration_s > 0.0,
        "source duration must be finite and positive",
    )
    yaw = np.unwrap(plan.base_xy_yaw[:, 2])
    midpoint = 0.5 * (yaw[:-1] + yaw[1:])
    delta_xy = np.diff(plan.base_xy_yaw[:, :2], axis=0)
    forward = np.abs(
        np.cos(midpoint) * delta_xy[:, 0]
        + np.sin(midpoint) * delta_xy[:, 1]
    )
    lateral = np.abs(
        -np.sin(midpoint) * delta_xy[:, 0]
        + np.cos(midpoint) * delta_xy[:, 1]
    )
    proxy_step = np.max(np.abs(np.diff(plan.proxy_gimbal_q, axis=0)), axis=1)
    execution_dt = np.maximum.reduce(
        (
            np.full(len(delta_xy), MINIMUM_EXECUTION_INTERVAL_S),
            forward / PLAYBACK_BASE_LINEAR_LIMIT_MPS,
            lateral / PLAYBACK_BASE_LATERAL_LIMIT_MPS,
            np.abs(np.diff(yaw)) / PLAYBACK_BASE_YAW_RATE_LIMIT_RAD_S,
            np.abs(np.diff(plan.riser_q)) / PLAYBACK_RISER_RATE_LIMIT_MPS,
            proxy_step / PLAYBACK_PROXY_RATE_LIMIT_RAD_S,
        )
    )
    if float(np.sum(execution_dt)) < source_duration_s:
        execution_dt *= source_duration_s / float(np.sum(execution_dt))
    execution_time = np.r_[0.0, np.cumsum(execution_dt)]
    base_feedforward, riser_feedforward, proxy_feedforward = _feedforward(
        plan, execution_time
    )
    result = replace(
        plan,
        time_s=execution_time,
        feedforward_v_wz=base_feedforward,
        feedforward_riser_velocity=riser_feedforward,
        feedforward_proxy_velocity=proxy_feedforward,
    )
    result.validate()
    return result


def _provisional_schedule(
    position_m: np.ndarray, source_time_s: np.ndarray
) -> np.ndarray:
    delta = np.diff(position_m, axis=0)
    execution_dt = np.maximum.reduce(
        (
            np.diff(source_time_s),
            np.linalg.norm(delta[:, :2], axis=1)
            / PLAYBACK_BASE_LINEAR_LIMIT_MPS,
            np.abs(delta[:, 2]) / PLAYBACK_RISER_RATE_LIMIT_MPS,
            np.full(len(delta), MINIMUM_EXECUTION_INTERVAL_S),
        )
    )
    target_duration = (
        MAXIMUM_EXECUTION_SOURCE_DURATION_RATIO * float(source_time_s[-1])
    )
    execution_dt *= target_duration / float(np.sum(execution_dt))
    return np.r_[0.0, np.cumsum(execution_dt)]


def _strategy(lookahead_m: float, heading_gain: float) -> str:
    return f"smoothed_preview_{lookahead_m:.2f}m_g{heading_gain:.2f}"


def transition_metrics(plan: RiserPlaybackPlan) -> dict[str, float]:
    return {
        "maximum_pre_densification_base_branch_step_rad": float(
            np.max(np.abs(np.diff(np.unwrap(plan.base_xy_yaw[:, 2]))))
        ),
        "maximum_pre_densification_proxy_branch_step_rad": float(
            np.max(np.abs(np.diff(plan.proxy_gimbal_q, axis=0)))
        ),
    }


@dataclass(frozen=True)
class SmoothedPlanResult:
    plan: RiserPlaybackPlan
    smoothed_position_source_frame_m: np.ndarray
    smoothing_sigma_samples: float
    smoothing_blend_factor: float
    lookahead_distance_m: float
    heading_gain: float
    reset_yaw_mode: str
    reset_yaw_rad: float
    path_metrics: dict[str, float]
    transition_metrics: dict[str, float]
    kinematic_metrics: dict[str, float]
    kinematic_checks: dict[str, bool]
    checks: dict[str, bool]
    attempts: tuple[dict[str, object], ...]

    @property
    def passed(self) -> bool:
        return all(self.checks.values()) and all(self.kinematic_checks.values())


def batch_unicycle_recovery_seed_eligible(
    result: SmoothedPlanResult, source_duration_s: float
) -> bool:
    """Admit only a near-gate seed with one position-p95 failure."""

    _require(
        math.isfinite(source_duration_s) and source_duration_s > 0.0,
        "source duration must be finite and positive",
    )
    failed_plan = [key for key, value in result.checks.items() if not value]
    failed_kinematic = [
        key for key, value in result.kinematic_checks.items() if not value
    ]
    duration_ratio = float(result.plan.time_s[-1] / source_duration_s)
    return (
        not failed_plan
        and set(failed_kinematic) == {"position_p95_bounded"}
        and result.kinematic_metrics["position_error_p95_m"]
        <= BATCH_RECOVERY_SEED_POSITION_P95_LIMIT_M
        and duration_ratio <= BATCH_RECOVERY_SEED_DURATION_RATIO_LIMIT
    )


def _build_batch_unicycle_recovery(
    source: ExactSourceRiserReference,
    kinematics: UrdfRiserCameraKinematics,
    seed: SmoothedPlanResult,
) -> SmoothedPlanResult:
    """Globally improve a near-gate seed using only legal unicycle controls."""

    import torch

    torch.set_num_threads(4)
    smoothed = seed.smoothed_position_source_frame_m
    provisional_time = _provisional_schedule(smoothed, source.source_time_s)
    reference = CorrectedRiserReference(
        case=source.case,
        path=source.source_json_path,
        positions_m=smoothed,
        semantic_dfr_quat_wxyz=source.planning_reference(
            source.source_time_s
        ).semantic_dfr_quat_wxyz,
        time_s=provisional_time,
        initial_base_yaw_rad=seed.reset_yaw_rad,
        metadata={
            "source": SMOOTHED_TARGET_SCHEMA,
            "source_manifest_sha256": source.package_manifest_sha256,
            "source_json_sha256": source.source_json_sha256,
        },
    )
    plan = seed.plan
    dt_np = np.diff(provisional_time)
    seed_yaw = np.unwrap(plan.base_xy_yaw[:, 2])
    seed_delta = np.diff(plan.base_xy_yaw[:, :2], axis=0)
    seed_mid_yaw = 0.5 * (seed_yaw[:-1] + seed_yaw[1:])
    seed_control_np = np.column_stack(
        (
            (
                np.cos(seed_mid_yaw) * seed_delta[:, 0]
                + np.sin(seed_mid_yaw) * seed_delta[:, 1]
            )
            / dt_np,
            np.diff(seed_yaw) / dt_np,
        )
    )
    seed_control_np = np.clip(
        seed_control_np,
        [-PLAYBACK_BASE_LINEAR_LIMIT_MPS, -PLAYBACK_BASE_YAW_RATE_LIMIT_RAD_S],
        [PLAYBACK_BASE_LINEAR_LIMIT_MPS, PLAYBACK_BASE_YAW_RATE_LIMIT_RAD_S],
    )

    offset_body_np = np.empty((len(seed_yaw), 2), dtype=np.float64)
    for index in range(len(seed_yaw)):
        camera_xy = kinematics.world_transform(
            plan.base_xy_yaw[index],
            float(plan.riser_q[index]),
            plan.proxy_gimbal_q[index],
        )[:2, 3]
        world_offset = camera_xy - plan.base_xy_yaw[index, :2]
        cosine = math.cos(float(seed_yaw[index]))
        sine = math.sin(float(seed_yaw[index]))
        offset_body_np[index] = (
            cosine * world_offset[0] + sine * world_offset[1],
            -sine * world_offset[0] + cosine * world_offset[1],
        )

    dt = torch.as_tensor(dt_np, dtype=torch.float64)
    seed_control = torch.as_tensor(seed_control_np, dtype=torch.float64)
    lower = torch.maximum(
        seed_control - BATCH_RECOVERY_RESIDUAL_BOUND,
        torch.as_tensor(
            [-PLAYBACK_BASE_LINEAR_LIMIT_MPS, -PLAYBACK_BASE_YAW_RATE_LIMIT_RAD_S],
            dtype=torch.float64,
        ),
    )
    upper = torch.minimum(
        seed_control + BATCH_RECOVERY_RESIDUAL_BOUND,
        torch.as_tensor(
            [PLAYBACK_BASE_LINEAR_LIMIT_MPS, PLAYBACK_BASE_YAW_RATE_LIMIT_RAD_S],
            dtype=torch.float64,
        ),
    )
    control = torch.nn.Parameter(seed_control.clone())
    target_xy = torch.as_tensor(
        plan.target_position_world_m[:, :2], dtype=torch.float64
    )
    offset_body = torch.as_tensor(offset_body_np, dtype=torch.float64)
    initial = torch.as_tensor(plan.base_xy_yaw[0], dtype=torch.float64)
    fixed_demand = torch.as_tensor(
        np.maximum(
            np.abs(np.diff(plan.riser_q)) / dt_np / PLAYBACK_RISER_RATE_LIMIT_MPS,
            np.max(np.abs(np.diff(plan.proxy_gimbal_q, axis=0)), axis=1)
            / dt_np
            / PLAYBACK_PROXY_RATE_LIMIT_RAD_S,
        ),
        dtype=torch.float64,
    )
    optimizer = torch.optim.Adam(
        [control], lr=BATCH_RECOVERY_LEARNING_RATE
    )

    def rollout(candidate_control: torch.Tensor) -> torch.Tensor:
        x, y, yaw = initial
        states = [torch.stack((x, y, yaw))]
        for index in range(len(dt)):
            velocity, yaw_rate = candidate_control[index]
            half_delta = 0.5 * yaw_rate * dt[index]
            scale = torch.sinc(half_delta / torch.pi)
            x = x + velocity * dt[index] * scale * torch.cos(yaw + half_delta)
            y = y + velocity * dt[index] * scale * torch.sin(yaw + half_delta)
            yaw = yaw + 2.0 * half_delta
            states.append(torch.stack((x, y, yaw)))
        return torch.stack(states)

    duration_ratio = torch.tensor(float("inf"), dtype=torch.float64)
    for _ in range(BATCH_RECOVERY_ITERATIONS):
        optimizer.zero_grad()
        state = rollout(control)
        cosine = torch.cos(state[:, 2])
        sine = torch.sin(state[:, 2])
        camera_xy = torch.column_stack(
            (
                state[:, 0]
                + cosine * offset_body[:, 0]
                - sine * offset_body[:, 1],
                state[:, 1]
                + sine * offset_body[:, 0]
                + cosine * offset_body[:, 1],
            )
        )
        error_norm = torch.linalg.vector_norm(camera_xy - target_xy, dim=1)
        tracking = torch.mean(error_norm**4) / PLAYBACK_POSITION_P95_LIMIT_M**4
        regularization = BATCH_RECOVERY_REGULARIZATION * torch.mean(
            ((control - seed_control) / BATCH_RECOVERY_RESIDUAL_BOUND) ** 2
        )
        interval_demand = torch.maximum(
            fixed_demand,
            torch.maximum(
                torch.abs(control[:, 0]) / PLAYBACK_BASE_LINEAR_LIMIT_MPS,
                torch.abs(control[:, 1]) / PLAYBACK_BASE_YAW_RATE_LIMIT_RAD_S,
            ),
        )
        duration_ratio = (
            torch.sum(dt * interval_demand) / float(source.source_time_s[-1])
        )
        duration_penalty = BATCH_RECOVERY_DURATION_WEIGHT * torch.relu(
            duration_ratio - BATCH_RECOVERY_DURATION_RATIO_TARGET
        ) ** 2
        (tracking + regularization + duration_penalty).backward()
        optimizer.step()
        with torch.no_grad():
            control.clamp_(lower, upper)

    optimized_base = rollout(control).detach().cpu().numpy()
    attitude = plan_rs4_attitude_commands(
        reference,
        accepted62_body_basis_rotation(),
        optimized_base[:, 2],
    )
    optimized_proxy = unwrap_proxy_joint_yaw(
        rs4_command_to_proxy_joint_order(attitude.command_yaw_roll_pitch_rad)
    )
    optimized_riser = plan.riser_q.copy()
    for index in range(len(optimized_riser)):
        optimized_riser[index] = kinematics.solve_position(
            plan.target_position_world_m[index],
            float(optimized_base[index, 2]),
            optimized_proxy[index],
        ).base_xy_yaw_riser[3]

    optimized_plan = replace(
        plan,
        base_xy_yaw=optimized_base,
        riser_q=optimized_riser,
        proxy_gimbal_q=optimized_proxy,
        planning_strategy="smoothed_batch_unicycle_v1",
    )
    base_ff, riser_ff, proxy_ff = _feedforward(
        optimized_plan, optimized_plan.time_s
    )
    optimized_plan = replace(
        optimized_plan,
        feedforward_v_wz=base_ff,
        feedforward_riser_velocity=riser_ff,
        feedforward_proxy_velocity=proxy_ff,
    )
    optimized_plan = retime_smoothed_plan_from_demands(
        optimized_plan, float(source.source_time_s[-1])
    )
    transitions = transition_metrics(optimized_plan)
    kinematic_metrics = riser_playback_kinematic_metrics(
        optimized_plan, kinematics
    )
    kinematic_checks = riser_playback_kinematic_gate(
        kinematic_metrics, kinematics
    )
    duration_ratio_value = float(
        optimized_plan.time_s[-1] / source.source_time_s[-1]
    )
    checks = dict(seed.checks)
    checks.update(
        {
            "execution_not_faster_than_source": duration_ratio_value
            >= 1.0 - 1e-12,
            "execution_duration_ratio_bounded": duration_ratio_value
            <= MAXIMUM_EXECUTION_SOURCE_DURATION_RATIO + 1e-12,
            "base_branch_step_bounded": transitions[
                "maximum_pre_densification_base_branch_step_rad"
            ]
            <= MAXIMUM_PRE_DENSIFICATION_BRANCH_STEP_RAD + 1e-12,
            "proxy_branch_step_bounded": transitions[
                "maximum_pre_densification_proxy_branch_step_rad"
            ]
            <= MAXIMUM_PRE_DENSIFICATION_BRANCH_STEP_RAD + 1e-12,
            "global_proxy_branch_feasible": bool(
                np.all(attitude.command_feasible)
            ),
            "execution_schedule_strict": bool(
                optimized_plan.time_s[0] == 0.0
                and np.all(np.diff(optimized_plan.time_s) > 0.0)
            ),
        }
    )
    return replace(
        seed,
        plan=optimized_plan,
        transition_metrics=transitions,
        kinematic_metrics={
            **kinematic_metrics,
            "batch_surrogate_duration_ratio": float(duration_ratio.detach()),
            "batch_maximum_control_delta": float(
                torch.max(torch.abs(control.detach() - seed_control))
            ),
        },
        kinematic_checks=kinematic_checks,
        checks=checks,
    )


def _build_candidate(
    source: ExactSourceRiserReference,
    kinematics: UrdfRiserCameraKinematics,
    *,
    sigma_samples: float,
    smoothing_blend_factor: float = 1.0,
    lookahead_distance_m: float,
    heading_gain: float,
    reset_yaw_mode: str = "source",
) -> SmoothedPlanResult:
    smoothed = smooth_source_positions(
        source.source_position_world_m, sigma_samples, smoothing_blend_factor
    )
    path = smoothed_path_metrics(source.source_position_world_m, smoothed)
    provisional_time = _provisional_schedule(smoothed, source.source_time_s)
    source_attitude_wxyz = source.planning_reference(
        source.source_time_s
    ).semantic_dfr_quat_wxyz
    reset_yaw = derived_reset_yaw_rad(
        source.source_position_world_m,
        source.initial_base_yaw_rad,
        reset_yaw_mode,
    )
    reference = CorrectedRiserReference(
        case=source.case,
        path=source.source_json_path,
        positions_m=smoothed,
        semantic_dfr_quat_wxyz=source_attitude_wxyz,
        time_s=provisional_time,
        initial_base_yaw_rad=reset_yaw,
        metadata={
            "source": SMOOTHED_TARGET_SCHEMA,
            "source_manifest_sha256": source.package_manifest_sha256,
            "source_json_sha256": source.source_json_sha256,
        },
    )
    vertical_shift_m, workspace_compatible = camera_envelope_vertical_shift(
        smoothed
    )
    kinematic_plan = _plan_preview_rs4_riser_reference(
        reference,
        kinematics,
        vertical_shift_m=vertical_shift_m,
        maximum_base_yaw_rate_rad_s=PLAYBACK_BASE_YAW_RATE_LIMIT_RAD_S,
        maximum_base_linear_velocity_mps=PLAYBACK_BASE_LINEAR_LIMIT_MPS,
        maximum_riser_rate_mps=PLAYBACK_RISER_RATE_LIMIT_MPS,
        position_scale_m=0.05,
        attitude_tolerance_rad=math.radians(2.0),
        lookahead_distance_m=lookahead_distance_m,
        heading_gain=heading_gain,
    )
    preliminary = playback_plan_from_kinematic_plan(
        reference,
        replace(
            kinematic_plan,
            planning_strategy=_strategy(lookahead_distance_m, heading_gain),
        ),
    )

    # Re-resolve the complete sequence so an Euler branch is never selected
    # greedily at one anchor and paid for later as artificial dwell time.
    attitude = plan_rs4_attitude_commands(
        reference,
        accepted62_body_basis_rotation(),
        preliminary.base_xy_yaw[:, 2],
    )
    proxy = unwrap_proxy_joint_yaw(
        rs4_command_to_proxy_joint_order(attitude.command_yaw_roll_pitch_rad)
    )
    provisional_proxy_plan = replace(preliminary, proxy_gimbal_q=proxy)
    base_ff, riser_ff, proxy_ff = _feedforward(
        provisional_proxy_plan, preliminary.time_s
    )
    provisional_proxy_plan = replace(
        provisional_proxy_plan,
        feedforward_v_wz=base_ff,
        feedforward_riser_velocity=riser_ff,
        feedforward_proxy_velocity=proxy_ff,
    )
    transitions = transition_metrics(provisional_proxy_plan)
    plan = retime_smoothed_plan_from_demands(
        provisional_proxy_plan, float(source.source_time_s[-1])
    )
    kinematic_metrics = riser_playback_kinematic_metrics(plan, kinematics)
    kinematic_checks = riser_playback_kinematic_gate(
        kinematic_metrics, kinematics
    )
    duration_ratio = float(plan.time_s[-1] / source.source_time_s[-1])
    checks = {
        "source_start_preserved": path["start_position_error_m"] <= 1e-12,
        "source_final_preserved": path["final_position_error_m"] <= 1e-12,
        "source_order_mapping_preserved": len(plan.time_s)
        == source.source_pose_count,
        "source_motion_direction_preserved": path[
            "opposed_segment_direction_count"
        ]
        == 0,
        "path_length_within_5_percent": abs(
            path["path_length_relative_drift"]
        )
        <= MAXIMUM_PATH_LENGTH_RELATIVE_DRIFT + 1e-12,
        "source_polyline_p95_bounded": path[
            "source_polyline_deviation_p95_m"
        ]
        <= PLAYBACK_POSITION_P95_LIMIT_M + 1e-12,
        "source_polyline_max_bounded": path[
            "source_polyline_deviation_max_m"
        ]
        <= PLAYBACK_POSITION_MAX_LIMIT_M + 1e-12,
        "execution_not_faster_than_source": duration_ratio >= 1.0 - 1e-12,
        "execution_duration_ratio_bounded": duration_ratio
        <= MAXIMUM_EXECUTION_SOURCE_DURATION_RATIO + 1e-12,
        "base_branch_step_bounded": transitions[
            "maximum_pre_densification_base_branch_step_rad"
        ]
        <= MAXIMUM_PRE_DENSIFICATION_BRANCH_STEP_RAD + 1e-12,
        "proxy_branch_step_bounded": transitions[
            "maximum_pre_densification_proxy_branch_step_rad"
        ]
        <= MAXIMUM_PRE_DENSIFICATION_BRANCH_STEP_RAD + 1e-12,
        "global_proxy_branch_feasible": bool(np.all(attitude.command_feasible)),
        "vertical_workspace_span_compatible": workspace_compatible,
        "execution_schedule_strict": bool(
            plan.time_s[0] == 0.0 and np.all(np.diff(plan.time_s) > 0.0)
        ),
    }
    return SmoothedPlanResult(
        plan=plan,
        smoothed_position_source_frame_m=smoothed,
        smoothing_sigma_samples=float(sigma_samples),
        smoothing_blend_factor=float(smoothing_blend_factor),
        lookahead_distance_m=float(lookahead_distance_m),
        heading_gain=float(heading_gain),
        reset_yaw_mode=reset_yaw_mode,
        reset_yaw_rad=reset_yaw,
        path_metrics=path,
        transition_metrics=transitions,
        kinematic_metrics=kinematic_metrics,
        kinematic_checks=kinematic_checks,
        checks=checks,
        attempts=(),
    )


def build_smoothed_riser_plan(
    source: ExactSourceRiserReference,
    kinematics: UrdfRiserCameraKinematics,
    *,
    smoothing_sigma_candidates: tuple[float, ...] = SMOOTHING_SIGMA_CANDIDATES,
    preview_configurations: tuple[tuple[float, float], ...] = PREVIEW_CONFIGURATIONS,
) -> SmoothedPlanResult:
    """Search a small deterministic canary portfolio and return fail-closed."""

    _require(bool(smoothing_sigma_candidates), "smoothing candidates are empty")
    _require(bool(preview_configurations), "preview configurations are empty")
    attempts: list[dict[str, object]] = []
    results: list[SmoothedPlanResult] = []
    for sigma_samples in smoothing_sigma_candidates:
        for lookahead_distance_m, heading_gain in preview_configurations:
            result = _build_candidate(
                source,
                kinematics,
                sigma_samples=sigma_samples,
                smoothing_blend_factor=1.0,
                lookahead_distance_m=lookahead_distance_m,
                heading_gain=heading_gain,
                reset_yaw_mode="source",
            )
            summary = {
                "smoothing_sigma_samples": float(sigma_samples),
                "smoothing_blend_factor": result.smoothing_blend_factor,
                "lookahead_distance_m": float(lookahead_distance_m),
                "heading_gain": float(heading_gain),
                "reset_yaw_mode": result.reset_yaw_mode,
                "reset_yaw_rad": result.reset_yaw_rad,
                "execution_source_duration_ratio": float(
                    result.plan.time_s[-1] / source.source_time_s[-1]
                ),
                "path_length_relative_drift": result.path_metrics[
                    "path_length_relative_drift"
                ],
                "position_error_p95_m": result.kinematic_metrics[
                    "position_error_p95_m"
                ],
                "position_error_max_m": result.kinematic_metrics[
                    "position_error_max_m"
                ],
                "failed_checks": [
                    key for key, value in result.checks.items() if not value
                ]
                + [
                    key
                    for key, value in result.kinematic_checks.items()
                    if not value
                ],
                "passed": result.passed,
            }
            attempts.append(summary)
            result = replace(result, attempts=tuple(attempts))
            results.append(result)
            if result.passed:
                return result

    for sigma_samples, blend, lookahead, gain, yaw_mode in RECOVERY_CONFIGURATIONS:
        result = _build_candidate(
            source,
            kinematics,
            sigma_samples=sigma_samples,
            smoothing_blend_factor=blend,
            lookahead_distance_m=lookahead,
            heading_gain=gain,
            reset_yaw_mode=yaw_mode,
        )
        summary = {
            "smoothing_sigma_samples": sigma_samples,
            "smoothing_blend_factor": blend,
            "lookahead_distance_m": lookahead,
            "heading_gain": gain,
            "reset_yaw_mode": yaw_mode,
            "reset_yaw_rad": result.reset_yaw_rad,
            "execution_source_duration_ratio": float(
                result.plan.time_s[-1] / source.source_time_s[-1]
            ),
            "path_length_relative_drift": result.path_metrics[
                "path_length_relative_drift"
            ],
            "position_error_p95_m": result.kinematic_metrics[
                "position_error_p95_m"
            ],
            "position_error_max_m": result.kinematic_metrics[
                "position_error_max_m"
            ],
            "failed_checks": [
                key for key, value in result.checks.items() if not value
            ]
            + [
                key
                for key, value in result.kinematic_checks.items()
                if not value
            ],
            "passed": result.passed,
        }
        attempts.append(summary)
        result = replace(result, attempts=tuple(attempts))
        results.append(result)
        if result.passed:
            return result

    batch_seed = results[-1]
    if batch_unicycle_recovery_seed_eligible(
        batch_seed, float(source.source_time_s[-1])
    ):
        result = _build_batch_unicycle_recovery(source, kinematics, batch_seed)
        summary = {
            "smoothing_sigma_samples": result.smoothing_sigma_samples,
            "smoothing_blend_factor": result.smoothing_blend_factor,
            "lookahead_distance_m": result.lookahead_distance_m,
            "heading_gain": result.heading_gain,
            "reset_yaw_mode": result.reset_yaw_mode,
            "reset_yaw_rad": result.reset_yaw_rad,
            "planning_strategy": result.plan.planning_strategy,
            "execution_source_duration_ratio": float(
                result.plan.time_s[-1] / source.source_time_s[-1]
            ),
            "path_length_relative_drift": result.path_metrics[
                "path_length_relative_drift"
            ],
            "position_error_p95_m": result.kinematic_metrics[
                "position_error_p95_m"
            ],
            "position_error_max_m": result.kinematic_metrics[
                "position_error_max_m"
            ],
            "failed_checks": [
                key for key, value in result.checks.items() if not value
            ]
            + [
                key
                for key, value in result.kinematic_checks.items()
                if not value
            ],
            "passed": result.passed,
        }
        attempts.append(summary)
        result = replace(result, attempts=tuple(attempts))
        results.append(result)
        if result.passed:
            return result

    def rank(result: SmoothedPlanResult) -> tuple[float, ...]:
        failed = sum(not value for value in result.checks.values()) + sum(
            not value for value in result.kinematic_checks.values()
        )
        return (
            float(failed),
            max(
                float(result.plan.time_s[-1] / source.source_time_s[-1])
                - MAXIMUM_EXECUTION_SOURCE_DURATION_RATIO,
                0.0,
            ),
            result.kinematic_metrics["position_error_p95_m"],
            result.kinematic_metrics["position_error_max_m"],
            abs(result.path_metrics["path_length_relative_drift"]),
        )

    best = min(results, key=rank)
    return replace(best, attempts=tuple(attempts))


def save_smoothed_riser_plan(
    path: Path,
    result: SmoothedPlanResult,
    source: ExactSourceRiserReference,
) -> None:
    plan = result.plan
    plan.validate()
    _require(
        len(plan.time_s) == source.source_pose_count,
        "smoothed canary must retain one execution state per source anchor",
    )
    trajectory_integrity_passed = all(
        result.checks[key]
        for key in (
            "source_start_preserved",
            "source_final_preserved",
            "source_order_mapping_preserved",
            "source_motion_direction_preserved",
            "path_length_within_5_percent",
            "source_polyline_p95_bounded",
            "source_polyline_max_bounded",
        )
    )
    metadata = {
        "schema": SMOOTHED_PLAN_SCHEMA,
        "case": source.case,
        "source_provenance": {
            "trajectory_integrity_contract": EXACT_SOURCE_CONTRACT,
            "source_manifest_sha256": source.package_manifest_sha256,
            "source_json_sha256": source.source_json_sha256,
            "source_pose_count": source.source_pose_count,
            "source_duration_s": float(source.source_time_s[-1]),
            "source_path_length_m": result.path_metrics[
                "source_path_length_m"
            ],
            "raw_arrays_preserved_verbatim": True,
        },
        "smoothed_target": {
            "schema": SMOOTHED_TARGET_SCHEMA,
            "state_count": len(plan.time_s),
            "smoothing_sigma_samples": result.smoothing_sigma_samples,
            "smoothing_blend_factor": result.smoothing_blend_factor,
            "lookahead_distance_m": result.lookahead_distance_m,
            "heading_gain": result.heading_gain,
            "reset_yaw_mode": result.reset_yaw_mode,
            "source_initial_base_yaw_rad": source.initial_base_yaw_rad,
            "planned_reset_base_yaw_rad": result.reset_yaw_rad,
            "planning_strategy": plan.planning_strategy,
            "vertical_shift_m": plan.vertical_shift_m,
            "execution_duration_s": float(plan.time_s[-1]),
            "execution_source_duration_ratio": float(
                plan.time_s[-1] / source.source_time_s[-1]
            ),
            "source_anchor_map_identity": True,
            "densification_required": False,
        },
        "frame_contract": {
            "target_link": "ee1_tool",
            "semantic_dfr_quaternion_order": "xyzw",
            "semantic_forward_axis": "+Y",
            "physical_gimbal_is_diagnostic_only": True,
        },
        "batch_unicycle_recovery": {
            "applied": plan.planning_strategy == "smoothed_batch_unicycle_v1",
            "iterations": BATCH_RECOVERY_ITERATIONS,
            "residual_bound": BATCH_RECOVERY_RESIDUAL_BOUND,
            "regularization": BATCH_RECOVERY_REGULARIZATION,
            "learning_rate": BATCH_RECOVERY_LEARNING_RATE,
            "duration_weight": BATCH_RECOVERY_DURATION_WEIGHT,
            "duration_ratio_target": BATCH_RECOVERY_DURATION_RATIO_TARGET,
            "lateral_control_available": False,
        },
        "path_metrics": result.path_metrics,
        "transition_metrics": result.transition_metrics,
        "kinematic_metrics": result.kinematic_metrics,
        "checks": result.checks,
        "kinematic_checks": result.kinematic_checks,
        "attempts": list(result.attempts),
        "trajectory_integrity_passed": trajectory_integrity_passed,
        "timing_transition_kinematic_gate_passed": result.passed,
        "thermal_gate_passed": False,
        "dynamic_quality_passed": False,
        "valid_for_training": False,
        "residual_capture_started": False,
        "bc_started": False,
        "ppo_started": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
        time_s=plan.time_s,
        execution_time_s=plan.time_s,
        target_position_world_m=plan.target_position_world_m,
        smoothed_target_position_source_frame_m=(
            result.smoothed_position_source_frame_m
        ),
        target_semantic_dfr_quat_wxyz=plan.target_semantic_dfr_quat_wxyz,
        base_xy_yaw=plan.base_xy_yaw,
        riser_q=plan.riser_q,
        proxy_gimbal_q=plan.proxy_gimbal_q,
        feedforward_v_wz=plan.feedforward_v_wz,
        feedforward_riser_velocity=plan.feedforward_riser_velocity,
        feedforward_proxy_velocity=plan.feedforward_proxy_velocity,
        source_time_s=source.source_time_s,
        source_target_position_world_m=source.source_position_world_m,
        source_target_semantic_dfr_quat_xyzw=(
            source.source_semantic_dfr_quat_xyzw
        ),
        source_anchor_execution_index=np.arange(
            source.source_pose_count, dtype=np.int64
        ),
        initialization_time_s=np.empty(0, dtype=np.float64),
        initialization_state=np.empty((0, 7), dtype=np.float64),
    )


def load_smoothed_riser_plan(path: Path) -> tuple[RiserPlaybackPlan, dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        _require(metadata.get("schema") == SMOOTHED_PLAN_SCHEMA, "wrong smoothed schema")
        plan = RiserPlaybackPlan(
            case=int(metadata["case"]),
            time_s=np.asarray(data["execution_time_s"], dtype=np.float64),
            target_position_world_m=np.asarray(
                data["target_position_world_m"], dtype=np.float64
            ),
            target_semantic_dfr_quat_wxyz=np.asarray(
                data["target_semantic_dfr_quat_wxyz"], dtype=np.float64
            ),
            base_xy_yaw=np.asarray(data["base_xy_yaw"], dtype=np.float64),
            riser_q=np.asarray(data["riser_q"], dtype=np.float64),
            proxy_gimbal_q=np.asarray(data["proxy_gimbal_q"], dtype=np.float64),
            feedforward_v_wz=np.asarray(
                data["feedforward_v_wz"], dtype=np.float64
            ),
            feedforward_riser_velocity=np.asarray(
                data["feedforward_riser_velocity"], dtype=np.float64
            ),
            feedforward_proxy_velocity=np.asarray(
                data["feedforward_proxy_velocity"], dtype=np.float64
            ),
            vertical_shift_m=float(metadata["smoothed_target"]["vertical_shift_m"]),
            planning_strategy=str(metadata["smoothed_target"]["planning_strategy"]),
            source_time_s=np.asarray(data["source_time_s"], dtype=np.float64),
        )
    plan.validate()
    return plan, metadata


def audit_smoothed_riser_plan(
    path: Path,
    source: ExactSourceRiserReference,
    kinematics: UrdfRiserCameraKinematics,
) -> dict[str, object]:
    plan, metadata = load_smoothed_riser_plan(path)
    with np.load(path, allow_pickle=False) as data:
        source_time = np.asarray(data["source_time_s"], dtype=np.float64)
        source_position = np.asarray(
            data["source_target_position_world_m"], dtype=np.float64
        )
        source_attitude = np.asarray(
            data["source_target_semantic_dfr_quat_xyzw"], dtype=np.float64
        )
        smoothed = np.asarray(
            data["smoothed_target_position_source_frame_m"], dtype=np.float64
        )
        anchor = np.asarray(
            data["source_anchor_execution_index"], dtype=np.int64
        )
        initialization_time = np.asarray(
            data["initialization_time_s"], dtype=np.float64
        )
        initialization_state = np.asarray(
            data["initialization_state"], dtype=np.float64
        )
    provenance = metadata.get("source_provenance", {})
    path_metrics = smoothed_path_metrics(source_position, smoothed)
    transitions = transition_metrics(plan)
    kinematic_metrics = riser_playback_kinematic_metrics(plan, kinematics)
    kinematic_checks = riser_playback_kinematic_gate(
        kinematic_metrics, kinematics
    )
    duration_ratio = float(plan.time_s[-1] / source.source_time_s[-1])
    checks = {
        "source_manifest_hash_bound": provenance.get("source_manifest_sha256")
        == source.package_manifest_sha256,
        "source_json_hash_bound": provenance.get("source_json_sha256")
        == source.source_json_sha256,
        "source_time_verbatim": bool(np.array_equal(source_time, source.source_time_s)),
        "source_position_verbatim": bool(
            np.array_equal(source_position, source.source_position_world_m)
        ),
        "source_attitude_verbatim": bool(
            np.array_equal(source_attitude, source.source_semantic_dfr_quat_xyzw)
        ),
        "source_anchor_map_identity": bool(
            np.array_equal(anchor, np.arange(source.source_pose_count))
        ),
        "source_motion_direction_preserved": path_metrics[
            "opposed_segment_direction_count"
        ]
        == 0,
        "source_start_preserved": path_metrics["start_position_error_m"] <= 1e-12,
        "source_final_preserved": path_metrics["final_position_error_m"] <= 1e-12,
        "path_length_within_5_percent": abs(
            path_metrics["path_length_relative_drift"]
        )
        <= MAXIMUM_PATH_LENGTH_RELATIVE_DRIFT + 1e-12,
        "source_polyline_p95_bounded": path_metrics[
            "source_polyline_deviation_p95_m"
        ]
        <= PLAYBACK_POSITION_P95_LIMIT_M + 1e-12,
        "source_polyline_max_bounded": path_metrics[
            "source_polyline_deviation_max_m"
        ]
        <= PLAYBACK_POSITION_MAX_LIMIT_M + 1e-12,
        "execution_duration_ratio_bounded": 1.0 - 1e-12
        <= duration_ratio
        <= MAXIMUM_EXECUTION_SOURCE_DURATION_RATIO + 1e-12,
        "base_branch_step_bounded": transitions[
            "maximum_pre_densification_base_branch_step_rad"
        ]
        <= MAXIMUM_PRE_DENSIFICATION_BRANCH_STEP_RAD + 1e-12,
        "proxy_branch_step_bounded": transitions[
            "maximum_pre_densification_proxy_branch_step_rad"
        ]
        <= MAXIMUM_PRE_DENSIFICATION_BRANCH_STEP_RAD + 1e-12,
        "initialization_separate_empty": initialization_time.shape == (0,)
        and initialization_state.shape[0] == 0,
        "training_closed": metadata.get("valid_for_training") is False
        and metadata.get("residual_capture_started") is False
        and metadata.get("bc_started") is False
        and metadata.get("ppo_started") is False,
    }
    passed = all(checks.values()) and all(kinematic_checks.values())
    return {
        "case": source.case,
        "file": path.name,
        "plan_sha256": sha256_file(path),
        "source_json_sha256": source.source_json_sha256,
        "source_pose_count": source.source_pose_count,
        "execution_state_count": len(plan.time_s),
        "source_duration_s": float(source.source_time_s[-1]),
        "execution_duration_s": float(plan.time_s[-1]),
        "execution_source_duration_ratio": duration_ratio,
        "path_metrics": path_metrics,
        "transition_metrics": transitions,
        "kinematic_metrics": kinematic_metrics,
        "checks": checks,
        "kinematic_checks": kinematic_checks,
        "timing_transition_kinematic_gate_passed": passed,
        "valid_for_training": False,
        "passed": passed,
    }
