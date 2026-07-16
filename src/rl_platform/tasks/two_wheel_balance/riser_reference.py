"""Strict corrected-reference loading and kinematic planning for the riser robot."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

import numpy as np
from scipy.optimize import least_squares

from .camera_attitude import (
    quaternion_matrix_wxyz,
    rotation_error_vector,
    semantic_dfr_to_physical_cam_quat_wxyz,
)
from .riser_kinematics import UrdfRiserCameraKinematics
from .whole_body_kinematics import integrate_unicycle


REFERENCE_RE = re.compile(r"^episode_(?P<case>\d{4})_split_teacher_v1\.json$")
EXPECTED_SOURCE = "corrected_physical_split_teacher"
EXPECTED_ATTITUDE_CONTRACT = "semantic_dfr_to_physical_cam_v1"
EXPECTED_OBSERVATION_FRAME = "physical_cam_link_fk"


@dataclass(frozen=True)
class CorrectedRiserReference:
    case: int
    path: Path
    positions_m: np.ndarray
    semantic_dfr_quat_wxyz: np.ndarray
    time_s: np.ndarray
    initial_base_yaw_rad: float
    metadata: dict[str, object]


@dataclass(frozen=True)
class RiserKinematicPlan:
    time_s: np.ndarray
    targets_m: np.ndarray
    base_xy_yaw: np.ndarray
    riser_q: np.ndarray
    gimbal_q: np.ndarray
    achieved_m: np.ndarray
    attitude_error_rad: np.ndarray
    attitude_converged: np.ndarray
    vertical_shift_m: float
    planning_strategy: str = "unspecified"


@dataclass(frozen=True)
class BoundedAttitudeAllocation:
    base_yaw_rad: np.ndarray
    gimbal_q: np.ndarray
    orientation_error_rad: np.ndarray
    converged: np.ndarray


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _normalize_quaternions_wxyz(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    _require(values.ndim == 2 and values.shape[1] == 4, "bad quaternion shape")
    _require(np.isfinite(values).all(), "non-finite quaternion")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    _require(bool(np.all(norms > 1e-12)), "zero-length quaternion")
    values = values / norms
    return values * np.where(values[:, :1] < 0.0, -1.0, 1.0)


def load_corrected_riser_reference(path: Path) -> CorrectedRiserReference:
    path = path.resolve()
    match = REFERENCE_RE.match(path.name)
    _require(match is not None, f"unexpected corrected reference name: {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata")
    poses = payload.get("poses")
    _require(isinstance(metadata, dict), f"missing metadata in {path}")
    _require(isinstance(poses, list) and len(poses) >= 2, f"bad poses in {path}")
    _require(metadata.get("source") == EXPECTED_SOURCE, f"wrong source in {path}")
    _require(metadata.get("quality_status") == "accepted", f"unaccepted case in {path}")
    _require(
        metadata.get("target_orientation_contract") == EXPECTED_ATTITUDE_CONTRACT,
        f"wrong attitude contract in {path}",
    )
    _require(
        metadata.get("observation_ee_frame") == EXPECTED_OBSERVATION_FRAME,
        f"wrong observation frame in {path}",
    )
    _require(metadata.get("recorded_quaternion_order") == "xyzw", f"wrong quaternion order in {path}")

    case = int(match.group("case"))
    _require(int(metadata.get("episode_index", -1)) == case, f"case mismatch in {path}")
    positions = np.asarray([item["position"] for item in poses], dtype=np.float64)
    xyzw = np.asarray([item["orientation"] for item in poses], dtype=np.float64)
    _require(positions.shape == (len(poses), 3), f"bad positions in {path}")
    _require(np.isfinite(positions).all(), f"non-finite positions in {path}")
    attitudes = _normalize_quaternions_wxyz(xyzw[:, [3, 0, 1, 2]])

    duration = float(metadata.get("duration_s", 0.0))
    waypoint_dt = float(metadata.get("waypoint_dt", 0.0))
    _require(duration > 0.0 and waypoint_dt > 0.0, f"bad timing in {path}")
    time_s = np.linspace(0.0, duration, len(poses), dtype=np.float64)
    _require(
        abs(float(np.median(np.diff(time_s))) - waypoint_dt) <= waypoint_dt * 0.02,
        f"duration/count mismatch in {path}",
    )
    initial_base = np.asarray(metadata.get("initial_base_pose_xyyaw"), dtype=np.float64)
    _require(initial_base.shape == (3,) and np.isfinite(initial_base).all(), f"bad initial base in {path}")
    return CorrectedRiserReference(
        case=case,
        path=path,
        positions_m=positions,
        semantic_dfr_quat_wxyz=attitudes,
        time_s=time_s,
        initial_base_yaw_rad=float(initial_base[2]),
        metadata=metadata,
    )


def discover_corrected_riser_stage(
    stage_dir: Path, *, expected_count: int = 62
) -> dict[int, CorrectedRiserReference]:
    references: dict[int, CorrectedRiserReference] = {}
    for path in sorted(stage_dir.resolve().glob("episode_*_split_teacher_v1.json")):
        reference = load_corrected_riser_reference(path)
        _require(reference.case not in references, f"duplicate case {reference.case}")
        references[reference.case] = reference
    _require(len(references) == expected_count, f"expected {expected_count} corrected cases, got {len(references)}")
    return references


def bidirectional_path_heading(xy: np.ndarray, initial_yaw_rad: float) -> np.ndarray:
    """Choose segment headings continuously while allowing signed forward speed."""

    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2 or not np.isfinite(xy).all():
        raise ValueError("xy path must be finite shape (N,2), N>=2")
    if not math.isfinite(initial_yaw_rad):
        raise ValueError("initial yaw must be finite")
    result = np.empty(len(xy), dtype=np.float64)
    previous = float(initial_yaw_rad)
    for index, delta in enumerate(np.diff(xy, axis=0)):
        if float(np.linalg.norm(delta)) > 1e-7:
            raw = math.atan2(float(delta[1]), float(delta[0]))
            candidates = np.array([raw + k * math.pi for k in range(-3, 4)])
            previous = float(candidates[np.argmin(np.abs(candidates - previous))])
        result[index] = previous
    result[-1] = previous
    return np.unwrap(result)


def yaw_quaternion_wxyz(yaw_rad: float) -> np.ndarray:
    return np.array([math.cos(yaw_rad / 2.0), 0.0, 0.0, math.sin(yaw_rad / 2.0)])


def select_initial_or_reverse_yaw(
    reference: CorrectedRiserReference,
    kinematics: UrdfRiserCameraKinematics,
) -> tuple[float, np.ndarray]:
    """Choose forward or reverse chassis heading by gimbal-limit margin."""

    candidates = []
    for yaw in (
        reference.initial_base_yaw_rad,
        reference.initial_base_yaw_rad + math.pi,
    ):
        attitude = kinematics.solve_semantic_attitude_robust(
            yaw_quaternion_wxyz(yaw),
            0.0,
            reference.semantic_dfr_quat_wxyz[0],
            np.zeros(3),
        )
        candidates.append(
            (
                attitude.converged,
                kinematics.normalized_gimbal_limit_margin(attitude.gimbal_q),
                -attitude.orientation_error_rad,
                yaw,
                attitude.gimbal_q,
            )
        )
    converged = [item for item in candidates if item[0]]
    selected = max(converged or candidates, key=lambda item: (item[1], item[2]))
    return float(selected[3]), selected[4].copy()


def select_path_aligned_initial_yaw(
    reference: CorrectedRiserReference,
    kinematics: UrdfRiserCameraKinematics,
    *,
    minimum_normalized_margin: float = 0.02,
) -> tuple[float, np.ndarray]:
    """Find the closest attitude-feasible heading to the initial path tangent."""

    displacement = reference.positions_m[:, :2] - reference.positions_m[0, :2]
    indices = np.flatnonzero(np.linalg.norm(displacement, axis=1) >= 0.01)
    if len(indices):
        delta = displacement[int(indices[0])]
        desired = math.atan2(float(delta[1]), float(delta[0]))
        while desired - reference.initial_base_yaw_rad > math.pi / 2.0:
            desired -= math.pi
        while desired - reference.initial_base_yaw_rad < -math.pi / 2.0:
            desired += math.pi
    else:
        desired = reference.initial_base_yaw_rad

    offsets = [0.0]
    for magnitude in np.arange(0.2, math.pi / 2.0 + 0.01, 0.2):
        offsets.extend((-float(magnitude), float(magnitude)))
    candidates = []
    for offset in offsets:
        yaw = desired + offset
        attitude = kinematics.solve_semantic_attitude_robust(
            yaw_quaternion_wxyz(yaw),
            0.0,
            reference.semantic_dfr_quat_wxyz[0],
            np.zeros(3),
        )
        margin = kinematics.normalized_gimbal_limit_margin(attitude.gimbal_q)
        candidates.append((attitude.converged, margin, abs(offset), yaw, attitude))
        if attitude.converged and margin >= minimum_normalized_margin:
            return yaw, attitude.gimbal_q.copy()

    fallback_yaw, fallback_gimbal = select_initial_or_reverse_yaw(reference, kinematics)
    return fallback_yaw, fallback_gimbal


def allocate_bounded_attitude_trajectory(
    reference: CorrectedRiserReference,
    kinematics: UrdfRiserCameraKinematics,
    *,
    maximum_base_yaw_rate_radps: float = 0.4,
    maximum_gimbal_rate_radps: float = 0.5,
    tolerance_rad: float = math.radians(0.1),
    orientation_scale_rad: float = 0.01,
    heading_weight: float = 0.05,
) -> BoundedAttitudeAllocation:
    """Use base yaw redundancy to keep the physical gimbal continuous."""

    if (
        maximum_base_yaw_rate_radps <= 0.0
        or maximum_gimbal_rate_radps <= 0.0
        or orientation_scale_rad <= 0.0
        or heading_weight < 0.0
    ):
        raise ValueError("attitude allocation rates must be positive")
    yaw0, gimbal0 = select_path_aligned_initial_yaw(reference, kinematics)
    path_heading = bidirectional_path_heading(
        reference.positions_m[:, :2], yaw0
    )
    count = len(reference.time_s)
    yaw = np.empty(count, dtype=np.float64)
    gimbal = np.empty((count, 3), dtype=np.float64)
    error = np.empty(count, dtype=np.float64)
    converged = np.empty(count, dtype=bool)
    yaw[0], gimbal[0] = yaw0, gimbal0

    targets = [
        quaternion_matrix_wxyz(semantic_dfr_to_physical_cam_quat_wxyz(item))
        for item in reference.semantic_dfr_quat_wxyz
    ]
    current = kinematics.world_rotation(yaw_quaternion_wxyz(yaw0), 0.0, gimbal0)
    error[0] = float(np.linalg.norm(rotation_error_vector(current, targets[0])))
    converged[0] = error[0] <= tolerance_rad

    for index in range(1, count):
        dt = float(reference.time_s[index] - reference.time_s[index - 1])
        previous = np.r_[yaw[index - 1], gimbal[index - 1]]
        lower = np.r_[
            yaw[index - 1] - maximum_base_yaw_rate_radps * dt,
            np.maximum(
                kinematics.gimbal_lower,
                gimbal[index - 1] - maximum_gimbal_rate_radps * dt,
            ),
        ]
        upper = np.r_[
            yaw[index - 1] + maximum_base_yaw_rate_radps * dt,
            np.minimum(
                kinematics.gimbal_upper,
                gimbal[index - 1] + maximum_gimbal_rate_radps * dt,
            ),
        ]
        scale = np.maximum(upper - lower, 1e-9)

        def residual(candidate: np.ndarray) -> np.ndarray:
            physical = kinematics.world_rotation(
                yaw_quaternion_wxyz(float(candidate[0])), 0.0, candidate[1:]
            )
            orientation = (
                rotation_error_vector(physical, targets[index])
                / orientation_scale_rad
            )
            continuity = 0.002 * (candidate - previous) / scale
            heading_error = (
                candidate[0] - path_heading[index] + math.pi / 2.0
            ) % math.pi - math.pi / 2.0
            return np.r_[orientation, continuity, heading_weight * heading_error]

        solution = least_squares(
            residual,
            previous,
            bounds=(lower, upper),
            max_nfev=30,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
        )
        yaw[index] = solution.x[0]
        gimbal[index] = solution.x[1:]
        physical = kinematics.world_rotation(
            yaw_quaternion_wxyz(yaw[index]), 0.0, gimbal[index]
        )
        error[index] = float(
            np.linalg.norm(rotation_error_vector(physical, targets[index]))
        )
        converged[index] = error[index] <= tolerance_rad
    return BoundedAttitudeAllocation(yaw, gimbal, error, converged)


def retarget_bounded_unicycle_pose(
    reference: CorrectedRiserReference,
    kinematics: UrdfRiserCameraKinematics,
    *,
    vertical_shift_m: float = 0.0,
    maximum_linear_velocity_mps: float = 0.4,
    maximum_base_yaw_rate_radps: float = 0.4,
    maximum_riser_rate_mps: float = 1.0,
    maximum_gimbal_rate_radps: float = 0.5,
    position_scale_m: float = 0.03,
    orientation_scale_rad: float = 0.03,
    attitude_tolerance_rad: float = math.radians(0.1),
) -> RiserKinematicPlan:
    """Jointly retarget physical camera pose through bounded unicycle motion."""

    values = (
        maximum_linear_velocity_mps,
        maximum_base_yaw_rate_radps,
        maximum_riser_rate_mps,
        maximum_gimbal_rate_radps,
        position_scale_m,
        orientation_scale_rad,
        attitude_tolerance_rad,
    )
    if any(value <= 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("retarget limits and scales must be finite and positive")
    targets = reference.positions_m.copy()
    targets[:, 2] += vertical_shift_m
    count = len(targets)
    base = np.empty((count, 3), dtype=np.float64)
    riser = np.empty(count, dtype=np.float64)
    gimbal = np.empty((count, 3), dtype=np.float64)
    achieved = np.empty_like(targets)
    attitude_error = np.empty(count, dtype=np.float64)
    attitude_converged = np.empty(count, dtype=bool)
    target_rotation = [
        quaternion_matrix_wxyz(semantic_dfr_to_physical_cam_quat_wxyz(item))
        for item in reference.semantic_dfr_quat_wxyz
    ]

    yaw0, gimbal0 = select_path_aligned_initial_yaw(reference, kinematics)
    initial = kinematics.solve_position(targets[0], yaw0, gimbal0)
    base[0] = initial.base_xy_yaw_riser[:3]
    riser[0] = initial.base_xy_yaw_riser[3]
    gimbal[0] = gimbal0
    initial_transform = kinematics.world_transform(base[0], riser[0], gimbal[0])
    achieved[0] = initial_transform[:3, 3]
    initial_rotation = initial_transform[:3, :3]
    attitude_error[0] = float(
        np.linalg.norm(rotation_error_vector(initial_rotation, target_rotation[0]))
    )
    attitude_converged[0] = attitude_error[0] <= attitude_tolerance_rad

    previous_control = np.zeros(6, dtype=np.float64)
    for index in range(1, count):
        dt = float(reference.time_s[index] - reference.time_s[index - 1])
        riser_delta_limit = maximum_riser_rate_mps * dt
        gimbal_delta_limit = maximum_gimbal_rate_radps * dt
        lower = np.r_[
            -maximum_linear_velocity_mps,
            -maximum_base_yaw_rate_radps,
            max(-riser_delta_limit, kinematics.riser_lower - riser[index - 1]),
            np.maximum(
                -gimbal_delta_limit,
                kinematics.gimbal_lower - gimbal[index - 1],
            ),
        ]
        upper = np.r_[
            maximum_linear_velocity_mps,
            maximum_base_yaw_rate_radps,
            min(riser_delta_limit, kinematics.riser_upper - riser[index - 1]),
            np.minimum(
                gimbal_delta_limit,
                kinematics.gimbal_upper - gimbal[index - 1],
            ),
        ]
        initial_control = np.clip(previous_control, lower, upper)

        def candidate(control: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
            next_base = integrate_unicycle(
                base[index - 1], float(control[0]), float(control[1]), dt
            )
            next_riser = float(riser[index - 1] + control[2])
            next_gimbal = gimbal[index - 1] + control[3:]
            pose = kinematics.world_transform(next_base, next_riser, next_gimbal)
            return next_base, next_riser, next_gimbal, pose

        def residual(control: np.ndarray) -> np.ndarray:
            _, _, _, pose = candidate(control)
            position = (pose[:3, 3] - targets[index]) / position_scale_m
            orientation = (
                rotation_error_vector(pose[:3, :3], target_rotation[index])
                / orientation_scale_rad
            )
            normalized = np.r_[
                control[0] / maximum_linear_velocity_mps,
                control[1] / maximum_base_yaw_rate_radps,
                control[2] / max(riser_delta_limit, 1e-9),
                control[3:] / max(gimbal_delta_limit, 1e-9),
            ]
            regularization = 0.002 * normalized
            return np.r_[position, orientation, regularization]

        solution = least_squares(
            residual,
            initial_control,
            bounds=(lower, upper),
            max_nfev=60,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
        )
        previous_control = solution.x
        base[index], riser[index], gimbal[index], pose = candidate(solution.x)
        achieved[index] = pose[:3, 3]
        attitude_error[index] = float(
            np.linalg.norm(rotation_error_vector(pose[:3, :3], target_rotation[index]))
        )
        attitude_converged[index] = attitude_error[index] <= attitude_tolerance_rad

    return RiserKinematicPlan(
        time_s=reference.time_s.copy(),
        targets_m=targets,
        base_xy_yaw=base,
        riser_q=riser,
        gimbal_q=gimbal,
        achieved_m=achieved,
        attitude_error_rad=attitude_error,
        attitude_converged=attitude_converged,
        vertical_shift_m=vertical_shift_m,
    )


def plan_corrected_riser_reference(
    reference: CorrectedRiserReference,
    kinematics: UrdfRiserCameraKinematics,
    *,
    vertical_shift_m: float = 0.0,
    heading_iterations: int = 2,
    heading_mode: str = "bounded_attitude_allocation",
    orientation_scale_rad: float = 0.01,
    heading_weight: float = 0.05,
    maximum_base_yaw_rate_radps: float = 0.4,
    maximum_gimbal_rate_radps: float = 0.5,
    maximum_linear_velocity_mps: float = 0.4,
    maximum_riser_rate_mps: float = 1.0,
    position_scale_m: float = 0.03,
    attitude_tolerance_rad: float = math.radians(0.1),
) -> RiserKinematicPlan:
    """Solve an ideal full-pose plan before applying dynamic control limits."""

    if vertical_shift_m < 0.0 or not math.isfinite(vertical_shift_m):
        raise ValueError("vertical shift must be finite and non-negative")
    if heading_iterations < 1:
        raise ValueError("heading_iterations must be positive")
    if attitude_tolerance_rad <= 0.0 or not math.isfinite(attitude_tolerance_rad):
        raise ValueError("attitude tolerance must be finite and positive")
    if heading_mode not in {
        "initial_constant",
        "initial_or_reverse_constant",
        "bidirectional_fixed_point",
        "bounded_attitude_allocation",
        "bounded_unicycle_pose",
    }:
        raise ValueError(f"unsupported heading mode {heading_mode!r}")
    if heading_mode == "bounded_unicycle_pose":
        return retarget_bounded_unicycle_pose(
            reference,
            kinematics,
            vertical_shift_m=vertical_shift_m,
            maximum_linear_velocity_mps=maximum_linear_velocity_mps,
            maximum_base_yaw_rate_radps=maximum_base_yaw_rate_radps,
            maximum_riser_rate_mps=maximum_riser_rate_mps,
            maximum_gimbal_rate_radps=maximum_gimbal_rate_radps,
            position_scale_m=position_scale_m,
            orientation_scale_rad=orientation_scale_rad,
            attitude_tolerance_rad=attitude_tolerance_rad,
        )
    targets = reference.positions_m.copy()
    targets[:, 2] += vertical_shift_m
    initial_gimbal_seed = np.zeros(3, dtype=np.float64)
    allocated_attitude = None
    if heading_mode == "bounded_attitude_allocation":
        allocated_attitude = allocate_bounded_attitude_trajectory(
            reference,
            kinematics,
            maximum_base_yaw_rate_radps=maximum_base_yaw_rate_radps,
            maximum_gimbal_rate_radps=maximum_gimbal_rate_radps,
            orientation_scale_rad=orientation_scale_rad,
            heading_weight=heading_weight,
            tolerance_rad=attitude_tolerance_rad,
        )
        yaw = allocated_attitude.base_yaw_rad
    elif heading_mode == "initial_or_reverse_constant":
        selected_yaw, initial_gimbal_seed = select_initial_or_reverse_yaw(
            reference, kinematics
        )
        yaw = np.full(len(targets), selected_yaw, dtype=np.float64)
    elif heading_mode == "initial_constant":
        yaw = np.full(len(targets), reference.initial_base_yaw_rad, dtype=np.float64)
    else:
        yaw = bidirectional_path_heading(targets[:, :2], reference.initial_base_yaw_rad)

    count = len(targets)
    base = np.empty((count, 3), dtype=np.float64)
    riser = np.empty(count, dtype=np.float64)
    gimbal = np.empty((count, 3), dtype=np.float64)
    achieved = np.empty_like(targets)
    attitude_error = np.empty(count, dtype=np.float64)
    attitude_converged = np.empty(count, dtype=bool)

    if allocated_attitude is not None:
        gimbal[:] = allocated_attitude.gimbal_q
        attitude_error[:] = allocated_attitude.orientation_error_rad
        attitude_converged[:] = allocated_attitude.converged
        for index in range(count):
            position = kinematics.solve_position(
                targets[index], float(yaw[index]), gimbal[index]
            )
            base[index] = position.base_xy_yaw_riser[:3]
            riser[index] = position.base_xy_yaw_riser[3]
            achieved[index] = kinematics.world_transform(
                base[index], riser[index], gimbal[index]
            )[:3, 3]
    else:
        for iteration in range(heading_iterations):
            seed = initial_gimbal_seed.copy()
            for index in range(count):
                attitude = kinematics.solve_semantic_attitude_robust(
                    yaw_quaternion_wxyz(float(yaw[index])),
                    0.0,
                    reference.semantic_dfr_quat_wxyz[index],
                    seed,
                    tolerance_rad=attitude_tolerance_rad,
                )
                gimbal[index] = attitude.gimbal_q
                attitude_error[index] = attitude.orientation_error_rad
                attitude_converged[index] = attitude.converged
                seed = attitude.gimbal_q
                position = kinematics.solve_position(
                    targets[index], float(yaw[index]), attitude.gimbal_q
                )
                base[index] = position.base_xy_yaw_riser[:3]
                riser[index] = position.base_xy_yaw_riser[3]
                achieved[index] = kinematics.world_transform(
                    base[index], riser[index], gimbal[index]
                )[:3, 3]
            if heading_mode == "bidirectional_fixed_point" and iteration + 1 < heading_iterations:
                yaw = bidirectional_path_heading(base[:, :2], float(yaw[0]))
            elif heading_mode in {"initial_constant", "initial_or_reverse_constant"}:
                break

    return RiserKinematicPlan(
        time_s=reference.time_s.copy(),
        targets_m=targets,
        base_xy_yaw=base,
        riser_q=riser,
        gimbal_q=gimbal,
        achieved_m=achieved,
        attitude_error_rad=attitude_error,
        attitude_converged=attitude_converged,
        vertical_shift_m=vertical_shift_m,
    )


def plan_rate_metrics(plan: RiserKinematicPlan) -> dict[str, float]:
    dt = np.diff(plan.time_s)
    delta_xy = np.diff(plan.base_xy_yaw[:, :2], axis=0)
    unwrapped_yaw = np.unwrap(plan.base_xy_yaw[:, 2])
    # A constant-twist unicycle follows an arc during each sample. Projecting
    # chord displacement at the interval midpoint avoids reporting that arc as
    # non-holonomic lateral velocity.
    yaw = 0.5 * (unwrapped_yaw[:-1] + unwrapped_yaw[1:])
    c, s = np.cos(yaw), np.sin(yaw)
    vx = (c * delta_xy[:, 0] + s * delta_xy[:, 1]) / dt
    vy = (-s * delta_xy[:, 0] + c * delta_xy[:, 1]) / dt
    wz = np.diff(unwrapped_yaw) / dt
    riser_rate = np.diff(plan.riser_q) / dt
    gimbal_rate = np.diff(plan.gimbal_q, axis=0) / dt[:, None]
    position_error = np.linalg.norm(plan.achieved_m - plan.targets_m, axis=1)
    return {
        "position_error_max_m": float(np.max(position_error)),
        "position_error_p95_m": float(np.percentile(position_error, 95)),
        "attitude_error_max_deg": math.degrees(float(np.max(plan.attitude_error_rad))),
        "attitude_error_p95_deg": math.degrees(
            float(np.percentile(plan.attitude_error_rad, 95))
        ),
        "attitude_ik_converged_ratio": float(np.mean(plan.attitude_converged)),
        "maximum_abs_base_linear_velocity_mps": float(np.max(np.abs(vx))),
        "maximum_abs_base_lateral_velocity_mps": float(np.max(np.abs(vy))),
        "maximum_abs_base_yaw_rate_radps": float(np.max(np.abs(wz))),
        "maximum_abs_riser_rate_mps": float(np.max(np.abs(riser_rate))),
        "maximum_abs_gimbal_rate_radps": float(np.max(np.abs(gimbal_rate))),
        "minimum_riser_position_m": float(np.min(plan.riser_q)),
        "maximum_riser_position_m": float(np.max(plan.riser_q)),
    }
