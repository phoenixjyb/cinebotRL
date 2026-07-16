"""Exact-source trajectory loading and playback artifact integrity checks."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .riser_playback import PLAYBACK_SCHEMA, RiserPlaybackPlan
from .riser_reference import CorrectedRiserReference, bidirectional_path_heading


EXACT_SOURCE_CONTRACT = "exact_source_v1"
EXACT_SOURCE_PACKAGE_SCHEMA = "gik_exact_source_reference_package_v1"
MINIMUM_CAMERA_HEIGHT_M = 0.60
MAXIMUM_CAMERA_HEIGHT_M = 1.80


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _path_length(position_m: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(position_m, axis=0), axis=1)))


def _normalize_wxyz(source_xyzw: np.ndarray) -> np.ndarray:
    wxyz = np.asarray(source_xyzw, dtype=np.float64)[:, [3, 0, 1, 2]]
    norm = np.linalg.norm(wxyz, axis=1, keepdims=True)
    _require(bool(np.all(norm > 1e-12)), "source contains zero-length quaternion")
    return wxyz / norm


def _initial_yaw(position_m: np.ndarray, preferred_vx_mps: float) -> float:
    displacement = position_m[:, :2] - position_m[0, :2]
    nontrivial = np.flatnonzero(np.linalg.norm(displacement, axis=1) >= 0.01)
    if not len(nontrivial):
        return 0.0
    delta = displacement[int(nontrivial[0])]
    tangent = math.atan2(float(delta[1]), float(delta[0]))
    return tangent + (math.pi if preferred_vx_mps < 0.0 else 0.0)


@dataclass(frozen=True)
class ExactSourceRiserReference:
    case: int
    package_manifest_path: Path
    package_manifest_sha256: str
    source_json_path: Path
    source_json_sha256: str
    source_time_s: np.ndarray
    source_position_world_m: np.ndarray
    source_semantic_dfr_quat_xyzw: np.ndarray
    initial_base_yaw_rad: float
    package_item: dict[str, object]

    @property
    def source_pose_count(self) -> int:
        return len(self.source_time_s)

    def planning_reference(self, execution_time_s: np.ndarray) -> CorrectedRiserReference:
        execution_time_s = np.asarray(execution_time_s, dtype=np.float64)
        _require(
            execution_time_s.shape == self.source_time_s.shape,
            "execution schedule must have one state per source anchor",
        )
        _require(
            execution_time_s[0] == 0.0
            and bool(np.all(np.diff(execution_time_s) > 0.0)),
            "execution schedule must start at zero and increase strictly",
        )
        return CorrectedRiserReference(
            case=self.case,
            path=self.source_json_path,
            positions_m=self.source_position_world_m.copy(),
            semantic_dfr_quat_wxyz=_normalize_wxyz(
                self.source_semantic_dfr_quat_xyzw
            ),
            time_s=execution_time_s.copy(),
            initial_base_yaw_rad=self.initial_base_yaw_rad,
            metadata={
                "source": "exact_source_v1_reference_only",
                "source_manifest_sha256": self.package_manifest_sha256,
                "source_json_sha256": self.source_json_sha256,
                "quality_status": "reference_only",
                "target_orientation_contract": "semantic_dfr_to_physical_cam_v1",
                "observation_ee_frame": "physical_cam_link_fk",
                "recorded_quaternion_order": "xyzw",
            },
        )


def execution_schedule_for_source(
    reference: ExactSourceRiserReference,
    *,
    minimum_dt_s: float = 0.005,
    planning_horizontal_speed_mps: float = 0.30,
    planning_vertical_speed_mps: float = 0.80,
    planning_attitude_rate_rad_s: float = math.radians(20.0),
    planning_heading_rate_rad_s: float = 0.20,
) -> np.ndarray:
    """Retain N anchors while making actuator-aware execution timing explicit."""

    limits = (
        minimum_dt_s,
        planning_horizontal_speed_mps,
        planning_vertical_speed_mps,
        planning_attitude_rate_rad_s,
        planning_heading_rate_rad_s,
    )
    _require(all(math.isfinite(item) and item > 0.0 for item in limits), "bad retiming limits")
    source_dt = np.diff(reference.source_time_s)
    delta_position = np.diff(reference.source_position_world_m, axis=0)
    horizontal_dt = np.linalg.norm(delta_position[:, :2], axis=1) / planning_horizontal_speed_mps
    vertical_dt = np.abs(delta_position[:, 2]) / planning_vertical_speed_mps
    quaternion = _normalize_wxyz(reference.source_semantic_dfr_quat_xyzw)
    dots = np.abs(np.sum(quaternion[:-1] * quaternion[1:], axis=1))
    attitude_angle = 2.0 * np.arccos(np.clip(dots, -1.0, 1.0))
    attitude_dt = attitude_angle / planning_attitude_rate_rad_s
    path_heading = bidirectional_path_heading(
        reference.source_position_world_m[:, :2], reference.initial_base_yaw_rad
    )
    heading_dt = np.abs(np.diff(path_heading)) / planning_heading_rate_rad_s
    execution_dt = np.maximum.reduce(
        (
            source_dt,
            np.full_like(source_dt, minimum_dt_s),
            horizontal_dt,
            vertical_dt,
            attitude_dt,
            heading_dt,
        )
    )
    return np.r_[0.0, np.cumsum(execution_dt)]


def camera_envelope_vertical_shift(
    source_position_world_m: np.ndarray,
    *,
    minimum_camera_height_m: float = MINIMUM_CAMERA_HEIGHT_M,
    maximum_camera_height_m: float = MAXIMUM_CAMERA_HEIGHT_M,
) -> tuple[float, bool]:
    """Choose the smallest constant shift that fits the camera-Z envelope."""

    position = np.asarray(source_position_world_m, dtype=np.float64)
    _require(
        position.ndim == 2
        and position.shape[1] == 3
        and len(position) >= 2
        and np.isfinite(position).all(),
        "source positions must be finite shape (N,3)",
    )
    _require(
        math.isfinite(minimum_camera_height_m)
        and math.isfinite(maximum_camera_height_m)
        and minimum_camera_height_m < maximum_camera_height_m,
        "invalid camera height envelope",
    )
    lower_shift = minimum_camera_height_m - float(np.min(position[:, 2]))
    upper_shift = maximum_camera_height_m - float(np.max(position[:, 2]))
    compatible = lower_shift <= upper_shift + 1e-12
    if compatible:
        shift = float(np.clip(0.0, lower_shift, upper_shift))
    else:
        # Preserve the lower safety boundary and let the explicit upper gate reject.
        shift = lower_shift
    return shift, compatible


def refine_execution_schedule_for_plan(
    source: ExactSourceRiserReference,
    plan: RiserPlaybackPlan,
    *,
    maximum_base_linear_velocity_mps: float = 0.30,
    maximum_base_yaw_rate_rad_s: float = 0.20,
    maximum_riser_rate_mps: float = 0.80,
    maximum_proxy_rate_rad_s: float = math.radians(12.0),
) -> np.ndarray:
    """Stretch intervals from selected-plan demand without changing anchors."""

    plan.validate()
    _require(
        len(plan.time_s) == source.source_pose_count,
        "selected plan must retain one state per source anchor",
    )
    limits = (
        maximum_base_linear_velocity_mps,
        maximum_base_yaw_rate_rad_s,
        maximum_riser_rate_mps,
        maximum_proxy_rate_rad_s,
    )
    _require(all(math.isfinite(item) and item > 0.0 for item in limits), "bad plan limits")
    delta_xy = np.diff(plan.base_xy_yaw[:, :2], axis=0)
    yaw = np.unwrap(plan.base_xy_yaw[:, 2])
    midpoint = 0.5 * (yaw[:-1] + yaw[1:])
    forward_distance = np.abs(
        np.cos(midpoint) * delta_xy[:, 0]
        + np.sin(midpoint) * delta_xy[:, 1]
    )
    proxy_delta = np.abs(np.diff(plan.proxy_gimbal_q, axis=0))
    execution_dt = np.maximum.reduce(
        (
            np.diff(plan.time_s),
            forward_distance / maximum_base_linear_velocity_mps,
            np.abs(np.diff(yaw)) / maximum_base_yaw_rate_rad_s,
            np.abs(np.diff(plan.riser_q)) / maximum_riser_rate_mps,
            np.max(proxy_delta, axis=1) / maximum_proxy_rate_rad_s,
        )
    )
    return np.r_[0.0, np.cumsum(execution_dt)]


def load_exact_source_package(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    expected_count: int = 79,
) -> dict[int, ExactSourceRiserReference]:
    manifest_path = manifest_path.resolve()
    actual_manifest_sha256 = sha256_file(manifest_path)
    _require(
        actual_manifest_sha256 == expected_manifest_sha256,
        "exact-source package manifest hash mismatch",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("schema") == EXACT_SOURCE_PACKAGE_SCHEMA, "wrong package schema")
    _require(
        manifest.get("trajectory_integrity_contract") == EXACT_SOURCE_CONTRACT,
        "wrong trajectory integrity contract",
    )
    _require(manifest.get("integrity_passed") is True, "package integrity not passed")
    _require(
        manifest.get("quality_qualified_teacher") is False
        and manifest.get("valid_for_training") is False,
        "reference ingest requires an explicitly non-training package",
    )
    frame = manifest.get("frame_contract", {})
    _require(frame.get("pose_target_link") == "ee1_tool", "wrong target link")
    _require(frame.get("semantic_forward_axis") == "+y in ee1_tool", "wrong forward axis")
    items = manifest.get("items")
    _require(isinstance(items, list) and len(items) == expected_count, "wrong item count")

    references: dict[int, ExactSourceRiserReference] = {}
    for item in items:
        case = int(item.get("episode_index", -1))
        _require(1 <= case <= expected_count and case not in references, "bad or duplicate case")
        _require(item.get("trajectory_integrity_contract") == EXACT_SOURCE_CONTRACT, "bad item contract")
        _require(item.get("integrity_passed") is True, "item integrity not passed")
        _require(
            item.get("quality_qualified_teacher") is False
            and item.get("valid_for_training") is False,
            "item must remain reference-only",
        )
        source_path = manifest_path.parent / str(item.get("bundled_source_json", ""))
        _require(source_path.is_file(), f"missing source JSON for case {case}")
        source_sha256 = sha256_file(source_path)
        _require(source_sha256 == item.get("source_json_sha256"), f"source hash mismatch for case {case}")
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        poses = payload.get("poses")
        _require(isinstance(poses, list) and len(poses) >= 2, f"bad poses for case {case}")
        position = np.asarray([pose["position"] for pose in poses], dtype=np.float64)
        quaternion_xyzw = np.asarray([pose["orientation"] for pose in poses], dtype=np.float64)
        time_s = np.asarray([pose["time"] for pose in poses], dtype=np.float64)
        count = len(poses)
        _require(position.shape == (count, 3), f"bad position shape for case {case}")
        _require(quaternion_xyzw.shape == (count, 4), f"bad attitude shape for case {case}")
        _require(
            np.isfinite(position).all()
            and np.isfinite(quaternion_xyzw).all()
            and np.isfinite(time_s).all(),
            f"non-finite source value for case {case}",
        )
        _require(
            time_s[0] == 0.0 and bool(np.all(np.diff(time_s) > 0.0)),
            f"bad source timestamps for case {case}",
        )
        _require(count == item.get("source_pose_count"), f"pose count mismatch for case {case}")
        _require(
            abs(float(time_s[-1]) - float(item.get("source_duration_s", -1.0))) <= 1e-9,
            f"duration mismatch for case {case}",
        )
        _require(
            abs(_path_length(position) - float(item.get("source_path_length_m", -1.0))) <= 1e-9,
            f"path length mismatch for case {case}",
        )
        normalized = np.linalg.norm(quaternion_xyzw, axis=1)
        _require(bool(np.allclose(normalized, 1.0, atol=1e-10)), f"bad quaternion norm for case {case}")
        preferred_vx = float(payload.get("first_point_vx_prefer", 0.4))
        references[case] = ExactSourceRiserReference(
            case=case,
            package_manifest_path=manifest_path,
            package_manifest_sha256=actual_manifest_sha256,
            source_json_path=source_path,
            source_json_sha256=source_sha256,
            source_time_s=time_s,
            source_position_world_m=position,
            source_semantic_dfr_quat_xyzw=quaternion_xyzw,
            initial_base_yaw_rad=_initial_yaw(position, preferred_vx),
            package_item=dict(item),
        )
    _require(sorted(references) == list(range(1, expected_count + 1)), "cases are not contiguous")
    return references


def save_exact_source_playback_plan(
    path: Path,
    plan: RiserPlaybackPlan,
    source: ExactSourceRiserReference,
) -> None:
    plan.validate()
    _require(len(plan.time_s) == source.source_pose_count, "plan/source anchor count mismatch")
    metadata = {
        "schema": PLAYBACK_SCHEMA,
        "case": plan.case,
        "vertical_shift_m": plan.vertical_shift_m,
        "planning_strategy": plan.planning_strategy,
        "target_attitude_contract": "semantic_dfr_to_physical_cam_v1",
        "proxy_joint_order": ["pitch", "roll", "continuous_yaw"],
        "trajectory_integrity_contract": EXACT_SOURCE_CONTRACT,
        "source_manifest_sha256": source.package_manifest_sha256,
        "source_json_sha256": source.source_json_sha256,
        "source_pose_count": source.source_pose_count,
        "source_timestamp_count": source.source_pose_count,
        "retargeted_waypoint_state_count": source.source_pose_count,
        "transition_count": source.source_pose_count - 1,
        "execution_state_count": len(plan.time_s),
        "execution_transition_count": len(plan.time_s) - 1,
        "ordered_target_geometry_preserved": True,
        "source_timestamps_preserved": True,
        "initialization_separated": True,
        "initialization_state_count": 0,
        "target_link": "ee1_tool",
        "semantic_dfr_quaternion_order": "xyzw",
        "semantic_forward_axis": "+Y",
        "physical_gimbal_is_diagnostic_only": True,
        "trajectory_integrity_passed": True,
        "quality_gate_passed": False,
        "valid_for_training": False,
        "training_started": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
        time_s=plan.time_s,
        execution_time_s=plan.time_s,
        target_position_world_m=plan.target_position_world_m,
        target_semantic_dfr_quat_wxyz=plan.target_semantic_dfr_quat_wxyz,
        base_xy_yaw=plan.base_xy_yaw,
        riser_q=plan.riser_q,
        proxy_gimbal_q=plan.proxy_gimbal_q,
        feedforward_v_wz=plan.feedforward_v_wz,
        feedforward_riser_velocity=plan.feedforward_riser_velocity,
        feedforward_proxy_velocity=plan.feedforward_proxy_velocity,
        source_time_s=source.source_time_s,
        source_target_position_world_m=source.source_position_world_m,
        source_target_semantic_dfr_quat_xyzw=source.source_semantic_dfr_quat_xyzw,
        source_anchor_execution_index=np.arange(source.source_pose_count, dtype=np.int64),
        initialization_time_s=np.empty(0, dtype=np.float64),
        initialization_state=np.empty((0, 7), dtype=np.float64),
    )


def audit_exact_source_playback_plan(
    path: Path,
    source: ExactSourceRiserReference,
) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        source_time = np.asarray(data["source_time_s"], dtype=np.float64)
        source_position = np.asarray(data["source_target_position_world_m"], dtype=np.float64)
        source_quaternion = np.asarray(data["source_target_semantic_dfr_quat_xyzw"], dtype=np.float64)
        execution_time = np.asarray(data["execution_time_s"], dtype=np.float64)
        anchor = np.asarray(data["source_anchor_execution_index"], dtype=np.int64)
        target_position = np.asarray(data["target_position_world_m"], dtype=np.float64)
        target_wxyz = np.asarray(data["target_semantic_dfr_quat_wxyz"], dtype=np.float64)
        initialization_time = np.asarray(data["initialization_time_s"], dtype=np.float64)
        initialization_state = np.asarray(data["initialization_state"], dtype=np.float64)
    expected_position = source.source_position_world_m.copy()
    expected_position[:, 2] += float(metadata.get("vertical_shift_m", math.nan))
    source_wxyz = _normalize_wxyz(source.source_semantic_dfr_quat_xyzw)
    anchor_shape_valid = anchor.shape == (source.source_pose_count,)
    anchor_bounds_valid = bool(
        anchor_shape_valid
        and np.all(anchor >= 0)
        and np.all(anchor < len(execution_time))
    )
    if anchor_bounds_valid:
        mapped_position = target_position[anchor]
        mapped_quaternion = target_wxyz[anchor]
        attitude_dot = np.abs(np.sum(mapped_quaternion * source_wxyz, axis=1))
        position_error = np.linalg.norm(mapped_position - expected_position, axis=1)
    else:
        mapped_position = np.empty((0, 3), dtype=np.float64)
        attitude_dot = np.empty(0, dtype=np.float64)
        position_error = np.empty(0, dtype=np.float64)
    checks = {
        "source_manifest_hash_bound": metadata.get("source_manifest_sha256")
        == source.package_manifest_sha256,
        "source_json_hash_bound": metadata.get("source_json_sha256")
        == source.source_json_sha256,
        "source_time_verbatim": bool(np.array_equal(source_time, source.source_time_s)),
        "source_position_verbatim": bool(
            np.array_equal(source_position, source.source_position_world_m)
        ),
        "source_attitude_verbatim": bool(
            np.array_equal(source_quaternion, source.source_semantic_dfr_quat_xyzw)
        ),
        "execution_time_strict": bool(
            execution_time.shape == source.source_time_s.shape
            and execution_time[0] == 0.0
            and np.all(np.diff(execution_time) > 0.0)
        ),
        "anchor_count_exact": anchor_shape_valid,
        "anchor_map_strict": bool(
            anchor_bounds_valid
            and np.array_equal(anchor, np.arange(source.source_pose_count))
        ),
        "mapped_position_exact": bool(
            anchor_bounds_valid and np.max(position_error) <= 1e-12
        ),
        "mapped_attitude_exact": bool(
            anchor_bounds_valid and np.min(attitude_dot) >= 1.0 - 1e-12
        ),
        "initialization_separate_empty": bool(
            initialization_time.shape == (0,)
            and initialization_state.shape[0] == 0
            and metadata.get("initialization_state_count") == 0
            and metadata.get("initialization_separated") is True
        ),
        "semantic_contract": metadata.get("target_link") == "ee1_tool"
        and metadata.get("semantic_dfr_quaternion_order") == "xyzw"
        and metadata.get("semantic_forward_axis") == "+Y",
        "reference_only": metadata.get("trajectory_integrity_passed") is True
        and metadata.get("quality_gate_passed") is False
        and metadata.get("valid_for_training") is False
        and metadata.get("training_started") is False,
    }
    passed = all(checks.values())
    return {
        "case": source.case,
        "file": path.name,
        "plan_sha256": sha256_file(path),
        "source_json_sha256": source.source_json_sha256,
        "source_pose_count": source.source_pose_count,
        "source_timestamp_count": len(source_time),
        "retargeted_waypoint_state_count": len(anchor),
        "transition_count": source.source_pose_count - 1,
        "execution_state_count": len(execution_time),
        "execution_transition_count": len(execution_time) - 1,
        "source_duration_s": float(source.source_time_s[-1]),
        "execution_duration_s": float(execution_time[-1]),
        "source_path_length_m": _path_length(source.source_position_world_m),
        "mapped_target_path_length_m": (
            _path_length(mapped_position) if len(mapped_position) >= 2 else math.nan
        ),
        "maximum_mapped_position_error_m": (
            float(np.max(position_error)) if len(position_error) else math.inf
        ),
        "maximum_mapped_attitude_error_deg": (
            float(
                np.degrees(
                    2.0 * np.arccos(np.clip(np.min(attitude_dot), -1.0, 1.0))
                )
            )
            if len(attitude_dot)
            else math.inf
        ),
        "ordered_target_geometry_preserved": checks["mapped_position_exact"],
        "source_timestamps_preserved": checks["source_time_verbatim"],
        "initialization_separated": checks["initialization_separate_empty"],
        "trajectory_integrity_passed": passed,
        "quality_gate_passed": False,
        "valid_for_training": False,
        "checks": checks,
        "passed": passed,
    }
