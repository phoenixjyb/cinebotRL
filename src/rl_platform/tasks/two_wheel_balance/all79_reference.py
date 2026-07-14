"""Strict loaders and metrics for the corrected all-79 GIK reference corpus."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np


V3_PACKAGE_SCHEMA = "cinebotrl_gik_monorepo_ee1_curriculum_package_v2"
V3_EPISODE_SCHEMA = "cinebotrl_gik_monorepo_ee1_split_teacher_v2"
EXPECTED_ACTION_ORDER = (
    "joint6_arm_yaw_abs_norm",
    "joint5_arm_pitch_abs_norm",
    "joint4_elbow_pitch_abs_norm",
    "base_vx_body_norm",
    "base_vy_body_norm",
    "base_wz_norm",
)
FULL_STAGE_CASE_RE = re.compile(r"^(?P<case>\d{4})_.*\.json$")


@dataclass(frozen=True)
class FullReference:
    case: int
    path: Path
    positions_m: np.ndarray
    attitudes_wxyz: np.ndarray
    time_s: np.ndarray
    metadata: dict[str, object]


@dataclass(frozen=True)
class SparseTeacher:
    case: int
    path: Path
    base_arm_q: np.ndarray
    time_s: np.ndarray
    dfr_attitudes_wxyz: np.ndarray
    action_order: tuple[str, ...]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_acquisition_time_scale_overrides(value: str) -> dict[int, float]:
    overrides: dict[int, float] = {}
    if not value.strip():
        return overrides
    for item in value.split(","):
        fields = item.strip().split(":")
        require(len(fields) == 2, f"invalid acquisition scale override {item!r}")
        try:
            case = int(fields[0])
            scale = float(fields[1])
        except ValueError as exc:
            raise ValueError(f"invalid acquisition scale override {item!r}") from exc
        require(1 <= case <= 79, f"acquisition override case out of range: {case}")
        require(math.isfinite(scale) and scale >= 1.0, f"invalid acquisition scale: {scale}")
        require(case not in overrides, f"duplicate acquisition override case: {case}")
        overrides[case] = scale
    return overrides


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar(data: np.lib.npyio.NpzFile, key: str) -> object:
    require(key in data.files, f"missing {key}")
    value = np.asarray(data[key])
    require(value.size == 1, f"{key} must be scalar, got {value.shape}")
    return value.reshape(-1)[0].item()


def normalize_quaternions_wxyz(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 2 and values.shape[1] == 4, f"bad quaternion shape {values.shape}")
    require(np.isfinite(values).all(), "quaternions contain non-finite values")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    require(bool(np.all(norms > 1e-12)), "quaternion has zero norm")
    values = values / norms
    return values * np.where(values[:, :1] < 0.0, -1.0, 1.0)


def load_full_reference(path: Path) -> FullReference:
    path = path.resolve()
    match = FULL_STAGE_CASE_RE.match(path.name)
    require(match is not None, f"cannot derive case id from {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    poses = payload.get("poses")
    metadata = payload.get("metadata")
    require(isinstance(poses, list) and len(poses) >= 2, f"invalid poses in {path}")
    require(isinstance(metadata, dict), f"invalid metadata in {path}")
    positions = np.asarray([item["position"] for item in poses], dtype=np.float64)
    quat_xyzw = np.asarray([item["orientation"] for item in poses], dtype=np.float64)
    require(positions.shape == (len(poses), 3), f"bad position shape in {path}: {positions.shape}")
    require(np.isfinite(positions).all(), f"non-finite positions in {path}")
    require(quat_xyzw.shape == (len(poses), 4), f"bad orientation shape in {path}")
    attitudes = normalize_quaternions_wxyz(quat_xyzw[:, [3, 0, 1, 2]])
    waypoint_dt = float(metadata.get("waypoint_dt", 0.0))
    duration = float(metadata.get("duration_s", 0.0))
    require(waypoint_dt > 0.0 and duration > 0.0, f"invalid timing in {path}")
    time_s = np.linspace(0.0, duration, len(poses), dtype=np.float64)
    require(
        abs(float(np.median(np.diff(time_s))) - waypoint_dt) <= waypoint_dt * 0.02,
        f"duration/count mismatch in {path}",
    )
    return FullReference(
        case=int(match.group("case")),
        path=path,
        positions_m=positions,
        attitudes_wxyz=attitudes,
        time_s=time_s,
        metadata=metadata,
    )


def discover_full_stage(stage_dir: Path, expected_cases: int = 79) -> dict[int, FullReference]:
    references: dict[int, FullReference] = {}
    for path in sorted(stage_dir.resolve().glob("*.json")):
        if FULL_STAGE_CASE_RE.match(path.name) is None:
            continue
        reference = load_full_reference(path)
        require(reference.case not in references, f"duplicate full-stage case {reference.case}")
        references[reference.case] = reference
    expected = set(range(1, expected_cases + 1))
    require(set(references) == expected, f"full stage cases differ: {sorted(set(references) ^ expected)}")
    return references


def load_sparse_teacher(path: Path) -> SparseTeacher:
    path = path.resolve()
    with np.load(path, allow_pickle=False) as data:
        require(_scalar(data, "schema") == V3_EPISODE_SCHEMA, f"legacy schema in {path}")
        require(bool(_scalar(data, "valid_for_training")), f"teacher not valid in {path}")
        require(not bool(_scalar(data, "runtime_approved")), f"offline teacher marked runtime-approved: {path}")
        require(_scalar(data, "position_target_link") == "ee1_tool", f"wrong target link in {path}")
        require(_scalar(data, "quaternion_order") == "wxyz", f"wrong quaternion order in {path}")
        action_order = tuple(str(item) for item in np.asarray(data["action_order"]).tolist())
        require(action_order == EXPECTED_ACTION_ORDER, f"wrong action order in {path}: {action_order}")
        current = np.asarray(data["q_current_base_arm_6"], dtype=np.float64)
        next_q = np.asarray(data["q_next_base_arm_6"], dtype=np.float64)
        sample_time = np.asarray(data["time_s"], dtype=np.float64).reshape(-1)
        dt = np.asarray(data["dt_s"], dtype=np.float64).reshape(-1)
        attitudes = normalize_quaternions_wxyz(
            np.asarray(data["gimbal_attitude_target_world_dfr_quat_wxyz"], dtype=np.float64)
        )
        case = int(_scalar(data, "episode_index"))
    require(current.ndim == 2 and current.shape[1] == 6, f"bad current q shape in {path}")
    require(next_q.shape == current.shape, f"q pair mismatch in {path}")
    require(sample_time.shape == dt.shape == (len(current),), f"timing mismatch in {path}")
    require(attitudes.shape == (len(current), 4), f"attitude length mismatch in {path}")
    require(np.isfinite(current).all() and np.isfinite(next_q).all(), f"non-finite q in {path}")
    require(np.all(dt > 0.0) and np.all(np.diff(sample_time) > 0.0), f"non-monotonic time in {path}")
    require(np.allclose(current[1:], next_q[:-1], atol=2e-6), f"discontinuous q pairs in {path}")
    q_path = np.vstack((current, next_q[-1]))
    initial_time = sample_time[0] - dt[0]
    time_s = np.concatenate(([initial_time], sample_time))
    return SparseTeacher(
        case=case,
        path=path,
        base_arm_q=q_path,
        time_s=time_s,
        dfr_attitudes_wxyz=attitudes,
        action_order=action_order,
    )


def discover_v3_package(package_dir: Path, expected_cases: int = 79) -> dict[int, SparseTeacher]:
    package_dir = package_dir.resolve()
    manifest = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))
    require(manifest.get("schema") == V3_PACKAGE_SCHEMA, "wrong all-79 package schema")
    require(manifest.get("valid_for_training") is True, "package is not teacher-valid")
    require(manifest.get("runtime_approved") is False, "offline package cannot be runtime-approved")
    require(tuple(manifest.get("learned_action_order", ())) == EXPECTED_ACTION_ORDER, "package action order changed")
    require(manifest.get("accepted_count") == expected_cases, "package accepted count changed")
    require(manifest.get("rejected_count") == 0, "package contains rejected cases")
    teachers: dict[int, SparseTeacher] = {}
    for item in manifest.get("accepted", []):
        path = package_dir / item["npz"]
        require(path.is_file(), f"missing teacher file {path}")
        require(sha256(path) == item["sha256"], f"checksum mismatch for {path}")
        teacher = load_sparse_teacher(path)
        require(teacher.case == int(item["episode_index"]), f"case mismatch for {path}")
        require(teacher.case not in teachers, f"duplicate teacher case {teacher.case}")
        teachers[teacher.case] = teacher
    expected = set(range(1, expected_cases + 1))
    require(set(teachers) == expected, f"teacher cases differ: {sorted(set(teachers) ^ expected)}")
    return teachers


def source_body_velocities(teacher: SparseTeacher) -> np.ndarray:
    q = teacher.base_arm_q
    dt = np.diff(teacher.time_s)
    delta_xy = np.diff(q[:, :2], axis=0)
    yaw = q[:-1, 2]
    c = np.cos(yaw)
    s = np.sin(yaw)
    vx = (c * delta_xy[:, 0] + s * delta_xy[:, 1]) / dt
    vy = (-s * delta_xy[:, 0] + c * delta_xy[:, 1]) / dt
    wz = np.diff(np.unwrap(q[:, 2])) / dt
    return np.column_stack((vx, vy, wz))


def regenerate_acquisition_prefix(
    reference: FullReference,
    home_position_m: np.ndarray,
    acquisition_end_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    home_position_m = np.asarray(home_position_m, dtype=np.float64)
    require(home_position_m.shape == (3,), "home position must have shape (3,)")
    require(np.isfinite(home_position_m).all(), "home position contains non-finite values")
    require(
        1 <= acquisition_end_index < len(reference.time_s),
        f"invalid acquisition end index {acquisition_end_index} for case {reference.case}",
    )
    targets = reference.positions_m.copy()
    phase = reference.time_s[: acquisition_end_index + 1] / reference.time_s[
        acquisition_end_index
    ]
    blend = 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5
    semantic_start = targets[acquisition_end_index].copy()
    targets[: acquisition_end_index + 1] = (
        home_position_m[None, :]
        + blend[:, None] * (semantic_start - home_position_m)[None, :]
    )
    return targets, semantic_start


def monotonic_pose_match(
    full: FullReference,
    positions_m: np.ndarray,
    attitudes_wxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions_m = np.asarray(positions_m, dtype=np.float64)
    attitudes_wxyz = normalize_quaternions_wxyz(attitudes_wxyz)
    require(positions_m.shape == (len(attitudes_wxyz), 3), "sparse pose shape mismatch")
    indices = []
    position_errors = []
    attitude_errors = []
    previous = -1
    for position, attitude in zip(positions_m, attitudes_wxyz, strict=True):
        start = previous + 1
        require(start < len(full.positions_m), "cannot complete monotonic pose match")
        distances = np.linalg.norm(full.positions_m[start:] - position, axis=1)
        index = start + int(np.argmin(distances))
        previous = index
        dot = abs(float(np.dot(full.attitudes_wxyz[index], attitude)))
        angle_deg = math.degrees(2.0 * math.acos(np.clip(dot, -1.0, 1.0)))
        indices.append(index)
        position_errors.append(float(distances[index - start]))
        attitude_errors.append(angle_deg)
    return (
        np.asarray(indices, dtype=np.int32),
        np.asarray(position_errors, dtype=np.float64),
        np.asarray(attitude_errors, dtype=np.float64),
    )
