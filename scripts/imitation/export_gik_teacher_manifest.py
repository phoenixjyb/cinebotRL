#!/usr/bin/env python3
"""Export QA-filtered MATLAB GIK teacher manifests to CineBotRL demo NPZ files.

The input manifest is produced by the MATLAB GIK benchmark pipeline and contains
one accepted teacher episode per trajectory.  This converter intentionally uses
only manifest entries that pass QA and preserves the QA metadata in the output
manifest so failed episodes can stay reserved for RL curriculum/evaluation.

The newer ``gik_offline_teacher_manifest_v1`` schema separates strict
``accepted`` rows from ``near_pass`` and ``contact_near_pass`` rows.  This
script defaults to accepted-only export; non-strict tiers require explicit
opt-in flags.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ARM_LOWER_SAFE = np.array([-1.0, 0.55, -2.0, -1.0, -0.8, -0.8], dtype=np.float32)
ARM_UPPER_SAFE = np.array([1.0, 1.45, -0.4, 1.0, 0.8, 0.8], dtype=np.float32)
MAX_LINEAR_VELOCITY = 1.5
MAX_ANGULAR_VELOCITY = 2.0
ROBOT_FOOTPRINT_RADIUS = 0.35
Q13_TO_SIM9_INDICES = np.asarray([0, 1, 2, 3, 4, 5, 10, 11, 12], dtype=np.int64)


@dataclass
class ExportStats:
    source_mat: str
    source_json: str
    output_npz: str
    index: int
    method: str
    video_id: str
    obstacle_case: str
    obstacle_fractions: Any
    qa_pass: bool
    quality_status: str
    max_position_error_m: float
    max_orientation_error_rad: float
    footprint_margin_m: float
    solve_elapsed_s: float | None
    num_q_samples: int
    num_action_samples: int
    duration_s: float
    rate_hz_median: float
    completed_waypoints: int | None
    success_count: int | None
    arm_clip_fraction: float
    arm_clip_fraction_by_joint: list[float]
    base_clip_fraction: float
    base_clip_fraction_by_axis: list[float]
    max_abs_action: float
    max_abs_base_action: float
    max_abs_arm_action: float
    min_obstacle_clearance_m: float | None
    obstacle_count: int


def read_dataset(group: h5py.Group, path: str) -> np.ndarray | None:
    if path not in group:
        return None
    return np.array(group[path])


def read_scalar(group: h5py.Group, path: str) -> float | None:
    arr = read_dataset(group, path)
    if arr is None or arr.size == 0:
        return None
    if getattr(arr, "dtype", None) is not None and arr.dtype.kind == "O":
        return None
    try:
        return float(np.ravel(arr)[0])
    except (TypeError, ValueError):
        return None


def read_matlab_string(group: h5py.Group, path: str) -> str | None:
    arr = read_dataset(group, path)
    if arr is None:
        return None
    try:
        flat = np.ravel(arr)
        return "".join(chr(int(v)) for v in flat if int(v) != 0)
    except Exception:
        return None


def as_samples_by_dim(arr: np.ndarray, dim: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D array, got shape {arr.shape}")
    if arr.shape[1] == dim:
        return arr
    if arr.shape[0] == dim:
        return arr.T
    raise ValueError(f"cannot orient array with shape {arr.shape} as [N,{dim}]")


def as_qtraj_sim9(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the CineBotRL 9D sim state and the original MATLAB qTraj."""

    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D qTraj array, got shape {arr.shape}")
    if arr.shape[1] in (9, 13):
        q_full = arr
    elif arr.shape[0] in (9, 13):
        q_full = arr.T
    else:
        raise ValueError(f"cannot orient qTraj with shape {arr.shape} as [N,9] or [N,13]")
    if q_full.shape[1] == 9:
        return q_full, q_full
    return q_full[:, Q13_TO_SIM9_INDICES], q_full


def as_pose_samples(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"expected 3D pose array, got shape {arr.shape}")
    if arr.shape[1:] == (4, 4):
        return arr
    if arr.shape[:2] == (4, 4):
        return np.moveaxis(arr, 2, 0)
    raise ValueError(f"cannot orient pose array with shape {arr.shape} as [N,4,4]")


def normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    norm = np.where(norm > 1e-12, norm, 1.0)
    quat = quat / norm
    sign = np.where(quat[..., :1] < 0.0, -1.0, 1.0)
    return quat * sign


def rotmat_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    rot = np.asarray(rot, dtype=np.float64)
    flat = rot.reshape(-1, 3, 3)
    out = np.empty((flat.shape[0], 4), dtype=np.float64)
    for idx, r in enumerate(flat):
        trace = np.trace(r)
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            out[idx] = [
                0.25 * s,
                (r[2, 1] - r[1, 2]) / s,
                (r[0, 2] - r[2, 0]) / s,
                (r[1, 0] - r[0, 1]) / s,
            ]
        else:
            diag = np.diagonal(r)
            axis = int(np.argmax(diag))
            if axis == 0:
                s = math.sqrt(max(1.0 + r[0, 0] - r[1, 1] - r[2, 2], 0.0)) * 2.0
                out[idx] = [
                    (r[2, 1] - r[1, 2]) / s,
                    0.25 * s,
                    (r[0, 1] + r[1, 0]) / s,
                    (r[0, 2] + r[2, 0]) / s,
                ]
            elif axis == 1:
                s = math.sqrt(max(1.0 + r[1, 1] - r[0, 0] - r[2, 2], 0.0)) * 2.0
                out[idx] = [
                    (r[0, 2] - r[2, 0]) / s,
                    (r[0, 1] + r[1, 0]) / s,
                    0.25 * s,
                    (r[1, 2] + r[2, 1]) / s,
                ]
            else:
                s = math.sqrt(max(1.0 + r[2, 2] - r[0, 0] - r[1, 1], 0.0)) * 2.0
                out[idx] = [
                    (r[1, 0] - r[0, 1]) / s,
                    (r[0, 2] + r[2, 0]) / s,
                    (r[1, 2] + r[2, 1]) / s,
                    0.25 * s,
                ]
    return normalize_quat_wxyz(out.reshape(rot.shape[:-2] + (4,)))


def concat_prefix(prefix: np.ndarray | None, tail: np.ndarray, total_len: int, label: str) -> np.ndarray:
    tail = np.asarray(tail, dtype=np.float64)
    prefix_len = total_len - tail.shape[0]
    if prefix_len < 0:
        raise ValueError(f"{label} has {tail.shape[0]} tail samples but only {total_len} action samples")
    if prefix_len == 0:
        return tail
    if prefix is None or prefix.shape[0] < prefix_len:
        if tail.shape[0] == 0:
            raise ValueError(f"{label} cannot synthesize {prefix_len} prefix samples without tail data")
        synthesized = np.repeat(tail[:1], prefix_len, axis=0)
    else:
        synthesized = np.asarray(prefix[:prefix_len], dtype=np.float64)
    return np.concatenate([synthesized, tail], axis=0)


def read_vec2(group: h5py.Group, path: str) -> np.ndarray | None:
    arr = read_dataset(group, path)
    if arr is None:
        return None
    if getattr(arr, "dtype", None) is not None and arr.dtype.kind == "O":
        return None
    try:
        return np.ravel(as_samples_by_dim(arr, 2)[0]).astype(np.float32)
    except Exception:
        try:
            flat = np.ravel(np.asarray(arr, dtype=np.float64))
        except (TypeError, ValueError):
            return None
        return flat[:2].astype(np.float32) if flat.size >= 2 else None


def read_numeric_entries(group: h5py.Group, path: str) -> list[np.ndarray]:
    """Read numeric MATLAB arrays or cell arrays stored as HDF5 references."""

    if path not in group:
        return []
    dataset = group[path]
    arr = np.array(dataset)
    if arr.dtype.kind != "O":
        return [np.asarray(arr, dtype=np.float64)]

    entries: list[np.ndarray] = []
    for ref in arr.reshape(-1):
        if not isinstance(ref, h5py.Reference) or not ref:
            continue
        target = np.array(group.file[ref])
        if target.dtype.kind == "O":
            continue
        entries.append(np.asarray(target, dtype=np.float64))
    return entries


def read_obstacle_geometry(group: h5py.Group) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prefix = "log/floorDiscs" if "log/floorDiscs/Center" in group else "floorDiscs"
    center_entries = read_numeric_entries(group, f"{prefix}/Center")
    radius_entries = read_numeric_entries(group, f"{prefix}/Radius")
    margin_entries = read_numeric_entries(group, f"{prefix}/SafetyMargin")

    centers: list[np.ndarray] = []
    if len(center_entries) == 1:
        raw = np.asarray(center_entries[0], dtype=np.float64)
        if raw.ndim == 2 and 2 in raw.shape:
            oriented = raw if raw.shape[1] == 2 else raw.T
            centers.extend(row[:2] for row in oriented)
        elif raw.size >= 2:
            centers.append(raw.reshape(-1)[:2])
    else:
        centers.extend(entry.reshape(-1)[:2] for entry in center_entries if entry.size >= 2)

    count = len(centers)

    def scalar_values(entries: list[np.ndarray], default: float) -> np.ndarray:
        if not entries:
            return np.full(count, default, dtype=np.float32)
        if len(entries) == 1 and entries[0].size >= count:
            values = entries[0].reshape(-1)[:count]
        else:
            values = np.asarray([entry.reshape(-1)[0] for entry in entries if entry.size], dtype=np.float64)
        if values.size < count:
            values = np.pad(values, (0, count - values.size), constant_values=default)
        return values[:count].astype(np.float32)

    center_array = np.asarray(centers, dtype=np.float32).reshape(count, 2) if count else np.zeros((0, 2), dtype=np.float32)
    return center_array, scalar_values(radius_entries, np.nan), scalar_values(margin_entries, 0.0)


def compute_obstacle_clearance(
    q: np.ndarray,
    centers_xy: np.ndarray,
    radii: np.ndarray,
    safety_margins: np.ndarray,
) -> np.ndarray | None:
    if centers_xy.size == 0:
        return None
    xy = q[:-1, :2]
    clearance = np.linalg.norm(xy[:, None, :] - centers_xy[None, :, :], axis=2)
    clearance -= radii[None, :]
    clearance -= ROBOT_FOOTPRINT_RADIUS
    return clearance.astype(np.float32)


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def normalize_arm_targets(q_arm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = 2.0 * (q_arm - ARM_LOWER_SAFE) / (ARM_UPPER_SAFE - ARM_LOWER_SAFE) - 1.0
    clipped = np.clip(raw, -1.0, 1.0)
    return clipped.astype(np.float32), raw


def finite_difference_base_actions(q: np.ndarray, time: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    base = q[:, :3]
    dt = np.diff(time)
    if np.any(dt <= 0):
        bad = int(np.sum(dt <= 0))
        raise ValueError(f"time must be strictly increasing; {bad} non-positive step(s)")
    dxy_world = np.diff(base[:, :2], axis=0) / dt[:, None]
    dyaw = wrap_to_pi(np.diff(base[:, 2])) / dt
    yaw = base[:-1, 2]
    c = np.cos(yaw)
    s = np.sin(yaw)
    vx_body = c * dxy_world[:, 0] + s * dxy_world[:, 1]
    vy_body = -s * dxy_world[:, 0] + c * dxy_world[:, 1]
    raw = np.stack(
        [
            vx_body / MAX_LINEAR_VELOCITY,
            vy_body / MAX_LINEAR_VELOCITY,
            dyaw / MAX_ANGULAR_VELOCITY,
        ],
        axis=1,
    )
    clipped = np.clip(raw, -1.0, 1.0)
    meta = {
        "max_abs_vx_body": float(np.max(np.abs(vx_body))) if vx_body.size else 0.0,
        "max_abs_vy_body": float(np.max(np.abs(vy_body))) if vy_body.size else 0.0,
        "max_abs_wz": float(np.max(np.abs(dyaw))) if dyaw.size else 0.0,
    }
    return clipped.astype(np.float32), {"raw": raw, **meta}


def load_log(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as f:
        q_raw = read_dataset(f, "log/qTraj")
        t_raw = read_dataset(f, "log/time")
        if q_raw is None:
            raise ValueError("missing log/qTraj")
        if t_raw is None:
            raise ValueError("missing log/time")
        q, q_full = as_qtraj_sim9(q_raw)
        time = np.ravel(np.asarray(t_raw, dtype=np.float64))
        if time.size != q.shape[0]:
            if time.size == q.shape[0] - 1:
                dt = float(np.median(np.diff(time))) if time.size > 1 else 0.1
                time = np.concatenate([time, [time[-1] + dt]])
            else:
                raise ValueError(f"log/time length {time.size} does not match qTraj samples {q.shape[0]}")

        target_positions = read_dataset(f, "log/targetPositions")
        target_orientations = read_dataset(f, "log/targetOrientations")
        gimbal_attitude_target_orientations = read_dataset(f, "log/targetLinkDiagnostics/poseTargetOrientations")
        if gimbal_attitude_target_orientations is None:
            gimbal_attitude_target_orientations = target_orientations
        ee_positions = read_dataset(f, "log/eePositions")
        ee_orientations = read_dataset(f, "log/eeOrientations")
        ramp_poses_raw = read_dataset(f, "log/ramp/Poses")
        ramp_poses = as_pose_samples(ramp_poses_raw) if ramp_poses_raw is not None else None
        display_positions_raw = read_dataset(f, "log/referenceTrajectory/DisplayEndEffectorPositions")
        obstacle_centers_xy, obstacle_radii, obstacle_safety_margins = read_obstacle_geometry(f)

        return {
            "q": q,
            "q_full": q_full,
            "time": time,
            "target_positions": None if target_positions is None else as_samples_by_dim(target_positions, 3),
            "target_orientations": None if target_orientations is None else normalize_quat_wxyz(as_samples_by_dim(target_orientations, 4)),
            "gimbal_attitude_target_orientations": (
                None
                if gimbal_attitude_target_orientations is None
                else normalize_quat_wxyz(as_samples_by_dim(gimbal_attitude_target_orientations, 4))
            ),
            "target_display_positions": None if display_positions_raw is None else as_samples_by_dim(display_positions_raw, 3),
            "ee_positions": None if ee_positions is None else as_samples_by_dim(ee_positions, 3),
            "ee_orientations": None if ee_orientations is None else normalize_quat_wxyz(as_samples_by_dim(ee_orientations, 4)),
            "ramp_positions": None if ramp_poses is None else ramp_poses[:, :3, 3],
            "ramp_orientations": None if ramp_poses is None else rotmat_to_quat_wxyz(ramp_poses[:, :3, :3]),
            "completed_waypoints": read_scalar(f, "log/completedWaypoints"),
            "success_mask": read_dataset(f, "log/successMask"),
            "source_json": read_matlab_string(f, "log/referenceTrajectory/SourceJsonPath"),
            "profile": read_matlab_string(f, "cfg/meta/profile"),
            "variant": read_matlab_string(f, "cfg/robot/variant"),
            "frame_mode": read_matlab_string(f, "cfg/trajectory/frame_mode"),
            "obstacle_centers_xy": obstacle_centers_xy,
            "obstacle_radii": obstacle_radii,
            "obstacle_safety_margins": obstacle_safety_margins,
        }


def aligned_ee_samples(log: dict[str, Any], num_actions: int) -> dict[str, np.ndarray]:
    target_pos_tail = log["target_positions"]
    target_quat_tail = log["target_orientations"]
    gimbal_attitude_tail = log["gimbal_attitude_target_orientations"]
    actual_pos_tail = log["ee_positions"]
    actual_quat_tail = log["ee_orientations"]
    if target_pos_tail is None or target_quat_tail is None:
        raise ValueError("missing target end-effector pose arrays")
    if actual_pos_tail is None or actual_quat_tail is None:
        raise ValueError("missing actual end-effector pose arrays")

    ramp_pos = log["ramp_positions"]
    ramp_quat = log["ramp_orientations"]
    target_pos = concat_prefix(ramp_pos, target_pos_tail, num_actions, "target_pos")
    display_pos = log["target_display_positions"]
    if display_pos is not None and display_pos.shape[0] == num_actions:
        target_pos = display_pos
    target_quat = concat_prefix(ramp_quat, target_quat_tail, num_actions, "target_quat")
    gimbal_attitude_quat = concat_prefix(
        ramp_quat,
        gimbal_attitude_tail if gimbal_attitude_tail is not None else target_quat_tail,
        num_actions,
        "gimbal_attitude_target_quat",
    )
    actual_pos = concat_prefix(ramp_pos, actual_pos_tail, num_actions, "actual_ee_pos")
    actual_quat = concat_prefix(ramp_quat, actual_quat_tail, num_actions, "actual_ee_quat")

    for label, value, dim in (
        ("target_pos", target_pos, 3),
        ("target_quat_wxyz", target_quat, 4),
        ("gimbal_attitude_target_quat_wxyz", gimbal_attitude_quat, 4),
        ("actual_ee_pos", actual_pos, 3),
        ("actual_ee_quat_wxyz", actual_quat, 4),
    ):
        if value.shape != (num_actions, dim):
            raise ValueError(f"{label} aligned to {value.shape}, expected {(num_actions, dim)}")
        if not np.isfinite(value).all():
            raise ValueError(f"{label} contains non-finite values")

    return {
        "target_pos": target_pos.astype(np.float32),
        "target_quat_wxyz": normalize_quat_wxyz(target_quat).astype(np.float32),
        "gimbal_attitude_target_quat_wxyz": normalize_quat_wxyz(gimbal_attitude_quat).astype(np.float32),
        "actual_ee_pos": actual_pos.astype(np.float32),
        "actual_ee_quat_wxyz": normalize_quat_wxyz(actual_quat).astype(np.float32),
    }


def remap_source_path(raw_path: str, *, source_path_prefix: str, source_path_replacement: Path | None) -> Path:
    if not source_path_prefix or source_path_replacement is None:
        return Path(raw_path)
    raw = str(raw_path).replace("\\", "/")
    prefix = source_path_prefix.replace("\\", "/").rstrip("/")
    if raw.startswith(prefix):
        suffix = raw[len(prefix) :].lstrip("/\\")
        return source_path_replacement / suffix
    return Path(raw_path)


def export_one(
    item: dict[str, Any],
    out_dir: Path,
    *,
    mat_root: Path | None,
    source_path_prefix: str,
    source_path_replacement: Path | None,
    min_duration_s: float,
    copy_mat_dir: Path | None,
    skipped: list[dict[str, Any]],
) -> ExportStats | None:
    source_mat = remap_source_path(
        str(item["mat"]),
        source_path_prefix=source_path_prefix,
        source_path_replacement=source_path_replacement,
    )
    if mat_root is not None and not source_mat.is_absolute():
        source_mat = mat_root / source_mat
    if not source_mat.exists():
        raise FileNotFoundError(f"missing source mat: {source_mat}")

    log = load_log(source_mat)
    q = np.asarray(log["q"], dtype=np.float32)
    q_full = np.asarray(log.get("q_full", q), dtype=np.float32)
    time = np.asarray(log["time"], dtype=np.float64)
    duration = float(time[-1] - time[0]) if time.size else 0.0
    if q.shape[0] < 2 or duration < min_duration_s:
        skipped.append(
            {
                "index": int(item["index"]),
                "source_mat": str(source_mat),
                "duration_s": duration,
                "num_q_samples": int(q.shape[0]),
                "reason": "duration_below_minimum" if duration < min_duration_s else "insufficient_q_samples",
            }
        )
        return None

    arm_actions, arm_raw = normalize_arm_targets(q[1:, 3:9])
    base_actions, base_meta = finite_difference_base_actions(q, time)
    actions = np.concatenate([arm_actions, base_actions], axis=1).astype(np.float32)
    ee_samples = aligned_ee_samples(log, actions.shape[0])
    obstacle_centers_xy = np.asarray(log["obstacle_centers_xy"], dtype=np.float32)
    obstacle_radii = np.asarray(log["obstacle_radii"], dtype=np.float32)
    obstacle_safety_margins = np.asarray(log["obstacle_safety_margins"], dtype=np.float32)
    declared_obstacle_count = (item.get("teacher_metadata") or {}).get("obstacle_count")
    if declared_obstacle_count is not None:
        obstacle_count = max(0, int(declared_obstacle_count))
        obstacle_centers_xy = obstacle_centers_xy[:obstacle_count]
        obstacle_radii = obstacle_radii[:obstacle_count]
        obstacle_safety_margins = obstacle_safety_margins[:obstacle_count]
    obstacle_clearance = compute_obstacle_clearance(
        q,
        obstacle_centers_xy,
        obstacle_radii,
        obstacle_safety_margins,
    )
    quality_status = str((item.get("teacher_metadata") or {}).get("quality_status") or "accepted")
    strict_qa_pass = quality_status == "accepted"

    arm_clip_mask = np.abs(arm_raw) > 1.0
    arm_valid_mask = ~arm_clip_mask
    base_raw = base_meta.pop("raw")
    base_clip_mask = np.abs(base_raw) > 1.0
    base_valid_mask = ~base_clip_mask
    action_valid_mask = np.concatenate([arm_valid_mask, base_valid_mask], axis=1)
    dt = np.diff(time)
    success_mask = log["success_mask"]
    success_count = None if success_mask is None else int(np.sum(np.asarray(success_mask).astype(bool)))

    stem = f"{int(item['index']):04d}_{item.get('video_id') or source_mat.stem}_{item['method']}"
    out_path = out_dir / f"{stem}.npz"
    copied_mat_path = ""
    if copy_mat_dir is not None:
        copy_mat_dir.mkdir(parents=True, exist_ok=True)
        copied = copy_mat_dir / source_mat.name
        if not copied.exists():
            shutil.copy2(source_mat, copied)
        copied_mat_path = str(copied)

    np.savez_compressed(
        out_path,
        actions=actions,
        action_valid_mask=action_valid_mask.astype(bool),
        arm_action_unclipped=arm_raw.astype(np.float32),
        arm_action_valid_mask=arm_valid_mask.astype(bool),
        arm_target_physical=q[1:, 3:9].astype(np.float32),
        base_action_unclipped=base_raw.astype(np.float32),
        base_action_valid_mask=base_valid_mask.astype(bool),
        target_pos=ee_samples["target_pos"],
        target_quat_wxyz=ee_samples["target_quat_wxyz"],
        gimbal_attitude_target_quat_wxyz=ee_samples["gimbal_attitude_target_quat_wxyz"],
        actual_ee_pos=ee_samples["actual_ee_pos"],
        actual_ee_quat_wxyz=ee_samples["actual_ee_quat_wxyz"],
        obstacle_centers_xy=obstacle_centers_xy,
        obstacle_radii=obstacle_radii,
        obstacle_safety_margins=obstacle_safety_margins,
        robot_footprint_radius=np.float32(ROBOT_FOOTPRINT_RADIUS),
        obstacle_valid_mask=np.ones(obstacle_centers_xy.shape[0], dtype=bool),
        obstacle_center_xy=obstacle_centers_xy[0] if obstacle_centers_xy.size else np.asarray([np.nan, np.nan], dtype=np.float32),
        obstacle_radius=np.float32(obstacle_radii[0] if obstacle_radii.size else np.nan),
        obstacle_safety_margin=np.float32(obstacle_safety_margins[0] if obstacle_safety_margins.size else np.nan),
        obstacle_clearance_m=np.empty((actions.shape[0], 0), dtype=np.float32) if obstacle_clearance is None else obstacle_clearance,
        min_obstacle_dist=(
            np.full(actions.shape[0], np.nan, dtype=np.float32)
            if obstacle_clearance is None
            else np.min(obstacle_clearance, axis=1)
        ),
        q_current=q[:-1],
        q_next=q[1:],
        q=q,
        q_full=q_full,
        q_full_dim=np.asarray(q_full.shape[1], dtype=np.int32),
        q_sim9_source_indices=Q13_TO_SIM9_INDICES if q_full.shape[1] == 13 else np.arange(9, dtype=np.int64),
        time=time.astype(np.float64),
        dt=dt.astype(np.float32),
        source_mat=str(source_mat),
        copied_source_mat=copied_mat_path,
        source_json=str(item.get("source_file") or log["source_json"] or ""),
        profile=str(item.get("profile") or ""),
        variant=log["variant"] or "",
        frame_mode=str(item.get("frame_mode") or log["frame_mode"] or ""),
        teacher_method=str(item["method"]),
        teacher_metadata=json.dumps(item.get("teacher_metadata") or {}),
        teacher_index=np.asarray(int(item["index"])),
        qa_pass=np.asarray(strict_qa_pass),
        strict_qa_pass=np.asarray(strict_qa_pass),
        quality_status=np.asarray(quality_status),
        max_position_error_m=np.asarray(float(item["max_position_error_m"]), dtype=np.float32),
        max_orientation_error_rad=np.asarray(float(item["max_orientation_error_rad"]), dtype=np.float32),
        footprint_margin_m=np.asarray(float(item["footprint_margin_m"]), dtype=np.float32),
        obstacle_case=str(item.get("obstacle_case") or ""),
        obstacle_fractions=np.asarray(item.get("obstacle_fractions", np.nan), dtype=np.float32),
        arm_lower_safe=ARM_LOWER_SAFE,
        arm_upper_safe=ARM_UPPER_SAFE,
        max_linear_velocity=np.float32(MAX_LINEAR_VELOCITY),
        max_angular_velocity=np.float32(MAX_ANGULAR_VELOCITY),
        base_velocity_meta=json.dumps(base_meta),
    )

    return ExportStats(
        source_mat=str(source_mat),
        source_json=str(item.get("source_file") or log["source_json"] or ""),
        output_npz=str(out_path),
        index=int(item["index"]),
        method=str(item["method"]),
        video_id=str(item.get("video_id") or ""),
        obstacle_case=str(item.get("obstacle_case") or ""),
        obstacle_fractions=item.get("obstacle_fractions"),
        qa_pass=strict_qa_pass,
        quality_status=quality_status,
        max_position_error_m=float(item["max_position_error_m"]),
        max_orientation_error_rad=float(item["max_orientation_error_rad"]),
        footprint_margin_m=float(item["footprint_margin_m"]),
        solve_elapsed_s=None if item.get("solve_elapsed_s") is None else float(item["solve_elapsed_s"]),
        num_q_samples=int(q.shape[0]),
        num_action_samples=int(actions.shape[0]),
        duration_s=duration,
        rate_hz_median=float(1.0 / np.median(dt)) if dt.size else 0.0,
        completed_waypoints=None if log["completed_waypoints"] is None else int(log["completed_waypoints"]),
        success_count=success_count,
        arm_clip_fraction=float(np.mean(arm_clip_mask)),
        arm_clip_fraction_by_joint=np.mean(arm_clip_mask, axis=0).astype(float).tolist(),
        base_clip_fraction=float(np.mean(base_clip_mask)),
        base_clip_fraction_by_axis=np.mean(base_clip_mask, axis=0).astype(float).tolist(),
        max_abs_action=float(np.max(np.abs(actions))) if actions.size else 0.0,
        max_abs_base_action=float(np.max(np.abs(base_actions))) if base_actions.size else 0.0,
        max_abs_arm_action=float(np.max(np.abs(arm_actions))) if arm_actions.size else 0.0,
        min_obstacle_clearance_m=None if obstacle_clearance is None else float(np.min(obstacle_clearance)),
        obstacle_count=int(obstacle_centers_xy.shape[0]),
    )


def validate_manifest_items(items: list[dict[str, Any]]) -> None:
    for item in items:
        metadata = item.get("teacher_metadata") or {}
        quality_status = str(metadata.get("quality_status") or "accepted")
        max_position = float(item.get("max_position_error_m", math.inf))
        max_orientation = float(item.get("max_orientation_error_rad", math.inf))
        p95_position_raw = item.get("p95_position_error_m")
        p95_position = math.inf if p95_position_raw is None else float(p95_position_raw)
        footprint_margin = float(item.get("footprint_margin_m", -math.inf))
        if quality_status == "near_pass":
            if max_position > 0.12 or p95_position > 0.09 or max_orientation > 0.25 or footprint_margin < 0.0:
                raise ValueError(f"index {item.get('index')} violates near_pass thresholds")
        elif quality_status == "contact_near_pass":
            if max_position > 0.10 or max_orientation > 0.25 or footprint_margin < -0.001:
                raise ValueError(f"index {item.get('index')} violates contact_near_pass thresholds")
        else:
            if max_position > 0.10:
                raise ValueError(f"index {item.get('index')} exceeds position QA threshold")
            if footprint_margin < 0.0:
                raise ValueError(f"index {item.get('index')} has negative footprint margin")


def normalize_offline_manifest_item(item: dict[str, Any], source_index: int) -> dict[str, Any]:
    """Map gik_offline_teacher_manifest_v1 rows to the legacy exporter shape."""

    quality_status = str(item.get("quality_status") or "")
    scenario = str(item.get("scenario") or "")
    obstacle_case = str(item.get("obstacle_case") or "")
    episode_index = int(item.get("episode_index") or source_index + 1)
    method = str(item.get("teacher_method") or "gik_teacher")
    min_clearance = item.get("min_footprint_clearance_m")
    if min_clearance is None:
        min_clearance = item.get("footprint_margin_m")
    if min_clearance is None and int(item.get("obstacle_count") or 0) == 0:
        min_clearance = math.inf
    normalized = {
        "mat": item.get("mat"),
        "source_file": item.get("source_json"),
        "index": source_index + 1,
        "episode_index": episode_index,
        "method": method,
        "video_id": item.get("video_id") or "",
        "obstacle_case": obstacle_case,
        "obstacle_fractions": item.get("obstacle_fractions", []),
        "max_position_error_m": item.get("max_position_error_m"),
        "max_orientation_error_rad": item.get("max_orientation_error_rad"),
        "p95_position_error_m": item.get("p95_position_error_m"),
        "footprint_margin_m": min_clearance,
        "solve_elapsed_s": item.get("solve_elapsed_s"),
        "profile": item.get("profile") or "",
        "frame_mode": item.get("frame_mode") or "",
        "teacher_metadata": {
            "quality_status": quality_status,
            "policy_role": item.get("policy_role"),
            "scenario": scenario,
            "obstacle_count": item.get("obstacle_count"),
            "obstacle_case": obstacle_case,
            "episode_index": episode_index,
            "teacher_method": method,
            "quality_source": item.get("quality_source"),
            "p95_position_error_m": item.get("p95_position_error_m"),
            "video_id": item.get("video_id") or "",
            "notes": item.get("notes") or "",
            "source_json": item.get("source_json") or "",
            "source_mat": item.get("mat") or "",
        },
    }
    missing = [key for key in ("mat", "max_position_error_m", "max_orientation_error_rad", "footprint_margin_m") if normalized.get(key) is None]
    if missing:
        raise ValueError(f"manifest item {source_index + 1} missing required fields after normalization: {missing}")
    return normalized


def load_manifest_items(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = json.loads(args.manifest.read_text(encoding="utf-8"))
    source_meta: dict[str, Any] = {"source_schema": "legacy_list"}
    if isinstance(raw, dict):
        source_meta = {
            "source_schema": raw.get("schema", "dict_manifest"),
            "source_num_items": raw.get("num_items"),
            "source_num_accepted": raw.get("num_accepted"),
            "source_num_near_pass": raw.get("num_near_pass"),
            "source_num_contact_near_pass": raw.get("num_contact_near_pass"),
        }
        raw_items = raw.get("items", [])
        if raw.get("schema") == "gik_offline_teacher_manifest_v1":
            allowed_statuses = {"accepted"}
            if args.include_near_pass:
                allowed_statuses.add("near_pass")
            if args.include_contact_near_pass:
                allowed_statuses.add("contact_near_pass")
            selected = [
                normalize_offline_manifest_item(item, idx)
                for idx, item in enumerate(raw_items)
                if str(item.get("quality_status") or "") in allowed_statuses
            ]
            source_meta["selected_quality_statuses"] = sorted(allowed_statuses)
            return selected, source_meta
        return raw_items, source_meta
    if isinstance(raw, list):
        return raw, source_meta
    raise TypeError(f"unsupported manifest root type: {type(raw).__name__}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="QA-passing teacher manifest JSON.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/gik_teacher_one_obstacle_pass"))
    parser.add_argument("--mat-root", type=Path, default=None, help="Optional root for relative manifest mat paths.")
    parser.add_argument(
        "--source-path-prefix",
        default="",
        help="Optional source path prefix to replace before opening manifest .mat paths.",
    )
    parser.add_argument(
        "--source-path-replacement",
        type=Path,
        default=None,
        help="Replacement root used with --source-path-prefix.",
    )
    parser.add_argument("--min-duration-s", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--copy-mats", action="store_true", help="Copy source .mat files into output-dir/source_mats.")
    parser.add_argument(
        "--dry-run-manifest",
        action="store_true",
        help="Validate manifest filtering/QA thresholds without opening source .mat files.",
    )
    parser.add_argument(
        "--include-near-pass",
        action="store_true",
        help="Opt in to two-obstacle near_pass rows from gik_offline_teacher_manifest_v1.",
    )
    parser.add_argument(
        "--include-contact-near-pass",
        action="store_true",
        help="Opt in to contact_near_pass rows from gik_offline_teacher_manifest_v1.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    args.manifest = manifest_path
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path_replacement = args.source_path_replacement
    if source_path_replacement is not None:
        source_path_replacement = source_path_replacement.expanduser()
        if not source_path_replacement.is_absolute():
            source_path_replacement = (Path.cwd() / source_path_replacement).resolve()

    items, source_meta = load_manifest_items(args)
    if args.limit > 0:
        items = items[: args.limit]
    if not isinstance(items, list) or not items:
        raise SystemExit(f"manifest contains no teacher items: {manifest_path}")
    validate_manifest_items(items)
    if args.dry_run_manifest:
        statuses = collections.Counter(
            str((item.get("teacher_metadata") or {}).get("quality_status") or "accepted") for item in items
        )
        scenarios = collections.Counter(
            str((item.get("teacher_metadata") or {}).get("scenario") or item.get("obstacle_case") or "") for item in items
        )
        print(f"Manifest:       {manifest_path}")
        print(f"Selected items: {len(items)}")
        print(f"Statuses:       {dict(statuses)}")
        print(f"Scenarios:      {dict(scenarios)}")
        print("Dry run:        true")
        return 0

    copy_mat_dir = output_dir / "source_mats" if args.copy_mats else None
    stats: list[ExportStats] = []
    failures: list[dict[str, str]] = []
    skipped: list[dict[str, Any]] = []
    for item in items:
        try:
            exported = export_one(
                item,
                output_dir,
                mat_root=args.mat_root,
                source_path_prefix=args.source_path_prefix,
                source_path_replacement=source_path_replacement,
                min_duration_s=args.min_duration_s,
                copy_mat_dir=copy_mat_dir,
                skipped=skipped,
            )
            if exported is not None:
                stats.append(exported)
        except Exception as exc:
            failures.append({"index": str(item.get("index")), "method": str(item.get("method")), "source_mat": str(item.get("mat")), "error": str(exc)})

    total_samples = sum(item.num_action_samples for item in stats)
    manifest = {
        "schema": "cinebotrl_gik_teacher_manifest_v1",
        "source_manifest": str(manifest_path),
        **source_meta,
        "action_contract": "[arm6 normalized absolute targets, base_vx, base_vy, base_wz]",
        "qa_contract": "qa_pass && max_position_error_m <= 0.10 && footprint_margin_m >= 0",
        "arm_lower_safe": ARM_LOWER_SAFE.tolist(),
        "arm_upper_safe": ARM_UPPER_SAFE.tolist(),
        "max_linear_velocity": MAX_LINEAR_VELOCITY,
        "max_angular_velocity": MAX_ANGULAR_VELOCITY,
        "output_dir": str(output_dir),
        "num_manifest_items": len(items),
        "num_logs": len(stats),
        "num_failures": len(failures),
        "num_skipped": len(skipped),
        "total_action_samples": total_samples,
        "items": [asdict(item) for item in stats],
        "failures": failures,
        "skipped": skipped,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "source_manifest.json").write_text(json.dumps(items, indent=2), encoding="utf-8")

    print(f"Manifest:       {manifest_path}")
    print(f"Output dir:     {output_dir}")
    print(f"Exported logs:  {len(stats)} / {len(items)}")
    print(f"Action samples: {total_samples}")
    print(f"Failures:       {len(failures)}")
    print(f"Skipped:        {len(skipped)}")
    if stats:
        print("Arm valid:      " + " ".join(f"{1.0 - v:.3f}" for v in np.mean([s.arm_clip_fraction_by_joint for s in stats], axis=0)))
        print("Base valid:     " + " ".join(f"{1.0 - v:.3f}" for v in np.mean([s.base_clip_fraction_by_axis for s in stats], axis=0)))
        print(f"Min footprint:  {min(s.footprint_margin_m for s in stats):.4f}")
        print(f"Max pos err:    {max(s.max_position_error_m for s in stats):.4f}")
    return 0 if stats and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
