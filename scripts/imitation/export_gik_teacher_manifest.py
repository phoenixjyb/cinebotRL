#!/usr/bin/env python3
"""Export QA-filtered MATLAB GIK teacher manifests to CineBotRL demo NPZ files.

The input manifest is produced by the MATLAB GIK benchmark pipeline and contains
one accepted teacher episode per trajectory.  This converter intentionally uses
only manifest entries that pass QA and preserves the QA metadata in the output
manifest so failed episodes can stay reserved for RL curriculum/evaluation.
"""

from __future__ import annotations

import argparse
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


def read_dataset(group: h5py.Group, path: str) -> np.ndarray | None:
    if path not in group:
        return None
    return np.array(group[path])


def read_scalar(group: h5py.Group, path: str) -> float | None:
    arr = read_dataset(group, path)
    if arr is None or arr.size == 0:
        return None
    return float(np.ravel(arr)[0])


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
    try:
        return np.ravel(as_samples_by_dim(arr, 2)[0]).astype(np.float32)
    except Exception:
        flat = np.ravel(np.asarray(arr, dtype=np.float64))
        return flat[:2].astype(np.float32) if flat.size >= 2 else None


def compute_obstacle_clearance(q: np.ndarray, center_xy: np.ndarray | None, radius: float | None, safety_margin: float | None) -> np.ndarray | None:
    if center_xy is None or radius is None:
        return None
    margin = float(safety_margin or 0.0)
    xy = q[:-1, :2]
    clearance = np.linalg.norm(xy - center_xy[None, :2], axis=1) - float(radius) - margin
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
        q = as_samples_by_dim(q_raw, 9)
        time = np.ravel(np.asarray(t_raw, dtype=np.float64))
        if time.size != q.shape[0]:
            if time.size == q.shape[0] - 1:
                dt = float(np.median(np.diff(time))) if time.size > 1 else 0.1
                time = np.concatenate([time, [time[-1] + dt]])
            else:
                raise ValueError(f"log/time length {time.size} does not match qTraj samples {q.shape[0]}")

        target_positions = read_dataset(f, "log/targetPositions")
        target_orientations = read_dataset(f, "log/targetOrientations")
        ee_positions = read_dataset(f, "log/eePositions")
        ee_orientations = read_dataset(f, "log/eeOrientations")
        ramp_poses_raw = read_dataset(f, "log/ramp/Poses")
        ramp_poses = as_pose_samples(ramp_poses_raw) if ramp_poses_raw is not None else None
        display_positions_raw = read_dataset(f, "log/referenceTrajectory/DisplayEndEffectorPositions")
        obstacle_center_xy = read_vec2(f, "log/floorDiscs/Center")
        if obstacle_center_xy is None:
            obstacle_center_xy = read_vec2(f, "floorDiscs/Center")
        obstacle_radius = read_scalar(f, "log/floorDiscs/Radius")
        if obstacle_radius is None:
            obstacle_radius = read_scalar(f, "floorDiscs/Radius")
        obstacle_safety_margin = read_scalar(f, "log/floorDiscs/SafetyMargin")
        if obstacle_safety_margin is None:
            obstacle_safety_margin = read_scalar(f, "floorDiscs/SafetyMargin")

        return {
            "q": q,
            "time": time,
            "target_positions": None if target_positions is None else as_samples_by_dim(target_positions, 3),
            "target_orientations": None if target_orientations is None else normalize_quat_wxyz(as_samples_by_dim(target_orientations, 4)),
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
            "obstacle_center_xy": obstacle_center_xy,
            "obstacle_radius": obstacle_radius,
            "obstacle_safety_margin": obstacle_safety_margin,
        }


def aligned_ee_samples(log: dict[str, Any], num_actions: int) -> dict[str, np.ndarray]:
    target_pos_tail = log["target_positions"]
    target_quat_tail = log["target_orientations"]
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
    actual_pos = concat_prefix(ramp_pos, actual_pos_tail, num_actions, "actual_ee_pos")
    actual_quat = concat_prefix(ramp_quat, actual_quat_tail, num_actions, "actual_ee_quat")

    for label, value, dim in (
        ("target_pos", target_pos, 3),
        ("target_quat_wxyz", target_quat, 4),
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
        "actual_ee_pos": actual_pos.astype(np.float32),
        "actual_ee_quat_wxyz": normalize_quat_wxyz(actual_quat).astype(np.float32),
    }


def export_one(item: dict[str, Any], out_dir: Path, *, mat_root: Path | None, min_duration_s: float, copy_mat_dir: Path | None) -> ExportStats | None:
    source_mat = Path(item["mat"])
    if mat_root is not None and not source_mat.is_absolute():
        source_mat = mat_root / source_mat
    if not source_mat.exists():
        raise FileNotFoundError(f"missing source mat: {source_mat}")

    log = load_log(source_mat)
    q = np.asarray(log["q"], dtype=np.float32)
    time = np.asarray(log["time"], dtype=np.float64)
    duration = float(time[-1] - time[0]) if time.size else 0.0
    if q.shape[0] < 2 or duration < min_duration_s:
        return None

    arm_actions, arm_raw = normalize_arm_targets(q[1:, 3:9])
    base_actions, base_meta = finite_difference_base_actions(q, time)
    actions = np.concatenate([arm_actions, base_actions], axis=1).astype(np.float32)
    ee_samples = aligned_ee_samples(log, actions.shape[0])
    obstacle_clearance = compute_obstacle_clearance(q, log["obstacle_center_xy"], log["obstacle_radius"], log["obstacle_safety_margin"])

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
        actual_ee_pos=ee_samples["actual_ee_pos"],
        actual_ee_quat_wxyz=ee_samples["actual_ee_quat_wxyz"],
        obstacle_center_xy=np.asarray(log["obstacle_center_xy"] if log["obstacle_center_xy"] is not None else [np.nan, np.nan], dtype=np.float32),
        obstacle_radius=np.float32(log["obstacle_radius"] if log["obstacle_radius"] is not None else np.nan),
        obstacle_safety_margin=np.float32(log["obstacle_safety_margin"] if log["obstacle_safety_margin"] is not None else np.nan),
        min_obstacle_dist=np.full(actions.shape[0], np.nan, dtype=np.float32) if obstacle_clearance is None else obstacle_clearance,
        q_current=q[:-1],
        q_next=q[1:],
        q=q,
        time=time.astype(np.float64),
        dt=dt.astype(np.float32),
        source_mat=str(source_mat),
        copied_source_mat=copied_mat_path,
        source_json=str(item.get("source_file") or log["source_json"] or ""),
        profile=str(item.get("profile") or ""),
        variant=log["variant"] or "",
        frame_mode=str(item.get("frame_mode") or log["frame_mode"] or ""),
        teacher_method=str(item["method"]),
        teacher_index=np.asarray(int(item["index"])),
        qa_pass=np.asarray(True),
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
        qa_pass=True,
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
    )


def validate_manifest_items(items: list[dict[str, Any]]) -> None:
    for item in items:
        if float(item.get("max_position_error_m", math.inf)) > 0.10:
            raise ValueError(f"index {item.get('index')} exceeds position QA threshold")
        if float(item.get("footprint_margin_m", -math.inf)) < 0.0:
            raise ValueError(f"index {item.get('index')} has negative footprint margin")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="QA-passing teacher manifest JSON.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/gik_teacher_one_obstacle_pass"))
    parser.add_argument("--mat-root", type=Path, default=None, help="Optional root for relative manifest mat paths.")
    parser.add_argument("--min-duration-s", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--copy-mats", action="store_true", help="Copy source .mat files into output-dir/source_mats.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    items = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.limit > 0:
        items = items[: args.limit]
    if not isinstance(items, list) or not items:
        raise SystemExit(f"manifest contains no teacher items: {manifest_path}")
    validate_manifest_items(items)

    copy_mat_dir = output_dir / "source_mats" if args.copy_mats else None
    stats: list[ExportStats] = []
    failures: list[dict[str, str]] = []
    for item in items:
        try:
            exported = export_one(item, output_dir, mat_root=args.mat_root, min_duration_s=args.min_duration_s, copy_mat_dir=copy_mat_dir)
            if exported is not None:
                stats.append(exported)
        except Exception as exc:
            failures.append({"index": str(item.get("index")), "method": str(item.get("method")), "source_mat": str(item.get("mat")), "error": str(exc)})

    total_samples = sum(item.num_action_samples for item in stats)
    manifest = {
        "schema": "cinebotrl_gik_teacher_manifest_v1",
        "source_manifest": str(manifest_path),
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
        "total_action_samples": total_samples,
        "items": [asdict(item) for item in stats],
        "failures": failures,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "source_manifest.json").write_text(json.dumps(items, indent=2), encoding="utf-8")

    print(f"Manifest:       {manifest_path}")
    print(f"Output dir:     {output_dir}")
    print(f"Exported logs:  {len(stats)} / {len(items)}")
    print(f"Action samples: {total_samples}")
    print(f"Failures:       {len(failures)}")
    if stats:
        print("Arm valid:      " + " ".join(f"{1.0 - v:.3f}" for v in np.mean([s.arm_clip_fraction_by_joint for s in stats], axis=0)))
        print("Base valid:     " + " ".join(f"{1.0 - v:.3f}" for v in np.mean([s.base_clip_fraction_by_axis for s in stats], axis=0)))
        print(f"Min footprint:  {min(s.footprint_margin_m for s in stats):.4f}")
        print(f"Max pos err:    {max(s.max_position_error_m for s in stats):.4f}")
    return 0 if stats and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
