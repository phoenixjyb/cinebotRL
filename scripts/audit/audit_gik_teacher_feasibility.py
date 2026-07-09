#!/usr/bin/env python3
"""Audit GIK teacher/action-contract feasibility for CineBotRL trajectories.

This is a layered audit.  The default implementation is intentionally offline:
it reads the exported stage JSON files and matching GIK NPZ teacher demos, then
reports static timing/action/FK-label consistency metrics.  Expensive Isaac
open-loop/action replay can be added later without changing the output schema.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


ACTION_DIM = 9
BASE_ACTION_SLICE = slice(6, 9)
GIMBAL_ACTION_INDICES = [3, 4, 5]
DEFAULT_WAYPOINT_DT = 0.1

CSV_FIELDS = [
    "trajectory_id",
    "trajectory_path",
    "duration_s",
    "num_waypoints",
    "tag",
    "tag_reason",
    "fk_replay_ee_pos_mean_m",
    "fk_replay_ee_pos_p95_m",
    "fk_replay_ee_ori_mean_deg",
    "action_replay_ee_pos_mean_m",
    "action_replay_ee_pos_p95_m",
    "base_vy_abs_mean",
    "base_vy_abs_max",
    "base_vy_required",
    "base_vy_sign_suspect",
    "base_velocity_violation_pct",
    "base_acceleration_violation_pct",
    "base_jerk_violation_pct",
    "joint_limit_saturation_pct",
    "gimbal_saturation_pct",
    "gimbal_rate_saturation_pct",
    "orientation_discontinuity_count",
    "ee_velocity_spike_count",
    "ee_acceleration_spike_count",
    "reachability_margin_min_m",
    "final_target_reachable",
    "closed_loop_teacher_success",
    "closed_loop_teacher_error_mean_m",
    "notes",
]


@dataclass
class AuditThresholds:
    clean_fk_mean_m: float = 0.10
    clean_fk_p95_m: float = 0.25
    repair_fk_mean_m: float = 0.25
    repair_fk_p95_m: float = 0.60
    reject_fk_mean_m: float = 0.50
    reject_fk_p95_m: float = 1.00
    clean_saturation_pct: float = 5.0
    repair_saturation_pct: float = 35.0
    reject_saturation_pct: float = 65.0
    clean_orientation_deg: float = 10.0
    reject_orientation_deg: float = 45.0
    base_vy_required_m: float = 0.05
    base_sign_corr_suspect: float = -0.20
    velocity_spike_mps: float = 3.0
    acceleration_spike_mps2: float = 6.0
    max_base_accel_norm_per_s: float = 8.0
    max_base_jerk_norm_per_s2: float = 40.0
    curriculum_late_duration_s: float = 25.0
    curriculum_late_waypoints: int = 250


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, help="Stage name under trajectoryToLearn/.")
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--demo_dir", default=Path("data/gik_ik_demos"), type=Path)
    parser.add_argument("--max_trajectories", type=int, default=None)
    parser.add_argument("--waypoint_dt", type=float, default=DEFAULT_WAYPOINT_DT)
    parser.add_argument("--same_action_contract_as_env", action="store_true")
    parser.add_argument("--report_fk_replay", action="store_true")
    parser.add_argument("--report_action_replay", action="store_true")
    parser.add_argument("--report_base_vy_usage", action="store_true")
    parser.add_argument("--report_gimbal_saturation", action="store_true")
    parser.add_argument("--report_timing", action="store_true")
    parser.add_argument("--report_frame_invariance", action="store_true")
    return parser.parse_args()


def read_manifest(stage_dir: Path) -> list[Path]:
    manifest = stage_dir / "manifest.txt"
    if not manifest.exists():
        raise FileNotFoundError(f"missing stage manifest: {manifest}")
    paths: list[Path] = []
    for raw_line in manifest.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        path = Path(line)
        if not path.is_absolute():
            path = Path.cwd() / path
        paths.append(path)
    if not paths:
        raise RuntimeError(f"manifest has no trajectory paths: {manifest}")
    return paths


def trajectory_id_from_path(path: Path) -> str:
    parts = path.stem.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:3])
    return path.stem


def case_id_from_path(path: Path) -> str:
    return path.stem.split("_", 1)[0]


def load_stage_poses(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    poses = data.get("poses") if isinstance(data, dict) else data
    if not poses:
        raise ValueError(f"no poses in {path}")
    positions = np.asarray([[float(x) for x in pose["position"]] for pose in poses], dtype=np.float64)
    # Stage JSON stores xyzw.
    quat_xyzw = np.asarray([[float(x) for x in pose["orientation"]] for pose in poses], dtype=np.float64)
    metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
    return positions, quat_xyzw, metadata


def find_npz(demo_dir: Path, trajectory_path: Path, metadata: dict[str, Any]) -> Path | None:
    candidates: list[Path] = []
    source_npz = metadata.get("source_npz")
    if source_npz:
        candidates.append(Path(source_npz))
        candidates.append(demo_dir / Path(source_npz).name)
    stem = trajectory_id_from_path(trajectory_path)
    candidates.append(demo_dir / f"{stem}.npz")
    candidates.append(demo_dir / f"{trajectory_path.stem}.npz")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def quat_wxyz_normalize(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return quat / np.maximum(norm, 1e-12)


def quat_angle_error_deg(q1_wxyz: np.ndarray, q2_wxyz: np.ndarray) -> np.ndarray:
    q1 = quat_wxyz_normalize(q1_wxyz.astype(np.float64))
    q2 = quat_wxyz_normalize(q2_wxyz.astype(np.float64))
    dot = np.abs(np.sum(q1 * q2, axis=-1))
    dot = np.clip(dot, -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def orientation_discontinuity_count(quat_wxyz: np.ndarray, threshold_deg: float = 45.0) -> int:
    if len(quat_wxyz) < 2:
        return 0
    err = quat_angle_error_deg(quat_wxyz[:-1], quat_wxyz[1:])
    return int(np.count_nonzero(err > threshold_deg))


def safe_percentile(values: np.ndarray, percentile: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.percentile(values.astype(np.float64), percentile))


def safe_mean(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.mean(values.astype(np.float64)))


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def pct(mask: np.ndarray) -> float | None:
    if mask.size == 0:
        return None
    return float(np.mean(mask.astype(np.float64)) * 100.0)


def compute_spikes(positions: np.ndarray, dt: np.ndarray, velocity_threshold: float, acceleration_threshold: float) -> tuple[int, int]:
    if len(positions) < 3:
        return 0, 0
    dt = np.asarray(dt, dtype=np.float64).reshape(-1)
    if len(dt) < len(positions) - 1:
        dt = np.full((len(positions) - 1,), DEFAULT_WAYPOINT_DT, dtype=np.float64)
    dt = np.maximum(dt[: len(positions) - 1], 1e-6)
    velocity = np.diff(positions, axis=0) / dt[:, None]
    speed = np.linalg.norm(velocity, axis=1)
    v_spikes = int(np.count_nonzero(speed > velocity_threshold))
    if len(velocity) < 2:
        return v_spikes, 0
    acc_dt = np.maximum(dt[1:], 1e-6)
    acceleration = np.diff(velocity, axis=0) / acc_dt[:, None]
    acc_norm = np.linalg.norm(acceleration, axis=1)
    a_spikes = int(np.count_nonzero(acc_norm > acceleration_threshold))
    return v_spikes, a_spikes


def base_motion_stats(base_actions: np.ndarray, dt: np.ndarray, thresholds: AuditThresholds) -> tuple[float | None, float | None]:
    if base_actions.size == 0 or len(base_actions) < 2:
        return None, None
    dt = np.asarray(dt, dtype=np.float64).reshape(-1)
    if len(dt) < len(base_actions):
        dt = np.full((len(base_actions),), DEFAULT_WAYPOINT_DT, dtype=np.float64)
    dt_step = np.maximum(dt[1: len(base_actions)], 1e-6)
    acc = np.diff(base_actions.astype(np.float64), axis=0) / dt_step[:, None]
    acc_norm = np.linalg.norm(acc, axis=1)
    acc_violation = pct(acc_norm > thresholds.max_base_accel_norm_per_s)
    if len(acc) < 2:
        return acc_violation, None
    jerk_dt = np.maximum(dt[2: len(base_actions)], 1e-6)
    jerk = np.diff(acc, axis=0) / jerk_dt[:, None]
    jerk_norm = np.linalg.norm(jerk, axis=1)
    return acc_violation, pct(jerk_norm > thresholds.max_base_jerk_norm_per_s2)


def sign_suspect(positions: np.ndarray, base_vy: np.ndarray, threshold: AuditThresholds) -> bool:
    if len(positions) < 3 or len(base_vy) < 2:
        return False
    dy = np.diff(positions[: len(base_vy) + 1, 1])
    n = min(len(dy), len(base_vy))
    if n < 3 or np.std(dy[:n]) < 1e-9 or np.std(base_vy[:n]) < 1e-9:
        return False
    corr = float(np.corrcoef(dy[:n], base_vy[:n])[0, 1])
    return math.isfinite(corr) and corr < threshold.base_sign_corr_suspect


def audit_one(trajectory_path: Path, demo_dir: Path, thresholds: AuditThresholds, args: argparse.Namespace) -> dict[str, Any]:
    notes: list[str] = []
    positions, quat_xyzw, metadata = load_stage_poses(trajectory_path)
    waypoint_dt = float(metadata.get("waypoint_dt") or args.waypoint_dt)
    duration_s = finite_float(metadata.get("duration_s")) or float(len(positions) * waypoint_dt)
    npz_path = find_npz(demo_dir, trajectory_path, metadata)

    row: dict[str, Any] = {field: None for field in CSV_FIELDS}
    row.update(
        {
            "trajectory_id": case_id_from_path(trajectory_path),
            "trajectory_path": str(trajectory_path.relative_to(Path.cwd()) if trajectory_path.is_absolute() else trajectory_path),
            "duration_s": duration_s,
            "num_waypoints": int(len(positions)),
            "base_vy_required": bool(np.ptp(positions[:, 1]) > thresholds.base_vy_required_m),
            "orientation_discontinuity_count": orientation_discontinuity_count(
                np.column_stack([quat_xyzw[:, 3], quat_xyzw[:, 0], quat_xyzw[:, 1], quat_xyzw[:, 2]])
            ),
        }
    )

    dt_for_pose = np.full((max(len(positions) - 1, 0),), waypoint_dt, dtype=np.float64)
    v_spikes, a_spikes = compute_spikes(
        positions,
        dt_for_pose,
        thresholds.velocity_spike_mps,
        thresholds.acceleration_spike_mps2,
    )
    row["ee_velocity_spike_count"] = v_spikes
    row["ee_acceleration_spike_count"] = a_spikes

    if npz_path is None:
        notes.append("missing_matching_npz")
        tag, reason = "reject", "missing matching GIK NPZ demo"
        row["tag"] = tag
        row["tag_reason"] = reason
        row["notes"] = "; ".join(notes)
        return row

    with np.load(npz_path, allow_pickle=False) as npz:
        actions = np.asarray(npz["actions"], dtype=np.float64) if "actions" in npz else np.empty((0, ACTION_DIM))
        action_valid_mask = np.asarray(npz["action_valid_mask"], dtype=bool) if "action_valid_mask" in npz else np.ones_like(actions, dtype=bool)
        arm_valid_mask = (
            np.asarray(npz["arm_action_valid_mask"], dtype=bool)
            if "arm_action_valid_mask" in npz
            else action_valid_mask[:, :6] if action_valid_mask.size else np.empty((0, 6), dtype=bool)
        )
        base_valid_mask = (
            np.asarray(npz["base_action_valid_mask"], dtype=bool)
            if "base_action_valid_mask" in npz
            else action_valid_mask[:, BASE_ACTION_SLICE] if action_valid_mask.size else np.empty((0, 3), dtype=bool)
        )
        target_pos = np.asarray(npz["target_pos"], dtype=np.float64) if "target_pos" in npz else np.empty((0, 3))
        actual_ee_pos = np.asarray(npz["actual_ee_pos"], dtype=np.float64) if "actual_ee_pos" in npz else np.empty((0, 3))
        target_quat = np.asarray(npz["target_quat_wxyz"], dtype=np.float64) if "target_quat_wxyz" in npz else np.empty((0, 4))
        actual_quat = np.asarray(npz["actual_ee_quat_wxyz"], dtype=np.float64) if "actual_ee_quat_wxyz" in npz else np.empty((0, 4))
        dt = np.asarray(npz["dt"], dtype=np.float64).reshape(-1) if "dt" in npz else np.full((len(actions),), waypoint_dt)
        base_action_unclipped = (
            np.asarray(npz["base_action_unclipped"], dtype=np.float64)
            if "base_action_unclipped" in npz
            else actions[:, BASE_ACTION_SLICE] if actions.size else np.empty((0, 3))
        )
        arm_action_unclipped = (
            np.asarray(npz["arm_action_unclipped"], dtype=np.float64)
            if "arm_action_unclipped" in npz
            else actions[:, :6] if actions.size else np.empty((0, 6))
        )

    if target_pos.size and actual_ee_pos.size:
        n = min(len(target_pos), len(actual_ee_pos))
        pos_err = np.linalg.norm(target_pos[:n] - actual_ee_pos[:n], axis=1)
        row["fk_replay_ee_pos_mean_m"] = safe_mean(pos_err)
        row["fk_replay_ee_pos_p95_m"] = safe_percentile(pos_err, 95)
    else:
        notes.append("missing_target_or_actual_ee_pos")

    if target_quat.size and actual_quat.size:
        n = min(len(target_quat), len(actual_quat))
        row["fk_replay_ee_ori_mean_deg"] = safe_mean(quat_angle_error_deg(target_quat[:n], actual_quat[:n]))
    else:
        notes.append("missing_target_or_actual_ee_quat")

    if actions.size:
        base_vy = actions[:, 7]
        row["base_vy_abs_mean"] = float(np.mean(np.abs(base_vy)))
        row["base_vy_abs_max"] = float(np.max(np.abs(base_vy)))
        row["base_vy_sign_suspect"] = bool(row["base_vy_required"] and sign_suspect(positions, base_vy, thresholds))
        row["joint_limit_saturation_pct"] = pct(~arm_valid_mask) if arm_valid_mask.size else pct(np.abs(actions[:, :6]) >= 0.999)
        row["gimbal_saturation_pct"] = pct(~arm_valid_mask[:, GIMBAL_ACTION_INDICES]) if arm_valid_mask.size else pct(np.abs(actions[:, GIMBAL_ACTION_INDICES]) >= 0.999)
        if len(arm_action_unclipped) >= 2:
            gimbal_dt = np.maximum(dt[1: len(arm_action_unclipped)], 1e-6)
            gimbal_rate = np.abs(np.diff(arm_action_unclipped[:, GIMBAL_ACTION_INDICES], axis=0) / gimbal_dt[:, None])
            row["gimbal_rate_saturation_pct"] = pct(gimbal_rate > 5.0)
        row["base_velocity_violation_pct"] = pct(~base_valid_mask) if base_valid_mask.size else pct(np.abs(actions[:, BASE_ACTION_SLICE]) > 1.0)
        acc_vio, jerk_vio = base_motion_stats(base_action_unclipped, dt, thresholds)
        row["base_acceleration_violation_pct"] = acc_vio
        row["base_jerk_violation_pct"] = jerk_vio
    else:
        notes.append("missing_actions")

    if args.report_action_replay:
        notes.append("action_replay_not_implemented_offline")
    if args.report_frame_invariance:
        notes.append("frame_invariance_static_only")
    if args.same_action_contract_as_env:
        notes.append("assumed_env_action_contract_9d_sim_6joint_gimbal_v1")

    tag, reason = classify(row, thresholds)
    row["tag"] = tag
    row["tag_reason"] = reason
    row["notes"] = "; ".join(notes)
    return row


def classify(row: dict[str, Any], thresholds: AuditThresholds) -> tuple[str, str]:
    duration = finite_float(row.get("duration_s")) or 0.0
    fk_mean = finite_float(row.get("fk_replay_ee_pos_mean_m"))
    fk_p95 = finite_float(row.get("fk_replay_ee_pos_p95_m"))
    ori_mean = finite_float(row.get("fk_replay_ee_ori_mean_deg"))
    joint_sat = finite_float(row.get("joint_limit_saturation_pct")) or 0.0
    gimbal_sat = finite_float(row.get("gimbal_saturation_pct")) or 0.0
    base_vio = finite_float(row.get("base_velocity_violation_pct")) or 0.0
    ori_jumps = int(row.get("orientation_discontinuity_count") or 0)
    v_spikes = int(row.get("ee_velocity_spike_count") or 0)
    a_spikes = int(row.get("ee_acceleration_spike_count") or 0)
    num_waypoints = int(row.get("num_waypoints") or 0)

    if duration < 5.0:
        return "reject", "duration shorter than 5s"
    if fk_mean is None or fk_p95 is None:
        return "reject", "missing FK replay target/actual metrics"
    if fk_mean > thresholds.reject_fk_mean_m or fk_p95 > thresholds.reject_fk_p95_m:
        return "reject", f"FK replay error too high mean={fk_mean:.3f} p95={fk_p95:.3f}"
    if max(joint_sat, gimbal_sat) > thresholds.reject_saturation_pct:
        return "reject", f"teacher saturation too high joint={joint_sat:.1f}% gimbal={gimbal_sat:.1f}%"
    if ori_mean is not None and ori_mean > thresholds.reject_orientation_deg:
        return "reject", f"orientation error too high mean={ori_mean:.1f}deg"
    if ori_jumps > 0:
        return "repairable", f"orientation discontinuities={ori_jumps}"
    if v_spikes > 0 or a_spikes > 0:
        return "repairable", f"trajectory spikes velocity={v_spikes} acceleration={a_spikes}"
    if (
        fk_mean <= thresholds.clean_fk_mean_m
        and fk_p95 <= thresholds.clean_fk_p95_m
        and max(joint_sat, gimbal_sat, base_vio) <= thresholds.clean_saturation_pct
        and (ori_mean is None or ori_mean <= thresholds.clean_orientation_deg)
    ):
        if duration >= thresholds.curriculum_late_duration_s or num_waypoints >= thresholds.curriculum_late_waypoints:
            return "curriculum_late", "clean replay but long trajectory"
        return "clean", "low replay error and low saturation"
    if (
        fk_mean <= thresholds.repair_fk_mean_m
        and fk_p95 <= thresholds.repair_fk_p95_m
        and max(joint_sat, gimbal_sat) <= thresholds.repair_saturation_pct
    ):
        if duration >= thresholds.curriculum_late_duration_s or num_waypoints >= thresholds.curriculum_late_waypoints:
            return "curriculum_late", "feasible but long/difficult"
        return "repairable", "moderate replay error or saturation"
    return "curriculum_late", "trackable labels but difficult under current thresholds"


def jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: jsonable(row.get(field)) for field in CSV_FIELDS})


def write_tagged_manifests(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    by_tag: dict[str, list[str]] = {tag: [] for tag in ("clean", "repairable", "curriculum_late", "reject")}
    for row in rows:
        by_tag[str(row["tag"])].append(str(row["trajectory_path"]))
    for tag, paths in by_tag.items():
        (out_dir / f"tagged_manifest_{tag}.txt").write_text(
            "\n".join(paths) + ("\n" if paths else ""),
            encoding="utf-8",
        )


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace, thresholds: AuditThresholds) -> dict[str, Any]:
    counts = Counter(str(row["tag"]) for row in rows)
    numeric_fields = [
        "fk_replay_ee_pos_mean_m",
        "fk_replay_ee_pos_p95_m",
        "fk_replay_ee_ori_mean_deg",
        "base_vy_abs_mean",
        "base_vy_abs_max",
        "joint_limit_saturation_pct",
        "gimbal_saturation_pct",
        "base_velocity_violation_pct",
    ]
    metrics: dict[str, Any] = {}
    for field in numeric_fields:
        values = [finite_float(row.get(field)) for row in rows]
        values = [value for value in values if value is not None]
        if values:
            metrics[field] = {
                "mean": statistics.fmean(values),
                "p50": statistics.median(values),
                "max": max(values),
            }
    return {
        "stage": args.stage,
        "num_trajectories": len(rows),
        "tag_counts": dict(counts),
        "thresholds": asdict(thresholds),
        "metrics": metrics,
        "unsupported_layers": {
            "action_replay": "not implemented in offline audit; reserved output fields are null",
            "closed_loop_teacher": "not implemented in offline audit; reserved output fields are null",
            "reachability_margin": "not implemented in this first layered script",
        },
        "rows": rows,
    }


def main() -> int:
    args = parse_args()
    stage_dir = Path("trajectoryToLearn") / args.stage
    demo_dir = args.demo_dir
    thresholds = AuditThresholds()
    paths = read_manifest(stage_dir)
    if args.max_trajectories is not None:
        paths = paths[: args.max_trajectories]

    rows = [audit_one(path, demo_dir, thresholds, args) for path in paths]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "per_trajectory.csv", rows)
    write_tagged_manifests(args.out_dir, rows)
    summary = summarize(rows, args, thresholds)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=jsonable) + "\n",
        encoding="utf-8",
    )

    print(f"[audit] stage={args.stage}")
    print(f"[audit] trajectories={len(rows)}")
    print(f"[audit] tags={summary['tag_counts']}")
    print(f"[audit] wrote {args.out_dir / 'summary.json'}")
    print(f"[audit] wrote {args.out_dir / 'per_trajectory.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
