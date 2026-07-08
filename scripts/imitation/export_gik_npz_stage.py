#!/usr/bin/env python3
"""Export accepted GIK/ARCore NPZ teachers as an Isaac trajectory stage.

The accepted teacher NPZ files contain sparse target poses and action labels.
Isaac's MultiTrajectoryLoader consumes JSON pose lists at a fixed waypoint dt
(0.1s by default), so this exporter resamples each sparse target sequence to
the manifest duration before writing `trajectoryToLearn/<stage>/manifest.txt`.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo-dir",
        type=Path,
        default=Path("data/gik_offline_teachers_20260701_142322/accepted_npz"),
        help="Directory containing accepted teacher NPZ files and manifest.",
    )
    parser.add_argument("--manifest", default="manifest_no_obstacle79.json")
    parser.add_argument(
        "--stage",
        default="stage_gik_no_obstacle79_nominal",
        help="Output stage name under trajectoryToLearn/.",
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--waypoint-dt", type=float, default=0.1)
    parser.add_argument("--min-duration", type=float, default=5.0)
    parser.add_argument("--scenario", default="no_obstacle")
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the output stage directory before writing.",
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def resolve_npz_path(item: dict[str, Any], demo_dir: Path) -> Path:
    raw = Path(str(item["output_npz"]))
    candidates = []
    if raw.exists():
        candidates.append(raw)
    candidates.append(demo_dir / raw.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"missing npz for manifest item: {raw}")


def normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = quat.astype(np.float64, copy=True)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    quat = quat / np.maximum(norm, 1e-12)
    sign = np.where(quat[:, :1] < 0.0, -1.0, 1.0)
    return quat * sign


def resample_positions(values: np.ndarray, src_t: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
    return np.stack(
        [np.interp(dst_t, src_t, values[:, axis]) for axis in range(values.shape[1])],
        axis=1,
    ).astype(np.float32)


def resample_quats_wxyz(quat: np.ndarray, src_t: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
    quat = normalize_quat_wxyz(quat)
    # Keep neighbouring quaternions on the same hemisphere before interpolation.
    for idx in range(1, quat.shape[0]):
        if float(np.dot(quat[idx - 1], quat[idx])) < 0.0:
            quat[idx] *= -1.0
    interp = np.stack(
        [np.interp(dst_t, src_t, quat[:, axis]) for axis in range(quat.shape[1])],
        axis=1,
    )
    return normalize_quat_wxyz(interp).astype(np.float32)


def safe_float_list(values: np.ndarray) -> list[float]:
    return [float(x) for x in values.tolist()]


def write_text_lf(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def stage_file_name(index: int, item: dict[str, Any], npz_path: Path) -> str:
    metadata = item.get("teacher_metadata") or {}
    video_id = str(metadata.get("video_id") or "")
    episode = int(metadata.get("episode_index") or index + 1)
    if video_id:
        return f"{episode:04d}_{video_id}_{npz_path.stem[:10]}.json"
    return f"{index + 1:04d}_{npz_path.stem}.json"


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    demo_dir = args.demo_dir
    if not demo_dir.is_absolute():
        demo_dir = project_root / demo_dir
    manifest_path = demo_dir / args.manifest
    require(manifest_path.exists(), f"missing manifest: {manifest_path}")
    require(args.waypoint_dt > 0.0, "--waypoint-dt must be positive")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = manifest.get("items", [])
    require(items, "manifest contains no items")

    selected: list[tuple[dict[str, Any], Path]] = []
    for item in items:
        metadata = item.get("teacher_metadata") or {}
        if args.scenario and metadata.get("scenario") != args.scenario:
            continue
        if float(item.get("duration_s", 0.0)) < args.min_duration:
            continue
        selected.append((item, resolve_npz_path(item, demo_dir)))
        if args.max_trajectories is not None and len(selected) >= args.max_trajectories:
            break
    require(selected, "no manifest items selected")

    stage_dir = project_root / "trajectoryToLearn" / args.stage
    if args.clean and stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[Path] = []
    reset_offsets: list[np.ndarray] = []
    summaries: list[dict[str, Any]] = []

    for index, (item, npz_path) in enumerate(selected):
        with np.load(npz_path, allow_pickle=False) as data:
            target_pos = data["target_pos"].astype(np.float32)
            target_quat = data["target_quat_wxyz"].astype(np.float32)
            q_current = data["q_current"].astype(np.float32)
        require(target_pos.ndim == 2 and target_pos.shape[1] == 3, f"bad target_pos shape in {npz_path}")
        require(target_quat.ndim == 2 and target_quat.shape[1] == 4, f"bad target_quat shape in {npz_path}")
        require(target_pos.shape[0] == target_quat.shape[0], f"pose/quaternion length mismatch in {npz_path}")
        require(target_pos.shape[0] >= 2, f"need at least two target poses in {npz_path}")

        duration_s = float(item["duration_s"])
        dst_count = max(2, int(math.ceil(duration_s / args.waypoint_dt)) + 1)
        src_t = np.linspace(0.0, duration_s, target_pos.shape[0], dtype=np.float64)
        dst_t = np.linspace(0.0, duration_s, dst_count, dtype=np.float64)
        pos_out = resample_positions(target_pos, src_t, dst_t)
        quat_wxyz_out = resample_quats_wxyz(target_quat, src_t, dst_t)

        # JSON loader expects xyzw and converts back to wxyz internally.
        quat_xyzw_out = np.column_stack(
            [quat_wxyz_out[:, 1], quat_wxyz_out[:, 2], quat_wxyz_out[:, 3], quat_wxyz_out[:, 0]]
        )
        poses = [
            {
                "position": safe_float_list(pos),
                "orientation": safe_float_list(quat),
            }
            for pos, quat in zip(pos_out, quat_xyzw_out)
        ]

        reset_offsets.append(target_pos[0, :2] - q_current[0, :2])
        output_name = stage_file_name(index, item, npz_path)
        output_path = stage_dir / output_name
        payload = {
            "poses": poses,
            "metadata": {
                "source": "accepted_gik_arcore_npz",
                "source_npz": npz_path.name,
                "source_manifest": str(manifest_path.relative_to(project_root)),
                "duration_s": duration_s,
                "waypoint_dt": args.waypoint_dt,
                "source_pose_count": int(target_pos.shape[0]),
                "resampled_pose_count": int(len(poses)),
                "initial_arm_joint_pos": safe_float_list(q_current[0, 3:9]),
                "scenario": (item.get("teacher_metadata") or {}).get("scenario"),
                "video_id": (item.get("teacher_metadata") or {}).get("video_id"),
                "quality_status": (item.get("teacher_metadata") or {}).get("quality_status"),
            },
        }
        write_text_lf(output_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        manifest_entries.append(output_path)
        summaries.append(
            {
                "file": output_path.name,
                "duration_s": duration_s,
                "source_pose_count": int(target_pos.shape[0]),
                "resampled_pose_count": int(len(poses)),
                "z_min": float(np.min(pos_out[:, 2])),
                "z_max": float(np.max(pos_out[:, 2])),
            }
        )

    stage_manifest = stage_dir / "manifest.txt"
    with stage_manifest.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"# Generated by scripts/imitation/export_gik_npz_stage.py for {args.stage}\n")
        f.write(f"# Source manifest: {manifest_path.relative_to(project_root).as_posix()}\n")
        for path in manifest_entries:
            f.write(f"{path.relative_to(project_root).as_posix()}\n")

    offsets = np.stack(reset_offsets, axis=0)
    reset_config = {
        "reset_base_x_offset": float(np.median(offsets[:, 0])),
        "reset_base_y_offset": float(np.median(offsets[:, 1])),
        "reset_arm_to_trajectory_metadata": True,
        "notes": "Use with --reset_base_to_trajectory_start for the accepted GIK/ARCore nominal stage.",
    }
    write_text_lf(stage_dir / "reset_config.json", json.dumps(reset_config, ensure_ascii=False, indent=2) + "\n")

    summary = {
        "stage": args.stage,
        "source_manifest": str(manifest_path.relative_to(project_root)),
        "num_trajectories": len(summaries),
        "waypoint_dt": args.waypoint_dt,
        "min_duration": args.min_duration,
        "reset_config": reset_config,
        "trajectories": summaries,
    }
    write_text_lf(stage_dir / "export_summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    durations = [item["duration_s"] for item in summaries]
    counts = [item["resampled_pose_count"] for item in summaries]
    print(f"Stage:          {stage_dir.relative_to(project_root)}")
    print(f"Manifest:       {stage_manifest.relative_to(project_root)}")
    print(f"Trajectories:   {len(summaries)}")
    print(f"Duration range: {min(durations):.2f}-{max(durations):.2f}s")
    print(f"Pose counts:    {min(counts)}-{max(counts)}")
    print(
        "Reset offset:   "
        f"x={reset_config['reset_base_x_offset']:.6f}, "
        f"y={reset_config['reset_base_y_offset']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
