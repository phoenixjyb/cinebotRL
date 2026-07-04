"""Export a recorded-trajectory manifest as a GIK teacher-label request.

The output is a compact contract between CineBotRL and the MATLAB/GIK side:
it lists the exact trajectory files, timing assumptions, reset/start-window
settings, and simple path statistics for the distribution that needs labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("trajectoryToLearn/stage1_recovery/manifest.txt"),
        help="Manifest listing trajectory JSON files. Relative paths resolve from --project-root.",
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--waypoint-dt", type=float, default=0.1)
    parser.add_argument("--min-duration", type=float, default=5.0)
    parser.add_argument("--start-waypoint-min-fraction", type=float, default=0.25)
    parser.add_argument("--start-waypoint-max-fraction", type=float, default=0.70)
    parser.add_argument("--reset-anchor-target-blend", type=float, default=0.35)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def resolve_path(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else base / path


def read_manifest(manifest_path: Path, project_root: Path) -> list[Path]:
    require(manifest_path.exists(), f"manifest not found: {manifest_path}")
    entries: list[Path] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            entries.append(resolve_path(Path(line), project_root))
    require(entries, f"manifest contains no entries: {manifest_path}")
    return entries


def load_positions(path: Path) -> tuple[list[dict[str, Any]], list[list[float]]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    poses = data.get("poses") if isinstance(data, dict) else data
    require(isinstance(poses, list) and poses, f"no poses found in {path}")
    positions = [[float(v) for v in pose["position"]] for pose in poses]
    return poses, positions


def path_length(positions: list[list[float]]) -> float:
    total = 0.0
    for prev, curr in zip(positions, positions[1:]):
        dx = curr[0] - prev[0]
        dy = curr[1] - prev[1]
        dz = curr[2] - prev[2]
        total += math.sqrt(dx * dx + dy * dy + dz * dz)
    return total


def start_index_bounds(length: int, min_fraction: float, max_fraction: float) -> tuple[int, int]:
    require(length > 0, "trajectory length must be positive")
    lo = max(0.0, min(1.0, min_fraction))
    hi = max(0.0, min(1.0, max_fraction))
    if hi < lo:
        lo, hi = hi, lo
    last = length - 1
    min_idx = int(round(lo * last))
    max_idx = int(round(hi * last))
    return min_idx, max(min_idx, min(max_idx, last))


def stats_for(path: Path, project_root: Path, waypoint_dt: float, min_fraction: float, max_fraction: float) -> dict[str, Any]:
    poses, positions = load_positions(path)
    length = len(positions)
    duration_s = length * waypoint_dt
    start = positions[0]
    end = positions[-1]
    x_vals = [p[0] for p in positions]
    y_vals = [p[1] for p in positions]
    z_vals = [p[2] for p in positions]
    dx_total = end[0] - start[0]
    dy_total = end[1] - start[1]
    min_start_idx, max_start_idx = start_index_bounds(length, min_fraction, max_fraction)
    rel_path = path.relative_to(project_root).as_posix() if path.is_relative_to(project_root) else str(path)
    category = path.parent.name
    return {
        "trajectory_path": rel_path,
        "category": category,
        "length": length,
        "duration_s": round(duration_s, 6),
        "waypoint_dt_s": waypoint_dt,
        "start_waypoint_min_idx": min_start_idx,
        "start_waypoint_max_idx": max_start_idx,
        "start_position": [round(float(v), 6) for v in start],
        "end_position": [round(float(v), 6) for v in end],
        "position_min": [round(min(axis), 6) for axis in (x_vals, y_vals, z_vals)],
        "position_max": [round(max(axis), 6) for axis in (x_vals, y_vals, z_vals)],
        "horizontal_displacement_m": round(math.sqrt(dx_total * dx_total + dy_total * dy_total), 6),
        "horizontal_range_m": round(
            math.sqrt((max(x_vals) - min(x_vals)) ** 2 + (max(y_vals) - min(y_vals)) ** 2),
            6,
        ),
        "path_length_m": round(path_length(positions), 6),
        "first_orientation_xyzw": [round(float(v), 6) for v in poses[0]["orientation"]],
        "last_orientation_xyzw": [round(float(v), 6) for v in poses[-1]["orientation"]],
    }


def write_csv(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trajectory_path",
        "category",
        "length",
        "duration_s",
        "waypoint_dt_s",
        "start_waypoint_min_idx",
        "start_waypoint_max_idx",
        "horizontal_displacement_m",
        "horizontal_range_m",
        "path_length_m",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            writer.writerow({name: item[name] for name in fieldnames})


def category_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        category = str(item["category"])
        counts[category] = counts.get(category, 0) + 1
    return dict(sorted(counts.items()))


def main() -> int:
    args = parse_args()
    require(args.waypoint_dt > 0.0, "--waypoint-dt must be positive")
    require(args.min_duration >= 0.0, "--min-duration must be non-negative")
    project_root = args.project_root.resolve()
    manifest_path = resolve_path(args.manifest, project_root).resolve()
    entries = read_manifest(manifest_path, project_root)

    items = [
        stats_for(path.resolve(), project_root, args.waypoint_dt, args.start_waypoint_min_fraction, args.start_waypoint_max_fraction)
        for path in entries
    ]
    items = [item for item in items if float(item["duration_s"]) >= args.min_duration]
    require(items, "all manifest entries were rejected by --min-duration")

    durations = [float(item["duration_s"]) for item in items]
    path_lengths = [float(item["path_length_m"]) for item in items]
    payload = {
        "schema": "cinebotrl_gik_teacher_request_v1",
        "source_manifest": manifest_path.relative_to(project_root).as_posix()
        if manifest_path.is_relative_to(project_root)
        else str(manifest_path),
        "waypoint_dt_s": args.waypoint_dt,
        "min_duration_s": args.min_duration,
        "start_waypoint_min_fraction": args.start_waypoint_min_fraction,
        "start_waypoint_max_fraction": args.start_waypoint_max_fraction,
        "reset_anchor_target_blend": args.reset_anchor_target_blend,
        "action_contract": "sim_6joint_gimbal_v1",
        "expected_action_order": [
            "joint6_arm_yaw",
            "joint5_arm_pitch",
            "joint4_elbow_pitch",
            "joint3_gimbal_yaw",
            "joint2_gimbal_roll",
            "joint1_gimbal_pitch",
            "base_vx",
            "base_vy",
            "base_wz",
        ],
        "notes": [
            "Generate teacher labels for these exact trajectories before PPO warm-start on stage1_recovery.",
            "Use the same waypoint_dt and start waypoint window when deriving progress-indexed labels.",
            "For DJI RS4/RS5 deployment, keep physical gimbal attitude handling separate from this sim contract.",
        ],
        "summary": {
            "count": len(items),
            "category_counts": category_counts(items),
            "duration_min_s": min(durations),
            "duration_max_s": max(durations),
            "path_length_min_m": min(path_lengths),
            "path_length_max_m": max(path_lengths),
        },
        "items": items,
    }

    output_json = resolve_path(args.output_json, project_root)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if args.output_csv:
        write_csv(resolve_path(args.output_csv, project_root), items)

    print(f"Exported {len(items)} teacher request trajectories")
    print(f"  manifest: {payload['source_manifest']}")
    print(f"  categories: {payload['summary']['category_counts']}")
    print(f"  output_json: {output_json}")
    if args.output_csv:
        print(f"  output_csv: {resolve_path(args.output_csv, project_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
