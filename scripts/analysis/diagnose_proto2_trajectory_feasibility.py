"""Diagnose Proto2 recorded trajectory feasibility without running RL.

This script mirrors the current MobileMM reset geometry and reachability-map
query path for recorded trajectories. It is intentionally deterministic: it
enumerates every possible randomized reset waypoint instead of sampling one.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from rl_platform.utils.reachability_map import ReachabilityMap  # noqa: E402


ARM_MOUNT_TRANSLATION = np.array([0.16, 0.0, 0.9465], dtype=np.float64)
RESET_BASE_X_OFFSET = 0.4415
RESET_BASE_Y_OFFSET = 0.2405
ARM_LOWER_SAFE = np.array([-1.0, 0.55, -2.0, -1.0, -0.8, -0.8], dtype=np.float64)
ARM_UPPER_SAFE = np.array([1.0, 1.45, -0.4, 1.0, 0.8, 0.8], dtype=np.float64)


@dataclass
class WindowStats:
    start_idx: int
    samples: int
    reachable_pct: float
    unreachable_pct: float
    mean_workspace_distance_m: float
    p95_workspace_distance_m: float
    max_workspace_distance_m: float
    mean_base_target_xy_m: float
    max_base_target_xy_m: float
    ideal_base_displacement_max_m: float
    ideal_base_path_length_m: float
    ideal_base_speed_p95_mps: float
    qexample_safe_pct: float
    qexample_violation_pct: float
    qexample_violation_by_joint_pct: list[float]
    qexample_min: list[float]
    qexample_max: list[float]
    worst_sample_step: int
    worst_waypoint_idx: int
    worst_workspace_distance_m: float
    worst_target_world: list[float]
    worst_target_arm: list[float]
    worst_qexample: list[float]
    worst_qexample_violation: list[bool]


def _load_poses(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    poses = data.get("poses") if isinstance(data, dict) else data
    if not poses:
        raise ValueError(f"No poses found in {path}")

    positions: list[list[float]] = []
    orientations_wxyz: list[list[float]] = []
    for pose in poses:
        positions.append([float(v) for v in pose["position"]])
        xyzw = [float(v) for v in pose["orientation"]]
        orientations_wxyz.append([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])

    return np.asarray(positions, dtype=np.float64), np.asarray(orientations_wxyz, dtype=np.float64)


def _start_index_range(length: int, min_fraction: float, max_fraction: float) -> range:
    if length < 1:
        raise ValueError("Trajectory must contain at least one waypoint")
    min_fraction = max(0.0, min(1.0, min_fraction))
    max_fraction = max(0.0, min(1.0, max_fraction))
    if max_fraction < min_fraction:
        min_fraction, max_fraction = max_fraction, min_fraction

    last_idx = length - 1
    min_idx = int(round(min_fraction * last_idx))
    max_idx = int(round(max_fraction * last_idx))
    max_idx = max(min_idx, min(max_idx, last_idx))
    return range(min_idx, max_idx + 1)


def _quat_to_euler_xyz_deg(q_wxyz: np.ndarray) -> list[float]:
    w, x, y, z = [float(v) for v in q_wxyz]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]


def _simulate_target_samples(
    positions: np.ndarray,
    start_idx: int,
    control_dt: float,
    waypoint_dt: float,
    episode_length_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce recorded trajectory interpolation for one static-base episode."""
    length = positions.shape[0]
    num_steps = int(round(episode_length_s / control_dt))
    current_idx = start_idx
    time_accum = 0.0

    targets: list[np.ndarray] = []
    waypoint_indices: list[int] = []
    for _ in range(num_steps):
        next_idx = (current_idx + 1) % length
        alpha = min(max(time_accum / waypoint_dt, 0.0), 1.0)
        target = (1.0 - alpha) * positions[current_idx] + alpha * positions[next_idx]
        targets.append(target)
        waypoint_indices.append(current_idx)

        time_accum += control_dt
        steps_to_advance = int(math.floor(time_accum / waypoint_dt))
        if steps_to_advance > 0:
            current_idx = (current_idx + steps_to_advance) % length
            time_accum -= steps_to_advance * waypoint_dt

    return np.asarray(targets, dtype=np.float64), np.asarray(waypoint_indices, dtype=np.int64)


def _reset_base_pose(positions: np.ndarray, start_idx: int, blend: float) -> np.ndarray:
    start_anchor = positions[0]
    first_target = positions[start_idx]
    reset_anchor = (1.0 - blend) * start_anchor + blend * first_target
    return np.array(
        [
            reset_anchor[0] - RESET_BASE_X_OFFSET,
            reset_anchor[1] - RESET_BASE_Y_OFFSET,
            0.0,
        ],
        dtype=np.float64,
    )


def _world_to_arm_frame_np(target_world: np.ndarray, base_pose: np.ndarray) -> np.ndarray:
    rel = target_world - np.array([base_pose[0], base_pose[1], 0.0], dtype=np.float64)
    theta = base_pose[2]
    cos_t = math.cos(-theta)
    sin_t = math.sin(-theta)
    x_base = rel[:, 0] * cos_t - rel[:, 1] * sin_t
    y_base = rel[:, 0] * sin_t + rel[:, 1] * cos_t
    z_base = rel[:, 2]
    in_mount = np.stack([x_base, y_base, z_base], axis=1) - ARM_MOUNT_TRANSLATION
    return np.stack([in_mount[:, 1], -in_mount[:, 0], in_mount[:, 2]], axis=1)


def _round_list(values: np.ndarray, digits: int = 6) -> list[float]:
    return [round(float(v), digits) for v in values.tolist()]


def _analyze_window(
    reach_map: ReachabilityMap,
    positions: np.ndarray,
    start_idx: int,
    control_dt: float,
    waypoint_dt: float,
    episode_length_s: float,
    reset_anchor_target_blend: float,
    tolerance: float,
) -> WindowStats:
    targets_world, waypoint_indices = _simulate_target_samples(
        positions=positions,
        start_idx=start_idx,
        control_dt=control_dt,
        waypoint_dt=waypoint_dt,
        episode_length_s=episode_length_s,
    )
    base_pose = _reset_base_pose(positions, start_idx, reset_anchor_target_blend)
    targets_arm = _world_to_arm_frame_np(targets_world, base_pose)

    targets_arm_t = torch.tensor(targets_arm, dtype=torch.float32, device=reach_map.device)
    distances = reach_map.distance_to_workspace(targets_arm_t).detach().cpu().numpy()
    reachable = distances < tolerance
    configs, _ = reach_map.get_best_configs(targets_arm_t, tolerance=tolerance)
    qexample = configs.detach().cpu().numpy().astype(np.float64)
    q_violations = (qexample < ARM_LOWER_SAFE) | (qexample > ARM_UPPER_SAFE)
    q_safe = ~q_violations.any(axis=1)

    base_xy = base_pose[:2]
    base_target_xy = np.linalg.norm(targets_world[:, :2] - base_xy, axis=1)
    ideal_base_xy = targets_world[:, :2] - np.array([RESET_BASE_X_OFFSET, RESET_BASE_Y_OFFSET], dtype=np.float64)
    ideal_base_delta = ideal_base_xy - base_xy
    ideal_base_step_dist = np.linalg.norm(np.diff(ideal_base_xy, axis=0), axis=1)
    worst_idx = int(np.argmax(distances))

    return WindowStats(
        start_idx=start_idx,
        samples=int(len(targets_world)),
        reachable_pct=float(reachable.mean() * 100.0),
        unreachable_pct=float((~reachable).mean() * 100.0),
        mean_workspace_distance_m=float(distances.mean()),
        p95_workspace_distance_m=float(np.percentile(distances, 95)),
        max_workspace_distance_m=float(distances.max()),
        mean_base_target_xy_m=float(base_target_xy.mean()),
        max_base_target_xy_m=float(base_target_xy.max()),
        ideal_base_displacement_max_m=float(np.linalg.norm(ideal_base_delta, axis=1).max()),
        ideal_base_path_length_m=float(ideal_base_step_dist.sum()),
        ideal_base_speed_p95_mps=float(np.percentile(ideal_base_step_dist / control_dt, 95)),
        qexample_safe_pct=float(q_safe.mean() * 100.0),
        qexample_violation_pct=float((~q_safe).mean() * 100.0),
        qexample_violation_by_joint_pct=_round_list(q_violations.mean(axis=0) * 100.0, 3),
        qexample_min=_round_list(qexample.min(axis=0)),
        qexample_max=_round_list(qexample.max(axis=0)),
        worst_sample_step=worst_idx,
        worst_waypoint_idx=int(waypoint_indices[worst_idx]),
        worst_workspace_distance_m=float(distances[worst_idx]),
        worst_target_world=_round_list(targets_world[worst_idx]),
        worst_target_arm=_round_list(targets_arm[worst_idx]),
        worst_qexample=_round_list(qexample[worst_idx]),
        worst_qexample_violation=[bool(v) for v in q_violations[worst_idx].tolist()],
    )


def _summarize_windows(windows: list[WindowStats]) -> dict[str, Any]:
    if not windows:
        raise ValueError("No start windows to summarize")
    unreachable = np.asarray([w.unreachable_pct for w in windows], dtype=np.float64)
    p95_dist = np.asarray([w.p95_workspace_distance_m for w in windows], dtype=np.float64)
    max_dist = np.asarray([w.max_workspace_distance_m for w in windows], dtype=np.float64)
    q_violation = np.asarray([w.qexample_violation_pct for w in windows], dtype=np.float64)
    q_violation_by_joint = np.asarray([w.qexample_violation_by_joint_pct for w in windows], dtype=np.float64)
    q_min = np.asarray([w.qexample_min for w in windows], dtype=np.float64)
    q_max = np.asarray([w.qexample_max for w in windows], dtype=np.float64)
    ideal_disp = np.asarray([w.ideal_base_displacement_max_m for w in windows], dtype=np.float64)
    ideal_speed = np.asarray([w.ideal_base_speed_p95_mps for w in windows], dtype=np.float64)
    best_idx = int(np.argmin(unreachable))
    worst_idx = int(np.argmax(unreachable))
    return {
        "num_start_indices": len(windows),
        "start_idx_min": min(w.start_idx for w in windows),
        "start_idx_max": max(w.start_idx for w in windows),
        "unreachable_pct_mean": float(unreachable.mean()),
        "unreachable_pct_min": float(unreachable.min()),
        "unreachable_pct_max": float(unreachable.max()),
        "workspace_distance_p95_mean_m": float(p95_dist.mean()),
        "workspace_distance_max_mean_m": float(max_dist.mean()),
        "ideal_base_displacement_max_mean_m": float(ideal_disp.mean()),
        "ideal_base_displacement_max_min_m": float(ideal_disp.min()),
        "ideal_base_displacement_max_max_m": float(ideal_disp.max()),
        "ideal_base_speed_p95_mean_mps": float(ideal_speed.mean()),
        "qexample_violation_pct_mean": float(q_violation.mean()),
        "qexample_violation_by_joint_pct_mean": _round_list(q_violation_by_joint.mean(axis=0), 3),
        "qexample_min_over_windows": _round_list(q_min.min(axis=0)),
        "qexample_max_over_windows": _round_list(q_max.max(axis=0)),
        "best_start_by_unreachable": asdict(windows[best_idx]),
        "worst_start_by_unreachable": asdict(windows[worst_idx]),
    }


def _trajectory_summary(
    reach_map: ReachabilityMap,
    trajectory_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    positions, orientations_wxyz = _load_poses(trajectory_path)
    start_indices = list(
        _start_index_range(
            len(positions),
            args.start_fraction_min,
            args.start_fraction_max,
        )
    )

    windows = [
        _analyze_window(
            reach_map=reach_map,
            positions=positions,
            start_idx=start_idx,
            control_dt=args.control_dt,
            waypoint_dt=args.waypoint_dt,
            episode_length_s=args.episode_length_s,
            reset_anchor_target_blend=args.reset_anchor_target_blend,
            tolerance=args.tolerance,
        )
        for start_idx in start_indices
    ]

    unique_orientations = np.unique(np.round(orientations_wxyz, 8), axis=0)
    position_min = positions.min(axis=0)
    position_max = positions.max(axis=0)
    return {
        "file": str(trajectory_path),
        "num_waypoints": int(len(positions)),
        "duration_seconds": float(len(positions) * args.waypoint_dt),
        "position_min": _round_list(position_min),
        "position_max": _round_list(position_max),
        "unique_orientation_count": int(len(unique_orientations)),
        "first_orientation_wxyz": _round_list(orientations_wxyz[0]),
        "first_orientation_euler_xyz_deg": _round_list(np.asarray(_quat_to_euler_xyz_deg(orientations_wxyz[0])), 3),
        "reset_start_indices": [int(v) for v in start_indices],
        "summary": _summarize_windows(windows),
        "windows": [asdict(window) for window in windows],
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append("# Proto2 Trajectory Feasibility Diagnosis")
    lines.append("")
    lines.append("This report is deterministic and non-training. It enumerates every randomized reset waypoint in the configured range and replays one static-base episode through the current reachability-map transform.")
    lines.append("")
    lines.append("## Runtime Contract")
    config = report["config"]
    for key in (
        "reach_map",
        "start_fraction_min",
        "start_fraction_max",
        "reset_anchor_target_blend",
        "control_dt",
        "waypoint_dt",
        "episode_length_s",
        "tolerance",
    ):
        lines.append(f"- `{key}`: `{config[key]}`")
    lines.append("- reset base xy: `reset_anchor.xy - [0.4415, 0.2405]`")
    lines.append("- arm frame: subtract `[0.16, 0, 0.9465]`, then rotate by `-90 deg` around z (`x_arm=y_mount`, `y_arm=-x_mount`).")
    lines.append("- RL-safe arm/gimbal envelope used for nearest `qExample` audit: `[-1.0, 0.55, -2.0, -1.0, -0.8, -0.8]` to `[1.0, 1.45, -0.4, 1.0, 0.8, 0.8]`.")
    lines.append("")
    lines.append("## Results")
    for traj in report["trajectories"]:
        summary = traj["summary"]
        best = summary["best_start_by_unreachable"]
        worst = summary["worst_start_by_unreachable"]
        lines.append(f"### `{Path(traj['file']).name}`")
        lines.append(f"- waypoints/duration: `{traj['num_waypoints']}` / `{traj['duration_seconds']:.2f}s`")
        lines.append(f"- position range: `{traj['position_min']}` to `{traj['position_max']}`")
        lines.append(f"- orientation: `{traj['unique_orientation_count']}` unique wxyz quaternion(s), first `{traj['first_orientation_wxyz']}`, Euler XYZ deg `{traj['first_orientation_euler_xyz_deg']}`")
        lines.append(f"- reset starts enumerated: `{summary['start_idx_min']}..{summary['start_idx_max']}` (`{summary['num_start_indices']}` starts)")
        lines.append(f"- unreachable mean/min/max: `{summary['unreachable_pct_mean']:.2f}%` / `{summary['unreachable_pct_min']:.2f}%` / `{summary['unreachable_pct_max']:.2f}%`")
        lines.append(f"- workspace distance p95 mean: `{summary['workspace_distance_p95_mean_m']:.4f}m`; max-distance mean: `{summary['workspace_distance_max_mean_m']:.4f}m`")
        lines.append(f"- ideal base motion from reset: max displacement mean/min/max `{summary['ideal_base_displacement_max_mean_m']:.3f}m` / `{summary['ideal_base_displacement_max_min_m']:.3f}m` / `{summary['ideal_base_displacement_max_max_m']:.3f}m`; p95 speed mean `{summary['ideal_base_speed_p95_mean_mps']:.3f}m/s`")
        lines.append(f"- nearest qExample envelope violation mean: `{summary['qexample_violation_pct_mean']:.2f}%`")
        lines.append(f"- nearest qExample per-joint violation mean: `{summary['qexample_violation_by_joint_pct_mean']}`")
        lines.append(f"- nearest qExample min/max over windows: `{summary['qexample_min_over_windows']}` to `{summary['qexample_max_over_windows']}`")
        lines.append(f"- best reset start: `{best['start_idx']}` with unreachable `{best['unreachable_pct']:.2f}%`, p95 distance `{best['p95_workspace_distance_m']:.4f}m`, max distance `{best['max_workspace_distance_m']:.4f}m`")
        lines.append(f"- worst reset start: `{worst['start_idx']}` with unreachable `{worst['unreachable_pct']:.2f}%`, p95 distance `{worst['p95_workspace_distance_m']:.4f}m`, max distance `{worst['max_workspace_distance_m']:.4f}m`")
        lines.append(f"- worst sample target world/arm: `{worst['worst_target_world']}` / `{worst['worst_target_arm']}` at waypoint `{worst['worst_waypoint_idx']}`")
        lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("For these two `crane_down` files, the static reset contract is not feasible: even the best randomized reset start leaves about half the 20s episode outside the reach map, and the average start leaves about two thirds unreachable. The required ideal base motion is modest relative to the configured base velocity limit, so base motion should be treated as required but not sufficient.")
    lines.append("")
    lines.append("The nearest reach-map `qExample` values also sit outside the current conservative action envelope, especially `joint5_arm_pitch` and the simulated wrist/gimbal axes. That means a base-only curriculum can improve reachability but still leave the policy unable to command the arm/gimbal posture implied by the reach map.")
    lines.append("")
    lines.append("The trajectory orientation is constant at wxyz `[-0.5, 0.5, -0.5, 0.5]` / Euler XYZ `[-90, 0, -90]`. This script does not prove whether that camera-frame orientation is correct for the current `cam_link`; that remains a separate FK/camera-frame check.")
    lines.append("")
    lines.append("- If unreachable percentages are low while policy EE error is high, the primary blocker is unlikely to be static reachability or chassis `vy`; inspect action-to-FK semantics, camera-frame target orientation, and whether qExample labels sit inside the training envelope.")
    lines.append("- If nearest qExample violation is high, imitation labels or reach-map examples are outside the RL-safe envelope even when positions are geometrically reachable.")
    lines.append("- If unreachable percentages are high, adjust reset-anchor/base-assist geometry before burning more PPO budget.")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", action="append", required=True, help="Trajectory JSON path. May be repeated.")
    parser.add_argument("--reach_map", default="matlab/reach_map_mobile_mm_arm_only.mat")
    parser.add_argument("--output", required=True, help="Output JSON report path.")
    parser.add_argument("--markdown-output", default=None, help="Optional Markdown report path.")
    parser.add_argument("--start-fraction-min", type=float, default=0.25)
    parser.add_argument("--start-fraction-max", type=float, default=0.70)
    parser.add_argument("--reset-anchor-target-blend", type=float, default=0.35)
    parser.add_argument("--control-dt", type=float, default=0.05)
    parser.add_argument("--waypoint-dt", type=float, default=0.1)
    parser.add_argument("--episode-length-s", type=float, default=20.0)
    parser.add_argument("--tolerance", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reach_map = ReachabilityMap(args.reach_map, device=args.device)

    report = {
        "config": {
            "reach_map": args.reach_map,
            "start_fraction_min": args.start_fraction_min,
            "start_fraction_max": args.start_fraction_max,
            "reset_anchor_target_blend": args.reset_anchor_target_blend,
            "control_dt": args.control_dt,
            "waypoint_dt": args.waypoint_dt,
            "episode_length_s": args.episode_length_s,
            "tolerance": args.tolerance,
            "device": args.device,
        },
        "trajectories": [
            _trajectory_summary(reach_map, Path(path), args)
            for path in args.trajectory
        ],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.markdown_output:
        _write_markdown(report, Path(args.markdown_output))

    print(f"Wrote JSON report: {output_path}")
    if args.markdown_output:
        print(f"Wrote Markdown report: {args.markdown_output}")


if __name__ == "__main__":
    main()
