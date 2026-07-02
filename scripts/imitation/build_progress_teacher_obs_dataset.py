#!/usr/bin/env python3
"""Build a BC-ready dataset from progress-indexed MATLAB GIK teacher exports.

The input schema is produced by gikWBC9DOF's progress teacher exporter:

    teacher_export_manifest.json
    case_*/teacher_samples.csv

Rows are keyed by normalized path progress, not by MoveIt execution time. This
script converts those labels into the current CineBotRL sim_6joint_gimbal_v1
policy contract:

    [joint6, joint5, joint4, joint3, joint2, joint1, base_vx, base_vy, base_wz]

The output .npz follows the existing BC trainer contract:

    observations, actions, action_valid_mask
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rl_platform.tasks.mobile_mm.observations import compose_observation, get_observation_dimensions  # noqa: E402


ACTION_DIM = 9
MAX_LINEAR_VELOCITY = 1.5
MAX_ANGULAR_VELOCITY = 2.0
DEFAULT_SAMPLE_TIME = 0.1

ARM_ACTION_NAMES = [
    "joint6_arm_yaw",
    "joint5_arm_pitch",
    "joint4_elbow_pitch",
    "joint3_gimbal_yaw",
    "joint2_gimbal_roll",
    "joint1_gimbal_pitch",
]
ARM_SAFE_HOME = np.array([0.0, 1.0, -1.2, 0.0, 0.0, 0.0], dtype=np.float32)
ARM_ACTION_RADIUS = np.array([1.0, 0.45, 0.8, 1.0, 0.8, 0.8], dtype=np.float32)
ARM_SAFE_LOWER = ARM_SAFE_HOME - ARM_ACTION_RADIUS
ARM_SAFE_UPPER = ARM_SAFE_HOME + ARM_ACTION_RADIUS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=Path("data/gik_stage2_parity_progress_20260702"),
        help="Directory containing teacher_export_manifest.json and case folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/gik_stage2_parity_progress_20260702/obs_dataset_progress_sim9.npz"),
    )
    parser.add_argument("--lookahead-steps", type=int, default=3)
    parser.add_argument("--action-history-length", type=int, default=2)
    parser.add_argument("--num-contacts", type=int, default=1)
    parser.add_argument("--safety-radius", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Mask non-base action labels. Useful for initial base-head BC smoke.",
    )
    parser.add_argument(
        "--append-progress",
        action="store_true",
        help="Append normalized path progress to observations. This is not compatible with the current 85D env yet.",
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return float("nan")
    return float(value)


def yaw_to_quat_wxyz(yaw: np.ndarray) -> np.ndarray:
    half = 0.5 * yaw
    return np.stack([np.cos(half), np.zeros_like(yaw), np.zeros_like(yaw), np.sin(half)], axis=1)


def quat_conj(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[:, 1:] *= -1.0
    return out


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=1,
    )


def quat_to_axis_angle(q: np.ndarray) -> np.ndarray:
    q = q.copy()
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    sign = np.where(q[:, :1] < 0.0, -1.0, 1.0)
    q *= sign
    xyz = q[:, 1:]
    norm = np.linalg.norm(xyz, axis=1, keepdims=True)
    angle = 2.0 * np.arctan2(norm, np.clip(q[:, :1], -1.0, 1.0))
    axis = xyz / np.maximum(norm, 1e-8)
    return axis * angle


def finite_difference(values: np.ndarray, dt: float) -> np.ndarray:
    if values.shape[0] == 1:
        return np.zeros_like(values, dtype=np.float32)
    diff = np.zeros_like(values, dtype=np.float32)
    diff[:-1] = (values[1:] - values[:-1]) / dt
    diff[-1] = diff[-2]
    return diff


def angular_velocity_from_quats(quat: np.ndarray, dt: float) -> np.ndarray:
    if quat.shape[0] == 1:
        return np.zeros((1, 3), dtype=np.float32)
    rel = quat_multiply(quat[1:], quat_conj(quat[:-1]))
    axis_angle = quat_to_axis_angle(rel)
    out = np.zeros((quat.shape[0], 3), dtype=np.float32)
    out[:-1] = axis_angle / dt
    out[-1] = out[-2]
    return out


def build_lookahead(target_pos: np.ndarray, steps: int) -> np.ndarray:
    idx = np.arange(target_pos.shape[0])[:, None] + np.arange(1, steps + 1)[None, :]
    idx = np.clip(idx, 0, target_pos.shape[0] - 1)
    return target_pos[idx]


def build_action_history(actions: np.ndarray, history_len: int) -> np.ndarray:
    history = np.zeros((actions.shape[0], history_len, actions.shape[1]), dtype=np.float32)
    for i in range(actions.shape[0]):
        for h in range(history_len):
            src = i - history_len + h
            if src >= 0:
                history[i, h] = actions[src]
    return history


def normalize_arm_targets(q_arm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = 2.0 * (q_arm - ARM_SAFE_LOWER[None, :]) / (ARM_SAFE_UPPER[None, :] - ARM_SAFE_LOWER[None, :]) - 1.0
    valid = np.isfinite(raw) & (np.abs(raw) <= 1.0)
    return np.clip(raw, -1.0, 1.0).astype(np.float32), valid


def read_case_csv(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def resolve_case_csv(export_dir: Path, case: dict) -> Path:
    csv_path = Path(case["teacher_samples_csv"])
    if csv_path.exists():
        return csv_path
    fallback = export_dir / case["case_name"] / "teacher_samples.csv"
    require(fallback.exists(), f"missing teacher samples: {csv_path} and {fallback}")
    return fallback


def case_sample_time(csv_path: Path) -> float:
    manifest_path = csv_path.with_name("teacher_manifest.json")
    if not manifest_path.exists():
        return DEFAULT_SAMPLE_TIME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    limits = manifest.get("velocity_limits", {})
    return float(limits.get("SampleTime", DEFAULT_SAMPLE_TIME))


def arrays_from_rows(rows: list[dict[str, str]], sample_time: float) -> dict[str, np.ndarray]:
    n = len(rows)
    require(n > 0, "case has no teacher samples")

    progress = np.asarray([parse_float(r, "progress") for r in rows], dtype=np.float32)
    base_x = np.asarray([parse_float(r, "base_x_m") for r in rows], dtype=np.float32)
    base_y = np.asarray([parse_float(r, "base_y_m") for r in rows], dtype=np.float32)
    base_yaw = np.asarray([parse_float(r, "base_yaw_rad") for r in rows], dtype=np.float32)
    base_pos = np.zeros((n, 3), dtype=np.float32)
    base_pos[:, 0] = base_x
    base_pos[:, 1] = base_y
    base_quat = yaw_to_quat_wxyz(base_yaw).astype(np.float32)

    target_pos = np.asarray(
        [[parse_float(r, "desired_x_m"), parse_float(r, "desired_y_m"), parse_float(r, "desired_z_m")] for r in rows],
        dtype=np.float32,
    )
    target_quat = np.asarray(
        [[parse_float(r, "desired_qw"), parse_float(r, "desired_qx"), parse_float(r, "desired_qy"), parse_float(r, "desired_qz")] for r in rows],
        dtype=np.float32,
    )
    ee_pos = np.asarray(
        [[parse_float(r, "actual_x_m"), parse_float(r, "actual_y_m"), parse_float(r, "actual_z_m")] for r in rows],
        dtype=np.float32,
    )
    ee_quat = np.asarray(
        [[parse_float(r, "actual_qw"), parse_float(r, "actual_qx"), parse_float(r, "actual_qy"), parse_float(r, "actual_qz")] for r in rows],
        dtype=np.float32,
    )

    q_base = np.asarray(
        [
            [parse_float(r, "q_base_joint_vx"), parse_float(r, "q_base_joint_vy"), parse_float(r, "q_base_joint_wz")]
            for r in rows
        ],
        dtype=np.float32,
    )
    q_arm = np.asarray([[parse_float(r, f"q_{name}") for name in ARM_ACTION_NAMES] for r in rows], dtype=np.float32)
    joint_pos = np.concatenate([q_base, q_arm], axis=1).astype(np.float32)

    dq_base = np.asarray(
        [
            [
                parse_float(r, "dq_next_base_joint_vx"),
                parse_float(r, "dq_next_base_joint_vy"),
                parse_float(r, "dq_next_base_joint_wz"),
            ]
            for r in rows
        ],
        dtype=np.float32,
    )
    dq_arm = np.asarray([[parse_float(r, f"dq_next_{name}") for name in ARM_ACTION_NAMES] for r in rows], dtype=np.float32)
    joint_vel = np.concatenate([dq_base, dq_arm], axis=1).astype(np.float32) / sample_time

    base_vel_world = dq_base[:, 0:2] / sample_time
    cos_yaw = np.cos(base_yaw)
    sin_yaw = np.sin(base_yaw)
    base_vx_body = cos_yaw * base_vel_world[:, 0] + sin_yaw * base_vel_world[:, 1]
    base_vy_body = -sin_yaw * base_vel_world[:, 0] + cos_yaw * base_vel_world[:, 1]
    base_wz = dq_base[:, 2] / sample_time

    actions = np.zeros((n, ACTION_DIM), dtype=np.float32)
    arm_actions, arm_valid = normalize_arm_targets(q_arm)
    actions[:, 0:6] = arm_actions
    actions[:, 6] = np.clip(base_vx_body / MAX_LINEAR_VELOCITY, -1.0, 1.0)
    actions[:, 7] = np.clip(base_vy_body / MAX_LINEAR_VELOCITY, -1.0, 1.0)
    actions[:, 8] = np.clip(base_wz / MAX_ANGULAR_VELOCITY, -1.0, 1.0)

    action_valid_mask = np.zeros_like(actions, dtype=bool)
    action_valid_mask[:, 0:6] = arm_valid
    base_raw = np.stack(
        [
            base_vx_body / MAX_LINEAR_VELOCITY,
            base_vy_body / MAX_LINEAR_VELOCITY,
            base_wz / MAX_ANGULAR_VELOCITY,
        ],
        axis=1,
    )
    action_valid_mask[:, 6:9] = np.isfinite(base_raw) & (np.abs(base_raw) <= 1.0)

    base_lin_vel = np.zeros((n, 3), dtype=np.float32)
    base_lin_vel[:, 0] = base_vel_world[:, 0] / MAX_LINEAR_VELOCITY
    base_lin_vel[:, 1] = base_vel_world[:, 1] / MAX_LINEAR_VELOCITY
    base_ang_vel = np.zeros((n, 3), dtype=np.float32)
    base_ang_vel[:, 2] = base_wz / MAX_ANGULAR_VELOCITY

    return {
        "progress": progress,
        "actions": actions,
        "action_valid_mask": action_valid_mask,
        "base_pos": base_pos,
        "base_quat": base_quat,
        "base_lin_vel": base_lin_vel,
        "base_ang_vel": base_ang_vel,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "ee_pos": ee_pos,
        "ee_quat": ee_quat,
        "ee_lin_vel": finite_difference(ee_pos, sample_time),
        "ee_ang_vel": angular_velocity_from_quats(ee_quat, sample_time),
        "target_pos": target_pos,
        "target_quat": target_quat,
    }


def compose_batches(components: dict[str, np.ndarray], args: argparse.Namespace) -> np.ndarray:
    outputs: list[np.ndarray] = []
    count = components["actions"].shape[0]
    lookahead = build_lookahead(components["target_pos"], args.lookahead_steps).astype(np.float32)
    action_history = build_action_history(components["actions"], args.action_history_length)
    contact = np.zeros((count, args.num_contacts), dtype=np.float32)
    obstacle = np.full((count, 1), 5.0, dtype=np.float32)

    for start in range(0, count, args.batch_size):
        end = min(start + args.batch_size, count)
        kwargs = {
            "base_pos": torch.from_numpy(components["base_pos"][start:end]).float(),
            "base_quat": torch.from_numpy(components["base_quat"][start:end]).float(),
            "base_lin_vel": torch.from_numpy(components["base_lin_vel"][start:end]).float(),
            "base_ang_vel": torch.from_numpy(components["base_ang_vel"][start:end]).float(),
            "joint_pos": torch.from_numpy(components["joint_pos"][start:end]).float(),
            "joint_vel": torch.from_numpy(components["joint_vel"][start:end]).float(),
            "ee_pos": torch.from_numpy(components["ee_pos"][start:end]).float(),
            "ee_quat": torch.from_numpy(components["ee_quat"][start:end]).float(),
            "ee_lin_vel": torch.from_numpy(components["ee_lin_vel"][start:end]).float(),
            "ee_ang_vel": torch.from_numpy(components["ee_ang_vel"][start:end]).float(),
            "target_pos": torch.from_numpy(components["target_pos"][start:end]).float(),
            "target_quat": torch.from_numpy(components["target_quat"][start:end]).float(),
            "lookahead_pos": torch.from_numpy(lookahead[start:end]).float(),
            "action_history": torch.from_numpy(action_history[start:end]).float(),
            "contact_forces": torch.from_numpy(contact[start:end]).float(),
            "min_obstacle_dist": torch.from_numpy(obstacle[start:end]).float(),
        }
        with torch.no_grad():
            obs = compose_observation(**kwargs).cpu().numpy().astype(np.float32)
        outputs.append(obs)
    return np.concatenate(outputs, axis=0)


def main() -> int:
    args = parse_args()
    export_dir = args.export_dir.resolve()
    manifest_path = export_dir / "teacher_export_manifest.json"
    require(manifest_path.exists(), f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = [case for case in manifest.get("cases", []) if case.get("ok", False)]
    require(cases, "manifest contains no ok cases")

    all_obs: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []
    all_progress: list[np.ndarray] = []
    source_index: list[np.ndarray] = []
    source_cases: list[str] = []

    for idx, case in enumerate(cases):
        csv_path = resolve_case_csv(export_dir, case)
        sample_time = case_sample_time(csv_path)
        components = arrays_from_rows(read_case_csv(csv_path), sample_time)
        obs = compose_batches(components, args)
        actions = components["actions"]
        mask = components["action_valid_mask"].copy()
        if args.base_only:
            mask[:, :6] = False

        all_obs.append(obs)
        all_actions.append(actions)
        all_masks.append(mask)
        all_progress.append(components["progress"])
        source_index.append(np.full(actions.shape[0], idx, dtype=np.int32))
        source_cases.append(case["case_name"])

    observations = np.concatenate(all_obs, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    action_valid_mask = np.concatenate(all_masks, axis=0)
    progress = np.concatenate(all_progress, axis=0)
    source_index_arr = np.concatenate(source_index, axis=0)

    if args.append_progress:
        observations = np.concatenate([observations, progress[:, None].astype(np.float32)], axis=1)

    expected_dim = get_observation_dimensions(
        num_joints=6,
        num_contacts=args.num_contacts,
        use_lookahead=True,
        lookahead_steps=args.lookahead_steps,
        use_action_history=True,
        action_history_length=args.action_history_length,
        action_dim=ACTION_DIM,
        use_obstacles=True,
    )
    expected_output_dim = expected_dim + (1 if args.append_progress else 0)
    require(observations.shape[1] == expected_output_dim, f"obs dim {observations.shape[1]} != expected {expected_output_dim}")
    require(np.isfinite(observations).all(), "observations contain non-finite values")
    require(np.isfinite(actions).all(), "actions contain non-finite values")
    require(np.max(np.abs(actions)) <= 1.000001, "actions outside [-1,1]")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        observations=observations,
        actions=actions,
        action_valid_mask=action_valid_mask,
        progress=progress,
        source_index=source_index_arr,
        source_cases=np.asarray(source_cases),
        manifest=str(manifest_path),
        schema=np.asarray("cinebotrl_progress_teacher_obs_dataset_v1"),
        action_contract=np.asarray("sim_6joint_gimbal_v1"),
        action_names=np.asarray(ARM_ACTION_NAMES + ["base_vx", "base_vy", "base_wz"]),
        base_only=np.asarray(args.base_only),
        append_progress=np.asarray(args.append_progress),
        observation_dim=np.asarray(observations.shape[1]),
    )

    print(f"Manifest:      {manifest_path}")
    print(f"Output:        {args.output}")
    print(f"Cases:         {len(cases)}")
    print(f"Samples:       {observations.shape[0]}")
    print(f"Obs dim:       {observations.shape[1]}")
    print(f"Append prog.:  {args.append_progress}")
    print(f"Progress:      {float(np.nanmin(progress)):.3f}..{float(np.nanmax(progress)):.3f}")
    print(f"Label mask:    {action_valid_mask.mean(axis=0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
