#!/usr/bin/env python3
"""Build a BC-ready dataset from progress-indexed MATLAB GIK teacher exports.

The input schema is produced by gikWBC9DOF's progress teacher exporter:

    teacher_export_manifest.json
    case_*/teacher_samples.csv

Rows are keyed by normalized path progress, not by MoveIt execution time. This
script converts those labels into one of the named CineBotRL policy contracts:

    [joint6, joint5, joint4, joint3, joint2, joint1, base_vx, base_vy, base_wz]

or the DJI/RS4-oriented experimental contract:

    [arm_yaw, arm_pitch, arm_elbow,
     rs4_yaw_rate, rs4_pitch_rate, rs4_roll_rate,
     base_vx, base_vy, base_wz]

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
from rl_platform.tasks.mobile_mm.action_envelopes import normalize_arm_targets as normalize_arm_targets_for_profile  # noqa: E402
from rl_platform.tasks.mobile_mm.rs4_adapter import (  # noqa: E402
    Rs4RateAdapterConfig,
    quaternion_tracking_policy_rates_deg_s,
    slew_limit_policy_rate_sequence_deg_s,
)


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
RS4_ACTION_NAMES = [
    "arm_yaw",
    "arm_pitch",
    "arm_elbow",
    "rs4_yaw_rate",
    "rs4_pitch_rate",
    "rs4_roll_rate",
]


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
        "--action-contract",
        choices=["sim_6joint_gimbal_v1", "rs4_attitude_rate_v1"],
        default="sim_6joint_gimbal_v1",
        help=(
            "Output action semantics. rs4_attitude_rate_v1 uses explicit "
            "gimbal_attitude_target_q* labels instead of physical gimbal q_* joints."
        ),
    )
    parser.add_argument(
        "--arm-envelope-profile",
        choices=["proto2_safe_v1", "teacher_wide_v1"],
        default="proto2_safe_v1",
    )
    parser.add_argument(
        "--enable-roll",
        action="store_true",
        help="Include RS4 roll-rate labels. Default masks roll for the current DJI path.",
    )
    parser.add_argument(
        "--attitude-response-horizon-s",
        type=float,
        default=0.5,
        help="Time horizon used to convert camera-attitude residual into a corrective rate.",
    )
    parser.add_argument(
        "--control-dt-s",
        type=float,
        default=0.05,
        help="Fixed runtime control period used for RS4 acceleration slew limiting.",
    )
    parser.add_argument(
        "--allow-desired-pose-gimbal-fallback",
        action="store_true",
        help=(
            "For older progress exports only: derive RS4 attitude labels from desired_q* "
            "if explicit gimbal_attitude_target_q* columns are missing."
        ),
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


def normalize_arm_targets(q_arm: np.ndarray, profile: str) -> tuple[np.ndarray, np.ndarray]:
    actions, _, valid = normalize_arm_targets_for_profile(q_arm, profile=profile)
    return actions, valid


def normalize_real_arm_targets(q_arm: np.ndarray, profile: str) -> tuple[np.ndarray, np.ndarray]:
    padded = np.zeros((q_arm.shape[0], 6), dtype=np.float32)
    padded[:, :3] = q_arm
    actions, valid = normalize_arm_targets(padded, profile)
    return actions[:, :3], valid[:, :3]


def read_quat_columns(rows: list[dict[str, str]], prefix: str) -> np.ndarray | None:
    keys = [f"{prefix}_qw", f"{prefix}_qx", f"{prefix}_qy", f"{prefix}_qz"]
    missing = [key for key in keys if key not in rows[0]]
    if missing:
        return None
    quat = np.asarray([[parse_float(r, key) for key in keys] for r in rows], dtype=np.float32)
    require(np.isfinite(quat).all(), f"{prefix} contains non-finite values")
    return quat


def read_rs4_target_quat(rows: list[dict[str, str]], *, allow_desired_pose_fallback: bool) -> tuple[np.ndarray, str]:
    target_quat = read_quat_columns(rows, "gimbal_attitude_target")
    if target_quat is not None:
        return target_quat, "gimbal_attitude_target_qwxyz"
    require(
        allow_desired_pose_fallback,
        "missing gimbal_attitude_target_q* columns; rerun the updated GIK exporter or pass "
        "--allow-desired-pose-gimbal-fallback for legacy progress exports",
    )
    fallback = read_quat_columns(rows, "desired")
    require(fallback is not None, "legacy fallback requested but desired_q* columns are missing")
    return fallback, "desired_qwxyz_legacy_fallback"


def build_rs4_attitude_rate_actions(
    q_arm_full: np.ndarray,
    base_actions: np.ndarray,
    base_valid: np.ndarray,
    rows: list[dict[str, str]],
    sample_time: float,
    *,
    enable_roll: bool,
    allow_desired_pose_fallback: bool,
    arm_envelope_profile: str,
    response_horizon_s: float,
    control_dt_s: float,
) -> dict[str, np.ndarray]:
    arm_actions, arm_valid = normalize_real_arm_targets(q_arm_full[:, :3], arm_envelope_profile)
    target_quat, attitude_source = read_rs4_target_quat(
        rows,
        allow_desired_pose_fallback=allow_desired_pose_fallback,
    )
    current_quat = read_quat_columns(rows, "actual")
    require(current_quat is not None, "missing actual_q* columns required for closed-loop RS4 labels")

    adapter_cfg = Rs4RateAdapterConfig(enable_roll=enable_roll)
    desired_rates, feedforward_rates, attitude_residual_deg = quaternion_tracking_policy_rates_deg_s(
        current_quat,
        target_quat,
        dt_s=sample_time,
        response_horizon_s=response_horizon_s,
        config=adapter_cfg,
    )
    attitude_rates = slew_limit_policy_rate_sequence_deg_s(
        desired_rates,
        control_dt_s,
        adapter_cfg,
    )
    rate_actions = (attitude_rates / adapter_cfg.max_policy_order_rates[None, :]).astype(np.float32)
    rate_valid = np.isfinite(rate_actions)
    if not enable_roll:
        rate_actions[:, 2] = 0.0
        rate_valid[:, 2] = False

    actions = np.zeros((q_arm_full.shape[0], ACTION_DIM), dtype=np.float32)
    actions[:, 0:3] = arm_actions
    actions[:, 3:6] = rate_actions
    actions[:, 6:9] = base_actions

    mask = np.zeros_like(actions, dtype=bool)
    mask[:, 0:3] = arm_valid
    mask[:, 3:6] = rate_valid
    mask[:, 6:9] = base_valid

    return {
        "actions": actions,
        "action_valid_mask": mask,
        "gimbal_attitude_target_quat_wxyz": target_quat.astype(np.float32),
        "camera_attitude_residual_deg": attitude_residual_deg.astype(np.float32),
        "camera_attitude_feedforward_rate_deg_s": feedforward_rates.astype(np.float32),
        "camera_attitude_rate_deg_s": attitude_rates.astype(np.float32),
        "max_rs4_rate_deg_s": adapter_cfg.max_policy_order_rates.astype(np.float32),
        "rs4_attitude_label_source": np.asarray(attitude_source),
    }


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


def arrays_from_rows(rows: list[dict[str, str]], sample_time: float, args: argparse.Namespace) -> dict[str, np.ndarray]:
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
    q_arm_full = np.asarray([[parse_float(r, f"q_{name}") for name in ARM_ACTION_NAMES] for r in rows], dtype=np.float32)
    joint_pos = np.concatenate([q_base, q_arm_full], axis=1).astype(np.float32)

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

    base_raw = np.stack(
        [
            base_vx_body / MAX_LINEAR_VELOCITY,
            base_vy_body / MAX_LINEAR_VELOCITY,
            base_wz / MAX_ANGULAR_VELOCITY,
        ],
        axis=1,
    )
    base_actions = np.clip(base_raw, -1.0, 1.0).astype(np.float32)
    base_valid = np.isfinite(base_raw) & (np.abs(base_raw) <= 1.0)

    extra: dict[str, np.ndarray] = {}
    if args.action_contract == "rs4_attitude_rate_v1":
        rs4 = build_rs4_attitude_rate_actions(
            q_arm_full,
            base_actions,
            base_valid,
            rows,
            sample_time,
            enable_roll=args.enable_roll,
            allow_desired_pose_fallback=args.allow_desired_pose_gimbal_fallback,
            arm_envelope_profile=args.arm_envelope_profile,
            response_horizon_s=args.attitude_response_horizon_s,
            control_dt_s=args.control_dt_s,
        )
        actions = rs4.pop("actions")
        action_valid_mask = rs4.pop("action_valid_mask")
        extra.update(rs4)
    else:
        actions = np.zeros((n, ACTION_DIM), dtype=np.float32)
        arm_actions, arm_valid = normalize_arm_targets(q_arm_full, args.arm_envelope_profile)
        actions[:, 0:6] = arm_actions
        actions[:, 6:9] = base_actions

        action_valid_mask = np.zeros_like(actions, dtype=bool)
        action_valid_mask[:, 0:6] = arm_valid
        action_valid_mask[:, 6:9] = base_valid

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
        **extra,
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
    all_gimbal_target_quat: list[np.ndarray] = []
    all_attitude_residual: list[np.ndarray] = []
    all_attitude_rates: list[np.ndarray] = []
    all_feedforward_rates: list[np.ndarray] = []
    rs4_attitude_label_sources: set[str] = set()
    source_index: list[np.ndarray] = []
    source_cases: list[str] = []

    for idx, case in enumerate(cases):
        csv_path = resolve_case_csv(export_dir, case)
        sample_time = case_sample_time(csv_path)
        components = arrays_from_rows(read_case_csv(csv_path), sample_time, args)
        obs = compose_batches(components, args)
        actions = components["actions"]
        mask = components["action_valid_mask"].copy()
        if args.base_only:
            mask[:, :6] = False

        all_obs.append(obs)
        all_actions.append(actions)
        all_masks.append(mask)
        all_progress.append(components["progress"])
        if args.action_contract == "rs4_attitude_rate_v1":
            all_gimbal_target_quat.append(components["gimbal_attitude_target_quat_wxyz"])
            all_attitude_residual.append(components["camera_attitude_residual_deg"])
            all_attitude_rates.append(components["camera_attitude_rate_deg_s"])
            all_feedforward_rates.append(components["camera_attitude_feedforward_rate_deg_s"])
            rs4_attitude_label_sources.add(str(components["rs4_attitude_label_source"]))
        source_index.append(np.full(actions.shape[0], idx, dtype=np.int32))
        source_cases.append(case["case_name"])

    if args.action_contract == "rs4_attitude_rate_v1":
        action_names = RS4_ACTION_NAMES + ["base_vx", "base_vy", "base_wz"]
    else:
        action_names = ARM_ACTION_NAMES + ["base_vx", "base_vy", "base_wz"]

    observations = np.concatenate(all_obs, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    action_valid_mask = np.concatenate(all_masks, axis=0)
    progress = np.concatenate(all_progress, axis=0)
    source_index_arr = np.concatenate(source_index, axis=0)
    extra_outputs = {}
    if args.action_contract == "rs4_attitude_rate_v1":
        label_source = ",".join(sorted(rs4_attitude_label_sources))
        extra_outputs = {
            "gimbal_attitude_target_quat_wxyz": np.concatenate(all_gimbal_target_quat, axis=0),
            "camera_attitude_residual_deg": np.concatenate(all_attitude_residual, axis=0),
            "camera_attitude_rate_deg_s": np.concatenate(all_attitude_rates, axis=0),
            "camera_attitude_feedforward_rate_deg_s": np.concatenate(all_feedforward_rates, axis=0),
            "max_rs4_rate_deg_s": Rs4RateAdapterConfig(enable_roll=args.enable_roll).max_policy_order_rates,
            "attitude_frame_convention": np.asarray(f"local_camera_rotation_vector_zyx from {label_source}"),
            "attitude_label_mode": np.asarray("quaternion_tracking_slew_limited_v1"),
            "attitude_response_horizon_s": np.asarray(args.attitude_response_horizon_s, dtype=np.float32),
            "control_dt_s": np.asarray(args.control_dt_s, dtype=np.float32),
            "rs4_attitude_label_source": np.asarray(label_source),
        }

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
        action_contract=np.asarray(args.action_contract),
        action_names=np.asarray(action_names),
        gimbal_label_contract=np.asarray(
            "gimbal_attitude_target_q*_closed_loop_quaternion_residual"
            if args.action_contract == "rs4_attitude_rate_v1"
            else "physical_sim_urdf_gimbal_joint_targets"
        ),
        roll_enabled=np.asarray(args.enable_roll),
        base_only=np.asarray(args.base_only),
        append_progress=np.asarray(args.append_progress),
        observation_dim=np.asarray(observations.shape[1]),
        arm_envelope_profile=np.asarray(args.arm_envelope_profile),
        **extra_outputs,
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
