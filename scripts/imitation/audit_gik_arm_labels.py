#!/usr/bin/env python3
"""Audit GIK arm/wrist labels before enabling arm imitation learning.

This validates the six RL-controlled arm channels against the Proto2 action
contract and performs lightweight URDF FK to compare labeled joint states with
exported end-effector poses.  Virtual MoveIt-style EE gimbal joints are not part
of the RL policy and are kept at zero in this audit.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Keep this script Isaac-free. Importing rl_platform task modules can trigger
# optional Isaac imports on Windows; these names mirror mobile_mm/joint_names.py.
ARM_JOINT_NAMES = [
    "joint6_arm_yaw",
    "joint5_arm_pitch",
    "joint4_elbow_pitch",
    "joint3_gimbal_yaw",
    "joint2_gimbal_roll",
    "joint1_gimbal_pitch",
]
EE_LINK_NAME = "cam_link"
EE_VIRTUAL_JOINT_NAMES = ["ee1_rot_z", "ee1_rot_y", "ee1_rot_x"]

ARM_SAFE_HOME = np.array([0.0, 1.0, -1.2, 0.0, 0.0, 0.0], dtype=np.float64)
ARM_ACTION_RADIUS = np.array([1.0, 0.45, 0.8, 1.0, 0.8, 0.8], dtype=np.float64)
JOINT_LIMIT_MARGIN = 0.1
MAX_JOINT_ACCEL = 6.0
CONTROL_DT = 0.05
MAX_ARM_TARGET_DELTA = MAX_JOINT_ACCEL * CONTROL_DT * CONTROL_DT


@dataclass
class Joint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float | None
    upper: float | None
    velocity: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-dir", type=Path, default=Path("data/gik_ik_demos"))
    parser.add_argument("--manifest", type=str, default="manifest.json")
    parser.add_argument("--urdf", type=Path, default=Path("assets_own/recomoProto2-1190_moveit.urdf"))
    parser.add_argument("--output", type=Path, default=Path("data/gik_ik_demos/arm_label_audit.json"))
    parser.add_argument("--sample-stride", type=int, default=1, help="Evaluate FK on every Nth sample.")
    parser.add_argument("--position-warn-m", type=float, default=0.05)
    parser.add_argument("--orientation-warn-rad", type=float, default=0.20)
    return parser.parse_args()


def parse_vec(text: str | None, default: tuple[float, ...]) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(x) for x in text.split()], dtype=np.float64)


def transform_from_xyz_rpy(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
    t[:3, 3] = xyz
    return t


def motion_transform(joint_type: str, axis: np.ndarray, value: float) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    if joint_type == "fixed":
        return t
    if joint_type == "prismatic":
        t[:3, 3] = axis * value
        return t
    if joint_type in {"revolute", "continuous"}:
        t[:3, :3] = R.from_rotvec(axis * value).as_matrix()
        return t
    raise ValueError(f"unsupported joint type: {joint_type}")


def parse_urdf(path: Path) -> dict[str, Joint]:
    root = ET.parse(path).getroot()
    joints: dict[str, Joint] = {}
    for elem in root.findall("joint"):
        name = elem.get("name") or ""
        parent = elem.find("parent").get("link")
        child = elem.find("child").get("link")
        origin_elem = elem.find("origin")
        axis_elem = elem.find("axis")
        limit_elem = elem.find("limit")
        xyz = parse_vec(origin_elem.get("xyz") if origin_elem is not None else None, (0.0, 0.0, 0.0))
        rpy = parse_vec(origin_elem.get("rpy") if origin_elem is not None else None, (0.0, 0.0, 0.0))
        axis = parse_vec(axis_elem.get("xyz") if axis_elem is not None else None, (0.0, 0.0, 1.0))
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-12:
            axis = axis / axis_norm
        lower = upper = velocity = None
        if limit_elem is not None:
            lower = float(limit_elem.get("lower", "nan")) if limit_elem.get("lower") is not None else None
            upper = float(limit_elem.get("upper", "nan")) if limit_elem.get("upper") is not None else None
            velocity = float(limit_elem.get("velocity", "nan")) if limit_elem.get("velocity") is not None else None
        joints[name] = Joint(
            name=name,
            joint_type=elem.get("type") or "fixed",
            parent=parent,
            child=child,
            origin=transform_from_xyz_rpy(xyz, rpy),
            axis=axis,
            lower=lower,
            upper=upper,
            velocity=velocity,
        )
    return joints


def chain_to_link(joints: dict[str, Joint], root_link: str, target_link: str) -> list[Joint]:
    by_child = {joint.child: joint for joint in joints.values()}
    chain: list[Joint] = []
    link = target_link
    while link != root_link:
        if link not in by_child:
            raise ValueError(f"no parent joint found for link {link!r} while resolving {target_link!r}")
        joint = by_child[link]
        chain.append(joint)
        link = joint.parent
    chain.reverse()
    return chain


def fk(chain: list[Joint], values: dict[str, float]) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    for joint in chain:
        value = values.get(joint.name, 0.0)
        t = t @ joint.origin @ motion_transform(joint.joint_type, joint.axis, value)
    return t


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    return R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()


def matrix_to_quat_wxyz(mat: np.ndarray) -> np.ndarray:
    q = R.from_matrix(mat).as_quat()
    return np.asarray([q[3], q[0], q[1], q[2]], dtype=np.float64)


def quat_angle_error_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    q1 = q1 / np.maximum(np.linalg.norm(q1, axis=-1, keepdims=True), 1e-12)
    q2 = q2 / np.maximum(np.linalg.norm(q2, axis=-1, keepdims=True), 1e-12)
    dot = np.abs(np.sum(q1 * q2, axis=-1))
    return 2.0 * np.arccos(np.clip(dot, -1.0, 1.0))


def resolve_npz_path(item: dict[str, Any], demo_dir: Path) -> Path:
    path = Path(item["output_npz"])
    if path.exists():
        return path
    fallback = demo_dir / path.name
    if not fallback.exists():
        raise FileNotFoundError(f"missing npz for manifest item: {path}")
    return fallback


def pct(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


def summarize_bool(mask: np.ndarray) -> dict[str, Any]:
    return {
        "valid_fraction_by_joint": np.mean(mask, axis=0).astype(float).tolist(),
        "valid_fraction_all_joints": float(np.mean(np.all(mask, axis=1))),
        "valid_fraction_all_values": float(np.mean(mask)),
    }


def summarize_error(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": pct(values, 50),
        "p90": pct(values, 90),
        "p95": pct(values, 95),
        "p99": pct(values, 99),
        "max": float(np.max(values)),
    }


def audit_one(npz_path: Path, chain: list[Joint], lower: np.ndarray, upper: np.ndarray, safe_lower: np.ndarray, safe_upper: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    with np.load(npz_path) as data:
        q_current = data["q_current"].astype(np.float64)
        q_next = data["q_next"].astype(np.float64)
        actions = data["actions"].astype(np.float64)
        dt = data["dt"].astype(np.float64)
        actual_pos = data["actual_ee_pos"].astype(np.float64)
        actual_quat = data["actual_ee_quat_wxyz"].astype(np.float64)
        target_pos = data["target_pos"].astype(np.float64)
        target_quat = data["target_quat_wxyz"].astype(np.float64)
        stored_mask = data["arm_action_valid_mask"].astype(bool) if "arm_action_valid_mask" in data else data["action_valid_mask"][:, :6].astype(bool)

    arm_current = q_current[:, 3:9]
    arm_next = q_next[:, 3:9]
    range_mask_next = (arm_next >= lower[None, :]) & (arm_next <= upper[None, :])
    safe_mask_next = (arm_next >= safe_lower[None, :]) & (arm_next <= safe_upper[None, :])
    action_range_mask = np.abs(actions[:, :6]) <= 1.0 + 1e-6
    arm_delta = np.abs(arm_next - arm_current)
    arm_vel = arm_delta / dt[:, None]
    slew_mask = arm_delta <= MAX_ARM_TARGET_DELTA + 1e-9

    indices = np.arange(0, actions.shape[0], max(args.sample_stride, 1), dtype=int)
    fk_current_pos = []
    fk_next_pos = []
    fk_current_quat = []
    fk_next_quat = []
    for i in indices:
        for q_state, pos_list, quat_list in ((q_current[i], fk_current_pos, fk_current_quat), (q_next[i], fk_next_pos, fk_next_quat)):
            values = {
                "base_joint_vx": float(q_state[0]),
                "base_joint_vy": float(q_state[1]),
                "base_joint_wz": float(q_state[2]),
            }
            values.update({name: float(q_state[3 + j]) for j, name in enumerate(ARM_JOINT_NAMES)})
            values.update({name: 0.0 for name in EE_VIRTUAL_JOINT_NAMES})
            t = fk(chain, values)
            pos_list.append(t[:3, 3])
            quat_list.append(matrix_to_quat_wxyz(t[:3, :3]))

    fk_current_pos_arr = np.asarray(fk_current_pos)
    fk_next_pos_arr = np.asarray(fk_next_pos)
    fk_current_quat_arr = np.asarray(fk_current_quat)
    fk_next_quat_arr = np.asarray(fk_next_quat)
    actual_pos_s = actual_pos[indices]
    actual_quat_s = actual_quat[indices]
    target_pos_s = target_pos[indices]
    target_quat_s = target_quat[indices]

    current_actual_pos_err = np.linalg.norm(fk_current_pos_arr - actual_pos_s, axis=1)
    next_actual_pos_err = np.linalg.norm(fk_next_pos_arr - actual_pos_s, axis=1)
    current_target_pos_err = np.linalg.norm(fk_current_pos_arr - target_pos_s, axis=1)
    next_target_pos_err = np.linalg.norm(fk_next_pos_arr - target_pos_s, axis=1)
    current_actual_ori_err = quat_angle_error_wxyz(fk_current_quat_arr, actual_quat_s)
    next_actual_ori_err = quat_angle_error_wxyz(fk_next_quat_arr, actual_quat_s)
    current_target_ori_err = quat_angle_error_wxyz(fk_current_quat_arr, target_quat_s)
    next_target_ori_err = quat_angle_error_wxyz(fk_next_quat_arr, target_quat_s)

    best_pos_err = np.minimum(current_actual_pos_err, next_actual_pos_err)
    best_ori_err = np.minimum(current_actual_ori_err, next_actual_ori_err)

    return {
        "file": npz_path.name,
        "samples": int(actions.shape[0]),
        "fk_samples": int(indices.size),
        "stored_mask": summarize_bool(stored_mask),
        "action_range_mask": summarize_bool(action_range_mask),
        "urdf_limit_mask_next": summarize_bool(range_mask_next),
        "rl_safe_envelope_mask_next": summarize_bool(safe_mask_next),
        "slew_mask": summarize_bool(slew_mask),
        "arm_delta_abs_rad": {
            "mean_by_joint": np.mean(arm_delta, axis=0).astype(float).tolist(),
            "p95_by_joint": np.percentile(arm_delta, 95, axis=0).astype(float).tolist(),
            "max_by_joint": np.max(arm_delta, axis=0).astype(float).tolist(),
            "max_allowed_per_step": float(MAX_ARM_TARGET_DELTA),
        },
        "arm_velocity_abs_rad_s": {
            "mean_by_joint": np.mean(arm_vel, axis=0).astype(float).tolist(),
            "p95_by_joint": np.percentile(arm_vel, 95, axis=0).astype(float).tolist(),
            "max_by_joint": np.max(arm_vel, axis=0).astype(float).tolist(),
        },
        "fk_current_vs_actual_pos_m": summarize_error(current_actual_pos_err),
        "fk_next_vs_actual_pos_m": summarize_error(next_actual_pos_err),
        "fk_best_vs_actual_pos_m": summarize_error(best_pos_err),
        "fk_current_vs_actual_ori_rad": summarize_error(current_actual_ori_err),
        "fk_next_vs_actual_ori_rad": summarize_error(next_actual_ori_err),
        "fk_best_vs_actual_ori_rad": summarize_error(best_ori_err),
        "fk_current_vs_target_pos_m": summarize_error(current_target_pos_err),
        "fk_next_vs_target_pos_m": summarize_error(next_target_pos_err),
        "fk_current_vs_target_ori_rad": summarize_error(current_target_ori_err),
        "fk_next_vs_target_ori_rad": summarize_error(next_target_ori_err),
        "pass_flags": {
            "all_actions_in_range": bool(np.all(action_range_mask)),
            "all_targets_in_urdf_limits": bool(np.all(range_mask_next)),
            "all_targets_in_rl_safe_envelope": bool(np.all(safe_mask_next)),
            "all_steps_within_env_slew_limit": bool(np.all(slew_mask)),
            "fk_best_pos_p95_ok": bool(pct(best_pos_err, 95) <= args.position_warn_m),
            "fk_best_ori_p95_ok": bool(pct(best_ori_err, 95) <= args.orientation_warn_rad),
        },
    }


def aggregate(items: list[dict[str, Any]], joint_names: list[str]) -> dict[str, Any]:
    weighted: dict[str, Any] = {"total_samples": sum(item["samples"] for item in items), "total_fk_samples": sum(item["fk_samples"] for item in items)}
    for key in ["stored_mask", "action_range_mask", "urdf_limit_mask_next", "rl_safe_envelope_mask_next", "slew_mask"]:
        vals = np.asarray([item[key]["valid_fraction_by_joint"] for item in items], dtype=np.float64)
        weights = np.asarray([item["samples"] for item in items], dtype=np.float64)
        weighted[key] = {
            "joint_names": joint_names,
            "valid_fraction_by_joint": np.average(vals, axis=0, weights=weights).astype(float).tolist(),
            "valid_fraction_all_values_mean_by_file": float(np.average([item[key]["valid_fraction_all_values"] for item in items], weights=weights)),
            "valid_fraction_all_joints_mean_by_file": float(np.average([item[key]["valid_fraction_all_joints"] for item in items], weights=weights)),
        }
    for key in [
        "fk_best_vs_actual_pos_m",
        "fk_best_vs_actual_ori_rad",
        "fk_next_vs_actual_pos_m",
        "fk_next_vs_actual_ori_rad",
    ]:
        weighted[key] = {
            "mean_by_file": float(np.average([item[key]["mean"] for item in items], weights=[item["fk_samples"] for item in items])),
            "p95_max_by_file": float(max(item[key]["p95"] for item in items)),
            "max_by_file": float(max(item[key]["max"] for item in items)),
        }
    return weighted


def main() -> int:
    args = parse_args()
    demo_dir = args.demo_dir.resolve()
    manifest_path = demo_dir / args.manifest
    if not manifest_path.exists():
        raise SystemExit(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = manifest.get("items", [])
    if not items:
        raise SystemExit("manifest contains no items")

    urdf_path = args.urdf.resolve()
    joints = parse_urdf(urdf_path)
    chain = chain_to_link(joints, "base_root", EE_LINK_NAME)
    chain_names = [joint.name for joint in chain]
    unexpected_virtual = [name for name in EE_VIRTUAL_JOINT_NAMES if name in chain_names]
    if unexpected_virtual:
        raise SystemExit(f"virtual EE joints unexpectedly appear in FK chain to {EE_LINK_NAME}: {unexpected_virtual}")

    lower = np.asarray([joints[name].lower for name in ARM_JOINT_NAMES], dtype=np.float64)
    upper = np.asarray([joints[name].upper for name in ARM_JOINT_NAMES], dtype=np.float64)
    safe_lower = np.maximum(lower + JOINT_LIMIT_MARGIN, ARM_SAFE_HOME - ARM_ACTION_RADIUS)
    safe_upper = np.minimum(upper - JOINT_LIMIT_MARGIN, ARM_SAFE_HOME + ARM_ACTION_RADIUS)

    results = []
    failures = []
    for item in items:
        try:
            results.append(audit_one(resolve_npz_path(item, demo_dir), chain, lower, upper, safe_lower, safe_upper, args))
        except Exception as exc:
            failures.append({"item": item.get("output_npz", "<unknown>"), "error": str(exc)})

    report = {
        "schema": "cinebotrl_gik_arm_label_audit_v1",
        "manifest": str(manifest_path),
        "urdf": str(urdf_path),
        "ee_link": EE_LINK_NAME,
        "fk_chain": chain_names,
        "rl_controlled_arm_joints": ARM_JOINT_NAMES,
        "virtual_ee_joints_excluded": EE_VIRTUAL_JOINT_NAMES,
        "urdf_lower": lower.astype(float).tolist(),
        "urdf_upper": upper.astype(float).tolist(),
        "rl_safe_lower": safe_lower.astype(float).tolist(),
        "rl_safe_upper": safe_upper.astype(float).tolist(),
        "max_arm_target_delta_rad_per_step": MAX_ARM_TARGET_DELTA,
        "num_items": len(results),
        "num_failures": len(failures),
        "aggregate": aggregate(results, ARM_JOINT_NAMES) if results else {},
        "items": results,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    agg = report["aggregate"]
    print(f"Manifest:       {manifest_path}")
    print(f"URDF:           {urdf_path}")
    print(f"Output:         {args.output}")
    print(f"Items/failures: {len(results)} / {len(failures)}")
    if agg:
        print("RL safe valid:  " + " ".join(f"{v:.3f}" for v in agg["rl_safe_envelope_mask_next"]["valid_fraction_by_joint"]))
        print("URDF valid:     " + " ".join(f"{v:.3f}" for v in agg["urdf_limit_mask_next"]["valid_fraction_by_joint"]))
        print("Slew valid:     " + " ".join(f"{v:.3f}" for v in agg["slew_mask"]["valid_fraction_by_joint"]))
        print(f"FK pos mean:    {agg['fk_best_vs_actual_pos_m']['mean_by_file']:.4f} m")
        print(f"FK pos p95 max: {agg['fk_best_vs_actual_pos_m']['p95_max_by_file']:.4f} m")
        print(f"FK ori mean:    {agg['fk_best_vs_actual_ori_rad']['mean_by_file']:.4f} rad")
        print(f"FK ori p95 max: {agg['fk_best_vs_actual_ori_rad']['p95_max_by_file']:.4f} rad")
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
