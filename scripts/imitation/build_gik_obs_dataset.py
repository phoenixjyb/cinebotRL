#!/usr/bin/env python3
"""Build an offline CineBotRL observation/action dataset from exported GIK demos.

The exporter stores MATLAB IK trajectories as action labels plus aligned target
and actual end-effector poses.  This script composes the same observation vector
used by the Proto2 tracking environment without launching Isaac Sim, which makes
it suitable for bounded behavior-cloning and dataset sanity checks.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rl_platform.tasks.mobile_mm.observations import compose_observation, get_observation_dimensions  # noqa: E402
from rl_platform.tasks.mobile_mm.action_envelopes import (  # noqa: E402
    ARM_ENVELOPE_PROFILES,
    normalize_arm_targets,
)


ACTION_DIM = 9
MAX_LINEAR_VELOCITY = 1.5
MAX_ANGULAR_VELOCITY = 2.0
ARM_JOINT_NAMES = [
    "joint6_arm_yaw",
    "joint5_arm_pitch",
    "joint4_elbow_pitch",
    "joint3_gimbal_yaw",
    "joint2_gimbal_roll",
    "joint1_gimbal_pitch",
]
EE_VIRTUAL_JOINT_NAMES = ["ee1_rot_z", "ee1_rot_y", "ee1_rot_x"]
EE_LINK_NAME = "cam_link"


@dataclass
class Joint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-dir", type=Path, default=Path("data/gik_ik_demos"))
    parser.add_argument("--manifest", type=str, default="manifest_strict.json")
    parser.add_argument("--output", type=Path, default=Path("data/gik_ik_demos/obs_dataset_strict.npz"))
    parser.add_argument("--base-only", action="store_true", help="Mask imitation labels to base_vx/base_vy/base_wz only.")
    parser.add_argument("--lookahead-steps", type=int, default=3)
    parser.add_argument("--action-history-length", type=int, default=2)
    parser.add_argument("--num-contacts", type=int, default=1)
    parser.add_argument("--urdf", type=Path, default=Path("assets_own/recomoProto2-1190_moveit.urdf"))
    parser.add_argument(
        "--ee-state-source",
        choices=["stored", "fk_current"],
        default="stored",
        help="Source for observation EE pose. fk_current recomputes cam_link pose from q_current.",
    )
    parser.add_argument(
        "--velocity-source",
        choices=["action", "lagged_q"],
        default="action",
        help="Source for observation velocities. lagged_q uses previous-current q deltas to avoid label leakage.",
    )
    parser.add_argument(
        "--target-shift-steps",
        type=int,
        default=0,
        help="Shift target pose rows forward to match env reset/step timing.",
    )
    parser.add_argument(
        "--flip-fk-ee-quat",
        action="store_true",
        help="Flip FK EE quaternion sign to match Isaac cam_link convention for this USD.",
    )
    parser.add_argument(
        "--use-obstacles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include obstacle-clearance observations. Disable for no-obstacle stages to keep obs dim aligned with env.",
    )
    parser.add_argument("--safety-radius", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--arm-envelope-profile",
        type=str,
        default="stored",
        choices=("stored",) + ARM_ENVELOPE_PROFILES,
        help=(
            "Arm label source. 'stored' preserves exported clipped labels. "
            "Named profiles rebuild rows [0:6] from q_next[:,3:9] using that "
            "physical action envelope."
        ),
    )
    parser.add_argument(
        "--resample-dt",
        type=float,
        default=0.0,
        help=(
            "If positive, stretch each accepted sparse GIK teacher over its "
            "manifest duration and resample rows at this fixed dt. This aligns "
            "BC labels with the dense trajectory stage used by Isaac playback."
        ),
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


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


def finite_difference(values: np.ndarray, dt: np.ndarray) -> np.ndarray:
    if values.shape[0] == 1:
        return np.zeros_like(values)
    diff = np.zeros_like(values)
    diff[:-1] = (values[1:] - values[:-1]) / dt[:-1, None]
    diff[-1] = diff[-2]
    return diff


def backward_difference(values: np.ndarray, dt: np.ndarray) -> np.ndarray:
    diff = np.zeros_like(values, dtype=np.float32)
    if values.shape[0] <= 1:
        return diff
    diff[1:] = (values[1:] - values[:-1]) / dt[1:, None]
    return diff.astype(np.float32)


def angular_velocity_from_quats(quat: np.ndarray, dt: np.ndarray) -> np.ndarray:
    if quat.shape[0] == 1:
        return np.zeros((1, 3), dtype=np.float32)
    rel = quat_multiply(quat[1:], quat_conj(quat[:-1]))
    axis_angle = quat_to_axis_angle(rel)
    out = np.zeros((quat.shape[0], 3), dtype=np.float32)
    out[:-1] = axis_angle / dt[:-1, None]
    out[-1] = out[-2]
    return out


def backward_angular_velocity_from_quats(quat: np.ndarray, dt: np.ndarray) -> np.ndarray:
    if quat.shape[0] == 1:
        return np.zeros((1, 3), dtype=np.float32)
    rel = quat_multiply(quat[1:], quat_conj(quat[:-1]))
    axis_angle = quat_to_axis_angle(rel)
    out = np.zeros((quat.shape[0], 3), dtype=np.float32)
    out[1:] = axis_angle / dt[1:, None]
    return out


def normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = quat.astype(np.float64, copy=True)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    quat = quat / np.maximum(norm, 1e-12)
    sign = np.where(quat[:, :1] < 0.0, -1.0, 1.0)
    return (quat * sign).astype(np.float32)


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
        xyz = parse_vec(origin_elem.get("xyz") if origin_elem is not None else None, (0.0, 0.0, 0.0))
        rpy = parse_vec(origin_elem.get("rpy") if origin_elem is not None else None, (0.0, 0.0, 0.0))
        axis = parse_vec(axis_elem.get("xyz") if axis_elem is not None else None, (0.0, 0.0, 1.0))
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-12:
            axis = axis / axis_norm
        joints[name] = Joint(
            name=name,
            joint_type=elem.get("type") or "fixed",
            parent=parent,
            child=child,
            origin=transform_from_xyz_rpy(xyz, rpy),
            axis=axis,
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


def matrix_to_quat_wxyz(mat: np.ndarray) -> np.ndarray:
    q = R.from_matrix(mat).as_quat()
    return np.asarray([q[3], q[0], q[1], q[2]], dtype=np.float64)


def fk_ee_from_q(q_state: np.ndarray, chain: list[Joint]) -> tuple[np.ndarray, np.ndarray]:
    ee_pos = np.zeros((q_state.shape[0], 3), dtype=np.float32)
    ee_quat = np.zeros((q_state.shape[0], 4), dtype=np.float32)
    for i, row in enumerate(q_state):
        values = {
            "base_joint_vx": float(row[0]),
            "base_joint_vy": float(row[1]),
            "base_joint_wz": float(row[2]),
        }
        values.update({name: float(row[3 + j]) for j, name in enumerate(ARM_JOINT_NAMES)})
        values.update({name: 0.0 for name in EE_VIRTUAL_JOINT_NAMES})
        t = fk(chain, values)
        ee_pos[i] = t[:3, 3].astype(np.float32)
        ee_quat[i] = matrix_to_quat_wxyz(t[:3, :3]).astype(np.float32)
    return ee_pos, ee_quat.astype(np.float32)


def shift_target_rows(components: dict[str, np.ndarray], steps: int) -> None:
    if steps <= 0:
        return
    count = components["actions"].shape[0]
    idx = np.clip(np.arange(count, dtype=np.int64) + int(steps), 0, count - 1)
    components["target_pos"] = components["target_pos"][idx].astype(np.float32)
    components["target_quat"] = components["target_quat"][idx].astype(np.float32)


def interp_rows(values: np.ndarray, src_t: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.stack(
        [np.interp(dst_t, src_t, values[:, axis]) for axis in range(values.shape[1])],
        axis=1,
    ).astype(np.float32)


def interp_quats_wxyz(quat: np.ndarray, src_t: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
    quat = normalize_quat_wxyz(quat)
    for idx in range(1, quat.shape[0]):
        if float(np.dot(quat[idx - 1], quat[idx])) < 0.0:
            quat[idx] *= -1.0
    return normalize_quat_wxyz(interp_rows(quat, src_t, dst_t))


def nearest_rows(values: np.ndarray, src_t: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    right = np.searchsorted(src_t, dst_t, side="left")
    right = np.clip(right, 0, len(src_t) - 1)
    left = np.clip(right - 1, 0, len(src_t) - 1)
    use_left = np.abs(dst_t - src_t[left]) <= np.abs(src_t[right] - dst_t)
    indices = np.where(use_left, left, right)
    return values[indices]


def resample_components(
    components: dict[str, np.ndarray],
    duration_s: float,
    resample_dt: float,
) -> dict[str, np.ndarray]:
    count = components["actions"].shape[0]
    require(count >= 2, "resampling requires at least two source action rows")
    require(duration_s > 0.0, "resampling requires a positive manifest duration")
    require(resample_dt > 0.0, "--resample-dt must be positive")

    dst_count = max(2, int(math.ceil(duration_s / resample_dt)) + 1)
    src_t = np.linspace(0.0, duration_s, count, dtype=np.float64)
    dst_t = np.linspace(0.0, duration_s, dst_count, dtype=np.float64)

    out: dict[str, np.ndarray] = {}
    linear_keys = {
        "actions",
        "base_pos",
        "base_lin_vel",
        "base_ang_vel",
        "joint_pos",
        "ee_pos",
        "ee_lin_vel",
        "ee_ang_vel",
        "target_pos",
        "min_obstacle_dist",
    }
    quat_keys = {"base_quat", "ee_quat", "target_quat"}
    nearest_keys = {"action_valid_mask", "contact_forces"}

    for key, value in components.items():
        if key in linear_keys:
            out[key] = interp_rows(value, src_t, dst_t)
        elif key in quat_keys:
            out[key] = interp_quats_wxyz(value, src_t, dst_t)
        elif key in nearest_keys:
            out[key] = nearest_rows(value, src_t, dst_t)
        else:
            out[key] = value

    dt = np.full((dst_count,), float(resample_dt), dtype=np.float32)
    joint_pos = out["joint_pos"].astype(np.float32)
    joint_next = np.empty_like(joint_pos)
    joint_next[:-1] = joint_pos[1:]
    joint_next[-1] = joint_pos[-1]
    out["joint_vel"] = ((joint_next - joint_pos) / dt[:, None]).astype(np.float32)
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


def apply_env_state_observation_sources(
    components: dict[str, np.ndarray],
    args: argparse.Namespace,
    dt_value: float,
) -> None:
    dt = np.full((components["actions"].shape[0],), float(dt_value), dtype=np.float32)
    joint_pos = components["joint_pos"].astype(np.float32)
    if args.velocity_source == "lagged_q":
        q_delta = backward_difference(joint_pos, dt)
        yaw = joint_pos[:, 2]
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        base_lin_vel = np.zeros((joint_pos.shape[0], 3), dtype=np.float32)
        base_lin_vel[:, 0] = (cos_yaw * q_delta[:, 0] - sin_yaw * q_delta[:, 1]) / MAX_LINEAR_VELOCITY
        base_lin_vel[:, 1] = (sin_yaw * q_delta[:, 0] + cos_yaw * q_delta[:, 1]) / MAX_LINEAR_VELOCITY
        base_ang_vel = np.zeros((joint_pos.shape[0], 3), dtype=np.float32)
        base_ang_vel[:, 2] = q_delta[:, 2] / MAX_ANGULAR_VELOCITY
        components["base_lin_vel"] = base_lin_vel
        components["base_ang_vel"] = base_ang_vel
        components["joint_vel"] = q_delta

    if args.ee_state_source == "fk_current":
        ee_pos, ee_quat = fk_ee_from_q(joint_pos, args.fk_chain)
        if args.flip_fk_ee_quat:
            ee_quat = -ee_quat
        components["ee_pos"] = ee_pos
        components["ee_quat"] = ee_quat
        components["ee_lin_vel"] = backward_difference(ee_pos, dt)
        components["ee_ang_vel"] = backward_angular_velocity_from_quats(ee_quat, dt)


def apply_arm_envelope_profile(
    actions: np.ndarray,
    mask: np.ndarray,
    q_next: np.ndarray,
    profile: str,
) -> tuple[np.ndarray, np.ndarray]:
    if profile == "stored":
        return actions, mask
    rebuilt_actions, _, arm_valid = normalize_arm_targets(q_next[:, 3:9], profile=profile)
    actions = actions.copy()
    mask = mask.copy()
    actions[:, :6] = rebuilt_actions
    mask[:, :6] = arm_valid
    return actions, mask


def resolve_npz_path(item: dict, demo_dir: Path) -> Path:
    npz_path = Path(item["output_npz"])
    if npz_path.exists():
        return npz_path
    fallback = demo_dir / npz_path.name
    require(fallback.exists(), f"missing npz: {npz_path}")
    return fallback


def build_components(
    data: np.lib.npyio.NpzFile,
    args: argparse.Namespace,
    duration_s: float | None = None,
) -> dict[str, np.ndarray]:
    actions = data["actions"].astype(np.float32)
    q_current = data["q_current"].astype(np.float32)
    q_next = data["q_next"].astype(np.float32)
    action_valid_mask = data["action_valid_mask"].astype(bool)
    actions, action_valid_mask = apply_arm_envelope_profile(
        actions,
        action_valid_mask,
        q_next,
        args.arm_envelope_profile,
    )
    dt = data["dt"].astype(np.float32)
    target_pos = data["target_pos"].astype(np.float32)
    target_quat = data["target_quat_wxyz"].astype(np.float32)
    ee_pos = data["actual_ee_pos"].astype(np.float32)
    ee_quat = data["actual_ee_quat_wxyz"].astype(np.float32)

    yaw = q_current[:, 2]
    base_pos = np.zeros((actions.shape[0], 3), dtype=np.float32)
    base_pos[:, :2] = q_current[:, :2]
    base_quat = yaw_to_quat_wxyz(yaw).astype(np.float32)

    vx_body = actions[:, 6] * MAX_LINEAR_VELOCITY
    vy_body = actions[:, 7] * MAX_LINEAR_VELOCITY
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    base_lin_vel = np.zeros((actions.shape[0], 3), dtype=np.float32)
    base_lin_vel[:, 0] = (cos_yaw * vx_body - sin_yaw * vy_body) / MAX_LINEAR_VELOCITY
    base_lin_vel[:, 1] = (sin_yaw * vx_body + cos_yaw * vy_body) / MAX_LINEAR_VELOCITY
    base_ang_vel = np.zeros((actions.shape[0], 3), dtype=np.float32)
    base_ang_vel[:, 2] = actions[:, 8]

    joint_pos = q_current.astype(np.float32)
    joint_vel = ((q_next - q_current) / dt[:, None]).astype(np.float32)
    ee_lin_vel = finite_difference(ee_pos, dt).astype(np.float32)
    ee_ang_vel = angular_velocity_from_quats(ee_quat, dt).astype(np.float32)
    lookahead = build_lookahead(target_pos, args.lookahead_steps).astype(np.float32)
    action_history = build_action_history(actions, args.action_history_length).astype(np.float32)
    contact = np.zeros((actions.shape[0], args.num_contacts), dtype=np.float32)

    components = {
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
        "ee_lin_vel": ee_lin_vel,
        "ee_ang_vel": ee_ang_vel,
        "target_pos": target_pos,
        "target_quat": target_quat,
        "lookahead_pos": lookahead,
        "action_history": action_history,
        "contact_forces": contact,
    }
    if args.use_obstacles:
        clearance = data["min_obstacle_dist"].astype(np.float32) if "min_obstacle_dist" in data else None
        if clearance is None or not np.isfinite(clearance).any():
            obstacle = np.full((actions.shape[0], 1), 5.0, dtype=np.float32)
        else:
            clearance = np.nan_to_num(clearance, nan=5.0 * args.safety_radius)
            obstacle = np.clip(clearance / max(args.safety_radius, 1e-6), -2.0, 5.0)[:, None].astype(np.float32)
        components["min_obstacle_dist"] = obstacle
    if args.resample_dt > 0.0:
        require(duration_s is not None, "--resample-dt requires manifest items with duration_s")
        components = resample_components(components, float(duration_s), float(args.resample_dt))
        components["action_history"] = build_action_history(
            components["actions"],
            args.action_history_length,
        ).astype(np.float32)
        apply_env_state_observation_sources(components, args, float(args.resample_dt))
    else:
        apply_env_state_observation_sources(components, args, float(np.median(dt)))
    shift_target_rows(components, args.target_shift_steps)
    components["lookahead_pos"] = build_lookahead(components["target_pos"], args.lookahead_steps).astype(np.float32)
    return components


def compose_batches(components: dict[str, np.ndarray], batch_size: int) -> np.ndarray:
    outputs: list[np.ndarray] = []
    count = components["actions"].shape[0]
    tensor_keys = [k for k in components if k not in {"actions", "action_valid_mask"}]
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        kwargs = {
            key: torch.from_numpy(components[key][start:end]).float()
            for key in tensor_keys
        }
        with torch.no_grad():
            obs = compose_observation(**kwargs).cpu().numpy().astype(np.float32)
        outputs.append(obs)
    return np.concatenate(outputs, axis=0)


def main() -> int:
    args = parse_args()
    demo_dir = args.demo_dir.resolve()
    manifest_path = demo_dir / args.manifest
    require(manifest_path.exists(), f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = manifest.get("items", [])
    require(items, "manifest contains no items")
    if args.ee_state_source == "fk_current":
        urdf_path = args.urdf.resolve()
        joints = parse_urdf(urdf_path)
        args.fk_chain = chain_to_link(joints, "base_root", EE_LINK_NAME)

    all_obs: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []
    source_index: list[np.ndarray] = []
    source_files: list[str] = []

    for idx, item in enumerate(items):
        npz_path = resolve_npz_path(item, demo_dir)
        with np.load(npz_path) as data:
            components = build_components(data, args, duration_s=item.get("duration_s"))
            obs = compose_batches(components, args.batch_size)
            actions = components["actions"]
            mask = components["action_valid_mask"].copy()
            if args.base_only:
                mask[:, :6] = False
        all_obs.append(obs)
        all_actions.append(actions)
        all_masks.append(mask)
        source_index.append(np.full(actions.shape[0], idx, dtype=np.int32))
        source_files.append(npz_path.name)

    observations = np.concatenate(all_obs, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    action_valid_mask = np.concatenate(all_masks, axis=0)
    source_index_arr = np.concatenate(source_index, axis=0)

    expected_dim = get_observation_dimensions(
        num_joints=6,
        num_contacts=args.num_contacts,
        use_lookahead=True,
        lookahead_steps=args.lookahead_steps,
        use_action_history=True,
        action_history_length=args.action_history_length,
        action_dim=ACTION_DIM,
        use_obstacles=args.use_obstacles,
    )
    require(observations.shape[1] == expected_dim, f"obs dim {observations.shape[1]} != expected {expected_dim}")
    require(np.isfinite(observations).all(), "observations contain non-finite values")
    require(np.isfinite(actions).all(), "actions contain non-finite values")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        observations=observations,
        actions=actions,
        action_valid_mask=action_valid_mask,
        source_index=source_index_arr,
        source_files=np.asarray(source_files),
        manifest=str(manifest_path),
        base_only=np.asarray(args.base_only),
        observation_dim=np.asarray(observations.shape[1]),
        use_obstacles=np.asarray(bool(args.use_obstacles)),
        resample_dt=np.asarray(float(args.resample_dt), dtype=np.float32),
        arm_envelope_profile=np.asarray(args.arm_envelope_profile),
        ee_state_source=np.asarray(args.ee_state_source),
        velocity_source=np.asarray(args.velocity_source),
        target_shift_steps=np.asarray(int(args.target_shift_steps), dtype=np.int32),
        flip_fk_ee_quat=np.asarray(bool(args.flip_fk_ee_quat)),
    )
    print(f"Manifest:      {manifest_path}")
    print(f"Output:        {args.output}")
    print(f"Trajectories:  {len(items)}")
    print(f"Samples:       {observations.shape[0]}")
    print(f"Obs dim:       {observations.shape[1]}")
    print(f"Use obstacles: {args.use_obstacles}")
    print(f"EE source:     {args.ee_state_source}")
    print(f"Velocity src:  {args.velocity_source}")
    print(f"Target shift:  {args.target_shift_steps}")
    print(f"Flip FK quat:  {args.flip_fk_ee_quat}")
    if args.resample_dt > 0.0:
        print(f"Resample dt:   {args.resample_dt:.4f}s")
    print(f"Arm envelope:  {args.arm_envelope_profile}")
    print(f"Base labels:   {action_valid_mask[:, 6:].mean(axis=0)}")
    print(f"Arm labels:    {action_valid_mask[:, :6].mean(axis=0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
