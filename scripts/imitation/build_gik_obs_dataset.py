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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-dir", type=Path, default=Path("data/gik_ik_demos"))
    parser.add_argument("--manifest", type=str, default="manifest_strict.json")
    parser.add_argument("--output", type=Path, default=Path("data/gik_ik_demos/obs_dataset_strict.npz"))
    parser.add_argument("--base-only", action="store_true", help="Mask imitation labels to base_vx/base_vy/base_wz only.")
    parser.add_argument("--lookahead-steps", type=int, default=3)
    parser.add_argument("--action-history-length", type=int, default=2)
    parser.add_argument("--num-contacts", type=int, default=1)
    parser.add_argument("--safety-radius", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=4096)
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


def angular_velocity_from_quats(quat: np.ndarray, dt: np.ndarray) -> np.ndarray:
    if quat.shape[0] == 1:
        return np.zeros((1, 3), dtype=np.float32)
    rel = quat_multiply(quat[1:], quat_conj(quat[:-1]))
    axis_angle = quat_to_axis_angle(rel)
    out = np.zeros((quat.shape[0], 3), dtype=np.float32)
    out[:-1] = axis_angle / dt[:-1, None]
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


def resolve_npz_path(item: dict, demo_dir: Path) -> Path:
    npz_path = Path(item["output_npz"])
    if npz_path.exists():
        return npz_path
    fallback = demo_dir / npz_path.name
    require(fallback.exists(), f"missing npz: {npz_path}")
    return fallback


def build_components(data: np.lib.npyio.NpzFile, args: argparse.Namespace) -> dict[str, np.ndarray]:
    actions = data["actions"].astype(np.float32)
    q_current = data["q_current"].astype(np.float32)
    q_next = data["q_next"].astype(np.float32)
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

    clearance = data["min_obstacle_dist"].astype(np.float32) if "min_obstacle_dist" in data else None
    if clearance is None or not np.isfinite(clearance).any():
        obstacle = np.full((actions.shape[0], 1), 5.0, dtype=np.float32)
    else:
        clearance = np.nan_to_num(clearance, nan=5.0 * args.safety_radius)
        obstacle = np.clip(clearance / max(args.safety_radius, 1e-6), -2.0, 5.0)[:, None].astype(np.float32)

    return {
        "actions": actions,
        "action_valid_mask": data["action_valid_mask"].astype(bool),
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
        "min_obstacle_dist": obstacle,
    }


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

    all_obs: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []
    source_index: list[np.ndarray] = []
    source_files: list[str] = []

    for idx, item in enumerate(items):
        npz_path = resolve_npz_path(item, demo_dir)
        with np.load(npz_path) as data:
            components = build_components(data, args)
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
        use_obstacles=True,
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
    )
    print(f"Manifest:      {manifest_path}")
    print(f"Output:        {args.output}")
    print(f"Trajectories:  {len(items)}")
    print(f"Samples:       {observations.shape[0]}")
    print(f"Obs dim:       {observations.shape[1]}")
    print(f"Base labels:   {action_valid_mask[:, 6:].mean(axis=0)}")
    print(f"Arm labels:    {action_valid_mask[:, :6].mean(axis=0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
