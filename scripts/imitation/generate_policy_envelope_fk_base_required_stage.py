#!/usr/bin/env python3
"""Generate a small base-required stage from a validated policy-envelope FK stage.

The source FK stage already gives self-consistent arm/gimbal targets and labels.
This script adds a smooth, low-amplitude world-frame base offset to each target
path and labels the base rows so a policy running with ``--base_action_scale`` can
track the moving target by moving the chassis while preserving the arm labels.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

ACTION_DIM = 9
ARM_SAFE_HOME = np.asarray([0.0, 1.0, -1.2, 0.0, 0.0, 0.0], dtype=np.float32)
ARM_ACTION_RADIUS = np.asarray([1.0, 0.45, 0.8, 1.0, 0.8, 0.8], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_stage", default="stage0_policy_envelope_fk_large08")
    parser.add_argument(
        "--source_dataset",
        type=Path,
        default=Path("data/policy_envelope_fk_large08/obs_dataset_policy_envelope_fk_large08_arm6.npz"),
    )
    parser.add_argument("--stage", default="stage0_policy_envelope_fk_base025")
    parser.add_argument(
        "--output_dataset",
        type=Path,
        default=Path("data/policy_envelope_fk_base025/obs_dataset_policy_envelope_fk_base025_arm6_base3.npz"),
    )
    parser.add_argument("--base_action_scale", type=float, default=0.25)
    parser.add_argument("--base_offset_radius", type=float, default=0.10)
    parser.add_argument("--max_linear_velocity", type=float, default=1.5)
    parser.add_argument("--waypoint_dt", type=float, default=0.1)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def xyzw_to_wxyz(quat_xyzw: list[float]) -> list[float]:
    return [float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])]


def wxyz_to_xyzw(quat_wxyz: np.ndarray) -> list[float]:
    return [
        round(float(quat_wxyz[1]), 6),
        round(float(quat_wxyz[2]), 6),
        round(float(quat_wxyz[3]), 6),
        round(float(quat_wxyz[0]), 6),
    ]


def quat_conj_np(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def quat_multiply_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = [q1[..., i] for i in range(4)]
    w2, x2, y2, z2 = [q2[..., i] for i in range(4)]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def quat_to_axis_angle_np(q: np.ndarray) -> np.ndarray:
    q = q.copy()
    q /= np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    q *= np.where(q[..., :1] < 0.0, -1.0, 1.0)
    xyz = q[..., 1:]
    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(norm, np.clip(q[..., :1], -1.0, 1.0))
    return xyz / np.maximum(norm, 1e-8) * angle


def load_source_stage(source_stage: str) -> tuple[list[Path], list[list[dict]]]:
    stage_dir = PROJECT_ROOT / "trajectoryToLearn" / source_stage
    manifest = stage_dir / "manifest.txt"
    require(manifest.exists(), f"source manifest not found: {manifest}")
    paths: list[Path] = []
    trajectories: list[list[dict]] = []
    for raw_line in manifest.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        path = PROJECT_ROOT / line
        require(path.exists(), f"source trajectory missing: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        poses = data.get("poses")
        require(isinstance(poses, list) and poses, f"invalid poses in {path}")
        paths.append(path)
        trajectories.append(poses)
    return paths, trajectories


def smooth_base_offset(traj_idx: int, steps: int, radius: float) -> np.ndarray:
    offset = np.zeros((steps, 3), dtype=np.float32)
    phase = traj_idx * math.pi / 6.0
    direction = np.asarray([math.cos(phase), math.sin(phase)], dtype=np.float32)
    lateral = np.asarray([-direction[1], direction[0]], dtype=np.float32)
    for step in range(steps):
        u = step / max(steps - 1, 1)
        forward = radius * (0.5 - 0.5 * math.cos(math.pi * u))
        weave = 0.25 * radius * math.sin(2.0 * math.pi * u + phase)
        offset[step, :2] = forward * direction + weave * lateral
    offset[0, :] = 0.0
    return offset


def main() -> int:
    args = parse_args()
    require(0.0 < args.base_action_scale <= 1.0, "--base_action_scale must be in (0, 1]")
    require(args.base_offset_radius > 0.0, "--base_offset_radius must be positive")
    require(args.waypoint_dt > 0.0, "--waypoint_dt must be positive")

    from rl_platform.tasks.mobile_mm.observations import compose_observation, get_observation_dimensions

    source_dataset = args.source_dataset
    if not source_dataset.is_absolute():
        source_dataset = PROJECT_ROOT / source_dataset
    require(source_dataset.exists(), f"source dataset not found: {source_dataset}")
    with np.load(source_dataset, allow_pickle=False) as data:
        source_actions = data["actions"].astype(np.float32)
    require(source_actions.shape[1] == ACTION_DIM, f"expected action dim {ACTION_DIM}, got {source_actions.shape[1]}")

    _, source_trajectories = load_source_stage(args.source_stage)
    num_trajectories = len(source_trajectories)
    require(num_trajectories > 0, "source stage has no trajectories")
    num_waypoints = len(source_trajectories[0])
    require(num_waypoints >= 50, "source trajectories must be at least 5s at 0.1s dt")
    require(source_actions.shape[0] == num_trajectories * num_waypoints, "dataset/stage size mismatch")
    require(all(len(traj) == num_waypoints for traj in source_trajectories), "source trajectories have uneven lengths")

    stage_dir = PROJECT_ROOT / "trajectoryToLearn" / args.stage
    generated_dir = stage_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    manifest_lines = [
        f"# Generated by scripts/imitation/generate_policy_envelope_fk_base_required_stage.py for {args.stage}",
        "# Paths are relative to the project root.",
    ]
    observations = []
    actions = []
    masks = []
    base_action_abs_max = 0.0
    base_offset_abs_max = 0.0

    for traj_idx, poses in enumerate(source_trajectories):
        start = traj_idx * num_waypoints
        end = start + num_waypoints
        source_action_rows = source_actions[start:end].copy()
        arm_actions = source_action_rows[:, :6]

        source_pos = np.asarray([pose["position"] for pose in poses], dtype=np.float32)
        source_quat = np.asarray([xyzw_to_wxyz(pose["orientation"]) for pose in poses], dtype=np.float32)
        base_offset = smooth_base_offset(traj_idx, num_waypoints, args.base_offset_radius)
        target_pos = source_pos + base_offset
        target_quat = source_quat.copy()
        base_offset_abs_max = max(base_offset_abs_max, float(np.max(np.linalg.norm(base_offset[:, :2], axis=1))))

        base_vel = np.zeros((num_waypoints, 3), dtype=np.float32)
        base_vel[:-1] = np.diff(base_offset, axis=0) / args.waypoint_dt
        base_vel[-1] = base_vel[-2]
        base_actions = np.zeros((num_waypoints, 3), dtype=np.float32)
        denom = args.max_linear_velocity * args.base_action_scale
        base_actions[:, 0:2] = base_vel[:, 0:2] / denom
        base_actions = np.clip(base_actions, -1.0, 1.0)
        base_action_abs_max = max(base_action_abs_max, float(np.max(np.abs(base_actions))))

        full_actions = np.zeros((num_waypoints, ACTION_DIM), dtype=np.float32)
        full_actions[:, :6] = arm_actions
        full_actions[:, 6:9] = base_actions

        joint_targets = ARM_SAFE_HOME[None, :] + arm_actions * ARM_ACTION_RADIUS[None, :]
        joint_pos = np.zeros((num_waypoints, 9), dtype=np.float32)
        joint_vel = np.zeros((num_waypoints, 9), dtype=np.float32)
        joint_pos[:, 3:9] = joint_targets
        joint_vel[:-1, 3:9] = np.diff(joint_targets, axis=0) / args.waypoint_dt
        joint_vel[-1, 3:9] = joint_vel[-2, 3:9]

        ee_lin_vel = np.zeros((num_waypoints, 3), dtype=np.float32)
        ee_lin_vel[:-1] = np.diff(target_pos, axis=0) / args.waypoint_dt
        ee_lin_vel[-1] = ee_lin_vel[-2]
        ee_ang_vel = np.zeros((num_waypoints, 3), dtype=np.float32)
        rel = quat_multiply_np(target_quat[1:], quat_conj_np(target_quat[:-1]))
        ee_ang_vel[:-1] = quat_to_axis_angle_np(rel) / args.waypoint_dt
        ee_ang_vel[-1] = ee_ang_vel[-2]

        lookahead = []
        for step in range(num_waypoints):
            idx = np.clip(step + np.arange(1, 4), 0, num_waypoints - 1)
            lookahead.append(target_pos[idx])
        lookahead_np = np.asarray(lookahead, dtype=np.float32)

        action_history = np.zeros((num_waypoints, 2, ACTION_DIM), dtype=np.float32)
        for step in range(num_waypoints):
            if step - 2 >= 0:
                action_history[step, 0] = full_actions[step - 2]
            if step - 1 >= 0:
                action_history[step, 1] = full_actions[step - 1]

        with torch.no_grad():
            obs = compose_observation(
                base_pos=torch.from_numpy(base_offset).float(),
                base_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).repeat(num_waypoints, 1),
                base_lin_vel=torch.from_numpy(base_vel).float(),
                base_ang_vel=torch.zeros((num_waypoints, 3), dtype=torch.float32),
                joint_pos=torch.from_numpy(joint_pos).float(),
                joint_vel=torch.from_numpy(joint_vel).float(),
                ee_pos=torch.from_numpy(target_pos).float(),
                ee_quat=torch.from_numpy(target_quat).float(),
                ee_lin_vel=torch.from_numpy(ee_lin_vel).float(),
                ee_ang_vel=torch.from_numpy(ee_ang_vel).float(),
                target_pos=torch.from_numpy(target_pos).float(),
                target_quat=torch.from_numpy(target_quat).float(),
                lookahead_pos=torch.from_numpy(lookahead_np).float(),
                action_history=torch.from_numpy(action_history).float(),
                contact_forces=torch.zeros((num_waypoints, 1), dtype=torch.float32),
                min_obstacle_dist=None,
            ).cpu().numpy().astype(np.float32)

        observations.append(obs)
        actions.append(full_actions)
        mask = np.zeros_like(full_actions, dtype=bool)
        mask[:, :9] = True
        masks.append(mask)

        out_poses = [
            {
                "position": [round(float(v), 6) for v in target_pos[step]],
                "orientation": wxyz_to_xyzw(target_quat[step]),
            }
            for step in range(num_waypoints)
        ]
        path = generated_dir / f"{traj_idx:03d}_policy_envelope_fk_base_required.json"
        path.write_text(json.dumps({"poses": out_poses}, indent=2) + "\n", encoding="utf-8")
        manifest_lines.append(path.relative_to(PROJECT_ROOT).as_posix())

    (stage_dir / "manifest.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    first_pos = np.asarray(source_trajectories[0][0]["position"], dtype=np.float32)
    (stage_dir / "reset_config.json").write_text(
        json.dumps(
            {
                "schema": "policy_envelope_fk_base_required_reset_config_v1",
                "reset_base_x_offset": float(first_pos[0]),
                "reset_base_y_offset": float(first_pos[1]),
                "source": "first source FK waypoint; generated base offset starts at zero",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    observations_np = np.concatenate(observations, axis=0)
    actions_np = np.concatenate(actions, axis=0)
    masks_np = np.concatenate(masks, axis=0)
    expected_dim = get_observation_dimensions(
        num_joints=6,
        num_contacts=1,
        use_lookahead=True,
        lookahead_steps=3,
        use_action_history=True,
        action_history_length=2,
        action_dim=ACTION_DIM,
        use_obstacles=False,
    )
    require(observations_np.shape[1] == expected_dim, f"obs dim {observations_np.shape[1]} != {expected_dim}")
    require(np.isfinite(observations_np).all(), "observations contain non-finite values")
    require(np.isfinite(actions_np).all(), "actions contain non-finite values")
    require(float(np.max(np.abs(actions_np))) <= 1.000001, "actions outside [-1,1]")

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "schema": "policy_envelope_fk_base_required_stage_v1",
        "source_stage": args.source_stage,
        "stage": args.stage,
        "manifest": str(stage_dir / "manifest.txt"),
        "samples": int(observations_np.shape[0]),
        "obs_dim": int(observations_np.shape[1]),
        "act_dim": int(actions_np.shape[1]),
        "num_trajectories": int(num_trajectories),
        "num_waypoints": int(num_waypoints),
        "valid_action_counts": masks_np.sum(axis=0).astype(int).tolist(),
        "base_action_scale": float(args.base_action_scale),
        "base_offset_radius": float(args.base_offset_radius),
        "max_base_offset_xy_m": float(base_offset_abs_max),
        "max_abs_base_action": float(base_action_abs_max),
    }
    output_dataset = args.output_dataset
    if not output_dataset.is_absolute():
        output_dataset = PROJECT_ROOT / output_dataset
    output_dataset.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dataset,
        observations=observations_np,
        actions=actions_np,
        action_valid_mask=masks_np,
        metadata=json.dumps(metadata, indent=2),
        action_contract=np.asarray("sim_6joint_gimbal_base_v1"),
        observation_dim=np.asarray(observations_np.shape[1], dtype=np.int32),
    )
    print(f"stage manifest: {stage_dir / 'manifest.txt'}")
    print(f"dataset: {output_dataset}")
    print(f"observations: {observations_np.shape}, actions: {actions_np.shape}")
    print(f"valid action counts: {metadata['valid_action_counts']}")
    print(f"max base offset xy: {metadata['max_base_offset_xy_m']:.4f} m")
    print(f"max abs base action: {metadata['max_abs_base_action']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
