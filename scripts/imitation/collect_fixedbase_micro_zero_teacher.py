#!/usr/bin/env python3
"""Collect a masked BC baseline for the fixed-base micro stage.

This is intentionally a conservative baseline, not a full IK teacher.  It
samples real observations from the IsaacLab environment while applying zero
actions, then labels the six arm/gimbal policy rows as zero.  In the current
``sim_6joint_gimbal_v1`` contract, zero is the normalized safe-home absolute
joint target.  Base rows are left unlabelled because Stage A freezes them with
``--freeze_base_actions``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--trajectory_stage", default="stage0_fixedbase_micro")
    parser.add_argument("--max_trajectories", type=int, default=24)
    parser.add_argument("--min_trajectory_duration", type=float, default=5.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/fixedbase_micro_zero_teacher/obs_dataset_zero_arm6.npz"),
    )
    return parser.parse_args()


def tensor_np(value):
    if isinstance(value, tuple):
        value = value[0]
    if isinstance(value, dict):
        value = value.get("policy", next(iter(value.values())))
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    import numpy as np

    return np.asarray(value, dtype=np.float32)


def main() -> int:
    args = parse_args()

    import numpy as np
    import torch
    from isaaclab.app import AppLauncher

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    best_device = 0
    best_compute = 0.0
    for index in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(index)
        compute = major + 0.1 * minor
        if compute > best_compute:
            best_compute = compute
            best_device = index

    app_launcher = AppLauncher(
        headless=args.headless,
        enable_cameras=False,
        device=f"cuda:{best_device}",
    )
    simulation_app = app_launcher.app

    try:
        from rl_platform.tasks.mobile_mm import MobileMMTrackEEEnv, MobileMMTrackEEEnvCfg
        from rl_platform.tasks.mobile_mm.config import TrajectoryConfig

        stage_dir = PROJECT_ROOT / "trajectoryToLearn" / args.trajectory_stage
        manifest = stage_dir / "manifest.txt"
        if not manifest.exists():
            raise FileNotFoundError(f"stage manifest not found: {manifest}")

        env_cfg = MobileMMTrackEEEnvCfg()
        env_cfg.num_envs = args.num_envs
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = args.seed
        env_cfg.task_config.obstacles.enable_obstacles = False
        env_cfg.task_config.base_assist.enable = False
        env_cfg.task_config.trajectory = TrajectoryConfig(
            type="multi_recorded",
            trajectory_dir=str(PROJECT_ROOT),
            trajectory_pattern="**/*.json",
            trajectory_manifest_file=str(manifest),
            max_trajectories=args.max_trajectories,
            min_duration_seconds=args.min_trajectory_duration,
            randomize_start_waypoint=False,
            reset_base_to_trajectory_start=False,
        )

        env = MobileMMTrackEEEnv(cfg=env_cfg)
        obs = env.reset()

        obs_chunks: list[np.ndarray] = []
        action_chunks: list[np.ndarray] = []
        mask_chunks: list[np.ndarray] = []
        pos_error_chunks: list[np.ndarray] = []

        collected = 0
        zero_actions = torch.zeros((args.num_envs, 9), dtype=torch.float32, device=env.device)
        while collected < args.samples:
            obs_np = tensor_np(obs)
            take = min(args.samples - collected, obs_np.shape[0])
            obs_chunks.append(obs_np[:take].copy())

            actions = np.zeros((take, 9), dtype=np.float32)
            mask = np.zeros((take, 9), dtype=bool)
            mask[:, 0:6] = True
            action_chunks.append(actions)
            mask_chunks.append(mask)

            if hasattr(env, "ee_pos_error_buf"):
                pos_error = env.ee_pos_error_buf.detach().cpu().numpy().astype(np.float32)
                pos_error_chunks.append(pos_error[:take].copy())

            collected += take
            obs, _rewards, _terminated, _truncated, _infos = env.step(zero_actions)

        observations = np.concatenate(obs_chunks, axis=0)
        actions = np.concatenate(action_chunks, axis=0)
        action_valid_mask = np.concatenate(mask_chunks, axis=0)
        pos_errors = np.concatenate(pos_error_chunks, axis=0) if pos_error_chunks else np.empty((0, 3), dtype=np.float32)

        metadata = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "schema": "fixedbase_micro_zero_teacher_v1",
            "trajectory_stage": args.trajectory_stage,
            "manifest": str(manifest),
            "samples": int(observations.shape[0]),
            "obs_dim": int(observations.shape[1]),
            "act_dim": int(actions.shape[1]),
            "labelled_action_rows": [0, 1, 2, 3, 4, 5],
            "label_semantics": "zero normalized sim_6joint_gimbal_v1 arm/gimbal safe-home targets",
            "base_rows": "unlabelled; use --freeze_base_actions during Stage A",
            "zero_action_pos_error_mean_m": float(np.linalg.norm(pos_errors, axis=1).mean()) if pos_errors.size else None,
            "zero_action_pos_error_p95_m": float(np.percentile(np.linalg.norm(pos_errors, axis=1), 95)) if pos_errors.size else None,
        }

        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            observations=observations,
            actions=actions,
            action_valid_mask=action_valid_mask,
            metadata=json.dumps(metadata, indent=2),
            observation_dim=np.asarray(observations.shape[1], dtype=np.int32),
            action_contract=np.asarray("sim_6joint_gimbal_v1"),
            action_names=np.asarray(
                [
                    "joint6_arm_yaw",
                    "joint5_arm_pitch",
                    "joint4_elbow_pitch",
                    "joint3_gimbal_yaw",
                    "joint2_gimbal_roll",
                    "joint1_gimbal_pitch",
                    "base_vx",
                    "base_vy",
                    "base_wz",
                ]
            ),
        )
        print(f"saved: {args.output}")
        print(f"observations: {observations.shape}, actions: {actions.shape}")
        print(f"valid action counts: {action_valid_mask.sum(axis=0).astype(int).tolist()}")
        print(f"metadata: {json.dumps(metadata, indent=2)}")
        env.close()
    finally:
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
