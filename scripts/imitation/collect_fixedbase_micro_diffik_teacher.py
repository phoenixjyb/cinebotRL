#!/usr/bin/env python3
"""Collect a DifferentialIK masked BC dataset for stage0_fixedbase_micro.

The output follows the existing BC trainer schema:

    observations, actions, action_valid_mask

Rows 0..5 are normalized absolute arm/gimbal joint targets for the current
``sim_6joint_gimbal_v1`` action contract.  Base rows 6..8 are left unlabelled
because this Stage A gate freezes base actions.
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
    parser.add_argument("--ik_damping", type=float, default=0.05)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/fixedbase_micro_diffik_teacher/obs_dataset_diffik_arm6.npz"),
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


def normalize_targets(targets, lower, upper):
    import torch

    raw = 2.0 * (targets - lower.unsqueeze(0)) / (upper - lower).unsqueeze(0) - 1.0
    valid = torch.isfinite(raw) & (torch.abs(raw) <= 1.0)
    return torch.clamp(raw, -1.0, 1.0), valid


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

    app_launcher = AppLauncher(headless=args.headless, enable_cameras=False, device="cuda:0")
    simulation_app = app_launcher.app

    try:
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
        from rl_platform.tasks.mobile_mm import MobileMMTrackEEEnv, MobileMMTrackEEEnvCfg
        from rl_platform.tasks.mobile_mm.config import TrajectoryConfig
        from rl_platform.tasks.mobile_mm.joint_names import ARM_JOINT_NAMES

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
        env._initialize_ee_body_idx()
        env._verify_joint_mapping()
        env._initialize_joint_limits()
        arm_ids = env._get_joint_ids(ARM_JOINT_NAMES, "_arm_joint_ids")

        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": args.ik_damping},
        )
        ik = DifferentialIKController(cfg=ik_cfg, num_envs=args.num_envs, device=env.device)

        lower = env.arm_safe_home.to(env.device) - env.arm_action_radius.to(env.device)
        upper = env.arm_safe_home.to(env.device) + env.arm_action_radius.to(env.device)

        obs_chunks: list[np.ndarray] = []
        action_chunks: list[np.ndarray] = []
        mask_chunks: list[np.ndarray] = []
        raw_target_chunks: list[np.ndarray] = []
        pos_error_chunks: list[np.ndarray] = []

        collected = 0
        while collected < args.samples:
            obs_np = tensor_np(obs)
            take = min(args.samples - collected, obs_np.shape[0])

            target_pos, target_quat = env.trajectory_manager.get_target_pose()
            ee_pos = env.robot.data.body_pos_w[:, env._ee_body_idx, :]
            ee_quat = env.robot.data.body_quat_w[:, env._ee_body_idx, :]
            joint_pos = env.robot.data.joint_pos[:, arm_ids]
            jacobians = env.robot.root_physx_view.get_jacobians()
            if jacobians.ndim != 4:
                raise RuntimeError(f"expected 4D jacobian tensor, got {tuple(jacobians.shape)}")
            jacobian = jacobians[:, env._ee_body_idx, :, arm_ids]

            ik.reset()
            ik.set_command(torch.cat([target_pos, target_quat], dim=-1))
            target_joints = ik.compute(
                ee_pos=ee_pos,
                ee_quat=ee_quat,
                jacobian=jacobian,
                joint_pos=joint_pos,
            )
            arm_actions, arm_valid = normalize_targets(target_joints, lower, upper)
            actions_t = torch.zeros((args.num_envs, 9), dtype=torch.float32, device=env.device)
            actions_t[:, 0:6] = arm_actions

            mask_t = torch.zeros_like(actions_t, dtype=torch.bool)
            mask_t[:, 0:6] = arm_valid

            obs_chunks.append(obs_np[:take].copy())
            action_chunks.append(actions_t[:take].detach().cpu().numpy().astype(np.float32))
            mask_chunks.append(mask_t[:take].detach().cpu().numpy().astype(bool))
            raw_target_chunks.append(target_joints[:take].detach().cpu().numpy().astype(np.float32))
            if hasattr(env, "ee_pos_error_buf"):
                pos_error_chunks.append(env.ee_pos_error_buf[:take].detach().cpu().numpy().astype(np.float32))

            collected += take
            obs, _rewards, _terminated, _truncated, _infos = env.step(actions_t)

        observations = np.concatenate(obs_chunks, axis=0)
        actions = np.concatenate(action_chunks, axis=0)
        action_valid_mask = np.concatenate(mask_chunks, axis=0)
        raw_targets = np.concatenate(raw_target_chunks, axis=0)
        pos_errors = np.concatenate(pos_error_chunks, axis=0) if pos_error_chunks else np.empty((0, 3), dtype=np.float32)

        valid_counts = action_valid_mask.sum(axis=0).astype(int).tolist()
        valid_fraction = action_valid_mask.mean(axis=0).astype(float).tolist()
        metadata = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "schema": "fixedbase_micro_diffik_teacher_v1",
            "trajectory_stage": args.trajectory_stage,
            "manifest": str(manifest),
            "samples": int(observations.shape[0]),
            "obs_dim": int(observations.shape[1]),
            "act_dim": int(actions.shape[1]),
            "labelled_action_rows": [0, 1, 2, 3, 4, 5],
            "valid_action_counts": valid_counts,
            "valid_action_fraction": valid_fraction,
            "ik_damping": float(args.ik_damping),
            "base_rows": "unlabelled; use --freeze_base_actions during Stage A",
            "teacher_rollout_pos_error_mean_m": float(np.linalg.norm(pos_errors, axis=1).mean()) if pos_errors.size else None,
            "teacher_rollout_pos_error_p95_m": float(np.percentile(np.linalg.norm(pos_errors, axis=1), 95)) if pos_errors.size else None,
        }

        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            observations=observations,
            actions=actions,
            action_valid_mask=action_valid_mask,
            teacher_joint_targets=raw_targets,
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
        print(f"valid action counts: {valid_counts}")
        print(f"metadata: {json.dumps(metadata, indent=2)}")
        env.close()
    finally:
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
