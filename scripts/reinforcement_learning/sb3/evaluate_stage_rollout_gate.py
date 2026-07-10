#!/usr/bin/env python3
"""Minimal fixed-step rollout gate for Stage A policy checks.

This evaluator intentionally mirrors the known-working training env setup:
stage manifest loading, optional stage reset_config.json, IsaacLab->SB3
observation adaptation, VecNormalize loading, and optional base-action freeze.
It avoids episode-finish bookkeeping so short gates can fail fast.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--recovery_checkpoint",
        default=None,
        help="Optional second PPO checkpoint used only when conditional recovery routing is active.",
    )
    parser.add_argument(
        "--recovery_route_min_waypoint_fraction",
        type=float,
        default=0.65,
        help="Route envs to recovery checkpoint only after this recorded trajectory fraction.",
    )
    parser.add_argument(
        "--recovery_route_min_pos_error",
        type=float,
        default=0.0,
        help="Optional minimum EE position error in meters required for recovery routing.",
    )
    parser.add_argument(
        "--recovery_route_min_base_target_distance",
        type=float,
        default=0.0,
        help="Optional minimum base-target XY distance in meters required for recovery routing.",
    )
    parser.add_argument(
        "--recovery_route_latch_once",
        action="store_true",
        help="Once an env meets the recovery route condition, keep routing it for the rest of the rollout.",
    )
    parser.add_argument(
        "--row_blend_checkpoint",
        default=None,
        help=(
            "Optional second policy whose selected action rows are blended into "
            "the primary policy for diagnostic hybrid gates."
        ),
    )
    parser.add_argument(
        "--row_blend_action_indices",
        default=None,
        help="Comma-separated action rows to blend from --row_blend_checkpoint, e.g. '3,4,5'.",
    )
    parser.add_argument(
        "--row_blend_weight",
        type=float,
        default=1.0,
        help="Blend weight for selected rows: 1.0 fully uses row_blend_checkpoint rows.",
    )
    parser.add_argument("--vec_normalize", required=True)
    parser.add_argument(
        "--disable_vec_normalize",
        action="store_true",
        help="Do not load VecNormalize stats. Use this for raw-observation BC policies.",
    )
    parser.add_argument("--trajectory_stage", default="stage0_policy_envelope_fk")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument(
        "--episode_length_s",
        type=float,
        default=0.0,
        help="Optional env episode length override in seconds. Keep 0 to use the task default.",
    )
    parser.add_argument("--max_trajectories", type=int, default=None)
    parser.add_argument(
        "--assign_loaded_trajectories_once",
        action="store_true",
        help=(
            "Assign loaded trajectories sequentially to envs on reset instead of "
            "sampling with replacement. Use with num_envs equal to the loaded "
            "trajectory count for one-pass per-trajectory gates."
        ),
    )
    parser.add_argument("--min_trajectory_duration", type=float, default=5.0)
    parser.add_argument(
        "--enable_obstacles",
        action="store_true",
        help="Enable obstacle spawning and obstacle-safety metrics during the rollout gate.",
    )
    parser.add_argument(
        "--obstacles_from_trajectory_metadata",
        action="store_true",
        help=(
            "Place each env's obstacle from the currently assigned trajectory "
            "metadata.obstacle.center_xy during reset. Use with exported GIK "
            "one-obstacle stages."
        ),
    )
    parser.add_argument("--obstacle_x", type=float, default=0.0)
    parser.add_argument("--obstacle_y", type=float, default=0.5)
    parser.add_argument("--obstacle_radius", type=float, default=None)
    parser.add_argument("--obstacle_height", type=float, default=None)
    parser.add_argument(
        "--disable_obstacle_randomization",
        action="store_true",
        help="Keep the obstacle at --obstacle_x/--obstacle_y instead of randomizing per reset.",
    )
    parser.add_argument("--obstacle_x_range", type=float, nargs=2, default=(-0.35, 0.35))
    parser.add_argument("--obstacle_y_range", type=float, nargs=2, default=(0.45, 1.0))
    parser.add_argument("--min_obstacle_start_clearance", type=float, default=0.10)
    parser.add_argument(
        "--random_start_waypoint",
        action="store_true",
        help="Start each recorded trajectory from a random waypoint during reset.",
    )
    parser.add_argument("--start_waypoint_min_fraction", type=float, default=0.0)
    parser.add_argument("--start_waypoint_max_fraction", type=float, default=0.0)
    parser.add_argument(
        "--reset_base_to_trajectory_start",
        action="store_true",
        help="Anchor the base near waypoint zero even when target playback starts later.",
    )
    parser.add_argument("--reset_anchor_target_blend", type=float, default=0.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--freeze_base_actions", action="store_true")
    parser.add_argument(
        "--arm_action_envelope_profile",
        type=str,
        default="proto2_safe_v1",
        choices=["proto2_safe_v1", "teacher_wide_v1"],
        help="Physical scaling envelope used for normalized arm action rows [0:6].",
    )
    parser.add_argument(
        "--base_action_scale",
        type=float,
        default=1.0,
        help=(
            "Multiply base action rows [6,7,8] by this factor before env dynamics. "
            "Ignored when --freeze_base_actions is set."
        ),
    )
    parser.add_argument(
        "--enable_initial_joint_randomization",
        action="store_true",
        help="Keep startup joint noise enabled. Default disables it for deterministic Stage A contract checks.",
    )
    parser.add_argument(
        "--open_loop_actions_npz",
        default=None,
        help="Optional .npz dataset whose actions are replayed instead of model predictions.",
    )
    parser.add_argument("--action_sequence_index", type=int, default=0)
    parser.add_argument("--output_json", default=None)
    parser.add_argument(
        "--output_dataset_npz",
        default=None,
        help=(
            "Optional .npz dataset of pre-step rollout observations and actions. "
            "Useful for closed-loop DAgger/distillation from the actual gate state distribution."
        ),
    )
    parser.add_argument(
        "--output_base_teacher_npz",
        default=None,
        help="Optional masked-action dataset with direct base-correction labels from pre-step rollout states.",
    )
    parser.add_argument(
        "--base_teacher_mode",
        choices=["target_direction", "target_offset_follow"],
        default="target_offset_follow",
    )
    parser.add_argument("--base_teacher_activation_distance", type=float, default=0.25)
    parser.add_argument("--base_teacher_full_speed_distance", type=float, default=0.90)
    parser.add_argument("--base_teacher_max_action", type=float, default=1.0)
    parser.add_argument("--base_teacher_lookahead_steps", type=int, default=0)
    parser.add_argument("--base_teacher_include_yaw", action="store_true")
    parser.add_argument("--base_teacher_yaw_max_action", type=float, default=0.5)
    parser.add_argument("--base_teacher_yaw_full_error", type=float, default=1.2)
    parser.add_argument(
        "--base_teacher_active_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only save rows where the direct teacher is active.",
    )
    parser.add_argument("--base_teacher_sample_weight_distance_threshold", type=float, default=0.55)
    parser.add_argument("--base_teacher_sample_weight_full_distance", type=float, default=1.20)
    parser.add_argument("--base_teacher_sample_weight_max", type=float, default=4.0)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def tensor_np(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def parse_action_indices(raw: str | None) -> list[int]:
    if raw is None or not raw.strip():
        return []
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def load_stage_reset_config(stage: str) -> dict[str, object]:
    path = PROJECT_ROOT / "trajectoryToLearn" / stage / "reset_config.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, object] = {}
    if "reset_base_x_offset" in data:
        out["reset_base_x_offset"] = float(data["reset_base_x_offset"])
    if "reset_base_y_offset" in data:
        out["reset_base_y_offset"] = float(data["reset_base_y_offset"])
    if "reset_arm_to_trajectory_metadata" in data:
        out["reset_arm_to_trajectory_metadata"] = bool(data["reset_arm_to_trajectory_metadata"])
    raw_reward_overrides = data.get("reward_overrides", {})
    if isinstance(raw_reward_overrides, dict):
        out["reward_overrides"] = {
            str(name): float(value)
            for name, value in raw_reward_overrides.items()
        }
    return out


def quat_to_yaw_torch(quat):
    import torch

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))


def compute_direct_base_teacher(raw_env, args):
    import torch

    target_pos, _ = raw_env.trajectory_manager.get_target_pose()
    target_xy = target_pos[:, :2]
    lead_steps = max(int(args.base_teacher_lookahead_steps), 0)
    if lead_steps > 0:
        lead_pos, _ = raw_env.trajectory_manager.get_lookahead(
            steps=lead_steps,
            lookahead_dt=float(raw_env.task_cfg.lookahead_dt),
        )
        target_xy = lead_pos[:, -1, :2]

    base_xy = raw_env.robot.data.root_pos_w[:, :2]
    base_to_target = target_xy - base_xy
    if args.base_teacher_mode == "target_offset_follow":
        traj_cfg = raw_env.task_cfg.trajectory
        desired_offset = torch.tensor(
            [
                float(getattr(traj_cfg, "reset_base_x_offset", 0.4415)),
                float(getattr(traj_cfg, "reset_base_y_offset", 0.2405)),
            ],
            dtype=target_xy.dtype,
            device=target_xy.device,
        )
        expert_vector_world = target_xy - desired_offset.unsqueeze(0) - base_xy
        expert_distance = torch.norm(expert_vector_world, dim=-1)
    else:
        expert_vector_world = base_to_target
        expert_distance = torch.norm(expert_vector_world, dim=-1)

    target_dir_world = expert_vector_world / (expert_distance.unsqueeze(-1) + 1e-6)
    theta = quat_to_yaw_torch(raw_env.robot.data.root_quat_w)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    expert_body = torch.stack(
        (
            cos_theta * target_dir_world[:, 0] + sin_theta * target_dir_world[:, 1],
            -sin_theta * target_dir_world[:, 0] + cos_theta * target_dir_world[:, 1],
        ),
        dim=-1,
    )

    activation_distance = max(float(args.base_teacher_activation_distance), 1e-6)
    full_speed_distance = max(float(args.base_teacher_full_speed_distance), activation_distance + 1e-6)
    speed_fraction = torch.clamp(
        (expert_distance - activation_distance) / (full_speed_distance - activation_distance),
        min=0.0,
        max=1.0,
    )
    expert_xy = expert_body * (float(args.base_teacher_max_action) * speed_fraction).unsqueeze(-1)
    expert_xy = torch.nan_to_num(expert_xy, nan=0.0, posinf=0.0, neginf=0.0)
    expert_xy = torch.clamp(expert_xy, -1.0, 1.0)
    active = expert_distance > activation_distance

    expert_wz = torch.zeros((raw_env.num_envs, 1), dtype=expert_xy.dtype, device=expert_xy.device)
    if bool(args.base_teacher_include_yaw):
        target_heading = torch.atan2(base_to_target[:, 1], base_to_target[:, 0])
        yaw_error = torch.atan2(torch.sin(target_heading - theta), torch.cos(target_heading - theta))
        yaw_full_error = max(float(args.base_teacher_yaw_full_error), 1e-6)
        expert_wz[:, 0] = torch.clamp(yaw_error / yaw_full_error, -1.0, 1.0) * float(
            args.base_teacher_yaw_max_action
        )
        expert_wz = torch.nan_to_num(expert_wz, nan=0.0, posinf=0.0, neginf=0.0)
        expert_wz = torch.clamp(expert_wz, -1.0, 1.0)
    return expert_xy, expert_wz, active, expert_distance


def main() -> int:
    args = parse_args()
    if args.base_action_scale < 0.0 or args.base_action_scale > 1.0:
        raise ValueError("--base_action_scale must be in [0, 1]")
    checkpoint = Path(args.checkpoint)
    recovery_checkpoint = Path(args.recovery_checkpoint) if args.recovery_checkpoint else None
    row_blend_checkpoint = Path(args.row_blend_checkpoint) if args.row_blend_checkpoint else None
    row_blend_action_indices = parse_action_indices(args.row_blend_action_indices)
    vec_normalize = Path(args.vec_normalize)
    require(checkpoint.exists(), f"checkpoint not found: {checkpoint}")
    if recovery_checkpoint is not None:
        require(recovery_checkpoint.exists(), f"recovery checkpoint not found: {recovery_checkpoint}")
    if row_blend_checkpoint is not None:
        require(row_blend_checkpoint.exists(), f"row blend checkpoint not found: {row_blend_checkpoint}")
        require(row_blend_action_indices, "--row_blend_action_indices is required with --row_blend_checkpoint")
        require(0.0 <= args.row_blend_weight <= 1.0, "--row_blend_weight must be in [0, 1]")
        invalid = [idx for idx in row_blend_action_indices if idx < 0 or idx >= 9]
        require(not invalid, f"invalid row blend action indices: {invalid}")
    if not args.disable_vec_normalize:
        require(vec_normalize.exists(), f"vec_normalize not found: {vec_normalize}")
    require(args.num_envs > 0, "--num_envs must be positive")
    require(args.steps > 0, "--steps must be positive")
    if args.start_waypoint_min_fraction < 0.0 or args.start_waypoint_min_fraction > 1.0:
        raise ValueError("--start_waypoint_min_fraction must be in [0, 1]")
    if args.start_waypoint_max_fraction < 0.0 or args.start_waypoint_max_fraction > 1.0:
        raise ValueError("--start_waypoint_max_fraction must be in [0, 1]")
    if args.reset_anchor_target_blend < 0.0 or args.reset_anchor_target_blend > 1.0:
        raise ValueError("--reset_anchor_target_blend must be in [0, 1]")
    if args.recovery_route_min_waypoint_fraction < 0.0 or args.recovery_route_min_waypoint_fraction > 1.0:
        raise ValueError("--recovery_route_min_waypoint_fraction must be in [0, 1]")
    if args.recovery_route_min_pos_error < 0.0:
        raise ValueError("--recovery_route_min_pos_error must be non-negative")
    if args.recovery_route_min_base_target_distance < 0.0:
        raise ValueError("--recovery_route_min_base_target_distance must be non-negative")
    if args.obstacles_from_trajectory_metadata and not args.enable_obstacles:
        raise ValueError("--obstacles_from_trajectory_metadata requires --enable_obstacles")

    from isaaclab.app import AppLauncher

    print("[gate] launching Isaac", flush=True)
    app_launcher = AppLauncher(headless=args.headless, enable_cameras=False, device="cuda:0")
    simulation_app = app_launcher.app

    try:
        import torch
        from gymnasium import spaces
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import VecEnvWrapper, VecNormalize

        from task_spec import register_isaac_lab_tasks
        from rl_platform.tasks.mobile_mm import MobileMMTrackEEEnv, MobileMMTrackEEEnvCfg
        from rl_platform.tasks.mobile_mm.config import TrajectoryConfig

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        register_isaac_lab_tasks()

        stage_dir = PROJECT_ROOT / "trajectoryToLearn" / args.trajectory_stage
        manifest = stage_dir / "manifest.txt"
        require(manifest.exists(), f"stage manifest not found: {manifest}")
        reset_config = load_stage_reset_config(args.trajectory_stage)
        print(f"[gate] stage={args.trajectory_stage}", flush=True)
        print(f"[gate] manifest={manifest}", flush=True)
        if reset_config:
            print(
                "[gate] reset offset "
                f"x={reset_config.get('reset_base_x_offset', 0.4415):.4f} "
                f"y={reset_config.get('reset_base_y_offset', 0.2405):.4f}",
                flush=True,
            )
        if args.random_start_waypoint:
            print(
                "[gate] random start waypoint "
                f"{args.start_waypoint_min_fraction:.2f}-{args.start_waypoint_max_fraction:.2f}, "
                f"reset_base_to_trajectory_start={args.reset_base_to_trajectory_start}, "
                f"anchor_blend={args.reset_anchor_target_blend:.2f}",
                flush=True,
            )

        env_cfg = MobileMMTrackEEEnvCfg()
        env_cfg.num_envs = args.num_envs
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = args.seed
        if args.episode_length_s > 0.0:
            env_cfg.episode_length_s = float(args.episode_length_s)
            print(f"[gate] episode_length_s override={env_cfg.episode_length_s:g}", flush=True)
        env_cfg.task_config.obstacles.enable_obstacles = bool(args.enable_obstacles)
        env_cfg.task_config.obstacles.disc_position_xy = (float(args.obstacle_x), float(args.obstacle_y))
        if args.obstacle_radius is not None:
            env_cfg.task_config.obstacles.disc_radius = float(args.obstacle_radius)
        if args.obstacle_height is not None:
            env_cfg.task_config.obstacles.disc_height = float(args.obstacle_height)
        env_cfg.task_config.obstacles.randomize_per_reset = not bool(args.disable_obstacle_randomization)
        env_cfg.task_config.obstacles.disc_position_x_range = tuple(float(x) for x in args.obstacle_x_range)
        env_cfg.task_config.obstacles.disc_position_y_range = tuple(float(y) for y in args.obstacle_y_range)
        env_cfg.task_config.obstacles.min_start_clearance = float(args.min_obstacle_start_clearance)
        if args.enable_obstacles:
            env_cfg.scene = env_cfg._create_scene_config()
            env_cfg.scene.num_envs = args.num_envs
            obstacle_cfg = env_cfg.task_config.obstacles
            print(
                "[gate] obstacles enabled "
                f"pos=({args.obstacle_x:.2f},{args.obstacle_y:.2f}) "
                f"radius={obstacle_cfg.disc_radius:.2f}m "
                f"height={obstacle_cfg.disc_height:.2f}m "
                f"randomized={obstacle_cfg.randomize_per_reset} "
                f"x_range=({args.obstacle_x_range[0]:.2f},{args.obstacle_x_range[1]:.2f}) "
                f"y_range=({args.obstacle_y_range[0]:.2f},{args.obstacle_y_range[1]:.2f})",
                flush=True,
            )
        env_cfg.task_config.base_assist.enable = False
        env_cfg.task_config.arm_action_envelope_profile = args.arm_action_envelope_profile
        env_cfg.task_config.randomize_initial_joint_positions = bool(args.enable_initial_joint_randomization)
        print(f"[gate] arm action envelope={env_cfg.task_config.arm_action_envelope_profile}", flush=True)
        reward_overrides = reset_config.get("reward_overrides", {})
        if isinstance(reward_overrides, dict):
            for name, value in reward_overrides.items():
                setattr(env_cfg.task_config.rewards, name, float(value))
            if reward_overrides:
                print(
                    "[gate] reward overrides "
                    + ", ".join(f"{name}={float(value):g}" for name, value in reward_overrides.items()),
                    flush=True,
                )
        env_cfg.task_config.trajectory = TrajectoryConfig(
            type="multi_recorded",
            trajectory_dir=str(PROJECT_ROOT),
            trajectory_manifest_file=str(manifest),
            max_trajectories=args.max_trajectories,
            min_duration_seconds=args.min_trajectory_duration,
            randomize_start_waypoint=bool(args.random_start_waypoint),
            start_waypoint_min_fraction=args.start_waypoint_min_fraction,
            start_waypoint_max_fraction=args.start_waypoint_max_fraction,
            reset_base_to_trajectory_start=bool(args.reset_base_to_trajectory_start),
            reset_anchor_target_blend=args.reset_anchor_target_blend,
            reset_base_x_offset=reset_config.get("reset_base_x_offset", 0.4415),
            reset_base_y_offset=reset_config.get("reset_base_y_offset", 0.2405),
            reset_arm_to_trajectory_metadata=reset_config.get("reset_arm_to_trajectory_metadata", False),
        )

        print("[gate] creating env", flush=True)
        base_env = MobileMMTrackEEEnv(cfg=env_cfg)
        print("[gate] env created", flush=True)
        raw_env = base_env.unwrapped if hasattr(base_env, "unwrapped") else base_env

        def install_metadata_obstacle_randomizer() -> None:
            require(
                getattr(raw_env, "obstacles_enabled", False),
                "--obstacles_from_trajectory_metadata requires an obstacle-enabled env",
            )
            original_randomize_obstacles = raw_env._randomize_obstacles

            def metadata_randomize_obstacles(env_ids, base_xy_local):
                original_randomize_obstacles(env_ids, base_xy_local)
                metadata = getattr(raw_env.trajectory_manager, "current_trajectory_metadata", []) or []
                rows = []
                used = 0
                for local_idx, env_id in enumerate(env_ids.detach().cpu().tolist()):
                    row = raw_env.obstacle_disc_xy_local[env_id].clone()
                    if env_id < len(metadata) and isinstance(metadata[env_id], dict):
                        raw_meta = metadata[env_id].get("metadata", {})
                        obstacle_meta = raw_meta.get("obstacle") if isinstance(raw_meta, dict) else None
                        center_xy = obstacle_meta.get("center_xy") if isinstance(obstacle_meta, dict) else None
                        if isinstance(center_xy, (list, tuple)) and len(center_xy) == 2:
                            candidate = torch.tensor(center_xy, dtype=row.dtype, device=row.device)
                            if torch.isfinite(candidate).all():
                                row = candidate
                                used += 1
                    rows.append(row)
                if rows:
                    raw_env.obstacle_disc_xy_local[env_ids] = torch.stack(rows, dim=0)
                    raw_env._obstacle_xy_buf = raw_env.obstacle_disc_xy_local.clone()
                    raw_env._write_obstacle_poses_to_sim(env_ids)
                if not hasattr(raw_env, "_metadata_obstacle_randomizer_logged"):
                    print(
                        "[gate] metadata obstacle placement active: "
                        f"applied {used}/{int(env_ids.numel())} env(s) on first reset",
                        flush=True,
                    )
                    raw_env._metadata_obstacle_randomizer_logged = True

            raw_env._randomize_obstacles = metadata_randomize_obstacles
            print("[gate] installed trajectory-metadata obstacle placement", flush=True)

        if args.obstacles_from_trajectory_metadata:
            install_metadata_obstacle_randomizer()

        def install_sequential_trajectory_sampler() -> None:
            manager = raw_env.trajectory_manager
            loader = getattr(manager, "multi_loader", None)
            require(loader is not None, "--assign_loaded_trajectories_once requires multi-recorded trajectories")
            trajectories = list(getattr(loader, "trajectories", []))
            require(trajectories, "trajectory loader has no loaded trajectories")
            require(
                args.num_envs <= len(trajectories),
                f"--num_envs {args.num_envs} exceeds loaded trajectories {len(trajectories)}",
            )

            selected = trajectories[: args.num_envs]
            max_length = int(getattr(loader, "max_length", 0) or max(t["length"] for t in selected))

            def sequential_sample_trajectories_with_lengths(num_envs: int):
                require(
                    num_envs <= len(selected),
                    f"sequential gate requested {num_envs} envs but only {len(selected)} selected",
                )
                sampled = selected[:num_envs]
                loader.last_sampled_metadata = [
                    {
                        "file": traj.get("file", "unknown"),
                        "category": traj.get("category", "unknown"),
                        "length": traj.get("length", 0),
                        "metadata": traj.get("metadata", {}),
                    }
                    for traj in sampled
                ]
                positions_list = []
                orientations_list = []
                lengths_list = []
                for traj in sampled:
                    pos = traj["positions"]
                    ori = traj["orientations"]
                    length = int(traj["length"])
                    lengths_list.append(length)
                    if length < max_length:
                        pad_length = max_length - length
                        pos = torch.cat([pos, pos[-1:].repeat(pad_length, 1)], dim=0)
                        ori = torch.cat([ori, ori[-1:].repeat(pad_length, 1)], dim=0)
                    positions_list.append(pos)
                    orientations_list.append(ori)
                return (
                    torch.stack(positions_list, dim=0),
                    torch.stack(orientations_list, dim=0),
                    torch.tensor(lengths_list, dtype=torch.long, device=loader.device),
                )

            loader.sample_trajectories_with_lengths = sequential_sample_trajectories_with_lengths
            print(
                "[gate] sequential trajectory assignment enabled: "
                f"{len(selected)} envs/files, first={selected[0].get('file')}, "
                f"last={selected[-1].get('file')}",
                flush=True,
            )

        if args.assign_loaded_trajectories_once:
            install_sequential_trajectory_sampler()

        class IsaacLabToSB3VecEnvWrapper(VecEnvWrapper):
            def __init__(
                self,
                venv,
                expected_obs_dim: int | None,
                freeze_base_actions: bool,
                base_action_scale: float,
            ) -> None:
                super().__init__(venv)
                self.expected_obs_dim = expected_obs_dim
                self.freeze_base_actions = bool(freeze_base_actions)
                self.base_action_scale = float(base_action_scale)
                self._base_adapter_logged = False
                self._obs_space_updated = False
                if hasattr(venv.action_space, "shape") and len(venv.action_space.shape) > 1:
                    action_dim = venv.action_space.shape[-1]
                    self.action_space = spaces.Box(
                        low=venv.action_space.low.flatten()[0],
                        high=venv.action_space.high.flatten()[0],
                        shape=(action_dim,),
                        dtype=venv.action_space.dtype,
                    )

            def _adapt_obs_dim(self, obs: np.ndarray) -> np.ndarray:
                if self.expected_obs_dim is None or obs.shape[-1] == self.expected_obs_dim:
                    return obs
                if obs.ndim == 2 and obs.shape[1] == self.expected_obs_dim - 1:
                    return np.concatenate([obs, np.zeros((obs.shape[0], 1), dtype=np.float32)], axis=1)
                if obs.ndim == 2 and obs.shape[1] == self.expected_obs_dim + 1:
                    return obs[:, : self.expected_obs_dim]
                return obs

            def _obs_to_numpy(self, obs):
                if isinstance(obs, tuple):
                    obs = obs[0]
                if isinstance(obs, dict):
                    obs = obs.get("policy", list(obs.values())[0])
                obs = tensor_np(obs).astype(np.float32, copy=False)
                obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
                obs = self._adapt_obs_dim(obs)
                if not self._obs_space_updated:
                    obs_shape = obs.shape[1:] if obs.ndim > 1 else obs.shape
                    self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs_shape, dtype=np.float32)
                    self._obs_space_updated = True
                return obs

            def reset(self):
                return self._obs_to_numpy(self.venv.reset())

            def step_async(self, actions):
                if isinstance(actions, np.ndarray):
                    actions = torch.from_numpy(actions).float().to(self.venv.unwrapped.device)
                if actions.shape[-1] < 9:
                    raise ValueError(f"base action adapter requires at least 9 action dims, got {actions.shape[-1]}")
                if self.freeze_base_actions or abs(self.base_action_scale - 1.0) >= 1e-9:
                    actions = actions.clone()
                    if not self._base_adapter_logged:
                        if self.freeze_base_actions:
                            print("[gate action-adapter] Freezing base action rows [6,7,8]", flush=True)
                        else:
                            print(
                                "[gate action-adapter] Scaling base action rows [6,7,8] "
                                f"by {self.base_action_scale:.3f}",
                                flush=True,
                            )
                        self._base_adapter_logged = True
                    if self.freeze_base_actions:
                        actions[..., 6:9] = 0.0
                    else:
                        actions[..., 6:9] *= self.base_action_scale
                self._actions = actions

            def step_wait(self):
                result = self.venv.step(self._actions)
                if len(result) == 5:
                    obs, rewards, terminated, truncated, infos = result
                    dones = terminated | truncated
                else:
                    obs, rewards, dones, infos = result
                obs = self._obs_to_numpy(obs)
                rewards = tensor_np(rewards).astype(np.float32, copy=False)
                dones = tensor_np(dones).astype(bool, copy=False)
                if isinstance(infos, dict):
                    infos = [infos.copy() for _ in range(len(rewards))]
                elif not isinstance(infos, list):
                    infos = [{} for _ in range(len(rewards))]
                return obs, rewards, dones, infos

        expected_obs_dim = int(np.prod(PPO.load(str(checkpoint), device="cpu").observation_space.shape))
        env = IsaacLabToSB3VecEnvWrapper(
            base_env,
            expected_obs_dim,
            args.freeze_base_actions,
            args.base_action_scale,
        )
        _ = env.reset()
        if not args.disable_vec_normalize:
            env = VecNormalize.load(str(vec_normalize), env)
            env.training = False
            env.norm_reward = False
        obs = env.reset()
        model = PPO.load(str(checkpoint), env=env, device="cuda:0")
        row_blend_model = None
        if row_blend_checkpoint is not None:
            row_blend_model = PPO.load(str(row_blend_checkpoint), env=env, device="cuda:0")
            print(
                "[gate row-blend] enabled "
                f"checkpoint={row_blend_checkpoint} rows={row_blend_action_indices} "
                f"weight={args.row_blend_weight:.3f}",
                flush=True,
            )
        recovery_model = None
        if recovery_checkpoint is not None:
            recovery_model = PPO.load(str(recovery_checkpoint), env=env, device="cuda:0")
            print(
                "[gate router] conditional recovery enabled "
                f"checkpoint={recovery_checkpoint} "
                f"min_waypoint_fraction={args.recovery_route_min_waypoint_fraction:.3f} "
                f"min_pos_error={args.recovery_route_min_pos_error:.4f} "
                f"min_base_target_distance={args.recovery_route_min_base_target_distance:.4f}",
                flush=True,
            )
        open_loop_actions = None
        if args.open_loop_actions_npz:
            with np.load(args.open_loop_actions_npz, allow_pickle=False) as data:
                actions = data["actions"].astype(np.float32)
            start = args.action_sequence_index * args.steps
            end = start + args.steps
            require(end <= actions.shape[0], f"action sequence slice {start}:{end} exceeds {actions.shape[0]}")
            open_loop_actions = actions[start:end]
            print(
                f"[gate] using open-loop actions {args.open_loop_actions_npz} "
                f"rows={start}:{end}",
                flush=True,
            )
        def route_recovery_mask(target_pos, ee_pos, base_pos):
            trajectory_manager = raw_env.trajectory_manager
            if not hasattr(trajectory_manager, "current_waypoint_idx"):
                return np.zeros((args.num_envs,), dtype=bool), np.zeros((args.num_envs,), dtype=np.float32), np.zeros((args.num_envs,), dtype=np.float32), np.zeros((args.num_envs,), dtype=np.float32)

            waypoint_idx = tensor_np(trajectory_manager.current_waypoint_idx).astype(np.float32)
            lengths = getattr(trajectory_manager, "recorded_lengths", None)
            if lengths is None:
                waypoint_fraction = np.zeros_like(waypoint_idx, dtype=np.float32)
            else:
                lengths_np = np.maximum(tensor_np(lengths).astype(np.float32) - 1.0, 1.0)
                waypoint_fraction = np.clip(waypoint_idx / lengths_np, 0.0, 1.0)

            pos_error = tensor_np(torch.linalg.norm(target_pos - ee_pos, dim=-1)).astype(np.float32)
            base_target_distance = tensor_np(torch.linalg.norm(target_pos[:, :2] - base_pos[:, :2], dim=-1)).astype(np.float32)
            route = waypoint_fraction >= float(args.recovery_route_min_waypoint_fraction)
            if args.recovery_route_min_pos_error > 0.0:
                route &= pos_error >= float(args.recovery_route_min_pos_error)
            if args.recovery_route_min_base_target_distance > 0.0:
                route &= base_target_distance >= float(args.recovery_route_min_base_target_distance)
            return route, waypoint_fraction, pos_error, base_target_distance

        pos_errors: list[np.ndarray] = []
        ori_errors: list[np.ndarray] = []
        rewards: list[np.ndarray] = []
        obstacle_clearances: list[np.ndarray] = []
        route_counts: list[int] = []
        latched_route_mask = np.zeros((args.num_envs,), dtype=bool)
        route_waypoint_fractions: list[np.ndarray] = []
        route_pos_errors: list[np.ndarray] = []
        route_base_target_distances: list[np.ndarray] = []
        dataset_observations: list[np.ndarray] = []
        dataset_actions: list[np.ndarray] = []
        dataset_policy_actions: list[np.ndarray] = []
        base_teacher_observations: list[np.ndarray] = []
        base_teacher_actions: list[np.ndarray] = []
        base_teacher_masks: list[np.ndarray] = []
        base_teacher_weights: list[np.ndarray] = []
        base_teacher_distances: list[np.ndarray] = []
        dones_count = 0
        done_counts_per_env = np.zeros((args.num_envs,), dtype=np.int64)
        first_done_step_per_env = np.full((args.num_envs,), -1, dtype=np.int64)
        target_pos0, target_quat0 = raw_env.trajectory_manager.get_target_pose()
        ee_pos0 = raw_env.robot.data.body_pos_w[:, raw_env._ee_body_idx, :]
        base_pos0 = raw_env.robot.data.root_pos_w
        initial_pos_error = tensor_np(torch.linalg.norm(target_pos0 - ee_pos0, dim=-1))
        initial_obstacle_clearance = None
        initial_obstacle_xy = None
        if getattr(raw_env, "obstacles_enabled", False):
            initial_obstacle_clearance = tensor_np(raw_env._get_obstacle_clearance(base_pos0))
            initial_obstacle_xy = tensor_np(raw_env.obstacle_disc_xy_local).astype(np.float64)
            obstacle_clearances.append(initial_obstacle_clearance.reshape(-1))
        metadata0 = list(getattr(raw_env.trajectory_manager, "current_trajectory_metadata", []) or [])
        lengths0 = tensor_np(getattr(raw_env.trajectory_manager, "recorded_lengths", None))
        start_waypoint0 = tensor_np(getattr(raw_env.trajectory_manager, "current_waypoint_idx", None))
        print(
            "[gate] initial env0 "
            f"base={tensor_np(base_pos0[0]).round(4).tolist()} "
            f"ee={tensor_np(ee_pos0[0]).round(4).tolist()} "
            f"target={tensor_np(target_pos0[0]).round(4).tolist()} "
            f"pos_err={float(initial_pos_error[0]):.4f}",
            flush=True,
        )
        print("[gate] rollout start", flush=True)
        for step in range(args.steps):
            obs_before_step = np.asarray(obs, dtype=np.float32).copy()
            primary_action_for_dataset = None
            if open_loop_actions is None:
                action, _ = model.predict(obs, deterministic=True)
                primary_action_for_dataset = np.asarray(action, dtype=np.float32).copy()
                if row_blend_model is not None:
                    blend_action, _ = row_blend_model.predict(obs, deterministic=True)
                    action = np.asarray(action, dtype=np.float32).copy()
                    blend_action = np.asarray(blend_action, dtype=np.float32)
                    rows = np.asarray(row_blend_action_indices, dtype=np.int64)
                    action[:, rows] = (
                        (1.0 - float(args.row_blend_weight)) * action[:, rows]
                        + float(args.row_blend_weight) * blend_action[:, rows]
                    )
                if recovery_model is not None:
                    target_pos_pre, _ = raw_env.trajectory_manager.get_target_pose()
                    ee_pos_pre = raw_env.robot.data.body_pos_w[:, raw_env._ee_body_idx, :]
                    base_pos_pre = raw_env.robot.data.root_pos_w
                    route_mask, waypoint_fraction, pre_pos_error, base_target_distance = route_recovery_mask(
                        target_pos_pre,
                        ee_pos_pre,
                        base_pos_pre,
                    )
                    if args.recovery_route_latch_once:
                        latched_route_mask |= route_mask
                        route_mask = latched_route_mask.copy()
                    recovery_action, _ = recovery_model.predict(obs, deterministic=True)
                    if np.any(route_mask):
                        action = np.asarray(action, dtype=np.float32).copy()
                        action[route_mask] = recovery_action[route_mask]
                    route_counts.append(int(np.count_nonzero(route_mask)))
                    route_waypoint_fractions.append(waypoint_fraction)
                    route_pos_errors.append(pre_pos_error)
                    route_base_target_distances.append(base_target_distance)
            else:
                action = np.repeat(open_loop_actions[step : step + 1], args.num_envs, axis=0)
                primary_action_for_dataset = np.asarray(action, dtype=np.float32).copy()
            if args.output_dataset_npz:
                dataset_observations.append(obs_before_step)
                dataset_actions.append(np.asarray(action, dtype=np.float32).copy())
                dataset_policy_actions.append(np.asarray(primary_action_for_dataset, dtype=np.float32).copy())
            if args.output_base_teacher_npz:
                expert_xy, expert_wz, active, expert_distance = compute_direct_base_teacher(raw_env, args)
                expert_xy_np = tensor_np(expert_xy).astype(np.float32)
                expert_wz_np = tensor_np(expert_wz).astype(np.float32)
                active_np = tensor_np(active).astype(bool)
                distance_np = tensor_np(expert_distance).astype(np.float32)
                labels = np.zeros_like(np.asarray(action, dtype=np.float32))
                mask = np.zeros_like(labels, dtype=np.float32)
                labels[:, 6:8] = expert_xy_np[:, 0:2]
                mask[:, 6:8] = active_np[:, None].astype(np.float32)
                if args.base_teacher_include_yaw:
                    labels[:, 8] = expert_wz_np[:, 0]
                    mask[:, 8] = active_np.astype(np.float32)
                denom = max(
                    float(args.base_teacher_sample_weight_full_distance)
                    - float(args.base_teacher_sample_weight_distance_threshold),
                    1e-6,
                )
                excess = np.clip(
                    (distance_np - float(args.base_teacher_sample_weight_distance_threshold)) / denom,
                    0.0,
                    1.0,
                )
                weights = 1.0 + (float(args.base_teacher_sample_weight_max) - 1.0) * excess
                if args.base_teacher_active_only:
                    keep = active_np
                    if np.any(keep):
                        base_teacher_observations.append(obs_before_step[keep])
                        base_teacher_actions.append(labels[keep])
                        base_teacher_masks.append(mask[keep])
                        base_teacher_weights.append(weights[keep].astype(np.float32))
                        base_teacher_distances.append(distance_np[keep])
                else:
                    base_teacher_observations.append(obs_before_step)
                    base_teacher_actions.append(labels)
                    base_teacher_masks.append(mask)
                    base_teacher_weights.append(weights.astype(np.float32))
                    base_teacher_distances.append(distance_np)
            obs, reward, done, _ = env.step(action)
            rewards.append(np.asarray(reward, dtype=np.float64))
            done_arr = np.asarray(done, dtype=bool)
            dones_count += int(np.count_nonzero(done_arr))
            done_counts_per_env += done_arr.astype(np.int64)
            first_done_mask = done_arr & (first_done_step_per_env < 0)
            first_done_step_per_env[first_done_mask] = step

            target_pos, target_quat = raw_env.trajectory_manager.get_target_pose()
            ee_pos = raw_env.robot.data.body_pos_w[:, raw_env._ee_body_idx, :]
            ee_quat = raw_env.robot.data.body_quat_w[:, raw_env._ee_body_idx, :]
            pos_errors.append(tensor_np(torch.linalg.norm(target_pos - ee_pos, dim=-1)))
            quat_dot = torch.abs(torch.sum(target_quat * ee_quat, dim=-1)).clamp(max=1.0)
            ori_errors.append(tensor_np(2.0 * torch.acos(quat_dot)))
            if getattr(raw_env, "obstacles_enabled", False) and hasattr(raw_env, "_obstacle_clearance_buf"):
                obstacle_clearances.append(tensor_np(raw_env._obstacle_clearance_buf).reshape(-1))
            if step % 50 == 0:
                if recovery_model is not None and route_counts:
                    print(
                        f"[gate router] step={step} routed={route_counts[-1]}/{args.num_envs}",
                        flush=True,
                    )
                print(f"[gate] step={step}/{args.steps}", flush=True)

        pos = np.concatenate([x.reshape(-1) for x in pos_errors]).astype(np.float64)
        ori = np.concatenate([x.reshape(-1) for x in ori_errors]).astype(np.float64)
        rew = np.concatenate([x.reshape(-1) for x in rewards]).astype(np.float64)
        pos_by_step = np.stack(pos_errors, axis=0).astype(np.float64)
        ori_by_step = np.stack(ori_errors, axis=0).astype(np.float64)
        rew_by_step = np.stack(rewards, axis=0).astype(np.float64)
        first_episode_pos_chunks: list[np.ndarray] = []
        first_episode_ori_chunks: list[np.ndarray] = []
        first_episode_rew_chunks: list[np.ndarray] = []
        first_episode_final_pos_error = np.zeros((args.num_envs,), dtype=np.float64)
        first_episode_steps = np.zeros((args.num_envs,), dtype=np.int64)
        for env_id in range(args.num_envs):
            stop = int(first_done_step_per_env[env_id])
            if stop < 0:
                stop = int(args.steps)
            # Exclude the terminal sample because the vectorized env may already
            # have reset that environment for the next episode on the done step.
            stop = max(0, min(stop, int(args.steps)))
            first_episode_steps[env_id] = stop
            if stop > 0:
                first_episode_pos_chunks.append(pos_by_step[:stop, env_id])
                first_episode_ori_chunks.append(ori_by_step[:stop, env_id])
                first_episode_rew_chunks.append(rew_by_step[:stop, env_id])
                first_episode_final_pos_error[env_id] = float(pos_by_step[stop - 1, env_id])
            else:
                first_episode_final_pos_error[env_id] = float(initial_pos_error[env_id])
        first_episode_pos = (
            np.concatenate(first_episode_pos_chunks).astype(np.float64)
            if first_episode_pos_chunks
            else np.asarray([], dtype=np.float64)
        )
        first_episode_ori = (
            np.concatenate(first_episode_ori_chunks).astype(np.float64)
            if first_episode_ori_chunks
            else np.asarray([], dtype=np.float64)
        )
        first_episode_rew = (
            np.concatenate(first_episode_rew_chunks).astype(np.float64)
            if first_episode_rew_chunks
            else np.asarray([], dtype=np.float64)
        )
        target_pos_end, _ = raw_env.trajectory_manager.get_target_pose()
        ee_pos_end = raw_env.robot.data.body_pos_w[:, raw_env._ee_body_idx, :]
        base_pos_end = raw_env.robot.data.root_pos_w
        final_pos_error = tensor_np(torch.linalg.norm(target_pos_end - ee_pos_end, dim=-1))
        final_obstacle_clearance = None
        final_obstacle_xy = None
        if getattr(raw_env, "obstacles_enabled", False):
            final_obstacle_clearance = tensor_np(raw_env._get_obstacle_clearance(base_pos_end))
            final_obstacle_xy = tensor_np(raw_env.obstacle_disc_xy_local).astype(np.float64)
        arm_joint_ids = raw_env._get_joint_ids(
            ["joint6_arm_yaw", "joint5_arm_pitch", "joint4_elbow_pitch", "joint3_gimbal_yaw", "joint2_gimbal_roll", "joint1_gimbal_pitch"],
            "_gate_arm_joint_ids",
        )
        final_arm_joint_pos = raw_env.robot.data.joint_pos[:, arm_joint_ids]
        final_arm_target = getattr(raw_env, "filtered_arm_targets", final_arm_joint_pos)
        print(
            "[gate] final env0 "
            f"base={tensor_np(base_pos_end[0]).round(4).tolist()} "
            f"ee={tensor_np(ee_pos_end[0]).round(4).tolist()} "
            f"target={tensor_np(target_pos_end[0]).round(4).tolist()} "
            f"pos_err={float(final_pos_error[0]):.4f}",
            flush=True,
        )
        print(
            "[gate] final arm env0 "
            f"joint_pos={tensor_np(final_arm_joint_pos[0]).round(4).tolist()} "
            f"target={tensor_np(final_arm_target[0]).round(4).tolist()}",
            flush=True,
        )
        metrics = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "checkpoint": str(checkpoint),
            "recovery_checkpoint": str(recovery_checkpoint) if recovery_checkpoint is not None else None,
            "row_blend_checkpoint": str(row_blend_checkpoint) if row_blend_checkpoint is not None else None,
            "row_blend_action_indices": row_blend_action_indices,
            "row_blend_weight": float(args.row_blend_weight),
            "recovery_route_min_waypoint_fraction": float(args.recovery_route_min_waypoint_fraction),
            "recovery_route_min_pos_error": float(args.recovery_route_min_pos_error),
            "recovery_route_min_base_target_distance": float(args.recovery_route_min_base_target_distance),
            "recovery_route_latch_once": bool(args.recovery_route_latch_once),
            "vec_normalize": str(vec_normalize),
            "disable_vec_normalize": bool(args.disable_vec_normalize),
            "trajectory_stage": args.trajectory_stage,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "episode_length_s": float(args.episode_length_s) if args.episode_length_s > 0.0 else None,
            "enable_obstacles": bool(args.enable_obstacles),
            "obstacles_from_trajectory_metadata": bool(args.obstacles_from_trajectory_metadata),
            "samples": int(pos.size),
            "freeze_base_actions": bool(args.freeze_base_actions),
            "base_action_scale": float(args.base_action_scale),
            "arm_action_envelope_profile": str(args.arm_action_envelope_profile),
            "initial_joint_randomization": bool(args.enable_initial_joint_randomization),
            "initial_ee_pos_error_mean_m": float(np.mean(initial_pos_error)),
            "final_ee_pos_error_mean_m": float(np.mean(final_pos_error)),
            "open_loop_actions_npz": args.open_loop_actions_npz,
            "action_sequence_index": int(args.action_sequence_index),
            "ee_pos_error_mean_m": float(np.mean(pos)),
            "ee_pos_error_p50_m": float(np.percentile(pos, 50)),
            "ee_pos_error_p95_m": float(np.percentile(pos, 95)),
            "ee_pos_error_max_m": float(np.max(pos)),
            "ee_ori_error_mean_deg": float(np.degrees(np.mean(ori))),
            "ee_ori_error_p95_deg": float(np.degrees(np.percentile(ori, 95))),
            "reward_mean": float(np.mean(rew)),
            "reward_p50": float(np.percentile(rew, 50)),
            "dones_count": int(dones_count),
            "first_episode_samples": int(first_episode_pos.size),
            "first_episode_ee_pos_error_mean_m": float(np.mean(first_episode_pos)) if first_episode_pos.size else None,
            "first_episode_ee_pos_error_p50_m": float(np.percentile(first_episode_pos, 50)) if first_episode_pos.size else None,
            "first_episode_ee_pos_error_p95_m": float(np.percentile(first_episode_pos, 95)) if first_episode_pos.size else None,
            "first_episode_ee_pos_error_max_m": float(np.max(first_episode_pos)) if first_episode_pos.size else None,
            "first_episode_ee_ori_error_mean_deg": float(np.degrees(np.mean(first_episode_ori))) if first_episode_ori.size else None,
            "first_episode_reward_mean": float(np.mean(first_episode_rew)) if first_episode_rew.size else None,
            "first_episode_done_envs": int(np.count_nonzero(first_done_step_per_env >= 0)),
        }
        if obstacle_clearances:
            clearance = np.concatenate([x.reshape(-1) for x in obstacle_clearances]).astype(np.float64)
            safety_radius = float(getattr(raw_env, "reward_weights", {}).get("safety_radius", 0.2))
            metrics.update(
                {
                    "obstacle_safety_radius_m": safety_radius,
                    "initial_obstacle_clearance_mean_m": float(np.mean(initial_obstacle_clearance)) if initial_obstacle_clearance is not None else None,
                    "initial_obstacle_clearance_min_m": float(np.min(initial_obstacle_clearance)) if initial_obstacle_clearance is not None else None,
                    "final_obstacle_clearance_mean_m": float(np.mean(final_obstacle_clearance)) if final_obstacle_clearance is not None else None,
                    "final_obstacle_clearance_min_m": float(np.min(final_obstacle_clearance)) if final_obstacle_clearance is not None else None,
                    "obstacle_clearance_mean_m": float(np.mean(clearance)),
                    "obstacle_clearance_p05_m": float(np.percentile(clearance, 5)),
                    "obstacle_clearance_min_m": float(np.min(clearance)),
                    "obstacle_unsafe_pct": float(np.mean(clearance < safety_radius) * 100.0),
                    "obstacle_collision_pct": float(np.mean(clearance < 0.0) * 100.0),
                }
            )
            if hasattr(raw_env, "obstacle_disc_xy_local"):
                obstacle_xy = tensor_np(raw_env.obstacle_disc_xy_local).astype(np.float64)
                metrics.update(
                    {
                        "obstacle_x_mean": float(np.mean(obstacle_xy[:, 0])),
                        "obstacle_x_std": float(np.std(obstacle_xy[:, 0])),
                        "obstacle_y_mean": float(np.mean(obstacle_xy[:, 1])),
                        "obstacle_y_std": float(np.std(obstacle_xy[:, 1])),
                    }
                )
        if args.assign_loaded_trajectories_once:
            waypoint_end = tensor_np(getattr(raw_env.trajectory_manager, "current_waypoint_idx", None))
            per_env = []
            for env_id in range(args.num_envs):
                metadata = metadata0[env_id] if env_id < len(metadata0) and isinstance(metadata0[env_id], dict) else {}
                per_env.append(
                    {
                        "env_id": int(env_id),
                        "trajectory_file": str(metadata.get("file", "unknown")),
                        "trajectory_category": str(metadata.get("category", "unknown")),
                        "trajectory_length": int(lengths0[env_id]) if lengths0 is not None and env_id < len(lengths0) else None,
                        "start_waypoint_idx": int(start_waypoint0[env_id]) if start_waypoint0 is not None and env_id < len(start_waypoint0) else None,
                        "end_waypoint_idx": int(waypoint_end[env_id]) if waypoint_end is not None and env_id < len(waypoint_end) else None,
                        "initial_ee_pos_error_m": float(initial_pos_error[env_id]),
                        "final_ee_pos_error_m": float(final_pos_error[env_id]),
                        "ee_pos_error_mean_m": float(np.mean(pos_by_step[:, env_id])),
                        "ee_pos_error_p50_m": float(np.percentile(pos_by_step[:, env_id], 50)),
                        "ee_pos_error_p95_m": float(np.percentile(pos_by_step[:, env_id], 95)),
                        "ee_pos_error_max_m": float(np.max(pos_by_step[:, env_id])),
                        "ee_ori_error_mean_deg": float(np.degrees(np.mean(ori_by_step[:, env_id]))),
                        "reward_mean": float(np.mean(rew_by_step[:, env_id])),
                        "dones_count": int(done_counts_per_env[env_id]),
                        "first_done_step": int(first_done_step_per_env[env_id]),
                        "first_episode_steps": int(first_episode_steps[env_id]),
                        "first_episode_final_ee_pos_error_m": float(first_episode_final_pos_error[env_id]),
                        "first_episode_ee_pos_error_mean_m": (
                            float(np.mean(pos_by_step[: first_episode_steps[env_id], env_id]))
                            if first_episode_steps[env_id] > 0
                            else None
                        ),
                        "first_episode_ee_pos_error_p50_m": (
                            float(np.percentile(pos_by_step[: first_episode_steps[env_id], env_id], 50))
                            if first_episode_steps[env_id] > 0
                            else None
                        ),
                        "first_episode_ee_pos_error_p95_m": (
                            float(np.percentile(pos_by_step[: first_episode_steps[env_id], env_id], 95))
                            if first_episode_steps[env_id] > 0
                            else None
                        ),
                        "first_episode_ee_pos_error_max_m": (
                            float(np.max(pos_by_step[: first_episode_steps[env_id], env_id]))
                            if first_episode_steps[env_id] > 0
                            else None
                        ),
                        "first_episode_ee_ori_error_mean_deg": (
                            float(np.degrees(np.mean(ori_by_step[: first_episode_steps[env_id], env_id])))
                            if first_episode_steps[env_id] > 0
                            else None
                        ),
                        "first_episode_reward_mean": (
                            float(np.mean(rew_by_step[: first_episode_steps[env_id], env_id]))
                            if first_episode_steps[env_id] > 0
                            else None
                        ),
                    }
                )
                if initial_obstacle_clearance is not None and env_id < len(initial_obstacle_clearance):
                    per_env[-1]["initial_obstacle_clearance_m"] = float(initial_obstacle_clearance[env_id])
                if final_obstacle_clearance is not None and env_id < len(final_obstacle_clearance):
                    per_env[-1]["final_obstacle_clearance_m"] = float(final_obstacle_clearance[env_id])
                if initial_obstacle_xy is not None and env_id < len(initial_obstacle_xy):
                    per_env[-1]["initial_obstacle_xy"] = [
                        float(initial_obstacle_xy[env_id, 0]),
                        float(initial_obstacle_xy[env_id, 1]),
                    ]
                if final_obstacle_xy is not None and env_id < len(final_obstacle_xy):
                    per_env[-1]["final_obstacle_xy"] = [
                        float(final_obstacle_xy[env_id, 0]),
                        float(final_obstacle_xy[env_id, 1]),
                    ]
            metrics["per_env"] = per_env
        if route_counts:
            route_count_arr = np.asarray(route_counts, dtype=np.float64)
            route_fraction_arr = np.concatenate([x.reshape(-1) for x in route_waypoint_fractions]).astype(np.float64)
            route_pos_arr = np.concatenate([x.reshape(-1) for x in route_pos_errors]).astype(np.float64)
            route_base_arr = np.concatenate([x.reshape(-1) for x in route_base_target_distances]).astype(np.float64)
            metrics.update(
                {
                    "recovery_route_steps": int(len(route_counts)),
                    "recovery_route_env_selections": int(np.sum(route_count_arr)),
                    "recovery_route_fraction": float(np.sum(route_count_arr) / (len(route_counts) * args.num_envs)),
                    "recovery_route_count_mean": float(np.mean(route_count_arr)),
                    "recovery_route_waypoint_fraction_mean": float(np.mean(route_fraction_arr)),
                    "recovery_route_waypoint_fraction_p95": float(np.percentile(route_fraction_arr, 95)),
                    "recovery_route_pos_error_mean_m": float(np.mean(route_pos_arr)),
                    "recovery_route_base_target_distance_mean_m": float(np.mean(route_base_arr)),
                }
            )
        print(json.dumps(metrics, indent=2), flush=True)
        if args.output_json:
            output = Path(args.output_json)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
            print(f"[gate] wrote {output}", flush=True)
        if args.output_dataset_npz:
            dataset_obs = np.concatenate(dataset_observations, axis=0).astype(np.float32)
            dataset_actions_arr = np.concatenate(dataset_actions, axis=0).astype(np.float32)
            dataset_policy_actions_arr = np.concatenate(dataset_policy_actions, axis=0).astype(np.float32)
            dataset_mask = np.ones_like(dataset_actions_arr, dtype=np.float32)
            if row_blend_action_indices:
                dataset_mask[:] = 0.0
                dataset_mask[:, np.asarray(row_blend_action_indices, dtype=np.int64)] = 1.0
            dataset_metadata = {
                "created_at": metrics["created_at"],
                "checkpoint": str(checkpoint),
                "row_blend_checkpoint": str(row_blend_checkpoint) if row_blend_checkpoint is not None else None,
                "row_blend_action_indices": row_blend_action_indices,
                "row_blend_weight": float(args.row_blend_weight),
                "trajectory_stage": args.trajectory_stage,
                "num_envs": int(args.num_envs),
                "steps": int(args.steps),
                "samples": int(dataset_obs.shape[0]),
                "enable_obstacles": bool(args.enable_obstacles),
                "obstacles_from_trajectory_metadata": bool(args.obstacles_from_trajectory_metadata),
            }
            output_dataset = Path(args.output_dataset_npz)
            output_dataset.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output_dataset,
                observations=dataset_obs,
                actions=dataset_actions_arr,
                action_valid_mask=dataset_mask,
                policy_actions=dataset_policy_actions_arr,
                metadata=json.dumps(dataset_metadata, sort_keys=True),
            )
            print(f"[gate] wrote dataset {output_dataset} rows={dataset_obs.shape[0]}", flush=True)
        if args.output_base_teacher_npz:
            require(base_teacher_observations, "base teacher produced no active rows")
            teacher_obs = np.concatenate(base_teacher_observations, axis=0).astype(np.float32)
            teacher_actions = np.concatenate(base_teacher_actions, axis=0).astype(np.float32)
            teacher_mask = np.concatenate(base_teacher_masks, axis=0).astype(np.float32)
            teacher_weights = np.concatenate(base_teacher_weights, axis=0).astype(np.float32)
            teacher_distances = np.concatenate(base_teacher_distances, axis=0).astype(np.float32)
            require(float(teacher_mask.sum()) > 0.0, "base teacher produced no valid labels")
            teacher_metadata = {
                "created_at": metrics["created_at"],
                "schema": "direct_base_teacher_dataset_v1",
                "checkpoint": str(checkpoint),
                "trajectory_stage": args.trajectory_stage,
                "num_envs": int(args.num_envs),
                "steps": int(args.steps),
                "samples": int(teacher_obs.shape[0]),
                "mode": args.base_teacher_mode,
                "activation_distance": float(args.base_teacher_activation_distance),
                "full_speed_distance": float(args.base_teacher_full_speed_distance),
                "max_action": float(args.base_teacher_max_action),
                "lookahead_steps": int(args.base_teacher_lookahead_steps),
                "include_yaw": bool(args.base_teacher_include_yaw),
                "active_only": bool(args.base_teacher_active_only),
                "distance_mean_m": float(np.mean(teacher_distances)),
                "distance_p95_m": float(np.percentile(teacher_distances, 95)),
                "valid_action_counts": teacher_mask.sum(axis=0).astype(float).tolist(),
            }
            output_teacher = Path(args.output_base_teacher_npz)
            output_teacher.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output_teacher,
                observations=teacher_obs,
                actions=teacher_actions,
                action_valid_mask=teacher_mask,
                sample_weight=teacher_weights,
                source_base_target_distance_m=teacher_distances,
                metadata=json.dumps(teacher_metadata, indent=2),
            )
            print(
                f"[gate] wrote base teacher dataset {output_teacher} "
                f"rows={teacher_obs.shape[0]} valid={teacher_metadata['valid_action_counts']}",
                flush=True,
            )
        env.close()
    finally:
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
