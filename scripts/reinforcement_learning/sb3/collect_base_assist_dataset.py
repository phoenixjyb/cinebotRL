"""Collect normalized observation labels from the stage1 base-assist teacher.

The output follows the existing BC dataset schema:
``observations`` [N, obs_dim], ``actions`` [N, act_dim], and
``action_valid_mask`` [N, act_dim].  Only base rows are valid by default.
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
    parser = argparse.ArgumentParser(description="Collect base-assist distillation labels.")
    parser.add_argument("--checkpoint", required=True, help="PPO checkpoint/final_model.zip used to visit states.")
    parser.add_argument("--vec_normalize", required=True, help="VecNormalize pkl paired with the checkpoint.")
    parser.add_argument("--output", required=True, help="Output .npz path.")
    parser.add_argument("--task", default="RecomoProto2TrackEE-v0")
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--num_steps", type=int, default=512)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trajectory_stage", default="stage1_recovery")
    parser.add_argument(
        "--trajectory_categories",
        default="",
        help="Optional comma-separated category filter applied to the trajectory stage manifest.",
    )
    parser.add_argument(
        "--trajectory_files",
        default="",
        help="Optional comma-separated JSON file-name filter applied to the trajectory stage manifest.",
    )
    parser.add_argument("--start_waypoint_min_fraction", type=float, default=0.25)
    parser.add_argument("--start_waypoint_max_fraction", type=float, default=0.70)
    parser.add_argument("--reset_anchor_target_blend", type=float, default=0.35)
    parser.add_argument("--enable_obstacles", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--base_assist_blend", type=float, default=0.70)
    parser.add_argument("--base_assist_activation_distance", type=float, default=0.45)
    parser.add_argument("--base_assist_full_speed_distance", type=float, default=0.90)
    parser.add_argument("--base_assist_max_action", type=float, default=1.0)
    parser.add_argument("--base_action_indices", default="6,7", help="Comma-separated base rows to label.")
    parser.add_argument("--include_yaw", action="store_true", help="Also label base_wz row 8 from yaw assist.")
    parser.add_argument("--base_assist_yaw_max_action", type=float, default=0.6)
    parser.add_argument("--base_assist_yaw_full_error", type=float, default=1.2)
    parser.add_argument("--active_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--sample_weight_mode",
        choices=("uniform", "base_distance"),
        default="uniform",
        help="Optional per-row weighting saved as sample_weight for downstream weighted aux training.",
    )
    parser.add_argument(
        "--sample_weight_distance_threshold",
        type=float,
        default=0.70,
        help="Base-target distance where base_distance weights start increasing.",
    )
    parser.add_argument(
        "--sample_weight_full_distance",
        type=float,
        default=1.20,
        help="Base-target distance where base_distance weights reach their maximum.",
    )
    parser.add_argument(
        "--sample_weight_max",
        type=float,
        default=4.0,
        help="Maximum per-row weight for base_distance weighting.",
    )
    return parser.parse_args()


def tensor_np(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def unwrap_isaac_env(env):
    raw = env
    while hasattr(raw, "venv"):
        raw = raw.venv
    if hasattr(raw, "unwrapped"):
        raw = raw.unwrapped
    return raw


def parse_indices(raw: str) -> list[int]:
    indices = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not indices:
        raise ValueError("at least one action index is required")
    return indices


def parse_csv_set(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def build_filtered_manifest(
    source_manifest: Path,
    output_dir: Path,
    categories: set[str],
    files: set[str],
) -> Path:
    if not categories and not files:
        return source_manifest
    if not source_manifest.exists():
        raise FileNotFoundError(f"trajectory manifest not found: {source_manifest}")

    selected: list[str] = []
    total = 0
    for raw_line in source_manifest.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        total += 1
        rel_path = Path(line)
        category_ok = not categories or rel_path.parent.name in categories
        file_ok = not files or rel_path.name in files
        if category_ok and file_ok:
            selected.append(line)

    if not selected:
        raise ValueError(
            "trajectory filter selected zero files: "
            f"categories={sorted(categories)}, files={sorted(files)}, source={source_manifest}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix_parts = []
    if categories:
        suffix_parts.append("cat-" + "-".join(sorted(categories)))
    if files:
        suffix_parts.append("files-" + str(len(files)))
    suffix = "_".join(suffix_parts)
    filtered_manifest = output_dir / f"{source_manifest.stem}_{suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    header = [
        "# Generated by collect_base_assist_dataset.py trajectory filters.",
        f"# Source manifest: {source_manifest}",
        f"# Categories: {', '.join(sorted(categories)) if categories else '*'}",
        f"# Files: {', '.join(sorted(files)) if files else '*'}",
        f"# Selected: {len(selected)} / {total}",
    ]
    filtered_manifest.write_text("\n".join(header + selected) + "\n", encoding="utf-8")
    print(
        "trajectory filter: "
        f"selected {len(selected)}/{total} files -> {filtered_manifest}"
    )
    return filtered_manifest


def main() -> int:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    vec_path = Path(args.vec_normalize)
    if not checkpoint.exists():
        print(f"checkpoint not found: {checkpoint}")
        return 1
    if not vec_path.exists():
        print(f"VecNormalize stats not found: {vec_path}")
        return 1

    action_indices = parse_indices(args.base_action_indices)
    trajectory_categories = parse_csv_set(args.trajectory_categories)
    trajectory_files = parse_csv_set(args.trajectory_files)
    if args.include_yaw and 8 not in action_indices:
        action_indices.append(8)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app

    try:
        import torch
        from gymnasium import spaces
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import VecEnvWrapper, VecNormalize

        from task_spec import register_isaac_lab_tasks
        from rl_platform.tasks.mobile_mm import MobileMMTrackEEEnv, MobileMMTrackEEEnvCfg
        from rl_platform.tasks.mobile_mm.config import TrajectoryConfig

        register_isaac_lab_tasks()
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        env_cfg = MobileMMTrackEEEnvCfg()
        env_cfg.num_envs = args.num_envs
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.task_config.obstacles.enable_obstacles = bool(args.enable_obstacles)
        env_cfg.task_config.obstacles.randomize_per_reset = True
        env_cfg.task_config.obstacles.disc_radius = 0.20
        env_cfg.task_config.obstacles.disc_height = 0.50
        env_cfg.task_config.obstacles.disc_position_x_range = (-0.35, 0.35)
        env_cfg.task_config.obstacles.disc_position_y_range = (0.45, 1.00)
        env_cfg.task_config.obstacles.min_start_clearance = 0.10
        if args.enable_obstacles:
            env_cfg.scene = env_cfg._create_scene_config()
            env_cfg.scene.num_envs = args.num_envs

        output_path = Path(args.output)
        manifest = PROJECT_ROOT / "trajectoryToLearn" / args.trajectory_stage / "manifest.txt"
        manifest = build_filtered_manifest(
            manifest,
            output_path.parent / "filtered_manifests",
            trajectory_categories,
            trajectory_files,
        )
        env_cfg.task_config.trajectory = TrajectoryConfig(
            type="multi_recorded",
            trajectory_dir=str(PROJECT_ROOT),
            trajectory_manifest_file=str(manifest),
            min_duration_seconds=5.0,
            randomize_start_waypoint=True,
            start_waypoint_min_fraction=args.start_waypoint_min_fraction,
            start_waypoint_max_fraction=args.start_waypoint_max_fraction,
            reset_base_to_trajectory_start=True,
            reset_anchor_target_blend=args.reset_anchor_target_blend,
        )

        assist_cfg = env_cfg.task_config.base_assist
        assist_cfg.enable = True
        assist_cfg.initial_blend = args.base_assist_blend
        assist_cfg.final_blend = args.base_assist_blend
        assist_cfg.decay_steps = 1
        assist_cfg.activation_distance = args.base_assist_activation_distance
        assist_cfg.full_speed_distance = args.base_assist_full_speed_distance
        assist_cfg.max_action = args.base_assist_max_action
        assist_cfg.imitation_weight = 0.0
        assist_cfg.yaw_enable = bool(args.include_yaw)
        assist_cfg.yaw_max_action = args.base_assist_yaw_max_action
        assist_cfg.yaw_full_error = args.base_assist_yaw_full_error

        base_env = MobileMMTrackEEEnv(cfg=env_cfg)

        class IsaacLabToSB3VecEnvWrapper(VecEnvWrapper):
            def __init__(self, venv):
                super().__init__(venv)
                if hasattr(venv.action_space, "shape") and len(venv.action_space.shape) > 1:
                    action_dim = venv.action_space.shape[-1]
                    self.action_space = spaces.Box(
                        low=venv.action_space.low.flatten()[0],
                        high=venv.action_space.high.flatten()[0],
                        shape=(action_dim,),
                        dtype=venv.action_space.dtype,
                    )
                dummy_obs = self._convert_obs(venv.reset())
                obs_shape = (dummy_obs.shape[1],) if len(dummy_obs.shape) > 1 else dummy_obs.shape
                self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)

            @staticmethod
            def _convert_obs(obs):
                if isinstance(obs, tuple):
                    obs = obs[0]
                if isinstance(obs, dict):
                    obs = obs.get("policy", list(obs.values())[0])
                if hasattr(obs, "detach"):
                    obs = obs.detach()
                if hasattr(obs, "cpu"):
                    obs = obs.cpu()
                return np.asarray(obs, dtype=np.float32)

            def reset(self):
                return self._convert_obs(self.venv.reset())

            def step_async(self, actions):
                if isinstance(actions, np.ndarray):
                    device = self.venv.unwrapped.device if hasattr(self.venv.unwrapped, "device") else "cuda:0"
                    actions = torch.from_numpy(actions).float().to(device)
                self._actions = actions

            def step_wait(self):
                result = self.venv.step(self._actions)
                if len(result) == 5:
                    obs, rewards, terminated, truncated, infos = result
                    dones = terminated | truncated
                else:
                    obs, rewards, dones, infos = result
                obs = self._convert_obs(obs)
                rewards = tensor_np(rewards).astype(np.float32)
                dones = tensor_np(dones).astype(bool)
                if isinstance(infos, dict):
                    infos = [infos.copy() for _ in range(len(rewards))]
                elif not isinstance(infos, list):
                    infos = [{} for _ in range(len(rewards))]
                return obs, rewards, dones, infos

        env = IsaacLabToSB3VecEnvWrapper(base_env)
        env = VecNormalize.load(str(vec_path), env)
        env.training = False
        env.norm_reward = False
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = PPO.load(str(checkpoint), env=env, device=device)
        raw_env = unwrap_isaac_env(env)

        obs = env.reset()
        observations: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        sample_weights: list[np.ndarray] = []
        active_count = 0

        for step in range(args.num_steps):
            obs_before = np.asarray(obs, dtype=np.float32).copy()
            policy_actions, _ = model.predict(obs, deterministic=args.deterministic)
            obs, _, _, _ = env.step(policy_actions)

            expert_xy = tensor_np(getattr(raw_env, "_base_assist_expert_action", None))
            coeff = tensor_np(getattr(raw_env, "_base_assist_coeff", None))
            if expert_xy is None or coeff is None:
                raise RuntimeError("base assist expert tensors were not produced by the environment")
            base_target_distance = tensor_np(getattr(raw_env, "_base_target_distance_buf", None))
            if args.sample_weight_mode == "base_distance" and base_target_distance is None:
                raise RuntimeError("_base_target_distance_buf was not produced by the environment")

            labels = np.zeros((args.num_envs, model.action_space.shape[0]), dtype=np.float32)
            valid = np.zeros_like(labels)
            active = coeff > 0.0
            labels[:, 6:8] = expert_xy[:, 0:2]
            valid[:, 6:8] = active[:, None].astype(np.float32)
            weights = np.ones((args.num_envs,), dtype=np.float32)
            if args.sample_weight_mode == "base_distance":
                denom = max(args.sample_weight_full_distance - args.sample_weight_distance_threshold, 1e-6)
                excess = np.clip((base_target_distance - args.sample_weight_distance_threshold) / denom, 0.0, 1.0)
                weights = 1.0 + (args.sample_weight_max - 1.0) * excess.astype(np.float32)

            if args.include_yaw:
                expert_wz = tensor_np(getattr(raw_env, "_base_assist_expert_wz_action", None))
                if expert_wz is not None:
                    labels[:, 8] = expert_wz[:, 0]
                    valid[:, 8] = active.astype(np.float32)

            if args.active_only:
                keep = active
                observations.append(obs_before[keep])
                actions.append(labels[keep])
                masks.append(valid[keep])
                sample_weights.append(weights[keep])
                active_count += int(np.sum(keep))
            else:
                observations.append(obs_before)
                actions.append(labels)
                masks.append(valid)
                sample_weights.append(weights)
                active_count += int(np.sum(active))

            if (step + 1) % 50 == 0 or step + 1 == args.num_steps:
                print(f"step {step + 1}/{args.num_steps}: collected={sum(x.shape[0] for x in observations):,} active={active_count:,}")

        obs_arr = np.concatenate(observations, axis=0)
        act_arr = np.concatenate(actions, axis=0)
        mask_arr = np.concatenate(masks, axis=0)
        weight_arr = np.concatenate(sample_weights, axis=0).astype(np.float32)
        out_path = output_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "checkpoint": str(checkpoint),
            "vec_normalize": str(vec_path),
            "num_envs": args.num_envs,
            "num_steps": args.num_steps,
            "active_only": args.active_only,
            "trajectory_stage": args.trajectory_stage,
            "trajectory_manifest_file": str(manifest),
            "trajectory_categories": sorted(trajectory_categories),
            "trajectory_files": sorted(trajectory_files),
            "action_indices": action_indices,
            "include_yaw": args.include_yaw,
            "base_assist_blend": args.base_assist_blend,
            "sample_weight_mode": args.sample_weight_mode,
            "sample_weight_distance_threshold": args.sample_weight_distance_threshold,
            "sample_weight_full_distance": args.sample_weight_full_distance,
            "sample_weight_max": args.sample_weight_max,
        }
        output_payload = {
            "observations": obs_arr,
            "actions": act_arr,
            "action_valid_mask": mask_arr,
            "metadata": json.dumps(metadata, indent=2),
        }
        if args.sample_weight_mode != "uniform":
            output_payload["sample_weight"] = weight_arr
        np.savez_compressed(out_path, **output_payload)
        valid_counts = mask_arr.sum(axis=0)
        print(f"saved: {out_path}")
        print(f"observations: {obs_arr.shape}, actions: {act_arr.shape}")
        print(f"valid action counts: {valid_counts.tolist()}")
        if args.sample_weight_mode != "uniform":
            print(
                "sample weights: "
                f"mean={weight_arr.mean():.4f}, min={weight_arr.min():.4f}, max={weight_arr.max():.4f}"
            )
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
