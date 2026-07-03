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

        manifest = PROJECT_ROOT / "trajectoryToLearn" / args.trajectory_stage / "manifest.txt"
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
        active_count = 0

        for step in range(args.num_steps):
            obs_before = np.asarray(obs, dtype=np.float32).copy()
            policy_actions, _ = model.predict(obs, deterministic=args.deterministic)
            obs, _, _, _ = env.step(policy_actions)

            expert_xy = tensor_np(getattr(raw_env, "_base_assist_expert_action", None))
            coeff = tensor_np(getattr(raw_env, "_base_assist_coeff", None))
            if expert_xy is None or coeff is None:
                raise RuntimeError("base assist expert tensors were not produced by the environment")

            labels = np.zeros((args.num_envs, model.action_space.shape[0]), dtype=np.float32)
            valid = np.zeros_like(labels)
            active = coeff > 0.0
            labels[:, 6:8] = expert_xy[:, 0:2]
            valid[:, 6:8] = active[:, None].astype(np.float32)

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
                active_count += int(np.sum(keep))
            else:
                observations.append(obs_before)
                actions.append(labels)
                masks.append(valid)
                active_count += int(np.sum(active))

            if (step + 1) % 50 == 0 or step + 1 == args.num_steps:
                print(f"step {step + 1}/{args.num_steps}: collected={sum(x.shape[0] for x in observations):,} active={active_count:,}")

        obs_arr = np.concatenate(observations, axis=0)
        act_arr = np.concatenate(actions, axis=0)
        mask_arr = np.concatenate(masks, axis=0)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "checkpoint": str(checkpoint),
            "vec_normalize": str(vec_path),
            "num_envs": args.num_envs,
            "num_steps": args.num_steps,
            "active_only": args.active_only,
            "action_indices": action_indices,
            "include_yaw": args.include_yaw,
            "base_assist_blend": args.base_assist_blend,
        }
        np.savez_compressed(
            out_path,
            observations=obs_arr,
            actions=act_arr,
            action_valid_mask=mask_arr,
            metadata=json.dumps(metadata, indent=2),
        )
        valid_counts = mask_arr.sum(axis=0)
        print(f"saved: {out_path}")
        print(f"observations: {obs_arr.shape}, actions: {act_arr.shape}")
        print(f"valid action counts: {valid_counts.tolist()}")
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
