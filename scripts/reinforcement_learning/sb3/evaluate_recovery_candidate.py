"""Evaluate a stage1 recovery SB3 checkpoint under the matching Proto2 task setup."""

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
    parser = argparse.ArgumentParser(description="Evaluate Proto2 stage1 recovery candidate.")
    parser.add_argument("--checkpoint", required=True, help="PPO checkpoint/final_model.zip.")
    parser.add_argument("--vec_normalize", default=None, help="VecNormalize pkl. Defaults to checkpoint parent vec_normalize.pkl.")
    parser.add_argument("--task", default="RecomoProto2TrackEE-v0")
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--num_episodes", type=int, default=256)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trajectory_stage", default="stage1_recovery")
    parser.add_argument("--start_waypoint_min_fraction", type=float, default=0.25)
    parser.add_argument("--start_waypoint_max_fraction", type=float, default=0.70)
    parser.add_argument("--reset_anchor_target_blend", type=float, default=0.35)
    parser.add_argument("--enable_obstacles", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable_base_assist", action="store_true")
    parser.add_argument("--base_assist_blend", type=float, default=0.70)
    parser.add_argument("--base_assist_activation_distance", type=float, default=0.45)
    parser.add_argument("--base_assist_full_speed_distance", type=float, default=0.90)
    parser.add_argument("--base_assist_max_action", type=float, default=1.0)
    parser.add_argument("--output_dir", default="evaluation_results/recovery_candidate")
    return parser.parse_args()


class ScalarCollector:
    def __init__(self) -> None:
        self.values: dict[str, list[float]] = {}

    def add(self, name: str, value: float) -> None:
        self.values.setdefault(name, []).append(float(value))

    def summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for name, values in self.values.items():
            arr = np.asarray(values, dtype=np.float64)
            out[name] = {
                "mean": float(np.mean(arr)),
                "last": float(arr[-1]),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "p95": float(np.percentile(arr, 95)),
            }
        return out


def unwrap_isaac_env(env):
    raw = env
    while hasattr(raw, "venv"):
        raw = raw.venv
    if hasattr(raw, "unwrapped"):
        raw = raw.unwrapped
    return raw


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


def main() -> int:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        print(f"checkpoint not found: {checkpoint}")
        return 1

    vec_path = Path(args.vec_normalize) if args.vec_normalize else checkpoint.parent / "vec_normalize.pkl"
    if not vec_path.exists():
        print(f"VecNormalize stats not found: {vec_path}")
        return 1

    print("=" * 80)
    print("Proto2 Stage1 Recovery Candidate Evaluation")
    print("=" * 80)
    print(f"checkpoint: {checkpoint}")
    print(f"vec_normalize: {vec_path}")
    print(f"mode: {'assisted' if args.enable_base_assist else 'raw-policy'}")
    print(f"episodes/envs: {args.num_episodes}/{args.num_envs}")

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
        assist_cfg.enable = bool(args.enable_base_assist)
        assist_cfg.initial_blend = args.base_assist_blend
        assist_cfg.final_blend = args.base_assist_blend
        assist_cfg.decay_steps = 1
        assist_cfg.activation_distance = args.base_assist_activation_distance
        assist_cfg.full_speed_distance = args.base_assist_full_speed_distance
        assist_cfg.max_action = args.base_assist_max_action
        assist_cfg.imitation_weight = 0.0

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
                self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
                )

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
        model = PPO.load(str(checkpoint), env=env, device="cuda" if torch.cuda.is_available() else "cpu")

        raw_env = unwrap_isaac_env(env)
        obs = env.reset()
        collector = ScalarCollector()
        rewards_by_env = np.zeros(args.num_envs, dtype=np.float64)
        lengths_by_env = np.zeros(args.num_envs, dtype=np.int32)
        episode_rewards: list[float] = []
        episode_lengths: list[int] = []
        step = 0
        max_steps = max(10_000, int(args.num_episodes / max(args.num_envs, 1) * 800) + 1_000)

        while len(episode_rewards) < args.num_episodes and step < max_steps:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, rewards, dones, _infos = env.step(action)
            rewards_by_env += rewards
            lengths_by_env += 1

            base_dist = tensor_np(getattr(raw_env, "_base_target_distance_buf", None))
            workspace = tensor_np(getattr(raw_env, "_workspace_distance_buf", None))
            pos_error = tensor_np(getattr(raw_env, "ee_pos_error_buf", None))
            ori_error = tensor_np(getattr(raw_env, "ee_ori_error_buf", None))
            assist_coeff = tensor_np(getattr(raw_env, "_base_assist_coeff", None))

            if base_dist is not None:
                collector.add("nonfinite_reachability_pct", float(np.mean(~np.isfinite(base_dist)) * 100.0))
                finite_base_dist = base_dist[np.isfinite(base_dist)]
                if finite_base_dist.size:
                    collector.add("base_target_dist_mean", float(np.mean(finite_base_dist)))
                    collector.add("base_target_dist_max", float(np.max(finite_base_dist)))
                    collector.add("optimal_zone_pct", float(np.mean(finite_base_dist < 0.40) * 100.0))
                    collector.add("acceptable_zone_pct", float(np.mean((finite_base_dist >= 0.40) & (finite_base_dist <= 0.70)) * 100.0))
                    collector.add("unreachable_zone_pct", float(np.mean(finite_base_dist > 0.70) * 100.0))
            if workspace is not None:
                collector.add("workspace_soft_exceed_pct", float(np.mean(workspace > 0.20) * 100.0))
                collector.add("workspace_hard_exceed_pct", float(np.mean(workspace > 0.70) * 100.0))
                collector.add("workspace_distance_max", float(np.max(workspace)))
            if pos_error is not None:
                pos_norm = np.linalg.norm(pos_error, axis=1)
                pos_norm = pos_norm[np.isfinite(pos_norm)]
                if pos_norm.size:
                    collector.add("ee_pos_error_mean_m", float(np.mean(pos_norm)))
                    collector.add("ee_pos_error_p95_m", float(np.percentile(pos_norm, 95)))
            if ori_error is not None:
                finite_ori = ori_error[np.isfinite(ori_error)]
                if finite_ori.size:
                    collector.add("ee_ori_error_mean_deg", float(np.degrees(np.mean(finite_ori))))
            if assist_coeff is not None:
                collector.add("base_assist_active_pct", float(np.mean(assist_coeff > 0.0) * 100.0))
                collector.add("base_assist_coeff_mean", float(np.mean(assist_coeff)))
            if getattr(raw_env, "obstacles_enabled", False):
                clearance = tensor_np(raw_env._get_obstacle_clearance(raw_env.robot.data.root_pos_w))
                collector.add("obstacle_unsafe_pct", float(np.mean(clearance < 0.20) * 100.0))
                collector.add("obstacle_collision_pct", float(np.mean(clearance < 0.0) * 100.0))
                collector.add("obstacle_clearance_min", float(np.min(clearance)))

            for idx, done in enumerate(dones):
                if done:
                    episode_rewards.append(float(rewards_by_env[idx]))
                    episode_lengths.append(int(lengths_by_env[idx]))
                    rewards_by_env[idx] = 0.0
                    lengths_by_env[idx] = 0
                    if len(episode_rewards) >= args.num_episodes:
                        break

            step += 1

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "timestamp": timestamp,
            "checkpoint": str(checkpoint),
            "vec_normalize": str(vec_path),
            "mode": "assisted" if args.enable_base_assist else "raw-policy",
            "num_envs": args.num_envs,
            "episodes_completed": len(episode_rewards),
            "steps": step,
            "episode_reward_mean": float(np.mean(episode_rewards)) if episode_rewards else None,
            "episode_length_mean": float(np.mean(episode_lengths)) if episode_lengths else None,
            "metrics": collector.summary(),
        }
        out_file = output_dir / f"recovery_eval_{summary['mode']}_{timestamp}.json"
        out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print("=" * 80)
        print("Evaluation summary")
        print("=" * 80)
        print(json.dumps(summary, indent=2))
        print(f"saved: {out_file}")

        env.close()
        simulation_app.close()
        return 0 if len(episode_rewards) >= args.num_episodes else 2
    except Exception:
        import traceback

        traceback.print_exc()
        simulation_app.close()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
