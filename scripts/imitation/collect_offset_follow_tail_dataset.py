#!/usr/bin/env python3
"""Collect base labels from the offset-follow teacher on late-start Stage B states.

The collector runs a checkpoint in Isaac, records the policy observations visited
under the current policy, and labels base action rows from the offset-preserving
teacher:

    desired_base_xy = target_xy - [reset_base_x_offset, reset_base_y_offset]

Labels are stored in the raw policy action space.  If rollout/evaluation uses
``--base_action_scale 0.25``, the teacher action seen inside the env is divided
by 0.25 before saving so the trained policy is not scaled down twice.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vec_normalize", required=True)
    parser.add_argument("--disable_vec_normalize", action="store_true")
    parser.add_argument("--trajectory_stage", default="stage0_policy_envelope_fk_base040")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--max_trajectories", type=int, default=24)
    parser.add_argument("--min_trajectory_duration", type=float, default=5.0)
    parser.add_argument("--base_action_scale", type=float, default=0.25)
    parser.add_argument("--start_waypoint_min_fraction", type=float, default=0.65)
    parser.add_argument("--start_waypoint_max_fraction", type=float, default=0.95)
    parser.add_argument("--base_assist_activation_distance", type=float, default=0.010)
    parser.add_argument("--base_assist_full_speed_distance", type=float, default=0.080)
    parser.add_argument("--base_assist_max_action", type=float, default=0.60)
    parser.add_argument(
        "--teacher_blend",
        type=float,
        default=0.0,
        help="Executed assist blend. Keep 0.0 to collect unassisted visited states.",
    )
    parser.add_argument(
        "--min_expert_env_action_norm",
        type=float,
        default=0.005,
        help="Rows below this env-action norm are masked out unless --keep_inactive is set.",
    )
    parser.add_argument("--keep_inactive", action="store_true")
    parser.add_argument(
        "--sample_weight_mode",
        choices=("uniform", "expert_norm", "pos_error"),
        default="expert_norm",
    )
    parser.add_argument("--sample_weight_max", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


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


def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {"mean": float("nan"), "p50": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


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
    raw_reward_overrides = data.get("reward_overrides", {})
    if isinstance(raw_reward_overrides, dict):
        out["reward_overrides"] = {str(k): float(v) for k, v in raw_reward_overrides.items()}
    return out


def main() -> int:
    args = parse_args()
    require(args.num_envs > 0, "--num_envs must be positive")
    require(args.steps > 0, "--steps must be positive")
    require(0.0 < args.base_action_scale <= 1.0, "--base_action_scale must be in (0, 1]")
    require(0.0 <= args.start_waypoint_min_fraction <= 1.0, "--start_waypoint_min_fraction must be in [0, 1]")
    require(0.0 <= args.start_waypoint_max_fraction <= 1.0, "--start_waypoint_max_fraction must be in [0, 1]")
    require(0.0 <= args.teacher_blend <= 1.0, "--teacher_blend must be in [0, 1]")
    require(args.base_assist_full_speed_distance >= args.base_assist_activation_distance, "full speed distance must be >= activation distance")
    require(args.sample_weight_max >= 1.0, "--sample_weight_max must be >= 1")

    checkpoint = Path(args.checkpoint)
    vec_normalize = Path(args.vec_normalize)
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    if not vec_normalize.is_absolute():
        vec_normalize = PROJECT_ROOT / vec_normalize
    require(checkpoint.exists(), f"checkpoint not found: {checkpoint}")
    if not args.disable_vec_normalize:
        require(vec_normalize.exists(), f"vec_normalize not found: {vec_normalize}")

    from isaaclab.app import AppLauncher

    print("[offset-dataset] launching Isaac", flush=True)
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

        env_cfg = MobileMMTrackEEEnvCfg()
        env_cfg.num_envs = args.num_envs
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = args.seed
        env_cfg.task_config.obstacles.enable_obstacles = False
        env_cfg.task_config.randomize_initial_joint_positions = False
        reward_overrides = reset_config.get("reward_overrides", {})
        if isinstance(reward_overrides, dict):
            for name, value in reward_overrides.items():
                setattr(env_cfg.task_config.rewards, name, float(value))
        env_cfg.task_config.trajectory = TrajectoryConfig(
            type="multi_recorded",
            trajectory_dir=str(PROJECT_ROOT),
            trajectory_manifest_file=str(manifest),
            max_trajectories=args.max_trajectories,
            min_duration_seconds=args.min_trajectory_duration,
            randomize_start_waypoint=True,
            start_waypoint_min_fraction=args.start_waypoint_min_fraction,
            start_waypoint_max_fraction=args.start_waypoint_max_fraction,
            reset_base_to_trajectory_start=False,
            reset_anchor_target_blend=0.0,
            reset_base_x_offset=reset_config.get("reset_base_x_offset", 0.4415),
            reset_base_y_offset=reset_config.get("reset_base_y_offset", 0.2405),
        )

        assist_cfg = env_cfg.task_config.base_assist
        assist_cfg.enable = True
        assist_cfg.mode = "target_offset_follow"
        assist_cfg.initial_blend = args.teacher_blend
        assist_cfg.final_blend = args.teacher_blend
        assist_cfg.decay_steps = 1
        assist_cfg.activation_distance = args.base_assist_activation_distance
        assist_cfg.full_speed_distance = args.base_assist_full_speed_distance
        assist_cfg.max_action = args.base_assist_max_action
        assist_cfg.imitation_weight = 0.0
        assist_cfg.lookahead_steps = 0

        print("[offset-dataset] creating env", flush=True)
        base_env = MobileMMTrackEEEnv(cfg=env_cfg)
        print("[offset-dataset] env created", flush=True)

        class IsaacLabToSB3VecEnvWrapper(VecEnvWrapper):
            def __init__(self, venv, expected_obs_dim: int | None) -> None:
                super().__init__(venv)
                self.expected_obs_dim = expected_obs_dim
                self._obs_space_updated = False
                self._base_adapter_logged = False
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
                    raise ValueError(f"expected at least 9 action dims, got {actions.shape[-1]}")
                if abs(args.base_action_scale - 1.0) >= 1e-9:
                    actions = actions.clone()
                    if not self._base_adapter_logged:
                        print(
                            f"[offset-dataset action-adapter] scaling base rows [6,7,8] by {args.base_action_scale:.3f}",
                            flush=True,
                        )
                        self._base_adapter_logged = True
                    actions[..., 6:9] *= args.base_action_scale
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
        env = IsaacLabToSB3VecEnvWrapper(base_env, expected_obs_dim)
        _ = env.reset()
        if not args.disable_vec_normalize:
            env = VecNormalize.load(str(vec_normalize), env)
            env.training = False
            env.norm_reward = False
        obs = env.reset()
        model = PPO.load(str(checkpoint), env=env, device="cuda:0")
        raw_env = base_env.unwrapped if hasattr(base_env, "unwrapped") else base_env
        action_dim = int(model.action_space.shape[0])

        observations: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        policy_actions: list[np.ndarray] = []
        expert_env_actions: list[np.ndarray] = []
        pos_errors: list[np.ndarray] = []
        base_target_distances: list[np.ndarray] = []
        expert_norms: list[np.ndarray] = []
        sample_weights: list[np.ndarray] = []
        waypoint_rows: list[np.ndarray] = []

        for step in range(args.steps):
            obs_before = np.asarray(obs, dtype=np.float32).copy()
            action, _ = model.predict(obs, deterministic=True)
            target_pos, _ = raw_env.trajectory_manager.get_target_pose()
            ee_pos = raw_env.robot.data.body_pos_w[:, raw_env._ee_body_idx, :]
            pos_error_before = tensor_np(torch.linalg.norm(target_pos - ee_pos, dim=-1)).astype(np.float32)
            waypoint_idx = tensor_np(getattr(raw_env.trajectory_manager, "current_waypoint_idx", None))
            obs, _, _, _ = env.step(action)

            expert_env = tensor_np(getattr(raw_env, "_base_assist_expert_action", None))
            base_dist = tensor_np(getattr(raw_env, "_base_target_distance_buf", None))
            require(expert_env is not None, "env did not produce _base_assist_expert_action")
            require(base_dist is not None, "env did not produce _base_target_distance_buf")
            expert_norm = np.linalg.norm(expert_env[:, :2], axis=1).astype(np.float32)
            active = expert_norm >= float(args.min_expert_env_action_norm)
            keep = np.ones((args.num_envs,), dtype=bool) if args.keep_inactive else active

            raw_label = np.zeros((args.num_envs, action_dim), dtype=np.float32)
            env_label = np.zeros((args.num_envs, action_dim), dtype=np.float32)
            valid = np.zeros((args.num_envs, action_dim), dtype=bool)
            env_label[:, 6:8] = expert_env[:, :2]
            raw_label[:, 6:8] = np.clip(expert_env[:, :2] / float(args.base_action_scale), -1.0, 1.0)
            valid[:, 6:8] = active[:, None]

            if args.sample_weight_mode == "expert_norm":
                denom = max(float(args.base_assist_max_action), 1e-6)
                normalized = np.clip(expert_norm / denom, 0.0, 1.0)
                weights = 1.0 + (float(args.sample_weight_max) - 1.0) * normalized
            elif args.sample_weight_mode == "pos_error":
                normalized = np.clip(pos_error_before / 0.20, 0.0, 1.0)
                weights = 1.0 + (float(args.sample_weight_max) - 1.0) * normalized
            else:
                weights = np.ones((args.num_envs,), dtype=np.float32)

            observations.append(obs_before[keep])
            labels.append(raw_label[keep])
            masks.append(valid[keep])
            policy_actions.append(action[keep, :action_dim].astype(np.float32, copy=True))
            expert_env_actions.append(env_label[keep])
            pos_errors.append(pos_error_before[keep])
            base_target_distances.append(base_dist[keep].astype(np.float32))
            expert_norms.append(expert_norm[keep])
            sample_weights.append(weights[keep].astype(np.float32))
            if waypoint_idx is not None:
                waypoint_rows.append(waypoint_idx[keep].astype(np.int32))

            if step % 20 == 0 or step + 1 == args.steps:
                total = sum(chunk.shape[0] for chunk in observations)
                print(
                    f"[offset-dataset] step={step}/{args.steps} collected={total:,} "
                    f"active={int(active.sum())}/{args.num_envs} "
                    f"expert_norm_mean={float(expert_norm.mean()):.4f}",
                    flush=True,
                )

        obs_arr = np.concatenate(observations, axis=0).astype(np.float32)
        action_arr = np.concatenate(labels, axis=0).astype(np.float32)
        mask_arr = np.concatenate(masks, axis=0).astype(bool)
        policy_arr = np.concatenate(policy_actions, axis=0).astype(np.float32)
        expert_env_arr = np.concatenate(expert_env_actions, axis=0).astype(np.float32)
        pos_arr = np.concatenate(pos_errors, axis=0).astype(np.float32)
        base_dist_arr = np.concatenate(base_target_distances, axis=0).astype(np.float32)
        expert_norm_arr = np.concatenate(expert_norms, axis=0).astype(np.float32)
        weight_arr = np.concatenate(sample_weights, axis=0).astype(np.float32)
        waypoint_arr = np.concatenate(waypoint_rows, axis=0).astype(np.int32) if waypoint_rows else np.zeros((obs_arr.shape[0],), dtype=np.int32)

        output_npz = Path(args.output_npz)
        if not output_npz.is_absolute():
            output_npz = PROJECT_ROOT / output_npz
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        clip_fraction = float(np.mean(np.abs(expert_env_arr[:, 6:8] / float(args.base_action_scale)) > 1.0))
        metadata = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "schema": "offset_follow_tail_base_dataset_v1",
            "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT) if checkpoint.is_relative_to(PROJECT_ROOT) else checkpoint),
            "trajectory_stage": args.trajectory_stage,
            "num_envs": int(args.num_envs),
            "steps": int(args.steps),
            "samples": int(obs_arr.shape[0]),
            "obs_dim": int(obs_arr.shape[1]),
            "act_dim": int(action_arr.shape[1]),
            "valid_counts": mask_arr.sum(axis=0).astype(int).tolist(),
            "base_action_scale": float(args.base_action_scale),
            "label_space": "raw_policy_pre_base_action_scale",
            "teacher_env_action_rows": [6, 7],
            "teacher_env_action": "target_offset_follow",
            "teacher_blend": float(args.teacher_blend),
            "base_assist_activation_distance": float(args.base_assist_activation_distance),
            "base_assist_full_speed_distance": float(args.base_assist_full_speed_distance),
            "base_assist_max_action": float(args.base_assist_max_action),
            "min_expert_env_action_norm": float(args.min_expert_env_action_norm),
            "keep_inactive": bool(args.keep_inactive),
            "sample_weight_mode": args.sample_weight_mode,
            "sample_weight_max": float(args.sample_weight_max),
            "start_waypoint_min_fraction": float(args.start_waypoint_min_fraction),
            "start_waypoint_max_fraction": float(args.start_waypoint_max_fraction),
            "pos_error_m": summarize(pos_arr),
            "base_target_distance_m": summarize(base_dist_arr),
            "expert_env_action_norm": summarize(expert_norm_arr),
            "raw_label_clip_fraction": clip_fraction,
        }
        np.savez_compressed(
            output_npz,
            observations=obs_arr,
            actions=action_arr,
            action_valid_mask=mask_arr,
            sample_weight=weight_arr,
            policy_actions=policy_arr,
            expert_env_actions=expert_env_arr,
            source_pos_error_m=pos_arr,
            source_base_target_distance_m=base_dist_arr,
            expert_env_action_norm=expert_norm_arr,
            source_waypoint_idx=waypoint_arr,
            metadata=json.dumps(metadata, indent=2),
            action_contract="sim_6joint_gimbal_v1",
            observation_dim=np.asarray(obs_arr.shape[1], dtype=np.int32),
        )

        summary = {
            "output_npz": str(output_npz),
            "samples": int(obs_arr.shape[0]),
            "valid_counts": mask_arr.sum(axis=0).astype(int).tolist(),
            "pos_error_m": summarize(pos_arr),
            "base_target_distance_m": summarize(base_dist_arr),
            "expert_env_action_norm": summarize(expert_norm_arr),
            "sample_weight": summarize(weight_arr),
            "raw_label_clip_fraction": clip_fraction,
        }
        print(json.dumps(summary, indent=2), flush=True)
        if args.output_json:
            output_json = Path(args.output_json)
            if not output_json.is_absolute():
                output_json = PROJECT_ROOT / output_json
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(json.dumps({**metadata, **summary}, indent=2) + "\n", encoding="utf-8")
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
