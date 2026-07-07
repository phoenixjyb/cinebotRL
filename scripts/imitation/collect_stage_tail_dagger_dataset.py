#!/usr/bin/env python3
"""Collect DAgger-style tail states from a staged FK rollout.

The collector runs a checkpoint in Isaac, records actual visited observations,
then relabels selected late/high-error states with the generated-stage expert
action for the same trajectory and waypoint.  This is intentionally separate
from training so bad relabel distributions can be inspected before promotion.
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
    parser.add_argument(
        "--expert_dataset",
        default="data/policy_envelope_fk_base040/obs_dataset_policy_envelope_fk_base040_arm6_base3.npz",
    )
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--max_trajectories", type=int, default=8)
    parser.add_argument("--min_trajectory_duration", type=float, default=5.0)
    parser.add_argument("--base_action_scale", type=float, default=0.25)
    parser.add_argument("--tail_start_fraction", type=float, default=0.65)
    parser.add_argument("--error_percentile", type=float, default=80.0)
    parser.add_argument("--min_selected", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260707)
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


def trajectory_index_from_file(path_text: str) -> int:
    name = path_text.replace("\\", "/").rsplit("/", 1)[-1]
    suffix = "_policy_envelope_fk_base_required.json"
    if not name.endswith(suffix):
        raise ValueError(f"cannot infer trajectory index from file: {path_text}")
    return int(name[: -len(suffix)])


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


def main() -> int:
    args = parse_args()
    require(0.0 <= args.base_action_scale <= 1.0, "--base_action_scale must be in [0, 1]")
    require(0.0 <= args.tail_start_fraction < 1.0, "--tail_start_fraction must be in [0, 1)")
    require(0.0 <= args.error_percentile <= 100.0, "--error_percentile must be in [0, 100]")
    require(args.num_envs > 0, "--num_envs must be positive")
    require(args.steps > 0, "--steps must be positive")

    checkpoint = Path(args.checkpoint)
    vec_normalize = Path(args.vec_normalize)
    expert_dataset = Path(args.expert_dataset)
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    if not vec_normalize.is_absolute():
        vec_normalize = PROJECT_ROOT / vec_normalize
    if not expert_dataset.is_absolute():
        expert_dataset = PROJECT_ROOT / expert_dataset
    require(checkpoint.exists(), f"checkpoint not found: {checkpoint}")
    if not args.disable_vec_normalize:
        require(vec_normalize.exists(), f"vec_normalize not found: {vec_normalize}")
    require(expert_dataset.exists(), f"expert dataset not found: {expert_dataset}")

    with np.load(expert_dataset, allow_pickle=False) as data:
        expert_actions = data["actions"].astype(np.float32)
        expert_obs_dim = int(data["observations"].shape[1])
        expert_metadata = json.loads(str(data["metadata"])) if "metadata" in data else {}
    num_waypoints = int(expert_metadata.get("num_waypoints", args.steps))
    require(num_waypoints > 0, "expert metadata has invalid num_waypoints")

    from isaaclab.app import AppLauncher

    print("[dagger] launching Isaac", flush=True)
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
        env_cfg.task_config.base_assist.enable = False
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
            randomize_start_waypoint=False,
            reset_base_to_trajectory_start=False,
            reset_anchor_target_blend=0.0,
            reset_base_x_offset=reset_config.get("reset_base_x_offset", 0.4415),
            reset_base_y_offset=reset_config.get("reset_base_y_offset", 0.2405),
        )

        print("[dagger] creating env", flush=True)
        base_env = MobileMMTrackEEEnv(cfg=env_cfg)
        print("[dagger] env created", flush=True)

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
                            f"[dagger action-adapter] scaling base rows [6,7,8] by {args.base_action_scale:.3f}",
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
        require(expected_obs_dim == expert_obs_dim, f"checkpoint obs dim {expected_obs_dim} != expert obs dim {expert_obs_dim}")
        env = IsaacLabToSB3VecEnvWrapper(base_env, expected_obs_dim)
        _ = env.reset()
        if not args.disable_vec_normalize:
            env = VecNormalize.load(str(vec_normalize), env)
            env.training = False
            env.norm_reward = False
        obs = env.reset()
        model = PPO.load(str(checkpoint), env=env, device="cuda:0")
        raw_env = base_env.unwrapped if hasattr(base_env, "unwrapped") else base_env

        obs_rows: list[np.ndarray] = []
        action_rows: list[np.ndarray] = []
        mask_rows: list[np.ndarray] = []
        pos_errors: list[float] = []
        step_rows: list[int] = []
        waypoint_rows: list[int] = []
        trajectory_rows: list[int] = []
        policy_action_rows: list[np.ndarray] = []

        for step in range(args.steps):
            target_pos, _ = raw_env.trajectory_manager.get_target_pose()
            ee_pos = raw_env.robot.data.body_pos_w[:, raw_env._ee_body_idx, :]
            pos_error = tensor_np(torch.linalg.norm(target_pos - ee_pos, dim=-1)).astype(np.float32)
            waypoint_idx = tensor_np(getattr(raw_env.trajectory_manager, "current_waypoint_idx", None))
            metadata = getattr(raw_env.trajectory_manager, "current_trajectory_metadata", None) or []
            action, _ = model.predict(obs, deterministic=True)

            for env_id in range(args.num_envs):
                item = metadata[env_id] if env_id < len(metadata) and isinstance(metadata[env_id], dict) else {}
                traj_file = str(item.get("file", ""))
                traj_idx = trajectory_index_from_file(traj_file)
                wp_idx = int(waypoint_idx[env_id]) if waypoint_idx is not None else step
                wp_idx = max(0, min(wp_idx, num_waypoints - 1))
                dataset_row = traj_idx * num_waypoints + wp_idx
                require(dataset_row < expert_actions.shape[0], f"expert row out of range: {dataset_row}")
                obs_rows.append(obs[env_id].astype(np.float32, copy=True))
                action_rows.append(expert_actions[dataset_row].astype(np.float32, copy=True))
                mask_rows.append(np.ones((expert_actions.shape[1],), dtype=bool))
                pos_errors.append(float(pos_error[env_id]))
                step_rows.append(step)
                waypoint_rows.append(wp_idx)
                trajectory_rows.append(traj_idx)
                policy_action_rows.append(action[env_id, : expert_actions.shape[1]].astype(np.float32, copy=True))

            obs, _, _, _ = env.step(action)
            if step % 20 == 0:
                print(f"[dagger] step={step}/{args.steps}", flush=True)

        all_obs = np.asarray(obs_rows, dtype=np.float32)
        all_actions = np.asarray(action_rows, dtype=np.float32)
        all_masks = np.asarray(mask_rows, dtype=bool)
        all_errors = np.asarray(pos_errors, dtype=np.float32)
        all_steps = np.asarray(step_rows, dtype=np.int32)
        all_waypoints = np.asarray(waypoint_rows, dtype=np.int32)
        all_trajectories = np.asarray(trajectory_rows, dtype=np.int32)
        all_policy_actions = np.asarray(policy_action_rows, dtype=np.float32)

        error_threshold = float(np.percentile(all_errors, args.error_percentile))
        tail_step = int(np.floor(args.tail_start_fraction * max(args.steps - 1, 1)))
        selected = (all_steps >= tail_step) & (all_errors >= error_threshold)
        if int(np.count_nonzero(selected)) < args.min_selected:
            order = np.argsort(-all_errors)
            selected = np.zeros_like(all_errors, dtype=bool)
            selected[order[: min(args.min_selected, all_errors.size)]] = True

        selected_obs = all_obs[selected]
        selected_actions = all_actions[selected]
        selected_masks = all_masks[selected]
        selected_errors = all_errors[selected]
        selected_steps = all_steps[selected]
        selected_waypoints = all_waypoints[selected]
        selected_trajectories = all_trajectories[selected]
        selected_policy_actions = all_policy_actions[selected]

        output_npz = Path(args.output_npz)
        if not output_npz.is_absolute():
            output_npz = PROJECT_ROOT / output_npz
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "schema": "stage_tail_dagger_dataset_v1",
            "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT) if checkpoint.is_relative_to(PROJECT_ROOT) else checkpoint),
            "trajectory_stage": args.trajectory_stage,
            "expert_dataset": str(expert_dataset.relative_to(PROJECT_ROOT) if expert_dataset.is_relative_to(PROJECT_ROOT) else expert_dataset),
            "num_envs": int(args.num_envs),
            "steps": int(args.steps),
            "source_samples": int(all_obs.shape[0]),
            "selected_samples": int(selected_obs.shape[0]),
            "obs_dim": int(selected_obs.shape[1]),
            "act_dim": int(selected_actions.shape[1]),
            "tail_start_fraction": float(args.tail_start_fraction),
            "tail_step": int(tail_step),
            "error_percentile": float(args.error_percentile),
            "error_threshold_m": float(error_threshold),
            "source_error_m": summarize(all_errors),
            "selected_error_m": summarize(selected_errors),
            "num_waypoints": int(num_waypoints),
            "base_action_scale": float(args.base_action_scale),
        }
        np.savez_compressed(
            output_npz,
            observations=selected_obs,
            actions=selected_actions,
            action_valid_mask=selected_masks,
            source_pos_error_m=selected_errors,
            source_step=selected_steps,
            source_waypoint_idx=selected_waypoints,
            source_trajectory_idx=selected_trajectories,
            policy_actions=selected_policy_actions,
            metadata=json.dumps(metadata, indent=2),
            action_contract=np.asarray("sim_6joint_gimbal_base_v1"),
            observation_dim=np.asarray(selected_obs.shape[1], dtype=np.int32),
        )
        print(json.dumps(metadata, indent=2), flush=True)
        if args.output_json:
            output_json = Path(args.output_json)
            if not output_json.is_absolute():
                output_json = PROJECT_ROOT / output_json
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            print(f"[dagger] wrote {output_json}", flush=True)
        print(f"[dagger] wrote {output_npz}", flush=True)
        env.close()
    finally:
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
