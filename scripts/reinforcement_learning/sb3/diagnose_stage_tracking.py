#!/usr/bin/env python3
"""Per-trajectory tracking diagnostics for Stage A PPO checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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
    parser.add_argument("--vec_normalize", required=True)
    parser.add_argument("--disable_vec_normalize", action="store_true")
    parser.add_argument("--trajectory_stage", default="stage0_policy_envelope_fk")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--max_trajectories", type=int, default=None)
    parser.add_argument("--min_trajectory_duration", type=float, default=5.0)
    parser.add_argument("--random_start_waypoint", action="store_true")
    parser.add_argument("--start_waypoint_min_fraction", type=float, default=0.0)
    parser.add_argument("--start_waypoint_max_fraction", type=float, default=0.0)
    parser.add_argument("--reset_base_to_trajectory_start", action="store_true")
    parser.add_argument("--reset_anchor_target_blend", type=float, default=0.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--freeze_base_actions", action="store_true")
    parser.add_argument("--base_action_scale", type=float, default=1.0)
    parser.add_argument("--time_bins", type=int, default=6)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--output_json", required=True)
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
        out["reward_overrides"] = {
            str(name): float(value)
            for name, value in raw_reward_overrides.items()
        }
    return out


def infer_source(trajectory_file: str) -> str:
    if "base_required" in trajectory_file:
        return "base025"
    if trajectory_file.endswith("_policy_envelope_fk.json") or "policy_envelope_fk.json" in trajectory_file:
        return "large08"
    if "stage0_policy_envelope_fk_base025" in trajectory_file:
        return "base025"
    if "stage0_policy_envelope_fk_large08" in trajectory_file:
        return "large08"
    if "stage0_policy_envelope_fk_mix_large08_base025" in trajectory_file:
        return "mixed"
    return "other"


def percentile(values: np.ndarray, pct: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values.astype(np.float64), pct))


def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {"mean": float("nan"), "p50": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(values)),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "max": float(np.max(values)),
    }


def trajectory_snapshots(manager, num_envs: int) -> list[dict[str, object]]:
    metadata = getattr(manager, "current_trajectory_metadata", None) or []
    waypoint_idx = tensor_np(getattr(manager, "current_waypoint_idx", None))
    lengths = tensor_np(getattr(manager, "recorded_lengths", None))
    snapshots: list[dict[str, object]] = []
    for env_id in range(num_envs):
        item = metadata[env_id] if env_id < len(metadata) and isinstance(metadata[env_id], dict) else {}
        file_name = str(item.get("file", "unknown"))
        length = int(lengths[env_id]) if lengths is not None and env_id < len(lengths) else item.get("length")
        current_idx = int(waypoint_idx[env_id]) if waypoint_idx is not None and env_id < len(waypoint_idx) else None
        snapshots.append(
            {
                "env_id": env_id,
                "trajectory_file": file_name,
                "trajectory_category": str(item.get("category", "unknown")),
                "trajectory_source": infer_source(file_name),
                "trajectory_length": int(length) if length is not None else None,
                "start_waypoint_idx": current_idx,
            }
        )
    return snapshots


def main() -> int:
    args = parse_args()
    require(0.0 <= args.base_action_scale <= 1.0, "--base_action_scale must be in [0,1]")
    require(args.num_envs > 0, "--num_envs must be positive")
    require(args.steps > 0, "--steps must be positive")
    if args.start_waypoint_min_fraction < 0.0 or args.start_waypoint_min_fraction > 1.0:
        raise ValueError("--start_waypoint_min_fraction must be in [0, 1]")
    if args.start_waypoint_max_fraction < 0.0 or args.start_waypoint_max_fraction > 1.0:
        raise ValueError("--start_waypoint_max_fraction must be in [0, 1]")
    if args.reset_anchor_target_blend < 0.0 or args.reset_anchor_target_blend > 1.0:
        raise ValueError("--reset_anchor_target_blend must be in [0, 1]")
    require(args.time_bins > 0, "--time_bins must be positive")
    checkpoint = Path(args.checkpoint)
    vec_normalize = Path(args.vec_normalize)
    require(checkpoint.exists(), f"checkpoint not found: {checkpoint}")
    if not args.disable_vec_normalize:
        require(vec_normalize.exists(), f"vec_normalize not found: {vec_normalize}")

    from isaaclab.app import AppLauncher

    print("[diag] launching Isaac", flush=True)
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
        print(f"[diag] stage={args.trajectory_stage}", flush=True)
        print(f"[diag] manifest={manifest}", flush=True)

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
            if reward_overrides:
                print(
                    "[diag] reward overrides "
                    + ", ".join(f"{name}={float(value):g}" for name, value in reward_overrides.items()),
                    flush=True,
                )
        if args.random_start_waypoint:
            print(
                "[diag] random start waypoint "
                f"{args.start_waypoint_min_fraction:.2f}-{args.start_waypoint_max_fraction:.2f}, "
                f"reset_base_to_trajectory_start={args.reset_base_to_trajectory_start}, "
                f"anchor_blend={args.reset_anchor_target_blend:.2f}",
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
        )

        base_env = MobileMMTrackEEEnv(cfg=env_cfg)

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
                if args.freeze_base_actions or abs(args.base_action_scale - 1.0) >= 1e-9:
                    actions = actions.clone()
                    if not self._base_adapter_logged:
                        mode = "freezing" if args.freeze_base_actions else f"scaling by {args.base_action_scale:.3f}"
                        print(f"[diag action-adapter] base rows [6,7,8]: {mode}", flush=True)
                        self._base_adapter_logged = True
                    if args.freeze_base_actions:
                        actions[..., 6:9] = 0.0
                    else:
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
        snapshots = trajectory_snapshots(raw_env.trajectory_manager, args.num_envs)
        arm_joint_ids = raw_env._get_joint_ids(
            ["joint6_arm_yaw", "joint5_arm_pitch", "joint4_elbow_pitch", "joint3_gimbal_yaw", "joint2_gimbal_roll", "joint1_gimbal_pitch"],
            "_diag_arm_joint_ids",
        )

        pos_errors = np.zeros((args.steps, args.num_envs), dtype=np.float64)
        ori_errors = np.zeros((args.steps, args.num_envs), dtype=np.float64)
        arm_lag = np.zeros((args.steps, args.num_envs), dtype=np.float64)
        rewards = np.zeros((args.steps, args.num_envs), dtype=np.float64)
        action_abs = np.zeros((args.steps, args.num_envs, 9), dtype=np.float64)
        base_xy = np.zeros((args.steps, args.num_envs, 2), dtype=np.float64)
        target_xy = np.zeros((args.steps, args.num_envs, 2), dtype=np.float64)
        ee_xyz_error = np.zeros((args.steps, args.num_envs, 3), dtype=np.float64)
        dones_count = 0

        for step in range(args.steps):
            action, _ = model.predict(obs, deterministic=True)
            action_abs[step] = np.abs(action[:, :9])
            obs, reward, done, _ = env.step(action)
            rewards[step] = np.asarray(reward, dtype=np.float64)
            dones_count += int(np.count_nonzero(done))

            target_pos, target_quat = raw_env.trajectory_manager.get_target_pose()
            ee_pos = raw_env.robot.data.body_pos_w[:, raw_env._ee_body_idx, :]
            ee_quat = raw_env.robot.data.body_quat_w[:, raw_env._ee_body_idx, :]
            err_vec = target_pos - ee_pos
            ee_xyz_error[step] = tensor_np(err_vec)
            pos_errors[step] = tensor_np(torch.linalg.norm(err_vec, dim=-1))
            quat_dot = torch.abs(torch.sum(target_quat * ee_quat, dim=-1)).clamp(max=1.0)
            ori_errors[step] = tensor_np(2.0 * torch.acos(quat_dot))
            arm_joint_pos = raw_env.robot.data.joint_pos[:, arm_joint_ids]
            arm_target = getattr(raw_env, "filtered_arm_targets", arm_joint_pos)
            arm_lag[step] = tensor_np(torch.linalg.norm(arm_target - arm_joint_pos, dim=-1))
            base_xy[step] = tensor_np(raw_env.robot.data.root_pos_w[:, :2])
            target_xy[step] = tensor_np(target_pos[:, :2])
            if step % 20 == 0:
                print(f"[diag] step={step}/{args.steps}", flush=True)

        source_groups: dict[str, list[int]] = defaultdict(list)
        for item in snapshots:
            source_groups[str(item["trajectory_source"])].append(int(item["env_id"]))

        by_source = {}
        for source, env_ids in sorted(source_groups.items()):
            env_idx = np.asarray(env_ids, dtype=np.int64)
            by_source[source] = {
                "env_count": int(env_idx.size),
                "ee_pos_error_m": summarize(pos_errors[:, env_idx]),
                "ee_ori_error_deg": summarize(np.degrees(ori_errors[:, env_idx])),
                "arm_target_lag_rad": summarize(arm_lag[:, env_idx]),
                "reward": summarize(rewards[:, env_idx]),
                "base_action_abs_mean": float(np.mean(action_abs[:, env_idx, 6:9])),
                "arm_action_abs_mean": float(np.mean(action_abs[:, env_idx, 0:6])),
                "base_xy_motion_mean_m": float(np.mean(np.linalg.norm(base_xy[-1, env_idx] - base_xy[0, env_idx], axis=1))),
                "target_xy_motion_mean_m": float(np.mean(np.linalg.norm(target_xy[-1, env_idx] - target_xy[0, env_idx], axis=1))),
            }

        by_time_bin = []
        edges = np.linspace(0, args.steps, args.time_bins + 1, dtype=int)
        for bin_id in range(args.time_bins):
            start, end = int(edges[bin_id]), int(edges[bin_id + 1])
            if end <= start:
                continue
            by_time_bin.append(
                {
                    "bin": bin_id,
                    "step_start": start,
                    "step_end_exclusive": end,
                    "ee_pos_error_m": summarize(pos_errors[start:end]),
                    "arm_target_lag_rad": summarize(arm_lag[start:end]),
                    "reward": summarize(rewards[start:end]),
                }
            )

        env_rows = []
        for env_id, item in enumerate(snapshots):
            env_pos = pos_errors[:, env_id]
            env_lag = arm_lag[:, env_id]
            env_base_motion = float(np.linalg.norm(base_xy[-1, env_id] - base_xy[0, env_id]))
            env_target_motion = float(np.linalg.norm(target_xy[-1, env_id] - target_xy[0, env_id]))
            worst_step = int(np.argmax(env_pos))
            env_rows.append(
                {
                    **item,
                    "ee_pos_error_mean_m": float(np.mean(env_pos)),
                    "ee_pos_error_p95_m": percentile(env_pos, 95),
                    "ee_pos_error_max_m": float(np.max(env_pos)),
                    "worst_step": worst_step,
                    "worst_xyz_error_m": [float(v) for v in ee_xyz_error[worst_step, env_id]],
                    "arm_target_lag_mean_rad": float(np.mean(env_lag)),
                    "arm_target_lag_p95_rad": percentile(env_lag, 95),
                    "base_xy_motion_m": env_base_motion,
                    "target_xy_motion_m": env_target_motion,
                    "base_action_abs_mean": float(np.mean(action_abs[:, env_id, 6:9])),
                    "arm_action_abs_mean": float(np.mean(action_abs[:, env_id, 0:6])),
                    "reward_mean": float(np.mean(rewards[:, env_id])),
                }
            )
        worst_envs = sorted(env_rows, key=lambda row: row["ee_pos_error_p95_m"], reverse=True)[: args.top_k]

        result = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "checkpoint": str(checkpoint),
            "trajectory_stage": args.trajectory_stage,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "samples": int(args.num_envs * args.steps),
            "freeze_base_actions": bool(args.freeze_base_actions),
            "base_action_scale": float(args.base_action_scale),
            "random_start_waypoint": bool(args.random_start_waypoint),
            "start_waypoint_min_fraction": float(args.start_waypoint_min_fraction),
            "start_waypoint_max_fraction": float(args.start_waypoint_max_fraction),
            "reset_base_to_trajectory_start": bool(args.reset_base_to_trajectory_start),
            "reset_anchor_target_blend": float(args.reset_anchor_target_blend),
            "dones_count": int(dones_count),
            "summary": {
                "ee_pos_error_m": summarize(pos_errors),
                "ee_ori_error_deg": summarize(np.degrees(ori_errors)),
                "arm_target_lag_rad": summarize(arm_lag),
                "reward": summarize(rewards),
                "xyz_error_mean_m": [float(v) for v in np.mean(ee_xyz_error.reshape(-1, 3), axis=0)],
                "base_action_abs_mean": float(np.mean(action_abs[:, :, 6:9])),
                "arm_action_abs_mean": float(np.mean(action_abs[:, :, 0:6])),
                "base_xy_motion_mean_m": float(np.mean(np.linalg.norm(base_xy[-1] - base_xy[0], axis=1))),
                "target_xy_motion_mean_m": float(np.mean(np.linalg.norm(target_xy[-1] - target_xy[0], axis=1))),
            },
            "by_source": by_source,
            "by_time_bin": by_time_bin,
            "worst_envs": worst_envs,
        }
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result["summary"], indent=2), flush=True)
        print(f"[diag] wrote {output}", flush=True)
        env.close()
    finally:
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
