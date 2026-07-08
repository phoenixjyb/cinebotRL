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
    parser.add_argument("--vec_normalize", required=True)
    parser.add_argument(
        "--disable_vec_normalize",
        action="store_true",
        help="Do not load VecNormalize stats. Use this for raw-observation BC policies.",
    )
    parser.add_argument("--trajectory_stage", default="stage0_policy_envelope_fk")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--max_trajectories", type=int, default=None)
    parser.add_argument("--min_trajectory_duration", type=float, default=5.0)
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


def main() -> int:
    args = parse_args()
    if args.base_action_scale < 0.0 or args.base_action_scale > 1.0:
        raise ValueError("--base_action_scale must be in [0, 1]")
    checkpoint = Path(args.checkpoint)
    recovery_checkpoint = Path(args.recovery_checkpoint) if args.recovery_checkpoint else None
    vec_normalize = Path(args.vec_normalize)
    require(checkpoint.exists(), f"checkpoint not found: {checkpoint}")
    if recovery_checkpoint is not None:
        require(recovery_checkpoint.exists(), f"recovery checkpoint not found: {recovery_checkpoint}")
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
        env_cfg.task_config.obstacles.enable_obstacles = False
        env_cfg.task_config.base_assist.enable = False
        env_cfg.task_config.randomize_initial_joint_positions = bool(args.enable_initial_joint_randomization)
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
        )

        print("[gate] creating env", flush=True)
        base_env = MobileMMTrackEEEnv(cfg=env_cfg)
        print("[gate] env created", flush=True)

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
        raw_env = base_env.unwrapped if hasattr(base_env, "unwrapped") else base_env

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
        route_counts: list[int] = []
        latched_route_mask = np.zeros((args.num_envs,), dtype=bool)
        route_waypoint_fractions: list[np.ndarray] = []
        route_pos_errors: list[np.ndarray] = []
        route_base_target_distances: list[np.ndarray] = []
        dones_count = 0
        target_pos0, target_quat0 = raw_env.trajectory_manager.get_target_pose()
        ee_pos0 = raw_env.robot.data.body_pos_w[:, raw_env._ee_body_idx, :]
        base_pos0 = raw_env.robot.data.root_pos_w
        initial_pos_error = tensor_np(torch.linalg.norm(target_pos0 - ee_pos0, dim=-1))
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
            if open_loop_actions is None:
                action, _ = model.predict(obs, deterministic=True)
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
            obs, reward, done, _ = env.step(action)
            rewards.append(np.asarray(reward, dtype=np.float64))
            dones_count += int(np.count_nonzero(done))

            target_pos, target_quat = raw_env.trajectory_manager.get_target_pose()
            ee_pos = raw_env.robot.data.body_pos_w[:, raw_env._ee_body_idx, :]
            ee_quat = raw_env.robot.data.body_quat_w[:, raw_env._ee_body_idx, :]
            pos_errors.append(tensor_np(torch.linalg.norm(target_pos - ee_pos, dim=-1)))
            quat_dot = torch.abs(torch.sum(target_quat * ee_quat, dim=-1)).clamp(max=1.0)
            ori_errors.append(tensor_np(2.0 * torch.acos(quat_dot)))
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
        target_pos_end, _ = raw_env.trajectory_manager.get_target_pose()
        ee_pos_end = raw_env.robot.data.body_pos_w[:, raw_env._ee_body_idx, :]
        base_pos_end = raw_env.robot.data.root_pos_w
        final_pos_error = tensor_np(torch.linalg.norm(target_pos_end - ee_pos_end, dim=-1))
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
            "recovery_route_min_waypoint_fraction": float(args.recovery_route_min_waypoint_fraction),
            "recovery_route_min_pos_error": float(args.recovery_route_min_pos_error),
            "recovery_route_min_base_target_distance": float(args.recovery_route_min_base_target_distance),
            "recovery_route_latch_once": bool(args.recovery_route_latch_once),
            "vec_normalize": str(vec_normalize),
            "disable_vec_normalize": bool(args.disable_vec_normalize),
            "trajectory_stage": args.trajectory_stage,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "samples": int(pos.size),
            "freeze_base_actions": bool(args.freeze_base_actions),
            "base_action_scale": float(args.base_action_scale),
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
        }
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
        env.close()
    finally:
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
