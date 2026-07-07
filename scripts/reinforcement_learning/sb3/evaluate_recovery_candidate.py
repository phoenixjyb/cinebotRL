"""Evaluate a Proto2 SB3 checkpoint under a controlled recorded-trajectory gate."""

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
    parser = argparse.ArgumentParser(description="Evaluate a Proto2 recovery/feasibility candidate.")
    parser.add_argument("--checkpoint", required=True, help="PPO checkpoint/final_model.zip.")
    parser.add_argument("--vec_normalize", default=None, help="VecNormalize pkl. Defaults to checkpoint parent vec_normalize.pkl.")
    parser.add_argument("--task", default="RecomoProto2TrackEE-v0")
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--num_episodes", type=int, default=256)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trajectory_stage", default="stage1_recovery")
    parser.add_argument(
        "--trajectory_manifest",
        default=None,
        help=(
            "Optional manifest override. Relative paths are resolved from the project root. "
            "If omitted, trajectoryToLearn/{trajectory_stage}/manifest.txt is used."
        ),
    )
    parser.add_argument(
        "--trajectory_dir",
        default=None,
        help="Trajectory root passed to the loader. Defaults to the project root for stage manifests.",
    )
    parser.add_argument(
        "--trajectory_category",
        action="append",
        default=[],
        help="Restrict evaluation to trajectories whose parent directory matches this category. Repeatable.",
    )
    parser.add_argument(
        "--trajectory_file_contains",
        action="append",
        default=[],
        help="Restrict evaluation to trajectory paths containing this substring. Repeatable.",
    )
    parser.add_argument(
        "--max_trajectories",
        type=int,
        default=None,
        help="Limit the resolved manifest to the first N trajectories after filters.",
    )
    parser.add_argument(
        "--min_trajectory_duration",
        type=float,
        default=5.0,
        help="Reject recorded trajectories shorter than this many seconds.",
    )
    parser.add_argument(
        "--random_start_waypoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Randomize the start waypoint during evaluation.",
    )
    parser.add_argument("--start_waypoint_min_fraction", type=float, default=0.25)
    parser.add_argument("--start_waypoint_max_fraction", type=float, default=0.70)
    parser.add_argument(
        "--reset_base_to_trajectory_start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reset the base near the trajectory start anchor before playback.",
    )
    parser.add_argument("--reset_anchor_target_blend", type=float, default=0.35)
    parser.add_argument("--enable_obstacles", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable_base_assist", action="store_true")
    parser.add_argument("--base_assist_blend", type=float, default=0.70)
    parser.add_argument("--base_assist_activation_distance", type=float, default=0.45)
    parser.add_argument("--base_assist_full_speed_distance", type=float, default=0.90)
    parser.add_argument("--base_assist_max_action", type=float, default=1.0)
    parser.add_argument(
        "--freeze_base_actions",
        action="store_true",
        help=(
            "Zero base action rows [6,7,8] before env dynamics while preserving "
            "the checkpoint's 9D policy output."
        ),
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


def _read_manifest_entries(manifest_path: Path, trajectory_dir: Path) -> list[Path]:
    entries: list[Path] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            path = Path(line)
            if not path.is_absolute():
                path = trajectory_dir / path
            entries.append(path)
    return entries


def _resolve_trajectory_manifest(args: argparse.Namespace, output_dir: Path) -> tuple[Path, Path, list[str]]:
    """Resolve and optionally materialize a filtered manifest for narrow gates."""
    trajectory_dir = Path(args.trajectory_dir) if args.trajectory_dir else PROJECT_ROOT
    if not trajectory_dir.is_absolute():
        trajectory_dir = PROJECT_ROOT / trajectory_dir

    if args.trajectory_manifest:
        source_manifest = Path(args.trajectory_manifest)
        if not source_manifest.is_absolute():
            source_manifest = PROJECT_ROOT / source_manifest
    else:
        source_manifest = PROJECT_ROOT / "trajectoryToLearn" / args.trajectory_stage / "manifest.txt"

    if not source_manifest.exists():
        raise FileNotFoundError(f"trajectory manifest not found: {source_manifest}")

    entries = _read_manifest_entries(source_manifest, trajectory_dir)
    categories = set(args.trajectory_category or [])
    contains = list(args.trajectory_file_contains or [])
    if categories:
        entries = [p for p in entries if p.parent.name in categories]
    if contains:
        entries = [p for p in entries if all(fragment in str(p) for fragment in contains)]
    if args.max_trajectories is not None:
        if args.max_trajectories <= 0:
            raise ValueError("--max_trajectories must be positive")
        entries = entries[: args.max_trajectories]
    if not entries:
        raise ValueError("trajectory filters selected no files")

    if categories or contains or args.max_trajectories is not None or source_manifest != (
        PROJECT_ROOT / "trajectoryToLearn" / args.trajectory_stage / "manifest.txt"
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        resolved_manifest = output_dir / "resolved_feasibility_manifest.txt"
        resolved_manifest.write_text(
            "\n".join(str(path) for path in entries) + "\n",
            encoding="utf-8",
        )
    else:
        resolved_manifest = source_manifest

    return trajectory_dir, resolved_manifest, [str(path) for path in entries]


def _load_stage_reset_config(args: argparse.Namespace) -> dict[str, object]:
    """Load optional stage-local reset offsets without changing legacy defaults."""
    stage_dir = PROJECT_ROOT / "trajectoryToLearn" / args.trajectory_stage
    reset_config_file = stage_dir / "reset_config.json"
    if not reset_config_file.exists():
        return {}
    with reset_config_file.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    out: dict[str, object] = {}
    if "reset_base_x_offset" in raw:
        out["reset_base_x_offset"] = float(raw["reset_base_x_offset"])
    if "reset_base_y_offset" in raw:
        out["reset_base_y_offset"] = float(raw["reset_base_y_offset"])
    raw_reward_overrides = raw.get("reward_overrides", {})
    if isinstance(raw_reward_overrides, dict):
        out["reward_overrides"] = {
            str(name): float(value)
            for name, value in raw_reward_overrides.items()
        }
    return out


def get_trajectory_snapshots(raw_env, num_envs: int) -> list[dict[str, object]]:
    """Snapshot per-env trajectory metadata before it can be replaced by a reset."""
    manager = getattr(raw_env, "trajectory_manager", None)
    if manager is None:
        return [{} for _ in range(num_envs)]

    metadata = getattr(manager, "current_trajectory_metadata", None) or []
    waypoint_idx = tensor_np(getattr(manager, "current_waypoint_idx", None))
    lengths = tensor_np(getattr(manager, "recorded_lengths", None))
    snapshots: list[dict[str, object]] = []
    for env_id in range(num_envs):
        item = metadata[env_id] if env_id < len(metadata) and isinstance(metadata[env_id], dict) else {}
        current_idx = int(waypoint_idx[env_id]) if waypoint_idx is not None and env_id < len(waypoint_idx) else None
        length = int(lengths[env_id]) if lengths is not None and env_id < len(lengths) else item.get("length")
        length_int = int(length) if length is not None else None
        fraction = None
        if current_idx is not None and length_int is not None and length_int > 1:
            fraction = float(current_idx) / float(length_int - 1)
        snapshots.append(
            {
                "trajectory_file": str(item.get("file", "unknown")),
                "trajectory_category": str(item.get("category", "unknown")),
                "trajectory_length": length_int,
                "waypoint_idx": current_idx,
                "waypoint_fraction": fraction,
            }
        )
    return snapshots


def empty_episode_accumulators(num_envs: int) -> dict[str, object]:
    return {
        "base_sum": np.zeros(num_envs, dtype=np.float64),
        "base_count": np.zeros(num_envs, dtype=np.int32),
        "base_max": np.full(num_envs, np.nan, dtype=np.float64),
        "unreachable_count": np.zeros(num_envs, dtype=np.int32),
        "workspace_soft_count": np.zeros(num_envs, dtype=np.int32),
        "workspace_hard_count": np.zeros(num_envs, dtype=np.int32),
        "workspace_count": np.zeros(num_envs, dtype=np.int32),
        "workspace_max": np.full(num_envs, np.nan, dtype=np.float64),
        "obstacle_unsafe_count": np.zeros(num_envs, dtype=np.int32),
        "obstacle_collision_count": np.zeros(num_envs, dtype=np.int32),
        "obstacle_count": np.zeros(num_envs, dtype=np.int32),
        "obstacle_clearance_min": np.full(num_envs, np.nan, dtype=np.float64),
        "pos_errors": [[] for _ in range(num_envs)],
        "ori_errors": [[] for _ in range(num_envs)],
    }


def reset_episode_accumulator(acc: dict[str, object], env_id: int) -> None:
    for key in [
        "base_sum",
        "base_count",
        "base_max",
        "unreachable_count",
        "workspace_soft_count",
        "workspace_hard_count",
        "workspace_count",
        "workspace_max",
        "obstacle_unsafe_count",
        "obstacle_collision_count",
        "obstacle_count",
        "obstacle_clearance_min",
    ]:
        arr = acc[key]
        if key.endswith("_max") or key.endswith("_min"):
            arr[env_id] = np.nan
        else:
            arr[env_id] = 0
    acc["pos_errors"][env_id].clear()
    acc["ori_errors"][env_id].clear()


def append_episode_detail(
    details: list[dict[str, object]],
    acc: dict[str, object],
    env_id: int,
    trajectory_snapshot: dict[str, object],
    end_snapshot: dict[str, object],
    reward: float,
    length: int,
    done_step: int,
) -> None:
    base_count = int(acc["base_count"][env_id])
    workspace_count = int(acc["workspace_count"][env_id])
    obstacle_count = int(acc["obstacle_count"][env_id])
    pos_errors = np.asarray(acc["pos_errors"][env_id], dtype=np.float64)
    ori_errors = np.asarray(acc["ori_errors"][env_id], dtype=np.float64)
    detail = {
        "episode_index": len(details),
        "env_id": int(env_id),
        "done_step": int(done_step),
        "episode_reward": float(reward),
        "episode_length": int(length),
        **trajectory_snapshot,
        "end_waypoint_idx": end_snapshot.get("waypoint_idx"),
        "end_waypoint_fraction": end_snapshot.get("waypoint_fraction"),
        "base_target_dist_mean": float(acc["base_sum"][env_id] / base_count) if base_count else None,
        "base_target_dist_max": float(acc["base_max"][env_id]) if base_count else None,
        "unreachable_zone_pct": float(acc["unreachable_count"][env_id] / base_count * 100.0) if base_count else None,
        "workspace_soft_exceed_pct": (
            float(acc["workspace_soft_count"][env_id] / workspace_count * 100.0) if workspace_count else None
        ),
        "workspace_hard_exceed_pct": (
            float(acc["workspace_hard_count"][env_id] / workspace_count * 100.0) if workspace_count else None
        ),
        "workspace_distance_max": float(acc["workspace_max"][env_id]) if workspace_count else None,
        "obstacle_unsafe_pct": (
            float(acc["obstacle_unsafe_count"][env_id] / obstacle_count * 100.0) if obstacle_count else None
        ),
        "obstacle_collision_pct": (
            float(acc["obstacle_collision_count"][env_id] / obstacle_count * 100.0) if obstacle_count else None
        ),
        "obstacle_clearance_min": float(acc["obstacle_clearance_min"][env_id]) if obstacle_count else None,
        "ee_pos_error_mean_m": float(np.mean(pos_errors)) if pos_errors.size else None,
        "ee_pos_error_p95_m": float(np.percentile(pos_errors, 95)) if pos_errors.size else None,
        "ee_ori_error_mean_deg": float(np.degrees(np.mean(ori_errors))) if ori_errors.size else None,
    }
    details.append(detail)


def _metric_stats(values: list[float]) -> dict[str, float | int | None]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "mean": None, "p95": None, "max": None}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def summarize_episode_details(details: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    """Aggregate final JSON by trajectory category and file for tiny-set gates."""
    groups: dict[str, dict[str, list[float]]] = {}
    for detail in details:
        for prefix, group_key in [
            ("category", str(detail.get("trajectory_category", "unknown"))),
            ("file", str(detail.get("trajectory_file", "unknown"))),
        ]:
            key = f"{prefix}:{group_key}"
            bucket = groups.setdefault(
                key,
                {
                    "ee_pos_error_mean_m": [],
                    "ee_pos_error_p95_m": [],
                    "ee_ori_error_mean_deg": [],
                    "unreachable_zone_pct": [],
                    "workspace_hard_exceed_pct": [],
                    "obstacle_unsafe_pct": [],
                    "obstacle_collision_pct": [],
                },
            )
            for metric in bucket:
                value = detail.get(metric)
                if isinstance(value, (int, float)):
                    bucket[metric].append(float(value))

    summary: dict[str, dict[str, object]] = {}
    for key, metric_values in groups.items():
        summary[key] = {metric: _metric_stats(values) for metric, values in metric_values.items()}
    return summary


def main() -> int:
    args = parse_args()
    if args.base_action_scale < 0.0 or args.base_action_scale > 1.0:
        raise ValueError("--base_action_scale must be in [0, 1]")
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

        output_dir = Path(args.output_dir)
        trajectory_dir, manifest, selected_trajectories = _resolve_trajectory_manifest(args, output_dir)
        reset_config = _load_stage_reset_config(args)
        print(f"trajectory_dir: {trajectory_dir}")
        print(f"trajectory_manifest: {manifest}")
        if reset_config:
            print(
                "stage reset offset: "
                f"x={reset_config.get('reset_base_x_offset', 0.4415):.4f}, "
                f"y={reset_config.get('reset_base_y_offset', 0.2405):.4f}"
            )
        print(f"selected trajectories: {len(selected_trajectories)}")
        if len(selected_trajectories) <= 10:
            for item in selected_trajectories:
                print(f"  - {item}")

        checkpoint_obs_space = PPO.load(
            str(checkpoint), device="cpu", print_system_info=False
        ).observation_space
        expected_obs_dim = int(np.prod(checkpoint_obs_space.shape))

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
        reward_overrides = reset_config.get("reward_overrides", {})
        if isinstance(reward_overrides, dict):
            for name, value in reward_overrides.items():
                setattr(env_cfg.task_config.rewards, name, float(value))
            if reward_overrides:
                print(
                    "stage reward overrides: "
                    + ", ".join(f"{name}={float(value):g}" for name, value in reward_overrides.items())
                )

        env_cfg.task_config.trajectory = TrajectoryConfig(
            type="multi_recorded",
            trajectory_dir=str(trajectory_dir),
            trajectory_manifest_file=str(manifest),
            max_trajectories=None,
            min_duration_seconds=args.min_trajectory_duration,
            randomize_start_waypoint=bool(args.random_start_waypoint),
            start_waypoint_min_fraction=args.start_waypoint_min_fraction,
            start_waypoint_max_fraction=args.start_waypoint_max_fraction,
            reset_base_to_trajectory_start=bool(args.reset_base_to_trajectory_start),
            reset_anchor_target_blend=args.reset_anchor_target_blend,
            reset_base_x_offset=reset_config.get("reset_base_x_offset", 0.4415),
            reset_base_y_offset=reset_config.get("reset_base_y_offset", 0.2405),
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
            def __init__(
                self,
                venv,
                expected_dim: int | None = None,
                freeze_base_actions: bool = False,
                base_action_scale: float = 1.0,
            ):
                super().__init__(venv)
                self.expected_dim = expected_dim
                self._obs_adapter_logged = False
                self.freeze_base_actions = bool(freeze_base_actions)
                self.base_action_scale = float(base_action_scale)
                self._base_adapter_logged = False
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

            def _adapt_obs_dim(self, obs: np.ndarray) -> np.ndarray:
                if self.expected_dim is None:
                    return obs
                if obs.ndim == 2 and obs.shape[1] == self.expected_dim:
                    return obs
                if obs.ndim == 1 and obs.shape[0] == self.expected_dim:
                    return obs
                if obs.ndim == 2 and obs.shape[1] == self.expected_dim - 1:
                    if not self._obs_adapter_logged:
                        print(
                            "[obs-adapter] Appending trajectory progress column "
                            f"for checkpoint compatibility: {obs.shape[1]} -> {self.expected_dim}"
                        )
                        self._obs_adapter_logged = True
                    return np.concatenate([obs, self._trajectory_progress_column(obs)], axis=1)
                if obs.ndim == 1 and obs.shape[0] == self.expected_dim - 1:
                    if not self._obs_adapter_logged:
                        print(
                            "[obs-adapter] Appending zero progress column "
                            f"for checkpoint compatibility: {obs.shape[0]} -> {self.expected_dim}"
                        )
                        self._obs_adapter_logged = True
                    return np.concatenate([obs, np.zeros((1,), dtype=np.float32)], axis=0)
                # Some recorded-trajectory eval configs append progress while older
                # PPO checkpoints were trained on the raw 84D policy observation.
                if obs.ndim == 2 and obs.shape[1] == self.expected_dim + 1:
                    if not self._obs_adapter_logged:
                        print(
                            "[obs-adapter] Dropping final observation column "
                            f"for checkpoint compatibility: {obs.shape[1]} -> {self.expected_dim}"
                        )
                        self._obs_adapter_logged = True
                    return obs[:, : self.expected_dim]
                if obs.ndim == 1 and obs.shape[0] == self.expected_dim + 1:
                    if not self._obs_adapter_logged:
                        print(
                            "[obs-adapter] Dropping final observation column "
                            f"for checkpoint compatibility: {obs.shape[0]} -> {self.expected_dim}"
                        )
                        self._obs_adapter_logged = True
                    return obs[: self.expected_dim]
                return obs

            def _trajectory_progress_column(self, obs: np.ndarray) -> np.ndarray:
                raw_env = self.venv.unwrapped if hasattr(self.venv, "unwrapped") else self.venv
                manager = getattr(raw_env, "trajectory_manager", None)
                num_envs = obs.shape[0] if obs.ndim == 2 else 1
                if manager is None:
                    return np.zeros((num_envs, 1), dtype=np.float32)

                current_idx = getattr(manager, "current_waypoint_idx", None)
                lengths = getattr(manager, "recorded_lengths", None)
                if current_idx is None or lengths is None:
                    return np.zeros((num_envs, 1), dtype=np.float32)

                if hasattr(current_idx, "detach"):
                    current_idx = current_idx.detach()
                if hasattr(lengths, "detach"):
                    lengths = lengths.detach()
                if hasattr(current_idx, "cpu"):
                    current_idx = current_idx.cpu()
                if hasattr(lengths, "cpu"):
                    lengths = lengths.cpu()

                current_np = np.asarray(current_idx, dtype=np.float32).reshape(-1)
                lengths_np = np.asarray(lengths, dtype=np.float32).reshape(-1)
                if current_np.shape[0] < num_envs or lengths_np.shape[0] < num_envs:
                    return np.zeros((num_envs, 1), dtype=np.float32)
                denom = np.maximum(lengths_np[:num_envs] - 1.0, 1.0)
                progress = np.clip(current_np[:num_envs] / denom, 0.0, 1.0)
                return progress.reshape(num_envs, 1).astype(np.float32, copy=False)

            def _convert_obs(self, obs):
                if isinstance(obs, tuple):
                    obs = obs[0]
                if isinstance(obs, dict):
                    obs = obs.get("policy", list(obs.values())[0])
                if hasattr(obs, "detach"):
                    obs = obs.detach()
                if hasattr(obs, "cpu"):
                    obs = obs.cpu()
                return self._adapt_obs_dim(np.asarray(obs, dtype=np.float32))

            def reset(self):
                return self._convert_obs(self.venv.reset())

            def _adapt_actions(self, actions):
                if actions.shape[-1] < 9:
                    raise ValueError(
                        f"base action adapter requires at least 9 action dims, got {actions.shape[-1]}"
                    )
                if not self.freeze_base_actions and abs(self.base_action_scale - 1.0) < 1e-9:
                    return actions
                actions = actions.clone()
                if not self._base_adapter_logged:
                    if self.freeze_base_actions:
                        print("[eval action-adapter] Freezing base action rows [6,7,8]")
                    else:
                        print(
                            "[eval action-adapter] Scaling base action rows [6,7,8] "
                            f"by {self.base_action_scale:.3f}"
                        )
                    self._base_adapter_logged = True
                if self.freeze_base_actions:
                    actions[..., 6:9] = 0.0
                else:
                    actions[..., 6:9] *= self.base_action_scale
                return actions

            def step_async(self, actions):
                if isinstance(actions, np.ndarray):
                    device = self.venv.unwrapped.device if hasattr(self.venv.unwrapped, "device") else "cuda:0"
                    actions = torch.from_numpy(actions).float().to(device)
                actions = self._adapt_actions(actions)
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

        env = IsaacLabToSB3VecEnvWrapper(
            base_env,
            expected_obs_dim,
            freeze_base_actions=args.freeze_base_actions,
            base_action_scale=args.base_action_scale,
        )
        env = VecNormalize.load(str(vec_path), env)
        env.training = False
        env.norm_reward = False
        model = PPO.load(str(checkpoint), env=env, device="cuda" if torch.cuda.is_available() else "cpu")

        raw_env = unwrap_isaac_env(env)
        obs = env.reset()
        collector = ScalarCollector()
        episode_details: list[dict[str, object]] = []
        episode_acc = empty_episode_accumulators(args.num_envs)
        episode_start_snapshots = get_trajectory_snapshots(raw_env, args.num_envs)
        rewards_by_env = np.zeros(args.num_envs, dtype=np.float64)
        lengths_by_env = np.zeros(args.num_envs, dtype=np.int32)
        episode_rewards: list[float] = []
        episode_lengths: list[int] = []
        step = 0
        max_steps = max(10_000, int(args.num_episodes / max(args.num_envs, 1) * 800) + 1_000)

        while len(episode_rewards) < args.num_episodes and step < max_steps:
            pre_step_snapshots = get_trajectory_snapshots(raw_env, args.num_envs)
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
                finite_mask = np.isfinite(base_dist)
                if np.any(finite_mask):
                    episode_acc["base_sum"][finite_mask] += base_dist[finite_mask]
                    episode_acc["base_count"][finite_mask] += 1
                    current_max = episode_acc["base_max"][finite_mask]
                    episode_acc["base_max"][finite_mask] = np.where(
                        np.isnan(current_max),
                        base_dist[finite_mask],
                        np.maximum(current_max, base_dist[finite_mask]),
                    )
                    episode_acc["unreachable_count"][finite_mask] += (base_dist[finite_mask] > 0.70).astype(np.int32)
            if workspace is not None:
                collector.add("workspace_soft_exceed_pct", float(np.mean(workspace > 0.20) * 100.0))
                collector.add("workspace_hard_exceed_pct", float(np.mean(workspace > 0.70) * 100.0))
                collector.add("workspace_distance_max", float(np.max(workspace)))
                finite_mask = np.isfinite(workspace)
                if np.any(finite_mask):
                    episode_acc["workspace_count"][finite_mask] += 1
                    episode_acc["workspace_soft_count"][finite_mask] += (workspace[finite_mask] > 0.20).astype(np.int32)
                    episode_acc["workspace_hard_count"][finite_mask] += (workspace[finite_mask] > 0.70).astype(np.int32)
                    current_max = episode_acc["workspace_max"][finite_mask]
                    episode_acc["workspace_max"][finite_mask] = np.where(
                        np.isnan(current_max),
                        workspace[finite_mask],
                        np.maximum(current_max, workspace[finite_mask]),
                    )
            if pos_error is not None:
                pos_norm = np.linalg.norm(pos_error, axis=1)
                pos_norm = pos_norm[np.isfinite(pos_norm)]
                if pos_norm.size:
                    collector.add("ee_pos_error_mean_m", float(np.mean(pos_norm)))
                    collector.add("ee_pos_error_p95_m", float(np.percentile(pos_norm, 95)))
                all_pos_norm = np.linalg.norm(pos_error, axis=1)
                for env_id, value in enumerate(all_pos_norm):
                    if np.isfinite(value):
                        episode_acc["pos_errors"][env_id].append(float(value))
            if ori_error is not None:
                if ori_error.ndim > 1:
                    ori_values = np.linalg.norm(ori_error, axis=1)
                else:
                    ori_values = ori_error
                finite_ori = ori_values[np.isfinite(ori_values)]
                if finite_ori.size:
                    collector.add("ee_ori_error_mean_deg", float(np.degrees(np.mean(finite_ori))))
                for env_id, value in enumerate(ori_values):
                    if np.isfinite(value):
                        episode_acc["ori_errors"][env_id].append(float(value))
            if assist_coeff is not None:
                collector.add("base_assist_active_pct", float(np.mean(assist_coeff > 0.0) * 100.0))
                collector.add("base_assist_coeff_mean", float(np.mean(assist_coeff)))
            if getattr(raw_env, "obstacles_enabled", False):
                clearance = tensor_np(raw_env._get_obstacle_clearance(raw_env.robot.data.root_pos_w))
                collector.add("obstacle_unsafe_pct", float(np.mean(clearance < 0.20) * 100.0))
                collector.add("obstacle_collision_pct", float(np.mean(clearance < 0.0) * 100.0))
                collector.add("obstacle_clearance_min", float(np.min(clearance)))
                finite_mask = np.isfinite(clearance)
                if np.any(finite_mask):
                    episode_acc["obstacle_count"][finite_mask] += 1
                    episode_acc["obstacle_unsafe_count"][finite_mask] += (clearance[finite_mask] < 0.20).astype(np.int32)
                    episode_acc["obstacle_collision_count"][finite_mask] += (clearance[finite_mask] < 0.0).astype(np.int32)
                    current_min = episode_acc["obstacle_clearance_min"][finite_mask]
                    episode_acc["obstacle_clearance_min"][finite_mask] = np.where(
                        np.isnan(current_min),
                        clearance[finite_mask],
                        np.minimum(current_min, clearance[finite_mask]),
                    )

            for idx, done in enumerate(dones):
                if done:
                    append_episode_detail(
                        episode_details,
                        episode_acc,
                        idx,
                        episode_start_snapshots[idx],
                        pre_step_snapshots[idx],
                        float(rewards_by_env[idx]),
                        int(lengths_by_env[idx]),
                        step,
                    )
                    episode_rewards.append(float(rewards_by_env[idx]))
                    episode_lengths.append(int(lengths_by_env[idx]))
                    rewards_by_env[idx] = 0.0
                    lengths_by_env[idx] = 0
                    reset_episode_accumulator(episode_acc, idx)
                    if len(episode_rewards) >= args.num_episodes:
                        break
            if np.any(dones):
                latest_snapshots = get_trajectory_snapshots(raw_env, args.num_envs)
                for idx, done in enumerate(dones):
                    if done:
                        episode_start_snapshots[idx] = latest_snapshots[idx]

            step += 1

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "timestamp": timestamp,
            "checkpoint": str(checkpoint),
            "vec_normalize": str(vec_path),
            "mode": "assisted" if args.enable_base_assist else "raw-policy",
            "trajectory_source": {
                "trajectory_stage": args.trajectory_stage,
                "trajectory_dir": str(trajectory_dir),
                "trajectory_manifest": str(manifest),
                "selected_count": len(selected_trajectories),
                "selected_trajectories": selected_trajectories,
                "min_trajectory_duration": args.min_trajectory_duration,
                "random_start_waypoint": bool(args.random_start_waypoint),
                "reset_base_to_trajectory_start": bool(args.reset_base_to_trajectory_start),
            },
            "num_envs": args.num_envs,
            "freeze_base_actions": bool(args.freeze_base_actions),
            "base_action_scale": float(args.base_action_scale),
            "episodes_completed": len(episode_rewards),
            "steps": step,
            "episode_reward_mean": float(np.mean(episode_rewards)) if episode_rewards else None,
            "episode_length_mean": float(np.mean(episode_lengths)) if episode_lengths else None,
            "metrics": collector.summary(),
            "episode_details": episode_details[: args.num_episodes],
            "episode_group_summary": summarize_episode_details(episode_details[: args.num_episodes]),
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
