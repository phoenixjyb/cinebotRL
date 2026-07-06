"""Record a deterministic Stage1 recovery rollout as data plus a top-down MP4.

This is intentionally not an Isaac Sim viewport recorder. It runs the same policy
and environment path as recovery evaluation, captures base/EE/target/obstacle
state from the simulator, and renders a compact top-down animation. That makes it
usable over SSH/headless runs and keeps visual comparisons reproducible.
"""

from __future__ import annotations

import argparse
import csv
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
    parser = argparse.ArgumentParser(description="Record a Proto2 recovery rollout animation.")
    parser.add_argument("--checkpoint", required=True, help="PPO checkpoint/final_model.zip or BC policy zip.")
    parser.add_argument("--vec_normalize", default=None, help="VecNormalize pkl. Defaults to checkpoint parent vec_normalize.pkl.")
    parser.add_argument("--task", default="RecomoProto2TrackEE-v0")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--env_id", type=int, default=0)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--frame_stride", type=int, default=2)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trajectory_stage", default="stage1_recovery")
    parser.add_argument("--trajectory_manifest", default=None)
    parser.add_argument("--trajectory_dir", default=None)
    parser.add_argument("--trajectory_category", action="append", default=[])
    parser.add_argument("--trajectory_file_contains", action="append", default=[])
    parser.add_argument("--max_trajectories", type=int, default=1)
    parser.add_argument("--min_trajectory_duration", type=float, default=5.0)
    parser.add_argument("--random_start_waypoint", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--start_waypoint_min_fraction", type=float, default=0.25)
    parser.add_argument("--start_waypoint_max_fraction", type=float, default=0.70)
    parser.add_argument("--reset_base_to_trajectory_start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reset_anchor_target_blend", type=float, default=0.35)
    parser.add_argument("--enable_obstacles", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--obstacle_radius", type=float, default=0.20)
    parser.add_argument("--obstacle_height", type=float, default=0.50)
    parser.add_argument("--output_dir", default="evaluation_results/videos/stage1_recovery_rollout")
    parser.add_argument("--name", default=None, help="Optional output basename.")
    return parser.parse_args()


def _tensor_np(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _convert_obs(obs):
    if isinstance(obs, tuple):
        obs = obs[0]
    if isinstance(obs, dict):
        obs = obs.get("policy", list(obs.values())[0])
    return np.asarray(_tensor_np(obs), dtype=np.float32)


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

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_manifest = output_dir / "resolved_record_manifest.txt"
    resolved_manifest.write_text("\n".join(str(path) for path in entries) + "\n", encoding="utf-8")
    return trajectory_dir, resolved_manifest, [str(path) for path in entries]


def _write_csv(path: Path, frames: list[dict[str, float]]) -> None:
    if not frames:
        return
    keys = list(frames[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(frames)


def _render_topdown(video_path: Path, png_path: Path, frames: list[dict[str, float]], fps: int) -> str:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FFMpegWriter
        from matplotlib.patches import Circle
    except Exception as exc:
        return f"matplotlib unavailable: {exc}"

    if not frames:
        return "no frames captured"

    xs = np.array([f["base_x"] for f in frames] + [f["ee_x"] for f in frames] + [f["target_x"] for f in frames])
    ys = np.array([f["base_y"] for f in frames] + [f["ee_y"] for f in frames] + [f["target_y"] for f in frames])
    finite = np.isfinite(xs) & np.isfinite(ys)
    if not np.any(finite):
        return "no finite xy data"

    margin = 0.75
    x_min, x_max = float(np.min(xs[finite]) - margin), float(np.max(xs[finite]) + margin)
    y_min, y_max = float(np.min(ys[finite]) - margin), float(np.max(ys[finite]) + margin)
    span = max(x_max - x_min, y_max - y_min, 1.0)
    cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    x_min, x_max = cx - span / 2.0, cx + span / 2.0
    y_min, y_max = cy - span / 2.0, cy + span / 2.0

    fig, ax = plt.subplots(figsize=(7, 7), dpi=130)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.grid(True, alpha=0.25)

    target_line, = ax.plot([], [], color="#d62728", lw=2.0, label="target")
    base_line, = ax.plot([], [], color="#1f77b4", lw=2.0, label="base")
    ee_line, = ax.plot([], [], color="#2ca02c", lw=1.6, label="ee")
    target_pt, = ax.plot([], [], "o", color="#d62728", ms=7)
    base_pt, = ax.plot([], [], "s", color="#1f77b4", ms=7)
    ee_pt, = ax.plot([], [], "o", color="#2ca02c", ms=6)
    link_line, = ax.plot([], [], color="#555555", lw=1.0, alpha=0.6)
    title = ax.set_title("")
    ax.legend(loc="upper right")

    first = frames[0]
    if np.isfinite(first.get("obstacle_x", np.nan)) and np.isfinite(first.get("obstacle_y", np.nan)):
        obstacle = Circle(
            (first["obstacle_x"], first["obstacle_y"]),
            radius=max(first.get("obstacle_radius", 0.20), 0.01),
            facecolor="#ff7f0e",
            edgecolor="#8c4b00",
            alpha=0.35,
            label="obstacle",
        )
        ax.add_patch(obstacle)

    def draw(i: int):
        f = frames[i]
        hist = frames[: i + 1]
        target_line.set_data([h["target_x"] for h in hist], [h["target_y"] for h in hist])
        base_line.set_data([h["base_x"] for h in hist], [h["base_y"] for h in hist])
        ee_line.set_data([h["ee_x"] for h in hist], [h["ee_y"] for h in hist])
        target_pt.set_data([f["target_x"]], [f["target_y"]])
        base_pt.set_data([f["base_x"]], [f["base_y"]])
        ee_pt.set_data([f["ee_x"]], [f["ee_y"]])
        link_line.set_data([f["base_x"], f["ee_x"]], [f["base_y"], f["ee_y"]])
        title.set_text(
            f"step={int(f['step'])}  base-target={f['base_target_dist']:.2f}m  "
            f"ee-error={f['ee_pos_error']:.2f}m  clearance={f['obstacle_clearance']:.2f}m"
        )
        return target_line, base_line, ee_line, target_pt, base_pt, ee_pt, link_line, title

    draw(len(frames) - 1)
    fig.savefig(png_path, bbox_inches="tight")

    try:
        writer = FFMpegWriter(fps=fps, metadata={"title": "Stage1 recovery rollout"})
        with writer.saving(fig, str(video_path), dpi=130):
            for i in range(len(frames)):
                draw(i)
                writer.grab_frame()
        plt.close(fig)
        return "mp4 ok"
    except Exception as mp4_exc:
        gif_path = video_path.with_suffix(".gif")
        try:
            from matplotlib.animation import PillowWriter

            writer = PillowWriter(fps=fps)
            with writer.saving(fig, str(gif_path), dpi=130):
                for i in range(len(frames)):
                    draw(i)
                    writer.grab_frame()
            plt.close(fig)
            return f"gif ok; mp4 failed: {mp4_exc}"
        except Exception as gif_exc:
            plt.close(fig)
            return f"mp4/gif failed, png written: {mp4_exc}; {gif_exc}"


def main() -> int:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        print(f"checkpoint not found: {checkpoint}")
        return 1
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vec_path = Path(args.vec_normalize) if args.vec_normalize else checkpoint.parent / "vec_normalize.pkl"
    use_vec_normalize = vec_path.exists()

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

        trajectory_dir, manifest, selected_trajectories = _resolve_trajectory_manifest(args, output_dir)

        env_cfg = MobileMMTrackEEEnvCfg()
        env_cfg.num_envs = args.num_envs
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.task_config.obstacles.enable_obstacles = bool(args.enable_obstacles)
        env_cfg.task_config.obstacles.randomize_per_reset = True
        env_cfg.task_config.obstacles.disc_radius = float(args.obstacle_radius)
        env_cfg.task_config.obstacles.disc_height = float(args.obstacle_height)
        env_cfg.task_config.obstacles.disc_position_x_range = (-0.35, 0.35)
        env_cfg.task_config.obstacles.disc_position_y_range = (0.45, 1.00)
        env_cfg.task_config.obstacles.min_start_clearance = 0.10
        if args.enable_obstacles:
            env_cfg.scene = env_cfg._create_scene_config()
            env_cfg.scene.num_envs = args.num_envs
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
        )

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
                dummy_obs = _convert_obs(venv.reset())
                obs_shape = (dummy_obs.shape[1],) if len(dummy_obs.shape) > 1 else dummy_obs.shape
                self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)

            def reset(self):
                return _convert_obs(self.venv.reset())

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
                obs = _convert_obs(obs)
                rewards = _tensor_np(rewards).astype(np.float32)
                dones = _tensor_np(dones).astype(bool)
                if isinstance(infos, dict):
                    infos = [infos.copy() for _ in range(len(rewards))]
                elif not isinstance(infos, list):
                    infos = [{} for _ in range(len(rewards))]
                return obs, rewards, dones, infos

        env = IsaacLabToSB3VecEnvWrapper(base_env)
        if use_vec_normalize:
            env = VecNormalize.load(str(vec_path), env)
            env.training = False
            env.norm_reward = False

        model = PPO.load(str(checkpoint), env=env, device="cuda" if torch.cuda.is_available() else "cpu")
        raw_env = env
        while hasattr(raw_env, "venv"):
            raw_env = raw_env.venv
        if hasattr(raw_env, "unwrapped"):
            raw_env = raw_env.unwrapped

        obs = env.reset()
        frames: list[dict[str, float]] = []
        env_id = min(max(args.env_id, 0), args.num_envs - 1)

        for step in range(args.steps):
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, _rewards, dones, _infos = env.step(action)
            if step % max(args.frame_stride, 1) == 0:
                target_pos, _ = raw_env.trajectory_manager.get_target_pose()
                base_pos = raw_env.robot.data.root_pos_w
                ee_pos = raw_env.robot.data.body_pos_w[:, raw_env._ee_body_idx, :]
                base_np = _tensor_np(base_pos[env_id])
                ee_np = _tensor_np(ee_pos[env_id])
                target_np = _tensor_np(target_pos[env_id])
                base_target_dist = float(np.linalg.norm(target_np[:2] - base_np[:2]))
                ee_pos_error = float(np.linalg.norm(target_np[:3] - ee_np[:3]))
                obstacle_x = obstacle_y = obstacle_clearance = np.nan
                if getattr(raw_env, "obstacles_enabled", False):
                    obstacle_xy = raw_env.scene.env_origins[:, :2] + raw_env.obstacle_disc_xy_local
                    obstacle_np = _tensor_np(obstacle_xy[env_id])
                    obstacle_x, obstacle_y = float(obstacle_np[0]), float(obstacle_np[1])
                    clearance = raw_env._get_obstacle_clearance(raw_env.robot.data.root_pos_w)
                    obstacle_clearance = float(_tensor_np(clearance[env_id]))
                frames.append(
                    {
                        "step": float(step),
                        "base_x": float(base_np[0]),
                        "base_y": float(base_np[1]),
                        "base_z": float(base_np[2]),
                        "ee_x": float(ee_np[0]),
                        "ee_y": float(ee_np[1]),
                        "ee_z": float(ee_np[2]),
                        "target_x": float(target_np[0]),
                        "target_y": float(target_np[1]),
                        "target_z": float(target_np[2]),
                        "base_target_dist": base_target_dist,
                        "ee_pos_error": ee_pos_error,
                        "obstacle_x": obstacle_x,
                        "obstacle_y": obstacle_y,
                        "obstacle_radius": float(args.obstacle_radius),
                        "obstacle_clearance": obstacle_clearance,
                    }
                )
            if bool(np.asarray(dones)[env_id]):
                break

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = args.name or f"rollout_{timestamp}"
        csv_path = output_dir / f"{stem}.csv"
        png_path = output_dir / f"{stem}.png"
        mp4_path = output_dir / f"{stem}.mp4"
        json_path = output_dir / f"{stem}.json"
        _write_csv(csv_path, frames)
        render_status = _render_topdown(mp4_path, png_path, frames, fps=args.fps)
        summary = {
            "checkpoint": str(checkpoint),
            "vec_normalize": str(vec_path) if use_vec_normalize else None,
            "trajectory_manifest": str(manifest),
            "selected_trajectories": selected_trajectories,
            "frames": len(frames),
            "csv": str(csv_path),
            "png": str(png_path),
            "mp4": str(mp4_path) if mp4_path.exists() else None,
            "render_status": render_status,
            "base_target_dist_mean": float(np.mean([f["base_target_dist"] for f in frames])) if frames else None,
            "base_target_dist_max": float(np.max([f["base_target_dist"] for f in frames])) if frames else None,
            "ee_pos_error_mean": float(np.mean([f["ee_pos_error"] for f in frames])) if frames else None,
            "ee_pos_error_max": float(np.max([f["ee_pos_error"] for f in frames])) if frames else None,
        }
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))

        env.close()
        simulation_app.close()
        return 0 if frames else 2
    except Exception:
        import traceback

        traceback.print_exc()
        simulation_app.close()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
