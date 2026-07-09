"""Record an Isaac-rendered rollout video from a trained Proto2 SB3 checkpoint.

This is the real Isaac Lab rendering path, not the top-down diagnostic plot.
It uses Gymnasium RecordVideo against DirectRLEnv.render_mode="rgb_array".
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

from evaluate_recovery_candidate import _resolve_trajectory_manifest, tensor_np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a real Isaac-rendered Proto2 recovery rollout.")
    parser.add_argument("--checkpoint", required=True, help="PPO checkpoint/final_model.zip.")
    parser.add_argument(
        "--recovery_checkpoint",
        default=None,
        help="Optional PPO checkpoint used when conditional recovery routing is active.",
    )
    parser.add_argument("--recovery_route_min_waypoint_fraction", type=float, default=0.65)
    parser.add_argument("--recovery_route_min_pos_error", type=float, default=0.0)
    parser.add_argument("--recovery_route_min_base_target_distance", type=float, default=0.0)
    parser.add_argument("--recovery_route_latch_once", action="store_true")
    parser.add_argument("--vec_normalize", default=None, help="VecNormalize pkl. Defaults to checkpoint parent.")
    parser.add_argument("--disable_vec_normalize", action="store_true")
    parser.add_argument("--base_action_scale", type=float, default=1.0)
    parser.add_argument("--task", default="RecomoProto2TrackEE-v0")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument(
        "--episode_length_s",
        type=float,
        default=0.0,
        help="Optional env episode length override in seconds. Keep 0 to use the task default.",
    )
    parser.add_argument(
        "--base_action_slew_limit",
        type=float,
        default=0.0,
        help="Optional env-level normalized per-step slew limit for base action rows [6,7,8].",
    )
    parser.add_argument(
        "--stop_on_done",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stop recording as soon as any environment reports done. Useful for full trajectory proof clips.",
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--headless", action="store_true", help="Use headless offscreen rendering.")
    parser.add_argument(
        "--enable_cameras",
        action="store_true",
        help="Force Isaac camera/render extensions. This is automatically enabled in headless mode.",
    )
    parser.add_argument(
        "--render_experience",
        type=str,
        default=None,
        help=(
            "Isaac/Kit experience file to use for rendering. Defaults to the local "
            "D3D12 headless rendering app when it exists."
        ),
    )
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
    parser.add_argument(
        "--camera_eye",
        default="3.0,-4.0,2.4",
        help="Overview camera eye as x,y,z in world coordinates.",
    )
    parser.add_argument(
        "--camera_target",
        default="0.0,0.4,0.55",
        help="Overview camera target as x,y,z in world coordinates.",
    )
    parser.add_argument(
        "--debug_vis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable trajectory/EE debug markers for rendered rollout videos.",
    )
    parser.add_argument(
        "--render_polish",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add visual-only lights and floor grid for readable proof videos.",
    )
    parser.add_argument(
        "--grid_half_extent",
        type=float,
        default=2.0,
        help="Half-width in meters of the visual floor grid when --render_polish is enabled.",
    )
    parser.add_argument(
        "--grid_spacing",
        type=float,
        default=0.25,
        help="Spacing in meters for the visual floor grid when --render_polish is enabled.",
    )
    parser.add_argument("--output_dir", default="evaluation_results/videos_rendered/stage1_recovery_rollout")
    parser.add_argument("--name", default=None, help="Optional output basename.")
    return parser.parse_args()


def resolve_render_experience(args: argparse.Namespace) -> str | None:
    """Pick the Isaac rendering app that works on the Windows/Blackwell host."""
    if args.render_experience:
        return args.render_experience

    candidates = [
        Path(r"G:\isaaclab\apps\isaaclab.python.headless.rendering.d3d12.kit"),
        Path("/mnt/g/isaaclab/apps/isaaclab.python.headless.rendering.d3d12.kit"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


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
    return out


def attach_record_video_vecenv_passthrough(video_env, raw_env):
    """Expose Isaac VecEnv attributes hidden by Gymnasium RecordVideo."""
    for name in ("num_envs", "device", "render_mode"):
        if name != "render_mode" and hasattr(video_env, name):
            continue
        if name == "render_mode" and getattr(video_env, name, None) is not None:
            continue
        for source in (raw_env, getattr(raw_env, "unwrapped", None)):
            if source is not None and hasattr(source, name):
                value = getattr(source, name)
                if name == "render_mode" and value is None:
                    continue
                setattr(video_env, name, value)
                break
        else:
            if name == "render_mode":
                setattr(video_env, name, "rgb_array")
    return video_env


def parse_xyz(value: str, *, arg_name: str) -> list[float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError(f"{arg_name} must have three comma-separated values, got: {value!r}")
    return [float(part) for part in parts]


def apply_render_polish(env, args: argparse.Namespace) -> dict[str, object]:
    """Add visual-only context that makes headless Isaac videos easier to read."""
    if not args.render_polish:
        return {"enabled": False}

    raw_env = getattr(env, "unwrapped", env)
    scene = getattr(raw_env, "scene", None)
    stage = getattr(scene, "stage", None)
    if stage is None:
        print("WARNING: Could not find USD stage for render polish")
        return {"enabled": True, "applied": False, "reason": "missing_stage"}

    try:
        from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

        render_root = UsdGeom.Xform.Define(stage, "/World/RenderPolish")

        dome = UsdLux.DomeLight.Define(stage, "/World/RenderPolish/SoftDome")
        dome.CreateIntensityAttr(450.0)
        dome.CreateColorAttr(Gf.Vec3f(0.82, 0.88, 1.0))

        key = UsdLux.DistantLight.Define(stage, "/World/RenderPolish/KeyLight")
        key.CreateIntensityAttr(420.0)
        key.CreateAngleAttr(0.35)
        key.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.86))
        key_xform = UsdGeom.Xformable(key.GetPrim())
        key_xform.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, -35.0))

        grid_root = UsdGeom.Xform.Define(stage, "/World/RenderPolish/FloorGrid")
        half = max(float(args.grid_half_extent), 0.25)
        spacing = max(float(args.grid_spacing), 0.05)
        line_width = 0.006
        grid_z = 0.004
        line_count = int(round((2.0 * half) / spacing)) + 1
        start = -0.5 * spacing * (line_count - 1)
        for index in range(line_count):
            coord = start + index * spacing
            intensity = 0.55 if abs(coord) < 1e-6 else 0.32

            x_line = UsdGeom.Cube.Define(stage, f"/World/RenderPolish/FloorGrid/X_{index:03d}")
            x_line.CreateSizeAttr(1.0)
            x_line.CreateDisplayColorAttr([Gf.Vec3f(intensity, intensity, intensity)])
            x_line_xform = UsdGeom.Xformable(x_line.GetPrim())
            x_line_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, coord, grid_z))
            x_line_xform.AddScaleOp().Set(Gf.Vec3d(2.0 * half, line_width, line_width))

            y_line = UsdGeom.Cube.Define(stage, f"/World/RenderPolish/FloorGrid/Y_{index:03d}")
            y_line.CreateSizeAttr(1.0)
            y_line.CreateDisplayColorAttr([Gf.Vec3f(intensity, intensity, intensity)])
            y_line_xform = UsdGeom.Xformable(y_line.GetPrim())
            y_line_xform.AddTranslateOp().Set(Gf.Vec3d(coord, 0.0, grid_z))
            y_line_xform.AddScaleOp().Set(Gf.Vec3d(line_width, 2.0 * half, line_width))

        robot_mat = UsdShade.Material.Define(stage, "/World/RenderPolish/RobotSlateMaterial")
        robot_shader = UsdShade.Shader.Define(stage, "/World/RenderPolish/RobotSlateMaterial/Shader")
        robot_shader.CreateIdAttr("UsdPreviewSurface")
        robot_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.06, 0.10, 0.14))
        robot_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.78)
        robot_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        robot_mat.CreateSurfaceOutput().ConnectToSource(robot_shader.ConnectableAPI(), "surface")

        robot_mesh_count = 0
        for prim in stage.Traverse():
            path = prim.GetPath().pathString
            if "/Robot/" not in path or not prim.IsA(UsdGeom.Mesh):
                continue
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                robot_mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants
            )
            UsdGeom.Mesh(prim).CreateDisplayColorAttr([Gf.Vec3f(0.06, 0.10, 0.14)])
            robot_mesh_count += 1

        render_root.GetPrim().SetActive(True)
        grid_root.GetPrim().SetActive(True)
        print(
            "Applied render polish: "
            f"grid={line_count}x{line_count}, half_extent={half:.2f}m, "
            f"spacing={spacing:.2f}m, robot_meshes={robot_mesh_count}"
        )
        return {
            "enabled": True,
            "applied": True,
            "grid_lines_per_axis": line_count,
            "grid_half_extent": half,
            "grid_spacing": spacing,
            "robot_meshes": robot_mesh_count,
        }
    except Exception as exc:
        print(f"WARNING: Could not apply render polish: {exc}")
        return {"enabled": True, "applied": False, "reason": str(exc)}


def configure_render_view(env, args: argparse.Namespace) -> None:
    """Set a useful overview camera and optional debug markers for proof videos."""
    eye = parse_xyz(args.camera_eye, arg_name="--camera_eye")
    target = parse_xyz(args.camera_target, arg_name="--camera_target")
    raw_env = getattr(env, "unwrapped", env)
    if hasattr(raw_env, "sim") and hasattr(raw_env.sim, "set_camera_view"):
        raw_env.sim.set_camera_view(eye=eye, target=target)
        print(f"Configured render camera: eye={eye}, target={target}")
    else:
        print("WARNING: Could not find Isaac sim.set_camera_view on rollout environment")
    if args.debug_vis and hasattr(raw_env, "set_debug_vis"):
        raw_env.set_debug_vis(True)
        print("Enabled debug visualization markers for rendered rollout")
    elif args.debug_vis:
        print("WARNING: Could not enable debug visualization on rollout environment")


def _convert_obs(obs):
    if isinstance(obs, tuple):
        obs = obs[0]
    if isinstance(obs, dict):
        obs = obs.get("policy", list(obs.values())[0])
    return np.asarray(tensor_np(obs), dtype=np.float32)


class IsaacLabToSB3VecEnvWrapper:
    """Small VecEnv adapter matching the project evaluation scripts."""

    def __init__(self, venv, *, base_action_scale: float = 1.0, expected_obs_dim: int | None = None):
        from gymnasium import spaces
        from stable_baselines3.common.vec_env import VecEnvWrapper

        class _Wrapper(VecEnvWrapper):
            def __init__(self, wrapped, scale: float, expected_dim: int | None):
                super().__init__(wrapped)
                self.base_action_scale = float(scale)
                self.expected_obs_dim = expected_dim
                self._base_adapter_logged = False
                if hasattr(wrapped.action_space, "shape") and len(wrapped.action_space.shape) > 1:
                    action_dim = wrapped.action_space.shape[-1]
                    self.action_space = spaces.Box(
                        low=wrapped.action_space.low.flatten()[0],
                        high=wrapped.action_space.high.flatten()[0],
                        shape=(action_dim,),
                        dtype=wrapped.action_space.dtype,
                    )
                dummy_obs = _convert_obs(wrapped.reset())
                dummy_obs = self._adapt_obs_dim(dummy_obs)
                obs_shape = (dummy_obs.shape[1],) if len(dummy_obs.shape) > 1 else dummy_obs.shape
                self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)

            def _adapt_obs_dim(self, obs: np.ndarray) -> np.ndarray:
                if self.expected_obs_dim is None or obs.shape[-1] == self.expected_obs_dim:
                    return obs
                if obs.ndim == 2 and obs.shape[1] == self.expected_obs_dim - 1:
                    return np.concatenate([obs, np.zeros((obs.shape[0], 1), dtype=np.float32)], axis=1)
                if obs.ndim == 2 and obs.shape[1] == self.expected_obs_dim + 1:
                    return obs[:, : self.expected_obs_dim]
                return obs

            def reset(self):
                return self._adapt_obs_dim(_convert_obs(self.venv.reset()))

            def step_async(self, actions):
                import torch

                if isinstance(actions, np.ndarray):
                    device = self.venv.unwrapped.device if hasattr(self.venv.unwrapped, "device") else "cuda:0"
                    actions = torch.from_numpy(actions).float().to(device)
                if abs(self.base_action_scale - 1.0) >= 1e-9:
                    if actions.shape[-1] < 9:
                        raise ValueError(f"base action scaling requires at least 9 dims, got {actions.shape[-1]}")
                    actions = actions.clone()
                    if not self._base_adapter_logged:
                        print(
                            f"[render action-adapter] Scaling base action rows [6,7,8] by {self.base_action_scale:.3f}",
                            flush=True,
                        )
                        self._base_adapter_logged = True
                    actions[..., 6:9] *= self.base_action_scale
                self._actions = actions

            def step_wait(self):
                result = self.venv.step(self._actions)
                if len(result) == 5:
                    obs, rewards, terminated, truncated, infos = result
                    dones = terminated | truncated
                else:
                    obs, rewards, dones, infos = result
                obs = self._adapt_obs_dim(_convert_obs(obs))
                rewards = tensor_np(rewards).astype(np.float32)
                dones = tensor_np(dones).astype(bool)
                if isinstance(infos, dict):
                    infos = [infos.copy() for _ in range(len(rewards))]
                elif not isinstance(infos, list):
                    infos = [{} for _ in range(len(rewards))]
                return obs, rewards, dones, infos

        self.env = _Wrapper(venv, base_action_scale, expected_obs_dim)


def main() -> int:
    args = parse_args()
    if args.base_action_scale < 0.0 or args.base_action_scale > 1.0:
        raise ValueError("--base_action_scale must be in [0, 1]")
    if args.recovery_route_min_waypoint_fraction < 0.0 or args.recovery_route_min_waypoint_fraction > 1.0:
        raise ValueError("--recovery_route_min_waypoint_fraction must be in [0, 1]")
    if args.recovery_route_min_pos_error < 0.0:
        raise ValueError("--recovery_route_min_pos_error must be non-negative")
    if args.recovery_route_min_base_target_distance < 0.0:
        raise ValueError("--recovery_route_min_base_target_distance must be non-negative")
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        print(f"checkpoint not found: {checkpoint}")
        return 1
    recovery_checkpoint = Path(args.recovery_checkpoint) if args.recovery_checkpoint else None
    if recovery_checkpoint is not None and not recovery_checkpoint.exists():
        print(f"recovery checkpoint not found: {recovery_checkpoint}")
        return 1

    vec_path = Path(args.vec_normalize) if args.vec_normalize else checkpoint.parent / "vec_normalize.pkl"
    if not args.disable_vec_normalize and not vec_path.exists():
        print(f"VecNormalize stats not found: {vec_path}")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    basename = args.name or f"rendered_rollout_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    video_dir = output_dir / basename
    video_dir.mkdir(parents=True, exist_ok=True)

    from isaaclab.app import AppLauncher

    render_experience = resolve_render_experience(args)
    app_launcher_kwargs = {
        "headless": args.headless,
        "enable_cameras": bool(args.enable_cameras or args.headless),
        "video": True,
    }
    if render_experience:
        app_launcher_kwargs["experience"] = render_experience
        print(f"Using Isaac render experience: {render_experience}")
    app_launcher = AppLauncher(**app_launcher_kwargs)
    simulation_app = app_launcher.app

    try:
        import gymnasium as gym
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import VecNormalize

        from task_spec import register_isaac_lab_tasks
        from rl_platform.tasks.mobile_mm import MobileMMTrackEEEnvCfg
        from rl_platform.tasks.mobile_mm.config import TrajectoryConfig

        register_isaac_lab_tasks()
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        trajectory_dir, manifest, selected_trajectories = _resolve_trajectory_manifest(args, output_dir)
        reset_config = load_stage_reset_config(args.trajectory_stage)
        if reset_config:
            print(
                "Using stage reset offset "
                f"x={reset_config.get('reset_base_x_offset', 0.4415):.4f} "
                f"y={reset_config.get('reset_base_y_offset', 0.2405):.4f}",
                flush=True,
            )

        env_cfg = MobileMMTrackEEEnvCfg()
        env_cfg.num_envs = 1
        env_cfg.scene.num_envs = 1
        if args.episode_length_s > 0.0:
            env_cfg.episode_length_s = float(args.episode_length_s)
        if args.base_action_slew_limit > 0.0:
            env_cfg.task_config.base_action_slew_limit = float(args.base_action_slew_limit)
        env_cfg.task_config.obstacles.enable_obstacles = bool(args.enable_obstacles)
        env_cfg.task_config.obstacles.randomize_per_reset = True
        env_cfg.task_config.obstacles.disc_radius = 0.20
        env_cfg.task_config.obstacles.disc_height = 0.50
        env_cfg.task_config.obstacles.disc_position_x_range = (-0.35, 0.35)
        env_cfg.task_config.obstacles.disc_position_y_range = (0.45, 1.00)
        env_cfg.task_config.obstacles.min_start_clearance = 0.10
        if args.enable_obstacles:
            env_cfg.scene = env_cfg._create_scene_config()
            env_cfg.scene.num_envs = 1

        camera_eye = tuple(parse_xyz(args.camera_eye, arg_name="--camera_eye"))
        camera_target = tuple(parse_xyz(args.camera_target, arg_name="--camera_target"))
        env_cfg.viewer.eye = camera_eye
        env_cfg.viewer.lookat = camera_target

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

        base_env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
        render_polish_meta = apply_render_polish(base_env, args)
        configure_render_view(base_env, args)
        base_env = attach_record_video_vecenv_passthrough(gym.wrappers.RecordVideo(
            base_env,
            str(video_dir),
            step_trigger=lambda step: step == 0,
            video_length=args.steps,
            fps=args.fps,
            disable_logger=True,
        ), base_env)

        expected_obs_dim = int(np.prod(PPO.load(str(checkpoint), device="cpu").observation_space.shape))
        env = IsaacLabToSB3VecEnvWrapper(
            base_env,
            base_action_scale=args.base_action_scale,
            expected_obs_dim=expected_obs_dim,
        ).env
        if not args.disable_vec_normalize:
            env = VecNormalize.load(str(vec_path), env)
            env.training = False
            env.norm_reward = False
        model = PPO.load(str(checkpoint), env=env, device="cuda" if torch.cuda.is_available() else "cpu")
        recovery_model = None
        if recovery_checkpoint is not None:
            recovery_model = PPO.load(
                str(recovery_checkpoint),
                env=env,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            print(
                "[render router] conditional recovery enabled "
                f"checkpoint={recovery_checkpoint} "
                f"min_waypoint_fraction={args.recovery_route_min_waypoint_fraction:.3f} "
                f"min_pos_error={args.recovery_route_min_pos_error:.4f} "
                f"min_base_target_distance={args.recovery_route_min_base_target_distance:.4f} "
                f"latch={args.recovery_route_latch_once}",
                flush=True,
            )

        raw_env = getattr(base_env, "unwrapped", base_env)
        latched_route_mask = np.zeros((1,), dtype=bool)
        route_counts: list[int] = []

        def route_recovery_mask():
            trajectory_manager = raw_env.trajectory_manager
            waypoint_idx = tensor_np(trajectory_manager.current_waypoint_idx).astype(np.float32)
            lengths = getattr(trajectory_manager, "recorded_lengths", None)
            if lengths is None:
                waypoint_fraction = np.zeros_like(waypoint_idx, dtype=np.float32)
            else:
                lengths_np = np.maximum(tensor_np(lengths).astype(np.float32) - 1.0, 1.0)
                waypoint_fraction = np.clip(waypoint_idx / lengths_np, 0.0, 1.0)
            target_pos, _ = trajectory_manager.get_target_pose()
            ee_pos = raw_env.robot.data.body_pos_w[:, raw_env._ee_body_idx, :]
            base_pos = raw_env.robot.data.root_pos_w
            pos_error = tensor_np(torch.linalg.norm(target_pos - ee_pos, dim=-1)).astype(np.float32)
            base_target_distance = tensor_np(torch.linalg.norm(target_pos[:, :2] - base_pos[:, :2], dim=-1)).astype(np.float32)
            route = waypoint_fraction >= float(args.recovery_route_min_waypoint_fraction)
            if args.recovery_route_min_pos_error > 0.0:
                route &= pos_error >= float(args.recovery_route_min_pos_error)
            if args.recovery_route_min_base_target_distance > 0.0:
                route &= base_target_distance >= float(args.recovery_route_min_base_target_distance)
            return route

        obs = env.reset()
        steps_executed = 0
        done_count = 0
        for _ in range(args.steps):
            action, _ = model.predict(obs, deterministic=args.deterministic)
            if recovery_model is not None:
                route_mask = route_recovery_mask()
                if args.recovery_route_latch_once:
                    latched_route_mask |= route_mask
                    route_mask = latched_route_mask.copy()
                recovery_action, _ = recovery_model.predict(obs, deterministic=args.deterministic)
                if np.any(route_mask):
                    action = np.asarray(action, dtype=np.float32).copy()
                    action[route_mask] = recovery_action[route_mask]
                route_counts.append(int(np.count_nonzero(route_mask)))
            obs, _, dones, _ = env.step(action)
            steps_executed += 1
            if bool(np.any(dones)):
                done_count += 1
                if args.stop_on_done:
                    break

        summary = {
            "checkpoint": str(checkpoint),
            "recovery_checkpoint": str(recovery_checkpoint) if recovery_checkpoint is not None else None,
            "vec_normalize": str(vec_path),
            "disable_vec_normalize": bool(args.disable_vec_normalize),
            "base_action_scale": float(args.base_action_scale),
            "recovery_route_min_waypoint_fraction": float(args.recovery_route_min_waypoint_fraction),
            "recovery_route_min_pos_error": float(args.recovery_route_min_pos_error),
            "recovery_route_min_base_target_distance": float(args.recovery_route_min_base_target_distance),
            "recovery_route_latch_once": bool(args.recovery_route_latch_once),
            "recovery_route_fraction": (
                float(np.sum(route_counts) / max(len(route_counts), 1)) if route_counts else 0.0
            ),
            "video_dir": str(video_dir),
            "steps_requested": args.steps,
            "steps_executed": steps_executed,
            "episode_length_s": float(args.episode_length_s) if args.episode_length_s > 0.0 else None,
            "base_action_slew_limit": (
                float(args.base_action_slew_limit) if args.base_action_slew_limit > 0.0 else None
            ),
            "stop_on_done": bool(args.stop_on_done),
            "done_count": done_count,
            "fps": args.fps,
            "render_polish": render_polish_meta,
            "camera_eye": parse_xyz(args.camera_eye, arg_name="--camera_eye"),
            "camera_target": parse_xyz(args.camera_target, arg_name="--camera_target"),
            "selected_trajectories": selected_trajectories,
        }
        summary_path = video_dir / "rendered_rollout_meta.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        env.close()
        simulation_app.close()
        return 0
    except Exception:
        import traceback

        traceback.print_exc()
        simulation_app.close()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
