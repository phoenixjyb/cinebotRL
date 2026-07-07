#!/usr/bin/env python3
"""Generate a policy-envelope FK stage and exact BC labels.

This fixes the previous Stage A mismatch by sampling normalized policy actions
first, then using Isaac FK to generate the EE target poses.  The same sampled
actions are saved as masked BC labels for rows 0..5, so every labelled target is
inside the current ``sim_6joint_gimbal_v1`` action envelope by construction.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


ACTION_DIM = 9
RESET_OFFSET_XY = (0.4415, 0.2405)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--num_trajectories", type=int, default=24)
    parser.add_argument("--num_waypoints", type=int, default=60)
    parser.add_argument("--stage", default="stage0_policy_envelope_fk")
    parser.add_argument("--candidate_count", type=int, default=4096)
    parser.add_argument("--seed_action_radius", type=float, default=0.85)
    parser.add_argument("--path_action_radius", type=float, default=0.18)
    parser.add_argument("--max_start_offset_error", type=float, default=0.18)
    parser.add_argument(
        "--start_action_mode",
        choices=["zero_safe_home", "search_reset_offset"],
        default="zero_safe_home",
        help=(
            "zero_safe_home starts every path from normalized zero action so reset "
            "state and first target are self-consistent; search_reset_offset keeps "
            "the older diagnostic search near the historical reset offset."
        ),
    )
    parser.add_argument(
        "--reference_manifest",
        type=Path,
        default=Path("trajectoryToLearn/stage0_fixedbase_micro/manifest.txt"),
        help="Small existing manifest used only to instantiate the env.",
    )
    parser.add_argument(
        "--output_dataset",
        type=Path,
        default=Path("data/policy_envelope_fk/obs_dataset_policy_envelope_fk_arm6.npz"),
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def wxyz_to_xyzw(quat):
    return [float(quat[1]), float(quat[2]), float(quat[3]), float(quat[0])]


def quat_conj_np(q):
    import numpy as np

    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def quat_multiply_np(q1, q2):
    import numpy as np

    w1, x1, y1, z1 = [q1[..., i] for i in range(4)]
    w2, x2, y2, z2 = [q2[..., i] for i in range(4)]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def quat_to_axis_angle_np(q):
    import numpy as np

    q = q.copy()
    q /= np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    q *= np.where(q[..., :1] < 0.0, -1.0, 1.0)
    xyz = q[..., 1:]
    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(norm, np.clip(q[..., :1], -1.0, 1.0))
    return xyz / np.maximum(norm, 1e-8) * angle


def build_action_paths(seed_actions, num_waypoints: int, radius: float, rng):
    import numpy as np

    paths = []
    for idx, seed in enumerate(seed_actions):
        action = np.zeros((num_waypoints, 6), dtype=np.float32)
        phases = rng.uniform(0.0, 2.0 * np.pi, size=6)
        freqs = rng.choice([0.5, 1.0, 1.5], size=6, replace=True)
        scales = rng.uniform(0.35, 1.0, size=6) * radius
        for step in range(num_waypoints):
            t = step / max(num_waypoints - 1, 1)
            wave = np.sin(2.0 * np.pi * freqs * t + phases)
            ease = 0.5 - 0.5 * math.cos(2.0 * math.pi * t)
            action[step] = seed + ease * scales * wave
        action = np.clip(action, -1.0, 1.0)
        action[0] = seed
        paths.append(action)
    return np.stack(paths, axis=0).astype(np.float32)


def main() -> int:
    args = parse_args()
    require(args.num_waypoints >= 50, "--num_waypoints must be >= 50 for 5s duration")
    require(args.num_trajectories > 0, "--num_trajectories must be positive")
    print(
        f"[policy-fk] start stage={args.stage} num_envs={args.num_envs} "
        f"num_trajectories={args.num_trajectories} num_waypoints={args.num_waypoints}",
        flush=True,
    )

    import numpy as np
    import torch
    from isaaclab.app import AppLauncher

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("[policy-fk] launching Isaac app", flush=True)
    app_launcher = AppLauncher(headless=args.headless, enable_cameras=False, device="cuda:0")
    simulation_app = app_launcher.app
    print("[policy-fk] Isaac app launched", flush=True)

    try:
        print("[policy-fk] importing env modules", flush=True)
        from rl_platform.tasks.mobile_mm import MobileMMTrackEEEnv, MobileMMTrackEEEnvCfg
        from rl_platform.tasks.mobile_mm.config import TrajectoryConfig
        from rl_platform.tasks.mobile_mm.joint_names import ARM_JOINT_NAMES
        from rl_platform.tasks.mobile_mm.observations import compose_observation, get_observation_dimensions
        print("[policy-fk] env modules imported", flush=True)

        reference_manifest = args.reference_manifest
        if not reference_manifest.is_absolute():
            reference_manifest = PROJECT_ROOT / reference_manifest
        require(reference_manifest.exists(), f"reference manifest not found: {reference_manifest}")
        print(f"[policy-fk] reference manifest={reference_manifest}", flush=True)

        env_cfg = MobileMMTrackEEEnvCfg()
        env_cfg.num_envs = args.num_envs
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = args.seed
        env_cfg.task_config.obstacles.enable_obstacles = False
        env_cfg.task_config.base_assist.enable = False
        env_cfg.task_config.trajectory = TrajectoryConfig(
            type="multi_recorded",
            trajectory_dir=str(PROJECT_ROOT),
            trajectory_pattern="**/*.json",
            trajectory_manifest_file=str(reference_manifest),
            max_trajectories=min(args.num_trajectories, 24),
            min_duration_seconds=5.0,
            randomize_start_waypoint=False,
            reset_base_to_trajectory_start=False,
        )
        print("[policy-fk] creating env", flush=True)
        env = MobileMMTrackEEEnv(cfg=env_cfg)
        print("[policy-fk] env created; resetting", flush=True)
        env.reset()
        print("[policy-fk] env reset complete; initializing helpers", flush=True)
        env._initialize_ee_body_idx()
        env._verify_joint_mapping()
        env._initialize_joint_limits()
        arm_ids = env._get_joint_ids(ARM_JOINT_NAMES, "_arm_joint_ids")
        print(f"[policy-fk] helper init complete; arm_ids={arm_ids}", flush=True)

        safe_home = env.arm_safe_home.to(env.device)
        action_radius = env.arm_action_radius.to(env.device)
        lower = safe_home - action_radius
        upper = safe_home + action_radius

        def fk(actions_np):
            actions_np = np.asarray(actions_np, dtype=np.float32)
            print(f"[policy-fk] fk start rows={actions_np.shape[0]}", flush=True)
            positions = []
            quats = []
            for start in range(0, actions_np.shape[0], args.num_envs):
                if start == 0 or start % max(args.num_envs * 8, 1) == 0:
                    print(f"[policy-fk] fk batch start={start}/{actions_np.shape[0]}", flush=True)
                batch = actions_np[start : start + args.num_envs]
                count = batch.shape[0]
                env_ids = torch.arange(count, dtype=torch.long, device=env.device)
                actions_t = torch.as_tensor(batch, dtype=torch.float32, device=env.device)
                targets = safe_home.unsqueeze(0) + actions_t * action_radius.unsqueeze(0)
                targets = torch.clamp(targets, lower.unsqueeze(0), upper.unsqueeze(0))
                zeros = torch.zeros_like(targets)
                env.robot.set_joint_position_target(targets, joint_ids=arm_ids, env_ids=env_ids)
                env.robot.write_joint_state_to_sim(targets, zeros, joint_ids=arm_ids, env_ids=env_ids)
                env._lock_base_ppr_joints(env_ids=env_ids)
                env._lock_passive_joints(env_ids=env_ids)
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(env.sim.get_physics_dt())
                base = env.robot.data.root_pos_w[:count]
                ee_pos = env.robot.data.body_pos_w[:count, env._ee_body_idx, :]
                ee_quat = env.robot.data.body_quat_w[:count, env._ee_body_idx, :]
                positions.append((ee_pos - base).detach().cpu().numpy().astype(np.float32))
                quats.append(ee_quat.detach().cpu().numpy().astype(np.float32))
            print(f"[policy-fk] fk complete rows={actions_np.shape[0]}", flush=True)
            return np.concatenate(positions, axis=0), np.concatenate(quats, axis=0)

        offset = np.asarray([RESET_OFFSET_XY[0], RESET_OFFSET_XY[1]], dtype=np.float32)
        if args.start_action_mode == "zero_safe_home":
            print("[policy-fk] using zero-safe-home start actions", flush=True)
            seed_actions = np.zeros((args.num_trajectories, 6), dtype=np.float32)
            seed_pos, _ = fk(seed_actions)
            start_error = np.linalg.norm(seed_pos[:, :2] - offset[None, :], axis=1)
            selected_start_errors = start_error
            selected_max_error = float(start_error.max())
            selected_mean_error = float(start_error.mean())
            print(
                "[policy-fk] zero-start offset diagnostic "
                f"mean={selected_mean_error:.4f}m max={selected_max_error:.4f}m; "
                "this is expected to differ from historical hand-authored target starts",
                flush=True,
            )
        else:
            print("[policy-fk] sampling candidate actions", flush=True)
            candidate_actions = rng.uniform(
                -args.seed_action_radius,
                args.seed_action_radius,
                size=(args.candidate_count, 6),
            ).astype(np.float32)
            candidate_pos, _ = fk(candidate_actions)
            start_error = np.linalg.norm(candidate_pos[:, :2] - offset[None, :], axis=1)
            order = np.argsort(start_error)
            best = order[: args.num_trajectories]
            selected_max_error = float(start_error[best[-1]])
            selected_mean_error = float(start_error[best].mean())
            print(
                "[policy-fk] selected seed offset error "
                f"mean={selected_mean_error:.4f}m max={selected_max_error:.4f}m "
                f"threshold={args.max_start_offset_error:.4f}m",
                flush=True,
            )
            if selected_max_error > args.max_start_offset_error:
                print(
                    "[policy-fk] warning: selected seeds exceed preferred reset offset; "
                    "continuing because this may be acceptable for diagnostic smoke runs",
                    flush=True,
                )
            seed_actions = candidate_actions[best]
            selected_start_errors = start_error[best]
        print("[policy-fk] building action paths", flush=True)
        action_paths = build_action_paths(seed_actions, args.num_waypoints, args.path_action_radius, rng)
        flat_actions = action_paths.reshape(-1, 6)
        fk_pos_rel, fk_quat = fk(flat_actions)
        fk_pos_rel = fk_pos_rel.reshape(args.num_trajectories, args.num_waypoints, 3)
        fk_quat = fk_quat.reshape(args.num_trajectories, args.num_waypoints, 4)
        reset_base_xy_offset = fk_pos_rel[0, 0, :2].astype(np.float32)
        print(
            "[policy-fk] generated reset offset "
            f"x={float(reset_base_xy_offset[0]):.4f} y={float(reset_base_xy_offset[1]):.4f}",
            flush=True,
        )

        stage_dir = PROJECT_ROOT / "trajectoryToLearn" / args.stage
        print(f"[policy-fk] writing stage files to {stage_dir}", flush=True)
        data_dir = stage_dir / "generated"
        data_dir.mkdir(parents=True, exist_ok=True)
        manifest_lines = [
            f"# Generated by scripts/imitation/generate_policy_envelope_fk_stage.py for {args.stage}",
            "# Paths are relative to the project root.",
        ]
        for traj_idx in range(args.num_trajectories):
            poses = []
            for step in range(args.num_waypoints):
                poses.append(
                    {
                        "position": [round(float(v), 6) for v in fk_pos_rel[traj_idx, step]],
                        "orientation": [round(v, 6) for v in wxyz_to_xyzw(fk_quat[traj_idx, step])],
                    }
                )
            path = data_dir / f"{traj_idx:03d}_policy_envelope_fk.json"
            path.write_text(json.dumps({"poses": poses}, indent=2) + "\n", encoding="utf-8")
            manifest_lines.append(path.relative_to(PROJECT_ROOT).as_posix())
        (stage_dir / "manifest.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
        (stage_dir / "reset_config.json").write_text(
            json.dumps(
                {
                    "schema": "policy_envelope_fk_reset_config_v1",
                    "reset_base_x_offset": float(reset_base_xy_offset[0]),
                    "reset_base_y_offset": float(reset_base_xy_offset[1]),
                    "source": "first waypoint FK relative XY",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print("[policy-fk] stage files written", flush=True)

        observations = []
        actions = []
        masks = []
        print("[policy-fk] composing BC observations", flush=True)
        for traj_idx in range(args.num_trajectories):
            base_pos = np.zeros((args.num_waypoints, 3), dtype=np.float32)
            base_pos[:, 0] = fk_pos_rel[traj_idx, 0, 0] - reset_base_xy_offset[0]
            base_pos[:, 1] = fk_pos_rel[traj_idx, 0, 1] - reset_base_xy_offset[1]
            base_quat = np.zeros((args.num_waypoints, 4), dtype=np.float32)
            base_quat[:, 0] = 1.0
            base_lin_vel = np.zeros((args.num_waypoints, 3), dtype=np.float32)
            base_ang_vel = np.zeros((args.num_waypoints, 3), dtype=np.float32)

            joint_pos = np.zeros((args.num_waypoints, 9), dtype=np.float32)
            joint_vel = np.zeros((args.num_waypoints, 9), dtype=np.float32)
            target_actions = action_paths[traj_idx]
            joint_targets = safe_home.detach().cpu().numpy()[None, :] + target_actions * action_radius.detach().cpu().numpy()[None, :]
            joint_pos[:, 3:9] = joint_targets
            if args.num_waypoints > 1:
                joint_vel[:-1, 3:9] = np.diff(joint_targets, axis=0) / 0.1
                joint_vel[-1, 3:9] = joint_vel[-2, 3:9]

            ee_pos = base_pos + fk_pos_rel[traj_idx]
            ee_quat = fk_quat[traj_idx]
            target_pos = ee_pos.copy()
            target_quat = ee_quat.copy()
            ee_lin_vel = np.zeros((args.num_waypoints, 3), dtype=np.float32)
            if args.num_waypoints > 1:
                ee_lin_vel[:-1] = np.diff(ee_pos, axis=0) / 0.1
                ee_lin_vel[-1] = ee_lin_vel[-2]
            ee_ang_vel = np.zeros((args.num_waypoints, 3), dtype=np.float32)
            if args.num_waypoints > 1:
                rel = quat_multiply_np(ee_quat[1:], quat_conj_np(ee_quat[:-1]))
                ee_ang_vel[:-1] = quat_to_axis_angle_np(rel) / 0.1
                ee_ang_vel[-1] = ee_ang_vel[-2]

            lookahead = []
            for step in range(args.num_waypoints):
                idx = np.clip(step + np.arange(1, 4), 0, args.num_waypoints - 1)
                lookahead.append(target_pos[idx])
            lookahead = np.asarray(lookahead, dtype=np.float32)

            action_history = np.zeros((args.num_waypoints, 2, ACTION_DIM), dtype=np.float32)
            full_actions = np.zeros((args.num_waypoints, ACTION_DIM), dtype=np.float32)
            full_actions[:, 0:6] = target_actions
            for step in range(args.num_waypoints):
                if step - 2 >= 0:
                    action_history[step, 0] = full_actions[step - 2]
                if step - 1 >= 0:
                    action_history[step, 1] = full_actions[step - 1]

            with torch.no_grad():
                obs = compose_observation(
                    base_pos=torch.from_numpy(base_pos).float(),
                    base_quat=torch.from_numpy(base_quat).float(),
                    base_lin_vel=torch.from_numpy(base_lin_vel).float(),
                    base_ang_vel=torch.from_numpy(base_ang_vel).float(),
                    joint_pos=torch.from_numpy(joint_pos).float(),
                    joint_vel=torch.from_numpy(joint_vel).float(),
                    ee_pos=torch.from_numpy(ee_pos).float(),
                    ee_quat=torch.from_numpy(ee_quat).float(),
                    ee_lin_vel=torch.from_numpy(ee_lin_vel).float(),
                    ee_ang_vel=torch.from_numpy(ee_ang_vel).float(),
                    target_pos=torch.from_numpy(target_pos).float(),
                    target_quat=torch.from_numpy(target_quat).float(),
                    lookahead_pos=torch.from_numpy(lookahead).float(),
                    action_history=torch.from_numpy(action_history).float(),
                    contact_forces=torch.zeros((args.num_waypoints, 1), dtype=torch.float32),
                    min_obstacle_dist=None,
                ).cpu().numpy().astype(np.float32)
            observations.append(obs)
            actions.append(full_actions)
            mask = np.zeros_like(full_actions, dtype=bool)
            mask[:, 0:6] = True
            masks.append(mask)

        print("[policy-fk] validating and saving dataset", flush=True)
        observations_np = np.concatenate(observations, axis=0)
        actions_np = np.concatenate(actions, axis=0)
        masks_np = np.concatenate(masks, axis=0)
        expected_dim = get_observation_dimensions(
            num_joints=6,
            num_contacts=1,
            use_lookahead=True,
            lookahead_steps=3,
            use_action_history=True,
            action_history_length=2,
            action_dim=ACTION_DIM,
            use_obstacles=False,
        )
        require(observations_np.shape[1] == expected_dim, f"obs dim {observations_np.shape[1]} != {expected_dim}")
        require(np.isfinite(observations_np).all(), "observations contain non-finite values")
        require(np.isfinite(actions_np).all(), "actions contain non-finite values")
        require(float(np.max(np.abs(actions_np))) <= 1.000001, "actions outside [-1,1]")

        metadata = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "schema": "policy_envelope_fk_stage_v1",
            "stage": args.stage,
            "manifest": str(stage_dir / "manifest.txt"),
            "samples": int(observations_np.shape[0]),
            "obs_dim": int(observations_np.shape[1]),
            "act_dim": int(actions_np.shape[1]),
            "num_trajectories": int(args.num_trajectories),
            "num_waypoints": int(args.num_waypoints),
            "valid_action_counts": masks_np.sum(axis=0).astype(int).tolist(),
            "seed_start_offset_error_mean_m": float(selected_start_errors.mean()),
            "seed_start_offset_error_max_m": float(selected_start_errors.max()),
            "reset_base_x_offset": float(reset_base_xy_offset[0]),
            "reset_base_y_offset": float(reset_base_xy_offset[1]),
            "seed_action_radius": float(args.seed_action_radius),
            "path_action_radius": float(args.path_action_radius),
        }
        args.output_dataset.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output_dataset,
            observations=observations_np,
            actions=actions_np,
            action_valid_mask=masks_np,
            metadata=json.dumps(metadata, indent=2),
            action_contract=np.asarray("sim_6joint_gimbal_v1"),
            observation_dim=np.asarray(observations_np.shape[1], dtype=np.int32),
        )
        print(f"stage manifest: {stage_dir / 'manifest.txt'}", flush=True)
        print(f"dataset: {args.output_dataset}", flush=True)
        print(f"observations: {observations_np.shape}, actions: {actions_np.shape}", flush=True)
        print(f"valid action counts: {metadata['valid_action_counts']}", flush=True)
        print(
            "seed offset error: "
            f"mean={metadata['seed_start_offset_error_mean_m']:.4f}m "
            f"max={metadata['seed_start_offset_error_max_m']:.4f}m",
            flush=True,
        )
        env.close()
    finally:
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
