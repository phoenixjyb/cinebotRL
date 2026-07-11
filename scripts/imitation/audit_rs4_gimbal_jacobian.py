#!/usr/bin/env python3
"""Audit the live cam_link rotational Jacobian for an RS4 differential-IK adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory_stage", default="stage_gik_no_obstacle79_nominal")
    parser.add_argument("--damping", type=float, default=0.02)
    parser.add_argument("--probe_rate_deg_s", type=float, default=10.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=args.headless, enable_cameras=False, device="cuda:0")
    simulation_app = app_launcher.app
    try:
        import numpy as np
        import torch

        from rl_platform.tasks.mobile_mm import MobileMMTrackEEEnv, MobileMMTrackEEEnvCfg
        from rl_platform.tasks.mobile_mm.config import TrajectoryConfig
        from rl_platform.tasks.mobile_mm.joint_names import ARM_JOINT_NAMES

        stage_dir = PROJECT_ROOT / "trajectoryToLearn" / args.trajectory_stage
        manifest = stage_dir / "manifest.txt"
        if not manifest.exists():
            raise FileNotFoundError(manifest)

        cfg = MobileMMTrackEEEnvCfg()
        cfg.num_envs = 1
        cfg.scene.num_envs = 1
        cfg.task_config.obstacles.enable_obstacles = False
        cfg.task_config.base_assist.enable = False
        cfg.task_config.trajectory = TrajectoryConfig(
            type="multi_recorded",
            trajectory_dir=str(PROJECT_ROOT),
            trajectory_manifest_file=str(manifest),
            max_trajectories=1,
            min_duration_seconds=5.0,
            randomize_start_waypoint=False,
        )
        env = MobileMMTrackEEEnv(cfg=cfg)
        env.reset()
        env._initialize_ee_body_idx()
        env._verify_joint_mapping()

        jacobians = env.robot.root_physx_view.get_jacobians()
        body_count = len(env.robot.body_names)
        joint_count = len(env.robot.joint_names)
        jacobian_body_count = int(jacobians.shape[1])
        jacobian_column_count = int(jacobians.shape[-1])
        body_offset = jacobian_body_count - body_count
        joint_offset = jacobian_column_count - joint_count
        ee_jacobian_index = int(env._ee_body_idx) + body_offset
        if not 0 <= ee_jacobian_index < jacobian_body_count:
            raise RuntimeError(
                f"resolved EE Jacobian index {ee_jacobian_index} outside {jacobian_body_count} bodies"
            )

        gimbal_names = ARM_JOINT_NAMES[3:6]
        gimbal_joint_ids = [env.robot.joint_names.index(name) for name in gimbal_names]
        gimbal_columns = [joint_offset + joint_id for joint_id in gimbal_joint_ids]
        rotational = jacobians[0, ee_jacobian_index, 3:6, gimbal_columns].to(torch.float64)
        singular_values = torch.linalg.svdvals(rotational)
        condition = float((singular_values.max() / singular_values.min().clamp_min(1e-12)).item())

        rate_rad_s = float(np.deg2rad(args.probe_rate_deg_s))
        identity = torch.eye(3, dtype=torch.float64, device=rotational.device)
        probes = []
        for axis in range(3):
            desired = torch.zeros(3, dtype=torch.float64, device=rotational.device)
            desired[axis] = rate_rad_s
            # Damped least-squares pseudoinverse: J^T (J J^T + lambda^2 I)^-1.
            joint_rate = rotational.T @ torch.linalg.solve(
                rotational @ rotational.T + float(args.damping) ** 2 * identity,
                desired,
            )
            reproduced = rotational @ joint_rate
            probes.append(
                {
                    "world_axis": "xyz"[axis],
                    "desired_rad_s": desired.cpu().tolist(),
                    "joint_rate_rad_s": joint_rate.cpu().tolist(),
                    "reproduced_rad_s": reproduced.cpu().tolist(),
                    "residual_norm_rad_s": float(torch.linalg.norm(reproduced - desired).item()),
                }
            )

        report = {
            "schema": "cinebotrl_rs4_gimbal_jacobian_audit_v1",
            "trajectory_stage": args.trajectory_stage,
            "body_count": body_count,
            "jacobian_body_count": jacobian_body_count,
            "body_offset": body_offset,
            "ee_body_index": int(env._ee_body_idx),
            "ee_jacobian_index": ee_jacobian_index,
            "joint_count": joint_count,
            "jacobian_column_count": jacobian_column_count,
            "joint_offset": joint_offset,
            "gimbal_joint_names": list(gimbal_names),
            "gimbal_joint_ids": gimbal_joint_ids,
            "gimbal_jacobian_columns": gimbal_columns,
            "rotational_jacobian": rotational.cpu().tolist(),
            "singular_values": singular_values.cpu().tolist(),
            "condition_number": condition,
            "damping": float(args.damping),
            "probe_rate_deg_s": float(args.probe_rate_deg_s),
            "probes": probes,
        }
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        env.close()
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
