"""Collect (observation, expert_action) demonstration pairs for IL pre-training.

Follows the Isaac Lab init pattern: AppLauncher is created before any
isaaclab module is imported.  Run via::

    isaaclab.bat -p src/rl_platform/tasks/recomoproto1/il_dataset.py \\
        --task RecomoProto1TrackEE-v0 --num_envs 1 --headless \\
        --num_episodes 500 --output_dir data/il_demos
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root setup (must precede any local imports)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent          # recomoproto1/
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent  # project root (4 levels up)
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect expert demonstrations for IL pre-training"
    )
    parser.add_argument("--task", type=str, default="RecomoProto1TrackEE-v0")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/il_demos",
        help="Directory to write the .npz demo file",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=500,
        help="Number of trajectory episodes to collect",
    )
    parser.add_argument(
        "--trajectory_dir",
        type=str,
        default="trajectoryToLearn/world_json",
        help="Directory containing expert trajectory JSON files",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Expert controller (defined here so it can be instantiated after AppLauncher)
# ---------------------------------------------------------------------------

class ExpertController:
    """Combines a DifferentialIK arm controller and a proportional base controller."""

    def __init__(
        self,
        robot_articulation,
        ee_body_idx: int,
        arm_joint_ids: list[int],
        base_joint_ids: list[int],
        cfg,        # ILConfig
        device: str,
    ):
        # Import after Isaac Sim init
        import torch
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

        self.device = device
        self.arm_joint_ids = arm_joint_ids
        self.base_joint_ids = base_joint_ids
        self.num_arm_joints = len(arm_joint_ids)
        self.cfg = cfg

        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            ik_method="dls",
            ik_params={"lambda_val": cfg.ik_damping},
        )
        num_envs = robot_articulation.num_instances
        self.ik_ctrl = DifferentialIKController(
            cfg=ik_cfg,
            num_envs=num_envs,
            device=device,
        )

        self._torch = torch

    # ------------------------------------------------------------------
    def compute_arm_action(
        self,
        current_ee_pos,   # [N, 3]
        current_ee_quat,  # [N, 4]  (x,y,z,w)
        target_pos,       # [N, 3]
        target_quat,      # [N, 4]
        jacobian,         # [N, 6, num_arm_joints]
        joint_pos,        # [N, num_arm_joints]  current joint positions
    ):
        """Return normalised arm joint position deltas [N, 6].

        Steps:
        1. Feed target pose + Jacobian into DifferentialIKController → joint velocity targets
        2. Treat velocities as position deltas (small-step approximation)
        3. Clip and normalise to [-1, 1]
        """
        torch = self._torch

        # Pack command: [target_pos | target_quat]  shape [N, 7]
        command = torch.cat([target_pos, target_quat], dim=-1)
        self.ik_ctrl.set_command(command)

        # joint_pos_subset [N, num_arm_joints]
        delta_joints = self.ik_ctrl.compute(
            ee_pos=current_ee_pos,
            ee_quat=current_ee_quat,
            jacobian=jacobian,
            joint_pos=joint_pos,
        )

        # Check for divergence and warn
        max_delta = delta_joints.abs().max().item()
        if max_delta > 5.0:
            warnings.warn(
                f"[ExpertController] Large IK delta detected ({max_delta:.2f} rad). "
                "Clipping to safe range."
            )
        delta_joints = delta_joints.clamp(-1.5, 1.5)

        # Normalise to [-1, 1] using the typical max joint velocity (2.0 rad/s)
        normalized = delta_joints / 2.0
        return normalized.clamp(-1.0, 1.0)  # [N, 6]

    # ------------------------------------------------------------------
    def compute_base_action(
        self,
        base_pos,   # [N, 3]
        base_quat,  # [N, 4]  (x,y,z,w)  — world-frame
        target_pos, # [N, 3]  — world-frame target (x,y only used)
    ):
        """Return normalised [vx, wz] commands [N, 2].

        Proportional controller:
        - vx  ∝ distance to target (clamped by base_max_vel)
        - wz  ∝ heading error (sin of angle between robot forward and target direction)
        """
        torch = self._torch

        # xy direction to target
        delta_xy = target_pos[:, :2] - base_pos[:, :2]          # [N, 2]
        dist = delta_xy.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # [N, 1]
        direction = delta_xy / dist                              # [N, 2]

        # Extract yaw from quaternion (x,y,z,w ordering)
        qx, qy, qz, qw = (base_quat[:, i] for i in range(4))
        yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

        # Robot forward unit vector in world xy
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        forward = torch.stack([cos_yaw, sin_yaw], dim=-1)       # [N, 2]

        # Heading error (signed): cross product in 2-D
        # cross(forward, direction) = fwd_x*dir_y - fwd_y*dir_x
        heading_err_sin = (forward[:, 0] * direction[:, 1] - forward[:, 1] * direction[:, 0])

        # vx: proportional to distance, scaled and clamped
        vx = (self.cfg.base_kp * dist.squeeze(-1)).clamp(-self.cfg.base_max_vel, self.cfg.base_max_vel)
        # wz: proportional to sin(heading_error)
        wz = (self.cfg.base_kp * heading_err_sin).clamp(-self.cfg.base_max_vel, self.cfg.base_max_vel)

        return torch.stack([vx, wz], dim=-1)  # [N, 2]


# ---------------------------------------------------------------------------
# Demo collection loop
# ---------------------------------------------------------------------------

def collect_demonstrations(env, controller, trajectory_loader, num_episodes: int, device: str) -> dict:
    """Step through trajectory waypoints and record (obs, action) pairs.

    Returns a dict with:
        "observations": np.ndarray [N, obs_dim]
        "actions":      np.ndarray [N, act_dim]
        "episode_lengths": list[int]
    """
    import numpy as np
    import torch

    all_obs: list[np.ndarray] = []
    all_acts: list[np.ndarray] = []
    episode_lengths: list[int] = []

    for ep_idx in range(num_episodes):
        traj = trajectory_loader.sample_trajectory()
        waypoints_pos  = traj["positions"]   # [L, 3]
        waypoints_quat = traj["orientations"] # [L, 4]
        num_waypoints  = waypoints_pos.shape[0]

        obs, _ = env.reset()
        ep_len = 0

        for wp_idx in range(num_waypoints):
            target_pos  = waypoints_pos[wp_idx].unsqueeze(0).to(device)   # [1, 3]
            target_quat = waypoints_quat[wp_idx].unsqueeze(0).to(device)  # [1, 4]

            # -------------------------------------------------------
            # Query robot state directly from environment internals
            # -------------------------------------------------------
            robot = env.unwrapped.robot  # ArticulationView / Articulation
            num_envs = robot.num_instances

            # EE pos/quat from robot body (index already resolved at env init)
            ee_body_idx = env.unwrapped.ee_body_idx
            # [N, num_bodies, 3] / [N, num_bodies, 4]
            bodies_pos  = robot.data.body_pos_w   # world-frame body positions
            bodies_quat = robot.data.body_quat_w
            current_ee_pos  = bodies_pos[:, ee_body_idx, :]    # [N, 3]
            current_ee_quat = bodies_quat[:, ee_body_idx, :]   # [N, 4]

            # Joint positions for arm
            arm_ids  = env.unwrapped.arm_joint_ids
            base_ids = env.unwrapped.base_joint_ids
            joint_pos_all = robot.data.joint_pos  # [N, total_joints]
            joint_pos_arm = joint_pos_all[:, arm_ids]  # [N, 6]

            # Base pose
            base_pos  = robot.data.root_pos_w   # [N, 3]
            base_quat = robot.data.root_quat_w  # [N, 4]

            # -------------------------------------------------------
            # Jacobian for DiffIK  shape [N, 6, total_joints+6]
            # Slice to arm joints only (skip the first 6 floating base columns)
            # TODO: Verify exact Isaac Lab Jacobian API for recomoProto1 robot
            # -------------------------------------------------------
            jacobians_full = robot.root_physx_view.get_jacobians()
            # Expected shape: [N, 6, num_joints + 6]; arm joints start at offset 6
            arm_offset = 6
            jac_arm = jacobians_full[:, :, arm_offset : arm_offset + len(arm_ids)]  # [N, 6, 6]

            # -------------------------------------------------------
            # Compute expert actions
            # -------------------------------------------------------
            arm_act = controller.compute_arm_action(
                current_ee_pos, current_ee_quat, target_pos, target_quat, jac_arm, joint_pos_arm
            )  # [N, 6]
            base_act = controller.compute_base_action(base_pos, base_quat, target_pos)  # [N, 2]

            action = torch.cat([arm_act, base_act], dim=-1)  # [N, 8]

            # -------------------------------------------------------
            # Record obs and step
            # -------------------------------------------------------
            if isinstance(obs, dict):
                obs_np = np.concatenate(
                    [v.cpu().numpy() if hasattr(v, "cpu") else v for v in obs.values()],
                    axis=-1,
                )
            elif hasattr(obs, "cpu"):
                obs_np = obs.cpu().numpy()
            else:
                obs_np = np.asarray(obs)

            act_np = action.cpu().numpy()
            all_obs.append(obs_np)
            all_acts.append(act_np)
            ep_len += 1

            obs, _, terminated, truncated, _ = env.step(action.cpu().numpy())

            if np.any(terminated) or np.any(truncated):
                break

        episode_lengths.append(ep_len)
        if (ep_idx + 1) % 50 == 0:
            print(f"  Collected {ep_idx + 1}/{num_episodes} episodes "
                  f"(total transitions so far: {sum(episode_lengths)})")

    observations = np.concatenate(all_obs, axis=0)
    actions      = np.concatenate(all_acts, axis=0)
    return {"observations": observations, "actions": actions, "episode_lengths": episode_lengths}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("=" * 70)
    print("IL Demo Collection for RecomoProto1")
    print("=" * 70)

    # Step 1: AppLauncher MUST be created before importing any Isaac Lab module
    print("\n[1/4] Initialising Isaac Sim...")
    from isaaclab.app import AppLauncher
    import torch

    app_launcher = AppLauncher(
        headless=args.headless,
        num_envs=args.num_envs,
    )
    simulation_app = app_launcher.app  # noqa: F841  (keeps simulation alive)

    # Step 2: Import Isaac Lab / task modules AFTER launcher is up
    print("[2/4] Importing task modules...")
    import gymnasium as gym

    from rl_platform.tasks.recomoproto1.config import ILConfig
    from rl_platform.tasks.recomoproto1.multi_trajectory import MultiTrajectoryLoader

    # Register the task
    from task_spec import register_isaac_lab_tasks
    register_isaac_lab_tasks()

    # Step 3: Build environment
    print(f"[3/4] Building env: {args.task} ...")
    traj_dir = (PROJECT_ROOT / args.trajectory_dir).resolve()
    env = gym.make(
        args.task,
        num_envs=args.num_envs,
        trajectory_dir=str(traj_dir),
    )

    device = env.unwrapped.device if hasattr(env.unwrapped, "device") else "cuda:0"

    il_cfg = ILConfig()
    trajectory_loader = MultiTrajectoryLoader(
        trajectory_dir=str(traj_dir),
        device=device,
    )

    robot = env.unwrapped.robot
    ee_body_idx  = env.unwrapped.ee_body_idx
    arm_joint_ids  = env.unwrapped.arm_joint_ids
    base_joint_ids = env.unwrapped.base_joint_ids

    controller = ExpertController(
        robot_articulation=robot,
        ee_body_idx=ee_body_idx,
        arm_joint_ids=arm_joint_ids,
        base_joint_ids=base_joint_ids,
        cfg=il_cfg,
        device=device,
    )

    # Step 4: Collect demonstrations
    print(f"[4/4] Collecting {args.num_episodes} episodes...")
    demos = collect_demonstrations(
        env=env,
        controller=controller,
        trajectory_loader=trajectory_loader,
        num_episodes=args.num_episodes,
        device=device,
    )

    # Save
    import numpy as np
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / il_cfg.demo_filename
    np.savez_compressed(
        str(output_path),
        observations=demos["observations"],
        actions=demos["actions"],
        episode_lengths=np.array(demos["episode_lengths"]),
    )

    total_transitions = demos["observations"].shape[0]
    print(f"\n{'='*70}")
    print(f"[OK] Demo collection complete!")
    print(f"     Total transitions : {total_transitions:,}")
    print(f"     Episodes collected: {args.num_episodes}")
    print(f"     Saved to          : {output_path}")
    print(f"{'='*70}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
