"""Mobile manipulator end-effector tracking environment for Isaac Lab.

This environment trains a mobile manipulator to track a reference trajectory
with its end-effector while avoiding obstacles and maintaining stability.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Any

# Isaac Lab imports (these will be available when running in Isaac Lab)
try:
    import omni.isaac.lab.sim as sim_utils
    from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
    from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
    from omni.isaac.lab.scene import InteractiveSceneCfg
    from omni.isaac.lab.sim import SimulationCfg
    from omni.isaac.lab.utils import configclass
    ISAAC_LAB_AVAILABLE = True
except ImportError:
    # Fallback for development without Isaac Lab
    ISAAC_LAB_AVAILABLE = False
    DirectRLEnv = object
    DirectRLEnvCfg = object
    configclass = lambda cls: dataclass(cls)

from rl_platform.robots.mobile_mm import get_mobile_mm_usd_path
from .config import MobileMMTrackConfig, RewardWeights
from .trajectories import TrajectoryManager
from .observations import compose_observation, get_observation_dimensions
from .rewards import compute_combined_reward


@configclass
class MobileMMTrackEEEnvCfg(DirectRLEnvCfg):
    """Configuration for the mobile manipulator tracking environment."""
    
    # Simulation settings
    decimation = 4
    episode_length_s = 20.0
    
    # Task-specific configuration
    task_config: MobileMMTrackConfig = MobileMMTrackConfig()
    
    # Scene configuration (will be populated in __post_init__)
    scene: InteractiveSceneCfg = None
    
    # Action/Observation spaces (computed based on robot)
    num_actions: int = 8  # 6 arm joints + 2 base DOF (v_x, omega_z)
    num_observations: int = 46  # Will be computed based on config
    
    def __post_init__(self):
        """Initialize scene configuration and compute dimensions."""
        if not ISAAC_LAB_AVAILABLE:
            return
            
        # Create scene configuration
        self.scene = self._create_scene_config()
        
        # Compute observation dimension
        self.num_observations = get_observation_dimensions(
            num_joints=6,  # Arm joints
            num_contacts=0,  # No contact sensors initially
            use_lookahead=self.task_config.use_lookahead,
            lookahead_steps=self.task_config.lookahead_steps,
            use_action_history=self.task_config.include_action_history,
            action_history_length=self.task_config.action_history_length,
            action_dim=self.num_actions,
            use_obstacles=self.task_config.obstacles.enable_obstacles,
        )
        
        # Set physics simulation parameters
        self.sim = SimulationCfg(
            dt=0.005,  # 200 Hz physics
            render_interval=self.decimation,
        )
    
    def _create_scene_config(self) -> InteractiveSceneCfg:
        """Create the scene configuration with robot and environment."""
        
        # Get robot USD path
        robot_usd_path = str(get_mobile_mm_usd_path())
        
        # Configure robot articulation
        robot_cfg = ArticulationCfg(
            spawn=sim_utils.UsdFileCfg(
                usd_path=robot_usd_path,
                activate_contact_sensors=False,
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                joint_pos={
                    # Arm joints (mid-range initialization)
                    "left_arm_joint1": 0.0,
                    "left_arm_joint2": 1.6,
                    "left_arm_joint3": -1.6,
                    "left_arm_joint4": 0.0,
                    "left_arm_joint5": 0.0,
                    "left_arm_joint6": 0.0,
                },
            ),
            actuators={
                "arm": sim_utils.ImplicitActuatorCfg(
                    joint_names_expr=["left_arm_joint[1-6]"],
                    stiffness=400.0,
                    damping=40.0,
                ),
            },
        )
        
        # Ground plane
        ground_cfg = AssetBaseCfg(
            spawn=sim_utils.GroundPlaneCfg(),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        )
        
        # Create scene
        scene_cfg = InteractiveSceneCfg(
            num_envs=1024,  # Will be overridden by training config
            env_spacing=4.0,
            replicate_physics=True,
        )
        
        # Add assets to scene
        scene_cfg.robot = robot_cfg
        scene_cfg.ground = ground_cfg
        
        return scene_cfg


class MobileMMTrackEEEnv(DirectRLEnv):
    """Mobile manipulator end-effector tracking environment.
    
    This environment trains the robot to track a reference trajectory with
    its end-effector while maintaining stability and avoiding obstacles.
    """
    
    cfg: MobileMMTrackEEEnvCfg
    
    def __init__(self, cfg: MobileMMTrackEEEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the environment.
        
        Args:
            cfg: Environment configuration
            render_mode: Rendering mode (None for headless)
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(cfg, render_mode, **kwargs)
        
        # Task configuration
        self.task_cfg = cfg.task_config
        self.reward_weights = {
            "position_tracking": self.task_cfg.rewards.position_tracking,
            "orientation_tracking": self.task_cfg.rewards.orientation_tracking,
            "progress_bonus": self.task_cfg.rewards.progress_bonus,
            "action_magnitude": self.task_cfg.rewards.action_magnitude,
            "action_rate": self.task_cfg.rewards.action_rate,
            "collision_penalty": self.task_cfg.rewards.collision_penalty,
            "stability_penalty": self.task_cfg.rewards.stability_penalty,
            "min_obstacle_distance_weight": self.task_cfg.rewards.min_obstacle_distance_weight,
            "safety_radius": self.task_cfg.rewards.safety_radius,
        }
        
        # Trajectory manager
        self.trajectory_manager = TrajectoryManager(
            traj_type=self.task_cfg.trajectory.type,
            num_envs=self.num_envs,
            device=self.device,
            amplitude=self.task_cfg.trajectory.amplitude,
            speed=self.task_cfg.trajectory.speed,
            height=self.task_cfg.trajectory.height,
            dt=self.physics_dt * self.cfg.decimation,
        )
        
        # State buffers
        self.prev_actions = torch.zeros(
            self.num_envs, self.cfg.num_actions, device=self.device
        )
        self.prev_tracking_error = torch.zeros(self.num_envs, device=self.device)
        
        # Action history buffer
        if self.task_cfg.include_action_history:
            self.action_history = torch.zeros(
                self.num_envs,
                self.task_cfg.action_history_length,
                self.cfg.num_actions,
                device=self.device,
            )
        else:
            self.action_history = None
        
        # Reward component tracking for logging
        self.reward_components = {}
        
        print(f"[MobileMMTrackEE] Environment initialized:")
        print(f"  - Num envs: {self.num_envs}")
        print(f"  - Observation dim: {self.cfg.num_observations}")
        print(f"  - Action dim: {self.cfg.num_actions}")
        print(f"  - Trajectory type: {self.task_cfg.trajectory.type}")
        print(f"  - Episode length: {self.max_episode_length} steps")
    
    def _setup_scene(self):
        """Setup the scene entities."""
        # Get robot articulation
        self.robot = self.scene["robot"]
        
        # Clone ground plane
        self.scene.clone_environments(copy_from_source=False)
        
        # Add lights if not headless
        if self.sim.render_mode != "headless":
            self.scene.add_default_ground_plane()
    
    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions before physics step.
        
        Args:
            actions: Actions from policy [num_envs, num_actions]
        """
        # Store previous actions
        self.prev_actions = actions.clone()
        
        # Update action history
        if self.action_history is not None:
            # Shift history and append new action
            self.action_history = torch.roll(self.action_history, shifts=-1, dims=1)
            self.action_history[:, -1, :] = actions
        
        # Apply actions to robot
        # For now, directly set joint position targets (will be refined)
        self.robot.set_joint_position_target(actions)
    
    def _apply_action(self):
        """Apply actions to the simulation (called by parent)."""
        pass  # Actions applied in _pre_physics_step
    
    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Compute observations for all environments.
        
        Returns:
            Dictionary with "policy" key containing observations
        """
        # Get robot state
        base_pos, base_quat = self.robot.data.root_pos_w, self.robot.data.root_quat_w
        base_lin_vel = self.robot.data.root_lin_vel_w
        base_ang_vel = self.robot.data.root_ang_vel_w
        joint_pos = self.robot.data.joint_pos
        joint_vel = self.robot.data.joint_vel
        
        # Get end-effector state (assuming last link is EE)
        # This will need to be updated based on actual robot structure
        ee_pos = self.robot.data.body_pos_w[:, -1, :]
        ee_quat = self.robot.data.body_quat_w[:, -1, :]
        ee_lin_vel = self.robot.data.body_lin_vel_w[:, -1, :]
        ee_ang_vel = self.robot.data.body_ang_vel_w[:, -1, :]
        
        # Get target from trajectory
        target_pos, target_quat = self.trajectory_manager.get_target_pose()
        
        # Optional: Lookahead
        lookahead_pos = None
        if self.task_cfg.use_lookahead:
            lookahead_pos, _ = self.trajectory_manager.get_lookahead(
                steps=self.task_cfg.lookahead_steps,
                lookahead_dt=self.task_cfg.lookahead_dt,
            )
        
        # Compose full observation
        obs = compose_observation(
            base_pos=base_pos,
            base_quat=base_quat,
            base_lin_vel=base_lin_vel,
            base_ang_vel=base_ang_vel,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            ee_pos=ee_pos,
            ee_quat=ee_quat,
            ee_lin_vel=ee_lin_vel,
            ee_ang_vel=ee_ang_vel,
            target_pos=target_pos,
            target_quat=target_quat,
            lookahead_pos=lookahead_pos,
            action_history=self.action_history,
            contact_forces=None,  # TODO: Add contact sensors
            min_obstacle_dist=None,  # TODO: Add obstacle distance computation
        )
        
        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards for all environments.
        
        Returns:
            Reward tensor [num_envs]
        """
        # Get current EE pose
        ee_pos = self.robot.data.body_pos_w[:, -1, :]
        ee_quat = self.robot.data.body_quat_w[:, -1, :]
        
        # Get target pose
        target_pos, target_quat = self.trajectory_manager.get_target_pose()
        
        # Get base velocities
        base_lin_vel = self.robot.data.root_lin_vel_w
        base_ang_vel = self.robot.data.root_ang_vel_w
        
        # Compute rewards
        rewards, self.reward_components = compute_combined_reward(
            current_ee_pos=ee_pos,
            current_ee_quat=ee_quat,
            target_pos=target_pos,
            target_quat=target_quat,
            prev_tracking_error=self.prev_tracking_error,
            actions=self.prev_actions,
            prev_actions=self.prev_actions,  # TODO: Store actual previous
            contact_forces=torch.zeros(self.num_envs, 1, device=self.device),  # Placeholder
            base_lin_vel=base_lin_vel,
            base_ang_vel=base_ang_vel,
            min_obstacle_dist=None,
            weights=self.reward_weights,
        )
        
        # Update tracking error for next step
        self.prev_tracking_error = torch.norm(target_pos - ee_pos, dim=-1)
        
        # Log reward components
        self.extras["reward_components"] = {
            k: v.mean().item() for k, v in self.reward_components.items()
        }
        
        return rewards
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute termination and timeout conditions.
        
        Returns:
            terminated: Environments that should terminate [num_envs]
            time_out: Environments that reached max episode length [num_envs]
        """
        # Check for excessive tracking error
        ee_pos = self.robot.data.body_pos_w[:, -1, :]
        target_pos, _ = self.trajectory_manager.get_target_pose()
        tracking_error = torch.norm(target_pos - ee_pos, dim=-1)
        
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        if self.task_cfg.terminate_on_tracking_error:
            terminated |= tracking_error > self.task_cfg.max_tracking_error
        
        # Timeout after max episode length
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        return terminated, time_out
    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset specified environments.
        
        Args:
            env_ids: Indices of environments to reset
        """
        if env_ids is None or len(env_ids) == 0:
            return
        
        super()._reset_idx(env_ids)
        
        # Reset trajectory phase
        self.trajectory_manager.reset(env_ids)
        
        # Reset state buffers
        self.prev_actions[env_ids] = 0.0
        self.prev_tracking_error[env_ids] = 0.0
        
        if self.action_history is not None:
            self.action_history[env_ids] = 0.0
        
        # Randomize initial joint positions
        if self.task_cfg.randomize_initial_joint_positions:
            noise = torch.randn(
                len(env_ids), self.robot.num_joints, device=self.device
            ) * self.task_cfg.initial_joint_noise_std
            
            default_joint_pos = self.robot.data.default_joint_pos[env_ids]
            self.robot.set_joint_position_target(
                default_joint_pos + noise, env_ids=env_ids
            )
        
        # Advance trajectory by one step to start
        self.trajectory_manager.step()
