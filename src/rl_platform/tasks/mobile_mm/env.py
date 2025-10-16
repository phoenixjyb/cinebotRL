"""Mobile manipulator end-effector tracking environment for Isaac Lab.

This environment trains a mobile manipulator to track a reference trajectory
with its end-effector while avoiding obstacles and maintaining stability.
"""

from __future__ import annotations

import sys
import torch
from dataclasses import dataclass, field
from typing import Any
import gymnasium as gym
import numpy as np

# Isaac Lab imports (correct for Isaac Lab 2.2.0 pip package)
# NOTE: Must import AFTER AppLauncher/SimulationApp is created!
try:
    # Isaac Lab 2.2.0 pip package uses 'isaaclab' not 'omni.isaac.lab'
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg
    from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg
    from isaaclab.utils import configclass
    ISAAC_LAB_AVAILABLE = True
except ImportError as e:
    # Fallback for development/testing without Isaac Sim running
    # When register_isaac_lab_tasks() is called, Isaac Sim will already
    # be running and the imports will succeed on the second pass
    ISAAC_LAB_AVAILABLE = False
    print(f"[env.py] ✗ Failed to import Isaac Lab: {e}", file=sys.stderr)
    DirectRLEnv = None
    DirectRLEnvCfg = None
    configclass = None
    # Create dummy versions for type hints
    class DirectRLEnv:  # type: ignore
        pass
    class DirectRLEnvCfg:  # type: ignore
        pass
    def configclass(cls):  # type: ignore
        return dataclass(cls)

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
    num_envs = 1  # Default to 1, can be overridden
    
    # Task-specific configuration
    task_config: MobileMMTrackConfig = field(default_factory=MobileMMTrackConfig)
    
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
        
        # Define observation space (continuous, normalized)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_observations,),
            dtype=np.float32,
        )
        
        # Define action space (continuous, normalized)
        # Actions: [arm_joint_targets (6), base_vel_x, base_angular_vel]
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_actions,),
            dtype=np.float32,
        )
        
        # Set physics simulation parameters
        # Note: device will be auto-selected by AppLauncher
        # No need to specify here - Isaac Lab will use the device from AppLauncher
        self.sim = SimulationCfg(
            dt=0.005,  # 200 Hz physics
            render_interval=self.decimation,
        )
    
    def _create_scene_config(self) -> InteractiveSceneCfg:
        """Create the scene configuration with robot and environment."""
        print(f"[MobileMMTrackEE] DEBUG: _create_scene_config called with self.num_envs = {self.num_envs}")
        
        # Get robot USD path
        robot_usd_path = str(get_mobile_mm_usd_path())
        
        # Configure robot articulation
        robot_cfg = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",  # USD prim path for robot
            spawn=sim_utils.UsdFileCfg(
                usd_path=robot_usd_path,
                activate_contact_sensors=True,  # Enable for self-collision detection
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
                "arm": ImplicitActuatorCfg(
                    joint_names_expr=["left_arm_joint[1-6]"],
                    stiffness=400.0,
                    damping=40.0,
                ),
            },
        )
        
        # Ground plane
        ground_cfg = AssetBaseCfg(
            prim_path="/World/Ground",  # USD prim path for ground
            spawn=sim_utils.GroundPlaneCfg(),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        )
        
        # Create scene
        scene_cfg = InteractiveSceneCfg(
            num_envs=self.num_envs,  # Use num_envs from parent config
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
    
    def __init__(self, cfg: MobileMMTrackEEEnvCfg | None = None, render_mode: str | None = None, **kwargs):
        """Initialize the environment.
        
        Args:
            cfg: Environment configuration (optional, created if None)
            render_mode: Rendering mode (None for headless) - currently unused
            **kwargs: Additional arguments (num_envs, etc.)
        """
        # Create default config if none provided
        if cfg is None:
            cfg = MobileMMTrackEEEnvCfg()
        
        # CRITICAL: Override num_envs BEFORE calling super().__init__()
        # DirectRLEnv reads cfg.num_envs during initialization to create the scene
        if 'num_envs' in kwargs:
            num_envs_override = kwargs.pop('num_envs')
            print(f"[MobileMMTrackEE] DEBUG: Before override, cfg.num_envs = {cfg.num_envs}")
            cfg.num_envs = num_envs_override
            print(f"[MobileMMTrackEE] DEBUG: After override, cfg.num_envs = {cfg.num_envs}")
            print(f"[MobileMMTrackEE] DEBUG: cfg object id = {id(cfg)}")
        
        # DirectRLEnv only takes cfg, not render_mode
        # By this point, cfg.num_envs should be set correctly
        print(f"[MobileMMTrackEE] DEBUG: Passing cfg with num_envs={cfg.num_envs} to DirectRLEnv")
        super().__init__(cfg, **kwargs)
        
        # Task configuration
        self.task_cfg = cfg.task_config
        
        # Build reward weights dictionary with new constraint penalties
        self.reward_weights = {
            "position_tracking": self.task_cfg.rewards.position_tracking,
            "orientation_tracking": self.task_cfg.rewards.orientation_tracking,
            "progress_bonus": self.task_cfg.rewards.progress_bonus,
            "action_magnitude": self.task_cfg.rewards.action_magnitude,
            "action_rate": self.task_cfg.rewards.action_rate,
            "action_smoothness": self.task_cfg.rewards.action_smoothness,
            "velocity_limit_penalty": self.task_cfg.rewards.velocity_limit_penalty,
            "acceleration_limit_penalty": self.task_cfg.rewards.acceleration_limit_penalty,
            "jerk_limit_penalty": self.task_cfg.rewards.jerk_limit_penalty,
            "joint_limit_penalty": self.task_cfg.rewards.joint_limit_penalty,
            "lateral_motion_penalty": self.task_cfg.rewards.lateral_motion_penalty,
            "self_collision_penalty": self.task_cfg.rewards.self_collision_penalty,
            "self_collision_threshold": self.task_cfg.rewards.self_collision_threshold,
            "self_collision_continuous": self.task_cfg.rewards.self_collision_continuous,
            "collision_penalty": self.task_cfg.rewards.collision_penalty,
            "stability_penalty": self.task_cfg.rewards.stability_penalty,
            "min_obstacle_distance_weight": self.task_cfg.rewards.min_obstacle_distance_weight,
            "safety_radius": self.task_cfg.rewards.safety_radius,
        }
        
        # Robot limits dictionary
        self.robot_limits = {
            "max_linear_velocity": self.task_cfg.robot_limits.max_linear_velocity,
            "max_angular_velocity": self.task_cfg.robot_limits.max_angular_velocity,
            "max_linear_acceleration": self.task_cfg.robot_limits.max_linear_acceleration,
            "max_linear_jerk": self.task_cfg.robot_limits.max_linear_jerk,
            "max_joint_velocity": self.task_cfg.robot_limits.max_joint_velocity,
            "max_joint_acceleration": self.task_cfg.robot_limits.max_joint_acceleration,
            "joint_limit_margin": self.task_cfg.robot_limits.joint_limit_margin,
        }
        
        # Control timestep (for derivative calculations)
        self.control_dt = self.physics_dt * self.cfg.decimation
        
        # Trajectory manager
        self.trajectory_manager = TrajectoryManager(
            traj_type=self.task_cfg.trajectory.type,
            num_envs=self.num_envs,
            device=self.device,
            amplitude=self.task_cfg.trajectory.amplitude,
            speed=self.task_cfg.trajectory.speed,
            height=self.task_cfg.trajectory.height,
            dt=self.control_dt,
        )
        
        # State buffers for tracking history (needed for derivatives)
        self.prev_actions = torch.zeros(
            self.num_envs, self.cfg.num_actions, device=self.device
        )
        self.prev_prev_actions = torch.zeros(
            self.num_envs, self.cfg.num_actions, device=self.device
        )
        self.prev_tracking_error = torch.zeros(self.num_envs, device=self.device)
        
        # Velocity history for acceleration calculation
        self.prev_base_lin_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.prev_joint_vel = torch.zeros(
            self.num_envs, 6, device=self.device  # 6 arm joints
        )
        
        # Acceleration history for jerk calculation
        self.prev_base_accel = torch.zeros(self.num_envs, 3, device=self.device)
        
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
        # End-effector body index (will be set in _setup_scene)
        self._ee_body_idx = None
        
        # Joint limits (will be extracted in _setup_scene)
        self.joint_lower_limits = None
        self.joint_upper_limits = None
        
        print(f"[MobileMMTrackEE] Environment initialized:")
        print(f"  - Num envs: {self.num_envs}")
        print(f"  - Observation dim: {self.cfg.num_observations}")
        print(f"  - Action dim: {self.cfg.num_actions}")
        print(f"  - Trajectory type: {self.task_cfg.trajectory.type}")
        print(f"  - Episode length: {self.max_episode_length} steps")
        print(f"  - Control frequency: {1.0 / self.control_dt:.1f} Hz")
        print(f"  - Trajectory dt: {self.task_cfg.trajectory_dt:.3f}s")
    
    def _setup_scene(self):
        """Setup the scene entities."""
        # Get robot articulation
        self.robot = self.scene["robot"]
        
        # Joint limits will be extracted lazily when needed (after first reset)
        self.joint_lower_limits = None
        self.joint_upper_limits = None
        self._joint_limits_initialized = False
        
        # End-effector body index will be found lazily
        self._ee_body_idx = None
        self._ee_body_idx_initialized = False
        
        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
    
    def _initialize_ee_body_idx(self):
        """Initialize end-effector body index (lazy initialization)."""
        if not self._ee_body_idx_initialized and hasattr(self.robot, '_root_physx_view'):
            ee_link_name = "left_gripper_link"
            if ee_link_name in self.robot.body_names:
                self._ee_body_idx = self.robot.body_names.index(ee_link_name)
                print(f"[MobileMMTrackEE] Found EE link '{ee_link_name}' at index {self._ee_body_idx}")
            else:
                # Fallback to last body
                self._ee_body_idx = -1
                print(f"[MobileMMTrackEE] WARNING: EE link '{ee_link_name}' not found, using last body")
            self._ee_body_idx_initialized = True
    
    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions before physics step.
        
        Args:
            actions: Actions from policy [num_envs, num_actions]
        """
        # DEBUG: Print actual action shape on first call
        if not hasattr(self, '_first_action_printed'):
            print(f"[MobileMMTrackEE] DEBUG: First action shape = {actions.shape}")
            print(f"[MobileMMTrackEE] DEBUG: self.num_envs = {self.num_envs}")
            print(f"[MobileMMTrackEE] DEBUG: Expected shape = [{self.num_envs}, {self.cfg.num_actions}]")
            self._first_action_printed = True
        
        # Ensure actions are 2D [num_envs, num_actions]
        # Sometimes actions come in as 3D [1, 1, 8] - squeeze to [1, 8]
        while actions.ndim > 2:
            if actions.shape[0] == 1:
                actions = actions.squeeze(0)
            elif actions.shape[1] == 1:
                actions = actions.squeeze(1)
            else:
                break
        
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)
        
        # Update action history for derivative calculations (jerk/smoothness)
        # Store 3 timesteps: current, t-1, t-2
        if not hasattr(self, '_actions_t_minus_2'):
            self._actions_t_minus_2 = torch.zeros_like(actions)
        
        self._actions_t_minus_2 = self.prev_prev_actions.clone()
        self.prev_prev_actions = self.prev_actions.clone()
        self.prev_actions = actions.clone()
        
        # Update action history buffer
        if self.action_history is not None:
            # Shift history and append new action
            self.action_history = torch.roll(self.action_history, shifts=-1, dims=1)
            self.action_history[:, -1, :] = actions
        
        # Apply actions to robot
        # Action space is 8D: [6 arm joint positions, vx, wz]
        # Robot has 9 DOF: [6 arm joints, 3 chassis: vx, vy, wz]
        # Differential drive can't move sideways, so vy is always 0
        
        # Split actions: first 6 are arm joints, last 2 are base commands (vx, wz)
        arm_actions = actions[:, :6]  # First 6: arm joint position targets (in [-1, 1])
        base_vx = actions[:, 6:7]     # vx: forward/backward velocity (in [-1, 1])
        base_wz = actions[:, 7:8]     # wz: angular velocity/rotation (in [-1, 1])
        
        # Scale arm actions from [-1, 1] to actual joint limits with safety margins
        arm_actions_scaled = self._scale_actions_to_joint_limits(arm_actions)
        
        # Apply arm joint position targets to the actuated arm joints only
        # The robot has 9 total joints: 6 arm + 3 chassis (vx, vy, wz)
        # We only control the 6 arm joints via position targets
        if not hasattr(self, '_arm_joint_ids'):
            # Find indices of arm joints (left_arm_joint1 through left_arm_joint6)
            arm_joint_names = [f"left_arm_joint{i}" for i in range(1, 7)]
            self._arm_joint_ids = []
            for name in arm_joint_names:
                if name in self.robot.joint_names:
                    idx = self.robot.joint_names.index(name)
                    self._arm_joint_ids.append(idx)
            self._arm_joint_ids = torch.tensor(self._arm_joint_ids, device=self.device)
        
        # Set scaled joint position targets for arm joints only
        self.robot.set_joint_position_target(arm_actions_scaled, joint_ids=self._arm_joint_ids)
        
        # Apply base velocity commands for mobile base (differential drive)
        # Initialize base joint indices on first call
        if not hasattr(self, '_base_joint_ids'):
            # Base joints: first 3 joints are typically base_link_x, base_link_y, base_link_z (or similar)
            # For differential drive: we control vx (forward), and wz (rotation), vy is always 0
            # Joint indices [0, 1, 2] correspond to [vx, vy, wz]
            self._base_joint_ids = torch.tensor([0, 1, 2], device=self.device)
            print(f"[MobileMMTrackEE] Base joint IDs initialized: {self._base_joint_ids.tolist()}")
        
        # Create base velocity command: [vx, 0, wz]
        # vy = 0 because differential drive cannot move sideways
        base_velocities = torch.cat([
            base_vx,                      # Forward/backward velocity
            torch.zeros_like(base_vx),    # vy = 0 (no sideways movement)
            base_wz                        # Angular velocity (rotation)
        ], dim=-1)
        
        # Apply velocity targets to base joints
        self.robot.set_joint_velocity_target(
            target=base_velocities,  # Correct parameter name is 'target'
            joint_ids=self._base_joint_ids
        )
    
    def _apply_action(self):
        """Apply actions to the simulation (called by parent)."""
        pass  # Actions applied in _pre_physics_step
    
    def _scale_actions_to_joint_limits(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Scale normalized actions from [-1, 1] to actual joint limits with safety margins.
        
        Args:
            actions: Normalized actions in [-1, 1], shape [num_envs, num_joints]
            
        Returns:
            Scaled actions in physical joint limits, shape [num_envs, num_joints]
        """
        # Ensure joint limits are initialized
        self._initialize_joint_limits()
        
        # Get joint limits (these are already for arm joints only)
        lower = self.joint_lower_limits  # Shape: [6]
        upper = self.joint_upper_limits  # Shape: [6]
        
        # Add safety margin (5% from each limit to avoid hard stops)
        range_size = upper - lower
        safety_margin = 0.05 * range_size
        lower_safe = lower + safety_margin
        upper_safe = upper - safety_margin
        
        # Scale from [-1, 1] to [lower_safe, upper_safe]
        # Formula: scaled = (action + 1) * 0.5 * (upper - lower) + lower
        actions_normalized = (actions + 1.0) * 0.5  # Convert [-1, 1] to [0, 1]
        scaled_actions = actions_normalized * (upper_safe - lower_safe) + lower_safe
        
        return scaled_actions
    
    def _initialize_joint_limits(self):
        """Initialize joint limits from robot data (lazy initialization)."""
        if not self._joint_limits_initialized and hasattr(self.robot, 'data'):
            # Extract joint limits from robot data (arm joints only, first 6)
            self.joint_lower_limits = self.robot.data.soft_joint_pos_limits[0, :6, 0]
            self.joint_upper_limits = self.robot.data.soft_joint_pos_limits[0, :6, 1]
            self._joint_limits_initialized = True
            print(f"[MobileMMTrackEE] Joint limits initialized:")
            print(f"  Lower: {self.joint_lower_limits}")
            print(f"  Upper: {self.joint_upper_limits}")
    
    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Compute environment observations.
        
        Returns:
            Dictionary with "policy" key containing observation tensor
        """
        # Initialize joint limits and EE body index on first call
        self._initialize_joint_limits()
        self._initialize_ee_body_idx()
        
        # Robot state
        base_pos, base_quat = self.robot.data.root_pos_w, self.robot.data.root_quat_w
        base_lin_vel = self.robot.data.root_lin_vel_w
        base_ang_vel = self.robot.data.root_ang_vel_w
        joint_pos = self.robot.data.joint_pos
        joint_vel = self.robot.data.joint_vel
        
        # Get end-effector state
        ee_pos = self.robot.data.body_pos_w[:, self._ee_body_idx, :]
        ee_quat = self.robot.data.body_quat_w[:, self._ee_body_idx, :]
        ee_lin_vel = self.robot.data.body_lin_vel_w[:, self._ee_body_idx, :]
        ee_ang_vel = self.robot.data.body_ang_vel_w[:, self._ee_body_idx, :]
        
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
        ee_pos = self.robot.data.body_pos_w[:, self._ee_body_idx, :]
        ee_quat = self.robot.data.body_quat_w[:, self._ee_body_idx, :]
        
        # Get target pose
        target_pos, target_quat = self.trajectory_manager.get_target_pose()
        
        # Get robot state
        base_lin_vel = self.robot.data.root_lin_vel_w
        base_ang_vel = self.robot.data.root_ang_vel_w
        joint_pos = self.robot.data.joint_pos[:, :6]  # First 6 joints (arm)
        joint_vel = self.robot.data.joint_vel[:, :6]
        
        # Get contact forces for self-collision detection
        # Isaac Lab 2.2.0 provides contact forces via PhysX view
        try:
            # Try to get net contact forces from PhysX view
            net_contact_forces = self.robot.root_physx_view.get_net_contact_forces()
        except AttributeError:
            # Fallback: try body_net_contact_force_w from robot data
            try:
                net_contact_forces = self.robot.data.body_net_contact_force_w
            except AttributeError:
                # Last resort: use zeros but warn once
                if not hasattr(self, '_contact_force_warning_shown'):
                    print("[WARNING] Contact forces API not found - collision detection disabled!")
                    self._contact_force_warning_shown = True
                net_contact_forces = torch.zeros(
                    (self.num_envs, len(self.robot.body_names), 3),
                    device=self.device
                )
        
        # Compute acceleration for current step
        base_accel = (base_lin_vel - self.prev_base_lin_vel) / self.control_dt
        
        # Compute rewards with all new constraint penalties
        rewards, self.reward_components = compute_combined_reward(
            current_ee_pos=ee_pos,
            current_ee_quat=ee_quat,
            target_pos=target_pos,
            target_quat=target_quat,
            prev_tracking_error=self.prev_tracking_error,
            actions=self.prev_actions,  # Current actions (just applied)
            prev_actions=self.prev_prev_actions,  # Actions from previous step
            prev_prev_actions=self._actions_t_minus_2,  # Actions from 2 steps ago (for jerk calculation)
            base_lin_vel=base_lin_vel,
            base_ang_vel=base_ang_vel,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            prev_base_lin_vel=self.prev_base_lin_vel,
            prev_joint_vel=self.prev_joint_vel,
            prev_base_accel=self.prev_base_accel,
            joint_lower=self.joint_lower_limits,
            joint_upper=self.joint_upper_limits,
            robot_limits=self.robot_limits,
            contact_forces=net_contact_forces,  # Use actual contact forces for self-collision
            min_obstacle_dist=None,  # Not using obstacles for now
            dt=self.control_dt,
            weights=self.reward_weights,
        )
        
        # Update history for next step
        self.prev_tracking_error = torch.norm(target_pos - ee_pos, dim=-1)
        self.prev_base_lin_vel = base_lin_vel.clone()
        self.prev_joint_vel = joint_vel.clone()
        self.prev_base_accel = base_accel.clone()
        
        # Advance trajectory to next target
        # This must happen after reward calculation so current target is used for this step
        self.trajectory_manager.step()
        
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
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Check for excessive tracking error
        if self.task_cfg.terminate_on_tracking_error:
            ee_pos = self.robot.data.body_pos_w[:, self._ee_body_idx, :]
            target_pos, _ = self.trajectory_manager.get_target_pose()
            tracking_error = torch.norm(target_pos - ee_pos, dim=-1)
            terminated |= tracking_error > self.task_cfg.max_tracking_error
        
        # Check for self-collision (CRITICAL for mobile manipulator!)
        if self.task_cfg.terminate_on_self_collision:
            # Get contact forces - use same method as in _get_rewards()
            try:
                net_contact_forces = self.robot.root_physx_view.get_net_contact_forces()
            except AttributeError:
                try:
                    net_contact_forces = self.robot.data.body_net_contact_force_w
                except AttributeError:
                    # If API not available, skip collision termination
                    net_contact_forces = None
            
            if net_contact_forces is not None:
                # Calculate maximum contact force magnitude per environment
                contact_force_mag = torch.norm(net_contact_forces, dim=-1)  # [num_envs, num_bodies]
                max_contact_force = torch.max(contact_force_mag, dim=-1)[0]  # [num_envs]
                terminated |= max_contact_force > self.task_cfg.self_collision_termination_threshold
        
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
        self.prev_prev_actions[env_ids] = 0.0
        self.prev_tracking_error[env_ids] = 0.0
        self.prev_base_lin_vel[env_ids] = 0.0
        self.prev_joint_vel[env_ids] = 0.0
        self.prev_base_accel[env_ids] = 0.0
        
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
