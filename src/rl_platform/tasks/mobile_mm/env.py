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
    # Debug visualization
    from isaaclab.markers import VisualizationMarkers
    from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, RED_ARROW_X_MARKER_CFG
    import isaaclab.utils.math as math_utils
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
    decimation = 10  # 200Hz physics / 10 = 20Hz control (realistic for mobile manipulators)
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
                "base": ImplicitActuatorCfg(
                    joint_names_expr=["joint_x", "joint_y", "joint_theta"],
                    stiffness=1000.0,   # k=1000 N/m → ω_n=31.6 rad/s (5Hz, controllable at 20Hz)
                    damping=316.0,      # ζ=0.5 underdamped for 20Hz control (96% in 1 step!)
                    effort_limit=1000.0,  # Override URDF effort=0
                    velocity_limit=2.0,   # Override URDF velocity=0
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
            **kwargs: Additional arguments that may override configuration.
        """
        print(f"\n{'='*70}")
        print(f"[MobileMMTrackEE] __init__ called:")
        print(f"  cfg provided: {cfg is not None}")
        if cfg is not None:
            print(f"  cfg.task_config.trajectory.type: {cfg.task_config.trajectory.type}")
            print(f"  cfg.task_config.trajectory.trajectory_dir: {cfg.task_config.trajectory.trajectory_dir}")
        print(f"  kwargs: {list(kwargs.keys())}")
        print(f"{'='*70}\n")
        
        # Extract overrides before building config
        num_envs_override = kwargs.pop("num_envs", None)
        trajectory_type_override = kwargs.pop("trajectory_type", None)
        trajectory_dir_override = kwargs.pop("trajectory_dir", None)
        trajectory_pattern_override = kwargs.pop("trajectory_pattern", None)
        trajectory_filter_override = kwargs.pop("trajectory_filter_indices", None)
        max_trajectories_override = kwargs.pop("max_trajectories", None)
        waypoint_file_override = kwargs.pop("waypoint_file", None)
        use_all_trajectories = kwargs.pop("use_all_trajectories", None)
        use_chassis_only = kwargs.pop("use_chassis_only", None)
        
        print(f"[MobileMMTrackEE] DEBUG: Before config handling, num_envs_override={num_envs_override}")
        
        if cfg is None:
            from dataclasses import replace
            cfg = MobileMMTrackEEEnvCfg()
            if num_envs_override is not None:
                cfg = replace(cfg, num_envs=num_envs_override)
                print(f"[MobileMMTrackEE] Created config with num_envs={cfg.num_envs}")
        else:
            if num_envs_override is not None:
                cfg.num_envs = num_envs_override
                print(f"[MobileMMTrackEE] Updated existing config to num_envs={cfg.num_envs}")
        
        if cfg.scene is not None and num_envs_override is not None:
            cfg.scene.num_envs = num_envs_override
        
        # Apply trajectory overrides (works for provided or auto-created cfg)
        traj_cfg = cfg.task_config.trajectory
        
        if trajectory_type_override is not None:
            traj_cfg.type = trajectory_type_override
        if trajectory_dir_override is not None:
            traj_cfg.trajectory_dir = trajectory_dir_override
        if trajectory_pattern_override is not None:
            traj_cfg.trajectory_pattern = trajectory_pattern_override
        if max_trajectories_override is not None:
            traj_cfg.max_trajectories = max_trajectories_override
        if waypoint_file_override is not None:
            traj_cfg.waypoint_file = waypoint_file_override
        
        if trajectory_filter_override is not None:
            traj_cfg.trajectory_filter_indices = trajectory_filter_override
        elif use_chassis_only and traj_cfg.type == "multi_recorded":
            traj_cfg.trajectory_filter_indices = self._load_chassis_required_indices(max_trajectories_override)
        elif use_all_trajectories:
            traj_cfg.trajectory_filter_indices = None
        
        print(f"[MobileMMTrackEE] DEBUG: About to call super().__init__() with cfg.num_envs={cfg.num_envs if cfg else 'None'}")
        
        # DirectRLEnv only takes cfg, not render_mode
        super().__init__(cfg, **kwargs)
        
        print(f"[MobileMMTrackEE] DEBUG: After super().__init__(), self.num_envs={self.num_envs}")
        
        # ============================================================================
        # COORDINATE FRAME CONVENTIONS
        # ============================================================================
        # This environment uses TWO coordinate systems:
        #
        # 1. WORLD FRAME (Isaac "_w" buffers) - GROUND TRUTH for learning & rewards
        #    - Source: root_pos_w, root_quat_w, root_lin_vel_w, root_ang_vel_w, body_*_w
        #    - Used for: observations, rewards, termination, logging
        #    - This is maintained by PhysX and reflects actual simulated state
        #
        # 2. PPR JOINT FRAME (joint_x, joint_y, joint_theta) - CONTROL INTERFACE
        #    - Source: joint_pos[:, 0:3] = [joint_x, joint_y, joint_theta]
        #    - Used for: commanding base movement (_pre_physics_step)
        #    - We write position targets here, PhysX integrates to root_pos_w
        #
        # WHY THIS SPLIT:
        # - PPR joints are HOW we command (relative position targets)
        # - root_pos_w is WHAT we get (actual world position after physics)
        # - Using root_pos_w for rewards ensures we reward actual movement,
        #   not just command changes
        #
        # CRITICAL: Never mix frames! EE (body_pos_w) and base (root_pos_w) are
        # both in world frame, so "EE relative to base" = body_pos_w - root_pos_w
        # ============================================================================
        
        # Task configuration
        self.task_cfg = cfg.task_config
        
        # Build reward weights dictionary with new constraint penalties
        self.reward_weights = {
            "position_tracking": self.task_cfg.rewards.position_tracking,
            "orientation_tracking": self.task_cfg.rewards.orientation_tracking,
            "progress_bonus": self.task_cfg.rewards.progress_bonus,
            "base_progress_reward": self.task_cfg.rewards.base_progress_reward,
            "target_distance_penalty": self.task_cfg.rewards.target_distance_penalty,
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
            "max_angular_acceleration": self.task_cfg.robot_limits.max_angular_acceleration,
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
            waypoint_dt=self.task_cfg.trajectory_dt,
            waypoint_file=self.task_cfg.trajectory.waypoint_file,
            trajectory_dir=self.task_cfg.trajectory.trajectory_dir,
            trajectory_pattern=self.task_cfg.trajectory.trajectory_pattern,
            trajectory_filter_indices=self.task_cfg.trajectory.trajectory_filter_indices,
            max_trajectories=self.task_cfg.trajectory.max_trajectories,
        )
        
        # State buffers for tracking history (needed for derivatives)
        self.prev_actions = torch.zeros(
            self.num_envs, self.cfg.num_actions, device=self.device
        )
        self.prev_prev_actions = torch.zeros(
            self.num_envs, self.cfg.num_actions, device=self.device
        )
        self.prev_tracking_error = torch.zeros(self.num_envs, device=self.device)
        
        # Position history for base progress tracking
        self.prev_base_pos = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Velocity history for acceleration calculation
        self.prev_base_lin_vel = torch.zeros(self.num_envs, 3, device=self.device)
        print(f"[MobileMMTrackEE] DEBUG: Initializing velocity buffers with num_envs={self.num_envs}")
        self.current_commanded_vel = torch.zeros(self.num_envs, 3, device=self.device)  # Rate-limited commanded velocities for THIS step
        self.prev_commanded_vel = torch.zeros(self.num_envs, 3, device=self.device)
        print(f"[MobileMMTrackEE] DEBUG: current_commanded_vel.shape = {self.current_commanded_vel.shape}")
        self.prev_commanded_accel = torch.zeros(self.num_envs, 3, device=self.device)
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
        
        # Visualization markers (initialized in _setup_scene)
        self._target_markers = None
        self._ee_markers = None
        self._error_line_markers = None
        self._visualization_enabled = False
        
        print(f"[MobileMMTrackEE] Environment initialized:")
        print(f"  - Num envs: {self.num_envs}")
        print(f"  - Observation dim: {self.cfg.num_observations}")
        print(f"  - Action dim: {self.cfg.num_actions}")
        print(f"  - Trajectory type: {self.task_cfg.trajectory.type}")
        print(f"  - Episode length: {self.max_episode_length} steps")
        print(f"  - Control frequency: {1.0 / self.control_dt:.1f} Hz")
        print(f"  - Trajectory dt: {self.task_cfg.trajectory_dt:.3f}s")

    @staticmethod
    def _load_chassis_required_indices(limit: int | None = None) -> list[int] | None:
        """Load chassis-required trajectory indices from known analysis files."""
        try:
            import json
            import re
            from pathlib import Path

            indices: list[int] | None = None

            txt_file = Path("chassis_required_indices.txt")
            if txt_file.exists():
                content = txt_file.read_text()
                match = re.search(r"CHASSIS_REQUIRED_INDICES\s*=\s*\[(.*?)\]", content, re.DOTALL)
                if match:
                    cleaned = match.group(1).replace("\n", " ")
                    parsed = [int(x.strip()) for x in cleaned.split(",") if x.strip()]
                    if limit is not None:
                        parsed = parsed[:limit]
                    print(f"[MobileMMTrackEE] Loaded {len(parsed)} chassis-required indices from {txt_file}")
                    return parsed

            json_file = Path("trajectoryToLearn/trajectory_analysis.json")
            if json_file.exists():
                data = json.loads(json_file.read_text())
                parsed = data.get("chassis_requiring_indices")
                if isinstance(parsed, list):
                    parsed_int = [int(x) for x in parsed]
                    if limit is not None:
                        parsed_int = parsed_int[:limit]
                    print(f"[MobileMMTrackEE] Loaded {len(parsed_int)} chassis-required indices from {json_file}")
                    return parsed_int

        except Exception as exc:
            print(f"[MobileMMTrackEE] WARNING: Unable to load chassis-required indices: {exc}")

        print("[MobileMMTrackEE] WARNING: use_chassis_only requested but no chassis indices found. Using all trajectories.")
        return None
    
    def _setup_scene(self):
        """Setup the scene entities."""
        # Get robot articulation
        self.robot = self.scene["robot"]
        
        # Joint mapping verification will happen lazily after PhysX view is initialized
        self._joint_mapping_verified = False
        
        # Joint limits will be extracted lazily when needed (after first reset)
        self.joint_lower_limits = None
        self.joint_upper_limits = None
        self._joint_limits_initialized = False
        
        # End-effector body index will be found lazily
        self._ee_body_idx = None
        self._ee_body_idx_initialized = False
        
        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        
        # Setup visualization markers (only in GUI mode)
        self._setup_visualization_markers()
    
    def _setup_visualization_markers(self):
        """Setup visual markers for trajectory visualization."""
        # For now, disable visual markers and use console output
        # TODO: Fix marker visualization in Isaac Lab 0.46.2
        self._visualization_enabled = False
        print("[MobileMMTrackEE] ℹ Visual markers disabled - using console output for trajectory tracking")
    
    def _update_visualization_markers(self, ee_pos: torch.Tensor, target_pos: torch.Tensor):
        """Update visualization markers for trajectory tracking.
        
        Args:
            ee_pos: End-effector positions [num_envs, 3]
            target_pos: Target positions [num_envs, 3]
        """
        if not self._visualization_enabled or self._target_markers is None:
            # Print console output every 50 steps for env 0
            if not hasattr(self, '_vis_step_count'):
                self._vis_step_count = 0
            self._vis_step_count += 1
            
            if self._vis_step_count % 50 == 0:
                tracking_error = torch.norm(target_pos - ee_pos, dim=-1)
                
                # Get base position (using root_pos_w - actual world position)
                base_pos_world = self.robot.data.root_pos_w
                
                # Also show PPR joints for comparison (these are relative offsets)
                base_ppr = self.robot.data.joint_pos[:, 0:3]  # [joint_x, joint_y, joint_theta]
                
                base_to_target_2d = torch.norm(target_pos[:, :2] - base_pos_world[:, :2], dim=-1)
                
                # Get trajectory interpolation info (if available)
                traj_info = ""
                if hasattr(self.trajectory_manager, 'current_waypoint_idx') and \
                   hasattr(self.trajectory_manager, '_recorded_time_accum') and \
                   hasattr(self.trajectory_manager, 'waypoint_dt'):
                    current_wp = self.trajectory_manager.current_waypoint_idx[0].item()
                    time_accum = self.trajectory_manager._recorded_time_accum[0].item()
                    wp_dt = self.trajectory_manager.waypoint_dt
                    alpha = min(time_accum / wp_dt, 1.0)
                    traj_info = f"\n  🎬 Waypoint: {current_wp} → {current_wp + 1} (α={alpha:.2f}, {time_accum*1000:.0f}ms/{wp_dt*1000:.0f}ms)"
                
                print(f"\n[TRACKING Step {self._vis_step_count}] Env 0:{traj_info}")
                print(f"  🎯 Target (WORLD):  [{target_pos[0, 0]:.3f}, {target_pos[0, 1]:.3f}, {target_pos[0, 2]:.3f}]")
                print(f"  🟢 EE Pos (WORLD):  [{ee_pos[0, 0]:.3f}, {ee_pos[0, 1]:.3f}, {ee_pos[0, 2]:.3f}]")
                print(f"  🚗 Base Pos (WORLD): [{base_pos_world[0, 0]:.3f}, {base_pos_world[0, 1]:.3f}, {base_pos_world[0, 2]:.3f}]")
                print(f"  🔧 Base PPR offsets:   [{base_ppr[0, 0]:.3f}, {base_ppr[0, 1]:.3f}, {base_ppr[0, 2]:.3f}] (X, Y, theta)")
                
                # Calculate EE position relative to base (both in world frame)
                ee_relative = ee_pos[0] - base_pos_world[0]
                print(f"  📍 EE relative to base: [{ee_relative[0]:.3f}, {ee_relative[1]:.3f}, {ee_relative[2]:.3f}]")
                print(f"  📍 EE distance from base: {torch.norm(ee_relative[:2]).item():.3f} m (should be < 0.65m for arm reach)")
                
                print(f"  📏 EE Error:     {tracking_error[0].item():.4f} m")
                print(f"  📐 Base-Target:  {base_to_target_2d[0].item():.4f} m (arm reach: 0.6m)")
                
                # Show if base should be moving
                if base_to_target_2d[0].item() > 0.6:
                    beyond_reach = base_to_target_2d[0].item() - 0.6
                    print(f"  ⚠️  Base SHOULD be moving! (target {beyond_reach:.3f}m beyond arm reach)")
                    print(f"  💸 Distance penalty: {10.0 * beyond_reach:.2f} points")
                    
                # Show reward components
                if hasattr(self, 'reward_components'):
                    base_mob = self.reward_components.get('base_mobilization', torch.zeros(1, device=self.device))
                    pos_track = self.reward_components.get('position_tracking', torch.zeros(1, device=self.device))
                    dist_pen = self.reward_components.get('target_distance_penalty', torch.zeros(1, device=self.device))
                    print(f"  💰 base_mobilization reward: {base_mob[0].item():.4f}")
                    print(f"  💰 position_tracking reward: {pos_track[0].item():.4f}")
                    print(f"  💸 target_distance_penalty: {dist_pen[0].item():.4f}")
            return
        
        try:
            # Update target markers (red spheres at target positions)
            self._target_markers.visualize(target_pos)
            
            # Update EE markers (green spheres at end-effector positions)
            self._ee_markers.visualize(ee_pos)
            
        except Exception as e:
            # Silently disable if visualization fails
            self._visualization_enabled = False
    
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
    
    def _verify_joint_mapping(self):
        """Verify joint mapping (lazy initialization after PhysX view is ready)."""
        if not self._joint_mapping_verified and hasattr(self.robot, '_root_physx_view'):
            print("\n" + "="*80)
            print("JOINT MAPPING VERIFICATION")
            print("="*80)
            print(f"Total joints in robot: {len(self.robot.joint_names)}")
            print(f"Joint names: {self.robot.joint_names}")
            
            # Expected mapping for mobile manipulator with PPR base
            expected_base_joints = ["joint_x", "joint_y", "joint_theta"]
            expected_arm_joints = [f"left_arm_joint{i}" for i in range(1, 7)]
            
            print(f"\nExpected BASE joints (indices 0-2): {expected_base_joints}")
            print(f"Expected ARM joints (indices 3-8): {expected_arm_joints}")
            
            # Verify actual indices
            for i, name in enumerate(self.robot.joint_names):
                joint_type = "UNKNOWN"
                if name in expected_base_joints:
                    expected_idx = expected_base_joints.index(name)
                    joint_type = f"BASE[{expected_idx}]"
                    if i != expected_idx:
                        print(f"⚠️  WARNING: {name} at index {i}, expected at {expected_idx}")
                elif name in expected_arm_joints:
                    expected_idx = expected_arm_joints.index(name) + 3  # ARM starts at index 3
                    joint_type = f"ARM[{expected_idx - 3}]"
                    if i != expected_idx:
                        print(f"⚠️  WARNING: {name} at index {i}, expected at {expected_idx}")
                print(f"  [{i}] {name:20s} -> {joint_type}")
            
            print("="*80 + "\n")
            self._joint_mapping_verified = True
    
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
            # Store raw actions [-1,1] for policy consistency
            # NOTE: Base actions will be scaled when applied to robot, but history keeps original policy outputs
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
        
        # Apply base position commands for mobile base (differential drive)
        # Initialize base joint indices on first call (lookup by name for safety)
        if not hasattr(self, '_base_joint_ids'):
            # Base joints: joint_x (prismatic X), joint_y (prismatic Y), joint_theta (revolute rotation)
            base_joint_names = ["joint_x", "joint_y", "joint_theta"]
            self._base_joint_ids = []
            for name in base_joint_names:
                if name in self.robot.joint_names:
                    idx = self.robot.joint_names.index(name)
                    self._base_joint_ids.append(idx)
            self._base_joint_ids = torch.tensor(self._base_joint_ids, device=self.device)
            print(f"[MobileMMTrackEE] Base joint IDs initialized: {self._base_joint_ids.tolist()}")
            print(f"[MobileMMTrackEE] Base joint names: {base_joint_names}")
        
        # Get current base positions from physics (no drift accumulation)
        current_base_pos = self.robot.data.joint_pos[:, self._base_joint_ids]  # [num_envs, 3]
        theta = current_base_pos[:, 2]  # Current orientation (joint_theta)
        
        # Scale base actions from [-1, 1] to actual velocity limits
        # CRITICAL FIX: Previously actions were used directly without scaling!
        base_vx_desired = base_vx * self.robot_limits["max_linear_velocity"]  # [-1.5, +1.5] m/s
        base_wz_desired = base_wz * self.robot_limits["max_angular_velocity"]  # [-2.0, +2.0] rad/s
        
        # Rate limit velocities to respect acceleration constraints
        # This ensures commanded velocities are physically realizable
        dt = self.cfg.sim.dt * self.cfg.decimation
        max_linear_accel = self.robot_limits["max_linear_acceleration"]
        max_angular_accel = self.robot_limits["max_angular_acceleration"]
        max_vel_delta_linear = max_linear_accel * dt  # Max velocity change per step (e.g., 0.1 m/s)
        max_vel_delta_angular = max_angular_accel * dt  # Max angular velocity change per step
        
        # Get previous commanded velocities (body frame) for rate limiting
        prev_vx = self.prev_commanded_vel[:, 0:1]
        prev_wz = self.prev_commanded_vel[:, 2:3]
        
        # Clamp velocity changes to respect acceleration limits
        vel_delta_x = base_vx_desired - prev_vx
        vel_delta_x_clamped = torch.clamp(vel_delta_x, -max_vel_delta_linear, max_vel_delta_linear)
        base_vx_scaled = prev_vx + vel_delta_x_clamped
        
        vel_delta_wz = base_wz_desired - prev_wz
        vel_delta_wz_clamped = torch.clamp(vel_delta_wz, -max_vel_delta_angular, max_vel_delta_angular)
        base_wz_scaled = prev_wz + vel_delta_wz_clamped
        
        # Store commanded velocities (body frame) for reward calculation
        self.current_commanded_vel.zero_()
        self.current_commanded_vel[:, 0:1] = base_vx_scaled  # Linear velocity (x)
        self.current_commanded_vel[:, 2:3] = base_wz_scaled  # Angular velocity (yaw)
        
        # Integrate rate-limited velocities to position deltas using differential drive kinematics
        dx = base_vx_scaled.squeeze(-1) * torch.cos(theta) * dt  # X displacement in global frame
        dy = base_vx_scaled.squeeze(-1) * torch.sin(theta) * dt  # Y displacement in global frame
        dtheta = base_wz_scaled.squeeze(-1) * dt  # Angular displacement
        
        # Compute new target positions
        position_deltas = torch.stack([dx, dy, dtheta], dim=1)  # [num_envs, 3]
        new_base_targets = current_base_pos + position_deltas
        
        # Apply position targets (PPR joints are position-controlled, NOT velocity-controlled)
        self.robot.set_joint_position_target(
            target=new_base_targets,
            joint_ids=self._base_joint_ids
        )
        
        # DEBUG: Print base movement on first few steps
        if not hasattr(self, '_base_debug_count'):
            self._base_debug_count = 0
        if self._base_debug_count < 5:
            print(f"\n[BASE DEBUG Step {self._base_debug_count}]")
            print(f"  base_vx action: {base_vx[0].item():.4f} -> scaled: {base_vx_scaled[0].item():.4f} m/s")
            print(f"  base_wz action: {base_wz[0].item():.4f} -> scaled: {base_wz_scaled[0].item():.4f} rad/s")
            print(f"  position_delta: [{dx[0].item():.4f}, {dy[0].item():.4f}, {dtheta[0].item():.4f}]")
            print(f"  current_pos: {current_base_pos[0].cpu().numpy()}")
            print(f"  new_target: {new_base_targets[0].cpu().numpy()}")
            self._base_debug_count += 1
    
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
            # Extract joint limits from robot data
            # Robot has 9 total joints: [0-2: base PPR, 3-8: arm 6-DOF]
            # We only care about ARM joints for limit violations (base has huge limits)
            self.joint_lower_limits = self.robot.data.soft_joint_pos_limits[0, 3:9, 0]  # ARM joints only
            self.joint_upper_limits = self.robot.data.soft_joint_pos_limits[0, 3:9, 1]  # ARM joints only
            self._joint_limits_initialized = True
            print(f"[MobileMMTrackEE] Joint limits initialized (ARM joints 3-8 only):")
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
        self._verify_joint_mapping()  # Verify joint mapping on first call
        
        # Robot state - use root_pos_w for base position (this is the actual world position maintained by PhysX)
        # PPR joints are used for COMMANDING movement, but root_pos_w reflects the actual simulated position
        joint_pos = self.robot.data.joint_pos
        joint_vel = self.robot.data.joint_vel
        
        # Base position from PhysX root (actual world position)
        base_pos = self.robot.data.root_pos_w.clone()
        base_quat = self.robot.data.root_quat_w
        
        # Use root velocities (these are in world frame and correct)
        base_lin_vel = self.robot.data.root_lin_vel_w
        base_ang_vel = self.robot.data.root_ang_vel_w
        
        # Get end-effector state
        ee_pos = self.robot.data.body_pos_w[:, self._ee_body_idx, :]
        ee_quat = self.robot.data.body_quat_w[:, self._ee_body_idx, :]
        ee_lin_vel = self.robot.data.body_lin_vel_w[:, self._ee_body_idx, :]
        ee_ang_vel = self.robot.data.body_ang_vel_w[:, self._ee_body_idx, :]
        
        # Get target from trajectory
        target_pos, target_quat = self.trajectory_manager.get_target_pose()
        
        # Update visualization markers (if enabled)
        self._update_visualization_markers(ee_pos, target_pos)
        
        # Optional: Lookahead
        lookahead_pos = None
        if self.task_cfg.use_lookahead:
            lookahead_pos, _ = self.trajectory_manager.get_lookahead(
                steps=self.task_cfg.lookahead_steps,
                lookahead_dt=self.task_cfg.lookahead_dt,
            )
        
        # Normalize base velocities for observations (to match policy's expected range [0,1])
        base_lin_vel_obs = base_lin_vel / self.robot_limits["max_linear_velocity"]  # [0, 1.5] -> [0, 1]
        base_ang_vel_obs = base_ang_vel / self.robot_limits["max_angular_velocity"]  # [0, 2.0] -> [0, 1]
        
        # Compose full observation
        obs = compose_observation(
            base_pos=base_pos,
            base_quat=base_quat,
            base_lin_vel=base_lin_vel_obs,  # Pass normalized velocities
            base_ang_vel=base_ang_vel_obs,  # Pass normalized velocities
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
        
        # Get robot state - use root_pos_w (actual world position maintained by PhysX)
        base_pos = self.robot.data.root_pos_w.clone()
        base_quat = self.robot.data.root_quat_w
        base_lin_vel = self.robot.data.root_lin_vel_w
        base_ang_vel = self.robot.data.root_ang_vel_w
        
        joint_pos = self.robot.data.joint_pos
        arm_joint_pos = joint_pos[:, 3:9]  # ARM joints only
        joint_vel = self.robot.data.joint_vel  # All joint velocities (needed for monitoring)
        arm_joint_vel = joint_vel[:, 3:9]  # ARM joint velocities
        
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
        
        # DIAGNOSTIC: Check contact force API and monitor continuously
        contact_force_mag = torch.norm(net_contact_forces, dim=-1)
        max_force = contact_force_mag.max().item()
        
        if not hasattr(self, '_contact_force_checked'):
            print(f"\n{'='*80}")
            print(f"CONTACT FORCE API VERIFICATION")
            print(f"{'='*80}")
            print(f"Contact forces shape: {net_contact_forces.shape}")
            print(f"Max contact force: {max_force:.4f} N")
            if max_force < 0.001:
                print(f"⚠️  WARNING: Contact forces are zero!")
                print(f"   Self-collision detection may NOT be working!")
            else:
                print(f"✅ Contact forces detected - API is working!")
            print(f"{'='*80}\n")
            self._contact_force_checked = True
            self._collision_step_count = 0
        
        # Monitor for actual collisions (contact forces > threshold)
        collision_threshold = self.reward_weights["self_collision_threshold"]
        if max_force > collision_threshold:
            if not hasattr(self, '_last_collision_warning_step'):
                self._last_collision_warning_step = -100
            
            # Print warning every 100 steps to avoid spam
            if self._collision_step_count - self._last_collision_warning_step >= 100:
                print(f"\n⚠️  [COLLISION DETECTED] Step {self._collision_step_count}")
                print(f"   Max contact force: {max_force:.2f} N (threshold: {collision_threshold:.2f} N)")
                
                # Find which body has the collision
                max_body_idx = contact_force_mag.max(dim=-1).indices[0].item()
                if max_body_idx < len(self.robot.body_names):
                    print(f"   Collision on body: {self.robot.body_names[max_body_idx]}")
                self._last_collision_warning_step = self._collision_step_count
        
        # Monitor for wild joint velocities
        max_joint_vel = torch.abs(joint_vel).max().item()
        if max_joint_vel > 5.0:  # rad/s - very fast!
            if not hasattr(self, '_last_wild_vel_warning_step'):
                self._last_wild_vel_warning_step = -100
            
            if self._collision_step_count - self._last_wild_vel_warning_step >= 100:
                print(f"\n⚠️  [WILD ARM MOTION] Step {self._collision_step_count}")
                print(f"   Max joint velocity: {max_joint_vel:.2f} rad/s")
                print(f"   Joint velocities [env 0]: {joint_vel[0].cpu().numpy()}")
                self._last_wild_vel_warning_step = self._collision_step_count
        
        # Monitor for joint limit violations (potential disengagement)
        # Robot has 9 joints total: [0-2: base PPR, 3-8: arm 6-DOF]
        base_joint_pos = self.robot.data.joint_pos[:, 0:3]  # BASE joints (X, Y, theta)
        arm_joint_pos = self.robot.data.joint_pos[:, 3:9]   # ARM joints (6-DOF)
        
        # Check ARM joint limits (base has huge limits ±50m, not useful to check)
        arm_lower_violations = (arm_joint_pos < self.joint_lower_limits).any(dim=-1)
        arm_upper_violations = (arm_joint_pos > self.joint_upper_limits).any(dim=-1)
        
        if arm_lower_violations.any() or arm_upper_violations.any():
            if not hasattr(self, '_last_limit_warning_step'):
                self._last_limit_warning_step = -100
            
            if self._collision_step_count - self._last_limit_warning_step >= 100:
                print(f"\n⚠️  [JOINT LIMIT VIOLATION] Step {self._collision_step_count}")
                print(f"   BASE joints [env 0]: {base_joint_pos[0].cpu().numpy()} (X_m, Y_m, theta_rad)")
                print(f"   ARM joints [env 0]: {arm_joint_pos[0].cpu().numpy()}")
                print(f"   ARM Lower limits: {self.joint_lower_limits.cpu().numpy()}")
                print(f"   ARM Upper limits: {self.joint_upper_limits.cpu().numpy()}")
                
                # Calculate ARM violations
                lower_diff = arm_joint_pos[0] - self.joint_lower_limits
                upper_diff = self.joint_upper_limits - arm_joint_pos[0]
                print(f"   ARM Lower margin: {lower_diff.cpu().numpy()}")
                print(f"   ARM Upper margin: {upper_diff.cpu().numpy()}")
                self._last_limit_warning_step = self._collision_step_count
        
        self._collision_step_count += 1
        
        # Normalize velocities for observations/diagnostics (policy still sees normalized values)
        base_lin_vel_normalized = base_lin_vel / self.robot_limits["max_linear_velocity"]
        base_ang_vel_normalized = base_ang_vel / self.robot_limits["max_angular_velocity"]
        prev_base_lin_vel_normalized = self.prev_base_lin_vel / self.robot_limits["max_linear_velocity"]

        # Commanded velocities (body frame) for rate-limited penalties
        commanded_vel = self.current_commanded_vel
        prev_commanded_vel = self.prev_commanded_vel
        commanded_linear = torch.zeros_like(commanded_vel)
        commanded_linear[:, 0:1] = commanded_vel[:, 0:1]
        prev_commanded_linear = torch.zeros_like(prev_commanded_vel)
        prev_commanded_linear[:, 0:1] = prev_commanded_vel[:, 0:1]
        commanded_ang = torch.zeros_like(commanded_vel)
        commanded_ang[:, 2:3] = commanded_vel[:, 2:3]

        commanded_linear_accel = torch.zeros_like(commanded_vel)
        commanded_linear_accel[:, 0:1] = (commanded_vel[:, 0:1] - prev_commanded_vel[:, 0:1]) / self.control_dt
        prev_commanded_linear_accel = self.prev_commanded_accel

        # Actual acceleration from simulator (for diagnostics only)
        actual_accel = (base_lin_vel - self.prev_base_lin_vel) / self.control_dt

        # Get base orientation for lateral penalty calculation
        base_quat = self.robot.data.root_quat_w
        
        # Compute rewards with all new constraint penalties
        # Use COMMANDED velocities for penalty calculation to avoid penalizing simulation artifacts
        rewards, self.reward_components = compute_combined_reward(
            current_ee_pos=ee_pos,
            current_ee_quat=ee_quat,
            target_pos=target_pos,
            target_quat=target_quat,
            prev_tracking_error=self.prev_tracking_error,
            actions=self.prev_actions,  # Current actions (just applied)
            prev_actions=self.prev_prev_actions,  # Actions from previous step
            prev_prev_actions=self._actions_t_minus_2,  # Actions from 2 steps ago (for jerk calculation)
            base_pos=base_pos,  # NEW: Current base position for progress tracking
            base_lin_vel=commanded_linear,
            base_ang_vel=commanded_ang,
            base_quat=base_quat,  # Base orientation for lateral penalty
            joint_pos=arm_joint_pos,  # ARM joints only [6]
            joint_vel=arm_joint_vel,  # ARM joint velocities only [6]
            prev_base_pos=self.prev_base_pos,  # NEW: Previous base position
            prev_base_lin_vel=prev_commanded_linear,
            prev_joint_vel=self.prev_joint_vel[:, 3:9],  # ARM joint velocities only [6]
            prev_base_accel=prev_commanded_linear_accel,
            joint_lower=self.joint_lower_limits,  # ARM limits [6]
            joint_upper=self.joint_upper_limits,  # ARM limits [6]
            robot_limits=self.robot_limits,
            contact_forces=net_contact_forces,  # Use actual contact forces for self-collision
            min_obstacle_dist=None,  # Not using obstacles for now
            dt=self.control_dt,
            weights=self.reward_weights,
        )
        
        # Update history for next step - store COMMANDED velocities for consistent penalty calculation
        self.prev_tracking_error = torch.norm(target_pos - ee_pos, dim=-1)
        self.prev_base_pos = base_pos.clone()  # NEW: Store base position for next step
        self.prev_base_lin_vel = base_lin_vel.clone()
        self.prev_joint_vel = joint_vel.clone()
        self.prev_base_accel = actual_accel.clone()
        self.prev_commanded_vel = commanded_vel.clone()
        self.prev_commanded_accel = commanded_linear_accel.clone()
        
        # Advance trajectory to next target
        # This must happen after reward calculation so current target is used for this step
        self.trajectory_manager.step()
        
        # Log reward components
        self.extras["reward_components"] = {
            k: v.mean().item() for k, v in self.reward_components.items()
        }
        
        # DIAGNOSTIC: Track base movement metrics
        self.extras["base_diagnostics"] = {
            "base_vel_x_mean": base_lin_vel[:, 0].mean().item(),
            "base_vel_x_std": base_lin_vel[:, 0].std().item(),
            "base_vel_x_max": base_lin_vel[:, 0].abs().max().item(),
            "base_vel_z_mean": base_ang_vel[:, 2].mean().item(),
            "base_vel_z_std": base_ang_vel[:, 2].std().item(),
            "base_vel_z_max": base_ang_vel[:, 2].abs().max().item(),
            "base_action_x_mean": self.prev_actions[:, 6].mean().item(),
            "base_action_x_std": self.prev_actions[:, 6].std().item(),
            "base_action_z_mean": self.prev_actions[:, 7].mean().item(),
            "base_action_z_std": self.prev_actions[:, 7].std().item(),
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
                # Calculate contact force magnitude per environment
                contact_force_mag = torch.norm(net_contact_forces, dim=-1)  # [num_envs, num_bodies]
                
                # CRITICAL: Exclude base link (index 0) to filter out ground contact
                # Ground reaction forces should NOT terminate episodes!
                if contact_force_mag.shape[1] > 1:
                    contact_force_mag = contact_force_mag[:, 1:]  # Only check arm links
                
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
        
        # CRITICAL FIX: Reset robot base position to match trajectory starting point
        # This ensures the robot starts near the trajectory, making it physically possible to track
        first_target_pos, _ = self.trajectory_manager.get_target_pose()
        
        # Set base position to match trajectory XY, keep current Z (floor level)
        new_root_state = self.robot.data.default_root_state[env_ids].clone()
        new_root_state[:, 0] = first_target_pos[env_ids, 0]  # X position
        new_root_state[:, 1] = first_target_pos[env_ids, 1]  # Y position
        # Keep Z position from default (floor level)
        # Keep orientation from default (facing forward)
        
        # Reset velocities to zero
        new_root_state[:, 7:10] = 0.0  # Linear velocity
        new_root_state[:, 10:13] = 0.0  # Angular velocity
        
        # Apply the new root state
        self.robot.write_root_state_to_sim(new_root_state, env_ids=env_ids)
        
        # Also reset base joint positions (PPR joints) to match
        # Initialize base joint indices if needed
        if not hasattr(self, '_base_joint_ids'):
            base_joint_names = ["joint_x", "joint_y", "joint_theta"]
            self._base_joint_ids = []
            for name in base_joint_names:
                if name in self.robot.joint_names:
                    idx = self.robot.joint_names.index(name)
                    self._base_joint_ids.append(idx)
            self._base_joint_ids = torch.tensor(self._base_joint_ids, device=self.device)
        
        # Set base joint positions to trajectory start (x, y, theta=0)
        base_joint_pos = torch.zeros(len(env_ids), 3, device=self.device)
        base_joint_pos[:, 0] = first_target_pos[env_ids, 0]  # joint_x
        base_joint_pos[:, 1] = first_target_pos[env_ids, 1]  # joint_y  
        base_joint_pos[:, 2] = 0.0  # joint_theta (facing forward)
        
        self.robot.set_joint_position_target(
            base_joint_pos, 
            joint_ids=self._base_joint_ids,
            env_ids=env_ids
        )
        
        print(f"[RESET] Env {env_ids[0].item() if len(env_ids) > 0 else 'N/A'}: Base moved to trajectory start [{first_target_pos[env_ids[0], 0]:.3f}, {first_target_pos[env_ids[0], 1]:.3f}, {first_target_pos[env_ids[0], 2]:.3f}]")
        
        # Reset state buffers - use root_pos_w (actual world position)
        self.prev_base_pos[env_ids] = new_root_state[:, 0:3]
        self.prev_actions[env_ids] = 0.0
        self.prev_prev_actions[env_ids] = 0.0
        self.prev_tracking_error[env_ids] = 0.0
        self.prev_base_lin_vel[env_ids] = 0.0
        self.prev_joint_vel[env_ids] = 0.0
        self.prev_base_accel[env_ids] = 0.0
        self.current_commanded_vel[env_ids] = 0.0
        self.prev_commanded_vel[env_ids] = 0.0
        self.prev_commanded_accel[env_ids] = 0.0
        
        if self.action_history is not None:
            self.action_history[env_ids] = 0.0
        
        # Randomize initial joint positions (ARM joints only, not base)
        if self.task_cfg.randomize_initial_joint_positions:
            # Only randomize the 6 arm joints, not the 3 base joints
            arm_joint_noise = torch.randn(
                len(env_ids), 6, device=self.device
            ) * self.task_cfg.initial_joint_noise_std
            
            # Get arm joint IDs (indices 3-8, after the 3 base joints)
            if not hasattr(self, '_arm_joint_ids'):
                arm_joint_names = [
                    "left_arm_joint1", "left_arm_joint2", "left_arm_joint3",
                    "left_arm_joint4", "left_arm_joint5", "left_arm_joint6"
                ]
                self._arm_joint_ids = []
                for name in arm_joint_names:
                    if name in self.robot.joint_names:
                        idx = self.robot.joint_names.index(name)
                        self._arm_joint_ids.append(idx)
                self._arm_joint_ids = torch.tensor(self._arm_joint_ids, device=self.device)
            
            default_arm_joint_pos = self.robot.data.default_joint_pos[env_ids][:, self._arm_joint_ids]
            self.robot.set_joint_position_target(
                default_arm_joint_pos + arm_joint_noise, 
                joint_ids=self._arm_joint_ids,
                env_ids=env_ids
            )
        
        # Advance trajectory by one step to start
        self.trajectory_manager.step()
