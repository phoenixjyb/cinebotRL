"""RecomoProto1 manipulator end-effector tracking environment for Isaac Lab.

This environment trains a RecomoProto1 manipulator to track a reference trajectory
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
    from isaaclab.sensors import ContactSensorCfg
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

from rl_platform.robots.recomoproto1 import get_recomoproto1_usd_path
from .config import RecomoProto1TrackConfig, RewardWeights
from .trajectories import TrajectoryManager
from .observations import compose_observation, get_observation_dimensions
from .rewards import compute_combined_reward


def quat_to_yaw(quat: torch.Tensor) -> torch.Tensor:
    """Extract yaw angle (rotation around Z-axis) from quaternion.
    
    Args:
        quat: Quaternion in (w, x, y, z) format, shape [..., 4]
        
    Returns:
        Yaw angles in radians, shape [...]
    """
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    return yaw


@configclass
class RecomoProto1TrackEEEnvCfg(DirectRLEnvCfg):
    """Configuration for the RecomoProto1 manipulator tracking environment."""
    
    # Simulation settings
    decimation = 10  # 200Hz physics / 10 = 20Hz control (realistic for RecomoProto1 manipulators)
    episode_length_s = 20.0
    num_envs = 1  # Default to 1, can be overridden
    
    # Task-specific configuration
    task_config: RecomoProto1TrackConfig = field(default_factory=RecomoProto1TrackConfig)
    
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
            num_contacts=1,  # Collision signal from contact sensors (single normalized scalar)
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
        print(f"[RecomoProto1TrackEE] DEBUG: _create_scene_config called with self.num_envs = {self.num_envs}")
        
        # Get robot USD path
        robot_usd_path = str(get_recomoproto1_usd_path())
        
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
        
        # Add contact sensor for chassis (to detect when arm links collide with it)
        # Following official Isaac Lab pattern from contact_sensor.py example
        # filter_prim_paths_expr limits to only report contacts with arm links
        # Contact sensors: Monitor both chassis and arm for comprehensive collision detection
        # 1. Chassis sensor: Detects arm-base collisions (filters out ground support)
        # 2. Arm sensor: Detects arm-ground collisions
        scene_cfg.contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/abstract_chassis_link",  # Monitor chassis
            update_period=0.0,  # Update every sim step (5ms physics)
            history_length=1,   # Only need current forces
            debug_vis=False,    # Disable visualization for performance
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/left_arm.*"],  # Only report arm-chassis contacts
        )
        
        # Add arm contact sensor to detect arm-ground collisions
        scene_cfg.arm_contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/left_arm.*",  # Monitor all arm links
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            filter_prim_paths_expr=["/World/Ground"],  # Only report arm-ground contacts (actual prim path)
        )
        
        return scene_cfg


class RecomoProto1TrackEEEnv(DirectRLEnv):
    """RecomoProto1 manipulator end-effector tracking environment.
    
    This environment trains the robot to track a reference trajectory with
    its end-effector while maintaining stability and avoiding obstacles.
    """
    
    cfg: RecomoProto1TrackEEEnvCfg
    
    def __init__(self, cfg: RecomoProto1TrackEEEnvCfg | None = None, render_mode: str | None = None, **kwargs):
        """Initialize the environment.
        
        Args:
            cfg: Environment configuration (optional, created if None)
            render_mode: Rendering mode (None for headless) - currently unused
            **kwargs: Additional arguments that may override configuration.
        """
        print(f"\n{'='*70}")
        print(f"[RecomoProto1TrackEE] __init__ called:")
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
        
        print(f"[RecomoProto1TrackEE] DEBUG: Before config handling, num_envs_override={num_envs_override}")
        
        if cfg is None:
            from dataclasses import replace
            cfg = RecomoProto1TrackEEEnvCfg()
            if num_envs_override is not None:
                cfg = replace(cfg, num_envs=num_envs_override)
                print(f"[RecomoProto1TrackEE] Created config with num_envs={cfg.num_envs}")
        else:
            if num_envs_override is not None:
                cfg.num_envs = num_envs_override
                print(f"[RecomoProto1TrackEE] Updated existing config to num_envs={cfg.num_envs}")
        
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
        
        print(f"[RecomoProto1TrackEE] DEBUG: About to call super().__init__() with cfg.num_envs={cfg.num_envs if cfg else 'None'}")
        
        # DirectRLEnv only takes cfg, not render_mode
        super().__init__(cfg, **kwargs)
        
        print(f"[RecomoProto1TrackEE] DEBUG: After super().__init__(), self.num_envs={self.num_envs}")
        
        # ============================================================================
        # COORDINATE FRAME CONVENTIONS (FIXED: Session 7c)
        # ============================================================================
        # BASE CONTROL FIX (2024-10-27): Resolved dual control mechanism bug
        # 
        # PREVIOUS BUG:
        # - Both write_root_state_to_sim() AND set_joint_position_target() controlled base
        # - PPR joints accumulated world-frame displacements (joint_y reached -6.3m!)
        # - root_pos_w stayed frozen (Y=0.08m), creating 6.38m discrepancies
        # - This caused all sessions (6, 7, 7b) to have ~0.002m base displacement
        #
        # CURRENT FIX:
        # - Base controlled ONLY via write_root_link_velocity_to_sim() (direct velocity)
        # - PPR joints set to ZERO (they are kinematic offsets, not world positions)
        # - Orientation unified: all transforms use root_quat_w (via quat_to_yaw helper)
        # - Reachability uses root_pos_w/root_quat_w (not joint_pos)
        #
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
            "base_target_alignment": self.task_cfg.rewards.base_target_alignment,  # BUGFIX: Was missing!
            "target_distance_penalty": self.task_cfg.rewards.target_distance_penalty,
            "excessive_base_movement_penalty": self.task_cfg.rewards.excessive_base_movement_penalty,  # BUGFIX: Was missing!
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
            # Session 8b: Reachability-aware base coordination
            "reachability_maintenance_reward": self.task_cfg.rewards.reachability_maintenance_reward,
            "reachability_distance_weight": self.task_cfg.rewards.reachability_distance_weight,
            "reachability_soft_margin": self.task_cfg.rewards.reachability_soft_margin,
            "reachability_hard_margin": self.task_cfg.rewards.reachability_hard_margin,
            "base_overshoot_penalty": self.task_cfg.rewards.base_overshoot_penalty,
            "mobilization_progress_cap": self.task_cfg.rewards.mobilization_progress_cap,
            "position_distance_penalty": self.task_cfg.rewards.position_distance_penalty,
        }
        
        # Session 8h: Curriculum learning - track stage and original weights
        self.use_curriculum = self.task_cfg.rewards.use_curriculum
        self.curriculum_stage_1_steps = self.task_cfg.rewards.curriculum_stage_1_steps
        self.current_training_step = 0  # Will be updated during training
        self.curriculum_stage = 1 if self.use_curriculum else 2  # Start in stage 1 if enabled
        
        # Store original weights for curriculum switching
        self.base_position_weight = self.reward_weights["position_tracking"]
        self.base_orientation_weight = self.reward_weights["orientation_tracking"]
        
        # Apply Stage 1 weights if curriculum enabled
        if self.use_curriculum:
            self.reward_weights["position_tracking"] = self.task_cfg.rewards.curriculum_stage_1_position_weight
            self.reward_weights["orientation_tracking"] = self.task_cfg.rewards.curriculum_stage_1_orientation_weight
            print(f"[Session 8h] Curriculum Stage 1 Active: "
                  f"position_weight={self.reward_weights['position_tracking']}, "
                  f"orientation_weight={self.reward_weights['orientation_tracking']} (1:3 ratio maintained)")
            print(f"[Session 8h] Gradual transition: 45M-55M linear ramp → ({self.task_cfg.rewards.curriculum_stage_2_position_weight}, "
                  f"{self.task_cfg.rewards.curriculum_stage_2_orientation_weight})")

        
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
        
        # Load reachability map for intelligent base planning
        print("[RecomoProto1TrackEE] Loading reachability map...")
        try:
            from rl_platform.utils.reachability_map import ReachabilityMap
            reach_map_path = "matlab/reach_map_mobile_mm_arm_only.mat"
            self.reach_map = ReachabilityMap(reach_map_path, device=self.device)
            print(f"[RecomoProto1TrackEE] ✓ Reachability map loaded with {len(self.reach_map.reachable_positions)} points")
        except Exception as e:
            print(f"[RecomoProto1TrackEE] ⚠ Failed to load reachability map: {e}")
            print(f"[RecomoProto1TrackEE] ⚠ Continuing without reachability guidance")
            self.reach_map = None
        
        # State buffers for tracking history (needed for derivatives)
        self.prev_actions = torch.zeros(
            self.num_envs, self.cfg.num_actions, device=self.device
        )
        self.prev_prev_actions = torch.zeros(
            self.num_envs, self.cfg.num_actions, device=self.device
        )
        self.prev_tracking_error = torch.zeros(self.num_envs, device=self.device)
        self.prev_ee_ori_error = torch.zeros(self.num_envs, device=self.device)  # SESSION 8i: Previous orientation error
        
        # Tracking error buffers for evaluation/logging
        self.ee_pos_error_buf = torch.zeros(self.num_envs, 3, device=self.device)  # [x, y, z] position error
        self.ee_ori_error_buf = torch.zeros(self.num_envs, device=self.device)  # Angular error (radians)
        
        # Position history for base progress tracking
        self.prev_base_pos = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Velocity history for acceleration calculation
        self.prev_base_lin_vel = torch.zeros(self.num_envs, 3, device=self.device)
        print(f"[RecomoProto1TrackEE] DEBUG: Initializing velocity buffers with num_envs={self.num_envs}")
        self.current_commanded_vel = torch.zeros(self.num_envs, 3, device=self.device)  # Rate-limited commanded velocities for THIS step
        self.prev_commanded_vel = torch.zeros(self.num_envs, 3, device=self.device)
        print(f"[RecomoProto1TrackEE] DEBUG: current_commanded_vel.shape = {self.current_commanded_vel.shape}")
        self.prev_commanded_accel = torch.zeros(self.num_envs, 3, device=self.device)
        self.prev_joint_vel = torch.zeros(
            self.num_envs, 9, device=self.device  # 9 total joints (3 base PPR + 6 arm)
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
        
        print(f"[RecomoProto1TrackEE] Environment initialized:")
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

            txt_file = Path("data/trajectory_filters/chassis_required_indices.txt")
            if txt_file.exists():
                content = txt_file.read_text()
                match = re.search(r"CHASSIS_REQUIRED_INDICES\s*=\s*\[(.*?)\]", content, re.DOTALL)
                if match:
                    cleaned = match.group(1).replace("\n", " ")
                    parsed = [int(x.strip()) for x in cleaned.split(",") if x.strip()]
                    if limit is not None:
                        parsed = parsed[:limit]
                    print(f"[RecomoProto1TrackEE] Loaded {len(parsed)} chassis-required indices from {txt_file}")
                    return parsed

            json_file = Path("trajectoryToLearn/trajectory_analysis.json")
            if json_file.exists():
                data = json.loads(json_file.read_text())
                parsed = data.get("chassis_requiring_indices")
                if isinstance(parsed, list):
                    parsed_int = [int(x) for x in parsed]
                    if limit is not None:
                        parsed_int = parsed_int[:limit]
                    print(f"[RecomoProto1TrackEE] Loaded {len(parsed_int)} chassis-required indices from {json_file}")
                    return parsed_int

        except Exception as exc:
            print(f"[RecomoProto1TrackEE] WARNING: Unable to load chassis-required indices: {exc}")

        print("[RecomoProto1TrackEE] WARNING: use_chassis_only requested but no chassis indices found. Using all trajectories.")
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

        # Trajectory visualization bookkeeping
        self._visited_waypoint_masks = [None for _ in range(self.num_envs)]
        self._last_waypoint_idx = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._marker_creation_attempts = 0
        self._debug_vis_requested = False
        self._debug_vis_active = False
        
        # Setup visualization markers (only in GUI mode)
        self._setup_visualization_markers()
    
    def _setup_visualization_markers(self):
        """Setup visual markers for trajectory visualization.
        
        Note: Markers are created lazily on first use after simulation starts,
        not during scene setup. This ensures USD stage is ready.
        """
        # Mark that we want visualization (actual creation happens later)
        self._visualization_enabled = True
        self._markers_created = False
        self._current_target_markers = None
        self._future_target_markers = None
        self._past_target_markers = None
        self._ee_markers = None
        print("[RecomoProto1TrackEE] ℹ Visual markers deferred (will create after first reset)")
    
    def _create_markers_if_needed(self):
        """Create markers on first use (lazy initialization)."""
        if self._markers_created or not self._visualization_enabled:
            return
        
        try:
            # Import Isaac Lab marker utilities
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
            import isaaclab.sim as sim_utils
            
            print(f"[RecomoProto1TrackEE] Attempting marker creation (num_envs={self.num_envs})...")
            
            # Red spheres for current target waypoint
            current_target_cfg = VisualizationMarkersCfg(
                prim_path="/World/Visuals/current_target_markers",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.06,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    )
                },
            )
            
            # Green spheres for future waypoints
            future_target_cfg = VisualizationMarkersCfg(
                prim_path="/World/Visuals/future_target_markers",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.04,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                    )
                },
            )
            
            # Blue spheres for past waypoints
            past_target_cfg = VisualizationMarkersCfg(
                prim_path="/World/Visuals/past_target_markers",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.03,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                    )
                },
            )
            
            # Yellow spheres for end-effector positions
            ee_marker_cfg = VisualizationMarkersCfg(
                prim_path="/World/Visuals/ee_markers",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.05,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
                    )
                },
            )
            
            # Initialize markers
            print("[RecomoProto1TrackEE] Creating current target markers...")
            self._current_target_markers = VisualizationMarkers(current_target_cfg)
            print(f"  ✓ Current: {self._current_target_markers is not None}")
            
            print("[RecomoProto1TrackEE] Creating future target markers...")
            self._future_target_markers = VisualizationMarkers(future_target_cfg)
            print(f"  ✓ Future: {self._future_target_markers is not None}")
            
            print("[RecomoProto1TrackEE] Creating past target markers...")
            self._past_target_markers = VisualizationMarkers(past_target_cfg)
            print(f"  ✓ Past: {self._past_target_markers is not None}")
            
            print("[RecomoProto1TrackEE] Creating EE markers...")
            self._ee_markers = VisualizationMarkers(ee_marker_cfg)
            print(f"  ✓ EE: {self._ee_markers is not None}")
            
            # Change dome light to white background for better marker visibility
            try:
                from pxr import Usd, UsdGeom, UsdLux
                stage = self.scene.stage
                # Set dome light to white with high intensity
                dome_light_path = "/World/defaultLight"
                dome_light = UsdLux.DomeLight.Get(stage, dome_light_path)
                if dome_light:
                    dome_light.GetIntensityAttr().Set(10000.0)
                    dome_light.GetColorAttr().Set((1.0, 1.0, 1.0))
                    print("[RecomoProto1TrackEE] ✓ Changed background to white for better visibility")
            except Exception as light_err:
                print(f"[RecomoProto1TrackEE] ⚠ Could not change background color: {light_err}")
            
            print("[RecomoProto1TrackEE] ✓ Visual markers enabled")
            print("  - Red (large) = Current target waypoint")
            print("  - Green (medium) = Future waypoints")
            print("  - Blue (small) = Past waypoints")
            print("  - Yellow (medium) = End-effector position")
            self._markers_created = True
            self._marker_creation_attempts = 0
        except Exception as e:
            import traceback
            self._marker_creation_attempts += 1
            self._current_target_markers = None
            self._future_target_markers = None
            self._past_target_markers = None
            self._ee_markers = None
            if self._marker_creation_attempts <= 3:
                print(f"[RecomoProto1TrackEE] ⚠ Visual marker creation failed (attempt {self._marker_creation_attempts}):")
                print(f"  Error: {str(e)}")
                print("  Traceback:")
                traceback.print_exc()
            return
    
    def _set_debug_vis_impl(self, debug_vis: bool):
        """Implementation of debug visualization control.
        
        Called by DirectRLEnv when set_debug_vis() is called.
        Controls visibility of trajectory markers.
        
        Args:
            debug_vis: Whether to enable debug visualization
        """
        # If enabling visualization, create markers first (lazy initialization)
        if debug_vis:
            self._create_markers_if_needed()
        
        # Check if ALL markers were successfully created
        self._debug_vis_requested = debug_vis

        if (not hasattr(self, "_current_target_markers") or self._current_target_markers is None or
            not hasattr(self, "_future_target_markers") or self._future_target_markers is None or
            not hasattr(self, "_past_target_markers") or self._past_target_markers is None or
            not hasattr(self, "_ee_markers") or self._ee_markers is None):
            if debug_vis:
                print("[RecomoProto1TrackEE] ⚠ Cannot enable debug vis: one or more markers not initialized")
            return
        
        if debug_vis:
            # Enable marker visibility
            self._current_target_markers.set_visibility(True)
            self._future_target_markers.set_visibility(True)
            self._past_target_markers.set_visibility(True)
            self._ee_markers.set_visibility(True)
            print("[RecomoProto1TrackEE] ✓ Marker visibility enabled")
            self._debug_vis_active = True
        else:
            # Disable marker visibility
            self._current_target_markers.set_visibility(False)
            self._future_target_markers.set_visibility(False)
            self._past_target_markers.set_visibility(False)
            self._ee_markers.set_visibility(False)
            print("[RecomoProto1TrackEE] ℹ Marker visibility disabled")
            self._debug_vis_active = False
    
    def _debug_vis_callback(self, event):
        """Debug visualization callback - called automatically by DirectRLEnv on render events.
        
        This is where we update marker positions. DirectRLEnv calls this automatically
        during render updates if debug_vis is enabled.
        
        Args:
            event: Event object (unused, required by callback signature)
        """
        if not self._visualization_enabled:
            return
        
        # Create markers on first call (lazy initialization after simulation starts)
        self._create_markers_if_needed()
        
        # Skip if marker creation failed
        if self._current_target_markers is None:
            return
        
        # Get current EE and target positions
        ee_pos = self.robot.data.body_pos_w[:, self._ee_body_idx, :]
        target_pos, _ = self.trajectory_manager.get_target_pose()
        
        # Call the marker update method
        self._update_visualization_markers(ee_pos, target_pos)
    
    def _update_visualization_markers(self, ee_pos: torch.Tensor, target_pos: torch.Tensor):
        """Update visualization markers for trajectory tracking.
        
        Shows all trajectory waypoints with different colors:
        - Red (large): Current target waypoint
        - Green (medium): Future waypoints
        - Blue (small): Past waypoints
        - Yellow (medium): Current end-effector position
        
        Args:
            ee_pos: End-effector positions [num_envs, 3]
            target_pos: Target positions [num_envs, 3]
        """
        if self._visualization_enabled and self._current_target_markers is not None:
            # Only visualize for the first environment to avoid clutter in the viewport
            env_id = 0

            if (hasattr(self.trajectory_manager, "recorded_positions")
                    and self.trajectory_manager.recorded_positions is not None):
                all_waypoints = self.trajectory_manager.recorded_positions[env_id]  # [num_waypoints, 3]
                num_waypoints = all_waypoints.shape[0]

                empty_positions = all_waypoints.new_empty((0, 3))

                if num_waypoints == 0:
                    self._current_target_markers.visualize(empty_positions)
                    self._future_target_markers.visualize(empty_positions)
                    self._past_target_markers.visualize(empty_positions)
                    ee_single = ee_pos[env_id:env_id + 1]
                    if ee_single.shape[0] == 0:
                        self._ee_markers.visualize(empty_positions)
                    else:
                        self._ee_markers.visualize(ee_single)
                    return

                # Ensure bookkeeping containers exist
                if not hasattr(self, "_visited_waypoint_masks"):
                    self._visited_waypoint_masks = [None for _ in range(self.num_envs)]
                if not hasattr(self, "_last_waypoint_idx"):
                    self._last_waypoint_idx = torch.full(
                        (self.num_envs,), -1, dtype=torch.long, device=self.device
                    )

                current_idx_int = int(self.trajectory_manager.current_waypoint_idx[env_id].item())
                mask = self._visited_waypoint_masks[env_id]

                if mask is None or mask.numel() != num_waypoints:
                    mask = torch.zeros(num_waypoints, dtype=torch.bool, device=all_waypoints.device)
                    self._visited_waypoint_masks[env_id] = mask
                    last_idx_int = -1
                else:
                    last_idx_int = int(self._last_waypoint_idx[env_id].item())

                if last_idx_int == -1 or current_idx_int < last_idx_int:
                    mask.zero_()
                    last_idx_int = current_idx_int

                if last_idx_int <= current_idx_int:
                    mask[last_idx_int:current_idx_int + 1] = True
                else:
                    mask[last_idx_int:] = True
                    mask[:current_idx_int + 1] = True

                self._last_waypoint_idx[env_id] = current_idx_int

                current_goal_idx = (current_idx_int + 1) % num_waypoints

                past_mask = mask.clone()
                past_mask[current_goal_idx] = False

                future_mask = torch.ones(num_waypoints, dtype=torch.bool, device=all_waypoints.device)
                future_mask[past_mask] = False
                future_mask[current_goal_idx] = False

                past_waypoints = all_waypoints[past_mask]
                future_waypoints = all_waypoints[future_mask]
                current_goal_pos = all_waypoints[current_goal_idx:current_goal_idx + 1]

                if current_goal_pos.shape[0] == 0:
                    self._current_target_markers.visualize(empty_positions)
                else:
                    self._current_target_markers.visualize(current_goal_pos)

                if future_waypoints.shape[0] == 0:
                    self._future_target_markers.visualize(empty_positions)
                else:
                    self._future_target_markers.visualize(future_waypoints)

                if past_waypoints.shape[0] == 0:
                    self._past_target_markers.visualize(empty_positions)
                else:
                    self._past_target_markers.visualize(past_waypoints)

                ee_single = ee_pos[env_id:env_id + 1]
                if ee_single.shape[0] == 0:
                    self._ee_markers.visualize(empty_positions)
                else:
                    self._ee_markers.visualize(ee_single)

            else:
                empty_positions = torch.empty((0, 3), device=self.device, dtype=torch.float32)
                self._current_target_markers.visualize(empty_positions)
                self._future_target_markers.visualize(empty_positions)
                self._past_target_markers.visualize(empty_positions)
                ee_single = ee_pos[0:1]
                if ee_single.shape[0] == 0:
                    self._ee_markers.visualize(empty_positions)
                else:
                    self._ee_markers.visualize(ee_single)

        else:
            # Print console output every 50 steps with statistics and diverse env samples
            if not hasattr(self, '_vis_step_count'):
                self._vis_step_count = 0
            self._vis_step_count += 1
            
            if self._vis_step_count % 50 == 0:
                tracking_error = torch.norm(target_pos - ee_pos, dim=-1)
                
                # Get base position (using root_pos_w - actual world position)
                base_pos_world = self.robot.data.root_pos_w
                
                # Show PPR joints (FIXED: should stay near zero now)
                base_ppr = self.robot.data.joint_pos[:, 0:3]  # [joint_x, joint_y, joint_theta]
                
                base_to_target_2d = torch.norm(target_pos[:, :2] - base_pos_world[:, :2], dim=-1)
                
                # Calculate EE distance from base for all envs
                ee_relative_all = ee_pos - base_pos_world
                ee_dist_from_base = torch.norm(ee_relative_all[:, :2], dim=-1)
                
                # Get reward components if available
                base_mob = torch.zeros(self.num_envs, device=self.device)
                pos_track = torch.zeros(self.num_envs, device=self.device)
                dist_pen = torch.zeros(self.num_envs, device=self.device)
                if hasattr(self, 'reward_components'):
                    base_mob = self.reward_components.get('base_mobilization', base_mob)
                    pos_track = self.reward_components.get('position_tracking', pos_track)
                    dist_pen = self.reward_components.get('target_distance_penalty', dist_pen)
                
                # ========== OVERALL STATISTICS ==========
                print(f"\n{'='*80}")
                print(f"[TRACKING Step {self._vis_step_count}] OVERALL STATISTICS ({self.num_envs} envs)")
                print(f"{'='*80}")
                
                # Count problematic environments
                num_broken = (tracking_error > 2.0).sum().item()
                num_excellent = (tracking_error < 0.1).sum().item()
                num_good = ((tracking_error >= 0.1) & (tracking_error < 0.3)).sum().item()
                
                print(f"📊 Environment Health:")
                print(f"   Excellent (<0.1m):  {num_excellent:4d} ({100*num_excellent/self.num_envs:.1f}%)")
                print(f"   Good (0.1-0.3m):    {num_good:4d} ({100*num_good/self.num_envs:.1f}%)")
                print(f"   Poor (0.3-2.0m):    {self.num_envs-num_excellent-num_good-num_broken:4d} ({100*(self.num_envs-num_excellent-num_good-num_broken)/self.num_envs:.1f}%)")
                print(f"   Broken (>2.0m):     {num_broken:4d} ({100*num_broken/self.num_envs:.1f}%)")
                
                print(f"\n📏 EE Tracking Error (m):")
                print(f"   min={tracking_error.min():.4f}  mean={tracking_error.mean():.4f}  max={tracking_error.max():.4f}  std={tracking_error.std():.4f}")
                
                print(f"📐 Base-Target Distance (m):")
                print(f"   min={base_to_target_2d.min():.4f}  mean={base_to_target_2d.mean():.4f}  max={base_to_target_2d.max():.4f}  std={base_to_target_2d.std():.4f}")
                
                # NEW: Track actual base movement since episode start
                base_displacement = torch.norm(base_pos_world[:, :2] - self._episode_start_base_pos[:, :2], dim=-1)
                print(f"🚗 Base Movement from Start (m):")
                print(f"   min={base_displacement.min():.4f}  mean={base_displacement.mean():.4f}  max={base_displacement.max():.4f}  std={base_displacement.std():.4f}")
                
                print(f"🤖 EE Distance from Base (m):")
                print(f"   min={ee_dist_from_base.min():.4f}  mean={ee_dist_from_base.mean():.4f}  max={ee_dist_from_base.max():.4f}  std={ee_dist_from_base.std():.4f}")
                
                print(f"\n💰 Rewards:")
                print(f"   base_mobilization:    min={base_mob.min():.4f}  mean={base_mob.mean():.4f}  max={base_mob.max():.4f}")
                print(f"   position_tracking:    min={pos_track.min():.4f}  mean={pos_track.mean():.4f}  max={pos_track.max():.4f}")
                print(f"   target_distance_pen:  min={dist_pen.min():.4f}  mean={dist_pen.mean():.4f}  max={dist_pen.max():.4f}")
                
                # ========== SAMPLE ENVIRONMENTS ==========
                # Pick 3 random envs, 1 best, 1 worst
                random_env_ids = torch.randperm(self.num_envs)[:3].tolist()
                best_env_id = tracking_error.argmin().item()
                worst_env_id = tracking_error.argmax().item()
                
                # Combine and remove duplicates while preserving order
                display_env_ids = []
                display_labels = []
                
                for env_id in random_env_ids:
                    if env_id not in display_env_ids:
                        display_env_ids.append(env_id)
                        display_labels.append("RANDOM")
                
                if best_env_id not in display_env_ids:
                    display_env_ids.append(best_env_id)
                    display_labels.append("✅ BEST")
                else:
                    idx = display_env_ids.index(best_env_id)
                    display_labels[idx] = "✅ BEST"
                
                if worst_env_id not in display_env_ids:
                    display_env_ids.append(worst_env_id)
                    display_labels.append("❌ WORST")
                else:
                    idx = display_env_ids.index(worst_env_id)
                    display_labels[idx] = "❌ WORST"
                
                # Display each selected environment
                for env_id, label in zip(display_env_ids, display_labels):
                    # Get trajectory interpolation info (if available)
                    traj_info = ""
                    if hasattr(self.trajectory_manager, 'current_waypoint_idx') and \
                       hasattr(self.trajectory_manager, '_recorded_time_accum') and \
                       hasattr(self.trajectory_manager, 'waypoint_dt'):
                        current_wp = self.trajectory_manager.current_waypoint_idx[env_id].item()
                        time_accum = self.trajectory_manager._recorded_time_accum[env_id].item()
                        wp_dt = self.trajectory_manager.waypoint_dt
                        alpha = min(time_accum / wp_dt, 1.0)
                        traj_info = f" | 🎬 WP {current_wp}→{current_wp + 1} (α={alpha:.2f})"
                    
                    print(f"\n{'-'*80}")
                    print(f"Env {env_id:4d} [{label}]{traj_info}")
                    print(f"{'-'*80}")
                    print(f"  🎯 Target:       [{target_pos[env_id, 0]:7.3f}, {target_pos[env_id, 1]:7.3f}, {target_pos[env_id, 2]:7.3f}]")
                    print(f"  🟢 EE Pos:       [{ee_pos[env_id, 0]:7.3f}, {ee_pos[env_id, 1]:7.3f}, {ee_pos[env_id, 2]:7.3f}]")
                    print(f"  🚗 Base Pos:     [{base_pos_world[env_id, 0]:7.3f}, {base_pos_world[env_id, 1]:7.3f}, {base_pos_world[env_id, 2]:7.3f}]")
                    print(f"  🔧 PPR offsets:  [{base_ppr[env_id, 0]:7.3f}, {base_ppr[env_id, 1]:7.3f}, {base_ppr[env_id, 2]:7.3f}] (X, Y, θ)")
                    
                    # Calculate EE position relative to base
                    ee_relative = ee_pos[env_id] - base_pos_world[env_id]
                    print(f"  📍 EE from base: [{ee_relative[0]:7.3f}, {ee_relative[1]:7.3f}, {ee_relative[2]:7.3f}] | dist={ee_dist_from_base[env_id]:.3f}m")
                    
                    print(f"  📏 EE Error:     {tracking_error[env_id].item():.4f} m")
                    print(f"  📐 Base-Target:  {base_to_target_2d[env_id].item():.4f} m (arm reach: 0.6m)")
                    print(f"  🚗 Base Moved:   {base_displacement[env_id].item():.4f} m (from episode start)")
                    
                    # Show if base should be moving
                    if base_to_target_2d[env_id].item() > 0.6:
                        beyond_reach = base_to_target_2d[env_id].item() - 0.6
                        print(f"  ⚠️  Base SHOULD move! (target {beyond_reach:.3f}m beyond reach → penalty {10.0 * beyond_reach:.2f} pts)")
                    
                    print(f"  💰 Rewards: base_mob={base_mob[env_id].item():7.4f} | pos_track={pos_track[env_id].item():7.4f} | dist_pen={dist_pen[env_id].item():7.4f}")
                
                print(f"{'='*80}\n")
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
                print(f"[RecomoProto1TrackEE] Found EE link '{ee_link_name}' at index {self._ee_body_idx}")
            else:
                # Fallback to last body
                self._ee_body_idx = -1
                print(f"[RecomoProto1TrackEE] WARNING: EE link '{ee_link_name}' not found, using last body")
            self._ee_body_idx_initialized = True
    
    def _verify_joint_mapping(self):
        """Verify joint mapping (lazy initialization after PhysX view is ready)."""
        if not self._joint_mapping_verified and hasattr(self.robot, '_root_physx_view'):
            print("\n" + "="*80)
            print("JOINT MAPPING VERIFICATION")
            print("="*80)
            print(f"Total joints in robot: {len(self.robot.joint_names)}")
            print(f"Joint names: {self.robot.joint_names}")
            
            # Expected mapping for RecomoProto1 manipulator with PPR base
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
            print(f"[RecomoProto1TrackEE] DEBUG: First action shape = {actions.shape}")
            print(f"[RecomoProto1TrackEE] DEBUG: self.num_envs = {self.num_envs}")
            print(f"[RecomoProto1TrackEE] DEBUG: Expected shape = [{self.num_envs}, {self.cfg.num_actions}]")
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
        
        # Apply base velocity commands via direct root control (FIXED: no more joint accumulation)
        # Get current base orientation from root state (unified source of truth)
        base_quat = self.robot.data.root_quat_w
        theta = quat_to_yaw(base_quat)  # Extract yaw for body-to-world transform
        
        # Scale base actions from [-1, 1] to actual velocity limits
        base_vx_desired = base_vx * self.robot_limits["max_linear_velocity"]  # [-1.5, +1.5] m/s
        base_wz_desired = base_wz * self.robot_limits["max_angular_velocity"]  # [-2.0, +2.0] rad/s
        
        # Rate limit velocities to respect acceleration constraints
        dt = self.cfg.sim.dt * self.cfg.decimation
        max_linear_accel = self.robot_limits["max_linear_acceleration"]
        max_angular_accel = self.robot_limits["max_angular_acceleration"]
        max_vel_delta_linear = max_linear_accel * dt  # Max velocity change per step
        max_vel_delta_angular = max_angular_accel * dt
        
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
        
        # Transform body-frame velocities to world frame
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        vel_world_x = base_vx_scaled.squeeze(-1) * cos_theta  # World X velocity
        vel_world_y = base_vx_scaled.squeeze(-1) * sin_theta  # World Y velocity
        
        # CRITICAL FIX (Session 8f): Write root state atomically to prevent control conflict
        # Previously: write_root_link_velocity_to_sim() then write_root_pose_to_sim()
        # Problem: Pose write could fight/wipe velocity write, making base "numb"
        # Solution: Single atomic write of pose + velocities together
        # Reference: mobile_mm_training_playbook.md §1
        
        root_state = torch.zeros(self.num_envs, 13, device=self.device)
        
        # Position [0:3] - with Z clamped to ground
        root_state[:, 0:3] = self.robot.data.root_pos_w
        root_state[:, 2] = 0.0  # Keep chassis at ground level
        
        # Orientation [3:7] - preserve current quaternion
        root_state[:, 3:7] = self.robot.data.root_quat_w
        
        # Linear velocity [7:10]
        root_state[:, 7] = vel_world_x  # X velocity (world frame)
        root_state[:, 8] = vel_world_y  # Y velocity (world frame)
        root_state[:, 9] = 0.0  # Z velocity (always 0 for ground robot)
        
        # Angular velocity [10:13]
        root_state[:, 10] = 0.0  # Roll rate (always 0)
        root_state[:, 11] = 0.0  # Pitch rate (always 0)
        root_state[:, 12] = base_wz_scaled.squeeze(-1)  # Yaw rate
        
        # Single atomic write - no control conflict!
        self.robot.write_root_state_to_sim(root_state)
        
        # DEBUG: Print base velocity on first few steps
        if not hasattr(self, '_base_debug_count'):
            self._base_debug_count = 0
        if self._base_debug_count < 5:
            root_pos = self.robot.data.root_pos_w[0]
            print(f"\n[BASE DEBUG Step {self._base_debug_count}]")
            print(f"  base_vx action: {base_vx[0].item():.4f} -> scaled: {base_vx_scaled[0].item():.4f} m/s")
            print(f"  base_wz action: {base_wz[0].item():.4f} -> scaled: {base_wz_scaled[0].item():.4f} rad/s")
            print(f"  root_vel_w (world): [{vel_world_x[0].item():.4f}, {vel_world_y[0].item():.4f}, {base_wz_scaled[0].item():.4f}]")
            print(f"  root_pos_w: [{root_pos[0].item():.4f}, {root_pos[1].item():.4f}, {root_pos[2].item():.4f}]")
            print(f"  yaw (from quat): {theta[0].item():.4f} rad")
            self._base_debug_count += 1
    
    def _update_curriculum_stage(self):
        """Update curriculum stage with gradual weight interpolation (Session 8h).
        
        Session 8g failed with instant transition @ 50M (value function shock).
        Session 8h uses 10M linear ramp (45M-55M) to prevent instability.
        """
        if not self.use_curriculum:
            return
        
        # Estimate current training step from episode count
        # This is approximate, will be more accurate with explicit step tracking
        self.current_training_step = self.episode_length_buf.max().item() * len(self.episode_length_buf)
        
        stage_1_end = self.curriculum_stage_1_steps  # 45M
        transition_end = stage_1_end + self.task_cfg.rewards.curriculum_transition_steps  # 55M
        
        if self.current_training_step < stage_1_end:
            # Stage 1: Keep reduced weights (4.0, 12.0)
            return
        
        elif self.current_training_step < transition_end:
            # Transition: Linear interpolation 45M-55M
            if self.curriculum_stage == 1:
                print(f"\n{'='*80}")
                print(f"[Session 8h] Starting Gradual Curriculum Transition @ {self.current_training_step:,} steps")
                print(f"  45M-55M: Linear interpolation (4.0, 12.0) → (10.0, 30.0)")
                print(f"  Session 8g lesson: Instant switch @ 50M caused collapse")
                print(f"{'='*80}\n")
                self.curriculum_stage = 1.5  # Mark as transitioning
            
            # Calculate interpolation progress [0.0, 1.0]
            progress = (self.current_training_step - stage_1_end) / self.task_cfg.rewards.curriculum_transition_steps
            progress = min(1.0, max(0.0, progress))  # Clamp to [0, 1]
            
            # Linear interpolation
            stage_1_pos = self.task_cfg.rewards.curriculum_stage_1_position_weight  # 4.0
            stage_1_ori = self.task_cfg.rewards.curriculum_stage_1_orientation_weight  # 12.0
            stage_2_pos = self.task_cfg.rewards.curriculum_stage_2_position_weight  # 10.0
            stage_2_ori = self.task_cfg.rewards.curriculum_stage_2_orientation_weight  # 30.0
            
            self.reward_weights["position_tracking"] = stage_1_pos + progress * (stage_2_pos - stage_1_pos)
            self.reward_weights["orientation_tracking"] = stage_1_ori + progress * (stage_2_ori - stage_1_ori)
            
            # Log every 1M steps during transition
            if self.current_training_step % 1_000_000 < 100_000:
                print(f"[Session 8h] Transition progress: {progress*100:.1f}% @ {self.current_training_step:,} steps | "
                      f"pos={self.reward_weights['position_tracking']:.1f}, "
                      f"ori={self.reward_weights['orientation_tracking']:.1f}")
        
        else:
            # Stage 2: Full weights reached (10.0, 30.0)
            if self.curriculum_stage < 2:
                self.reward_weights["position_tracking"] = self.task_cfg.rewards.curriculum_stage_2_position_weight
                self.reward_weights["orientation_tracking"] = self.task_cfg.rewards.curriculum_stage_2_orientation_weight
                self.curriculum_stage = 2
                print(f"\n{'='*80}")
                print(f"[Session 8h] Curriculum Stage 2 Complete @ {self.current_training_step:,} steps")
                print(f"  Position weight: {self.task_cfg.rewards.curriculum_stage_1_position_weight} → {self.reward_weights['position_tracking']}")
                print(f"  Orientation weight: {self.task_cfg.rewards.curriculum_stage_1_orientation_weight} → {self.reward_weights['orientation_tracking']}")
                print(f"  Transition method: 10M gradual ramp (vs 8g's instant switch)")
                print(f"{'='*80}\n")
    
    def _apply_action(self):
        """Apply actions to the simulation (called by parent)."""
        # Update curriculum stage if enabled
        self._update_curriculum_stage()
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
            print(f"[RecomoProto1TrackEE] Joint limits initialized (ARM joints 3-8 only):")
            print(f"  Lower: {self.joint_lower_limits}")
            print(f"  Upper: {self.joint_upper_limits}")
    
    def _get_filtered_contact_forces(self) -> torch.Tensor:
        """Get contact forces filtered to exclude normal base-ground contact.
        
        We use two sensors:
        1. Chassis sensor: Reports arm-chassis collisions (sensor on base, filtered to arm contacts)
        2. Arm sensor: Reports arm-ground collisions (sensor on arm, filtered to ground contacts)
        
        This excludes:
        - Chassis ↔ Ground (normal 2-3kN support load - not reported by either sensor)
        
        Returns:
            Tensor of shape [num_envs] with maximum filtered contact force per env
        """
        # Get forces from chassis sensor (arm-chassis collisions)
        chassis_sensor = self.scene["contact_sensor"]
        chassis_forces = chassis_sensor.data.net_forces_w  # [num_envs, num_bodies, 3] or [num_envs, 3]
        
        # Calculate chassis contact magnitudes
        if len(chassis_forces.shape) == 3:
            chassis_force_mag = torch.norm(chassis_forces, dim=-1).max(dim=-1)[0]  # [num_envs]
        else:
            chassis_force_mag = torch.norm(chassis_forces, dim=-1)  # [num_envs]
        
        # Get forces from arm sensor (arm-ground collisions)
        # Check both possible key formats: direct key or via sensors dict
        arm_sensor = None
        if "arm_contact_sensor" in self.scene.sensors:
            arm_sensor = self.scene.sensors["arm_contact_sensor"]
        elif hasattr(self.scene, "arm_contact_sensor"):
            arm_sensor = self.scene.arm_contact_sensor
        
        if arm_sensor is not None:
            arm_forces = arm_sensor.data.net_forces_w
            
            # Calculate arm contact magnitudes
            if len(arm_forces.shape) == 3:
                arm_force_mag = torch.norm(arm_forces, dim=-1).max(dim=-1)[0]  # [num_envs]
            else:
                arm_force_mag = torch.norm(arm_forces, dim=-1)  # [num_envs]
            
            # Combined: max of arm-chassis OR arm-ground
            contact_force_per_env = torch.maximum(chassis_force_mag, arm_force_mag)
        else:
            # Fallback: only chassis sensor available (should not happen with new config)
            contact_force_per_env = chassis_force_mag
        
        return contact_force_per_env
    
    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Compute environment observations.
        
        Returns:
            Dictionary with "policy" key containing observation tensor
        """
        # Initialize joint limits and EE body index on first call
        self._initialize_joint_limits()
        self._initialize_ee_body_idx()
        self._verify_joint_mapping()  # Verify joint mapping on first call
        
        # Robot state - FIXED: Use root_pos_w for base (unified source of truth)
        # Base position controlled directly via root velocity commands (no joint accumulation)
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
        base_target_distance = torch.norm(target_pos[:, :2] - base_pos[:, :2], dim=-1)
        
        if self._visualization_enabled:
            prev_markers_created = self._markers_created
            self._create_markers_if_needed()
            if self._debug_vis_requested and self._markers_created and not self._debug_vis_active:
                # Markers just became available; enable visibility to honor prior request
                self._set_debug_vis_impl(True)
            if self._current_target_markers is not None:
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

        # Collision signal from contact sensors: normalized force magnitude [0, 1]
        # 0 = no collision, approaches 1 as contact forces exceed the self-collision threshold.
        # Uses the same sensor data as self_collision_penalty in _get_rewards(), giving the
        # policy an explicit "touch" signal so it can learn to avoid collisions proactively.
        raw_contact = self._get_filtered_contact_forces()  # [num_envs]
        coll_threshold = self.reward_weights.get("self_collision_threshold", 1.0)
        contact_obs = torch.clamp(raw_contact / (coll_threshold * 10.0), 0.0, 1.0).unsqueeze(-1)  # [num_envs, 1]

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
            contact_forces=contact_obs,   # Collision severity [num_envs, 1]: 0=clean, →1 at heavy collision
            min_obstacle_dist=None,  # No obstacles in current scene; enable via ObstacleConfig.enable_obstacles
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
        base_target_distance = torch.norm(target_pos[:, :2] - base_pos[:, :2], dim=-1)
        
        joint_pos = self.robot.data.joint_pos
        arm_joint_pos = joint_pos[:, 3:9]  # ARM joints only
        joint_vel = self.robot.data.joint_vel  # All joint velocities (needed for monitoring)
        arm_joint_vel = joint_vel[:, 3:9]  # ARM joint velocities
        
        # Get contact forces from ContactSensor (Isaac Lab 2.2.0 pattern)
        # Use filtered contact forces (excludes base-ground static load)
        contact_force_mag_per_env = self._get_filtered_contact_forces()  # [num_envs]
        max_force = contact_force_mag_per_env.max().item()
        
        if not hasattr(self, '_contact_force_checked'):
            print(f"\n{'='*80}")
            print(f"CONTACT FORCE API VERIFICATION (Filtered)")
            print(f"{'='*80}")
            print(f"Max filtered contact force: {max_force:.4f} N")
            if max_force < 0.001:
                print(f"⚠️  WARNING: Filtered contact forces are zero!")
                print(f"   Either no collisions or filtering is too aggressive!")
            else:
                print(f"✅ Filtered contact forces detected - collision detection active!")
            print(f"   (Base-ground support load excluded)")
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
                print(f"   Max filtered contact force: {max_force:.2f} N (threshold: {collision_threshold:.2f} N)")
                print(f"   (Base-ground contact excluded from detection)")
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
        
        # ============================================================================
        # REACHABILITY-GUIDED BASE PLANNING
        # ============================================================================
        # Check if target EE position is reachable from current base position
        # If not reachable → encourage base movement toward target
        reachability_bonus = torch.zeros(self.num_envs, device=self.device)
        base_direction_reward = torch.zeros(self.num_envs, device=self.device)
        
        if self.reach_map is not None:
            # FIXED: Use root state (world pose) instead of joint offsets
            # Extract base world pose from root state
            base_world_x = base_pos[:, 0]  # Already have base_pos from root_pos_w
            base_world_y = base_pos[:, 1]
            base_world_yaw = quat_to_yaw(base_quat)  # Extract yaw from root quaternion
            base_pose = torch.stack([base_world_x, base_world_y, base_world_yaw], dim=1)  # [N, 3]
            
            # Transform target EE position from world frame to arm base frame
            target_in_arm_frame = self.reach_map.world_to_arm_frame(target_pos, base_pose)
            
            # Check reachability
            is_reachable = self.reach_map.query(target_in_arm_frame, tolerance=0.1)  # 10cm tolerance
            workspace_distance = self.reach_map.distance_to_workspace(target_in_arm_frame)
            
            # Count reachable/unreachable for logging
            n_reachable = is_reachable.sum().item()
            n_unreachable = (~is_reachable).sum().item()
            
            # === CASE 1: Target IS reachable from current base position ===
            if is_reachable.any():
                # Bonus for being in a good base position (Session 7c: Reduced to 1.0 from 2.0 - was too dominant)
                reachability_bonus[is_reachable] = 1.0
                
                # Optional: Get best arm config as IK hint (future enhancement)
                # arm_configs, _ = self.reach_map.get_best_configs(target_in_arm_frame[is_reachable])
            
            # === CASE 2: Target is NOT reachable from current base position ===
            if (~is_reachable).any():
                # Compute direction to target in world X-Y plane
                target_xy = target_pos[~is_reachable, :2]  # [M, 2]: world X, Y
                base_xy = base_pose[~is_reachable, :2]     # [M, 2]: world X, Y
                
                # Direction vector from base to target
                direction_to_target = target_xy - base_xy  # [M, 2]
                distance_to_target_xy = torch.norm(direction_to_target, dim=-1, keepdim=True)  # [M, 1]
                direction_normalized = direction_to_target / (distance_to_target_xy + 1e-6)  # [M, 2]
                
                # Transform base velocity from body frame to world frame
                # Base velocity in body frame: [vx_body, 0, wz] (differential drive can't strafe)
                base_theta = base_pose[~is_reachable, 2]  # [M]
                base_vx_body = commanded_linear[~is_reachable, 0]  # [M]
                
                # World frame velocity (rotation matrix applied to body velocity)
                base_vx_world = base_vx_body * torch.cos(base_theta)  # X component in world
                base_vy_world = base_vx_body * torch.sin(base_theta)  # Y component in world
                base_vel_xy_world = torch.stack([base_vx_world, base_vy_world], dim=-1)  # [M, 2]
                
                # Compute alignment: dot product of velocity with desired direction
                alignment = (base_vel_xy_world * direction_normalized).sum(dim=-1)  # [M]
                
                # Reward moving in correct direction (Session 7c: Reduced to 1.5 from 3.0)
                base_direction_reward[~is_reachable] = 1.5 * torch.clamp(alignment, min=0.0)
                
                # Bonus for higher speed when moving in right direction (Session 7c: Reduced to 0.5 from 1.0)
                speed_xy = torch.norm(base_vel_xy_world, dim=-1)  # [M]
                base_direction_reward[~is_reachable] += 0.5 * speed_xy * torch.clamp(alignment, min=0.0)
            
            # Log reachability statistics (every 100 steps to avoid spam)
            if not hasattr(self, '_reach_log_step'):
                self._reach_log_step = 0
            
            # Store stats for training monitor callback
            avg_alignment = alignment.mean().item() if n_unreachable > 0 else 0.0
            avg_distance = distance_to_target_xy.mean().item() if n_unreachable > 0 else 0.0
            self._last_reachability_stats = {
                'reachable': n_reachable,
                'unreachable': n_unreachable,
                'total': self.num_envs,
                'avg_alignment': avg_alignment,
                'avg_distance': avg_distance
            }
            
            if self._reach_log_step % 100 == 0:
                print(f"\n[Reachability Stats] Step {self._reach_log_step}")
                print(f"  Reachable: {n_reachable}/{self.num_envs} envs")
                print(f"  Unreachable: {n_unreachable}/{self.num_envs} envs")
                if n_unreachable > 0:
                    print(f"  Avg base→target alignment: {avg_alignment:.3f}")
                    print(f"  Avg base→target distance: {avg_distance:.3f} m")
            
            self._reach_log_step += 1
        else:
            workspace_distance = None
        
        # BUGFIX: Convert commanded velocities from body frame to world frame
        # Most reward functions expect world-frame velocities (lateral_motion_penalty, 
        # velocity_limit_penalty, etc.) but commanded_vel is in body frame.
        # Only base_target_alignment_reward expects body frame, so we'll handle that specially.
        
        # Extract yaw from quaternion for rotation
        w, x, y, z = base_quat[:, 0], base_quat[:, 1], base_quat[:, 2], base_quat[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))
        
        # Current commanded velocity: body → world
        vx_body = commanded_linear[:, 0]
        commanded_linear_world = torch.zeros_like(commanded_linear)
        commanded_linear_world[:, 0] = vx_body * torch.cos(yaw)  # X component in world
        commanded_linear_world[:, 1] = vx_body * torch.sin(yaw)  # Y component in world
        commanded_linear_world[:, 2] = commanded_linear[:, 2]     # Z unchanged (usually 0)
        
        # Previous commanded velocity: body → world
        vx_body_prev = prev_commanded_linear[:, 0]
        prev_commanded_linear_world = torch.zeros_like(prev_commanded_linear)
        prev_commanded_linear_world[:, 0] = vx_body_prev * torch.cos(yaw)
        prev_commanded_linear_world[:, 1] = vx_body_prev * torch.sin(yaw)
        prev_commanded_linear_world[:, 2] = prev_commanded_linear[:, 2]
        
        # Previous commanded acceleration: body → world
        ax_body = prev_commanded_linear_accel[:, 0]
        prev_commanded_linear_accel_world = torch.zeros_like(prev_commanded_linear_accel)
        prev_commanded_linear_accel_world[:, 0] = ax_body * torch.cos(yaw)
        prev_commanded_linear_accel_world[:, 1] = ax_body * torch.sin(yaw)
        prev_commanded_linear_accel_world[:, 2] = prev_commanded_linear_accel[:, 2]
        
        # SESSION 8i: Calculate current orientation error for distance-gated rewards
        dot_product_for_reward = torch.sum(ee_quat * target_quat, dim=-1).abs()
        dot_product_for_reward = torch.clamp(dot_product_for_reward, 0.0, 1.0)
        current_ee_ori_error = 2 * torch.acos(dot_product_for_reward)  # [num_envs] - angular error in radians
        
        # Get EE angular velocity for smoothness penalty
        ee_ang_vel = self.robot.data.body_ang_vel_w[:, self._ee_body_idx, :]  # [num_envs, 3]
        
        # Compute rewards with all new constraint penalties
        # Use COMMANDED velocities (now in world frame) for penalty calculation to avoid penalizing simulation artifacts
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
            base_lin_vel=commanded_linear_world,  # BUGFIX: Now in world frame
            base_ang_vel=commanded_ang,
            base_quat=base_quat,  # Base orientation for lateral penalty
            base_target_distance=base_target_distance,
            joint_pos=arm_joint_pos,  # ARM joints only [6]
            joint_vel=arm_joint_vel,  # ARM joint velocities only [6]
            prev_base_pos=self.prev_base_pos,  # NEW: Previous base position
            prev_base_lin_vel=prev_commanded_linear_world,  # BUGFIX: Now in world frame
            prev_joint_vel=self.prev_joint_vel[:, 3:9],  # ARM joint velocities only [6]
            prev_base_accel=prev_commanded_linear_accel_world,  # BUGFIX: Now in world frame
            joint_lower=self.joint_lower_limits,  # ARM limits [6]
            joint_upper=self.joint_upper_limits,  # ARM limits [6]
            robot_limits=self.robot_limits,
            contact_forces=contact_force_mag_per_env[:, None, None],  # Shape [num_envs, 1, 1] to match expected [num_envs, num_bodies, 3]
            dt=self.control_dt,
            weights=self.reward_weights,
            workspace_distance=workspace_distance,
            min_obstacle_dist=None,  # No obstacles in scene; requires ObstacleConfig.enable_obstacles + obstacle prims
            prev_ori_error=self.prev_ee_ori_error,  # SESSION 8i: Previous orientation error
            current_ori_error=current_ee_ori_error,  # SESSION 8i: Current orientation error
            ee_ang_vel=ee_ang_vel,  # SESSION 8i: EE angular velocity
        )

        if workspace_distance is None:
            workspace_distance_log = torch.zeros_like(base_target_distance)
        else:
            workspace_distance_log = workspace_distance
        self._workspace_distance_buf = workspace_distance_log.clone()
        self._base_target_distance_buf = base_target_distance.clone()
        
        # Update history for next step - store COMMANDED velocities for consistent penalty calculation
        self.prev_tracking_error = torch.norm(target_pos - ee_pos, dim=-1)
        self.prev_ee_ori_error = current_ee_ori_error.clone()  # SESSION 8i: Store orientation error for next step
        
        # Store tracking errors for evaluation/logging
        self.ee_pos_error_buf = target_pos - ee_pos  # [num_envs, 3] - vector error (x, y, z)
        
        # Compute orientation error (same as orientation_tracking_reward)
        dot_product = torch.sum(ee_quat * target_quat, dim=-1).abs()
        dot_product = torch.clamp(dot_product, 0.0, 1.0)
        self.ee_ori_error_buf = 2 * torch.acos(dot_product)  # [num_envs] - angular error in radians
        
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
        
        # === ADD REACHABILITY-GUIDED REWARDS ===
        rewards += reachability_bonus + base_direction_reward
        
        # Log reachability reward components
        if self.reach_map is not None:
            self.extras["reward_components"]["reachability_bonus"] = reachability_bonus.mean().item()
            self.extras["reward_components"]["base_direction_reward"] = base_direction_reward.mean().item()
        
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
        
        # Check for self-collision (CRITICAL for RecomoProto1 manipulator!)
        if self.task_cfg.terminate_on_self_collision:
            # Get filtered contact forces (excludes base-ground static load)
            contact_force_mag = self._get_filtered_contact_forces()  # [num_envs]
            
            # Terminate if filtered contact force exceeds threshold
            # This will only trigger on arm-ground or arm-base collisions, not normal base support
            terminated |= contact_force_mag > self.task_cfg.self_collision_termination_threshold
        
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
        
        # Reset trajectory phase BEFORE getting first waypoint
        self.trajectory_manager.reset(env_ids)
        
        # DEBUG: Verify trajectory was reset correctly
        if len(env_ids) > 0:
            test_wp_idx = self.trajectory_manager.current_waypoint_idx[env_ids[0]].item()
            test_time = self.trajectory_manager._recorded_time_accum[env_ids[0]].item()
            print(f"[DEBUG] After reset - Env {env_ids[0].item()}: waypoint_idx={test_wp_idx}, time_accum={test_time:.4f}")

        # Clear waypoint visualization state for the reset environments
        if hasattr(self, "_visited_waypoint_masks"):
            for env_id in env_ids.cpu().tolist():
                self._visited_waypoint_masks[env_id] = None
        if hasattr(self, "_last_waypoint_idx"):
            self._last_waypoint_idx[env_ids] = -1

        if self._visualization_enabled:
            self._create_markers_if_needed()
        
        # CRITICAL: Get first waypoint AFTER trajectory reset
        # This ensures we're positioning base relative to trajectory start, not mid-trajectory
        first_target_pos, _ = self.trajectory_manager.get_target_pose()
        
        # DEBUG: Print what we got from get_target_pose AND what's in recorded_positions
        if len(env_ids) > 0 and self.trajectory_manager.recorded_positions is not None:
            # Safety check: ensure env_ids is not empty and contains valid indices
            if env_ids.numel() > 0:
                env_idx = env_ids[0].item()
                from_get_target = first_target_pos[env_idx].cpu().numpy()
                from_recorded = self.trajectory_manager.recorded_positions[env_idx, 0].cpu().numpy()
                print(f"[DEBUG] Env {env_idx}:")
                print(f"  get_target_pose():    [{from_get_target[0]:.3f}, {from_get_target[1]:.3f}, {from_get_target[2]:.3f}]")
                print(f"  recorded_positions:   [{from_recorded[0]:.3f}, {from_recorded[1]:.3f}, {from_recorded[2]:.3f}]")
                print(f"  waypoint_idx:         {self.trajectory_manager.current_waypoint_idx[env_idx].item()}")
                print(f"  time_accum:           {self.trajectory_manager._recorded_time_accum[env_idx].item():.4f}")
        
        # Set base position with offset from target, accounting for arm kinematics
        # FK analysis (all joints at zero): EE is at [0.1415, 0.2405, 0.9465] in base frame
        # We offset base 44cm behind and 24cm left of target XY to allow 30cm forward reach
        # This gives natural reaching posture instead of forcing arm backwards
        new_root_state = self.robot.data.default_root_state[env_ids].clone()
        new_root_state[:, 0] = first_target_pos[env_ids, 0] - 0.4415  # X: 30cm forward reach room
        new_root_state[:, 1] = first_target_pos[env_ids, 1] - 0.2405  # Y: compensate for lateral offset
        new_root_state[:, 2] = 0.0  # Ground level (matches runtime clamping line 1162)
        # Keep orientation from default (facing forward)
        
        # Reset velocities to zero
        new_root_state[:, 7:10] = 0.0  # Linear velocity
        new_root_state[:, 10:13] = 0.0  # Angular velocity
        
        # Apply the new root state
        self.robot.write_root_state_to_sim(new_root_state, env_ids=env_ids)
        
        # FIXED: Set PPR joints to ZERO (they are offsets, not world positions!)
        # Initialize base joint indices if needed
        if not hasattr(self, '_base_joint_ids'):
            base_joint_names = ["joint_x", "joint_y", "joint_theta"]
            self._base_joint_ids = []
            for name in base_joint_names:
                if name in self.robot.joint_names:
                    idx = self.robot.joint_names.index(name)
                    self._base_joint_ids.append(idx)
            self._base_joint_ids = torch.tensor(self._base_joint_ids, device=self.device)
        
        # Set base joints to zero offset (root_pos_w controls world position directly)
        base_joint_pos = torch.zeros(len(env_ids), 3, device=self.device)
        self.robot.set_joint_position_target(
            base_joint_pos, 
            joint_ids=self._base_joint_ids,
            env_ids=env_ids
        )
        
        # Print actual base position and target for debugging
        print(f"[RESET] Env {env_ids[0].item() if len(env_ids) > 0 else 'N/A'}: Base=[{new_root_state[0, 0]:.3f}, {new_root_state[0, 1]:.3f}, {new_root_state[0, 2]:.3f}], Target=[{first_target_pos[env_ids[0], 0]:.3f}, {first_target_pos[env_ids[0], 1]:.3f}, {first_target_pos[env_ids[0], 2]:.3f}], Reach=0.30m forward")
        
        # NEW: Store episode start base position for movement tracking
        if not hasattr(self, '_episode_start_base_pos'):
            self._episode_start_base_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._episode_start_base_pos[env_ids] = new_root_state[:, 0:3]
        
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
