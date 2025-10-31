"""Configuration dataclasses for mobile manipulator tracking task."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class TrajectoryConfig:
    """Configuration for reference trajectory generation."""
    
    # Trajectory type
    type: Literal["line", "circle", "figure_eight", "recorded", "multi_recorded"] = "circle"
    
    # Parametric trajectory settings
    amplitude: float = 0.5  # meters
    speed: float = 0.2  # meters/second
    height: float = 1.0  # meters (z-coordinate)
    
    # Recorded trajectory settings
    waypoint_file: str | None = None
    loop_trajectory: bool = True
    
    # Multi-recorded trajectory settings
    trajectory_dir: str = "trajectoryToLearn/world_json"
    trajectory_pattern: str = "**/*.json"
    trajectory_filter_indices: list[int] | None = None  # Filter to specific indices
    max_trajectories: int | None = None  # Limit number of trajectories
    
    # Curriculum settings
    enable_curriculum: bool = True
    initial_amplitude: float = 0.3
    final_amplitude: float = 1.0
    curriculum_stages: int = 5


@dataclass
class ObstacleConfig:
    """Configuration for obstacle spawning and randomization."""
    
    # Obstacle placement
    enable_obstacles: bool = False
    num_obstacles: int = 3
    min_distance_from_robot: float = 1.0
    max_distance_from_robot: float = 3.0
    obstacle_radius_range: tuple[float, float] = (0.1, 0.3)
    
    # Randomization
    randomize_per_reset: bool = True
    seed: int | None = None


@dataclass
class RobotLimits:
    """Physical limits and constraints for the robot."""
    
    # Mobile base limits (differential drive - no lateral motion)
    max_linear_velocity: float = 1.5  # m/s
    max_angular_velocity: float = 2.0  # rad/s (yaw rate)
    max_linear_acceleration: float = 5.0  # m/s^2
    max_angular_acceleration: float = 10.0  # rad/s^2 (yaw acceleration)
    max_linear_jerk: float = 100.0  # m/s^3; high limit keeps chassis agile when closing reach gaps
    
    # Arm joint limits
    max_joint_velocity: float = 2.0  # rad/s (motor speed)
    max_joint_acceleration: float = 10.0  # rad/s^2
    
    # Joint position limits will be read from USD
    enforce_joint_limits: bool = True
    joint_limit_margin: float = 0.1  # radians (stay this far from limits)


@dataclass
class RewardWeights:
    """Reward term weights for the tracking task.

Session 8c-v2 introduced distance-aware reachability shaping:
- Soft/hard workspace margins (0.20 m / 0.60 m) separate gentle bonuses from steep penalties.
- Base mobilisation rewards capture chassis-only progress with a configurable cap.
- Legacy distance penalties remain with light weights so gradients persist when the robot falls far behind.

The groups below mirror the structure used in rewards.compute_combined_reward().
"""
    
    # ========================================
    # TRACKING REWARDS (Make these DOMINANT)
    # ========================================
    position_tracking: float = 200.0  # Dominant weight for EE position accuracy
    orientation_tracking: float = 100.0  # Secondary weight for EE orientation accuracy
    progress_bonus: float = 5.0  # Reward incremental error reductions between steps
    base_progress_reward: float = 450.0  # Credits chassis motion when it closes the base-target gap
    base_target_alignment: float = 30.0  # Rewards velocity that points toward an unreachable target
    target_distance_penalty: float = 1.0  # Legacy penalty; discounted 90% while the base is moving
    
    # ========================================
    # BASE COORDINATION (Session 8c-v3)
    # ========================================
    reachability_maintenance_reward: float = 40.0  # Bonus when the target remains inside the soft margin
    reachability_distance_weight: float = 80.0  # Penalty weight for exceeding the hard workspace margin
    reachability_soft_margin: float = 0.2  # Soft margin radius (m) for positive shaping
    reachability_hard_margin: float = 0.6  # Hard cutoff radius (m) that triggers quadratic penalties
    base_overshoot_penalty: float = 20.0  # Penalises chassis that rush past the target
    excessive_base_movement_penalty: float = 10.0  # Discourages back-and-forth oscillations
    mobilization_progress_cap: float = 0.35  # Maximum distance progress credited per step (meters)
    
    # ========================================
    # MOTION QUALITY PENALTIES (Reduce these)
    # ========================================
    action_magnitude: float = 0.002  # Penalises high torque commands
    action_rate: float = 0.005  # Penalises rapid action changes
    action_smoothness: float = 0.05  # Penalises jerk-like behaviour in the control sequence
    
    # ========================================
    # CONSTRAINT VIOLATIONS (Much gentler)
    # ========================================
    velocity_limit_penalty: float = 1.0  # Activates when base or joints exceed nominal velocity limits
    acceleration_limit_penalty: float = 1.5  # Penalises linear acceleration spikes
    jerk_limit_penalty: float = 0.005  # Soft penalty on large changes in acceleration
    joint_limit_penalty: float = 5.0  # Keeps joints away from hard stops
    lateral_motion_penalty: float = 1.0  # Penalises sideways slipping beyond differential-drive kinematics
    
    # ========================================
    # SAFETY PENALTIES (Keep reasonable)
    # ========================================
    self_collision_penalty: float = 1.0  # Penalises link-on-link impact excluding the base-ground contact
    collision_penalty: float = 10.0  # External collisions (not used for now)
    stability_penalty: float = 0.2  # Keeps linear/angular velocity within stable ranges
    
    # Self-collision detection settings (filtered to exclude base-ground contact)
    self_collision_threshold: float = 50.0  # Newtons - arm impact threshold (was 1.0, too sensitive for base-ground load)
    self_collision_continuous: bool = True  # Continuous vs binary penalty
    
    # Obstacle avoidance
    min_obstacle_distance_weight: float = 1.0
    safety_radius: float = 0.2  # meters

    # ========================================
    # AUXILIARY TRACKING SHAPING
    # ========================================
    position_distance_penalty: float = 40.0  # Linear fallback penalty so gradients remain informative when far


@dataclass
class DomainRandomization:
    """Domain randomization settings."""
    
    enable: bool = False
    
    # Mass randomization (fraction of nominal)
    mass_range: tuple[float, float] = (0.8, 1.2)
    
    # Friction randomization
    friction_range: tuple[float, float] = (0.5, 1.5)
    
    # Torque limit randomization (fraction of nominal)
    torque_limit_range: tuple[float, float] = (0.9, 1.1)


@dataclass
class MobileMMTrackConfig:
    """Master configuration for MobileMMTrackEE task."""
    
    # Sub-configurations
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    obstacles: ObstacleConfig = field(default_factory=ObstacleConfig)
    rewards: RewardWeights = field(default_factory=RewardWeights)
    robot_limits: RobotLimits = field(default_factory=RobotLimits)
    domain_rand: DomainRandomization = field(default_factory=DomainRandomization)
    
    # Episode settings
    episode_length_s: float = 20.0
    decimation: int = 10  # Control @ 20Hz (200Hz physics / 10 = 20Hz control)
    
    # Trajectory timing (must match recorded trajectory waypoint spacing)
    trajectory_dt: float = 0.1  # seconds (100ms waypoint spacing)
    
    # Initial state randomization
    randomize_initial_joint_positions: bool = True
    initial_joint_noise_std: float = 0.1  # radians
    
    # Observation settings
    use_lookahead: bool = True
    lookahead_steps: int = 3
    lookahead_dt: float = 0.1  # seconds
    include_action_history: bool = True
    action_history_length: int = 2
    
    # Termination conditions
    terminate_on_self_collision: bool = True  # CRITICAL: End episode if robot hits itself
    self_collision_termination_threshold: float = 10.0  # Newtons (higher than penalty threshold)
    terminate_on_collision: bool = False  # External collisions (not used)
    terminate_on_tracking_error: bool = True
    max_tracking_error: float = 2.0  # meters


