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
    max_linear_jerk: float = 100.0  # m/s^3 (Session 7b: 50→100, allow agile base reactions for reachability-guided movement)
    
    # Arm joint limits
    max_joint_velocity: float = 2.0  # rad/s (motor speed)
    max_joint_acceleration: float = 10.0  # rad/s^2
    
    # Joint position limits will be read from USD
    enforce_joint_limits: bool = True
    joint_limit_margin: float = 0.1  # radians (stay this far from limits)


@dataclass
class RewardWeights:
    """Reward term weights for the tracking task.
    
    Session 8 Configuration (Based on Session 7d evaluation):
    - Dramatically boost orientation tracking (2.0 → 75.0, 37.5× increase)
    - Significantly reduce velocity penalty (5.0 → 1.5, 70% reduction)
    - Reduce jerk penalty (0.05 → 0.01, 80% reduction) to stop crushing movement
    - Boost base mobilization incentives (250.0 → 400.0)
    - Target reward/penalty ratio: 44:1 (rewards dominate)
    
    Expected Results:
    - Position error: 3.64m → < 0.2m
    - Orientation error: 140.7° → < 10°
    - Episode reward: -5,120 → POSITIVE
    - Base velocity: < 0.01 m/s → > 0.2 m/s
    """
    
    # ========================================
    # TRACKING REWARDS (Make these DOMINANT)
    # ========================================
    position_tracking: float = 150.0  # Session 8: INCREASED 100→150 (50% boost)
    orientation_tracking: float = 75.0  # Session 8: INCREASED 2.0→75.0 (37.5× boost!) - NOW 50% OF POSITION
    progress_bonus: float = 5.0  # Session 8: INCREASED 1.0→5.0 (5× boost)
    base_progress_reward: float = 400.0  # Session 8: INCREASED 250→400 (60% boost) - scales base_mobilization_reward()
    base_target_alignment: float = 30.0  # Session 8: INCREASED 10→30 (3× boost)
    target_distance_penalty: float = 1.0  # Session 8: REDUCED 3.0→1.0 (67% reduction) - allow exploration
    excessive_base_movement_penalty: float = 5.0  # Session 8: REDUCED 10→5 (50% reduction)
    
    # ========================================
    # MOTION QUALITY PENALTIES (Reduce these)
    # ========================================
    action_magnitude: float = 0.002  # Session 8: REDUCED 0.005→0.002 (60% reduction)
    action_rate: float = 0.005  # Session 8: REDUCED 0.01→0.005 (50% reduction)
    action_smoothness: float = 0.05  # Session 8: REDUCED 0.15→0.05 (67% reduction) - was -1.72/step
    
    # ========================================
    # CONSTRAINT VIOLATIONS (Much gentler)
    # ========================================
    velocity_limit_penalty: float = 1.5  # Session 8: REDUCED 5.0→1.5 (70% reduction!) - was -15.5/step
    acceleration_limit_penalty: float = 1.5  # Session 8: REDUCED 5.0→1.5 (70% reduction)
    jerk_limit_penalty: float = 0.01  # Session 8: REDUCED 0.05→0.01 (80% reduction!) - was -14.0/step
    joint_limit_penalty: float = 5.0  # Session 8: REDUCED 10.0→5.0 (50% reduction)
    lateral_motion_penalty: float = 1.0  # Session 8: REDUCED 2.0→1.0 (50% reduction)
    
    # ========================================
    # SAFETY PENALTIES (Keep reasonable)
    # ========================================
    self_collision_penalty: float = 1.0  # Session 8: INCREASED 0.5→1.0 (2× boost)
    collision_penalty: float = 10.0  # External collisions (not used for now)
    stability_penalty: float = 0.2  # Session 8: INCREASED 0.1→0.2 (2× boost)
    
    # Self-collision detection settings (filtered to exclude base-ground contact)
    self_collision_threshold: float = 50.0  # Newtons - arm impact threshold (was 1.0, too sensitive for base-ground load)
    self_collision_continuous: bool = True  # Continuous vs binary penalty
    
    # Obstacle avoidance
    min_obstacle_distance_weight: float = 1.0
    safety_radius: float = 0.2  # meters


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
