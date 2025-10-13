"""Configuration dataclasses for mobile manipulator tracking task."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class TrajectoryConfig:
    """Configuration for reference trajectory generation."""
    
    # Trajectory type
    type: Literal["line", "circle", "figure_eight", "recorded"] = "circle"
    
    # Parametric trajectory settings
    amplitude: float = 0.5  # meters
    speed: float = 0.2  # meters/second
    height: float = 1.0  # meters (z-coordinate)
    
    # Recorded trajectory settings
    waypoint_file: str | None = None
    loop_trajectory: bool = True
    
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
class RewardWeights:
    """Reward term weights for the tracking task."""
    
    # Tracking rewards
    position_tracking: float = 10.0
    orientation_tracking: float = 2.0
    progress_bonus: float = 1.0
    
    # Penalties
    action_magnitude: float = 0.01
    action_rate: float = 0.01
    collision_penalty: float = 10.0
    stability_penalty: float = 0.1
    
    # Safety
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
    domain_rand: DomainRandomization = field(default_factory=DomainRandomization)
    
    # Episode settings
    episode_length_s: float = 20.0
    decimation: int = 4  # Control frequency = sim_frequency / decimation
    
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
    terminate_on_collision: bool = True
    terminate_on_tracking_error: bool = True
    max_tracking_error: float = 2.0  # meters
