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
    min_duration_seconds: float = 5.0  # Reject recorded trajectories shorter than this duration

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
    num_obstacles: int = 1
    disc_position_xy: tuple[float, float] = (0.0, 0.5)  # Local env-frame XY, on the default circle trajectory
    disc_radius: float = 0.18
    disc_height: float = 0.08
    disc_position_x_range: tuple[float, float] = (-0.35, 0.35)
    disc_position_y_range: tuple[float, float] = (0.45, 1.0)
    robot_footprint_radius: float = 0.35
    min_start_clearance: float = 0.10

    # Randomization
    randomize_per_reset: bool = True
    seed: int | None = None  # Reserved for deterministic obstacle sampling

    # Termination
    terminate_on_collision: bool = True
    collision_grace_steps: int = 12  # Ignore reset-settling transients for 0.6s at 20 Hz control


@dataclass
class RobotLimits:
    """Physical limits and constraints for the robot."""

    # Mobile base limits. max_linear_velocity applies to body-frame vx and vy.
    max_linear_velocity: float = 1.5  # m/s per planar axis
    max_angular_velocity: float = 2.0  # rad/s (yaw rate)
    max_linear_acceleration: float = 5.0  # m/s^2
    max_angular_acceleration: float = 10.0  # rad/s^2 (yaw acceleration)
    max_linear_jerk: float = 100.0  # m/s^3; high limit keeps chassis agile when closing reach gaps

    # Arm joint limits
    max_joint_velocity: float = 2.0  # rad/s (motor speed)
    max_joint_acceleration: float = 6.0  # rad/s^2; conservative target slew for early Proto2 policy training

    # Joint position limits will be read from USD
    enforce_joint_limits: bool = True
    joint_limit_margin: float = 0.1  # radians (stay this far from limits)


@dataclass
class RewardWeights:
    """Reward term weights for the tracking task.

Session 8i implements distance-gated orientation rewards (OBS: 70→73 dims):
- STRATEGY: Separate reach-mode (far, ori_weight=4.0) from align-mode (close, ori_weight=30.0)
- Distance gate threshold: 0.7m (comfort zone for precise orientation tuning)
- New observations: axis-angle error (+3 dims) provides shortest rotation path
- Goal: Improve orientation 135°→80-100° while maintaining position ~237cm
- Baseline: Session 8h @ 40M (237.3cm pos, 135.1° ori)
- Reference: docs/training_sessions/session_8i/SESSION_8I_IMPLEMENTATION_PLAN.md

Session 8f implements distance-gated penalty system + playbook fixes:
- CRITICAL: Distance-gated penalties (far=mobilization, near=precision)
- Two-zone linear reachability (0.35-0.5-0.6m with plateau, simpler than 8e's bell curve)
- Control conflict fix (atomic root state write)
- Heading cue in observations (+2 dims: sin/cos of base→target yaw error)
- Reference: mobile_mm_training_playbook.md §1-3

Session 8e attempted bell-shaped comfort zone but failed:
- Reachability bonus collapsed from 7.06 → 0.79 (89% drop!)
- Workspace distance drifted from 0.52m @ 50M to 0.58m @ 73M
- Narrow bell peak too brittle for dynamic tracking
- Root cause: Penalties fought mobilization at all distances

The groups below mirror the structure used in rewards.compute_combined_reward().
"""

    # ========================================
    # TRACKING REWARDS (Make these DOMINANT)
    # ========================================
    position_tracking: float = 200.0  # Dominant weight for EE position accuracy
    orientation_tracking: float = 200.0  # Emphasize EE orientation accuracy alongside reachability
    progress_bonus: float = 5.0  # Reward incremental error reductions between steps
    base_progress_reward: float = 450.0  # Credits chassis motion when it closes the base-target gap
    base_target_alignment: float = 50.0  # Stronger reward for goal-directed chassis motion
    target_distance_penalty: float = 1.0  # Legacy penalty; discounted 90% while the base is moving

    # ========================================
    # BASE COORDINATION (Session 8g - Expanded Workspace)
    # ========================================
    reachability_maintenance_reward: float = 40.0  # Bonus when in optimal working zone (KEEP at 40!)
    reachability_distance_weight: float = 30.0  # Penalty weight for exceeding hard margin (REDUCED from 60)
    reachability_soft_margin: float = 0.2  # Two-zone linear model: soft margin width (±0.2m around optimal)
    reachability_hard_margin: float = 0.7  # Hard cutoff radius (m) - EXPANDED from 0.6m to match FK workspace
    reachability_optimal_distance: float = 0.6  # Optimal working distance - FK median (was 0.5m)
    inner_margin_penalty: float = 15.0  # Penalty for base getting too close (<0.35m)
    inner_margin_min_distance: float = 0.35  # Minimum comfortable working distance
    base_overshoot_penalty: float = 30.0  # Penalises chassis that rush past the target
    excessive_base_movement_penalty: float = 10.0  # Discourages back-and-forth oscillations
    mobilization_progress_cap: float = 0.35  # Maximum distance progress credited per step (meters)

    # ========================================
    # CURRICULUM LEARNING (Session 8h - Balanced Weights + Gradual Transition)
    # ========================================
    use_curriculum: bool = True  # Enable two-stage curriculum
    curriculum_stage_1_steps: int = 45_000_000  # Stage 1: 0-45M steps (transition starts earlier)
    curriculum_transition_steps: int = 10_000_000  # 10M gradual ramp (45M-55M)

    # Stage 1: Balanced ratio from start (40% of final, maintains 1:3 ratio)
    # Session 8g FAILED with (5.0, 15.0) - orientation under-trained despite 1:3 ratio
    # Session 8h FIX: (4.0, 12.0) - both reduced proportionally, orientation gets adequate signal
    curriculum_stage_1_position_weight: float = 4.0  # 40% of 10.0 (was 5.0 = 50%)
    curriculum_stage_1_orientation_weight: float = 12.0  # 40% of 30.0 (was 15.0 = 50%)

    # Stage 2: Full precision tracking (all trajectories, full weights)
    curriculum_stage_2_position_weight: float = 10.0  # Restore full weight
    curriculum_stage_2_orientation_weight: float = 30.0  # Restore full weight
    # Optional boost if orientation still struggles @ 20M: increase to 40.0 for 4:1 ratio

    # ========================================
    # STABILITY MONITORING (Session 8h - Auto-pause on Instability)
    # ========================================
    enable_auto_pause: bool = True  # Pause training on instability (8g collapsed @ 100M)
    kl_threshold: float = 0.1  # Pause if KL divergence exceeds this
    variance_threshold: float = -0.3  # Pause if explained variance drops below this (allow negative in early training)
    checkpoint_frequency_steps: int = 2_000_000  # Save every 2M steps (finer granularity than 8g)

    # ========================================
    # SESSION 8i: DISTANCE-GATED ORIENTATION REWARDS
    # ========================================
    # Strategy: Separate "reaching mode" from "alignment mode" using distance threshold
    # - Far from target (>0.7m): Medium orientation weight (8.0), focus on base mobilization
    # - Close to target (<0.7m): High orientation weight (30.0), focus on precise alignment
    # Goal: Improve orientation from 135° → 80-100° while maintaining position ~237cm
    # Session 8i v1 FAILED @ 29M: 7.5x jump (4.0→30.0), hard threshold caused policy oscillation
    # Session 8i v2 FAILED @ 21M: 3.75x jump (8.0→30.0), still unstable (hard threshold is root cause)
    # Session 8i v3 FIX: Sigmoid smooth transition eliminates discontinuity at boundary

    distance_gate_threshold: float = 0.7  # Distance (m) separating reach-mode from align-mode
    distance_gate_smoothness: float = 0.15  # Sigmoid transition width (0.55m-0.85m zone) - v3 NEW
    orientation_tracking_far: float = 8.0  # Orientation weight when far (>threshold)
    orientation_tracking_close: float = 30.0  # Orientation weight when close (<threshold)
    orientation_progress_bonus: float = 0.0  # DISABLED for A/B test (was 2.0)
    angular_velocity_penalty: float = 0.0  # DISABLED for A/B test (was 1.0)

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
    min_obstacle_distance_weight: float = 2.0
    safety_radius: float = 0.2  # Desired clearance beyond robot footprint + obstacle radius

    # ========================================
    # AUXILIARY TRACKING SHAPING
    # ========================================
    position_distance_penalty: float = 80.0  # Linear fallback penalty so gradients remain informative when far


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
    initial_joint_noise_std: float = 0.03  # radians; conservative Proto2 startup noise for PPO stability

    # Observation settings
    use_lookahead: bool = True
    lookahead_steps: int = 3
    lookahead_dt: float = 0.1  # seconds
    include_action_history: bool = True
    action_history_length: int = 2

    # Termination conditions
    terminate_on_self_collision: bool = True  # CRITICAL: End episode if robot hits itself
    self_collision_termination_threshold: float = 10.0  # Newtons (higher than penalty threshold)
    contact_grace_steps: int = 12  # Ignore reset-settling contact for 0.6s at 20 Hz control
    terminate_on_collision: bool = False  # External collisions (not used)
    terminate_on_tracking_error: bool = True
    max_tracking_error: float = 2.0  # meters

