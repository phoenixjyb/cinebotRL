"""Reward and penalty terms for mobile manipulator tracking."""

from __future__ import annotations

import torch


def position_tracking_reward(
    current_pos: torch.Tensor,
    target_pos: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Reward for position tracking accuracy.
    
    Uses negative exponential of squared error.
    
    Args:
        current_pos: Current EE positions [num_envs, 3]
        target_pos: Target positions [num_envs, 3]
        scale: Scaling factor for error sensitivity
        
    Returns:
        Reward values [num_envs]
    """
    error = torch.norm(target_pos - current_pos, dim=-1)
    return torch.exp(-scale * error ** 2)


def orientation_tracking_reward(
    current_quat: torch.Tensor,
    target_quat: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Reward for orientation tracking accuracy.
    
    Uses negative exponential of quaternion distance.
    
    Args:
        current_quat: Current EE orientations [num_envs, 4] (wxyz)
        target_quat: Target orientations [num_envs, 4] (wxyz)
        scale: Scaling factor for error sensitivity
        
    Returns:
        Reward values [num_envs]
    """
    # Quaternion dot product (measures similarity)
    dot_product = torch.sum(current_quat * target_quat, dim=-1).abs()
    
    # Clamp to avoid numerical issues with acos
    dot_product = torch.clamp(dot_product, 0.0, 1.0)
    
    # Angular distance in radians
    angular_dist = 2 * torch.acos(dot_product)
    
    return torch.exp(-scale * angular_dist ** 2)


def progress_bonus(
    prev_error: torch.Tensor,
    current_error: torch.Tensor,
) -> torch.Tensor:
    """Bonus for reducing tracking error.
    
    Args:
        prev_error: Previous tracking error [num_envs]
        current_error: Current tracking error [num_envs]
        
    Returns:
        Bonus values [num_envs]
    """
    improvement = prev_error - current_error
    return torch.clamp(improvement, min=0.0)


def action_magnitude_penalty(
    actions: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Penalty for large actions (energy efficiency).
    
    Args:
        actions: Action values [num_envs, action_dim]
        scale: Penalty scale
        
    Returns:
        Penalty values [num_envs]
    """
    return scale * torch.sum(actions ** 2, dim=-1)


def action_rate_penalty(
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Penalty for rapid action changes (smoothness).
    
    Args:
        actions: Current action values [num_envs, action_dim]
        prev_actions: Previous action values [num_envs, action_dim]
        scale: Penalty scale
        
    Returns:
        Penalty values [num_envs]
    """
    action_diff = actions - prev_actions
    return scale * torch.sum(action_diff ** 2, dim=-1)


def collision_penalty(
    contact_forces: torch.Tensor,
    threshold: float = 1.0,
    scale: float = 1.0,
) -> torch.Tensor:
    """Penalty for collisions detected via contact sensors.
    
    DEPRECATED: Use self_collision_penalty for mobile manipulators.
    
    Args:
        contact_forces: Contact force magnitudes [num_envs, num_bodies]
        threshold: Force threshold to count as collision
        scale: Penalty scale
        
    Returns:
        Penalty values [num_envs]
    """
    # Check if any contact exceeds threshold
    has_collision = torch.any(contact_forces > threshold, dim=-1).float()
    return scale * has_collision


def self_collision_penalty(
    net_contact_forces: torch.Tensor,
    threshold: float = 1.0,
    scale: float = 1.0,
    continuous: bool = True,
) -> torch.Tensor:
    """Penalty for self-collisions within the robot.
    
    Self-collision occurs when robot links contact each other, which is
    critical to prevent for mobile manipulators (arm hitting base, etc.).
    
    Args:
        net_contact_forces: Net contact force vectors [num_envs, num_bodies, 3]
        threshold: Force threshold to count as collision (Newtons)
        scale: Penalty scale
        continuous: If True, penalty scales with force magnitude (softer).
                   If False, binary penalty (harsher).
        
    Returns:
        Penalty values [num_envs]
    """
    # Compute magnitude of net contact forces for each body
    contact_force_mag = torch.norm(net_contact_forces, dim=-1)  # [num_envs, num_bodies]
    
    if continuous:
        # Continuous penalty: scales with force magnitude
        # Only penalize forces above threshold
        violation = torch.clamp(contact_force_mag - threshold, min=0.0)
        penalty = torch.sum(violation, dim=-1)  # Sum over all bodies
        return scale * penalty
    else:
        # Binary penalty: any contact above threshold triggers full penalty
        has_collision = torch.any(contact_force_mag > threshold, dim=-1).float()
        return scale * has_collision


def stability_penalty(
    base_lin_vel: torch.Tensor,
    base_ang_vel: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Penalty for excessive base motion (stability).
    
    Args:
        base_lin_vel: Base linear velocity [num_envs, 3]
        base_ang_vel: Base angular velocity [num_envs, 3]
        scale: Penalty scale
        
    Returns:
        Penalty values [num_envs]
    """
    lin_vel_mag = torch.norm(base_lin_vel, dim=-1)
    ang_vel_mag = torch.norm(base_ang_vel, dim=-1)
    return scale * (lin_vel_mag ** 2 + ang_vel_mag ** 2)


def obstacle_distance_reward(
    min_distance: torch.Tensor,
    safety_radius: float,
    scale: float = 1.0,
) -> torch.Tensor:
    """Reward for maintaining safe distance from obstacles.
    
    Uses sigmoid function to smoothly transition from penalty to reward.
    
    Args:
        min_distance: Minimum distance to any obstacle [num_envs]
        safety_radius: Desired minimum safe distance
        scale: Reward/penalty scale
        
    Returns:
        Reward values [num_envs] (negative if too close)
    """
    # Normalized distance (0 at contact, 1 at safety_radius, >1 beyond)
    normalized_dist = min_distance / safety_radius
    
    # Sigmoid centered at 1.0
    return scale * (torch.sigmoid(5 * (normalized_dist - 1.0)) - 0.5)


def velocity_limit_penalty(
    base_lin_vel: torch.Tensor,
    joint_vel: torch.Tensor,
    max_linear_vel: float,
    max_joint_vel: float,
    scale: float = 1.0,
) -> torch.Tensor:
    """Penalty for exceeding velocity limits.
    
    Args:
        base_lin_vel: Base linear velocity [num_envs, 3] (only x and yaw matter)
        joint_vel: Joint velocities [num_envs, num_joints]
        max_linear_vel: Maximum linear velocity (m/s)
        max_joint_vel: Maximum joint velocity (rad/s)
        scale: Penalty scale
        
    Returns:
        Penalty values [num_envs]
    """
    # Base forward velocity (x-direction for differential drive)
    base_vel_x = base_lin_vel[:, 0].abs()
    base_vel_violation = torch.clamp(base_vel_x - max_linear_vel, min=0.0) ** 2
    
    # Joint velocities
    joint_vel_violation = torch.sum(
        torch.clamp(joint_vel.abs() - max_joint_vel, min=0.0) ** 2, dim=-1
    )
    
    return scale * (base_vel_violation + joint_vel_violation)


def acceleration_limit_penalty(
    current_vel: torch.Tensor,
    prev_vel: torch.Tensor,
    dt: float,
    max_accel: float,
    scale: float = 1.0,
) -> torch.Tensor:
    """Penalty for exceeding acceleration limits.
    
    Args:
        current_vel: Current velocities [num_envs, dim]
        prev_vel: Previous velocities [num_envs, dim]
        dt: Time step (seconds)
        max_accel: Maximum acceleration
        scale: Penalty scale
        
    Returns:
        Penalty values [num_envs]
    """
    accel = (current_vel - prev_vel) / dt
    accel_mag = torch.norm(accel, dim=-1)
    violation = torch.clamp(accel_mag - max_accel, min=0.0)
    return scale * violation ** 2


def jerk_penalty(
    current_accel: torch.Tensor,
    prev_accel: torch.Tensor,
    dt: float,
    max_jerk: float,
    scale: float = 1.0,
) -> torch.Tensor:
    """Penalty for exceeding jerk (rate of acceleration change) limits.
    
    Args:
        current_accel: Current acceleration [num_envs, dim]
        prev_accel: Previous acceleration [num_envs, dim]
        dt: Time step (seconds)
        max_jerk: Maximum jerk
        scale: Penalty scale
        
    Returns:
        Penalty values [num_envs]
    """
    jerk = (current_accel - prev_accel) / dt
    jerk_mag = torch.norm(jerk, dim=-1)
    violation = torch.clamp(jerk_mag - max_jerk, min=0.0)
    return scale * violation ** 2


def joint_limit_penalty(
    joint_pos: torch.Tensor,
    joint_lower: torch.Tensor,
    joint_upper: torch.Tensor,
    margin: float,
    scale: float = 1.0,
) -> torch.Tensor:
    """Penalty for approaching joint limits.
    
    Soft penalty that increases as joints approach limits.
    
    Args:
        joint_pos: Joint positions [num_envs, num_joints]
        joint_lower: Lower joint limits [num_joints]
        joint_upper: Upper joint limits [num_joints]
        margin: Safety margin from limits (radians)
        scale: Penalty scale
        
    Returns:
        Penalty values [num_envs]
    """
    # Distance from lower limit
    lower_dist = joint_pos - joint_lower
    lower_violation = torch.clamp(margin - lower_dist, min=0.0)
    
    # Distance from upper limit
    upper_dist = joint_upper - joint_pos
    upper_violation = torch.clamp(margin - upper_dist, min=0.0)
    
    # Sum over all joints
    total_violation = torch.sum(lower_violation ** 2 + upper_violation ** 2, dim=-1)
    
    return scale * total_violation


def lateral_motion_penalty(
    base_lin_vel: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Penalty for lateral (sideways) motion.
    
    Differential drive robots cannot move sideways, so any y-velocity
    indicates slipping or unrealistic motion.
    
    Args:
        base_lin_vel: Base linear velocity [num_envs, 3] (x, y, z)
        scale: Penalty scale
        
    Returns:
        Penalty values [num_envs]
    """
    # Penalize y-direction velocity (lateral)
    lateral_vel = base_lin_vel[:, 1].abs()
    return scale * lateral_vel ** 2


def action_smoothness_penalty(
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    prev_prev_actions: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Penalty for jerkiness in actions (second derivative).
    
    Measures how much the action change rate is changing.
    
    Args:
        actions: Current actions [num_envs, action_dim]
        prev_actions: Previous actions [num_envs, action_dim]
        prev_prev_actions: Actions from 2 steps ago [num_envs, action_dim]
        scale: Penalty scale
        
    Returns:
        Penalty values [num_envs]
    """
    # First derivative (velocity)
    vel_current = actions - prev_actions
    vel_prev = prev_actions - prev_prev_actions
    
    # Second derivative (acceleration/jerk)
    accel = vel_current - vel_prev
    
    return scale * torch.sum(accel ** 2, dim=-1)


def compute_combined_reward(
    # Tracking terms
    current_ee_pos: torch.Tensor,
    current_ee_quat: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    prev_tracking_error: torch.Tensor,
    
    # Action terms
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    prev_prev_actions: torch.Tensor,
    
    # Robot state
    base_lin_vel: torch.Tensor,
    base_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    
    # Previous state for derivatives
    prev_base_lin_vel: torch.Tensor,
    prev_joint_vel: torch.Tensor,
    prev_base_accel: torch.Tensor,
    
    # Robot limits
    joint_lower: torch.Tensor,
    joint_upper: torch.Tensor,
    robot_limits: dict[str, float],
    
    # Safety terms
    contact_forces: torch.Tensor,
    min_obstacle_dist: torch.Tensor | None,
    
    # Timing
    dt: float,
    
    # Weights
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute combined reward from all components with robot constraints.
    
    Args:
        current_ee_pos: Current EE positions [num_envs, 3]
        current_ee_quat: Current EE orientations [num_envs, 4]
        target_pos: Target positions [num_envs, 3]
        target_quat: Target orientations [num_envs, 4]
        prev_tracking_error: Previous tracking error [num_envs]
        actions: Current actions [num_envs, action_dim]
        prev_actions: Previous actions [num_envs, action_dim]
        prev_prev_actions: Actions from 2 steps ago [num_envs, action_dim]
        base_lin_vel: Base linear velocity [num_envs, 3]
        base_ang_vel: Base angular velocity [num_envs, 3]
        joint_pos: Joint positions [num_envs, num_joints]
        joint_vel: Joint velocities [num_envs, num_joints]
        prev_base_lin_vel: Previous base velocity [num_envs, 3]
        prev_joint_vel: Previous joint velocities [num_envs, num_joints]
        prev_base_accel: Previous base acceleration [num_envs, 3]
        joint_lower: Lower joint limits [num_joints]
        joint_upper: Upper joint limits [num_joints]
        robot_limits: Dictionary of robot limit values
        contact_forces: Contact forces [num_envs, num_bodies]
        min_obstacle_dist: Min obstacle distances [num_envs] or None
        dt: Control timestep (seconds)
        weights: Dictionary of reward weights
        
    Returns:
        total_reward: Combined reward [num_envs]
        reward_components: Dictionary of individual reward components
    """
    # Tracking rewards
    pos_reward = weights["position_tracking"] * position_tracking_reward(
        current_ee_pos, target_pos, scale=1.0
    )
    
    ori_reward = weights["orientation_tracking"] * orientation_tracking_reward(
        current_ee_quat, target_quat, scale=0.5
    )
    
    current_error = torch.norm(target_pos - current_ee_pos, dim=-1)
    prog_bonus = weights["progress_bonus"] * progress_bonus(
        prev_tracking_error, current_error
    )
    
    # Action penalties
    action_mag_penalty = action_magnitude_penalty(actions, scale=weights["action_magnitude"])
    action_rt_penalty = action_rate_penalty(actions, prev_actions, scale=weights["action_rate"])
    action_smooth_penalty = action_smoothness_penalty(
        actions, prev_actions, prev_prev_actions, scale=weights["action_smoothness"]
    )
    
    # Robot constraint penalties
    # Note: base_lin_vel and joint_vel are now normalized to [0,1] range
    # So limits should be 1.0 for normalized velocity penalty
    vel_limit_penalty = velocity_limit_penalty(
        base_lin_vel, joint_vel,
        1.0,  # Normalized velocity limit for base (was robot_limits["max_linear_velocity"])
        robot_limits["max_joint_velocity"],  # Joint limits still in rad/s (not normalized)
        scale=weights["velocity_limit_penalty"],
    )
    
    # Base acceleration (with normalized velocities)
    base_accel = (base_lin_vel - prev_base_lin_vel) / dt
    # Normalize acceleration limit: max_accel_normalized = max_accel / max_velocity 
    normalized_accel_limit = robot_limits["max_linear_acceleration"] / robot_limits["max_linear_velocity"]
    accel_limit_penalty = acceleration_limit_penalty(
        base_lin_vel[:, 0:1], prev_base_lin_vel[:, 0:1],  # Forward velocity only (normalized)
        dt, normalized_accel_limit,  # Use normalized acceleration limit
        scale=weights["acceleration_limit_penalty"],
    )
    
    # Jerk (rate of acceleration change) - normalize limit
    # max_jerk_normalized = max_jerk / max_velocity (since accel is already normalized by velocity)
    normalized_jerk_limit = robot_limits["max_linear_jerk"] / robot_limits["max_linear_velocity"]
    jerk_penalty_val = jerk_penalty(
        base_accel[:, 0:1], prev_base_accel[:, 0:1],  # Forward direction (normalized)
        dt, normalized_jerk_limit,  # Use normalized jerk limit
        scale=weights["jerk_limit_penalty"],
    )
    
    # Joint limits
    joint_limit_penalty_val = joint_limit_penalty(
        joint_pos, joint_lower, joint_upper,
        margin=robot_limits["joint_limit_margin"],
        scale=weights["joint_limit_penalty"],
    )
    
    # Lateral motion (should be zero for differential drive)
    lateral_penalty = lateral_motion_penalty(
        base_lin_vel, scale=weights["lateral_motion_penalty"]
    )
    
    # Safety penalties - SELF-COLLISION (critical for mobile manipulator!)
    self_coll_penalty = self_collision_penalty(
        contact_forces,  # Expects [num_envs, num_bodies, 3]
        threshold=weights.get("self_collision_threshold", 1.0),
        scale=weights["self_collision_penalty"],
        continuous=weights.get("self_collision_continuous", True),
    )
    
    # External collisions (not used for now, kept for compatibility)
    # coll_penalty = collision_penalty(contact_forces, scale=weights["collision_penalty"])
    
    stab_penalty = stability_penalty(base_lin_vel, base_ang_vel, scale=weights["stability_penalty"])
    
    # Obstacle distance (if enabled)
    if min_obstacle_dist is not None:
        obst_reward = obstacle_distance_reward(
            min_obstacle_dist,
            safety_radius=weights["safety_radius"],
            scale=weights["min_obstacle_distance_weight"],
        )
    else:
        obst_reward = torch.zeros_like(pos_reward)
    
    # Combine all terms
    total_reward = (
        pos_reward
        + ori_reward
        + prog_bonus
        - action_mag_penalty
        - action_rt_penalty
        - action_smooth_penalty
        - vel_limit_penalty
        - accel_limit_penalty
        - jerk_penalty_val
        - joint_limit_penalty_val
        - lateral_penalty
        - self_coll_penalty  # CRITICAL: Self-collision penalty
        - stab_penalty
        + obst_reward
    )
    
    # Store components for logging
    components = {
        "position_tracking": pos_reward,
        "orientation_tracking": ori_reward,
        "progress_bonus": prog_bonus,
        "action_magnitude_penalty": action_mag_penalty,
        "action_rate_penalty": action_rt_penalty,
        "action_smoothness_penalty": action_smooth_penalty,
        "velocity_limit_penalty": vel_limit_penalty,
        "acceleration_limit_penalty": accel_limit_penalty,
        "jerk_penalty": jerk_penalty_val,
        "joint_limit_penalty": joint_limit_penalty_val,
        "lateral_motion_penalty": lateral_penalty,
        "self_collision_penalty": self_coll_penalty,
        "stability_penalty": stab_penalty,
        "obstacle_reward": obst_reward,
    }
    
    return total_reward, components
