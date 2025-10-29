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


def base_mobilization_reward(
    base_pos: torch.Tensor,
    prev_base_pos: torch.Tensor,
    target_pos: torch.Tensor,
    arm_reach: float = 0.8,  # Maximum reach of arm from base (meters)
    scale: float = 1.0,
) -> torch.Tensor:
    """Reward chassis movement that genuinely reduces target distance.
    
    The previous implementation compared sequential target distances that were
    measured against the target at different timesteps. When the trajectory
    moved closer on its own, the policy received positive reward even if the
    chassis stayed still. Here we hold the target fixed at the CURRENT pose and
    compare how far it would be if the base had not moved. This isolates the
    contribution of chassis motion and only grants credit when the base actually
    closes the gap to an out-of-reach target.
    
    Args:
        base_pos: Current base position [num_envs, 3]
        prev_base_pos: Previous base position [num_envs, 3]
        target_pos: Current target position [num_envs, 3]
        arm_reach: Maximum reach of arm from base center (meters)
        scale: Reward scale factor
        
    Returns:
        Reward values [num_envs] - positive when base motion reduces distance
    """
    # Clamp numerical noise so zero displacement does not look like movement.
    eps = 1e-6
    
    target_xy = target_pos[:, :2]
    base_xy = base_pos[:, :2]
    prev_base_xy = prev_base_pos[:, :2]
    
    # Distance to the current target with and without the latest base motion.
    dist_current = torch.norm(target_xy - base_xy, dim=-1)
    dist_if_static = torch.norm(target_xy - prev_base_xy, dim=-1)
    
    # Positive when the chassis actually moved closer to the target.
    progress = dist_if_static - dist_current
    
    # CRITICAL FIX (Session 5b): Cap progress to prevent reward explosion!
    # Max 20cm progress per step → max reward = 150 × 0.2 = 30 points
    # (reasonable compared to position_tracking max = 50 points)
    progress = torch.clamp(progress, min=0.0, max=0.2)
    
    # Only reward motion when the goal is outside the arm workspace.
    out_of_reach = torch.sigmoid(((dist_current + dist_if_static) * 0.5 - arm_reach) * 5.0)
    
    # Suppress tiny numerical oscillations when the base did not really move.
    moved = torch.norm(base_xy - prev_base_xy, dim=-1) > eps
    progress = torch.where(moved, progress, torch.zeros_like(progress))
    
    return scale * progress * out_of_reach


def base_target_alignment_reward(
    base_pos: torch.Tensor,
    base_vel: torch.Tensor,
    base_quat: torch.Tensor,
    target_pos: torch.Tensor,
    arm_reach: float = 0.6,
    scale: float = 1.0,
) -> torch.Tensor:
    """Reward base movement that aligns toward the target.
    
    Encourages moving in the correct direction, even if not reducing
    distance much yet. Helps policy learn goal-directed navigation.
    
    BUGFIX: Now expects WORLD-frame velocity (changed from body-frame).
    All velocity inputs to compute_combined_reward are now world-frame for consistency.
    
    Args:
        base_pos: Current base position [num_envs, 3]
        base_vel: Base velocity in WORLD frame [num_envs, 3] (changed from body frame!)
        base_quat: Base orientation quaternion [num_envs, 4]
        target_pos: Target position [num_envs, 3]
        arm_reach: Arm workspace radius
        scale: Reward scale factor
        
    Returns:
        Reward [num_envs] - positive when moving toward target
    """
    # Only reward when target is out of reach
    base_xy = base_pos[:, :2]
    target_xy = target_pos[:, :2]
    dist_to_target = torch.norm(target_xy - base_xy, dim=-1)
    out_of_reach = (dist_to_target > arm_reach).float()
    
    # Direction from base to target (world frame)
    base_to_target = target_xy - base_xy
    base_to_target_norm = torch.norm(base_to_target, dim=-1, keepdim=True)
    base_to_target_unit = base_to_target / (base_to_target_norm + 1e-6)
    
    # BUGFIX: Velocity is already in world frame, no conversion needed!
    base_vel_world = base_vel[:, :2]  # Just extract XY components [num_envs, 2]
    
    # Compute speed and direction in world frame
    base_speed = torch.norm(base_vel_world, dim=-1, keepdim=True)
    base_vel_unit = base_vel_world / (base_speed + 1e-6)
    
    # Dot product: 1.0 = perfect alignment, 0.0 = perpendicular, -1.0 = opposite
    alignment = torch.sum(base_vel_unit * base_to_target_unit, dim=-1)
    alignment = torch.clamp(alignment, min=0.0, max=1.0)  # Only reward positive alignment
    
    # Scale by speed (moving faster in right direction = more reward)
    speed_scale = torch.clamp(base_speed.squeeze(-1), min=0.0, max=0.5)  # Cap at 0.5 m/s
    
    return scale * alignment * speed_scale * out_of_reach


def target_distance_penalty(
    base_pos: torch.Tensor,
    target_pos: torch.Tensor,
    prev_base_pos: torch.Tensor = None,
    arm_reach: float = 0.6,
    scale: float = 1.0,
    moving_discount: float = 0.9,
) -> torch.Tensor:
    """Penalty for distance between base and target.
    
    Heavily penalize when base is far from target AND not moving.
    Reduce penalty by 90% when base is actively moving (detected via velocity).
    
    Args:
        base_pos: Current base position [num_envs, 3]
        target_pos: Target position [num_envs, 3]
        prev_base_pos: Previous base position (for movement detection)
        arm_reach: Arm reach threshold
        scale: Base penalty scale
        moving_discount: Discount factor when base is moving (0.9 = 90% reduction)
        
    Returns:
        Distance penalty [num_envs]
    """
    # XY distance from base to target
    dist = torch.norm(target_pos[:, :2] - base_pos[:, :2], dim=-1)
    
    # Smooth penalty that ramps up when target is out of reach
    out_of_reach_penalty = torch.sigmoid((dist - arm_reach) * 5.0) * dist
    
    # If prev_base_pos provided, detect movement and reduce penalty
    if prev_base_pos is not None:
        base_movement = torch.norm(base_pos[:, :2] - prev_base_pos[:, :2], dim=-1)
        is_moving = base_movement > 0.01  # 1cm threshold
        penalty = torch.where(
            is_moving,
            out_of_reach_penalty * moving_discount,  # 90% reduction when moving
            out_of_reach_penalty  # Full penalty when static
        )
    else:
        penalty = out_of_reach_penalty
    
    return scale * penalty


def reachability_maintenance_reward(
    target_pos: torch.Tensor,
    base_pos: torch.Tensor,
    arm_optimal_reach: float = 0.4,
    arm_max_reach: float = 0.6,
    scale: float = 50.0,
) -> torch.Tensor:
    """Reward maintaining targets within optimal arm workspace.
    
    Session 8b: Added to fix reachability crisis where base drifts too far.
    
    Validation showed catastrophic failure:
    - Step 0: 32/32 reachable
    - Step 12800: 0/32 reachable (100% failure!)
    - Distance grew from 0.5m → 1.2m (double arm reach)
    
    This reward provides strong incentive to keep targets reachable:
    - Within optimal reach (0.3-0.4m): Full reward (+1.0)
    - Between optimal and max (0.4-0.6m): Reduced reward (+0.5)
    - Beyond max reach (>0.6m): Penalty (-2.0)
    
    Args:
        target_pos: Current target [num_envs, 3]
        base_pos: Current base [num_envs, 3]
        arm_optimal_reach: Optimal working distance (0.3-0.4m)
        arm_max_reach: Maximum arm reach (0.6m)
        scale: Reward weight
        
    Returns:
        Reward [num_envs] - positive when reachable, negative when beyond reach
    """
    base_xy = base_pos[:, :2]
    target_xy = target_pos[:, :2]
    dist = torch.norm(target_xy - base_xy, dim=-1)
    
    # Three zones:
    # 1. Optimal zone (< 0.4m): Full reward
    # 2. Acceptable zone (0.4-0.6m): Partial reward
    # 3. Unreachable zone (> 0.6m): Penalty
    
    optimal_mask = dist <= arm_optimal_reach
    acceptable_mask = (dist > arm_optimal_reach) & (dist <= arm_max_reach)
    unreachable_mask = dist > arm_max_reach
    
    reward = torch.where(
        optimal_mask,
        torch.ones_like(dist),  # +1.0 in optimal zone
        torch.where(
            acceptable_mask,
            torch.ones_like(dist) * 0.5,  # +0.5 in acceptable zone
            -2.0 * (dist - arm_max_reach),  # Penalty grows with distance
        )
    )
    
    return scale * reward


def base_overshoot_penalty(
    base_pos: torch.Tensor,
    target_pos: torch.Tensor,
    base_vel: torch.Tensor,
    base_quat: torch.Tensor,
    arm_optimal_reach: float = 0.4,
    scale: float = 20.0,
) -> torch.Tensor:
    """Penalize base moving past target waypoints (overshooting).
    
    Session 8b: Added to prevent base from "rushing ahead" of trajectory.
    
    Detection method: Check if base velocity points away from target.
    If base is moving away from target, it's either:
    1. Overshooting the current waypoint, or
    2. Moving in wrong direction
    
    Both cases should be penalized to maintain coordinated tracking.
    
    BUGFIX: Now expects WORLD-frame velocity (changed from body-frame).
    Consistent with all other reward functions.
    
    Args:
        base_pos: Current base [num_envs, 3]
        target_pos: Current target [num_envs, 3]
        base_vel: Base velocity in WORLD frame [num_envs, 3] (changed from body frame!)
        base_quat: Base orientation [num_envs, 4] (kept for consistency, but not used)
        arm_optimal_reach: Optimal distance (don't get closer than this)
        scale: Penalty weight
        
    Returns:
        Penalty [num_envs] - increases when moving past target
    """
    base_xy = base_pos[:, :2]
    target_xy = target_pos[:, :2]
    
    # BUGFIX: Velocity is already in world frame, no conversion needed!
    vel_world = base_vel[:, :2]  # Just extract XY components
    
    # Direction from base to target
    base_to_target = target_xy - base_xy
    base_to_target_norm = torch.norm(base_to_target, dim=-1, keepdim=True)
    base_to_target_unit = base_to_target / (base_to_target_norm + 1e-6)
    
    # Velocity alignment: positive = moving toward, negative = moving away
    vel_norm = torch.norm(vel_world, dim=-1, keepdim=True)
    vel_alignment = torch.sum(vel_world * base_to_target_unit, dim=-1) / (vel_norm.squeeze(-1) + 1e-6)
    
    # Current distance to target
    dist = base_to_target_norm.squeeze(-1)
    
    # Penalize moving away when:
    # 1. Already within optimal reach AND moving away (overshooting)
    # 2. Any distance AND moving strongly away (wrong direction)
    
    within_reach = dist < arm_optimal_reach
    moving_away = vel_alignment < -0.2  # Threshold: significantly away
    
    # Strong penalty for overshooting (within reach but moving away)
    overshoot_penalty = torch.where(
        within_reach & moving_away,
        -vel_alignment * 2.0,  # Penalty proportional to how fast moving away
        torch.zeros_like(vel_alignment)
    )
    
    # Mild penalty for wrong direction (any distance)
    wrong_dir_penalty = torch.clamp(-vel_alignment, min=0.0) * 0.5
    
    total_penalty = overshoot_penalty + wrong_dir_penalty
    return scale * total_penalty


def excessive_base_movement_penalty(
    base_pos: torch.Tensor,
    prev_base_pos: torch.Tensor,
    threshold: float = 0.1,
    scale: float = 10.0,
) -> torch.Tensor:
    """Heavily penalize excessive base movements to prevent wild behavior.
    
    Movements beyond threshold (default 10cm) are heavily penalized.
    This prevents the policy from exploiting unbounded rewards by moving wildly.
    
    Example: If base moves 1 meter in one step:
        excess = 1.0 - 0.1 = 0.9 meters
        penalty = 10.0 × 0.9 = 9.0 points
    
    Args:
        base_pos: Current base position [num_envs, 3]
        prev_base_pos: Previous base position [num_envs, 3]
        threshold: Maximum reasonable movement per step (meters)
        scale: Penalty scale factor
        
    Returns:
        Penalty values [num_envs] - zero for movements ≤ threshold
    """
    movement = torch.norm(base_pos[:, :2] - prev_base_pos[:, :2], dim=-1)
    excess = torch.clamp(movement - threshold, min=0.0)
    return scale * excess


def old_target_distance_penalty(
    base_pos: torch.Tensor,
    target_pos: torch.Tensor,
    prev_base_pos: torch.Tensor = None,
    arm_reach: float = 0.6,
    scale: float = 10.0,
) -> torch.Tensor:
    """Penalty for target being beyond arm reach from base.
    
    Measures PLANAR distance (X-Y only) since base can only move horizontally.
    Provides strong negative signal to force base mobilization when needed.
    
    SMART PENALTY: If base is actively moving toward target, penalty is reduced 90%
    to encourage base movement learning without overly harsh penalties during motion.
    
    This penalty creates a strong gradient for base movement by linearly penalizing
    distance beyond arm reach. Unlike exponential position tracking reward (which
    provides weak gradient when far), this penalty grows linearly with distance,
    ensuring the policy is strongly motivated to reduce large errors.
    
    Example: Target 1.4m beyond 0.6m reach → penalty = 10.0 × 1.4 = 14 points
             As base moves closer, penalty decreases linearly to zero at 0.6m
             If base is moving: penalty = 10.0 × 1.4 × 0.1 = 1.4 points (90% reduction!)
    
    Args:
        base_pos: Current base position [num_envs, 3]
        target_pos: Target position [num_envs, 3]
        prev_base_pos: Previous base position [num_envs, 3] (optional, for smart penalty)
        arm_reach: Maximum reach of arm from base (meters)
        scale: Penalty scale factor (points per meter beyond reach)
        
    Returns:
        Penalty values [num_envs] - higher when further beyond reach
    """
    # Planar distance only (base moves in X-Y plane)
    target_xy = target_pos[:, :2]
    base_xy = base_pos[:, :2]
    
    dist = torch.norm(target_xy - base_xy, dim=-1)
    beyond_reach = torch.clamp(dist - arm_reach, min=0.0)
    
    penalty = scale * beyond_reach  # Linear penalty
    
    # SMART PENALTY: Reduce if base is actively moving (>1cm threshold to ignore noise)
    if prev_base_pos is not None:
        prev_base_xy = prev_base_pos[:, :2]
        base_movement = torch.norm(base_xy - prev_base_xy, dim=-1)
        is_moving = base_movement > 0.01  # 1cm threshold
        
        # Apply 90% penalty reduction when moving (keep 10% to maintain gradient)
        penalty = torch.where(is_moving, penalty * 0.1, penalty)
    
    return penalty


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
    exclude_base: bool = True,
) -> torch.Tensor:
    """Penalty for self-collisions within the robot.
    
    Self-collision occurs when robot links contact each other, which is
    critical to prevent for mobile manipulators (arm hitting base, etc.).
    
    NOTE: Ground contact shows up in contact forces! We need to filter it out.
    By default, we exclude the base link (index 0) which has ground contact.
    
    Args:
        net_contact_forces: Net contact force vectors [num_envs, num_bodies, 3]
        threshold: Force threshold to count as collision (Newtons)
        scale: Penalty scale
        continuous: If True, penalty scales with force magnitude (softer).
                   If False, binary penalty (harsher).
        exclude_base: If True, exclude base link (index 0) which has ground contact
        
    Returns:
        Penalty values [num_envs]
    """
    # Compute magnitude of net contact forces for each body
    contact_force_mag = torch.norm(net_contact_forces, dim=-1)  # [num_envs, num_bodies]
    
    # Exclude base link if requested (to filter out ground contact)
    if exclude_base and contact_force_mag.shape[1] > 1:
        contact_force_mag = contact_force_mag[:, 1:]  # Skip first body (base)
    
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
    # BUGFIX: Use planar speed magnitude (‖v_xy‖) instead of just X component
    # This ensures limits are consistent regardless of chassis heading
    base_vel_xy = base_lin_vel[:, :2]  # [num_envs, 2] - planar velocity
    base_speed = torch.norm(base_vel_xy, dim=-1)  # [num_envs] - magnitude
    base_vel_violation = torch.clamp(base_speed - max_linear_vel, min=0.0) ** 2
    
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
    base_quat: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Penalty for lateral (sideways) motion in robot frame.
    
    Differential drive robots cannot move sideways, so any y-velocity
    in the ROBOT frame indicates slipping or unrealistic motion.
    
    Args:
        base_lin_vel: Base linear velocity in WORLD frame [num_envs, 3] (x, y, z)
        base_quat: Base orientation quaternion in WORLD frame [num_envs, 4] (w, x, y, z)
        scale: Penalty scale
        
    Returns:
        Penalty values [num_envs]
    """
    # Convert world-frame velocity to robot frame
    # Extract yaw from quaternion and rotate velocity
    # quat is (w, x, y, z), we need yaw angle
    
    # For a quaternion (w, x, y, z), yaw (rotation around Z) is:
    # yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
    w, x, y, z = base_quat[:, 0], base_quat[:, 1], base_quat[:, 2], base_quat[:, 3]
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    
    # Rotate world velocity to robot frame (2D rotation around Z)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    
    vel_x_robot = cos_yaw * base_lin_vel[:, 0] + sin_yaw * base_lin_vel[:, 1]
    vel_y_robot = -sin_yaw * base_lin_vel[:, 0] + cos_yaw * base_lin_vel[:, 1]
    
    # Penalize y-direction velocity in ROBOT frame (lateral)
    lateral_vel = vel_y_robot.abs()
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
    base_pos: torch.Tensor,  # NEW: Base position for progress tracking
    base_lin_vel: torch.Tensor,
    base_ang_vel: torch.Tensor,
    base_quat: torch.Tensor,  # Base orientation for lateral penalty
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    
    # Previous state for derivatives
    prev_base_pos: torch.Tensor,  # NEW: Previous base position
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
    
    # Base mobilization reward - only move chassis when target is out of arm reach
    base_mob_reward = base_mobilization_reward(
        base_pos, prev_base_pos, target_pos,
        arm_reach=0.6,  # Based on empirical observation: EE reaches ~0.6m from base
        scale=weights.get("base_progress_reward", 10.0)
    )
    
    # Session 7d: Base-target alignment reward - encourage moving toward unreachable targets
    base_alignment_reward = base_target_alignment_reward(
        base_pos, base_lin_vel, base_quat, target_pos,
        arm_reach=0.6,
        scale=weights.get("base_target_alignment", 10.0)
    )
    
    # Distance penalty - strong gradient for base mobilization
    dist_penalty = target_distance_penalty(
        base_pos, target_pos,
        prev_base_pos=prev_base_pos,  # NEW: Enable smart penalty (90% reduction when moving)
        arm_reach=0.6,
        scale=weights.get("target_distance_penalty", 10.0)
    )
    
    # ========================================
    # Session 8b: BASE COORDINATION REWARDS
    # ========================================
    
    # Reachability maintenance: Keep targets within arm workspace
    reach_maint_reward = reachability_maintenance_reward(
        target_pos, base_pos,
        arm_optimal_reach=0.4,
        arm_max_reach=0.6,
        scale=weights.get("reachability_maintenance_reward", 50.0)
    )
    
    # Overshoot penalty: Prevent base from moving past waypoints
    overshoot_penalty = base_overshoot_penalty(
        base_pos, target_pos, base_lin_vel, base_quat,
        arm_optimal_reach=0.4,
        scale=weights.get("base_overshoot_penalty", 20.0)
    )
    
    # Session 5b FIX: Excessive movement penalty (prevents wild base movements)
    # Session 8b: Increased from 5.0 to 15.0 to constrain base drift
    excessive_penalty = excessive_base_movement_penalty(
        base_pos, prev_base_pos,
        threshold=0.1,  # 10cm per step is maximum reasonable
        scale=weights.get("excessive_base_movement_penalty", 15.0)
    )
    
    # Action penalties
    action_mag_penalty = action_magnitude_penalty(actions, scale=weights["action_magnitude"])
    action_rt_penalty = action_rate_penalty(actions, prev_actions, scale=weights["action_rate"])
    action_smooth_penalty = action_smoothness_penalty(
        actions, prev_actions, prev_prev_actions, scale=weights["action_smoothness"]
    )
    
    # Robot constraint penalties
    # base_lin_vel is in physical units (m/s); limits come directly from robot specs.
    vel_limit_penalty = velocity_limit_penalty(
        base_lin_vel,
        joint_vel,
        robot_limits["max_linear_velocity"],
        robot_limits["max_joint_velocity"],
        scale=weights["velocity_limit_penalty"],
    )
    
    # Base acceleration in physical units (m/s^2)
    base_accel = (base_lin_vel - prev_base_lin_vel) / dt
    # BUGFIX: Pass full planar velocity (xy) for magnitude-based penalty
    accel_limit_penalty = acceleration_limit_penalty(
        base_lin_vel[:, :2],  # Planar velocity [num_envs, 2]
        prev_base_lin_vel[:, :2],  # Previous planar velocity [num_envs, 2]
        dt,
        robot_limits["max_linear_acceleration"],
        scale=weights["acceleration_limit_penalty"],
    )
    
    # Jerk (rate of change of acceleration) in physical units (m/s^3)
    # BUGFIX: Pass full planar acceleration (xy) for magnitude-based penalty
    jerk_penalty_val = jerk_penalty(
        base_accel[:, :2],  # Planar acceleration [num_envs, 2]
        prev_base_accel[:, :2],  # Previous planar acceleration [num_envs, 2]
        dt,
        robot_limits["max_linear_jerk"],
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
        base_lin_vel, base_quat, scale=weights["lateral_motion_penalty"]
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
        + base_mob_reward  # NEW: Reward base movement when target is far (NOW CAPPED!)
        + base_alignment_reward  # Session 7d: Reward moving toward target
        + reach_maint_reward  # Session 8b: Reward keeping targets reachable
        - dist_penalty  # NEW: Strong penalty for being beyond arm reach
        - overshoot_penalty  # Session 8b: Penalize moving past waypoints
        - excessive_penalty  # Session 8b: Prevent wild base movements (increased 5→15)
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
        "base_mobilization": base_mob_reward,  # NEW: Log base movement reward (NOW CAPPED!)
        "base_target_alignment": base_alignment_reward,  # Session 7d: Log alignment reward
        "reachability_maintenance_reward": reach_maint_reward,  # Session 8b: Log reachability reward
        "target_distance_penalty": dist_penalty,  # NEW: Log distance penalty
        "base_overshoot_penalty": overshoot_penalty,  # Session 8b: Log overshoot penalty
        "excessive_base_movement_penalty": excessive_penalty,  # Session 8b: Log movement constraint
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
