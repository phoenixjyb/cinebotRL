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
    
    # Safety terms
    contact_forces: torch.Tensor,
    base_lin_vel: torch.Tensor,
    base_ang_vel: torch.Tensor,
    min_obstacle_dist: torch.Tensor | None,
    
    # Weights
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute combined reward from all components.
    
    Args:
        current_ee_pos: Current EE positions [num_envs, 3]
        current_ee_quat: Current EE orientations [num_envs, 4]
        target_pos: Target positions [num_envs, 3]
        target_quat: Target orientations [num_envs, 4]
        prev_tracking_error: Previous tracking error [num_envs]
        actions: Current actions [num_envs, action_dim]
        prev_actions: Previous actions [num_envs, action_dim]
        contact_forces: Contact forces [num_envs, num_bodies]
        base_lin_vel: Base linear velocity [num_envs, 3]
        base_ang_vel: Base angular velocity [num_envs, 3]
        min_obstacle_dist: Min obstacle distances [num_envs] or None
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
    
    # Safety penalties
    coll_penalty = collision_penalty(contact_forces, scale=weights["collision_penalty"])
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
        - coll_penalty
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
        "collision_penalty": coll_penalty,
        "stability_penalty": stab_penalty,
        "obstacle_reward": obst_reward,
    }
    
    return total_reward, components
