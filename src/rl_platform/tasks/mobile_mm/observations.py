"""Observation composition for mobile manipulator tracking task."""

from __future__ import annotations

import torch


def compose_observation(
    # Robot state
    base_pos: torch.Tensor,
    base_quat: torch.Tensor,
    base_lin_vel: torch.Tensor,
    base_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    ee_pos: torch.Tensor,
    ee_quat: torch.Tensor,
    ee_lin_vel: torch.Tensor,
    ee_ang_vel: torch.Tensor,
    
    # Target state
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    
    # Optional components
    lookahead_pos: torch.Tensor | None = None,
    action_history: torch.Tensor | None = None,
    contact_forces: torch.Tensor | None = None,
    min_obstacle_dist: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compose full observation vector from components.
    
    Args:
        base_pos: Base position [num_envs, 3]
        base_quat: Base orientation [num_envs, 4]
        base_lin_vel: Base linear velocity [num_envs, 3]
        base_ang_vel: Base angular velocity [num_envs, 3]
        joint_pos: Joint positions [num_envs, 9] (all joints: [0-2: base PPR, 3-8: arm])
        joint_vel: Joint velocities [num_envs, 9] (all joints: [0-2: base PPR, 3-8: arm])
        ee_pos: End-effector position [num_envs, 3]
        ee_quat: End-effector orientation [num_envs, 4]
        ee_lin_vel: End-effector linear velocity [num_envs, 3]
        ee_ang_vel: End-effector angular velocity [num_envs, 3]
        target_pos: Target position [num_envs, 3]
        target_quat: Target orientation [num_envs, 4]
        lookahead_pos: Lookahead positions [num_envs, steps, 3] or None
        action_history: Previous actions [num_envs, history_len, action_dim] or None
        contact_forces: Contact forces [num_envs, num_bodies] or None
        min_obstacle_dist: Minimum obstacle distance [num_envs, 1] or None
        
    Returns:
        observations: Flattened observation tensor [num_envs, obs_dim]
    """
    components = []
    
    # Base state (13 dims: pos + quat + lin_vel + ang_vel)
    components.extend([base_pos, base_quat, base_lin_vel, base_ang_vel])
    
    # Joint state (2 * num_joints) - extract only arm joints [3:9] from full joint array
    # Robot has 9 DOF: [0-2: base PPR joints, 3-8: arm joints]
    # We only include arm joints in observations (6 joints × 2 = 12 dims)
    arm_joint_pos = joint_pos[:, 3:9]  # Only arm joints
    arm_joint_vel = joint_vel[:, 3:9]  # Only arm joints
    components.extend([arm_joint_pos, arm_joint_vel])
    
    # End-effector state (13 dims)
    components.extend([ee_pos, ee_quat, ee_lin_vel, ee_ang_vel])
    
    # Tracking error (7 dims: position error + orientation error)
    pos_error = target_pos - ee_pos
    quat_error = quat_diff(ee_quat, target_quat)  # Relative quaternion
    components.extend([pos_error, quat_error])
    
    # Optional: Lookahead targets
    if lookahead_pos is not None:
        # Flatten lookahead: [num_envs, steps, 3] -> [num_envs, steps*3]
        batch_size = lookahead_pos.shape[0]
        lookahead_flat = lookahead_pos.view(batch_size, -1)
        components.append(lookahead_flat)
    
    # Optional: Action history
    if action_history is not None:
        # Flatten history: [num_envs, history_len, action_dim] -> [num_envs, history_len*action_dim]
        batch_size = action_history.shape[0]
        history_flat = action_history.view(batch_size, -1)
        components.append(history_flat)
    
    # Optional: Contact forces
    if contact_forces is not None:
        components.append(contact_forces)
    
    # Optional: Obstacle distance
    if min_obstacle_dist is not None:
        # Ensure it's 2D [num_envs, 1]
        if min_obstacle_dist.ndim == 1:
            min_obstacle_dist = min_obstacle_dist.unsqueeze(-1)
        components.append(min_obstacle_dist)
    
    # Concatenate all components
    observations = torch.cat(components, dim=-1)
    
    return observations


def quat_diff(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Compute relative quaternion from q1 to q2.
    
    Returns q_diff such that q2 = q_diff * q1
    
    Args:
        q1: First quaternions [num_envs, 4] (wxyz)
        q2: Second quaternions [num_envs, 4] (wxyz)
        
    Returns:
        q_diff: Relative quaternions [num_envs, 4]
    """
    # Conjugate of q1
    q1_conj = q1.clone()
    q1_conj[:, 1:] = -q1_conj[:, 1:]  # Negate x, y, z components
    
    # q_diff = q2 * q1_conj
    return quat_multiply(q2, q1_conj)


def quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions.
    
    Args:
        q1: First quaternions [num_envs, 4] (wxyz)
        q2: Second quaternions [num_envs, 4] (wxyz)
        
    Returns:
        Product quaternions [num_envs, 4]
    """
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.stack([w, x, y, z], dim=-1)


def get_observation_dimensions(
    num_joints: int,
    num_contacts: int = 0,
    use_lookahead: bool = False,
    lookahead_steps: int = 3,
    use_action_history: bool = False,
    action_history_length: int = 2,
    action_dim: int = 8,
    use_obstacles: bool = False,
) -> int:
    """Calculate total observation dimension.
    
    Args:
        num_joints: Number of robot joints
        num_contacts: Number of contact sensors
        use_lookahead: Whether to include lookahead targets
        lookahead_steps: Number of lookahead steps
        use_action_history: Whether to include action history
        action_history_length: Length of action history
        action_dim: Dimension of action space
        use_obstacles: Whether obstacle distance is included
        
    Returns:
        Total observation dimension
    """
    dim = 0
    
    # Base state: pos(3) + quat(4) + lin_vel(3) + ang_vel(3) = 13
    dim += 13
    
    # Joint state: pos(num_joints) + vel(num_joints)
    dim += 2 * num_joints
    
    # EE state: pos(3) + quat(4) + lin_vel(3) + ang_vel(3) = 13
    dim += 13
    
    # Tracking error: pos_error(3) + quat_error(4) = 7
    dim += 7
    
    # Optional: Lookahead
    if use_lookahead:
        dim += lookahead_steps * 3  # Position only
    
    # Optional: Action history
    if use_action_history:
        dim += action_history_length * action_dim
    
    # Optional: Contact forces
    if num_contacts > 0:
        dim += num_contacts
    
    # Optional: Obstacle distance
    if use_obstacles:
        dim += 1
    
    return dim
