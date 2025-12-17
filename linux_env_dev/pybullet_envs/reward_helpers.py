import numpy as np

DEBUG=False

def compute_reward(
    base_pos,
    base_lin_vel,
    ee_pos,
    target_pos,
    base_yaw,
    wrap_angle_fn,
    remaining_ratio=None,
    *,
    dist_weight: float = 2.0,
    collision_threshold: float = 0.3,
    collision_ratio: float = 50.0,
    clip_abs: float = 5.0,
):
    total_reward = 0.0
    info = {}
    target_len = np.linalg.norm(np.array(target_pos[:2], dtype=float))
    base_len = np.linalg.norm(np.array(base_pos[:2], dtype=float))
    yaw_ratio = max(0.0, (target_len - 2.0 - base_len) / (target_len - 2.0)) * 10

    # 底盘位置误差——距离误差
    # dist_reward, dist_info = compute_distance_reward(base_pos, target_pos, 0.5)
    
    # 末端执行器误差——距离误差
    # dist_reward, dist_info = compute_nonlinear_distance_reward(ee_pos, target_pos, 2.0)
    dist_reward, dist_info = compute_distance_reward(ee_pos, target_pos, float(dist_weight)) # 使用线性的，要不然因为跟踪精度不愿意往前走
    
    # 底盘速度过快惩罚
    # vel_reward, vel_info = compute_velocity_reward(base_lin_vel, thresh=0.0, ratio=2.0)

    # 末端执行器高程误差
    # reward_ee_height, ee_height_info = compute_height_reward(ee_pos[2], target_pos[2])

    # 进度惩罚
    # progress_reward = compute_progress_reward(remaining_ratio, ratio=2.0)
    
    # 机械臂和本体的干涉惩罚
    collision_reward = 0.0
    if collision_ratio and float(collision_ratio) != 0.0 and collision_threshold is not None:
        collision_reward = compute_collision_reward(
            base_pos, ee_pos, thresh=float(collision_threshold), ratio=float(collision_ratio)
        )
    
    total_reward = dist_reward + collision_reward
    total_reward = float(np.clip(total_reward, -float(clip_abs), float(clip_abs)))
    
    info.update(dist_info)
    info.update(
        {
            "reward_dist": float(dist_reward),
            "reward_collision": float(collision_reward),
        }
    )
    # info.update(yaw_info)
    # info.update(ee_height_info)
    if DEBUG:
        print(f"Reward: {total_reward:.2f} (dist: {dist_reward:.2f}, collision: {collision_reward:.2f})")
    return total_reward, info

def compute_collision_reward(base_pos, ee_pos, thresh: float = 0.3, ratio: float = 1.0):
    reward = 0.0
    base_xy = np.array(base_pos[:2], dtype=float)
    ee_xy = np.array(ee_pos[:2], dtype=float)
    dist = float(np.linalg.norm(ee_xy - base_xy))
    if dist < float(thresh):
        reward = -1.0 * (float(thresh) - dist) * float(ratio)
    
    # 距离0.25，ratio=100，惩罚-50，ratio=1000，惩罚-500
    # 距离0.20，ratio=100，惩罚-100，ratio=1000，惩罚-1000

    if DEBUG:
        print(f"collision Reward: {reward:.2f} (dist={dist:.2f}, ratio={ratio:.2f})")

    return reward

def compute_progress_reward(remaining_ratio, ratio=1.0):
    """Compute progress-based reward (negative with scaling).

    Returns (reward, info) where info contains 'remaining_ratio'.
    """
    reward = -1.0 * remaining_ratio * ratio
    if DEBUG:
        print(f"progress reward: {reward:.2f} (remaining_ratio={remaining_ratio:.2f})")
    return reward

def compute_height_reward(ee_height, target_height):
    reward = 0.0
    height_diff = abs(ee_height - target_height)
    reward = -height_diff

    return reward, {"ee_height": ee_height, "target_height": target_height}

def compute_velocity_reward(base_lin_vel, thresh=1.0, ratio=1.0):
    reward = 0.0
    if base_lin_vel > thresh:
        reward = -abs(base_lin_vel - thresh) * ratio

    if DEBUG:
        print(f"velocity Reward: {reward:.2f} (lin_vel={base_lin_vel:.2f}, thresh={thresh:.2f}, ratio={ratio:.2f})")

    return reward, {}

def compute_nonlinear_distance_reward(base_pos, target_pos, dist_weight=1.0):
    """Compute nonlinear distance-based reward (negative with scaling).

    Returns (reward, info) where info contains 'distance'.
    """
    reward = 0.0
    base_xy = np.array(base_pos[:3], dtype=float)
    target_xy = np.array(target_pos[:3], dtype=float)
    dist = float(np.linalg.norm(target_xy - base_xy))
    
    if dist < 0.1:
        reward = -2.0 * dist
    else:
        reward = -0.2 - 1.0 * (dist - 0.1)
        
    # dist   reward
    # 0.1    -0.2
    # 0.2    -0.3
    # 0.3    -0.4
    # 1.0    -1.1

    reward *= dist_weight

    if DEBUG:
        print(f"nonlinear distance reward: {reward:.2f} (dist={dist:.2f}, weight={dist_weight})")
    return reward, {"ee_distance": dist}

def compute_distance_reward(base_pos, target_pos, dist_weight=1.0):
    """Compute distance-based reward (negative with scaling).

    Returns (reward, info) where info contains 'distance'.
    """
    base_xy = np.array(base_pos[:3], dtype=float)
    target_xy = np.array(target_pos[:3], dtype=float)
    dist = float(np.linalg.norm(target_xy - base_xy))
    reward_dist_term = float(dist_weight) * dist
    reward = -float(reward_dist_term)
    if DEBUG:
        print(f"distance reward: {reward:.2f} (dist={dist:.2f}, weight={dist_weight})")
    return reward, {"ee_distance": dist}


def compute_yaw_reward(base_pos, target_pos, current_yaw, wrap_angle_fn, yaw_weight=1.0):
    """Compute yaw-based reward (negative of scaled yaw error).

    Returns (reward, info) where info contains 'yaw_error' and 'desired_yaw'.
    """
    base_xy = np.array(base_pos[:2], dtype=float)
    target_xy = np.array(target_pos[:2], dtype=float)
    vec = target_xy - base_xy
    dist = float(np.linalg.norm(vec))
    desired_yaw = float(np.arctan2(vec[1], vec[0])) if dist > 1e-2 else float(current_yaw)
    yaw_error = float(wrap_angle_fn(desired_yaw - float(current_yaw)))

    # If yaw error exceeds 90 degrees, set yaw reward term to 0 to penalize large yaw errors more
    # if yaw_error > np.pi / 2 or yaw_error < -np.pi / 2:
    #     reward_yaw_term = 0.0
    #     print(f"Large yaw error {yaw_error:.2f} detected (>90 degrees). Setting yaw reward term to 0.")

    reward = -float(abs(yaw_error)) * yaw_weight
    if DEBUG:
        print(f"yaw reward: {reward:.2f} (desired_yaw = {desired_yaw:.2f}, current_yaw = {current_yaw:.2f}, yaw_weight = {yaw_weight:.2f})")
    return reward, {"yaw_error": yaw_error, "desired_yaw": desired_yaw}
