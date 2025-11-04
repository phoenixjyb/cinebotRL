import numpy as np

DEBUG=False

def compute_reward(base_pos, ee_pos, target_pos, base_yaw, wrap_angle_fn, base_lin_vel, base_ang_vel):
    is_base_over_target = False
    if base_pos[0] > target_pos[0] or (base_pos[1] * target_pos[1] > 0 and abs(base_pos[1]) > abs(target_pos[1])):
        is_base_over_target = True
        if DEBUG:
            print(f"Base is over target: base_pos={base_pos}, target_pos={target_pos}")

    total_reward = 0.0
    info = {}
    target_len = np.linalg.norm(np.array(target_pos[:2], dtype=float))
    base_len = np.linalg.norm(np.array(base_pos[:2], dtype=float))
    yaw_ratio = max(0.0, (target_len - 2.0 - base_len) / (target_len - 2.0)) * 10

    # 底盘位置误差——距离误差
    # dist_reward, dist_info = compute_distance_reward(base_pos, target_pos, 0.5)
    
    # 末端执行器误差——距离误差
    dist_reward, dist_info = compute_distance_reward(ee_pos, target_pos, 0.5)
    
    # 底盘角度误差
    # yaw_reward, yaw_info = compute_yaw_reward(base_pos, target_pos, base_yaw, wrap_angle_fn, yaw_ratio)
    
    # 末端执行器高程误差
    # reward_ee_height, ee_height_info = compute_height_reward(ee_pos[2], target_pos[2])

    # total_reward = dist_reward + yaw_reward + reward_vel
    total_reward = dist_reward
    info.update(dist_info)
    # info.update(yaw_info)
    # info.update(ee_height_info)
    if DEBUG:
        print(f"Reward: {total_reward:.2f} (dist: {dist_reward:.2f})")
    return total_reward, info

def compute_height_reward(ee_height, target_height):
    reward = 0.0
    height_diff = abs(ee_height - target_height)
    reward = -height_diff

    return reward, {"ee_height": ee_height, "target_height": target_height}

def compute_velocity_reward(base_lin_vel, base_ang_vel, is_base_over_target, max_vel, max_ang_vel):
    lin_reward, ang_reward = 0.0, 0.0
    if is_base_over_target:
        lin_reward = -abs(base_lin_vel)
        ang_reward = -abs(base_ang_vel)
    else:
        if base_lin_vel > max_vel:
            lin_reward = -(base_lin_vel - max_vel)
        if base_ang_vel > max_ang_vel:
            ang_reward = max(-1.0, -(base_ang_vel - max_ang_vel))
    reward = lin_reward + ang_reward

    print(f"Velocity Reward: lin_vel={base_lin_vel:.2f} (max {max_vel:.2f}), ang_vel={base_ang_vel:.2f} (max {max_ang_vel:.2f}),"
          f" lin_reward={lin_reward:.3f}, ang_reward={ang_reward:.3f}, total_reward={reward:.3f}")

    return reward, {"velocity_error": (lin_reward, ang_reward)}

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
