import os
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
import json
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from pybullet_envs.mobile_mm import MobileMMBulletEnv
from pybullet_envs.mobile_mm_traj import MobileMMTrajEnv
from pybullet_envs.target_generator import RandomTargetForEpisode, JSONNearestTargetGenerator
from pybullet_envs.transformer_extractor import TransformerFeaturesExtractor

def test_trained_model(
    model_path,
    num_episodes=1,
    json_paths=None,
    test_txt=None,
    should_vis=True,
    max_steps=500,
    render=False,
    *,
    robot: str = "mobile_mm",
    urdf_path: str | None = None,
    frame_skip: int = 24,
):
    """
    测试训练好的模型并生成可视化结果
    
    Args:
        model_path: 模型文件路径
        num_episodes: 测试的episode数量
        save_dir: 结果保存目录
    """
    from pathlib import Path
    # Add project to path
    save_dir = Path(model_path).parent
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载模型（注意：有可能模型使用了自定义的特征提取器/策略，
    # 确保相关模块可被导入。显式选择 device 以保证模型在 CUDA 上运行）
    print(f"Loading model from: {model_path}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device for inference: {device}")
    model = PPO.load(model_path, device=device)
    
    policy = model.policy
    print(policy)
    
    # 创建测试环境（render 参数可控）
    env_fn = lambda: MobileMMTrajEnv(
        robot=robot,
        urdf_path=urdf_path,
        frame_skip=frame_skip,
        max_steps=max_steps,
        render=render,
        target_generator=JSONNearestTargetGenerator(
            json_paths=json_paths or [],
            json_txt=test_txt,
            mode="seq",
        ),
    )
    vec_env = DummyVecEnv([env_fn])
    
    # 存储测试结果
    all_distances = []
    all_rewards = []
    all_traj_path = []
    success_count = 0
    
    for episode in range(num_episodes):
        print(f"\n=== Episode {episode + 1} ===")
        
        
        obs = vec_env.reset()
        # get underlying env instance from DummyVecEnv for visualization helpers
        env = vec_env.envs[0]
        
        # target = vec_env.

        # collect end-effector and base positions for trajectory overlay
        episode_ee_positions = []
        episode_base_positions = []
        episode_targets = []
        # collect per-step velocities (world frame and chassis frame)
        episode_world_vels = []  # list of (vx_world, vy_world)
        episode_chassis_vels = []  # list of (vx_chassis, vy_chassis)
        # collect per-step control_info dicts if env populates them
        episode_control_info = []
        # collect per-step traj id (index of current target along trajectory)
        episode_traj_ids = []
        # collect per-step abstract chassis yaw (radians)
        episode_chassis_yaw = []
        # collect per-step arm joint angles (6 joints expected)
        episode_joint_angles = []  # list of length-6 tuples
        # collect per-step end-effector heights (z)
        episode_ee_heights = []
        

        def _get_ee_pos_from_env(e):
            return e._get_ee_pos()

        def _get_base_pos_from_env(e):
            return e._get_base_pos()

        # initial ee pos (if available)
        init_ee = _get_ee_pos_from_env(env)
        target = None
        try:
            target = env._get_target_position()
        except Exception:
            try:
                # maybe stored as attribute
                target = getattr(env, '_target', None)
            except Exception:
                target = None
        if init_ee is not None:
            try:
                episode_ee_positions.append(tuple(float(x) for x in init_ee[:3]))
            except Exception:
                episode_ee_positions.append(init_ee)
        # record initial target if available
        if target is not None:
            try:
                episode_targets.append(tuple(float(x) for x in target[:3]))
            except Exception:
                episode_targets.append(target)
        # initial base pos (if available)
        init_base = _get_base_pos_from_env(env)
        if init_base is not None:
            try:
                episode_base_positions.append(tuple(float(x) for x in init_base[:3]))
            except Exception:
                episode_base_positions.append(init_base)

        episode_distances = []
        episode_rewards = []
        terminated = False
        step_count = 0
        all_traj_path.append(env._target_generator.get_traj_path())
        cur_traj_path = env._target_generator.get_traj_path()
        
        while not terminated:
            # 使用模型预测动作
            action, _states = model.predict(obs, deterministic=True)
            res = vec_env.step(action)

            # support both Gym (obs, rewards, dones, infos) and Gymnasium (obs, rewards, terminated, truncated, infos)
            if len(res) == 4:
                obs, rewards, dones, infos = res
                done_flag = bool(dones[0])
            else:
                obs, rewards, terminateds, truncateds, infos = res
                done_flag = bool(terminateds[0]) or bool(truncateds[0])

            # 记录数据 (vectorized env returns arrays/lists)
            # import pdb; pdb.set_trace()
            try:
                dist_val = infos[0].get('ee_distance', None)
            except Exception:
                dist_val = None
            episode_distances.append(dist_val)
            # rewards may be scalar or array-like
            r0 = rewards[0] if hasattr(rewards, '__len__') else rewards
            episode_rewards.append(float(r0))
            step_count += 1
            # append current ee and base pos (if available)
            try:
                cur_ee = _get_ee_pos_from_env(env)
                if cur_ee is not None:
                    episode_ee_positions.append(tuple(float(x) for x in cur_ee[:3]))
                    # record ee height
                    try:
                        episode_ee_heights.append(float(cur_ee[2]) )
                    except Exception:
                        episode_ee_heights.append(np.nan)
            except Exception:
                pass
            try:
                cur_base = _get_base_pos_from_env(env)
                if cur_base is not None:
                    episode_base_positions.append(tuple(float(x) for x in cur_base[:3]))
            except Exception:
                pass

            # record current target (time-varying)
            try:
                cur_target = env._get_target_position()
                if cur_target is not None:
                    try:
                        episode_targets.append(tuple(float(x) for x in cur_target[:3]))
                    except Exception:
                        episode_targets.append(cur_target)
            except Exception:
                # some envs may not expose _get_target_position
                try:
                    tattr = getattr(env, '_target', None)
                    if tattr is not None:
                        episode_targets.append(tuple(float(x) for x in tattr[:3]))
                except Exception:
                    pass

            # record current traj id if available
            try:
                traj_id = getattr(env, '_traj_id', None)
                if traj_id is None:
                    # some envs may store as attribute in unwrapped
                    traj_id = getattr(getattr(env, 'unwrapped', None), '_traj_id', None)
                episode_traj_ids.append(float(traj_id) if traj_id is not None else np.nan)
            except Exception:
                episode_traj_ids.append(np.nan)

            # collect env.control_info if present (store shallow copy to avoid later mutation)
            try:
                ci = getattr(env, 'control_info', None)
                if ci is not None:
                    # try to copy relevant numeric fields; keep dict if copy fails
                    try:
                        episode_control_info.append(dict(ci))
                    except Exception:
                        episode_control_info.append(ci)
                else:
                    episode_control_info.append(None)
            except Exception:
                episode_control_info.append(None)

            # collect velocities: prefer explicit attributes if present, else try to call _get_base_pos()
            try:
                # ensure env has updated base/link velocity stored
                try:
                    # calling _get_base_pos() will update internal velocity attributes in many env implementations
                    env._get_base_pos()
                except Exception:
                    pass

                v_world = None
                v_chassis = None
                # possible attribute names used in the environment
                if hasattr(env, '_abstract_chassis_lin_vel_world'):
                    v_world = getattr(env, '_abstract_chassis_lin_vel_world')
                elif hasattr(env, '_abstract_chassis_lin_vel'):
                    v_world = getattr(env, '_abstract_chassis_lin_vel')

                if hasattr(env, '_abstract_chassis_lin_vel_chassis'):
                    v_chassis = getattr(env, '_abstract_chassis_lin_vel_chassis')
                elif hasattr(env, '_abstract_chassis_lin_vel'):
                    # fallback: treat same as world if chassis not available
                    v_chassis = getattr(env, '_abstract_chassis_lin_vel')

                if v_world is not None:
                    episode_world_vels.append((float(v_world[0]), float(v_world[1])))
                else:
                    episode_world_vels.append((np.nan, np.nan))

                if v_chassis is not None:
                    episode_chassis_vels.append((float(v_chassis[0]), float(v_chassis[1])))
                else:
                    episode_chassis_vels.append((np.nan, np.nan))
            except Exception:
                # on any failure, append NaNs to keep lengths consistent
                episode_world_vels.append((np.nan, np.nan))
                episode_chassis_vels.append((np.nan, np.nan))

            # collect chassis yaw if available (fall back to NaN)
            try:
                yaw = None
                if hasattr(env, '_abstract_chassis_yaw'):
                    yaw = getattr(env, '_abstract_chassis_yaw')
                elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, '_abstract_chassis_yaw'):
                    yaw = env.unwrapped._abstract_chassis_yaw
                elif hasattr(env, 'env') and hasattr(env.env, '_abstract_chassis_yaw'):
                    yaw = env.env._abstract_chassis_yaw
                # normalize to float if possible
                # if abs(yaw - episode_chassis_yaw[-1]) > 0.02:
                #     import pdb; pdb.set_trace()
                episode_chassis_yaw.append(float(yaw) if yaw is not None else np.nan)
            except Exception:
                episode_chassis_yaw.append(np.nan)
            except Exception:
                # on any failure, append NaNs to keep lengths consistent
                episode_world_vels.append((np.nan, np.nan))
                episode_chassis_vels.append((np.nan, np.nan))

            # collect arm joint angles if the env exposes _get_arm_state()
            try:
                arm_state = None
                if hasattr(env, '_get_arm_state'):
                    arm_state = env._get_arm_state()
                elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, '_get_arm_state'):
                    arm_state = env.unwrapped._get_arm_state()
                elif hasattr(env, 'env') and hasattr(env.env, '_get_arm_state'):
                    arm_state = env.env._get_arm_state()

                if arm_state is None:
                    episode_joint_angles.append((np.nan,)*6)
                else:
                    arr = np.array(arm_state).ravel()
                    if arr.size < 6:
                        tmp = np.full(6, np.nan, dtype=float)
                        tmp[:arr.size] = arr
                        arr = tmp
                    episode_joint_angles.append(tuple(float(x) for x in arr[:6]))
            except Exception:
                episode_joint_angles.append((np.nan,)*6)

            terminated = done_flag
        
            # 检查是否成功到达目标
            final_distance = episode_distances[-1] if episode_distances else float('nan')
            
            if step_count % 1000 == 0:
                img_path = os.path.join(save_dir, f"episode_{episode+1}_step_{step_count:04d}.png")
                env.save_robot_image(
                    img_path,
                    width=800,
                    height=600,
                    traj=episode_ee_positions,
                    draw_origin=True,
                    target=(episode_targets[-1] if episode_targets else None),
                )
                
                
        if final_distance < 0.05:
            success_count += 1
            print(f"Episode {episode + 1}: SUCCESS (distance: {final_distance:.4f})")
        else:
            print(f"Episode {episode + 1}: FAILED (distance: {final_distance:.4f})")
        

        all_distances.append(episode_distances)
        all_rewards.append(episode_rewards)

        # accumulate base and ee positions for cross-episode plotting (still keep for summary if needed)
        if 'all_base_positions' not in locals():
            all_base_positions = []
        if 'all_ee_positions' not in locals():
            all_ee_positions = []
        # store collected trajectories and target per episode (trim possible trailing incomplete entries)
        episode_base_trimmed = episode_base_positions[:-2]
        episode_ee_trimmed = episode_ee_positions[:-2]
        all_base_positions.append(episode_base_trimmed)
        all_ee_positions.append(episode_ee_trimmed)
        if 'all_targets' not in locals():
            all_targets = []
        # store the per-step target sequence for this episode (as array)
        try:
            episode_targets_trimmed = episode_targets[:-2]
        except Exception:
            episode_targets_trimmed = episode_targets
        all_targets.append(np.array(episode_targets_trimmed) if len(episode_targets_trimmed) > 0 else None)
        # store control_info for this episode
        if 'all_control_info' not in locals():
            all_control_info = []
        try:
            all_control_info.append(episode_control_info[:-2])
        except Exception:
            all_control_info.append([])

        # Generate per-episode 3D trajectory plot immediately for this episode
        print(f"Mean error : {np.mean(episode_distances[:-2]):.2f} m")
        if should_vis or (np.mean(episode_distances[:-2]) > 0.1):
            # pass single-episode lists wrapped in lists to match function signature
                generate_3d_trajectory_plot([episode_base_trimmed], [episode_ee_trimmed], [episode_targets_trimmed],
                                            cur_traj_path, save_dir, episode=episode+1)
                generate_velocity_plot(episode_world_vels, episode_chassis_vels, save_dir, 
                                   episode=episode+1, joint_angles=episode_joint_angles,
                                   ee_heights=episode_ee_heights, traj_ids=episode_traj_ids[:-2],
                                   ee_distances=episode_distances[:-2], chassis_yaw=episode_chassis_yaw[:-2],
                                       control_infos=episode_control_info[:-2], traj_path=cur_traj_path)
    
    vec_env.close()
    print(f"\n")
    
    # 生成性能图表
    generate_performance_plots(all_distances, all_rewards, save_dir, success_count, num_episodes)
    
    # 生成测试报告
    generate_test_report(all_distances, all_rewards, success_count, num_episodes, save_dir)
    # 计算并打印所有 episode 的 ee_distance 平均值（忽略 None/NaN）
    try:
        flat_vals = []
        for d in all_distances:
            for v in d:
                try:
                    if v is None:
                        continue
                    fv = float(v)
                    if np.isnan(fv):
                        continue
                    flat_vals.append(fv)
                except Exception:
                    continue
        if len(flat_vals) > 0:
            avg_ee_distance = float(np.mean(flat_vals))
            print(f"Average ee_distance across all episodes and steps: {avg_ee_distance:.6f} m")
        else:
            print("No ee_distance values recorded across episodes.")
    except Exception as e:
        print(f"Failed to compute average ee_distance: {e}")

    return all_distances, all_rewards, success_count


def generate_performance_plots(all_distances, all_rewards, save_dir, success_count, num_episodes):
    """生成性能图表"""
    plt.figure(figsize=(15, 10))
    
    # 距离变化图
    plt.subplot(2, 2, 1)
    for i, distances in enumerate(all_distances):
        plt.plot(distances, label=f'Episode {i+1}')
    plt.xlabel('Steps')
    plt.ylabel('Distance to Target')
    plt.title('Distance to Target Over Time')
    plt.legend()
    plt.grid(True)
    
    # 奖励变化图
    plt.subplot(2, 2, 2)
    for i, rewards in enumerate(all_rewards):
        plt.plot(rewards, label=f'Episode {i+1}')
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.title('Reward Over Time')
    plt.legend()
    plt.grid(True)
    
    # 平均距离图
    plt.subplot(2, 2, 3)
    min_length = min(len(d) for d in all_distances)
    avg_distances = np.mean([d[:min_length] for d in all_distances], axis=0)
    plt.plot(avg_distances)
    plt.xlabel('Steps')
    plt.ylabel('Average Distance')
    plt.title('Average Distance Across Episodes')
    plt.grid(True)
    
    # 成功率饼图
    plt.subplot(2, 2, 4)
    success_rate = success_count / num_episodes * 100
    labels = ['Success', 'Failed']
    sizes = [success_count, num_episodes - success_count]
    colors = ['lightgreen', 'lightcoral']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title(f'Success Rate: {success_rate:.1f}%')
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, "performance_summary.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    
    print(f"Saved performance summary: {plot_path}")


def generate_test_report(all_distances, all_rewards, success_count, num_episodes, save_dir):
    """生成测试报告"""
    report_path = os.path.join(save_dir, "test_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("=== Trained Model Test Report ===\n\n")
        f.write(f"Number of episodes tested: {num_episodes}\n")
        f.write(f"Success count: {success_count}\n")
        f.write(f"Success rate: {success_count/num_episodes*100:.1f}%\n\n")
        
        f.write("Episode Performance Summary:\n")
        f.write("-" * 50 + "\n")
        
        for i, (distances, rewards) in enumerate(zip(all_distances, all_rewards)):
            f.write(f"Episode {i+1}:\n")
            f.write(f"  Steps: {len(distances)}\n")
            f.write(f"  Initial distance: {distances[0]:.4f}\n")
            f.write(f"  Final distance: {distances[-1]:.4f}\n")
            f.write(f"  Total reward: {sum(rewards):.4f}\n")
            f.write(f"  Success: {'Yes' if distances[-1] < 0.05 else 'No'}\n")
            f.write("\n")
        
        # 总体统计
        all_final_distances = [d[-1] for d in all_distances]
        all_total_rewards = [sum(r) for r in all_rewards]
        
        f.write("Overall Statistics:\n")
        f.write(f"  Average final distance: {np.mean(all_final_distances):.4f}\n")
        f.write(f"  Std final distance: {np.std(all_final_distances):.4f}\n")
        f.write(f"  Average total reward: {np.mean(all_total_rewards):.4f}\n")
        f.write(f"  Best final distance: {min(all_final_distances):.4f}\n")
        f.write(f"  Worst final distance: {max(all_final_distances):.4f}\n")
    
    print(f"Saved test report: {report_path}")


def generate_3d_trajectory_plot(all_base_positions, all_ee_positions, all_targets, traj_path, save_dir, episode=1):
    """生成3D轨迹图：机器人底座和末端执行器位置的变化（不渲染机器人）"""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    # Create a 2x2 layout:
    # [ 3D traj (0,0) | EE X/Y/Z vs idx (0,1) ]
    # [ target X vs idx (1,0) | target Y vs idx (1,1) ]
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 0.8])

    ax3d = fig.add_subplot(gs[0, 0], projection='3d')
    ax_ee = fig.add_subplot(gs[0, 1])
    ax_tx = fig.add_subplot(gs[1, 0])
    ax_ty = fig.add_subplot(gs[1, 1])

    colors = plt.cm.tab10.colors

    # Plot each episode's data (expect lists of per-episode trajectories)
    for i, (base_traj, ee_traj, target_p) in enumerate(zip(all_base_positions, all_ee_positions, all_targets)):
        try:
            if base_traj:
                base_arr = np.array(base_traj)
                if base_arr.ndim == 2 and base_arr.shape[1] >= 3:
                    ax3d.plot(base_arr[:, 0], base_arr[:, 1], base_arr[:, 2], linestyle='-', marker='o',
                              color=colors[i % len(colors)], label=f'Base p{i+1}')
                    ax3d.scatter(base_arr[0, 0], base_arr[0, 1], base_arr[0, 2], color=colors[i % len(colors)], marker='D', s=40)
                    ax3d.scatter(base_arr[-1, 0], base_arr[-1, 1], base_arr[-1, 2], color=colors[i % len(colors)], marker='X', s=60)

            if ee_traj:
                ee_arr = np.array(ee_traj)
                if ee_arr.ndim == 2 and ee_arr.shape[1] >= 3:
                    ax3d.plot(ee_arr[:, 0], ee_arr[:, 1], ee_arr[:, 2], linestyle='--', marker='.',
                              color=colors[(i+3) % len(colors)], label=f'EE p{i+1}')
                    ax3d.scatter(ee_arr[0, 0], ee_arr[0, 1], ee_arr[0, 2], color=colors[(i+3) % len(colors)], marker='s', s=30)
                    ax3d.scatter(ee_arr[-1, 0], ee_arr[-1, 1], ee_arr[-1, 2], color=colors[(i+3) % len(colors)], marker='*', s=70)

                    # plot EE components aligned to requested layout:
                    # top-right: EE X, bottom-left: EE Y, bottom-right: EE Z
                    steps = np.arange(ee_arr.shape[0])
                    ax_ee.plot(steps, ee_arr[:, 0], label='EE X', color='C0')
                    ax_tx.plot(steps, ee_arr[:, 1], label='EE Y', color='C1')
                    ax_ty.plot(steps, ee_arr[:, 2], label='EE Z', color='C2')

            # Prefer provided target_p sequence. If absent, fall back to reading a default JSON path.
            target_arr = None
            if target_p is not None:
                try:
                    t = np.array(target_p)
                    if t.ndim == 2 and t.shape[1] >= 3:
                        target_arr = t
                except Exception:
                    target_arr = None

            if target_arr is not None:
                try:
                    ax3d.plot(target_arr[:, 0], target_arr[:, 1], target_arr[:, 2], linestyle='--', color='green', label=f'traj p{i+1}')
                    # also plot target components overlaid with EE components:
                    steps_t = np.arange(target_arr.shape[0])
                    # top-right overlay: Target X
                    ax_ee.plot(steps_t, target_arr[:, 0], label='Target X', color='C3', linestyle='--')
                    ax_ee.set_xlabel('point index')
                    ax_ee.set_ylabel('position (m)')
                    ax_ee.grid(True, alpha=0.3)

                    # bottom-left overlay: Target Y
                    ax_tx.plot(steps_t, target_arr[:, 1], label='Target Y', color='C4', linestyle='--')
                    ax_tx.set_xlabel('point index')
                    ax_tx.set_ylabel('position (m)')
                    ax_tx.grid(True, alpha=0.3)

                    # bottom-right overlay: Target Z
                    ax_ty.plot(steps_t, target_arr[:, 2], label='Target Z', color='C5', linestyle='--')
                    ax_ty.set_xlabel('point index')
                    ax_ty.set_ylabel('position (m)')
                    ax_ty.grid(True, alpha=0.3)
                except Exception:
                    pass
        except Exception:
            # keep plotting other episodes even if one fails
            continue

    ax3d.set_xlabel('X (m)')
    ax3d.set_ylabel('Y (m)')
    ax3d.set_zlabel('Z (m)')
    # include traj path (short name) in the title if provided
    title_main = '3D Trajectories: Base (solid) and End-Effector (dashed)'
    if traj_path:
        try:
            short = os.path.basename(traj_path)
        except Exception:
            short = str(traj_path)
        ax3d.set_title(f"{title_main} — {short}")
    else:
        ax3d.set_title(title_main)
    ax3d.legend(loc='upper left')

    # finalize EE / target subplot appearance and add clear legends
    ax_ee.set_title('EE X vs Target X')
    ax_ee.set_xlabel('point index')
    ax_ee.set_ylabel('position (m)')
    ax_ee.legend(loc='best', fontsize='small')
    ax_ee.grid(True, alpha=0.3)

    ax_tx.set_title('EE Y vs Target Y')
    ax_tx.set_xlabel('point index')
    ax_tx.set_ylabel('position (m)')
    # show legend for EE Y and Target Y if present
    try:
        ax_tx.legend(loc='best', fontsize='small')
    except Exception:
        pass
    ax_tx.grid(True, alpha=0.3)

    ax_ty.set_title('EE Z vs Target Z')
    ax_ty.set_xlabel('point index')
    ax_ty.set_ylabel('position (m)')
    try:
        ax_ty.legend(loc='best', fontsize='small')
    except Exception:
        pass
    ax_ty.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(save_dir, f'trajectories_3d_episode_{episode}_{os.path.basename(traj_path)[:-5]}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved 3D trajectory plot: {out_path}")


def generate_velocity_plot(world_vels, chassis_vels, save_dir, episode=1,
                           joint_angles=None, ee_heights=None, traj_ids=None,
                           ee_distances=None, chassis_yaw=None, control_infos=None, traj_path=None):
    """Generate a 2-panel plot (side-by-side) comparing world-frame and chassis-frame
    longitudinal (vx) and lateral (vy) linear velocities over time for one episode.

    Args:
        world_vels: list of (vx_world, vy_world) per step
        chassis_vels: list of (vx_chassis, vy_chassis) per step
        save_dir: directory to save the PNG
        episode: episode number (used for filename)
    """
    # import pdb; pdb.set_trace()
    # convert to arrays (N,2)
    w = np.array(world_vels, dtype=np.float32) if len(world_vels) > 0 else np.zeros((0, 2), dtype=np.float32)
    c = np.array(chassis_vels, dtype=np.float32) if len(chassis_vels) > 0 else np.zeros((0, 2), dtype=np.float32)

    steps_w = np.arange(w.shape[0]) if w.shape[0] > 0 else np.arange(0)
    steps_c = np.arange(c.shape[0]) if c.shape[0] > 0 else np.arange(0)

    # create a figure with 5 rows: two side-by-side velocity plots on top row,
    # joint-angles and ee-height in the second row, traj_id and ee_distance in the third row,
    # chassis_yaw as a full-width fourth row, and control_info dx/dy comparison in fifth row
    fig = plt.figure(figsize=(14, 14))
    gs = fig.add_gridspec(5, 2, height_ratios=[1, 0.8, 0.6, 0.6, 0.6])

    ax = fig.add_subplot(gs[0, 0])
    # World-frame velocities
    if w.shape[0] > 0:
        ax.plot(steps_w, w[:, 0], label='vx_world (forward)', color='C0')
        ax.plot(steps_w, w[:, 1], label='vy_world (lateral)', color='C1')
    else:
        ax.plot([], [], label='no data')
    ax.set_xlabel('Step')
    ax.set_ylabel('Linear velocity (m/s)')
    ax.set_title('World-frame linear velocities')
    ax.grid(True)
    ax.legend()

    ax2 = fig.add_subplot(gs[0, 1], sharey=ax)
    # Chassis-frame velocities
    if c.shape[0] > 0:
        ax2.plot(steps_c, c[:, 0], label='vx_chassis (forward)', color='C0')
        ax2.plot(steps_c, c[:, 1], label='vy_chassis (lateral)', color='C1')
    else:
        ax2.plot([], [], label='no data')
    ax2.set_xlabel('Step')
    ax2.set_title('Chassis-frame linear velocities')
    ax2.grid(True)
    ax2.legend()

    # Second row: left = joint angles, right = ee height
    ax_joint = fig.add_subplot(gs[1, 0])
    if joint_angles is None or len(joint_angles) == 0:
        ax_joint.text(0.5, 0.5, 'No joint angle data', ha='center', va='center')
    else:
        ja = np.array(joint_angles, dtype=float)[:-2]
        steps_j = np.arange(ja.shape[0])
        # plot up to 6 joints
        colors = [f'C{i}' for i in range(6)]
        for i in range(min(6, ja.shape[1])):
            ax_joint.plot(steps_j, np.rad2deg(ja[:, i]), label=f'joint{i+1} (deg)', color=colors[i], linestyle='-')
        ax_joint.set_xlabel('Step')
        ax_joint.set_ylabel('Joint angle (deg)')
        ax_joint.set_title('Arm joint angles')
        ax_joint.grid(True)
        ax_joint.legend(ncol=3, fontsize='small')

    ax_ee = fig.add_subplot(gs[1, 1])
    if ee_heights is None or len(ee_heights) == 0:
        ax_ee.text(0.5, 0.5, 'No EE height data', ha='center', va='center')
    else:
        eh = np.array(ee_heights, dtype=float)[:-2]
        steps_e = np.arange(eh.shape[0])
        ax_ee.plot(steps_e, eh, label='EE height (m)', color='C2')
        ax_ee.set_xlabel('Step')
        ax_ee.set_ylabel('Height (m)')
        ax_ee.set_title('End-Effector Height')
        ax_ee.grid(True)
        ax_ee.legend()

    # include trajectory path (short name) in the suptitle when available
    title = f'Episode {episode}: Velocities, Joint Angles and Chassis Yaw'
    if traj_path:
        try:
            short = os.path.basename(traj_path)
        except Exception:
            short = str(traj_path)
        title = f"{title} — {short}"
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # bottom row: plot traj_ids over steps if provided

    try:
        # bottom row: left = traj_id, right = ee_distance. Create side-by-side axes.
        left_needed = traj_ids is not None and len(traj_ids) > 0
        right_needed = ee_distances is not None and len(ee_distances) > 0
        if left_needed or right_needed:
            ax_left = fig.add_subplot(gs[2, 0])
            ax_right = fig.add_subplot(gs[2, 1], sharex=ax_left)

            if left_needed:
                tid = np.array(traj_ids, dtype=float)
                steps_t = np.arange(tid.shape[0])
                ax_left.plot(steps_t, tid, linestyle='-', marker='.', color='C3', label='traj_id')
                ax_left.set_xlabel('Step')
                ax_left.set_ylabel('traj_id')
                ax_left.set_title('Trajectory index over steps')
                ax_left.grid(True)
                ax_left.legend()
            else:
                ax_left.text(0.5, 0.5, 'No traj_id data', ha='center', va='center')

            if right_needed:
                ed = np.array(ee_distances, dtype=float)
                steps_e = np.arange(ed.shape[0])
                ax_right.plot(steps_e, ed, linestyle='--', marker='.', color='C4', label='ee_distance (m)')
                ax_right.set_xlabel('Step')
                ax_right.set_ylabel('ee_distance (m)')
                ax_right.set_title('EE distance over steps')
                ax_right.grid(True)
                ax_right.legend()
            else:
                ax_right.text(0.5, 0.5, 'No EE distance data', ha='center', va='center')
    except Exception:
        pass

    # Fourth row: chassis yaw (full-width)
    try:
        ax_yaw = fig.add_subplot(gs[3, :])
        if chassis_yaw is None or len(chassis_yaw) == 0:
            ax_yaw.text(0.5, 0.5, 'No chassis yaw data', ha='center', va='center')
        else:
            cy = np.array(chassis_yaw, dtype=float)
            # trim to remove possible trailing entries used for state buffering
            cy = cy[:-2] if cy.shape[0] > 2 else cy
            steps_y = np.arange(cy.shape[0])
            ax_yaw.plot(steps_y, np.unwrap(cy), linestyle='-', marker='.', color='C5', label='chassis_yaw (rad)')
            ax_yaw.set_xlabel('Step')
            ax_yaw.set_ylabel('Yaw (rad)')
            ax_yaw.set_title('Abstract Chassis Yaw over Steps')
            ax_yaw.grid(True)
            ax_yaw.legend()
    except Exception:
        pass

    # Fifth row: control_info comparison (two side-by-side plots for dx and dy)
    try:
        ax_dx = fig.add_subplot(gs[4, 0])
        ax_dy = fig.add_subplot(gs[4, 1], sharey=ax_dx)

        if control_infos is None or len(control_infos) == 0:
            ax_dx.text(0.5, 0.5, 'No control_info data', ha='center', va='center')
            ax_dy.text(0.5, 0.5, 'No control_info data', ha='center', va='center')
        else:
            # extract target.dx, reality.dx and target.dy, reality.dy sequences
            targ_dx = []
            real_dx = []
            targ_dy = []
            real_dy = []
            for ci in control_infos:
                if ci is None:
                    targ_dx.append(np.nan); real_dx.append(np.nan); targ_dy.append(np.nan); real_dy.append(np.nan)
                    continue
                # prefer dict access; handle nested dicts
                try:
                    t = ci.get('target', {}) if isinstance(ci, dict) else {}
                    r = ci.get('reality', {}) if isinstance(ci, dict) else {}
                    td_x = t.get('dx', np.nan) if isinstance(t, dict) else np.nan
                    rd_x = r.get('dx', np.nan) if isinstance(r, dict) else np.nan
                    td_y = t.get('dy', np.nan) if isinstance(t, dict) else np.nan
                    rd_y = r.get('dy', np.nan) if isinstance(r, dict) else np.nan
                except Exception:
                    td_x = np.nan; rd_x = np.nan; td_y = np.nan; rd_y = np.nan
                try:
                    targ_dx.append(float(td_x))
                except Exception:
                    targ_dx.append(np.nan)
                try:
                    real_dx.append(float(rd_x))
                except Exception:
                    real_dx.append(np.nan)
                try:
                    targ_dy.append(float(td_y))
                except Exception:
                    targ_dy.append(np.nan)
                try:
                    real_dy.append(float(rd_y))
                except Exception:
                    real_dy.append(np.nan)

            steps_ci = np.arange(len(targ_dx))
            ax_dx.plot(steps_ci, targ_dx, label='target.dx', color='C0')
            ax_dx.plot(steps_ci, real_dx, label='reality.dx', color='C1', linestyle='--')
            ax_dx.set_xlabel('Step')
            ax_dx.set_ylabel('dx (m)')
            ax_dx.set_title('Control: target.dx vs reality.dx')
            ax_dx.grid(True)
            ax_dx.legend()

            ax_dy.plot(steps_ci, targ_dy, label='target.dy', color='C0')
            ax_dy.plot(steps_ci, real_dy, label='reality.dy', color='C1', linestyle='--')
            ax_dy.set_xlabel('Step')
            ax_dy.set_ylabel('dy (m)')
            ax_dy.set_title('Control: target.dy vs reality.dy')
            ax_dy.grid(True)
            ax_dy.legend()
    except Exception:
        pass

    out_path = os.path.join(save_dir, f'episode_{episode}_{os.path.basename(traj_path)[:-5]}_velocities.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved velocity plot: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to PPO .zip checkpoint")
    parser.add_argument("--robot", type=str, default="mobile_mm", choices=["mobile_mm", "recomo"])
    parser.add_argument("--urdf_path", type=str, default=None, help="Optional URDF override (robot-specific)")
    parser.add_argument("--test_txt", type=str, default="linux_env_dev/new_json_50/test.txt",
                        help="Txt file with one json path per line")
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--frame_skip", type=int, default=24)
    parser.add_argument("--render", action="store_true", help="Use PyBullet GUI")
    parser.add_argument("--should_vis", action="store_true", help="Always render plots (can be slow)")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    print("Starting model testing...")
    distances, rewards, success_count = test_trained_model(
        args.model_path,
        test_txt=args.test_txt,
        num_episodes=int(args.num_episodes),
        render=bool(args.render),
        max_steps=int(args.max_steps),
        should_vis=bool(args.should_vis),
        robot=args.robot,
        urdf_path=args.urdf_path,
        frame_skip=int(args.frame_skip),
    )
    print(
        f"\nTesting completed! Success rate: {success_count}/{args.num_episodes} "
        f"({(success_count / max(1, int(args.num_episodes))) * 100:.1f}%)"
    )
