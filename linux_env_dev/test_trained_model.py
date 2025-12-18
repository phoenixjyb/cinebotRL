import os
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from pybullet_envs.mobile_mm import MobileMMBulletEnv
from pybullet_envs.mobile_mm_traj import MobileMMTrajEnv
from pybullet_envs.target_generator import RandomTargetForEpisode, JSONNearestTargetGenerator
from pybullet_envs.transformer_extractor import TransformerFeaturesExtractor

def test_trained_model(
    model_path,
    num_episodes=1,
    max_steps=500,
    render=False,
    low=(4.0, -2.0, 0.5),
    high=(6.0, 2.0, 1.5),
    *,
    robot: str = "mobile_mm",
    urdf_path: str | None = None,
    frame_skip: int = 24,
    test_txt: str | None = None,
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
    # env_fn = lambda: MobileMMBulletEnv(render=render, max_steps=max_steps,
    #                                    target_generator=RandomTargetForEpisode(
    #                                        low=low,
    #                                        high=high
    #                                    ))
    # If test_txt is provided, iterate trajectories sequentially; else fall back to a single demo path.
    if test_txt is None:
        json_paths = ["trajectoryToLearn/world_json/scene_1/traj_1.json"]
        target_gen = JSONNearestTargetGenerator(json_paths=json_paths, mode="seq")
    else:
        target_gen = JSONNearestTargetGenerator(json_paths=[], json_txt=test_txt, mode="seq")

    env_fn = lambda: MobileMMTrajEnv(
        robot=robot,
        urdf_path=urdf_path,
        frame_skip=frame_skip,
        max_steps=max_steps,
        render=render,
        target_generator=target_gen,
    )
    vec_env = DummyVecEnv([env_fn])
    
    # 存储测试结果
    all_distances = []
    all_rewards = []
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
        # collect per-step velocities (world frame and chassis frame)
        episode_world_vels = []  # list of (vx_world, vy_world)
        episode_chassis_vels = []  # list of (vx_chassis, vy_chassis)
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
            
            # # 每50步保存一张图片
            # if step_count % 50 == 0:
            #     img_path = os.path.join(save_dir, f"episode_{episode+1}_step_{step_count}.png")
            #     try:
            #         # pass collected EE trajectory for overlay
            #         env.save_robot_image(img_path, width=800, height=600, traj=episode_ee_positions, draw_origin=True)
            #         print(f"Saved image: {img_path}")
            #     except Exception as e:
            #         print(f"Failed to save image: {e}")
        
            # 检查是否成功到达目标
            final_distance = episode_distances[-1] if episode_distances else float('nan')
        
        if final_distance < 0.05:
            success_count += 1
            print(f"Episode {episode + 1}: SUCCESS (distance: {final_distance:.4f})")
        else:
            print(f"Episode {episode + 1}: FAILED (distance: {final_distance:.4f})")
        
        # episode_ee_positions
        # episode_base_positions
        # import pdb; pdb.set_trace()

        all_distances.append(episode_distances)
        all_rewards.append(episode_rewards)
        
        # 保存该episode的轨迹图片（包含已经收集的EE轨迹），并传入target以在图中显示
        # final_img_path = os.path.join(save_dir, f"episode_{episode+1}_final.png")
        # try:
        #     env.save_robot_image(final_img_path, width=800, height=600, traj=episode_ee_positions, draw_origin=True, target=target)
        #     print(f"Saved final image: {final_img_path}")
        # except Exception as e:
        #     print(f"Failed to save final image: {e}")

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
        all_targets.append(target)

        # Generate per-episode 3D trajectory plot immediately for this episode
        try:
            # pass single-episode lists wrapped in lists to match function signature
            generate_3d_trajectory_plot([episode_base_trimmed], [episode_ee_trimmed], [target], save_dir, episode=episode+1)
        except Exception as e:
            print(f'Failed to generate per-episode 3D trajectory plot for episode {episode+1}: {e}')

        # Generate per-episode velocity plot (world-frame vs chassis-frame)
        try:
            generate_velocity_plot(episode_world_vels, episode_chassis_vels, save_dir, episode=episode+1, joint_angles=episode_joint_angles, ee_heights=episode_ee_heights)
        except Exception as e:
            print(f'Failed to generate per-episode velocity plot for episode {episode+1}: {e}')
    
    vec_env.close()
    
    # 生成性能图表
    generate_performance_plots(all_distances, all_rewards, save_dir, success_count, num_episodes)
    
    # 生成测试报告
    generate_test_report(all_distances, all_rewards, success_count, num_episodes, save_dir)
    # 生成3D轨迹图（底座与末端执行器）
    # try:
    #     if 'all_base_positions' in locals() and 'all_ee_positions' in locals():
    #         generate_3d_trajectory_plot(all_base_positions, all_ee_positions, all_targets, save_dir, episode)
    #     else:
    #         print('No trajectory data collected to plot 3D trajectories.')
    # except Exception as e:
    #     print(f'Failed to generate 3D trajectory plot: {e}')
    
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


def generate_3d_trajectory_plot(all_base_positions, all_ee_positions, all_targets, save_dir, episode=1):
    """生成3D轨迹图：机器人底座和末端执行器位置的变化（不渲染机器人）"""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # color cycle
    colors = plt.cm.tab10.colors

    # plot each episode
    for i, (base_traj, ee_traj, target_p) in enumerate(zip(all_base_positions, all_ee_positions, all_targets)):
        if base_traj:
            base_arr = np.array(base_traj)
            ax.plot(base_arr[:, 0], base_arr[:, 1], base_arr[:, 2], linestyle='-', marker='o',
                    color=colors[i % len(colors)], label=f'Base p{i+1}')
            # mark start and end
            ax.scatter(base_arr[0, 0], base_arr[0, 1], base_arr[0, 2], color=colors[i % len(colors)], marker='D', s=40)
            ax.scatter(base_arr[-1, 0], base_arr[-1, 1], base_arr[-1, 2], color=colors[i % len(colors)], marker='X', s=60)

        if ee_traj:
            ee_arr = np.array(ee_traj)
            ax.plot(ee_arr[:, 0], ee_arr[:, 1], ee_arr[:, 2], linestyle='--', marker='.',
                    color=colors[(i+3) % len(colors)], label=f'EE p{i+1}')
            ax.scatter(ee_arr[0, 0], ee_arr[0, 1], ee_arr[0, 2], color=colors[(i+3) % len(colors)], marker='s', s=30)
            ax.scatter(ee_arr[-1, 0], ee_arr[-1, 1], ee_arr[-1, 2], color=colors[(i+3) % len(colors)], marker='*', s=70)

        if target_p is not None:
            # target_p may be a single 3-vector (x,y,z) or a sequence of points
            try:
                target_arr = np.array(target_p)
                if target_arr.ndim == 1 and target_arr.size >= 3:
                    # single target point
                    tx, ty, tz = target_arr[:3]
                    ax.scatter([tx], [ty], [tz], color='magenta', marker='X', s=140, edgecolors='k', label=f'Target p{i+1}({tx:.1f},{ty:.1f},{tz:.1f})')
                elif target_arr.ndim == 2 and target_arr.shape[1] >= 3:
                    # sequence of target points (plot as dashed magenta line + markers)
                    ax.plot(target_arr[:, 0], target_arr[:, 1], target_arr[:, 2], linestyle=':', color='magenta', linewidth=1.0)
                    ax.scatter(target_arr[:, 0], target_arr[:, 1], target_arr[:, 2], color='magenta', marker='X', s=60, edgecolors='k')
                    # mark first/last explicitly
                    ax.scatter(target_arr[0, 0], target_arr[0, 1], target_arr[0, 2], color='magenta', marker='D', s=50)
                    ax.scatter(target_arr[-1, 0], target_arr[-1, 1], target_arr[-1, 2], color='magenta', marker='*', s=80)
                else:
                    # fallback: try to interpret as flat sequence
                    flat = target_arr.flatten()
                    if flat.size >= 3:
                        tx, ty, tz = flat[:3]
                        ax.scatter([tx], [ty], [tz], color='magenta', marker='X', s=140, edgecolors='k', label=f'Target Ep{i+1}')
            except Exception:
                # silently ignore malformed target data
                pass

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Trajectories: Base (solid) and End-Effector (dashed)')
    # place legend inside the axes in the upper-left to avoid external overlap
    ax.legend(loc='upper left')
    ax.set_aspect('equal')
    plt.tight_layout()

    out_path = os.path.join(save_dir, f'trajectories_3d_episode_{episode}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved 3D trajectory plot: {out_path}")


def generate_velocity_plot(world_vels, chassis_vels, save_dir, episode=1, joint_angles=None, ee_heights=None):
    """Generate a 2-panel plot (side-by-side) comparing world-frame and chassis-frame
    longitudinal (vx) and lateral (vy) linear velocities over time for one episode.

    Args:
        world_vels: list of (vx_world, vy_world) per step
        chassis_vels: list of (vx_chassis, vy_chassis) per step
        save_dir: directory to save the PNG
        episode: episode number (used for filename)
    """
    # convert to arrays (N,2)
    w = np.array(world_vels, dtype=np.float32) if len(world_vels) > 0 else np.zeros((0, 2), dtype=np.float32)
    c = np.array(chassis_vels, dtype=np.float32) if len(chassis_vels) > 0 else np.zeros((0, 2), dtype=np.float32)

    steps_w = np.arange(w.shape[0]) if w.shape[0] > 0 else np.arange(0)
    steps_c = np.arange(c.shape[0]) if c.shape[0] > 0 else np.arange(0)

    # create a figure with 3 rows: two side-by-side velocity plots on top row
    # and a full-width joint-angles plot on the bottom row
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.8])

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

    # Lower row: left = joint angles, right = ee height
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

    plt.suptitle(f'Episode {episode}: Velocities and Joint Angles')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_path = os.path.join(save_dir, f'episode_{episode}_velocities.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved velocity plot: {out_path}")


if __name__ == "__main__":
    # 模型路径 - 根据实际保存位置调整
    # 去掉角度误差
    model_path = "linux_env_dev/models/logs_20251031_112452/ppo_mobile_mm_final.zip"
    # 带高程误差惩罚
    model_path = "linux_env_dev/models/logs_20251031_113615/ppo_mobile_mm_final.zip"
    # 增大关节力矩
    model_path = "linux_env_dev/models/logs_20251103_175011/ppo_mobile_mm_final.zip"
    # 增加terminate的机械臂静止条件
    model_path = "linux_env_dev/models/logs_20251104_110708/ppo_mobile_mm_final.zip"
    # 优化学习率
    model_path = "linux_env_dev/models/logs_20251104_121650/ppo_mobile_mm_final.zip"
    # delta = 0.05
    model_path = "linux_env_dev/models/logs_20251105_000224/ppo_mobile_mm_final.zip"
    # Transformer特征提取器
    model_path = "linux_env_dev/models/logs_20251105_110047/ppo_mobile_mm_final.zip"
    # 轨迹点跟踪
    model_path = "linux_env_dev/models/logs_20251105_184132/checkpoints/ppo_mobile_mm_1600000_steps.zip"
    render = False
    
    if os.path.exists(model_path):
        print("Starting model testing...")
        # target_x, target_y, target_z = 5.5, -0.5, 0.7
        # target_x, target_y, target_z = 4.7, -1.7, 0.7
        # num_episode = 1
        # distances, rewards, success_count = test_trained_model(model_path, num_episodes=num_episode, render=render, max_steps=500,
        #                                                         # low=(4.0, -2.0, 0.5),
        #                                                         # high=(10.0, 2.0, 1.5))
        #                                                         low=(target_x, target_y, target_z),
        #                                                         high=(target_x, target_y, target_z))
        num_episode = 5
        distances, rewards, success_count = test_trained_model(model_path, num_episodes=num_episode, render=render, max_steps=500)
        print(f"\nTesting completed! Success rate: {success_count}/{num_episode} ({success_count/num_episode*100:.1f}%)")
    else:
        print(f"Model file not found: {model_path}")
        print("Please train the model first or check the model path.")
