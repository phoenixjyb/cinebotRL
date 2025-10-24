import os
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from pybullet_envs.mobile_mm import MobileMMBulletEnv


def test_trained_model(model_path, num_episodes=1, render=False):
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
    
    # 加载模型
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)
    
    # 创建测试环境（render 参数可控）
    env_fn = lambda: MobileMMBulletEnv(render=render, max_steps=1000)
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
                dist_val = infos[0].get('distance', None)
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
            except Exception:
                pass
            try:
                cur_base = _get_base_pos_from_env(env)
                if cur_base is not None:
                    episode_base_positions.append(tuple(float(x) for x in cur_base[:3]))
            except Exception:
                pass

            terminated = done_flag
            
            # 每50步保存一张图片
            if step_count % 50 == 0:
                img_path = os.path.join(save_dir, f"episode_{episode+1}_step_{step_count}.png")
                try:
                    # pass collected EE trajectory for overlay
                    env.save_robot_image(img_path, width=800, height=600, traj=episode_ee_positions, draw_origin=True)
                    print(f"Saved image: {img_path}")
                except Exception as e:
                    print(f"Failed to save image: {e}")
        
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
        final_img_path = os.path.join(save_dir, f"episode_{episode+1}_final.png")
        try:
            env.save_robot_image(final_img_path, width=800, height=600, traj=episode_ee_positions, draw_origin=True, target=target)
            print(f"Saved final image: {final_img_path}")
        except Exception as e:
            print(f"Failed to save final image: {e}")

        # accumulate base and ee positions for cross-episode plotting
        if 'all_base_positions' not in locals():
            all_base_positions = []
        if 'all_ee_positions' not in locals():
            all_ee_positions = []
        # store collected trajectories and target per episode (trim possible trailing incomplete entries)
        all_base_positions.append(episode_base_positions[:-2])
        all_ee_positions.append(episode_ee_positions[:-2])
        if 'all_targets' not in locals():
            all_targets = []
        all_targets.append(target)
    
    vec_env.close()
    
    # 生成性能图表
    generate_performance_plots(all_distances, all_rewards, save_dir, success_count, num_episodes)
    
    # 生成测试报告
    generate_test_report(all_distances, all_rewards, success_count, num_episodes, save_dir)
    # 生成3D轨迹图（底座与末端执行器）
    try:
        if 'all_base_positions' in locals() and 'all_ee_positions' in locals():
            generate_3d_trajectory_plot(all_base_positions, all_ee_positions, all_targets, save_dir)
        else:
            print('No trajectory data collected to plot 3D trajectories.')
    except Exception as e:
        print(f'Failed to generate 3D trajectory plot: {e}')
    
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


def generate_3d_trajectory_plot(all_base_positions, all_ee_positions, all_targets, save_dir):
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
                    ax.scatter([tx], [ty], [tz], color='magenta', marker='X', s=140, edgecolors='k', label=f'Target Ep{i+1}')
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
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.set_aspect('equal')
    plt.tight_layout()

    out_path = os.path.join(save_dir, 'trajectories_3d.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved 3D trajectory plot: {out_path}")


if __name__ == "__main__":
    # 模型路径 - 根据实际保存位置调整
    model_path = "linux_env_dev/models/checkpoints_20251024_073650/ppo_mobile_mm_1000000_steps.zip"
    render = False
    
    if os.path.exists(model_path):
        print("Starting model testing...")
        distances, rewards, success_count = test_trained_model(model_path, render=render)
        print(f"\nTesting completed! Success rate: {success_count}/3 ({success_count/3*100:.1f}%)")
    else:
        print(f"Model file not found: {model_path}")
        print("Please train the model first or check the model path.")
