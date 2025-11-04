from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import multiprocessing as mp
from pybullet_envs.mobile_mm import MobileMMBulletEnv
from pybullet_envs.target_generator import FixedTarget, RandomTarget, RandomTargetForEpisode, CurriculumTarget
import torch
import time

def compare_vec_envs(make_env, n_envs=4, test_steps=1000):
    """对比DummyVecEnv和SubprocVecEnv的性能"""
    
    # 测试DummyVecEnv
    print("=== 测试 DummyVecEnv ===")
    dummy_env = DummyVecEnv([make_env for _ in range(n_envs)])
    
    start_time = time.time()
    obs = dummy_env.reset()
    
    for step in range(test_steps):
        actions = [dummy_env.action_space.sample() for _ in range(n_envs)]
        obs, rewards, dones, infos = dummy_env.step(actions)
        
        if any(dones):
            reset_indices = [i for i, done in enumerate(dones) if done]
            for idx in reset_indices:
                # import pdb; pdb.set_trace()
                obs = dummy_env.envs[idx].reset()
    
    dummy_time = time.time() - start_time
    dummy_steps_per_second = (test_steps * n_envs) / dummy_time
    dummy_env.close()
    
    print(f"DummyVecEnv: {dummy_steps_per_second:.1f} 步/秒")
    
    # 测试SubprocVecEnv
    print("=== 测试 SubprocVecEnv ===")
    subproc_env = SubprocVecEnv([make_env for _ in range(n_envs)])
    
    start_time = time.time()
    obs = subproc_env.reset()
    
    for step in range(test_steps):
        actions = [subproc_env.action_space.sample() for _ in range(n_envs)]
        obs, rewards, dones, infos = subproc_env.step(actions)
        
    subproc_time = time.time() - start_time
    subproc_steps_per_second = (test_steps * n_envs) / subproc_time
    subproc_env.close()
    
    print(f"SubprocVecEnv: {subproc_steps_per_second:.1f} 步/秒")
    
    # 性能比较
    improvement = (subproc_steps_per_second - dummy_steps_per_second) / dummy_steps_per_second * 100
    print(f"\n性能比较:")
    print(f"SubprocVecEnv 比 DummyVecEnv {'快' if improvement > 0 else '慢'} {abs(improvement):.1f}%")
    
    if improvement > 20:
        print("推荐使用: SubprocVecEnv")
    elif improvement < -10:
        print("推荐使用: DummyVecEnv") 
    else:
        print("两者性能相近，推荐使用 DummyVecEnv（开销更小）")
    
    return dummy_steps_per_second, subproc_steps_per_second

if __name__ == '__main__':
    # 运行对比测试
    env_fn = lambda: MobileMMBulletEnv(render=False,
                                       target_generator=RandomTargetForEpisode(
                                           low=(4.0, -2.0, 0.5),
                                           high=(10.0, 2.0, 1.5)
                                       ))
    dummy_sps, subproc_sps = compare_vec_envs(env_fn, n_envs=16)