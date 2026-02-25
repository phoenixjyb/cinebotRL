#!/usr/bin/env python3
"""测试新URDF (recomoProto1-190) 能否正确加载"""

import sys
sys.path.insert(0, 'linux_env_dev')

from pybullet_envs.mobile_mm_traj import MobileMMTrajEnv
from pybullet_envs.target_generator import JSONNearestTargetGenerator

print("="*60)
print("测试加载新URDF: recomoProto1-190.urdf")
print("="*60)

try:
    env = MobileMMTrajEnv(
        render=False,
        target_generator=JSONNearestTargetGenerator(
            json_txt="linux_env_dev/new_json_50/train.txt",
            mode="random"
        )
    )
    print(f"✅ URDF加载成功！\n")
    
    print(f"关节总数: {env.n_j}")
    print(f"观测空间维度: {env.observation_space.shape[0]}")
    print(f"动作空间维度: {env.action_space.shape[0]}")
    
    print(f"\n可控关节列表:")
    print("-" * 60)
    for joint_name, joint_id in sorted(env.joint_name2ids.items(), key=lambda x: x[1]):
        limits = env.joint_limits.get(joint_name, "N/A")
        if limits != "N/A":
            print(f"  [{joint_id}] {joint_name:30s} → [{limits[0]:7.2f}, {limits[1]:7.2f}]")
        else:
            print(f"  [{joint_id}] {joint_name}")
    
    # 测试reset
    print(f"\n" + "="*60)
    print("测试环境reset...")
    reset_result = env.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    print(f"✅ Reset成功！观测维度: {obs.shape}")
    
    # 测试step
    print(f"\n测试环境step...")
    action = env.action_space.sample()
    step_result = env.step(action)
    obs, reward = step_result[0], step_result[1]
    done = step_result[2] if len(step_result) == 4 else (step_result[2] or step_result[3])
    info = step_result[-1]
    print(f"✅ Step成功！")
    print(f"  观测维度: {obs.shape}")
    print(f"  奖励: {reward:.4f}")
    print(f"  Done: {done}")
    
    env.close()
    print(f"\n" + "="*60)
    print(f"✅ 所有测试通过！新URDF可以正常使用。")
    print(f"="*60)
    
except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
