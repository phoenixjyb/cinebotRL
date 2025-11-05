from linux_env_dev.pybullet_envs.target_generator import JSONNearestTargetGenerator
from linux_env_dev.pybullet_envs.mobile_mm_traj import MobileMMTrajEnv
import os

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sample = os.path.join(repo_root, 'trajectoryToLearn', 'world_json', 'scene_1', 'traj_1.json')
print('Using sample:', sample)

gen = JSONNearestTargetGenerator(sample, lookahead_steps=0, lookahead_dt=1.0)
env = MobileMMTrajEnv(target_generator=gen, render=False)
obs, info = env.reset()
import pybullet as p
base_pos, base_orn = p.getBasePositionAndOrientation(env.robot)
print('Robot base after reset:', base_pos, base_orn)
env.close()
