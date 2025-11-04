"""
Quick test script to verify world-frame Y change when moving joint_x under different joint_theta values.
Saves outputs to stdout.
"""
import time
import numpy as np
import pybullet as p
from pybullet_envs.mobile_mm import MobileMMBulletEnv


def print_positions(env, label):
    print(f"current move: {label}")
    abstract_idx = env._link_index_by_name(env.robot, 'abstract_chassis_link')
    # link state gives position/orientation of the named link
    # import pdb; pdb.set_trace()
    # tmp_pos, _ = p.getBasePositionAndOrientation(self.robot)
    st_abs = p.getLinkState(env.robot, abstract_idx, computeForwardKinematics=True, computeLinkVelocity=True)
    abs_pos = np.array(st_abs[0], dtype=np.float32)
    euler = p.getEulerFromQuaternion(st_abs[1])   # (roll, pitch, yaw)
    abs_yaw = float(euler[2])

    abs_lin_vel = np.array(st_abs[6], dtype=np.float32)
    abs_ang_vel = np.array(st_abs[7], dtype=np.float32)

    pos_str = tuple(round(float(x),2) for x in abs_pos)
    yaw_str = round(abs_yaw,4)
    if abs_lin_vel is not None and abs_ang_vel is not None:
        # import pdb; pdb.set_trace()
        lin_str = tuple(round(float(x),6) for x in abs_lin_vel)
        ang_str = tuple(round(float(x),6) for x in abs_ang_vel)
        ang_str = tuple(round(float(x),6) for x in abs_ang_vel)
        print("abstract_chassis_link pos:", pos_str, "yaw:", yaw_str,
                "lin_vel:", lin_str, "ang_vel:", ang_str)
    print("\n")

def print_all_joints_state(jids,):
    for jidx in jids:
        st = p.getJointState(env.robot, jidx)
        print(f"  joint {jidx}: pos={st[0]:.4f}, vel={st[1]:.4f}")

# helper to move joints and step
def move_and_step(jidx, pos, steps=100, positionGain=1.0, velocityGain=0.1, max_force=1000):
    # read current joint position
    st_before_a = p.getJointState(env.robot, jidx)[0]

    # estimate required velocity to reach `pos` in `steps` physics steps
    # PyBullet default internal timestep is usually 1/240, but we'll query it if possible
    dt = 1.0 / 240.0
    info = p.getPhysicsEngineParameters()
    if 'fixedTimeStep' in info and info['fixedTimeStep'] > 0:
        dt = float(info['fixedTimeStep'])
        pass

    # velocity required per second to cover (pos - cur) in (steps * dt)
    total_time = max(steps * dt, 1e-6)
    # req_vel = float((pos - cur) / total_time)
    req_vel = 1

    p.setJointMotorControl2(env.robot, jidx, p.POSITION_CONTROL,
                            targetPosition=pos,
                            targetVelocity=req_vel,
                            positionGain=positionGain,
                            velocityGain=velocityGain,
                            force=max_force)

    for _ in range(steps):
        p.stepSimulation()
    st_after_a = p.getJointState(env.robot, jidx)[0]
    print(f"  move_and_step: jidx={jidx} from {st_before_a:.4f} to {st_after_a:.4f} (target={pos})")

# helper to move joints and step
def move_and_step_array(jidx, pos, steps=10, positionGain=1.0, velocityGain=0.1, max_force=100):
    # read current joint position
    st_before_a0 = p.getJointState(env.robot, jidx[0])[0]
    st_before_a1 = p.getJointState(env.robot, jidx[1])[0]
    
    p.setJointMotorControlArray(env.robot,
                        jointIndices=jidx,
                        controlMode=p.POSITION_CONTROL,
                        targetPositions=pos,
                        forces=[max_force, max_force])
    for _ in range(steps):
        p.stepSimulation()
    st_after_a0 = p.getJointState(env.robot, jidx[0])[0]
    st_after_a1 = p.getJointState(env.robot, jidx[1])[0]
    print(f"  move_and_step: jidx={jidx} from {st_before_a0:.4f}, {st_before_a1:.4f} "
          f"to {st_after_a0:.4f}, {st_after_a1:.4f} (target={pos})")

if __name__ == '__main__':
    env = MobileMMBulletEnv(render=False, max_steps=100)
    obs, _ = env.reset()

    # find joint indices
    idx_x = env._joint_index_by_name(env.robot, 'joint_x')
    idx_y = env._joint_index_by_name(env.robot, 'joint_y')
    idx_theta = env._joint_index_by_name(env.robot, 'joint_theta')
    idx_arm1 = env._joint_index_by_name(env.robot, 'left_arm_joint1')
    idx_arm2 = env._joint_index_by_name(env.robot, 'left_arm_joint2')
    idx_arm3 = env._joint_index_by_name(env.robot, 'left_arm_joint3')
    idx_arm4 = env._joint_index_by_name(env.robot, 'left_arm_joint4')
    idx_arm5 = env._joint_index_by_name(env.robot, 'left_arm_joint5')
    idx_arm6 = env._joint_index_by_name(env.robot, 'left_arm_joint6')
    
    print('Joint indices: idx_x=', idx_x, 'idx_y=', idx_y, 'idx_theta=', idx_theta)
    print('Arm joint indices:', idx_arm1, idx_arm2, idx_arm3, idx_arm4, idx_arm5, idx_arm6)
    
    # print_positions(env, 'initial')
    # move_and_step_array([idx_x, idx_y], [-0.1, -0.1], steps=200)
    # print_positions(env, 'x y ->1')
    # move_and_step_array([idx_x, idx_y], [0.1, 0.1], steps=200)
    # print_positions(env, 'x y ->-1')
    
    
    # env.save_robot_image('linux_env_dev/initial_0.0.png')
    # move_and_step(idx_arm1, 1.0)
    # print_positions(env, 'arm1->1.0')
    # env.save_robot_image('linux_env_dev/arm1_1.0.png')
    print_all_joints_state([idx_arm1, idx_arm2, idx_arm3, idx_arm4, idx_arm5, idx_arm6])
    move_and_step(idx_arm2, 1.0)
    print_positions(env, 'arm2->1.0')
    env.save_robot_image('linux_env_dev/arm2_1.0.png')
    print_all_joints_state([idx_arm1, idx_arm2, idx_arm3, idx_arm4, idx_arm5, idx_arm6])
    # move_and_step(idx_theta, 1.0)
    # print_positions(env, 'theta=1.0')
    # move_and_step(idx_x, -10)
    # print_positions(env, 'x->-10')
    # move_and_step(idx_x, 10)
    # print_positions(env, 'x->10')
    # move_and_step(idx_theta, -0.1)
    # print_positions(env, 'theta=-0.1')
    # move_and_step(idx_x, 1000)
    # move_and_step(idx_x, 1000)
    # move_and_step(idx_x, 1000)


    env.close()
