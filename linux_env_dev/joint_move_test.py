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
    idx_abs = env._link_index_by_name(env.robot, 'abstract_chassis_link')
    abs_pos = None
    abs_yaw = None
    abs_lin_vel = None
    abs_ang_vel = None
    if idx_abs is not None:
        # request velocities too
        st = p.getLinkState(env.robot, idx_abs, computeForwardKinematics=True, computeLinkVelocity=True)
        abs_pos = st[0]
        abs_yaw = p.getEulerFromQuaternion(st[1])[2]
        # getLinkState returns linear and angular velocities at indices 6 and 7 when computeLinkVelocity=True
        if len(st) > 6:
            abs_lin_vel = st[6]
            abs_ang_vel = st[7]

    if abs_pos is not None:
        pos_str = tuple(round(x,4) for x in abs_pos)
        yaw_str = round(abs_yaw,4)
        if abs_lin_vel is not None and abs_ang_vel is not None:
            lin_str = tuple(round(x,6) for x in abs_lin_vel)
            ang_str = tuple(round(x,6) for x in abs_ang_vel)
            print("abstract_chassis_link pos:", pos_str, "yaw:", yaw_str,
                  "lin_vel:", lin_str, "ang_vel:", ang_str)
        else:
            print("abstract_chassis_link pos:", pos_str, "yaw:", yaw_str)
    else:
        print("abstract_chassis_link not found")


if __name__ == '__main__':
    env = MobileMMBulletEnv(render=False, max_steps=100)
    obs, _ = env.reset()

    # find joint indices
    idx_x = env._joint_index_by_name(env.robot, 'joint_x')
    idx_theta = env._joint_index_by_name(env.robot, 'joint_theta')
    print('Joint indices: idx_x=', idx_x, 'idx_theta=', idx_theta)

    # helper to move joints and step
    def move_and_step(jidx, pos, steps=10, positionGain=0.1, velocityGain=10.0, max_force=100):
        # read current joint position
        try:
            cur = p.getJointState(env.robot, jidx)[0]
        except Exception:
            cur = 0.0
        # estimate required velocity to reach `pos` in `steps` physics steps
        # PyBullet default internal timestep is usually 1/240, but we'll query it if possible
        dt = 1.0 / 240.0
        try:
            info = p.getPhysicsEngineParameters()
            if 'fixedTimeStep' in info and info['fixedTimeStep'] > 0:
                dt = float(info['fixedTimeStep'])
        except Exception:
            pass

        # velocity required per second to cover (pos - cur) in (steps * dt)
        total_time = max(steps * dt, 1e-6)
        # req_vel = float((pos - cur) / total_time)
        req_vel = 0.01

        p.setJointMotorControl2(env.robot, jidx, p.POSITION_CONTROL,
                                targetPosition=pos,
                                targetVelocity=req_vel,
                                positionGain=positionGain,
                                velocityGain=velocityGain,
                                force=max_force)
        for _ in range(steps):
            p.stepSimulation()
            # time.sleep(0.001)

    # reset x
    p.resetJointState(env.robot, idx_x, targetValue=0.0)
    for _ in range(50): p.stepSimulation()

    # Case B: theta = 0.5 rad, move x by +0.1
    if idx_theta is not None:
        p.resetJointState(env.robot, idx_theta, targetValue=0.0)
    p.resetJointState(env.robot, idx_x, targetValue=0.0)
    move_and_step(idx_x, 0.1)
    print_positions(env, 'x->0.1')
    move_and_step(idx_theta, 0.1)
    print_positions(env, 'theta=0.1')
    move_and_step(idx_x, 0.1)
    print_positions(env, 'x->0.1')
    move_and_step(idx_x, 0.1)
    print_positions(env, 'x->0.1')
    move_and_step(idx_theta, -0.1)
    print_positions(env, 'theta=-0.1')
    move_and_step(idx_x, 0.1)
    print_positions(env, 'x->0.1')
    move_and_step(idx_x, 0.1)
    print_positions(env, 'x->0.1')
    move_and_step(idx_x, -0.2)
    print_positions(env, 'x->-0.2')
    move_and_step(idx_x, -0.2)
    print_positions(env, 'x->-0.2')

    env.close()
