import os
import numpy as np
from typing import Optional
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
try:
    from PIL import Image
except Exception:
    Image = None

from .reward_helpers import compute_reward, DEBUG
from .target_generator import TargetGenerator, FixedTarget, JSONNearestTargetGenerator

class MobileMMTrajEnv(gym.Env):
    """Minimal PyBullet-based mobile manipulator tracking env.

    Loads URDF from project: assets_own/mobile_manipulator_PPR_base_corrected.urdf
    Observation: [q, qdot, ee_pos, target_pos]
    Action: joint position deltas scaled by max_delta
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, urdf_path=None, frame_skip=24, max_steps=500, render=False,
                 save_image_on_reset=False, save_image_path=None, image_width=800, image_height=600,
                 reward_distance_weight: float = 1.0, reward_yaw_weight: float = 1.0,
                 target_generator: Optional[TargetGenerator] = None):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        # self.urdf_path = urdf_path or os.path.join(self.project_root, "assets_own", "mobile_manipulator_PPR_base_corrected.urdf")
        # self.urdf_path = urdf_path or os.path.join(self.project_root, "assets_own", "mobile_manipulator_PPR_theta_before_x.urdf")
        self.urdf_path = urdf_path or os.path.join(self.project_root, "assets_own", "mobile_manipulator_little_xy_link.urdf")
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.render = render

        # image saving options
        self.save_image_on_reset = bool(save_image_on_reset)
        self.save_image_path = save_image_path
        self.image_width = int(image_width)
        self.image_height = int(image_height)

        # Connect
        if self.render:
            self.cid = p.connect(p.GUI)
        else:
            self.cid = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # placeholders
        self.robot = None
        self.step_count = 0

        # build a temporary scene to identify joints
        self._load_robot_once()

        # action / obs spaces
        self.n_j = len(self.joint_ids) # 关节个数
        self.lookahead_num = 5
        # number of historical frames to include for chassis joints (joint_x, joint_y, joint_theta)
        self.hist_num = 3

        # 目前仅控制底盘的2自由度——x平移和theta旋转
        self.action_space = spaces.Box(low=-1, high=1, shape=(2+6,), dtype=np.float32) # 2个底盘自由度 + 6个机械臂自由度
        # obs layout: joint pos (n_j), joint vel (n_j), chassis_history (hist_num*3), base_pos(3), ee_pos(3),
        # target + lookahead (3*(1+lookahead_num)), remain_ratio(1)
        obs_dim = self.n_j * 2 + (self.hist_num * 3) + 3 + 3 + 3*(1 + self.lookahead_num) + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self._target_generator = target_generator
        self._target = self._target_generator.reset()
        self.finish_step1_step = None
        self.remain_traj_ratio = 1.0
        self._traj_id = 0
        # chassis joint history buffer: shape (hist_num, 3) columns = [joint_x, joint_y, joint_theta]
        self._chassis_hist = np.zeros((self.hist_num, 3), dtype=np.float32)
        self.last_chassis_vel = None

    def _load_robot_once(self):
        # load plane and robot, then remove (we re-create on reset)
        plane = p.loadURDF("plane.urdf")
        try:
            robot = p.loadURDF(self.urdf_path, useFixedBase=False)
        except Exception as e:
            raise FileNotFoundError(f"URDF not found or failed to load: {self.urdf_path} -> {e}")
        # collect non-fixed joint ids
        self.joint_ids = []
        self.joint_limits = {}
        self.joint_name2ids = {}
        self.joint_effort_limit = {}
        for i in range(p.getNumJoints(robot)):
            info = p.getJointInfo(robot, i)
            joint_name = info[1].decode()
            joint_idx = self._joint_index_by_name(robot, joint_name)
            jtype = info[2]
            if jtype != p.JOINT_FIXED:
                # p.JOINT_REVOLUTE（旋转）
                # p.JOINT_PRISMATIC（平移）
                self.joint_ids.append(i)
                self.joint_limits[joint_name] = (info[8], info[9])
                self.joint_name2ids[joint_name] = joint_idx
                self.joint_effort_limit[joint_name] = info[10]

        # clean up
        p.removeBody(robot)
        p.removeBody(plane)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self._target = self._target_generator.reset()
        self.finish_step1_step = None
        self.remain_traj_ratio = 1.0
        self.step_count = 0
        self._traj_id = 0
        self.last_chassis_vel = None
        self._chassis_hist = np.zeros((self.hist_num, 3), dtype=np.float32)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        self.robot = p.loadURDF(self.urdf_path, useFixedBase=False)

        start_pos = self._target_generator.traj.get_position(0, 0.0)

        # reset joints to random small values
        for idx in self.joint_ids:
            p.resetJointState(self.robot, idx, targetValue=0.0, targetVelocity=0.0)

        # p.resetJointState(self.robot, self.joint_name2ids['joint_x'],
        #                   targetValue=start_pos[0] + np.random.uniform(-0.5, 0.5), targetVelocity=0.0)
        # p.resetJointState(self.robot, self.joint_name2ids['joint_y'],
        #                   targetValue=start_pos[1] + np.random.uniform(-0.5, 0.5), targetVelocity=0.0)
        # p.resetJointState(self.robot, self.joint_name2ids['joint_theta'],
        #                   targetValue=np.random.uniform(-np.pi, np.pi), targetVelocity=0.0)
        
        # 初始点做简化，方向尽量一致，ee尽量靠近起点
        # p.resetJointState(self.robot, self.joint_name2ids['joint_x'],
        #                   targetValue=start_pos[0] + np.random.uniform(-1.0, -0.7), targetVelocity=0.0)
        # p.resetJointState(self.robot, self.joint_name2ids['joint_y'],
        #                   targetValue=start_pos[1] + np.random.uniform(-0.1, 0.1), targetVelocity=0.0)
        p.resetJointState(self.robot, self.joint_name2ids['joint_x'],
                          targetValue=start_pos[0] - 0.6, targetVelocity=0.0)
        p.resetJointState(self.robot, self.joint_name2ids['joint_y'],
                          targetValue=start_pos[1], targetVelocity=0.0)
        
        # 各机械臂设置为初始位置
        p.resetJointState(self.robot, self.joint_name2ids['left_arm_joint1'],
                          targetValue=-1.61, targetVelocity=0.0)
        p.resetJointState(self.robot, self.joint_name2ids['left_arm_joint2'],
                          targetValue=0.00, targetVelocity=0.0)
        p.resetJointState(self.robot, self.joint_name2ids['left_arm_joint3'],
                          targetValue=-0.32, targetVelocity=0.0)
        p.resetJointState(self.robot, self.joint_name2ids['left_arm_joint4'],
                          targetValue=-1.56, targetVelocity=0.0)
        p.resetJointState(self.robot, self.joint_name2ids['left_arm_joint5'],
                          targetValue=-1.13, targetVelocity=0.0)
        p.resetJointState(self.robot, self.joint_name2ids['left_arm_joint6'],
                          targetValue=0.0, targetVelocity=0.0)

        # optionally save image after robot spawned
        if self.save_image_on_reset:
            out_path = self.save_image_path or os.path.join(self.project_root, "linux_env_dev", "robot_pose.png")
            try:
                self.save_robot_image(out_path, width=self.image_width, height=self.image_height)
                print(f"[MobileMMBulletEnv] Saved robot pose image to: {out_path}")
            except Exception as e:
                print(f"[MobileMMBulletEnv] Warning: failed to save robot image: {e}")

        if DEBUG:
            print(f"Robot initial position: ")
            for jid in self.joint_ids:
                info = p.getJointInfo(self.robot, jid)
                joint_name = info[1].decode()
                link_name = info[12].decode()        # child link name
                st = p.getLinkState(self.robot, jid, computeForwardKinematics=True)
                pos_world = st[0] # position
                orn_world = st[1] # orientation
                rounded_pos = tuple(round(x, 1) for x in pos_world)
                # import pdb; pdb.set_trace()
                print(f"joint {jid} name={joint_name}: pos={rounded_pos}")
        return self._get_obs(), {}

    def save_robot_image(self, filepath: str, width: int = 800, height: int = 600,
                         distance: float = 1.2, yaw: float = 45.0, pitch: float = 30.0,
                         auto_fit: bool = True, draw_axes: bool = True, traj: list = None, draw_origin: bool = True,
                         target: list = None):
        """Render current robot pose and save RGB image to filepath.

        Parameters:
            filepath: output PNG path
            width, height: image size
            distance: camera distance from target (meters)
            yaw, pitch: camera angles in degrees (spherical)
        """
        if self.robot is None:
            raise RuntimeError("Robot not loaded; call reset() first.")

        # determine target (center) and distance automatically from robot bounding box
        target = np.array([0.0, 0.0, 0.0], dtype=float)
        try:
            if auto_fit:
                # include base (-1) and all links
                aabb_min = np.array([np.inf, np.inf, np.inf], dtype=float)
                aabb_max = np.array([-np.inf, -np.inf, -np.inf], dtype=float)
                try:
                    base_aabb = p.getAABB(self.robot, -1)
                    if base_aabb and base_aabb[0] is not None:
                        aabb_min = np.minimum(aabb_min, np.array(base_aabb[0], dtype=float))
                        aabb_max = np.maximum(aabb_max, np.array(base_aabb[1], dtype=float))
                except Exception:
                    pass
                for li in range(p.getNumJoints(self.robot)):
                    try:
                        aabb = p.getAABB(self.robot, li)
                        if aabb and aabb[0] is not None:
                            aabb_min = np.minimum(aabb_min, np.array(aabb[0], dtype=float))
                            aabb_max = np.maximum(aabb_max, np.array(aabb[1], dtype=float))
                    except Exception:
                        continue

                if np.isfinite(aabb_min).all() and np.isfinite(aabb_max).all():
                    target = (aabb_min + aabb_max) / 2.0
                    size = aabb_max - aabb_min
                    diag = float(np.linalg.norm(size))
                    # choose a distance that fits the whole robot
                    distance = max(distance, diag * 1.2)
                else:
                    base_pos, _ = p.getBasePositionAndOrientation(self.robot)
                    target = np.array(base_pos, dtype=float)
            else:
                base_pos, _ = p.getBasePositionAndOrientation(self.robot)
                target = np.array(base_pos, dtype=float)
        except Exception:
            target = np.array([0.0, 0.0, 0.0], dtype=float)

        # compute camera eye in simple spherical coords around target
        yaw_rad = float(yaw) * np.pi / 180.0
        pitch_rad = float(pitch) * np.pi / 180.0
        ex = float(distance * np.cos(pitch_rad) * np.cos(yaw_rad)) + target[0]
        ey = float(distance * np.cos(pitch_rad) * np.sin(yaw_rad)) + target[1]
        ez = float(distance * np.sin(pitch_rad)) + target[2] + 0.2

        view_mat = p.computeViewMatrix(cameraEyePosition=[ex, ey, ez],
                                       cameraTargetPosition=target.tolist(),
                                       cameraUpVector=[0, 0, 1])
        proj_mat = p.computeProjectionMatrixFOV(fov=60.0,
                                                aspect=float(width) / float(height),
                                                nearVal=0.01, farVal=100.0)

        # choose renderer: if we're in headless mode (DIRECT) prefer tiny renderer
        if not self.render or p.getConnectionInfo().get('connectionMethod', None) == p.DIRECT:
            renderer = p.ER_TINY_RENDERER
        else:
            # try hardware renderer then fall back to tiny renderer
            try:
                renderer = p.ER_BULLET_HARDWARE_OPENGL
            except Exception:
                try:
                    renderer = p.ER_TINY_RENDERER
                except Exception:
                    renderer = 0

        # optionally draw a small coordinate frame at target for orientation reference
        debug_ids = []
        if draw_axes:
            try:
                L = max(0.05, float(distance) * 0.2)
                origin = target.tolist()
                # X (red)
                debug_ids.append(p.addUserDebugLine(origin, [origin[0] + L, origin[1], origin[2]], [1, 0, 0], 2, 0))
                # Y (green)
                debug_ids.append(p.addUserDebugLine(origin, [origin[0], origin[1] + L, origin[2]], [0, 1, 0], 2, 0))
                # Z (blue)
                debug_ids.append(p.addUserDebugLine(origin, [origin[0], origin[1], origin[2] + L], [0, 0, 1], 2, 0))
            except Exception:
                debug_ids = []

        # draw world origin marker if requested
        if draw_origin:
            try:
                o = [0.0, 0.0, 0.0]
                L0 = max(0.02, float(distance) * 0.05)
                debug_ids.append(p.addUserDebugLine([o[0] - L0, o[1], o[2]], [o[0] + L0, o[1], o[2]], [1, 1, 0], 2, 0))
                debug_ids.append(p.addUserDebugLine([o[0], o[1] - L0, o[2]], [o[0], o[1] + L0, o[2]], [1, 1, 0], 2, 0))
                debug_ids.append(p.addUserDebugLine([o[0], o[1], o[2] - L0], [o[0], o[1], o[2] + L0], [1, 1, 0], 2, 0))
            except Exception:
                pass

        # draw trajectory if provided: list of (x,y,z)
        if traj:
            try:
                # ensure points are lists
                pts = [list(map(float, pnt)) for pnt in traj if pnt is not None]
                for a, b in zip(pts[:-1], pts[1:]):
                    debug_ids.append(p.addUserDebugLine(a, b, [1, 0.8, 0], 3, 0))
                # draw current EE as a small sphere-like cross
                last = pts[-1]
                s = max(0.02, float(distance) * 0.03)
                debug_ids.append(p.addUserDebugLine([last[0]-s, last[1], last[2]], [last[0]+s, last[1], last[2]], [1, 0.6, 0], 3, 0))
                debug_ids.append(p.addUserDebugLine([last[0], last[1]-s, last[2]], [last[0], last[1]+s, last[2]], [1, 0.6, 0], 3, 0))
                debug_ids.append(p.addUserDebugLine([last[0], last[1], last[2]-s], [last[0], last[1], last[2]+s], [1, 0.6, 0], 3, 0))
            except Exception:
                pass

        # draw explicit target marker if provided (magenta cross)
        if target is not None:
            try:
                t = list(map(float, target[:3])) if hasattr(target, '__len__') else [float(target)]
                s = max(0.02, float(distance) * 0.03)
                # X line
                debug_ids.append(p.addUserDebugLine([t[0]-s, t[1], t[2]], [t[0]+s, t[1], t[2]], [1, 0, 1], 4, 0))
                # Y line
                debug_ids.append(p.addUserDebugLine([t[0], t[1]-s, t[2]], [t[0], t[1]+s, t[2]], [1, 0, 1], 4, 0))
                # Z line
                debug_ids.append(p.addUserDebugLine([t[0], t[1], t[2]-s], [t[0], t[1], t[2]+s], [1, 0, 1], 4, 0))
            except Exception:
                pass

        img_arr = p.getCameraImage(width, height, view_mat, proj_mat, renderer=renderer)
        rgba = np.reshape(img_arr[2], (height, width, 4))
        rgb = (rgba[:, :, :3]).astype(np.uint8)

        # save using PIL if available, else try imageio
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        try:
            if Image is not None:
                img = Image.fromarray(rgb)
                img.save(filepath)
            else:
                try:
                    import imageio.v2 as imageio
                except Exception:
                    raise RuntimeError("Cannot save image: Pillow or imageio not installed. Install 'pillow' or 'imageio'.")
                imageio.imwrite(filepath, rgb)
        finally:
            # remove debug lines we added
            try:
                for did in debug_ids:
                    p.removeUserDebugItem(did)
            except Exception:
                pass
            return

    def _get_ee_pos(self, gripper_link_name='left_gripper_link'):
        gripper_link_id = self._link_index_by_name(self.robot, gripper_link_name)
        st = p.getLinkState(self.robot, gripper_link_id, computeForwardKinematics=True)
        ee_pos = np.array(st[0], dtype=np.float32)
        self._ee_pos = ee_pos

        return ee_pos

    def _joint_index_by_name(self, robot, name):
        for i in range(p.getNumJoints(robot)):
            if p.getJointInfo(robot, i)[1].decode() == name:
                return i
        return None

    def _link_index_by_name(self, robot, link_name):
        """Return the joint index whose child link has the given name.

        PyBullet reports the child link name in getJointInfo(...)[12]. If the
        requested link is the base (root) it won't be found here.
        """
        for i in range(p.getNumJoints(robot)):
            info = p.getJointInfo(robot, i)
            child_name = info[12].decode() if isinstance(info[12], bytes) else info[12]
            if child_name == link_name:
                return i
        return None

    def _get_linear_vel_chassis_frame(self, yaw, lin_vel_world_frame):
        yaw = float(yaw)
        cy = np.cos(yaw)
        sy = np.sin(yaw)
        
        vx_w, vy_w = float(lin_vel_world_frame[0]), float(lin_vel_world_frame[1])
        vx_r = cy * vx_w + sy * vy_w
        vy_r = -sy * vx_w + cy * vy_w
        return vx_r, vy_r

    def _get_arm_state(self):
        st_left_arm1 = p.getJointState(self.robot, self.joint_name2ids['left_arm_joint1'])[0]
        st_left_arm2 = p.getJointState(self.robot, self.joint_name2ids['left_arm_joint2'])[0]
        st_left_arm3 = p.getJointState(self.robot, self.joint_name2ids['left_arm_joint3'])[0]
        st_left_arm4 = p.getJointState(self.robot, self.joint_name2ids['left_arm_joint4'])[0]
        st_left_arm5 = p.getJointState(self.robot, self.joint_name2ids['left_arm_joint5'])[0]
        st_left_arm6 = p.getJointState(self.robot, self.joint_name2ids['left_arm_joint6'])[0]

        return st_left_arm1, st_left_arm2, st_left_arm3, st_left_arm4, st_left_arm5, st_left_arm6
        
    
    def _get_base_pos(self):
        abstract_idx = self._link_index_by_name(self.robot, 'abstract_chassis_link')
        st_abs = p.getLinkState(self.robot, abstract_idx, computeForwardKinematics=True, computeLinkVelocity=True)
        abs_pos = np.array(st_abs[0], dtype=np.float32)
        euler = p.getEulerFromQuaternion(st_abs[1])   # (roll, pitch, yaw)
        self._abstract_chassis_pos = abs_pos
        self._abstract_chassis_yaw = float(euler[2])
        theta = np.arctan2(abs_pos[1], abs_pos[0])
        
        # link_vel_from_getLinkState (world frame)
        link_lin_vel_world_frame = np.array(st_abs[6], dtype=np.float32)
        link_ang_vel_world_frame = np.array(st_abs[7], dtype=np.float32)
        
        # store base translational/angular velocities
        self._abstract_chassis_ang_vel = link_ang_vel_world_frame

        # compute robot/chassis-frame velocity by rotating world velocity
        # into the chassis yaw frame (so vx_r is forward, vy_r is lateral)
        vx_r, vy_r = self._get_linear_vel_chassis_frame(euler[2], link_lin_vel_world_frame)

        # store both world-frame and chassis-frame velocities
        self._abstract_chassis_lin_vel_world = link_lin_vel_world_frame
        self._abstract_chassis_lin_vel_chassis = np.array([vx_r, vy_r, link_lin_vel_world_frame[2]], dtype=np.float32)

        # print(f"cur chassis pos = ({abs_pos[0]:.2f}, {abs_pos[1]:.2f}), yaw = {euler[2]:.3f}, "
        #       f"chassis_lin_vel_world=({link_lin_vel_world_frame[0]:.3f}, {link_lin_vel_world_frame[1]:.3f}), "
        #       f"chassis_ang_vel={link_ang_vel_world_frame[2]:.3f}, "
        #       f"vx_r={vx_r:.3f}, vy_r={vy_r:.3f}")

        return abs_pos

    def _wrap_angle(self, a: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return (a + np.pi) % (2 * np.pi) - np.pi

    def _get_desired_yaw(self, base_pos, target_pos):
        vec = target_pos[:2] - base_pos[:2]
        dist = float(np.linalg.norm(vec))

        desired_yaw = float(np.arctan2(vec[1], vec[0])) \
            if dist > 1e-2 else float(self._abstract_chassis_yaw)
        self._abstract_chassis_desired_yaw = desired_yaw
        return desired_yaw

    def _get_target(self, ee_pos):
        # self._target_generator.test_vel()
        # self._target_generator.test_target_given_speed_profile()
        if self.finish_step1_step is None:
            self._traj_id, self._target = 0, self._target_generator.traj.get_position(0)
        else:
            self._traj_id, self._target = self._target_generator.cal_target_given_speed_profile(self.step_count - self.finish_step1_step)
            
        remaining_traj_num, self.remain_traj_ratio = self._target_generator.get_remaining_traj_nums(self._traj_id)
        # 预瞄2个点，每个点间隔0.1s
        future_xp = self._target_generator.get_lookahead(self._traj_id, 0.1, self.lookahead_num)
        future_xp = np.array(future_xp, dtype=np.float32).reshape(-1)
        
        if DEBUG:
            print(f"Current target position: {self._target}, traj_id={self._traj_id}, "
                  f"future_2p={future_xp}, "
                  f"remaining_traj_num={remaining_traj_num}, remain_traj_ratio={self.remain_traj_ratio:.2f}, "
                  f"finish_step1_step={self.finish_step1_step}")
        return self._target, future_xp

    def _get_hist_chassis(self):
        # update chassis joint history (joint_x, joint_y, joint_theta)
        jx_vel = float(p.getJointState(self.robot, self.joint_name2ids['joint_x'])[1])
        jy_vel = float(p.getJointState(self.robot, self.joint_name2ids['joint_y'])[1])
        jth_vel = float(p.getJointState(self.robot, self.joint_name2ids['joint_theta'])[1])

        # roll history forward and append newest
        if self._chassis_hist is None:
            self._chassis_hist = np.zeros((self.hist_num, 3), dtype=np.float32)
        else:
            # shift left
            self._chassis_hist = np.roll(self._chassis_hist, -1, axis=0)
        self._chassis_hist[-1, :] = np.array([jx_vel, jy_vel, jth_vel], dtype=np.float32)
        if DEBUG:
            print(f"Chassis joint history:\n{self._chassis_hist}")
    
    def _get_obs(self):
        q = [] # 关节位置
        qdot = [] # 关节速度
        for idx in self.joint_ids:
            s = p.getJointState(self.robot, idx)
            q.append(s[0])
            qdot.append(s[1])
        q = np.array(q, dtype=np.float32)
        qdot = np.array(qdot, dtype=np.float32)
        
        self._get_hist_chassis()
        ee_pos = self._get_ee_pos()
        base_pos = self._get_base_pos()
        self._joint_states = np.concatenate([q, qdot], axis=0)
        # desired_yaw = np.array([self._get_desired_yaw(base_pos, self._target)])
        target, future_5targets = self._get_target(ee_pos)

        # include flattened chassis history in observation
        chassis_hist_flat = self._chassis_hist.reshape(-1).astype(np.float32)

        obs = np.concatenate([q, qdot, chassis_hist_flat, base_pos, ee_pos, target, future_5targets,
                              np.array([self.remain_traj_ratio])]).astype(np.float32)
        return obs

    def _get_target_position(self):
        return self._target
    
    def print_cur_pos(self,):
        for i, jid in enumerate(self.joint_ids):
            cur = p.getJointState(self.robot, jid)[0]
            info = p.getJointInfo(self.robot, jid)
            joint_name = info[1].decode()
            if 'left_arm' in joint_name:
                continue
            print(f"joint {joint_name}: current={cur:.1f}")
    
    def limit_action(self, target_action, action_limits):
        limited_action = np.clip(target_action, action_limits[0], action_limits[1])
        return limited_action

    def step(self, action):
        self.step_count += 1
        self.control_info = {}
        self.control_info['target'] = {}
        self.control_info['reality'] = {}
        # if self.step_count > 200:
        #     import pdb; pdb.set_trace()
        action = np.array(action, dtype=np.float32).flatten()
        self._get_base_pos()
        last_state_left_arm1, last_state_left_arm2, last_state_left_arm3, last_state_left_arm4, \
            last_state_left_arm5, last_state_left_arm6 = self._get_arm_state()

        last_yaw = self._abstract_chassis_yaw
        last_pos = self._abstract_chassis_pos

        last_chassis_vel = self.last_chassis_vel if self.last_chassis_vel is not None else 0.0
        # ds = vt + 1/2*at^2
        action_ds = 0.1 * last_chassis_vel + 0.5 * (1.5 * action[0]) * 0.1 * 0.1 # 加速度限制1m/s^2
        action_dtheta = action[1] * 0.01
        joint_delta_limit = 0.02
        action_left_arm1 = action[2] * joint_delta_limit
        action_left_arm2 = action[3] * joint_delta_limit
        action_left_arm3 = action[4] * joint_delta_limit
        action_left_arm4 = action[5] * joint_delta_limit
        action_left_arm5 = action[6] * joint_delta_limit
        action_left_arm6 = action[7] * joint_delta_limit
        
        action_dx = action_ds * np.cos(last_yaw)
        action_dy = action_ds * np.sin(last_yaw)
        
        self.control_info['target'] = {
            'dx': action_dx,
            'dy': action_dy,
            'dtheta': action_dtheta,
            'left_arm1': action_left_arm1,
            'left_arm2': action_left_arm2,
            'left_arm3': action_left_arm3,
            'left_arm4': action_left_arm4,
            'left_arm5': action_left_arm5,
            'left_arm6': action_left_arm6
        }
        
        target_x = last_pos[0] + action_dx
        target_y = last_pos[1] + action_dy
        target_theta = self._wrap_angle(last_yaw + action_dtheta) # 底盘theta是不限制范围的，所以需要wrap一下

        target_left_arm1 = self.limit_action(last_state_left_arm1 + action_left_arm1, self.joint_limits['left_arm_joint1'])
        target_left_arm2 = self.limit_action(last_state_left_arm2 + action_left_arm2, self.joint_limits['left_arm_joint2'])
        target_left_arm3 = self.limit_action(last_state_left_arm3 + action_left_arm3, self.joint_limits['left_arm_joint3'])
        target_left_arm4 = self.limit_action(last_state_left_arm4 + action_left_arm4, self.joint_limits['left_arm_joint4'])
        target_left_arm5 = self.limit_action(last_state_left_arm5 + action_left_arm5, self.joint_limits['left_arm_joint5'])
        target_left_arm6 = self.limit_action(last_state_left_arm6 + action_left_arm6, self.joint_limits['left_arm_joint6'])

        if DEBUG:
            print(f"action: ds={action_ds:.3f}"
                  f"(dx={action_dx:.3f}, dy={action_dy:.3f}), dtheta={action_dtheta:.3f}, "
                  f"target=(x={target_x:.3f}, y={target_y:.3f}, theta={target_theta:.3f}), "
                  f"left_arm1={action_left_arm1:.3f}, left_arm2={action_left_arm2:.3f}, "
                  f"left_arm3={action_left_arm3:.3f}, left_arm4={action_left_arm4:.3f}, "
                  f"left_arm5={action_left_arm5:.3f}, left_arm6={action_left_arm6:.3f}")

        p.setJointMotorControlArray(self.robot,
                                    jointIndices=[self.joint_name2ids['joint_x'], self.joint_name2ids['joint_y']],
                                    controlMode=p.POSITION_CONTROL, targetPositions=[target_x, target_y],
                                    forces=[100, 100])

        p.setJointMotorControl2(self.robot, self.joint_name2ids['joint_theta'],
                                p.POSITION_CONTROL, targetPosition=target_theta, force=50)
        
        p.setJointMotorControl2(self.robot, self.joint_name2ids['left_arm_joint1'],
                                p.POSITION_CONTROL, targetPosition=target_left_arm1,
                                force=self.joint_effort_limit['left_arm_joint1'])

        p.setJointMotorControl2(self.robot, self.joint_name2ids['left_arm_joint2'],
                                p.POSITION_CONTROL, targetPosition=target_left_arm2,
                                force=self.joint_effort_limit['left_arm_joint2'])

        p.setJointMotorControl2(self.robot, self.joint_name2ids['left_arm_joint3'],
                                p.POSITION_CONTROL, targetPosition=target_left_arm3,
                                force=self.joint_effort_limit['left_arm_joint3'])

        p.setJointMotorControl2(self.robot, self.joint_name2ids['left_arm_joint4'],
                                p.POSITION_CONTROL, targetPosition=target_left_arm4,
                                force=self.joint_effort_limit['left_arm_joint4'])

        p.setJointMotorControl2(self.robot, self.joint_name2ids['left_arm_joint5'],
                                p.POSITION_CONTROL, targetPosition=target_left_arm5,
                                force=self.joint_effort_limit['left_arm_joint5'])

        p.setJointMotorControl2(self.robot, self.joint_name2ids['left_arm_joint6'],
                                p.POSITION_CONTROL, targetPosition=target_left_arm6,
                                force=self.joint_effort_limit['left_arm_joint6'])

        # step simulation
        for _ in range(self.frame_skip):
            p.stepSimulation()

        # get observation and base position
        obs = self._get_obs()

        base_pos = getattr(self, '_abstract_chassis_pos', np.array([0.0, 0.0, 0.0], dtype=float))
        ee_pos = getattr(self, '_ee_pos', np.array([0.0, 0.0, 0.0], dtype=float))
        base_yaw = float(getattr(self, '_abstract_chassis_yaw', 0.0))
        base_lin_vel_chassis_frame = getattr(self, '_abstract_chassis_lin_vel_chassis', np.array([0.0, 0.0, 0.0], dtype=float))
        base_ang_vel = getattr(self, '_abstract_chassis_ang_vel', np.array([0.0, 0.0, 0.0], dtype=float))
        
        base_lin_vel_norm = np.linalg.norm(base_lin_vel_chassis_frame)
        base_ang_vel_norm = np.linalg.norm(base_ang_vel)

        cur_state_left_arm1, cur_state_left_arm2, cur_state_left_arm3, cur_state_left_arm4, \
                cur_state_left_arm5, cur_state_left_arm6 = self._get_arm_state()
        
        if DEBUG:
            print(f"current arm states: 1={cur_state_left_arm1:.3f}, 2={cur_state_left_arm2:.3f}, 3={cur_state_left_arm3:.3f}, 4={cur_state_left_arm4:.3f}, 5={cur_state_left_arm5:.3f}, 6={cur_state_left_arm6:.3f}")
            print(f"actual movement: act dx = {base_pos[0] - last_pos[0]:.3f}, dy = {base_pos[1] - last_pos[1]:.3f}, dtheta = {self._wrap_angle(base_yaw - last_yaw):.3f}")
        
        self.control_info['reality'] = {
            'dx': base_pos[0] - last_pos[0],
            'dy': base_pos[1] - last_pos[1],
            'dtheta': self._wrap_angle(base_yaw - last_yaw),
            'left_arm1': cur_state_left_arm1 - last_state_left_arm1,
            'left_arm2': cur_state_left_arm2 - last_state_left_arm2,
            'left_arm3': cur_state_left_arm3 - last_state_left_arm3,
            'left_arm4': cur_state_left_arm4 - last_state_left_arm4,
            'left_arm5': cur_state_left_arm5 - last_state_left_arm5,
            'left_arm6': cur_state_left_arm6 - last_state_left_arm6
        }
        
        reward, info = compute_reward(base_pos, base_lin_vel_norm, ee_pos, self._target, base_yaw,
                                    wrap_angle_fn=self._wrap_angle,
                                    remaining_ratio=self.remain_traj_ratio)

        # terminated: 成功或者失败的自然终止
        static_arm_thr = 0.02
        is_joint1_static = bool(abs(self._joint_states[self.n_j+0]) < static_arm_thr)
        is_joint2_static = bool(abs(self._joint_states[self.n_j+1]) < static_arm_thr)
        is_joint3_static = bool(abs(self._joint_states[self.n_j+2]) < static_arm_thr)
        is_joint4_static = bool(abs(self._joint_states[self.n_j+3]) < static_arm_thr)
        is_joint5_static = bool(abs(self._joint_states[self.n_j+4]) < static_arm_thr)
        is_joint6_static = bool(abs(self._joint_states[self.n_j+5]) < static_arm_thr)
        is_joint7_static = bool(abs(self._joint_states[self.n_j+6]) < static_arm_thr)
        is_joint8_static = bool(abs(self._joint_states[self.n_j+7]) < static_arm_thr)
        is_joint9_static = bool(abs(self._joint_states[self.n_j+8]) < static_arm_thr)

        is_arm_joints_static = is_joint1_static and is_joint2_static and is_joint3_static and \
            is_joint4_static and is_joint5_static and is_joint6_static and \
            is_joint7_static and is_joint8_static and is_joint9_static
        

        if not self.finish_step1_step and bool(info.get("ee_distance", 9999.0) < 0.05):
            self.finish_step1_step = self.step_count

        reached_goal = bool(self._traj_id == self._target_generator.traj.get_position_len() - 1) and \
            bool(info.get("ee_distance", 9999.0) < 0.05) and \
            bool(base_lin_vel_norm < 0.1) and bool(base_ang_vel_norm < 0.1) and \
            is_arm_joints_static
        # if reached_goal and is_arm_joints_static:
        #     import pdb; pdb.set_trace()

        target_len = np.linalg.norm(self._target[:2])
        base_len = np.linalg.norm(base_pos[:2])

        # terminated is True for natural success OR terminal failure (out_of_bounds)
        terminated = reached_goal

        # truncated: episode ended because of time/step limit only
        truncated = bool(self.step_count >= self.max_steps)

        # annotate info for downstream wrappers / loggers
        if truncated:
            info['TimeLimit.truncated'] = True
        if reached_goal:
            info['is_success'] = True

        
        if DEBUG:
            print(f"Step {self.step_count}: finish_step1_step = {self.finish_step1_step}, Distance: {info['ee_distance']:.1f}m, "
                f"Target: ({self._target[0]:.3f}, {self._target[1]:.3f}, {self._target[2]:.3f})m, "
                f"base_pos = ({base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f}), "
                f"ee_pos = ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}), "
                f"Reward: {reward:.2f} \n")

        self.last_chassis_vel = base_lin_vel_norm
        # import pdb; pdb.set_trace()
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        # GUI mode is handled by PyBullet GUI; nothing to return unless rgb_array requested.
        pass

    def close(self):
        if p.isConnected():
            p.disconnect()
