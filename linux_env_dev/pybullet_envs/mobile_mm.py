import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
try:
    from PIL import Image
except Exception:
    Image = None


class MobileMMBulletEnv(gym.Env):
    """Minimal PyBullet-based mobile manipulator tracking env.

    Loads URDF from project: assets_own/mobile_manipulator_PPR_base_corrected.urdf
    Observation: [q, qdot, ee_pos, target_pos]
    Action: joint position deltas scaled by max_delta
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, urdf_path=None, frame_skip=10, max_steps=1000, render=False,
                 save_image_on_reset=False, save_image_path=None, image_width=800, image_height=600,
                 reward_distance_weight: float = 0.5, reward_yaw_weight: float = 0.5):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        # self.urdf_path = urdf_path or os.path.join(self.project_root, "assets_own", "mobile_manipulator_PPR_base_corrected.urdf")
        self.urdf_path = urdf_path or os.path.join(self.project_root, "assets_own", "mobile_manipulator_PPR_theta_before_x.urdf")
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
        self.joint_ids = []
        self.joint_types = []
        self.joint_lower = []
        self.joint_upper = []
        self.step_count = 0

        # build a temporary scene to identify joints
        self._load_robot_once()

        # action / obs spaces
        n_j = len(self.joint_ids)
        # per-joint max delta (in joint units): larger for prismatic (meters), smaller for revolute (radians)
        self._action_max_delta = []
        for i, jid in enumerate(self.joint_ids):
            jtype = self.joint_types[i]
            if jtype == p.JOINT_PRISMATIC:
                # allow up to 0.5 m per step by default
                self._action_max_delta.append(0.5)
            else:
                # revolute: default ±0.1 rad per step
                # if joint limits are available, scale to a fraction of the range
                lower = self.joint_lower[i]
                upper = self.joint_upper[i]
                if np.isfinite(lower) and np.isfinite(upper) and upper > lower:
                    # allow up to 5% of joint range per action step, clamped
                    rng = float(upper - lower)
                    self._action_max_delta.append(max(0.01, min(0.2, 0.05 * rng)))
                else:
                    self._action_max_delta.append(0.1)

        # action space uses normalized controls in [-1, 1] per-dimension; we'll scale inside step()
        self.action_space = spaces.Box(low=-1, high=1, shape=(n_j,), dtype=np.float32)
        obs_dim = n_j * 2 + 3 + 3 + 3  # joint pos, joint vel, ee_pos(3) + target_pos(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # simple fixed target for initial tests
        # ee初始位置是[0.4, 0.0, 0.9]
        # base_pos初始位置是[0.0, 0.0, 0.2]
        self._target = np.array([10.0, 1.0, 0.2], dtype=np.float32)

        # reward shaping weights
        self.reward_distance_weight = float(reward_distance_weight)
        self.reward_yaw_weight = float(reward_yaw_weight)

    def _load_robot_once(self):
        # load plane and robot, then remove (we re-create on reset)
        plane = p.loadURDF("plane.urdf")
        try:
            robot = p.loadURDF(self.urdf_path, useFixedBase=False)
        except Exception as e:
            raise FileNotFoundError(f"URDF not found or failed to load: {self.urdf_path} -> {e}")
        # collect non-fixed joint ids
        self.joint_ids = []
        self.joint_lower = []
        self.joint_upper = []
        is_y_prismatic_found = False
        for i in range(p.getNumJoints(robot)):
            info = p.getJointInfo(robot, i)
            if info[1] == b'joint_y':
                # skip y prismatic joint to fix lateral movement
                print("[MobileMMBulletEnv] Skipping y prismatic joint to fix lateral movement.")
                is_y_prismatic_found = True
                continue
            # import pdb; pdb.set_trace()
            jtype = info[2]
            if jtype != p.JOINT_FIXED:
                # p.JOINT_REVOLUTE（旋转）
                # p.JOINT_PRISMATIC（平移）
                self.joint_ids.append(i)
                self.joint_types.append(jtype)
                self.joint_lower.append(info[8])
                self.joint_upper.append(info[9])
        
        # 去掉y方向移动的自由度
        if not is_y_prismatic_found:
            print("[MobileMMBulletEnv] Warning: 'joint_y' prismatic joint not found; lateral movement may be possible.")
            raise RuntimeError("Expected 'joint_y' prismatic joint not found in URDF.")
        
        # clean up
        p.removeBody(robot)
        p.removeBody(plane)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.step_count = 0
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        self.robot = p.loadURDF(self.urdf_path, useFixedBase=False)
        # reset joints to random small values
        for idx in self.joint_ids:
            p.resetJointState(self.robot, idx, targetValue=0.0, targetVelocity=0.0)

        # optionally save image after robot spawned
        if self.save_image_on_reset:
            out_path = self.save_image_path or os.path.join(self.project_root, "linux_env_dev", "robot_pose.png")
            try:
                self.save_robot_image(out_path, width=self.image_width, height=self.image_height)
                print(f"[MobileMMBulletEnv] Saved robot pose image to: {out_path}")
            except Exception as e:
                print(f"[MobileMMBulletEnv] Warning: failed to save robot image: {e}")

        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot)
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

    def _get_ee_pos(self):
        # try to use last joint/link as end-effector; adjust if URDF link name known
        last_link = self.joint_ids[-1] if self.joint_ids else -1
        if last_link >= 0:
            st = p.getLinkState(self.robot, last_link, computeForwardKinematics=True)
            ee_pos = np.array(st[0], dtype=np.float32)
        else:
            ee_pos = np.zeros(3, dtype=np.float32)
        # print(f" Current EE Position: {ee_pos}")
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
    
    def _get_base_pos(self):
        # attempt to locate the abstract chassis link by name and record its world position
        abstract_idx = self._link_index_by_name(self.robot, 'abstract_chassis_link')
        st_abs = p.getLinkState(self.robot, abstract_idx, computeForwardKinematics=True, computeLinkVelocity=True)
        abs_pos = np.array(st_abs[0], dtype=np.float32)
        euler = p.getEulerFromQuaternion(st_abs[1])   # (roll, pitch, yaw)
        self._abstract_chassis_pos = abs_pos
        # store chassis yaw for reward computations (roll, pitch, yaw)
        self._abstract_chassis_yaw = float(euler[2])
        self._abstract_chassis_lin_vel = st_abs[6]
        self._abstract_chassis_ang_vel = st_abs[7]


        # print(f" Abstract chassis position: ({abs_pos[0]:.1f}, {abs_pos[1]:.5f}, {abs_pos[2]:.1f}), "
        #       f"orientation: {euler[2]:.5f}")

        return abs_pos

    def _wrap_angle(self, a: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return (a + np.pi) % (2 * np.pi) - np.pi

    def _get_obs(self):
        q = [] # 关节位置
        qdot = [] # 关节速度
        for idx in self.joint_ids:
            s = p.getJointState(self.robot, idx)
            q.append(s[0])
            qdot.append(s[1])
        q = np.array(q, dtype=np.float32)
        qdot = np.array(qdot, dtype=np.float32)
        ee_pos = self._get_ee_pos()
        base_pos = self._get_base_pos()
        obs = np.concatenate([q, qdot, base_pos, ee_pos, self._target]).astype(np.float32)
        # import pdb; pdb.set_trace()
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
            

    def step(self, action):
        self.step_count += 1
        action = np.array(action, dtype=np.float32).flatten()
        n = len(self.joint_ids)

        for i, jid in enumerate(self.joint_ids):
            cur = p.getJointState(self.robot, jid)[0]
            info = p.getJointInfo(self.robot, jid)
            effort_limit = info[10]
            joint_name = info[1].decode()

            joint_type = self.joint_types[i]
            # ignore left_arm joints for now
            if 'left_arm' in joint_name:
                target = cur
            else:
                # incoming action is normalized in [-1,1]; scale by per-joint delta
                max_delta = float(self._action_max_delta[i])
                scaled_delta = float(np.clip(action[i], -1.0, 1.0)) * max_delta

                if joint_type == p.JOINT_PRISMATIC:
                    target = cur + scaled_delta
                else:
                    # revolute: apply delta and wrap
                    target = cur + scaled_delta
                    target = float(self._wrap_angle(target))
                    if target > np.pi / 3:
                        target = np.pi / 3
                    elif target < -np.pi / 3:
                        target = -np.pi / 3

                # # clamp to joint limits if available
                # lower = self.joint_lower[i]
                # upper = self.joint_upper[i]
                # try:
                #     if np.isfinite(lower) and np.isfinite(upper):
                #         target = float(np.clip(target, float(lower), float(upper)))
                # except Exception:
                #     pass

                print(f"Action for joint {joint_name}: current={cur:.3f}, raw_action={action[i]:.3f}, target={target:.3f}, effort_limit={effort_limit:.1f}")
                p.setJointMotorControl2(self.robot, jid, p.POSITION_CONTROL, targetPosition=target, force=min(100, effort_limit / 2))
        # step simulation
        for _ in range(self.frame_skip):
            p.stepSimulation()

        obs = self._get_obs()
        ee = obs[-6:-3]
        base_pos = obs[-9:-6]
        # distance term (Euclidean in 3D)
        dist = float(np.linalg.norm(base_pos[:2] - self._target[:2]))

        vec_to_target = np.array(self._target[:2], dtype=float) - np.array(base_pos[:2], dtype=float)
        desired_yaw = float(np.arctan2(vec_to_target[1], vec_to_target[0]))
        current_yaw = float(getattr(self, '_abstract_chassis_yaw', 0.0))
        current_pos = getattr(self, '_abstract_chassis_pos', (0.0, 0.0, 0.0))
        current_lin_vel = getattr(self, '_abstract_chassis_lin_vel', (0.0, 0.0, 0.0))
        current_ang_vel = getattr(self, '_abstract_chassis_ang_vel', (0.0, 0.0, 0.0))
        # import pdb; pdb.set_trace()
        yaw_error = float(self._wrap_angle(desired_yaw - current_yaw))

        # combine into a scalar reward: negative weighted sum (smaller is better)
        reward_dist_term = self.reward_distance_weight * dist
        # penalize orientation error (use absolute yaw error)
        reward_yaw_term = self.reward_yaw_weight * abs(yaw_error)
        reward = -float(reward_dist_term + reward_yaw_term)
        # reward = -float(abs(current_yaw) ** 2)
        # print(f"reward = {reward:.4f} yaw_term={current_yaw:.4f})")

        terminated = bool(dist < 0.05 or self.step_count >= self.max_steps)
        truncated = False
        info = {"distance": float(dist), "yaw_error": float(yaw_error)}
        print(f"Step {self.step_count}: Distance: {dist:.4f}, YawErr: {yaw_error:.4f},"
              f" Target: {self._target}, base_pos = ({base_pos[0]:.1f}, {base_pos[1]:.1f}, {base_pos[2]:.1f})")
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        # GUI mode is handled by PyBullet GUI; nothing to return unless rgb_array requested.
        pass

    def close(self):
        if p.isConnected():
            p.disconnect()
