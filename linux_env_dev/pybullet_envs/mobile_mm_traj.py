import os
import numpy as np
from typing import Optional
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
from PIL import Image
import math

from .reward_helpers import compute_reward, DEBUG
from .target_generator import TargetGenerator, FixedTarget, JSONNearestTargetGenerator
from .robot_specs import get_robot_spec
from .urdf_utils import prepare_urdf_for_pybullet

class MobileMMTrajEnv(gym.Env):
    """Minimal PyBullet-based mobile manipulator tracking env.

    Loads URDF from project: assets_own/mobile_manipulator_PPR_base_corrected.urdf
    Observation: [q, qdot, ee_pos, target_pos]
    Action: joint position deltas scaled by max_delta
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, robot: str = "mobile_mm", urdf_path=None, frame_skip=24, max_steps=500, render=False,
                 save_image_on_reset=False, save_image_path=None, image_width=800, image_height=600,
                 reward_distance_weight: float = 1.0, reward_yaw_weight: float = 1.0,
                 reward_dist_weight: Optional[float] = None,
                 reward_collision_threshold: Optional[float] = None,
                 reward_collision_ratio: Optional[float] = None,
                 reward_clip_abs: Optional[float] = None,
                 target_generator: Optional[TargetGenerator] = None):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.robot_name = (robot or "mobile_mm").strip().lower()
        self.robot_spec = get_robot_spec(self.robot_name)

        raw_urdf_path = urdf_path or self.robot_spec.default_urdf_path
        cache_dir = os.path.join(self.project_root, "linux_env_dev", "_urdf_cache")
        self.urdf_path = prepare_urdf_for_pybullet(raw_urdf_path, self.robot_spec.package_rewrites, cache_dir)

        self.base_joint_x, self.base_joint_y, self.base_joint_yaw = self.robot_spec.base_joints_xyz_yaw
        self.arm_joints = list(self.robot_spec.arm_joints)
        self.stabilized_joints = list(self.robot_spec.stabilized_joints)
        self.base_link_name = self.robot_spec.base_link_name
        self.ee_link_name = self.robot_spec.ee_link_name

        self.frame_skip = int(frame_skip)
        self.max_steps = int(max_steps)
        self.render = bool(render)
        self.use_fixed_base = bool(self.robot_name == "recomo")

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
        try:
            p.setTimeStep(1.0 / 240.0)
        except Exception:
            pass
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

        # action dims: mobile_mm uses (2 base + 6 arm); recomo uses (3 base + 2 arm)
        self.base_action_dim = 3 if self.robot_name == "recomo" else 2
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.base_action_dim + len(self.arm_joints),), dtype=np.float32
        )
        # obs layout: joint pos (n_j), joint vel (n_j), chassis_history (hist_num*3), base_pos(3), ee_pos(3),
        # target + lookahead (3*(1+lookahead_num)), remain_ratio(1)
        obs_dim = self.n_j * 2 + (self.hist_num * 3) + 3 + 3 + 3*(1 + self.lookahead_num) + 1 + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        if target_generator is None:
            default_train_txt = os.path.join(self.project_root, "linux_env_dev", "new_json_50", "train.txt")
            if not os.path.isfile(default_train_txt):
                raise ValueError(
                    "MobileMMTrajEnv requires a trajectory-based target_generator, "
                    f"but none was provided and default file is missing: {default_train_txt}"
                )
            target_generator = JSONNearestTargetGenerator(json_paths=[], json_txt=default_train_txt, mode="random")
        self._target_generator = target_generator
        self._target = self._target_generator.reset()
        self.finish_step1_step = None
        self.finish_step1_step = 0
        self.remain_traj_ratio = 1.0
        self._traj_id = 0
        self._last_traj_id = 0
        # chassis joint history buffer: shape (hist_num, 3) columns = [joint_x, joint_y, joint_theta]
        self._chassis_hist = np.zeros((self.hist_num, 3), dtype=np.float32)
        self.last_chassis_vel = None

        # reward shaping (robot-aware)
        if self.robot_name == "recomo":
            # ee_tool is nearly above base in XY; disable base-vs-ee collision penalty
            self._reward_kwargs = {
                "dist_weight": 2.0,
                "collision_ratio": 0.0,
            }
        else:
            self._reward_kwargs = {
                "dist_weight": 2.0,
                "collision_threshold": 0.3,
                "collision_ratio": 50.0,
            }

        # Optional reward overrides (useful for hyperparameter sweeps)
        if reward_dist_weight is not None:
            self._reward_kwargs["dist_weight"] = float(reward_dist_weight)
        elif reward_distance_weight != 1.0:
            # Backward-compatible: legacy arg was previously unused; apply only if user overrides it.
            self._reward_kwargs["dist_weight"] = float(reward_distance_weight)
        if reward_collision_threshold is not None:
            self._reward_kwargs["collision_threshold"] = float(reward_collision_threshold)
        if reward_collision_ratio is not None:
            self._reward_kwargs["collision_ratio"] = float(reward_collision_ratio)
        if reward_clip_abs is not None:
            self._reward_kwargs["clip_abs"] = float(reward_clip_abs)
    
    def _load_robot_once(self):
        # load plane and robot, then remove (we re-create on reset)
        plane = p.loadURDF("plane.urdf")
        try:
            robot = p.loadURDF(self.urdf_path, useFixedBase=self.use_fixed_base)
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

        required_joints = [self.base_joint_x, self.base_joint_y, self.base_joint_yaw] + self.arm_joints
        missing = [jn for jn in required_joints if jn and jn not in self.joint_name2ids]
        if missing:
            raise KeyError(
                f"[{self.robot_name}] Required joint(s) not found in URDF: {missing}. "
                f"URDF={self.urdf_path}"
            )
        # stabilized joints (e.g. gimbal) are optional: keep only those present
        self.stabilized_joints = [jn for jn in self.stabilized_joints if jn in self.joint_name2ids]

        # clean up
        p.removeBody(robot)
        p.removeBody(plane)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self._target = self._target_generator.reset()
        self.finish_step1_step = None
        self.finish_step1_step = 0
        self.remain_traj_ratio = 1.0
        self.step_count = 0
        self._traj_id = 0
        self._last_traj_id = 0
        self.last_chassis_vel = None
        self._chassis_hist = np.zeros((self.hist_num, 3), dtype=np.float32)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        try:
            p.setTimeStep(1.0 / 240.0)
        except Exception:
            pass
        p.loadURDF("plane.urdf")
        self.robot = p.loadURDF(self.urdf_path, useFixedBase=self.use_fixed_base)

        start_pos = self._target_generator.traj.get_position(0, 0.0)

        # reset joints to random small values
        for idx in self.joint_ids:
            p.resetJointState(self.robot, idx, targetValue=0.0, targetVelocity=0.0)

        original_heading = self._target_generator.get_path_heading()
        
        # mobile_mm: start with base offset behind the target so the arm can reach.
        # recomo: holonomic base + short arm, so start near the target.
        if self.robot_name == "recomo":
            xy_jitter = 0.05
            dx = float(np.random.uniform(-xy_jitter, xy_jitter))
            dy = float(np.random.uniform(-xy_jitter, xy_jitter))
        else:
            ds_mv = 0.65 + 0.05 * np.random.uniform(-1.0, 1.0)
            dx = ds_mv * math.cos(original_heading)
            dy = ds_mv * math.sin(original_heading)
        dtheta = 0.05 * np.random.uniform(-1.0, 1.0)
        p.resetJointState(self.robot, self.joint_name2ids[self.base_joint_x],
                          targetValue=float(start_pos[0] - dx), targetVelocity=0.0)
        p.resetJointState(self.robot, self.joint_name2ids[self.base_joint_y],
                          targetValue=float(start_pos[1] - dy), targetVelocity=0.0)
        p.resetJointState(self.robot, self.joint_name2ids[self.base_joint_yaw],
                          targetValue=self._wrap_angle(original_heading + dtheta), targetVelocity=0.0)
        # p.resetJointState(self.robot, self.joint_name2ids['joint_theta'],
        #                   targetValue=np.pi / 2, targetVelocity=0.0)
        if DEBUG:
            print(f"Reset robot base to x={start_pos[0]-dx:.2f}, y={start_pos[1]-dy:.2f}, heading={original_heading:.2f} rad, "
                  f"dx = {dx:.2f}, dy={dy:.2f}")
        
        # arm (and optionally gimbal) initial configuration
        if self.robot_name == "mobile_mm":
            init_joints = {
                "left_arm_joint1": -1.61,
                "left_arm_joint2": 0.00,
                "left_arm_joint3": -0.32,
                "left_arm_joint4": -1.56,
                "left_arm_joint5": -1.13,
                "left_arm_joint6": 0.0,
            }
        elif self.robot_name == "recomo":
            init_joints = {
                "joint6_arm_yaw": 0.0,
                "joint5_arm_pitch": 0.0,
            }
        else:
            init_joints = {}

        for jn, jv in init_joints.items():
            if jn in self.joint_name2ids:
                p.resetJointState(self.robot, self.joint_name2ids[jn], targetValue=float(jv), targetVelocity=0.0)

        # keep stabilized joints (e.g. gimbal) at 0 on reset
        for jn in self.stabilized_joints:
            if jn in self.joint_name2ids:
                p.resetJointState(self.robot, self.joint_name2ids[jn], targetValue=0.0, targetVelocity=0.0)

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

    def _get_ee_pos(self, link_name: Optional[str] = None):
        link_name = link_name or self.ee_link_name
        gripper_link_id = self._link_index_by_name(self.robot, link_name)
        if gripper_link_id is None:
            raise KeyError(f"[{self.robot_name}] EE link not found: {link_name}")
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
        return tuple(
            p.getJointState(self.robot, self.joint_name2ids[jn])[0]
            for jn in self.arm_joints
            if jn in self.joint_name2ids
        )
        
    
    def _get_base_pos(self):
        abstract_idx = self._link_index_by_name(self.robot, self.base_link_name)
        if abstract_idx is None:
            raise KeyError(f"[{self.robot_name}] Base link not found: {self.base_link_name}")
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
        # import pdb; pdb.set_trace()

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
                #   f"future_2p={future_xp}, "
                  f"remaining_traj_num={remaining_traj_num}, remain_traj_ratio={self.remain_traj_ratio:.2f}, "
                  f"finish_step1_step={self.finish_step1_step}")
        return self._target, future_xp

    def _get_hist_chassis(self):
        # update chassis joint history (joint_x, joint_y, joint_theta)
        jx_vel = float(p.getJointState(self.robot, self.joint_name2ids[self.base_joint_x])[1])
        jy_vel = float(p.getJointState(self.robot, self.joint_name2ids[self.base_joint_y])[1])
        jth_vel = float(p.getJointState(self.robot, self.joint_name2ids[self.base_joint_yaw])[1])

        # roll history forward and append newest
        if self._chassis_hist is None:
            self._chassis_hist = np.zeros((self.hist_num, 3), dtype=np.float32)
        else:
            # shift left
            self._chassis_hist = np.roll(self._chassis_hist, -1, axis=0)
        self._chassis_hist[-1, :] = np.array([jx_vel, jy_vel, jth_vel], dtype=np.float32)
        # if DEBUG:
        #     print(f"Chassis joint history:\n{self._chassis_hist}")
    
    def _get_obs(self):
        q = [] # 关节位置
        qdot = [] # 关节速度
        for idx in self.joint_ids:
            s = p.getJointState(self.robot, idx)
            q.append(s[0])
            qdot.append(s[1])
        q = np.array(q, dtype=np.float32)
        yaw_joint_id = self.joint_name2ids[self.base_joint_yaw]
        yaw_pos = self.joint_ids.index(yaw_joint_id)
        q_with_ang = np.concatenate(
            [q[:yaw_pos], [np.cos(q[yaw_pos]), np.sin(q[yaw_pos])], q[yaw_pos + 1 :]]
        )
        qdot = np.array(qdot, dtype=np.float32)
        
        self._get_hist_chassis()
        ee_pos = self._get_ee_pos()
        base_pos = self._get_base_pos()
        self._joint_states = np.concatenate([q, qdot], axis=0)
        # desired_yaw = np.array([self._get_desired_yaw(base_pos, self._target)])
        target, future_5targets = self._get_target(ee_pos)

        # include flattened chassis history in observation
        chassis_hist_flat = self._chassis_hist.reshape(-1).astype(np.float32)
        # obs = np.concatenate([q, qdot, chassis_hist_flat, base_pos, ee_pos, target, future_5targets,
        #                       np.array([self.remain_traj_ratio])]).astype(np.float32)

        obs = np.concatenate([q_with_ang, qdot, chassis_hist_flat, base_pos, ee_pos, target, future_5targets,
                              np.array([self.remain_traj_ratio])]).astype(np.float32)

        if DEBUG:
            print(f"obs = {q[:3]}")
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
        try:
            low, high = float(action_limits[0]), float(action_limits[1])
        except Exception:
            return target_action
        # PyBullet reports (lower, upper) as (1, -1) for continuous/unbounded joints.
        if low > high:
            return target_action
        return float(np.clip(target_action, low, high))

    def step(self, action):
        self.step_count += 1
        self.control_info = {"target": {}, "reality": {}}
        action = np.array(action, dtype=np.float32).flatten()

        # cache last states
        self._get_base_pos()
        last_base_pos = getattr(self, "_abstract_chassis_pos", np.array([0.0, 0.0, 0.0], dtype=float))
        last_yaw = float(p.getJointState(self.robot, self.joint_name2ids[self.base_joint_yaw])[0])
        last_arm_state_by_name = {
            jn: float(p.getJointState(self.robot, self.joint_name2ids[jn])[0])
            for jn in self.arm_joints
            if jn in self.joint_name2ids
        }

        dt = float(self.frame_skip) / 240.0
        base_acc_limit = 1.5  # m/s^2
        base_speed_limit = 0.5  # m/s
        yaw_rate_limit = 0.1  # rad/s
        arm_delta_limit = 0.02  # rad

        base_x = float(p.getJointState(self.robot, self.joint_name2ids[self.base_joint_x])[0])
        base_y = float(p.getJointState(self.robot, self.joint_name2ids[self.base_joint_y])[0])
        base_yaw = last_yaw

        if self.robot_name == "recomo":
            if action.shape[0] != (3 + len(self.arm_joints)):
                raise ValueError(
                    f"[recomo] Expected action dim={3 + len(self.arm_joints)}, got {action.shape[0]}"
                )
            vx_cmd, vy_cmd, wz_cmd = float(action[0]), float(action[1]), float(action[2])
            dx_r = vx_cmd * base_speed_limit * dt
            dy_r = vy_cmd * base_speed_limit * dt
            cy = float(np.cos(base_yaw))
            sy = float(np.sin(base_yaw))
            dx_w = cy * dx_r - sy * dy_r
            dy_w = sy * dx_r + cy * dy_r
            dtheta = wz_cmd * yaw_rate_limit * dt
            base_act_dim = 3
        else:
            if action.shape[0] != (2 + len(self.arm_joints)):
                raise ValueError(
                    f"[mobile_mm] Expected action dim={2 + len(self.arm_joints)}, got {action.shape[0]}"
                )
            last_forward_vel = float(self.last_chassis_vel) if self.last_chassis_vel is not None else 0.0
            a_cmd, wz_cmd = float(action[0]), float(action[1])
            ds = dt * last_forward_vel + 0.5 * (base_acc_limit * a_cmd) * dt * dt
            dx_w = float(ds * np.cos(base_yaw))
            dy_w = float(ds * np.sin(base_yaw))
            dtheta = wz_cmd * yaw_rate_limit * dt
            base_act_dim = 2

        target_x = base_x + dx_w
        target_y = base_y + dy_w
        target_theta = base_yaw + dtheta

        self.control_info["target"].update({"dx": dx_w, "dy": dy_w, "dtheta": dtheta})

        arm_targets = {}
        for i, jn in enumerate(self.arm_joints):
            if jn not in self.joint_name2ids:
                continue
            cur = float(p.getJointState(self.robot, self.joint_name2ids[jn])[0])
            delta = float(action[base_act_dim + i]) * arm_delta_limit
            target = self.limit_action(cur + delta, self.joint_limits.get(jn, (1.0, -1.0)))
            arm_targets[jn] = target
            self.control_info["target"][jn] = delta

        if DEBUG:
            print(
                f"[{self.robot_name}] dt={dt:.3f} dx={dx_w:.3f} dy={dy_w:.3f} dtheta={dtheta:.3f} "
                f"target=(x={target_x:.3f}, y={target_y:.3f}, theta={target_theta:.3f})"
            )

        base_force_xy = 200 if self.robot_name == "recomo" else 100
        p.setJointMotorControlArray(
            self.robot,
            jointIndices=[self.joint_name2ids[self.base_joint_x], self.joint_name2ids[self.base_joint_y]],
            controlMode=p.POSITION_CONTROL,
            targetPositions=[target_x, target_y],
            forces=[base_force_xy, base_force_xy],
        )
        p.setJointMotorControl2(
            self.robot,
            self.joint_name2ids[self.base_joint_yaw],
            p.POSITION_CONTROL,
            targetPosition=target_theta,
            force=50,
        )

        for jn, target in arm_targets.items():
            force = float(self.joint_effort_limit.get(jn, 0.0))
            if force <= 0.0:
                force = 5.0
            p.setJointMotorControl2(
                self.robot, self.joint_name2ids[jn], p.POSITION_CONTROL, targetPosition=target, force=force
            )

        # hold stabilized joints (e.g. gimbal) fixed at 0.0
        for jn in self.stabilized_joints:
            if jn not in self.joint_name2ids:
                continue
            p.setJointMotorControl2(
                self.robot, self.joint_name2ids[jn], p.POSITION_CONTROL, targetPosition=0.0, force=5.0
            )

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

        cur_arm_state_by_name = {
            jn: float(p.getJointState(self.robot, self.joint_name2ids[jn])[0])
            for jn in self.arm_joints
            if jn in self.joint_name2ids
        }
        self.control_info["reality"] = {
            "dx": float(base_pos[0] - last_base_pos[0]),
            "dy": float(base_pos[1] - last_base_pos[1]),
            "dtheta": float(self._wrap_angle(base_yaw - last_yaw)),
        }
        for jn, cur_v in cur_arm_state_by_name.items():
            last_v = last_arm_state_by_name.get(jn, cur_v)
            self.control_info["reality"][jn] = float(cur_v - last_v)
        
        reward, info = compute_reward(base_pos, base_lin_vel_norm, ee_pos, self._target, base_yaw,
                                    wrap_angle_fn=self._wrap_angle,
                                    remaining_ratio=self.remain_traj_ratio,
                                    **self._reward_kwargs)
        info.update(
            {
                "traj_id": int(self._traj_id),
                "remain_traj_ratio": float(self.remain_traj_ratio),
                "base_yaw": float(base_yaw),
                "base_vx": float(base_lin_vel_chassis_frame[0]),
                "base_vy": float(base_lin_vel_chassis_frame[1]),
                "base_wz": float(base_ang_vel[2]) if base_ang_vel is not None and len(base_ang_vel) > 2 else 0.0,
                "base_lin_vel_norm": float(base_lin_vel_norm),
                "base_ang_vel_norm": float(base_ang_vel_norm),
            }
        )

        # terminated: 成功或者失败的自然终止
        static_arm_thr = 0.02
        qdot = None
        try:
            qdot = self._joint_states[self.n_j :]
        except Exception:
            qdot = None
        is_arm_joints_static = bool(qdot is not None and np.all(np.abs(qdot) < static_arm_thr))
        

        # if not self.finish_step1_step and bool(info.get("ee_distance", 9999.0) < 0.05):
        #     self.finish_step1_step = self.step_count

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
        if terminated or truncated:
            # SB3 EvalCallback expects 'is_success' to exist for all episodes (True/False),
            # otherwise it saves ragged success arrays and can crash.
            info['is_success'] = bool(reached_goal)

        
        if DEBUG:
            print(f"Step {self.step_count}: finish_step1_step = {self.finish_step1_step}, Distance: {info['ee_distance']:.1f}m, "
                f"Target: ({self._target[0]:.3f}, {self._target[1]:.3f}, {self._target[2]:.3f})m, "
                f"base_pos = ({base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f}), "
                f"ee_pos = ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}), "
                f"Reward: {reward:.2f} \n")

        self.last_chassis_vel = base_lin_vel_chassis_frame[0]
        if self._last_traj_id > self._traj_id:
            import pdb; pdb.set_trace()
        self._last_traj_id = self._traj_id
        # import pdb; pdb.set_trace()
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        # GUI mode is handled by PyBullet GUI; nothing to return unless rgb_array requested.
        pass

    def close(self):
        if p.isConnected():
            p.disconnect()

    def set_curriculum_stage2_prob(self, stage2_prob: float):
        """Update curriculum mixing prob inside the env target generator (if supported)."""
        try:
            setter = getattr(self._target_generator, "set_stage2_prob", None)
            if callable(setter):
                setter(stage2_prob)
        except Exception:
            pass
