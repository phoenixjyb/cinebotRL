#!/usr/bin/env python3
"""
ROS2 node for running trained policy inference on NVIDIA Orin.

Subscribes to:
  - /joint_states (sensor_msgs/JointState) - Current robot joint positions/velocities
  - /camera_target_pose (geometry_msgs/PoseStamped) - Target EE pose from trajectory
  - /base_pose (geometry_msgs/PoseStamped) - Current base pose

Publishes:
  - /joint_commands (trajectory_msgs/JointTrajectory) - Commanded joint positions

Installation on Orin:
    sudo apt install python3-onnxruntime  # or pip3 install onnxruntime-gpu
    pip3 install numpy
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np
import onnxruntime as ort
from pathlib import Path
from collections import deque
import tf_transformations as tf


class PolicyInferenceNode(Node):
    """ROS2 node for trained RL policy inference."""
    
    def __init__(self):
        super().__init__('policy_inference_node')
        
        # Parameters
        self.declare_parameter('model_path', 'policy_session_7d.onnx')
        self.declare_parameter('stats_path', 'normalization_stats.npz')
        self.declare_parameter('control_frequency', 20.0)  # Hz (matches training)
        self.declare_parameter('base_joints', ['joint_x', 'joint_y', 'joint_theta'])
        self.declare_parameter('arm_joints', [
            'left_arm_joint1', 'left_arm_joint2', 'left_arm_joint3',
            'left_arm_joint4', 'left_arm_joint5', 'left_arm_joint6'
        ])
        
        # Load model and normalization stats
        self.load_model()
        
        # State buffers (for velocity estimation)
        self.joint_pos_history = deque(maxlen=2)
        self.joint_vel_history = deque(maxlen=2)
        self.base_pos_history = deque(maxlen=2)
        
        # Current state
        self.current_joint_pos = None
        self.current_joint_vel = None
        self.current_base_pose = None
        self.current_target_pose = None
        
        # Subscribers
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )
        self.target_sub = self.create_subscription(
            PoseStamped, '/camera_target_pose', self.target_callback, 10
        )
        self.base_sub = self.create_subscription(
            PoseStamped, '/base_pose', self.base_pose_callback, 10
        )
        
        # Publisher
        self.cmd_pub = self.create_publisher(
            JointTrajectory, '/joint_commands', 10
        )
        
        # Control loop timer
        control_period = 1.0 / self.get_parameter('control_frequency').value
        self.timer = self.create_timer(control_period, self.control_loop)
        
        self.get_logger().info('Policy inference node initialized')
        self.get_logger().info(f'  Model: {self.get_parameter("model_path").value}')
        self.get_logger().info(f'  Control freq: {self.get_parameter("control_frequency").value} Hz')
    
    def load_model(self):
        """Load ONNX model and normalization statistics."""
        model_path = self.get_parameter('model_path').value
        stats_path = self.get_parameter('stats_path').value
        
        self.get_logger().info(f'Loading ONNX model from {model_path}...')
        
        # Load ONNX model (use GPU if available)
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.ort_session = ort.InferenceSession(model_path, providers=providers)
        
        self.get_logger().info(f'  Provider: {self.ort_session.get_providers()[0]}')
        
        # Load normalization statistics
        stats = np.load(stats_path)
        self.obs_mean = stats['obs_mean']
        self.obs_std = stats['obs_std']
        
        self.get_logger().info('  Normalization stats loaded')
        self.get_logger().info(f'  Observation dim: {len(self.obs_mean)}')
    
    def joint_state_callback(self, msg: JointState):
        """Update current joint positions and velocities."""
        # Map joint names to indices
        base_joints = self.get_parameter('base_joints').value
        arm_joints = self.get_parameter('arm_joints').value
        all_joints = base_joints + arm_joints
        
        joint_pos = np.zeros(9)
        joint_vel = np.zeros(9)
        
        for i, joint_name in enumerate(all_joints):
            if joint_name in msg.name:
                idx = msg.name.index(joint_name)
                joint_pos[i] = msg.position[idx]
                if len(msg.velocity) > idx:
                    joint_vel[i] = msg.velocity[idx]
        
        self.current_joint_pos = joint_pos
        self.current_joint_vel = joint_vel
        
        # Update history for velocity estimation
        self.joint_pos_history.append(joint_pos.copy())
        self.joint_vel_history.append(joint_vel.copy())
    
    def base_pose_callback(self, msg: PoseStamped):
        """Update current base pose."""
        pos = msg.pose.position
        quat = msg.pose.orientation
        
        # Extract yaw from quaternion
        euler = tf.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
        yaw = euler[2]
        
        self.current_base_pose = np.array([pos.x, pos.y, yaw])
        self.base_pos_history.append(self.current_base_pose.copy())
    
    def target_callback(self, msg: PoseStamped):
        """Update target EE pose."""
        pos = msg.pose.position
        quat = msg.pose.orientation
        
        self.current_target_pose = np.array([
            pos.x, pos.y, pos.z,
            quat.x, quat.y, quat.z, quat.w
        ])
    
    def build_observation(self):
        """
        Build observation vector matching training format.
        
        Observation (74 dims):
          - Joint positions (9)
          - Joint velocities (9)
          - Base velocity (3)
          - Arm joint velocity (6)
          - Target position (3)
          - Target orientation (4, quaternion)
          - Relative target position (3)
          - Previous action (8)
          - Distance to target (1)
          - Alignment to target (1)
          - Joint position errors (6)
          - Joint velocity errors (6)
          - Projected distance (1)
          - Lateral offset (1)
          - Height offset (1)
          - Base-target alignment (1)
          - ... (add any other observations from your task_spec.py)
        """
        
        # Check if all required data is available
        if (self.current_joint_pos is None or 
            self.current_joint_vel is None or
            self.current_base_pose is None or
            self.current_target_pose is None):
            return None
        
        obs = []
        
        # Joint positions (9)
        obs.extend(self.current_joint_pos)
        
        # Joint velocities (9)
        obs.extend(self.current_joint_vel)
        
        # Base velocity (3) - estimated from history
        if len(self.base_pos_history) >= 2:
            base_vel = (self.base_pos_history[-1] - self.base_pos_history[-2]) * \
                       self.get_parameter('control_frequency').value
        else:
            base_vel = np.zeros(3)
        obs.extend(base_vel)
        
        # Arm joint velocity (6)
        obs.extend(self.current_joint_vel[3:])
        
        # Target position (3)
        target_pos = self.current_target_pose[:3]
        obs.extend(target_pos)
        
        # Target orientation (4, quaternion)
        target_quat = self.current_target_pose[3:]
        obs.extend(target_quat)
        
        # Relative target position (3)
        rel_target = target_pos - self.current_base_pose[:2].tolist() + [0]
        obs.extend(rel_target)
        
        # Previous action (8) - use zeros for first step
        if not hasattr(self, 'prev_action'):
            self.prev_action = np.zeros(8)
        obs.extend(self.prev_action)
        
        # Distance to target (1)
        distance = np.linalg.norm(rel_target)
        obs.append(distance)
        
        # Alignment to target (1)
        base_yaw = self.current_base_pose[2]
        target_angle = np.arctan2(rel_target[1], rel_target[0])
        alignment = np.cos(target_angle - base_yaw)
        obs.append(alignment)
        
        # TODO: Add remaining observations to match your exact obs space (74 dims)
        # This is a simplified version - adjust based on your task_spec.py
        
        # Pad to 74 dims if needed
        while len(obs) < 74:
            obs.append(0.0)
        
        return np.array(obs[:74], dtype=np.float32)
    
    def control_loop(self):
        """Main control loop - run policy inference and publish commands."""
        
        # Build observation
        obs = self.build_observation()
        if obs is None:
            return  # Wait for all data to be available
        
        # Normalize observation
        obs_normalized = (obs - self.obs_mean) / (self.obs_std + 1e-8)
        
        # Run inference
        ort_inputs = {self.ort_session.get_inputs()[0].name: obs_normalized.reshape(1, -1)}
        action = self.ort_session.run(None, ort_inputs)[0][0]
        
        # Store for next observation
        self.prev_action = action.copy()
        
        # Scale action (training uses [-1, 1], scale to actual joint limits)
        # Base velocities (first 3): scale to max velocities
        base_vx = action[0] * 1.5  # m/s
        base_vy = action[1] * 1.5  # m/s  
        base_wz = action[2] * 2.0  # rad/s
        
        # Arm joint velocities (next 6): scale to joint velocity limits
        arm_velocities = action[3:] * 1.0  # rad/s
        
        # Create JointTrajectory message
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = self.get_parameter('base_joints').value + \
                         self.get_parameter('arm_joints').value
        
        point = JointTrajectoryPoint()
        
        # Convert velocities to positions (integrate over control period)
        dt = 1.0 / self.get_parameter('control_frequency').value
        
        # Base positions (velocity control)
        base_pos_cmd = self.current_base_pose.copy()
        base_pos_cmd[0] += base_vx * dt * np.cos(base_pos_cmd[2]) - base_vy * dt * np.sin(base_pos_cmd[2])
        base_pos_cmd[1] += base_vx * dt * np.sin(base_pos_cmd[2]) + base_vy * dt * np.cos(base_pos_cmd[2])
        base_pos_cmd[2] += base_wz * dt
        
        # Arm positions (velocity control)
        arm_pos_cmd = self.current_joint_pos[3:] + arm_velocities * dt
        
        point.positions = base_pos_cmd.tolist() + arm_pos_cmd.tolist()
        point.velocities = [base_vx, base_vy, base_wz] + arm_velocities.tolist()
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = int(dt * 1e9)
        
        msg.points = [point]
        
        # Publish
        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PolicyInferenceNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
