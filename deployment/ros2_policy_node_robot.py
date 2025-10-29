#!/usr/bin/env python3
"""
ROS2 node for running trained policy inference on NVIDIA Orin.
Updated to match actual robot ROS2 interface.

Subscribes to:
  - /hdas/feedback_arm_left (sensor_msgs/JointState) - Current arm joint states (6 DOF)
  - /camera_target_pose (geometry_msgs/PoseStamped) - Target EE pose from trajectory
  - /odom_wheel (nav_msgs/Odometry) - Mobile base odometry

Publishes:
  - /motion_target/target_joint_state_arm_left (sensor_msgs/JointState) - Arm joint positions
  - /mobile_base/commands/velocity (geometry_msgs/Twist) - Base velocity commands

Installation on Orin:
    sudo apt install python3-onnxruntime ros-humble-nav-msgs
    pip3 install numpy
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
import numpy as np
import onnxruntime as ort
from pathlib import Path
from collections import deque


class PolicyInferenceNode(Node):
    """ROS2 node for trained RL policy inference - Robot Interface Compatible."""
    
    def __init__(self):
        super().__init__('policy_inference_node')
        
        # Parameters
        self.declare_parameter('model_path', 'models/policy_demo.onnx')
        self.declare_parameter('stats_path', 'models/normalization_stats.npz')
        self.declare_parameter('control_frequency', 20.0)  # Hz (matches training)
        self.declare_parameter('base_linear_scale', 1.5)   # m/s
        self.declare_parameter('base_angular_scale', 2.0)  # rad/s
        self.declare_parameter('arm_velocity_scale', 1.0)  # rad/s
        
        # Joint names (must match robot configuration)
        # Note: Robot has 6 DOF arm, we exclude gripper for now
        self.arm_joint_names = [
            'joint1', 'joint2', 'joint3',  # Adjust to actual names
            'joint4', 'joint5', 'joint6'
        ]
        
        # Load model and normalization stats
        self.load_model()
        
        # State buffers
        self.current_arm_pos = None      # [6] arm joint positions
        self.current_arm_vel = None      # [6] arm joint velocities
        self.current_base_pos = None     # [3] base (x, y, yaw)
        self.current_base_vel = None     # [3] base velocities
        self.current_target_pose = None  # [7] target (x, y, z, qx, qy, qz, qw)
        
        # Integrated joint positions for arm (start from current, update with velocities)
        self.integrated_arm_pos = None
        
        # Previous time for dt calculation
        self.last_control_time = None
        
        # Subscribers (using robot's actual topics)
        self.arm_feedback_sub = self.create_subscription(
            JointState,
            '/hdas/feedback_arm_left',
            self.arm_feedback_callback,
            10
        )
        
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom_wheel',
            self.odom_callback,
            10
        )
        
        self.target_sub = self.create_subscription(
            PoseStamped,
            '/camera_target_pose',
            self.target_callback,
            10
        )
        
        # Publishers (using robot's actual topics)
        self.arm_cmd_pub = self.create_publisher(
            JointState,
            '/motion_target/target_joint_state_arm_left',
            10
        )
        
        self.base_cmd_pub = self.create_publisher(
            Twist,
            '/mobile_base/commands/velocity',
            10
        )
        
        # Control loop timer
        control_period = 1.0 / self.get_parameter('control_frequency').value
        self.timer = self.create_timer(control_period, self.control_loop)
        
        self.get_logger().info('=' * 60)
        self.get_logger().info('Policy Inference Node - Robot Interface')
        self.get_logger().info('=' * 60)
        self.get_logger().info(f'Model: {self.get_parameter("model_path").value}')
        self.get_logger().info(f'Control frequency: {self.get_parameter("control_frequency").value} Hz')
        self.get_logger().info(f'Inference provider: {self.ort_session.get_providers()[0]}')
        self.get_logger().info('')
        self.get_logger().info('Subscribed Topics:')
        self.get_logger().info('  - /hdas/feedback_arm_left (sensor_msgs/JointState)')
        self.get_logger().info('  - /odom_wheel (nav_msgs/Odometry)')
        self.get_logger().info('  - /camera_target_pose (geometry_msgs/PoseStamped)')
        self.get_logger().info('')
        self.get_logger().info('Publishing Topics:')
        self.get_logger().info('  - /motion_target/target_joint_state_arm_left (sensor_msgs/JointState)')
        self.get_logger().info('  - /mobile_base/commands/velocity (geometry_msgs/Twist)')
        self.get_logger().info('=' * 60)
    
    def load_model(self):
        """Load ONNX model and normalization statistics."""
        model_path = self.get_parameter('model_path').value
        stats_path = self.get_parameter('stats_path').value
        
        self.get_logger().info(f'Loading ONNX model: {model_path}')
        
        # Load ONNX model (prefer GPU, fallback to CPU)
        providers = ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CPUExecutionProvider']
        self.ort_session = ort.InferenceSession(model_path, providers=providers)
        
        # Load normalization statistics
        self.get_logger().info(f'Loading normalization stats: {stats_path}')
        stats = np.load(stats_path)
        self.obs_mean = stats['obs_mean']
        self.obs_var = stats['obs_var']
        self.obs_std = np.sqrt(self.obs_var + 1e-8)
        
        self.get_logger().info('Model and stats loaded successfully')
    
    def arm_feedback_callback(self, msg: JointState):
        """Callback for arm joint state feedback."""
        # Extract positions and velocities for the 6 arm joints
        # Robot publishes float64[] with 6 joint positions
        if len(msg.position) >= 6:
            self.current_arm_pos = np.array(msg.position[:6], dtype=np.float32)
            
            # Initialize integrated positions on first message
            if self.integrated_arm_pos is None:
                self.integrated_arm_pos = self.current_arm_pos.copy()
            
            # Use velocity if provided, otherwise estimate
            if len(msg.velocity) >= 6:
                self.current_arm_vel = np.array(msg.velocity[:6], dtype=np.float32)
            else:
                self.current_arm_vel = np.zeros(6, dtype=np.float32)
    
    def odom_callback(self, msg: Odometry):
        """Callback for base odometry (nav_msgs/Odometry)."""
        # Extract base position (x, y) and orientation (yaw)
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        # Convert quaternion to yaw
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        
        # Calculate yaw from quaternion
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        self.current_base_pos = np.array([x, y, yaw], dtype=np.float32)
        
        # Extract base velocities
        linear_x = msg.twist.twist.linear.x
        linear_y = msg.twist.twist.linear.y
        angular_z = msg.twist.twist.angular.z
        
        self.current_base_vel = np.array([linear_x, linear_y, angular_z], dtype=np.float32)
    
    def target_callback(self, msg: PoseStamped):
        """Callback for target camera pose."""
        # Extract 7D target pose (x, y, z, qx, qy, qz, qw)
        self.current_target_pose = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ], dtype=np.float32)
    
    def build_observation(self) -> np.ndarray:
        """Build 74-dimensional observation vector.
        
        Observation structure (matches training):
        - Joint positions (9): 3 base virtual joints + 6 arm joints
        - Joint velocities (9): 3 base velocities + 6 arm velocities
        - Base position (3): x, y, yaw
        - Target pose (7): x, y, z, qx, qy, qz, qw
        - Target relative to EE (7): relative pose
        - Previous action (8): last commanded action
        - Time/phase info: Additional features
        
        Total: 9 + 9 + 3 + 7 + 7 + 8 + ... = 74 dimensions
        """
        # For simplicity, we'll build a basic observation
        # You may need to adjust based on exact training observation structure
        
        # Joint positions: base virtual joints (set to base x, y, yaw) + arm joints
        joint_positions = np.concatenate([
            self.current_base_pos,      # [3] base x, y, yaw
            self.current_arm_pos        # [6] arm joints
        ])  # Total: 9
        
        # Joint velocities: base velocities + arm velocities
        joint_velocities = np.concatenate([
            self.current_base_vel,      # [3] base vx, vy, vyaw
            self.current_arm_vel        # [6] arm joint velocities
        ])  # Total: 9
        
        # Base position (redundant but matches training)
        base_pos = self.current_base_pos  # [3]
        
        # Target pose
        target_pose = self.current_target_pose  # [7]
        
        # Target relative to end-effector (simplified - would need FK in real implementation)
        # For now, use target pose directly (you'll need to compute relative transform)
        target_relative = self.current_target_pose  # [7] - PLACEHOLDER
        
        # Previous action (initialize to zeros, update in control loop)
        if not hasattr(self, 'prev_action'):
            self.prev_action = np.zeros(8, dtype=np.float32)
        
        # Additional features to reach 74 dims
        # This is a placeholder - adjust based on your actual observation structure
        additional = np.zeros(74 - 9 - 9 - 3 - 7 - 7 - 8, dtype=np.float32)
        
        # Concatenate all components
        obs = np.concatenate([
            joint_positions,    # 9
            joint_velocities,   # 9
            base_pos,          # 3
            target_pose,       # 7
            target_relative,   # 7
            self.prev_action,  # 8
            additional         # Remaining to reach 74
        ])
        
        assert obs.shape[0] == 74, f"Observation dimension mismatch: {obs.shape[0]} != 74"
        
        return obs
    
    def control_loop(self):
        """Main control loop - runs at specified frequency."""
        # Check if we have all required data
        if (self.current_arm_pos is None or 
            self.current_base_pos is None or 
            self.current_target_pose is None):
            return
        
        # Build observation vector
        obs = self.build_observation()
        
        # Normalize observation
        normalized_obs = (obs - self.obs_mean) / self.obs_std
        
        # Run inference
        input_name = self.ort_session.get_inputs()[0].name
        actions = self.ort_session.run(
            None, 
            {input_name: normalized_obs.reshape(1, -1).astype(np.float32)}
        )[0][0]  # Shape: [8]
        
        # Store for next observation
        self.prev_action = actions.copy()
        
        # Split actions: [0:3] base velocities, [3:9] arm velocities (we only have 6, so [3:9])
        base_actions = actions[:3]   # [3] (vx, vy, vyaw)
        arm_actions = actions[3:]    # [5] or [6] - adjust based on your action dim
        
        # If action dim is 8 and we have 6 arm joints, last 2 might be for gripper or padding
        # Take only first 6 for arm
        if len(arm_actions) > 6:
            arm_actions = arm_actions[:6]
        
        # Scale actions
        base_linear_scale = self.get_parameter('base_linear_scale').value
        base_angular_scale = self.get_parameter('base_angular_scale').value
        arm_velocity_scale = self.get_parameter('arm_velocity_scale').value
        
        # Publish base velocity commands
        base_cmd = Twist()
        base_cmd.linear.x = float(base_actions[0] * base_linear_scale)
        base_cmd.linear.y = float(base_actions[1] * base_linear_scale)
        base_cmd.angular.z = float(base_actions[2] * base_angular_scale)
        self.base_cmd_pub.publish(base_cmd)
        
        # Integrate arm velocities to positions
        current_time = self.get_clock().now()
        if self.last_control_time is not None:
            dt = (current_time - self.last_control_time).nanoseconds / 1e9
        else:
            dt = 1.0 / self.get_parameter('control_frequency').value
        
        self.last_control_time = current_time
        
        # Update integrated positions
        arm_velocities = arm_actions * arm_velocity_scale
        self.integrated_arm_pos += arm_velocities * dt
        
        # Publish arm position commands (robot expects positions, not velocities)
        arm_cmd = JointState()
        arm_cmd.header.stamp = self.get_clock().now().to_msg()
        arm_cmd.name = self.arm_joint_names
        arm_cmd.position = self.integrated_arm_pos.tolist()
        self.arm_cmd_pub.publish(arm_cmd)


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
