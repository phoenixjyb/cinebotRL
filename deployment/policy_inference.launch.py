"""
ROS2 launch file for policy inference on NVIDIA Orin.

Usage:
    ros2 launch cinebot_control policy_inference.launch.py \
        model_path:=/path/to/policy_session_7d.onnx \
        stats_path:=/path/to/normalization_stats.npz
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    
    # Declare launch arguments
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='policy_session_7d.onnx',
        description='Path to ONNX model file'
    )
    
    stats_path_arg = DeclareLaunchArgument(
        'stats_path',
        default_value='normalization_stats.npz',
        description='Path to normalization statistics'
    )
    
    control_freq_arg = DeclareLaunchArgument(
        'control_frequency',
        default_value='20.0',
        description='Control loop frequency in Hz'
    )
    
    # Policy inference node
    policy_node = Node(
        package='cinebot_control',
        executable='policy_inference_node.py',
        name='policy_inference',
        output='screen',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'stats_path': LaunchConfiguration('stats_path'),
            'control_frequency': LaunchConfiguration('control_frequency'),
            'base_joints': ['joint_x', 'joint_y', 'joint_theta'],
            'arm_joints': [
                'left_arm_joint1', 'left_arm_joint2', 'left_arm_joint3',
                'left_arm_joint4', 'left_arm_joint5', 'left_arm_joint6'
            ]
        }],
        remappings=[
            ('/joint_states', '/mobile_mm/joint_states'),
            ('/camera_target_pose', '/trajectory/target_pose'),
            ('/base_pose', '/mobile_mm/base_pose'),
            ('/joint_commands', '/mobile_mm/joint_commands')
        ]
    )
    
    return LaunchDescription([
        model_path_arg,
        stats_path_arg,
        control_freq_arg,
        policy_node
    ])
