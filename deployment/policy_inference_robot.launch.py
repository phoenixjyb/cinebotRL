#!/usr/bin/env python3
"""
Launch file for policy inference node - Robot Interface Compatible.
Updated to match actual robot ROS2 topics.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for policy inference node."""
    
    # Declare launch arguments
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='models/policy_demo.onnx',
        description='Path to ONNX model file'
    )
    
    stats_path_arg = DeclareLaunchArgument(
        'stats_path',
        default_value='models/normalization_stats.npz',
        description='Path to normalization stats file'
    )
    
    control_frequency_arg = DeclareLaunchArgument(
        'control_frequency',
        default_value='20.0',
        description='Control loop frequency in Hz'
    )
    
    base_linear_scale_arg = DeclareLaunchArgument(
        'base_linear_scale',
        default_value='1.5',
        description='Base linear velocity scale (m/s)'
    )
    
    base_angular_scale_arg = DeclareLaunchArgument(
        'base_angular_scale',
        default_value='2.0',
        description='Base angular velocity scale (rad/s)'
    )
    
    arm_velocity_scale_arg = DeclareLaunchArgument(
        'arm_velocity_scale',
        default_value='1.0',
        description='Arm joint velocity scale (rad/s)'
    )
    
    # Policy inference node
    policy_node = Node(
        package='cinebot_control',
        executable='policy_inference',
        name='policy_inference_node',
        output='screen',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'stats_path': LaunchConfiguration('stats_path'),
            'control_frequency': LaunchConfiguration('control_frequency'),
            'base_linear_scale': LaunchConfiguration('base_linear_scale'),
            'base_angular_scale': LaunchConfiguration('base_angular_scale'),
            'arm_velocity_scale': LaunchConfiguration('arm_velocity_scale'),
        }],
        # Robot-specific topic remapping (if needed)
        remappings=[
            # Input topics (from robot)
            ('/hdas/feedback_arm_left', '/hdas/feedback_arm_left'),
            ('/odom_wheel', '/odom_wheel'),
            ('/camera_target_pose', '/camera_target_pose'),
            # Output topics (to robot)
            ('/motion_target/target_joint_state_arm_left', '/motion_target/target_joint_state_arm_left'),
            ('/mobile_base/commands/velocity', '/mobile_base/commands/velocity'),
        ]
    )
    
    return LaunchDescription([
        model_path_arg,
        stats_path_arg,
        control_frequency_arg,
        base_linear_scale_arg,
        base_angular_scale_arg,
        arm_velocity_scale_arg,
        policy_node,
    ])
