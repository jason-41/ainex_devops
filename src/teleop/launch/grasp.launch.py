from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # General
        DeclareLaunchArgument('sim', default_value='False', description='Simulation mode'),
        DeclareLaunchArgument('use_camera', default_value='True', description='Use camera for pose estimation'),
        DeclareLaunchArgument('pose_topic', default_value='detected_object_pose', description='Topic for object pose'),
        DeclareLaunchArgument('lock_target_once', default_value='False', description='Lock target once detected'),

        # Hardcoded Camera (only used if use_camera is False)
        # Note: Arrays are harder to declutter as args, relying on defaults in code or yaml if needed.
        # But we can pass them if the user uses a param file.

        # Offsets
        DeclareLaunchArgument('pre_x_off', default_value='0.02', description='Pre-grasp X offset'),
        DeclareLaunchArgument('pre_z_off', default_value='0.015', description='Pre-grasp Z offset'),
        DeclareLaunchArgument('approach_x_off', default_value='0.0', description='Approach X offset'),
        DeclareLaunchArgument('lift_z', default_value='0.10', description='Lift Z height'),

        # Durations
        DeclareLaunchArgument('pre_duration', default_value='3.0', description='Duration for pre-grasp phase'),
        DeclareLaunchArgument('approach_duration', default_value='3.0', description='Duration for approach phase'),
        DeclareLaunchArgument('lift_duration', default_value='3.0', description='Duration for lift phase'),
        
        # Control
        DeclareLaunchArgument('dt_cmd', default_value='0.05', description='Control loop period'),
        DeclareLaunchArgument('feedback_hz', default_value='0.0', description='Feedback frequency'),

        # Gripper
        DeclareLaunchArgument('close_fraction', default_value='0.4', description='Gripper close fraction'),
        DeclareLaunchArgument('gripper_kp', default_value='6.0', description='Gripper P gain'),
        DeclareLaunchArgument('gripper_vel_max', default_value='2.0', description='Gripper max velocity'),
        DeclareLaunchArgument('gripper_eps', default_value='0.02', description='Gripper epsilon'),
        DeclareLaunchArgument('squeeze_time', default_value='0.5', description='Squeeze time'),
        DeclareLaunchArgument('phase_settle_s', default_value='0.25', description='Settle time between phases'),

        Node(
            package='teleop',
            executable='grasp',
            name='ainex_grasp_node',
            output='screen',
            parameters=[{
                'sim': LaunchConfiguration('sim'),
                'use_camera': LaunchConfiguration('use_camera'),
                'pose_topic': LaunchConfiguration('pose_topic'),
                'lock_target_once': LaunchConfiguration('lock_target_once'),
                
                'pre_x_off': LaunchConfiguration('pre_x_off'),
                'pre_z_off': LaunchConfiguration('pre_z_off'),
                'approach_x_off': LaunchConfiguration('approach_x_off'),
                'lift_z': LaunchConfiguration('lift_z'),
                
                'pre_duration': LaunchConfiguration('pre_duration'),
                'approach_duration': LaunchConfiguration('approach_duration'),
                'lift_duration': LaunchConfiguration('lift_duration'),
                
                'dt_cmd': LaunchConfiguration('dt_cmd'),
                'feedback_hz': LaunchConfiguration('feedback_hz'),

                'close_fraction': LaunchConfiguration('close_fraction'),
                'gripper_kp': LaunchConfiguration('gripper_kp'),
                'gripper_vel_max': LaunchConfiguration('gripper_vel_max'),
                'gripper_eps': LaunchConfiguration('gripper_eps'),
                'squeeze_time': LaunchConfiguration('squeeze_time'),
                'phase_settle_s': LaunchConfiguration('phase_settle_s'),
            }]
        )
    ])
