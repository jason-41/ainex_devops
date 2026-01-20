# src/vision/launch/object_pose_estimation.launch.py

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('vision')
    default_params = os.path.join(pkg_share, 'config', 'object_pose_estimator_params.yaml')

    image_topic = LaunchConfiguration('image_topic')
    params_file = LaunchConfiguration('params_file')

    return LaunchDescription([
        DeclareLaunchArgument('image_topic', default_value='/camera/color/image_raw'),
        DeclareLaunchArgument('params_file', default_value=default_params),

        Node(
            package='vision',
            executable='object_pose_estimator',
            name='object_pose_estimator',
            output='screen',
            parameters=[
                {'image_topic': image_topic},
                params_file,
            ]
        ),
    ])
