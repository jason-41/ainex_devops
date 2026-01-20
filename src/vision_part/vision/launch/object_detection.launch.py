# src/vision/launch/object_detection.launch.py

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('vision')
    default_hsv = os.path.join(pkg_share, 'config', 'hsv_thresholds.yaml')

    image_topic = LaunchConfiguration('image_topic')
    camera_frame = LaunchConfiguration('camera_frame')
    hsv_file = LaunchConfiguration('hsv_file')

    return LaunchDescription([
        DeclareLaunchArgument('image_topic', default_value='/camera/color/image_raw'),
        DeclareLaunchArgument('camera_frame', default_value='camera_optical_link'),
        DeclareLaunchArgument('hsv_file', default_value=default_hsv),

        Node(
            package='vision',
            executable='detect_object',
            name='detect_object',
            output='screen',
            parameters=[
                {'image_topic': image_topic, 'camera_frame': camera_frame},
                hsv_file,  # yaml 参数（如果你的 detect_object_node 里 declare_parameter 对上就能读）
            ]
        ),
    ])
