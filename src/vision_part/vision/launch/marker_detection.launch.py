# src/vision/launch/marker_detection.launch.py

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('vision')

    # 默认参数文件（你自己在 config 里放 camera_calibration.yaml 也行）
    default_calib = os.path.join(pkg_share, 'config', 'camera_calibration.yaml')

    use_compressed = LaunchConfiguration('use_compressed')
    image_topic = LaunchConfiguration('image_topic')
    camera_frame = LaunchConfiguration('camera_frame')
    marker_size = LaunchConfiguration('marker_size')
    publish_tf = LaunchConfiguration('publish_tf')
    publish_vis = LaunchConfiguration('publish_vis')

    # 可选：同时开 search_marker
    start_search = LaunchConfiguration('start_search')
    target_id = LaunchConfiguration('target_id')
    marker_topic = LaunchConfiguration('marker_topic')

    return LaunchDescription([
        DeclareLaunchArgument('use_compressed', default_value='true'),
        DeclareLaunchArgument('image_topic', default_value='/camera_image/compressed'),
        DeclareLaunchArgument('camera_frame', default_value='camera_optical_link'),
        DeclareLaunchArgument('marker_size', default_value='0.0485'),
        DeclareLaunchArgument('publish_tf', default_value='true'),
        DeclareLaunchArgument('publish_vis', default_value='true'),

        DeclareLaunchArgument('start_search', default_value='false'),
        DeclareLaunchArgument('target_id', default_value='1'),
        DeclareLaunchArgument('marker_topic', default_value='/aruco_markers'),

        # 1) ArUco detector（你 nodes/aruco_detector_node.py）
        Node(
            package='vision',
            executable='aruco_detector',   # 注意：这个名字必须和 setup.py 的 console_scripts 对上
            name='aruco_detector',
            output='screen',
            parameters=[{
                'use_compressed': use_compressed,
                'image_topic': image_topic,
                'camera_frame': camera_frame,
                'marker_size': marker_size,
                'publish_tf': publish_tf,
                'publish_vis': publish_vis,
            }]
        ),

        # 2) search_marker（可选开）
        Node(
            package='vision',
            executable='search_marker',
            name='search_marker',
            output='screen',
            condition=None,  # 我下面给你更稳的写法：用 launch argument 控制是否启动（见备注）
            parameters=[{
                'target_id': target_id,
                'topic_markers': marker_topic,
                'camera_frame': camera_frame,
                'publish_tf': 'true',
            }]
        ),
    ])
