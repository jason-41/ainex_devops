from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    # 1. Undistortion node
    undistortion_node = Node(
        package='vision',
        executable='undistortion',
        name='undistortion_node',
        output='screen'
    )

    # 2. ArUco marker detector
    aruco_detector_node = Node(
        package='vision',
        executable='aruco_detector_node',
        name='aruco_detector_node',
        output='screen'
    )

    return LaunchDescription([
        undistortion_node,
        aruco_detector_node,
    ])
