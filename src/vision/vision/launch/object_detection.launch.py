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

    # 2. Object detector (shape / color detection)
    object_detector_node = Node(
        package='vision',
        executable='object_detector',
        name='object_detector_node',
        output='screen'
    )

    # 3. Target object detector (logic / decision layer)
    target_object_detector_node = Node(
        package='vision',
        executable='target_object_detector_node',
        name='target_object_detector_node',
        output='screen'
    )

    return LaunchDescription([
        undistortion_node,
        object_detector_node,
        target_object_detector_node,
    ])
