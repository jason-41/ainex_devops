"""Launch Gazebo Harmonic with the grasp world and spawn ainex from URDF.

Usage (after `colcon build` and sourcing the workspace):
    ros2 launch /path/to/rl_grasp_gazebo/launch/spawn.launch.py
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription, ExecuteProcess, AppendEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import Command
from launch_ros.parameter_descriptions import ParameterValue


PKG_AINEX_DESCRIPTION = "ainex_description"
HERE = os.path.dirname(os.path.abspath(__file__))
RL_GAZEBO_ROOT = os.path.dirname(HERE)
URDF_XACRO = os.path.join(RL_GAZEBO_ROOT, "urdf", "ainex_gz.urdf.xacro")
WORLD_SDF = os.path.join(RL_GAZEBO_ROOT, "worlds", "grasp.sdf")


def generate_launch_description():
    pkg_ros_gz_sim = get_package_share_directory("ros_gz_sim")
    pkg_ainex_description = get_package_share_directory("ainex_description")
    # parent of `ainex_description/` so `model://ainex_description/meshes/...`
    # resolves to `<this>/ainex_description/meshes/...`
    ainex_share_parent = os.path.dirname(pkg_ainex_description)

    set_resource_path = AppendEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=ainex_share_parent,
    )

    # gz sim with our world
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={"gz_args": f"-r -v 4 {WORLD_SDF}"}.items(),
    )

    # robot_state_publisher: publishes /robot_description from xacro output
    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{
            "robot_description": ParameterValue(
                Command(["xacro ", URDF_XACRO]), value_type=str,
            ),
            "use_sim_time": True,
        }],
    )

    # spawn entity into running gz_sim by reading /robot_description topic
    spawn = Node(
        package="ros_gz_sim",
        executable="create",
        name="ainex_spawn",
        arguments=[
            "-name", "ainex",
            "-topic", "/robot_description",
            # base_link height is set by the world_to_base joint origin in URDF
        ],
        output="screen",
    )

    # bridge: gz <-> ros2 for clock and joint_states (handy for debugging)
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="gz_bridge",
        output="screen",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/world/grasp_world/model/ainex/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model",
        ],
    )

    return LaunchDescription([set_resource_path, gz_sim, rsp, spawn, bridge])
