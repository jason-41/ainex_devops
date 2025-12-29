from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    llm_server = Node(
        package="llm_interface",
        executable="llm_server_node",
        name="llm_server_node",
        output="screen"
    )

    # llm_cli = Node(
    #     package="llm_interface",
    #     executable="llm_cli_node",
    #     name="llm_cli_node",
    #     output="screen",
    #     #prefix="xterm -e"  # optional: open CLI in new terminal
    # )

    return LaunchDescription([
        llm_server,
        # llm_cli
    ])
