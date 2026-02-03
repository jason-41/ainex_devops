from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, TimerAction
import time

def generate_launch_description():
    # Deactivate walking first
    deactivate_walking = ExecuteProcess(
        cmd=['ros2', 'service', 'call', '/deactivate_walking', 'std_srvs/srv/Empty', '{}'],
        output='screen'
    )
    
    # Activate walking after a short delay
    activate_walking = TimerAction(
        period=1.0,
        actions=[
            ExecuteProcess(
                cmd=['ros2', 'service', 'call', '/activate_walking', 'std_srvs/srv/Empty', '{}'],
                output='screen'
            )
        ]
    )
    
    # Start main control node after service calls
    main_control_node = TimerAction(
        period=2.0,
        actions=[
            Node(
                package='teleop',
                executable='main_control',
                name='main_control_node',
                output='screen',
                emulate_tty=True
            )
        ]
    )
    
    return LaunchDescription([
        deactivate_walking,
        activate_walking,
        main_control_node
    ])
