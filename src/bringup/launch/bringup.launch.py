"""
File: bringup.launch.py

Purpose:
This launch file starts all core ROS2 nodes required for the humanoid robot
human–robot interaction pipeline, including perception, speech, and LLM-based
dialog components.

The launch configuration enables:
- Face detection and authentication
- Automatic speech recognition (ASR)
- LLM-based dialogue server
- Text-to-speech (TTS) output

Structure:
- generate_launch_description function:
  - Creates and returns a LaunchDescription instance
  - Defines and launches all required ROS2 nodes
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """
    Generate the ROS2 launch description for the full interaction system.

    Purpose:
    - Instantiate and configure all ROS2 nodes needed for the system
    - Group perception, speech, and LLM components in a single launch file

    Inputs:
    - None

    Outputs:
    - LaunchDescription: Launch configuration containing all nodes
    """

    return LaunchDescription([

        # =========================
        # Face recognition / authentication
        # =========================
        Node(
            package='ainex_vision',
            executable='face_detection_node',
            name='face_detection_node',
            output='screen'
        ),

        # =========================
        # Automatic Speech Recognition (Whisper + VAD)
        # =========================
        Node(
            package='speech_interface',
            executable='asr_node',
            name='asr_node',
            output='screen'
        ),

        # =========================
        # LLM Server Node
        # =========================
        Node(
            package='llm_interface',
            executable='llm_server_node',
            name='llm_server_node',
            output='screen'
        ),

        # =========================
        # Text-to-Speech (Piper)
        # =========================
        Node(
            package='speech_interface',
            executable='tts_node',
            name='tts_node',
            output='screen'
        ),
    ])

