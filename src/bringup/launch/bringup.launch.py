from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    return LaunchDescription([

        # =========================
        # Face recognition / auth
        # =========================
        Node(
            package='ainex_vision',
            executable='face_detection_node',
            name='face_detection_node',
            output='screen'
        ),

        # =========================
        # ASR (Whisper + VAD)
        # =========================
        Node(
            package='speech_interface',
            executable='asr_node',
            name='asr_node',
            output='screen'
        ),

        # =========================
        # LLM Server
        # =========================
        Node(
            package='llm_interface',
            executable='llm_server_node',
            name='llm_server_node',
            output='screen'
        ),

        # =========================
        # TTS (Piper)
        # =========================
        Node(
            package='speech_interface',
            executable='tts_node',
            name='tts_node',
            output='screen'
        ),
    ])
