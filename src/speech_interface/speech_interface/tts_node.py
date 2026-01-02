#!/usr/bin/env python3
"""
File: tts_node.py

Purpose:
This file implements a ROS2 Text-to-Speech (TTS) node based on the offline
Piper TTS engine. The node subscribes to text output from the dialogue system,
synthesizes speech audio, and plays it through the system audio device.

The node also publishes its activity state so that other components, such as
the ASR node, can temporarily pause listening while speech is being played.

Structure:
- TTSNode class:
  - Initializes Piper TTS configuration
  - Subscribes to text output messages
  - Generates and plays speech audio in a non-blocking manner
  - Publishes TTS activity state
- main function:
  - Initializes and spins the ROS2 node
"""

import subprocess
import tempfile
import os
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Bool


class TTSNode(Node):
    """
    ROS2 node providing offline text-to-speech functionality using Piper.
    """

    def __init__(self):
        """
        Initialize the TTS node.

        Purpose:
        - Configure the Piper TTS model and audio playback command
        - Initialize ROS2 subscriptions and publishers
        - Prepare synchronization primitives for safe audio playback

        Inputs:
        - None

        Outputs:
        - None
        """
        super().__init__('tts_node')

        # =========================
        # Piper model configuration
        # =========================

        # Path to the Piper TTS model file
        # TODO:This path should be adapted to the local system configuration
        self.model_path = (
            "/home/wbo/humanoid_robotics_system/hrs_groupE/src/speech_interface/models/tts/piper/en_US-amy-medium.onnx"
        )

        # Audio playback command (Linux)
        self.audio_player = "aplay"

        # Lock to prevent overlapping audio playback
        self.lock = threading.Lock()

        # =========================
        # ROS subscription
        # =========================

        # Subscribe to text output for speech synthesis
        self.sub = self.create_subscription(
            String,
            '/speech/text_output',
            self.on_text,
            10
        )

        # Publisher indicating whether TTS is currently active
        # Used by ASR to mute input during speech playback
        self.tts_state_pub = self.create_publisher(
            Bool,
            "/speech/tts_active",
            10
        )

        self.get_logger().info("Piper TTS node started.")

    def on_text(self, msg: String):
        """
        Handle incoming text messages for speech synthesis.

        Purpose:
        - Receive text output from the dialogue system
        - Spawn a separate thread to perform speech synthesis
        - Avoid blocking the ROS callback thread

        Inputs:
        - msg (String): Text message to be spoken

        Outputs:
        - None
        """
        text = msg.data.strip()
        if not text:
            return

        self.get_logger().info(f"[TTS] {text}")

        # Use a separate thread to avoid blocking ROS callbacks
        threading.Thread(
            target=self.speak,
            args=(text,),
            daemon=True
        ).start()

    def speak(self, text: str):
        """
        Generate and play speech audio for the given text.

        Purpose:
        - Invoke the Piper TTS engine to synthesize speech
        - Play the generated audio file
        - Publish TTS activity state for synchronization with ASR

        Inputs:
        - text (str): Text to be synthesized and spoken

        Outputs:
        - None
        """
        with self.lock:

            # Indicate that TTS playback has started
            self.tts_state_pub.publish(Bool(data=True))

            # Create a temporary WAV file for TTS output
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wav_path = tmp.name

            try:
                # Invoke Piper to generate speech audio
                cmd = [
                    "piper",
                    "--model", self.model_path,
                    "--output_file", wav_path
                ]

                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    text=True
                )

                proc.stdin.write(text)
                proc.stdin.close()
                proc.wait()

                # Play generated audio file
                subprocess.run(
                    [self.audio_player, wav_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )

            except Exception as e:
                self.get_logger().error(f"TTS error: {e}")

            finally:
                # Clean up temporary audio file
                if os.path.exists(wav_path):
                    os.remove(wav_path)

                # Indicate that TTS playback has finished
                self.tts_state_pub.publish(Bool(data=False))


def main():
    """
    Entry point for the TTS node.

    Purpose:
    - Initialize ROS2
    - Create and spin the TTS node
    - Ensure clean shutdown on exit

    Inputs:
    - None

    Outputs:
    - None
    """
    rclpy.init()
    node = TTSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


