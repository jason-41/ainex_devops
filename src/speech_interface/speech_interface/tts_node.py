#!/usr/bin/env python3
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
    Offline TTS using Piper.
    Subscribes to /speech/text_output and speaks the text.
    """

    def __init__(self):
        super().__init__('tts_node')

        # =========================
        # Piper model configuration
        # =========================
        # saved model location, should be changed to your own path
        self.model_path = (
            "/home/wbo/humanoid_robotics_system/models/tts/piper/"
            "en_US-amy-medium.onnx"
        )

        # play audio command (Linux universal)
        self.audio_player = "aplay"  

        # avoid preventing simultaneous playback
        self.lock = threading.Lock()

        # =========================
        # ROS subscription
        # =========================
        self.sub = self.create_subscription(
            String,
            '/speech/text_output',
            self.on_text,
            10
        )

        # create TTS state publisher, make sure if TTS is active, ASR can be paused
        self.tts_state_pub = self.create_publisher(
            Bool,
            "/speech/tts_active",
            10
        )


        self.get_logger().info("Piper TTS node started.")

    def on_text(self, msg: String):
        text = msg.data.strip()
        if not text:
            return

        self.get_logger().info(f"[TTS] {text}")

        # Use threading, avoid blocking ROS callback
        threading.Thread(
            target=self.speak,
            args=(text,),
            daemon=True
        ).start()

    def speak(self, text: str):
        with self.lock:

            self.tts_state_pub.publish(Bool(data=True))

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wav_path = tmp.name

            try:
                # call piper to generate the audio
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

                # play the audio
                subprocess.run(
                    [self.audio_player, wav_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )

            except Exception as e:
                self.get_logger().error(f"TTS error: {e}")

            finally:
                if os.path.exists(wav_path):
                    os.remove(wav_path)
                
                self.tts_state_pub.publish(Bool(data=False))


def main():
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
