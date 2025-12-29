#!/usr/bin/env python3
import json
import queue
import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class ASRNode(Node):
    """
    Offline ASR using Vosk.
    Publishes recognized text to /speech/text_input (std_msgs/String).
    """

    def __init__(self):
        super().__init__('asr_node')

        # Parameters
        #self.declare_parameter('model_path', '')
        self.declare_parameter('sample_rate', 16000)
        #self.declare_parameter('device', None)  # None -> default input device

        #model_path = self.get_parameter('model_path').get_parameter_value().string_value
        model_path = "/home/wbo/humanoid_robotics_system/models/speech_recogntion/vosk-model-small-en-us-0.15"
        self.sample_rate = int(self.get_parameter('sample_rate').value)
        #self.device = self.get_parameter('device').value
        self.device = 9  # you may need to change this to your microphone device ID

        # Publisher
        self.pub = self.create_publisher(String, '/speech/text_input', 10)

        # Audio queue
        self.q = queue.Queue()

        # Load model
        if not model_path:
            # 你需要下载一个 Vosk 模型并把路径填进 launch 或参数里
            # 例如: vosk-model-small-en-us-0.15 / vosk-model-small-cn-0.22
            raise RuntimeError("model_path is empty. Please set ROS parameter 'model_path' to a Vosk model directory.")

        self.get_logger().info(f"Loading Vosk model from: {model_path}")
        self.model = Model(model_path)
        self.rec = KaldiRecognizer(self.model, self.sample_rate)
        self.rec.SetWords(False)

        # Start audio stream
        self.stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=8000,
            device=self.device,
            dtype='int16',
            channels=1,
            callback=self.audio_callback
        )
        self.stream.start()

        self.get_logger().info("ASR started. Speak into microphone...")

        # Timer to process queue
        self.timer = self.create_timer(0.05, self.process_audio)

    def audio_callback(self, indata, frames, time, status):
        if status:
            # 这里不要狂刷日志，否则会卡；只放进队列
            pass
        self.q.put(bytes(indata))

    def process_audio(self):
        """Pull audio chunks and run recognition."""
        try:
            while not self.q.empty():
                data = self.q.get_nowait()
                if self.rec.AcceptWaveform(data):
                    result = json.loads(self.rec.Result())
                    text = (result.get("text") or "").strip()
                    if text:
                        msg = String()
                        msg.data = text
                        self.pub.publish(msg)
                        self.get_logger().info(f"[ASR] {text}")
                else:
                    # partial = json.loads(self.rec.PartialResult()).get("partial","")
                    # 需要的话可以 debug partial
                    pass
        except Exception as e:
            self.get_logger().error(f"ASR processing error: {e}")


def main():
    rclpy.init()
    node = ASRNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.stream.stop()
            node.stream.close()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
