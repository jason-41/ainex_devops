#!/usr/bin/env python3
import collections
import numpy as np
import sounddevice as sd
import soundfile as sf
import tempfile
import webrtcvad
import time

from faster_whisper import WhisperModel

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Bool



class ASRNode(Node):
    """
    Real-time ASR using:
    - WebRTC VAD (speech segmentation)
    - Whisper (faster-whisper) for transcription
    """

    def __init__(self):
        super().__init__('asr_node')

        # =========================
        # Audio / VAD parameters
        # =========================
        self.sample_rate = 16000
        self.device = 9                 # change to your microphone device
        self.frame_duration_ms = 30     # VAD support 10 / 20 / 30 ms
        self.silence_timeout_ms = 600   # mute for how long until silence is considered end of sentence

        self.frame_size = int(
            self.sample_rate * self.frame_duration_ms / 1000
        )

        self.max_silence_frames = int(
            self.silence_timeout_ms / self.frame_duration_ms
        )

        # =========================
        # Initialize VAD
        # =========================
          
        # the vad level 0~3, larger the value is more aggressive, 2 is balanced
        self.vad = webrtcvad.Vad(3)
        # =========================
        # Load Whisper
        # =========================
        self.get_logger().info("Loading Whisper model (faster-whisper)...")
        self.whisper = WhisperModel(
            "base",            # tiny / base / small
            device="cpu",
            compute_type="int8"
        )
        self.get_logger().info("Whisper loaded.")

        # =========================
        # ROS publisher
        # =========================
        self.pub = self.create_publisher(
            String,
            '/speech/text_input',
            10
        )

        # =========================
        # TTS activity gate (mute ASR while robot is speaking)
        # =========================
        self.tts_active = False
        self.create_subscription(
            Bool,
            "/speech/tts_active",
            self.tts_state_callback,
            10
        )


        # =========================
        # Runtime state
        # =========================
        self.audio_buffer = []
        self.silence_counter = 0
        self.is_speaking = False

        # =========================
        # Start audio stream
        # =========================
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            device=self.device,
            channels=1,
            dtype='int16',
            blocksize=self.frame_size,
            callback=self.audio_callback
        )
        self.stream.start()

        self.get_logger().info("Whisper + VAD ASR started. Speak naturally.")


    def tts_state_callback(self, msg: Bool):
        if self.tts_active != msg.data:
            self.tts_active = msg.data
            if self.tts_active:
                self.get_logger().info("TTS active -> ASR muted")
            else:
                self.get_logger().info("TTS finished -> ASR listening")


    def audio_callback(self, indata, frames, time_info, status):
        if status:
            return
        
        # ----- mute ASR while TTS is speaking -----
        if self.tts_active:
            self.audio_buffer.clear()
            self.silence_counter = 0
            self.is_speaking = False
            return

        audio_bytes = indata.tobytes()
        is_speech = self.vad.is_speech(audio_bytes, self.sample_rate)

        if is_speech:
            self.audio_buffer.append(indata.copy())
            self.silence_counter = 0
            self.is_speaking = True
        else:
            if self.is_speaking:
                self.silence_counter += 1

                if self.silence_counter > self.max_silence_frames:
                    self.is_speaking = False
                    self.silence_counter = 0
                    self.process_utterance()

    def process_utterance(self):
        if not self.audio_buffer:
            return

        self.get_logger().info("End of utterance detected, transcribing...")

        audio = np.concatenate(self.audio_buffer, axis=0)
        self.audio_buffer.clear()

        # transform to float32
        audio = audio.astype(np.float32) / 32768.0

        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            sf.write(tmp.name, audio, self.sample_rate)

            segments, _ = self.whisper.transcribe(
                tmp.name,
                beam_size=5,
                vad_filter=False,
                language=None   # auto-detect language
            )

            text = ""
            for seg in segments:
                text += seg.text.strip() + " "

            text = text.strip()

            if text:
                msg = String()
                msg.data = text
                self.pub.publish(msg)
                self.get_logger().info(f"[Whisper ASR] {text}")

    def destroy_node(self):
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = ASRNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
