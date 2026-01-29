#!/usr/bin/env python3
"""
File: asr_node.py

Purpose:
This file implements a real-time Automatic Speech Recognition (ASR) ROS2 node
using microphone audio input. The node combines WebRTC Voice Activity Detection
(VAD) for speech segmentation with a Whisper-based transcription model
(faster-whisper).

The node continuously listens to audio input, detects speech segments,
transcribes completed utterances, and publishes the recognized text to a ROS2
topic for downstream processing by other components such as an LLM server.

Structure:
- ASRNode class:
  - Initializes audio input, VAD, and Whisper model
  - Performs real-time speech segmentation and transcription
  - Publishes recognized text to a ROS2 topic
  - Handles muting while text-to-speech (TTS) is active
- main function:
  - Initializes and spins the ROS2 node
"""

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
    ROS2 node for real-time speech recognition using VAD-based segmentation
    and Whisper transcription.
    """

    def __init__(self):
        """
        Initialize the ASR node.

        Purpose:
        - Configure audio input and VAD parameters
        - Load the Whisper ASR model
        - Initialize ROS2 publishers and subscribers
        - Start the real-time audio input stream

        Inputs:
        - None

        Outputs:
        - None
        """
        super().__init__('asr_node')

        # =========================
        # Audio and VAD parameters
        # =========================
        self.sample_rate =8000
        # TODO: Adjust microphone device index if needed
        self.device = 6                 # Microphone device index
        self.frame_duration_ms = 30     # Supported values: 10 / 20 / 30 ms
        self.silence_timeout_ms = 800   # Duration of silence to mark end of utterance

        self.frame_size = int(
            self.sample_rate * self.frame_duration_ms / 1000
        )

        self.max_silence_frames = int(
            self.silence_timeout_ms / self.frame_duration_ms
        )

        # =========================
        # Initialize Voice Activity Detection
        # =========================

        # VAD aggressiveness level: 0 (least) to 3 (most aggressive)
        self.vad = webrtcvad.Vad(3)

        # ========================~
        # Load Whisper ASR model
        # =========================
        self.get_logger().info("Loading Whisper model (faster-whisper)...")
        self.whisper = WhisperModel(
            "base",            # Available models: tiny / base / small
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
        # TTS activity gate
        # Mute ASR while the robot is speaking
        # =========================
        self.tts_active = False
        self.create_subscription(
            Bool,
            "/speech/tts_active",
            self.tts_state_callback,
            10
        )

        # =========================
        # Runtime state variables
        # =========================
        self.audio_buffer = []
        self.silence_counter = 0
        self.is_speaking = False

        # =========================
        # Start audio input stream
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
        """
        Handle updates to the TTS activity state.

        Purpose:
        - Mute ASR input while the robot is producing speech
        - Resume listening once TTS playback is finished

        Inputs:
        - msg (Bool): Indicates whether TTS is currently active

        Outputs:
        - None
        """
        if self.tts_active != msg.data:
            self.tts_active = msg.data
            if self.tts_active:
                self.get_logger().info("TTS active -> ASR muted")
            else:
                self.get_logger().info("TTS finished -> ASR listening")

    def audio_callback(self, indata, frames, time_info, status):
        """
        Process incoming audio frames from the microphone.

        Purpose:
        - Apply voice activity detection to incoming audio frames
        - Buffer speech frames and detect end-of-utterance conditions
        - Trigger transcription when a speech segment is completed

        Inputs:
        - indata: Raw audio input buffer
        - frames: Number of frames
        - time_info: Timing information
        - status: Stream status flags

        Outputs:
        - None
        """
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
        """
        Transcribe a completed speech utterance.

        Purpose:
        - Concatenate buffered audio frames
        - Convert audio to the required format
        - Run Whisper transcription
        - Publish recognized text to a ROS2 topic

        Inputs:
        - None

        Outputs:
        - None
        """
        if not self.audio_buffer:
            return

        self.get_logger().info("End of utterance detected, transcribing...")

        # Concatenate buffered audio frames
        audio = np.concatenate(self.audio_buffer, axis=0)
        self.audio_buffer.clear()

        # Convert audio to float32 format
        audio = audio.astype(np.float32) / 32768.0

        # Write audio to temporary WAV file for Whisper input
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            sf.write(tmp.name, audio, self.sample_rate)

            segments, _ = self.whisper.transcribe(
                tmp.name,
                beam_size=5,
                vad_filter=False,
                language=None   # Automatically detect language
            )

            text = ""
            for seg in segments:
                text += seg.text.strip() + " "

            text = text.strip()

            # Publish recognized text if non-empty
            if text:
                msg = String()
                msg.data = text
                self.pub.publish(msg)
                self.get_logger().info(f"[Whisper ASR] {text}")

    def destroy_node(self):
        """
        Cleanly shut down the ASR node.

        Purpose:
        - Stop and close the audio input stream
        - Release associated resources before node destruction

        Inputs:
        - None

        Outputs:
        - None
        """
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass
        super().destroy_node()


def main():
    """
    Entry point for the ASR node.

    Purpose:
    - Initialize ROS2
    - Create and spin the ASR node
    - Ensure clean shutdown on exit

    Inputs:
    - None

    Outputs:
    - None
    """
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
