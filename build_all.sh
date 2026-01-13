#!/bin/bash
echo "Building all LLM packages..."
cd "$(dirname "$0")"
colcon build --packages-select llm_msgs servo_service speech_interface llm_interface
echo "Build complete!"
