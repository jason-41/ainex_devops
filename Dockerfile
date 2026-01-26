# ---------- Base image ----------
FROM ros:jazzy-ros-base

# ---------- Environment ----------
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

ENV ROS_DOMAIN_ID = 18

# ---------- System dependencies ----------
RUN apt-get update && apt-get install -y \
    python3-pip \
    build-essential \
    portaudio19-dev \
    libasound2-dev \
    v4l-utils \
    git \
    && rm -rf /var/lib/apt/lists/*

# ---------- Upgrade pip ----------
RUN python3 -m pip install --upgrade pip

# ---------- Install Python deps (direct pip, no venv) ----------
RUN pip install \
    mediapipe \
    piper-tts \
    faster-whisper \
    sounddevice \
    soundfile \
    numpy \
    webrtcvad

# ---------- Workspace ----------
WORKDIR /workspace
COPY . /workspace

# ---------- Build ROS packages ----------
SHELL ["/bin/bash", "-c"]
RUN source /opt/ros/jazzy/setup.bash && \
    colcon build --symlink-install

# ---------- Entrypoint ----------
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
