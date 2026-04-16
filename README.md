# Structure Guidance

## 项目技术栈速览（可直接用于 CV / GitHub Profile）

> 为减少中英文来回切换，以下按 **英文整段在前，中文整段在后** 的形式组织。

### English Version

#### 1) System & Middleware
- **ROS 2 Jazzy + colcon**: Built as a multi-package workspace with mixed `ament_python` / `ament_cmake` build types.
- **Modular architecture**: Core modules include `teleop` (state-machine control), `vision` (perception), `ainex_controller` (kinematics/execution), `llm_interface` (LLM instruction parsing), and `speech_interface` (voice interaction).

#### 2) Robot Control & Kinematics
- **Python robot-control stack**: Multi-node coordination is implemented with `rclpy`.
- **Kinematics & trajectory**: Uses **Pinocchio + NumPy + SciPy Rotation** for pose computation and end-effector trajectory handling.
- **Execution pipeline**: Custom messages/services (`servo_service`) are used to drive joint and posture control.

#### 3) Perception & Vision
- **OpenCV + cv_bridge**: Handles image ingestion, preprocessing, and target recognition.
- **Aruco detection & localization**: Uses `cv2.aruco` and TF-related processing for marker detection and pose publishing.
- **Object detection & camera calibration**: Supports threshold/model-based detection and image undistortion workflows.

#### 4) Intelligent Interaction
- **LLM integration**: OpenAI API-based dialogue/instruction nodes convert natural language into structured robot tasks.
- **Speech pipeline**: ASR uses `faster-whisper + webrtcvad + sounddevice`, and TTS uses `piper-tts` for closed-loop voice interaction.

#### 5) Resume/Profile-ready Project Bullets
- Built a **ROS 2-based humanoid manipulation pipeline** integrating perception, locomotion, and grasp control in a modular multi-node architecture.
- Implemented **Aruco-guided navigation and target-object localization** with OpenCV, enabling autonomous approach and pick-and-place behaviors.
- Developed **kinematics-driven arm/hand control** with Pinocchio and trajectory interpolation for stable grasp/degrasp execution.
- Integrated **LLM + speech interface** (ASR/TTS) to translate natural language instructions into executable robot actions.

#### 6) Keywords (ATS / GitHub Topics)
`ROS2` `rclpy` `Humanoid-Robot` `Robot-Control` `Pinocchio` `OpenCV` `Aruco` `Computer-Vision` `LLM` `OpenAI-API` `Speech-Recognition` `Whisper` `TTS` `State-Machine`

---

### 中文版本

#### 1) 系统与中间件
- **ROS 2 Jazzy + colcon**：基于多 package 工作空间组织，采用 `ament_python` / `ament_cmake` 混合构建。
- **模块化架构**：核心由 `teleop`（状态机控制）、`vision`（视觉检测）、`ainex_controller`（运动学与执行）、`llm_interface`（LLM 指令解析）、`speech_interface`（语音交互）组成。

#### 2) 机器人控制与运动学
- **Python 机器人控制栈**：使用 `rclpy` 编写多节点协同控制逻辑。
- **运动学/轨迹**：使用 **Pinocchio + NumPy + SciPy Rotation** 进行位姿计算与末端执行轨迹处理。
- **执行链路**：通过自定义消息/服务（`servo_service`）驱动关节与姿态控制。

#### 3) 感知与视觉
- **OpenCV + cv_bridge**：图像接入、预处理与目标识别。
- **Aruco 检测与定位**：使用 `cv2.aruco` 与 TF 相关处理完成标记检测与位姿发布。
- **目标检测与相机标定**：支持基于阈值/模型的目标识别与去畸变流程。

#### 4) 智能交互
- **LLM 接入**：基于 OpenAI API 的对话/指令服务节点，解析自然语言为机器人结构化任务。
- **语音链路**：ASR 使用 `faster-whisper + webrtcvad + sounddevice`，TTS 使用 `piper-tts`，实现语音输入输出闭环。

#### 5) 可写进简历/主页的项目描述（示例）
- 构建了基于 **ROS 2** 的人形机器人操作流水线，在模块化多节点架构中集成感知、移动与抓取控制。
- 基于 OpenCV 实现 **Aruco 引导导航与目标定位**，支持机器人自主接近与抓取放置。
- 基于 Pinocchio 与轨迹插值开发 **运动学驱动的手臂/手部控制**，实现稳定抓取与释放。
- 集成 **LLM + 语音接口**（ASR/TTS），将自然语言指令转换为可执行机器人动作。

#### 6) 关键词（ATS / GitHub Topics 推荐）
`ROS2` `机器人控制` `人形机器人` `运动学` `计算机视觉` `Aruco定位` `大语言模型` `语音识别` `语音合成` `状态机`

```bash
workspace/
└── src/
    └── llm/                    # Metapackage/Main package
        ├── CMakeLists.txt      # Main package's CMakeLists
        ├── package.xml         # Main package's package.xml
        ├── src/                # Main package's source code
        │   └── ...             # Main package's functions/code
        ├── speech_interface/   # Sub-package 1
        │   ├── setup.py
        │   └── package.xml
        ├── servo_service/      # Sub-package 2
        │   ├── CMakeLists.txt
        │   └── package.xml
        └── llm_msgs/           # Sub-package 3
            ├── CMakeLists.txt
            └── package.xml
```

# Env settings
- There are several version conflictions, so please follow this installation flow
```bash
    # order matters!
	python3 -m venv groupE_venv --symlinks
    touch groupE_venv/COLCON_IGNORE
	source groupE_venv/bin/activate

    python -m pip install "scipy==1.16.3"
	pip install mediapipe piper-tts faster-whisper sounddevice soundfile webrtcvad "numpy<2"
    pip install catkin_pkg empy lark pyyaml opencv-contrib-python==4.9.0.80
    python -m pip install "numpy==1.26.4"

    # fallback, not neccessary
    python -m pip install -r requirements.txt

    source /opt/ros/jazzy/setup.bash
	source groupE_venv/bin/activate
	
	colcon build --symlink-install
	source install/setup.bash

    export PYTHONPATH="$PWD/groupE_venv/lib/python3.12/site-packages:${PYTHONPATH}"
```


# Main workflow

## Trun on the robot, every node that listening to the robot need to be shut down.
- Check the node list, 
```bash
ros2 node list
```
- if all four node is opened
```bash
/Joint_Control
/camera_publisher
/sensor_node
/walking_node
```
## 1. Terminal: Activate LLM node (Only when 1. step finished)
```bash
ros2 launch bringup bringup.launch.py
```
## 2. Terminal: Activate detection node (Only when 1. step finished)
- open one new terminal
```bash
ros2 launch vision detector.launch.py 
```

## 3. Terminal: Main control loop (Only when 1. step finished)
- open one new terminal
```bash
ros2 launch teleop main_control.launch.py
```


## Tips code backup
### Fake LLM topic publisher
- you can change the color between [blue, green, red, purple], the shape between [cube, circle]
```bash
ros2 topic pub /instruction_after_llm servo_service/msg/InstructionAfterLLM '{object_color: "green", object_shape: "cube", pickup_location: 33, destination_location: 25}'
```
### Prepare to walk
```bash
ros2 service call /activate_walking std_srvs/srv/Empty {}
ros2 service call /deactivate_walking std_srvs/srv/Empty {}
```

export PYTHONPATH=/home/hrs2025/Documents/codeLibrary/groupE_final/groupE_final/groupE_venv/lib/python3.12/site-packages:$PYTHONPATH


### Lock or Unlock the joint
```bash
ros2 service call /Unlock_All_Joints std_srvs/Empty {}
```


# some notes:

Your can check all TODOs in the following packages

1. (llm_interface): If you want to use llm api calling, you should get into the .bashrc with the command: nano ~/.bashrc and write the following command in it: export OPENAI_API_KEY= "YOUR_CHATGPT_APIKEYS"
  


2. (speech_interface)
tts_node:adapt the model path to your own one;
asr_node:adapt the microphone id to your own one, if needed


3. (ainex_vision):
face_detection_node:adapt the topic name to match your camera setup

Contributions of me: Design of the overall control systems (simple state machine), train the grasping process using reinforcement learning(PPO), system integration and final testing/validation.
