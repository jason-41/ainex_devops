#  Project Description
### Project overview
This is a team project repository for the cource Humanoid Robotics Systems offered in winter semester of 2025 by Prof. Gordon Cheng of Institute for Cognitive Systems at Technical University of Munich. The project is based on an off‑the‑shelf robot model "AiNex Biped Humanoid Robot" and secondary development was conducted in order to achieve the course's final group project.  
<img width="200" height="200" alt="image" src="https://github.com/user-attachments/assets/68451164-dedd-4e9b-9ad3-4782b5561f56" />

### Project summary
(English)
- Built a **ROS 2-based humanoid manipulation pipeline** integrating perception, locomotion, and grasp control in a modular multi-node architecture.
- Integrated **LLM + speech interface** (ASR/TTS) to translate natural language instructions into executable robot actions.
- Implemented **Aruco-guided navigation and target-object localization** with OpenCV, enabling autonomous approach and pick-and-place behaviors.
- Developed **kinematics-driven arm/hand control** with Pinocchio and trajectory interpolation for stable grasp/degrasp execution.
- Developed a robotic grasping pipeline in Gazebo; leveraged the manufacturer-provided URDF and Pinocchio for robot kinematics/dynamics computation, and trained the policy with reinforcement learning (PPO).

(Chinese)
- 构建了基于 **ROS 2** 的人形机器人操作流水线，在模块化多节点架构中集成感知、移动与抓取控制。
- 集成 **LLM + 语音接口**（ASR/TTS），将自然语言指令转换为可执行机器人动作。
- 基于 OpenCV 实现 **Aruco 引导导航与目标定位**，支持机器人自主接近与抓取放置。
- 基于 Pinocchio 与轨迹插值开发 **运动学驱动的手臂/手部控制**，实现稳定抓取与释放。
- 使用 Gazebo 搭建机器人抓取仿真环境，基于厂家提供的 URDF，利用 Pinocchio 进行运动学/动力学计算，并采用强化学习的PPO算法训练抓取策略。

### Tech stack
(English)
#### 1) System & Middleware
- **ROS 2 Jazzy + colcon**: Built as a multi-package workspace with mixed `ament_python` / `ament_cmake` build types.
- **Modular architecture**: Core modules include `teleop` (state-machine control), `vision` (perception), `ainex_controller` (kinematics/execution), `llm_interface` (LLM instruction parsing), and `speech_interface` (voice interaction).

#### 2) Robot Control & Kinematics
- **Python robot-control stack**: Multi-node coordination is implemented with `rclpy`.
- **Kinematics & trajectory**: Uses **Pinocchio + NumPy + SciPy Rotation** for pose computation and end-effector trajectory handling.
- **Execution pipeline**: Custom messages/services (`servo_service`) are used to drive joint and posture control.
- The grasp controller is task-space driven: the desired end-effector trajectory is generated in Cartesian space, a PD law produces the desired Cartesian velocity, DLS inverse kinematics maps it to joint velocities, and these are integrated into joint position setpoints for the servo interface (Due to the nature of the robot, one can only send joint position commands to the robot by ROS2 communication).

#### 3) Perception & Vision
- **OpenCV + cv_bridge**: Handles image ingestion, preprocessing, and target recognition.
- **Aruco detection & localization**: Uses `cv2.aruco` and TF-related processing for marker detection and pose publishing.
- **Object detection & camera calibration**: Supports threshold/model-based detection and image undistortion workflows.

#### 4) Intelligent Interaction
- **LLM integration**: OpenAI API-based dialogue/instruction nodes convert natural language into structured robot tasks.
- **Speech pipeline**: ASR uses `faster-whisper + webrtcvad + sounddevice`, and TTS uses `piper-tts` for closed-loop voice interaction.

My contributions: Design of the overall control systems (simple state machine), train the grasping policy using reinforcement learning(PPO), system integration and final testing/validation.

(Chinese)
### 项目技术栈

#### 1) 系统与中间件
- **ROS 2 Jazzy + colcon**：基于多 package 工作空间组织，采用 `ament_python` / `ament_cmake` 混合构建。
- **模块化架构**：核心由 `teleop`（状态机控制）、`vision`（视觉检测）、`ainex_controller`（运动学与执行）、`llm_interface`（LLM 指令解析）、`speech_interface`（语音交互）组成。

#### 2) 机器人控制与运动学
- **Python 机器人控制栈**：使用 `rclpy` 编写多节点协同控制逻辑。
- **运动学/轨迹**：使用 **Pinocchio + NumPy + SciPy Rotation** 进行位姿计算与末端执行轨迹处理。
- **执行链路**：通过自定义消息/服务（`servo_service`）驱动关节与姿态控制。
- 抓取控制器是任务空间驱动的：先在笛卡尔空间生成末端参考轨迹，通过 PD 得到末端期望速度，再经 DLS 逆运动学映射为关节速度，最后将其积分为关节位置设定值发送给伺服（由于机器人的限制，只能由电脑经ros2发送关节坐标信息给机器人）。

#### 3) 感知与视觉
- **OpenCV + cv_bridge**：图像接入、预处理与目标识别。
- **Aruco 检测与定位**：使用 `cv2.aruco` 与 TF 相关处理完成标记检测与位姿发布。
- **目标检测与相机标定**：支持基于阈值/模型的目标识别与去畸变流程。

#### 4) 智能交互
- **LLM 接入**：基于 OpenAI API 的对话/指令服务节点，解析自然语言为机器人结构化任务。
- **语音链路**：ASR 使用 `faster-whisper + webrtcvad + sounddevice`，TTS 使用 `piper-tts`，实现语音输入输出闭环。

我的主要贡献包括：设计整体控制系统（采用简单状态机）、基于强化学习算法 PPO 训练机器人抓取、完成系统集成，并负责最终测试与验证。

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
