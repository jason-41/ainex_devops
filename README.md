#  Project Description
### Project overview
This is a team project repository for the cource Humanoid Robotics Systems offered in winter semester of 2025 by Prof. Gordon Cheng of Institute for Cognitive Systems at Technical University of Munich. The project is based on an off‑the‑shelf robot model "AiNex Biped Humanoid Robot" and secondary development was conducted in order to achieve the course's final group project.  
<img width="200" height="200" alt="image" src="https://github.com/user-attachments/assets/68451164-dedd-4e9b-9ad3-4782b5561f56" />

#### My contributions: 
- Designed the overall control architecture as a simple state machine on top of ROS 2.
- **Trained the grasping policy with PPO in MuJoCo** (Gymnasium env on the manufacturer URDF, `stable-baselines3` + `SubprocVecEnv`), then deployed/visualized the same policy in **Gazebo Harmonic** through a `gz-transport` bridge.
- System integration across perception / LLM / control nodes, plus final hardware testing & validation.

#### 我的主要贡献：
- 在 ROS 2 之上设计了基于状态机的整体控制架构。
- **在 MuJoCo 中用 PPO 训练抓取策略**（Gymnasium env 直接吃厂家 URDF，`stable-baselines3` + `SubprocVecEnv` 多进程并行）；训好的策略再通过 `gz-transport` 桥接部署/可视化到 **Gazebo Harmonic**。
- 完成感知 / LLM / 控制各节点的系统集成，以及最终的硬件测试与验证。

### Project summary
(English)
- Built a **ROS 2-based humanoid manipulation pipeline** integrating perception, locomotion, and grasp control in a modular multi-node architecture.
- Integrated **LLM + speech interface** (ASR/TTS) to translate natural language instructions into executable robot actions.
- Implemented **Aruco-guided navigation and target-object localization** with OpenCV, enabling autonomous approach and pick-and-place behaviors.
- Developed **kinematics-driven arm/hand control** with Pinocchio and trajectory interpolation for stable grasp/degrasp execution.
- Built a grasp policy with **MuJoCo + Stable-Baselines3 PPO** on the manufacturer-provided URDF; the trained policy is then deployed/visualized in **Gazebo Harmonic** (`gz_sim` + `ros_gz_bridge`) for system-level integration testing within the ROS 2 stack.

(Chinese)
- 构建了基于 **ROS 2** 的人形机器人操作流水线，在模块化多节点架构中集成感知、移动与抓取控制。
- 集成 **LLM + 语音接口**（ASR/TTS），将自然语言指令转换为可执行机器人动作。
- 基于 OpenCV 实现 **Aruco 引导导航与目标定位**，支持机器人自主接近与抓取放置。
- 基于 Pinocchio 与轨迹插值开发 **运动学驱动的手臂/手部控制**，实现稳定抓取与释放。
- 基于厂家 URDF 在 **MuJoCo + Stable-Baselines3 PPO** 中训练抓取策略；训完的策略再部署到 **Gazebo Harmonic**（`gz_sim` + `ros_gz_bridge`）做 ROS 2 集成测试与可视化验证。

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

#### 5) Reinforcement-Learning Grasp Policy
- **Training simulator**: **MuJoCo 3.x** drives physics & contact during PPO rollouts; chosen over ROS-native simulators (Gazebo) because MuJoCo is the de-facto RL standard (used by OpenAI, DeepMind, Google) — fast, deterministic, headless, no IPC overhead.
- **Algorithm**: **PPO** from `stable-baselines3 2.3` with `MlpPolicy` (64×64 hidden), 4 parallel envs via `SubprocVecEnv`, 1M timesteps, ~25 min on a single laptop GPU/CPU.
- **MDP**: continuous task-space action `(Δx, Δy, Δz, gripper)`; observation is `(ee_xyz, cube_xyz, cube−ee, gripper_q)` (10D); reward is shaped reach: `−‖cube − ee‖ + close-by bonus + touch bonus`; episode ends on reach or 200-step timeout.
- **IK**: PyBullet's damped-least-squares IK runs as a sidecar on the same URDF to convert task-space deltas into the ainex right-arm joint targets fed to MuJoCo.
- **Deployment / visualization**: a separate replay script loads the trained `.zip` policy, subscribes to **Gazebo Harmonic** joint-state via `gz-transport`, computes the action, and publishes back to `/ainex/<joint>/cmd_pos` topics — closing the sim-to-sim loop into the ROS 2 stack.

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

#### 5) 强化学习抓取策略
- **训练仿真器**：**MuJoCo 3.x** 负责训练阶段的物理与接触仿真。选择 MuJoCo 而非 ROS 原生的 Gazebo，是因为 MuJoCo 是 RL 业界事实标准（OpenAI / DeepMind / Google 都用）——快、确定性强、无头运行、无 IPC 开销。
- **算法**：`stable-baselines3 2.3` 的 **PPO**，`MlpPolicy`（隐层 64×64），`SubprocVecEnv` 4 路并行，1M timesteps，单机笔记本 GPU/CPU 约 25 min 训完。
- **MDP**：连续任务空间动作 `(Δx, Δy, Δz, gripper)`；观测 `(ee_xyz, cube_xyz, cube−ee, gripper_q)`（10 维）；reward 走 reach shaping：`−‖cube − ee‖ + 接近 bonus + 触到 bonus`；触到或 200 步超时则终止。
- **IK**：PyBullet 的 damped-least-squares IK 作为 sidecar，加载相同 URDF，把任务空间 delta 翻译成 ainex 右臂关节目标喂给 MuJoCo。
- **部署 / 可视化**：训完的 `.zip` 策略由 replay 脚本加载，通过 `gz-transport` 订阅 **Gazebo Harmonic** 的 joint state，算 action 后再发布回 `/ainex/<joint>/cmd_pos`——完成 sim-to-sim 串接到 ROS 2 体系内。

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
    pip install catkin_pkg "empy==3.3.4" lark pyyaml opencv-contrib-python==4.9.0.80
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


# Reinforcement-Learning Grasp Pipeline

The PPO grasp policy is trained **outside** the ROS 2 workspace (in MuJoCo, no ROS dependency) and then **optionally** bridged into Gazebo for sim-to-sim integration testing. Two parallel folders:

```
rl_grasp/                       MuJoCo training + PyBullet sidecar IK + Gym envs
├── envs/grasp_env_mujoco.py    main env (Gymnasium API)
├── envs/grasp_env_ainex.py     legacy PyBullet env (kept for comparison)
├── scripts/train_ppo.py        sb3 PPO training entrypoint
├── scripts/eval_ppo.py         interactive eval (mujoco.viewer OR pybullet GUI)
├── scripts/smoke_test.py       env sanity check
└── urdf/                       MuJoCo-friendly URDF + compiled MJCF + scene

rl_grasp_gazebo/                Gazebo Harmonic deployment of the trained policy
├── urdf/                       Harmonic-style <gazebo> blocks (controllers, surfaces)
├── worlds/grasp.sdf            ground / table / cube
├── launch/spawn.launch.py      gz_sim + robot_state_publisher + ros_gz_bridge
└── scripts/replay_pybullet_policy.py    loads the .zip and publishes joint cmds
```

## Setup (separate from groupE_venv)

The RL stack runs in its own venv to avoid clashing with ROS 2 pins:

```bash
python3 -m venv ~/rl_venv
source ~/rl_venv/bin/activate
pip install "stable-baselines3==2.3.2" "gymnasium==0.29.1" "torch>=2.1" \
    tensorboard pybullet mujoco
```

## Train (MuJoCo)

```bash
source ~/rl_venv/bin/activate
cd rl_grasp
python scripts/train_ppo.py --robot ainex_mujoco --total-steps 1000000
# ~25 min on a laptop CPU; SubprocVecEnv with 4 parallel envs
# checkpoints land in checkpoints/ainex_mujoco/ (gitignored)
# TensorBoard logs in logs/ainex_mujoco/
```

Other `--robot` choices: `ainex` (PyBullet env, legacy), `franka` (gripper-capable Franka Panda demo).

Monitor convergence:
```bash
tensorboard --logdir logs/ainex_mujoco
```

## Eval — option A: MuJoCo viewer (no ROS)

```bash
python scripts/eval_ppo.py --robot ainex_mujoco
```
Opens a native MuJoCo window; runs 10 deterministic episodes and prints success rate.

## Eval — option B: Gazebo Harmonic (ROS 2 integration)

```bash
# terminal 1 — start Gazebo + spawn robot
source /opt/ros/jazzy/setup.bash && source install/setup.bash
export GZ_IP=127.0.0.1
ros2 launch rl_grasp_gazebo/launch/spawn.launch.py

# terminal 2 — replay the trained policy through gz-transport
source ~/rl_venv/bin/activate
export GZ_IP=127.0.0.1
python rl_grasp_gazebo/scripts/replay_pybullet_policy.py
```

The replay script subscribes to `/world/grasp_world/model/ainex/joint_state` and `/world/grasp_world/dynamic_pose/info`, runs the policy + IK in-process, then publishes joint targets to `/ainex/<joint>/cmd_pos`.

## MDP cheat sheet (matches both envs)

| field | shape | meaning |
|---|---|---|
| obs   | (10,) | ee_xyz(3) + cube_xyz(3) + (cube − ee)(3) + gripper_q(1) |
| action| (4,)  | (Δx, Δy, Δz, gripper) in [−1, 1], task-space |
| reward| —    | `−‖cube − ee‖ + close-by bonus + touch bonus − λ·Σq² − μ·‖a‖` |
| done  | —    | reach within 4 cm OR 200 timesteps |

IK runs as a PyBullet sidecar on the same URDF — task-space deltas are converted to ainex right-arm joint targets, then handed to MuJoCo for physics integration.
