# Vision Module – HRS Final Project (Group E)

## 1. 项目背景

本 Vision 模块是 Human-Robot Systems（HRS）课程 Final Project 的视觉子系统。  
整个 Final Project 由多个前期 Tutorial 和作业逐步整合而成，最终形成一个包含多个场景（Scenes）的完整人形机器人任务流程。

Vision 模块负责机器人在任务执行过程中所有与视觉感知相关的功能，包括：

- ArUco Marker 的检测与定位
- 目标物体（绿色立方体）的颜色与形状识别
- 物体姿态估计与抓取面判断
- 坐标系（TF）发布与 RViz 可视化
- 为人工遥操作（Human-in-the-loop）提供可解释的感知信息

本模块不直接控制机器人运动，运动由操作者或其他模块完成。

---

## 2. 场景与 Vision 任务划分

### Scene 1 – Robot and Operator Area
- 机器人与人类操作者交互
- 上层模块（LLM）解析任务指令，例如：
  - “Pickup at Marker 1”
  - “Go to Marker 2”
- Vision 模块根据目标 ID（Marker 1 / Marker 2）：
  - 搜索指定 ArUco Marker
  - 判断是否检测到目标 Marker
  - 输出 Marker 到相机（机器人头部）的距离、方向角、法向量夹角
- 操作者根据 Vision 输出信息控制机器人移动

---

### Straight Path Segment
- 机器人沿直线路径行走
- Vision 持续输出 Marker 相对位姿
- 用于人工遥操作时的方向修正

---

### Scene 2 – Object Box Area
- Marker 1 位于物体箱上方，用于粗定位
- 操作者控制机器人低头
- Vision 模块执行：
  - 绿色立方体（green cube）的 HSV 颜色分割
  - 目标物体检测与定位
  - 物体最大朝向面（宽 / 窄）判断
- 输出结果用于辅助机器人抓取

---

### Delivery Area
- 抓取完成后，根据指令搜索 Marker 2
- Vision 模块提供 Marker 2 的定位信息
- 操作者完成最终交付

---

## 3. 目录结构说明

```text
vision/
├── config/
│   ├── camera_calibration.yaml
│   ├── hsv_thresholds.yaml
│   └── object_pose_estimator_params.yaml
│
├── launch/
│   ├── marker_detection.launch.py
│   ├── object_detection.launch.py
│   └── object_pose_estimation.launch.py
│
├── scripts/
│   ├── search_marker.py
│   ├── detect_object.py
│   ├── object_pose_estimator.py
│   └── tf_debug_publisher.py
│
├── vision/
│   ├── __init__.py
│   ├── utils.py
│   └── nodes/
│       ├── __init__.py
│       ├── aruco_detector_node.py
│       ├── search_marker_node.py
│       ├── detect_object_node.py
│       ├── object_pose_estimator_node.py
│       └── tf_debug_publisher_node.py
│
├── CMakeLists.txt
├── package.xml
└── setup.py

# 4. 核心功能说明（功能与运行方式）

本章节对 Vision 模块中各个核心功能文件进行说明，重点描述每个文件的职责、输入输出信息以及运行方式。

所有功能均以 **ROS2 Node** 的形式运行，并通过 Topic、TF 和参数文件进行解耦与协作。

---

## 4.1 ArUco Marker Detection (`aruco_detector_node`)

### 功能说明
`aruco_detector_node` 负责从机器人头部相机图像中检测 ArUco Marker，并计算 Marker 相对于相机坐标系的空间位姿信息。该节点是 Vision 模块中所有 Marker 相关任务的基础节点。

**主要功能：**
* 从相机图像中检测 ArUco Marker
* 支持同时检测多个 Marker（Marker 1 / Marker 2）
* 计算 Marker 到相机的三维位置与姿态
* 计算并输出距离、方向角和法向量夹角
* 发布 Marker 对应的 TF 坐标系
* 支持 RViz 中的可视化调试

**接口信息：**
* **输入：** 相机图像（RGB 或 Compressed Image）、相机内参（硬编码或 CameraInfo）、Marker 尺寸参数（通过 launch 或 yaml 设置）
* **输出：** * `/aruco_markers` (MarkerInfoArray)：包含所有检测到的 Marker 信息
    * `/aruco_pose` (PoseStamped)：兼容接口，发布第一个 Marker 位姿
    * `/tf`：发布 `camera_frame` → `marker_<id>` 的坐标变换
    * 可选的可视化图像 Topic（用于调试）

**运行方式：**
* 通常通过 `marker_detection.launch.py` 启动
* 在 Scene 1、Scene 2 和 Delivery Area 中均会使用

---

## 4.2 Marker Search (`search_marker_node`)

### 功能说明
`search_marker_node` 用于根据给定的目标 Marker ID（1 或 2），判断目标 Marker 是否被当前视觉系统检测到，并输出用于人工遥操作的导航参考信息。该节点不负责控制机器人运动，仅提供状态判断和可解释的空间信息。

**主要功能：**
* 从 `/aruco_markers` 中筛选目标 Marker
* 判断目标 Marker 是否存在
* 输出 Marker 到相机的距离和方向角
* 判断是否已接近 Marker（距离 < 10 cm）
* 输出“已找到 / 未找到 / 已接近”的状态信息

**接口信息：**
* **输入：** `/aruco_markers` (MarkerInfoArray)、目标 Marker ID（参数传入）
* **输出：** 控制台日志信息（FOUND / NOT FOUND / NEAR）、可选的 Marker 位姿 Topic

**运行方式：**
* 通常与 `aruco_detector_node` 同时运行
* 在 Scene 1 和 Delivery Area 中使用，由 launch 文件或上层任务逻辑指定目标 Marker ID

---

## 4.3 Object Detection (`detect_object_node`)

### 功能说明
负责在 Scene 2 中对箱内目标物体进行颜色与形状识别，当前任务中主要识别 **绿色立方体（green cube）**。该节点基于 HSV 颜色空间进行分割，适用于现场光照条件下的快速调整。

**主要功能：**
* 将相机图像转换为 HSV 颜色空间
* 根据配置文件进行颜色阈值分割
* 提取符合条件的物体轮廓并过滤噪声
* 计算目标物体在相机坐标系下的位置

**接口信息：**
* **输入：** 相机图像、HSV 阈值参数（来自 `hsv_thresholds.yaml`）
* **输出：** 目标物体的位置与基本几何信息、可选的调试图像（显示分割结果）

**运行方式：**
* 通过 `object_detection.launch.py` 启动
* 仅在 Scene 2 中使用，HSV 参数可通过 YAML 文件快速调整

---

## 4.4 Object Pose Estimation (`object_pose_estimator_node`)

### 功能说明
用于在物体检测完成后，对目标物体进行姿态分析，并判断最适合抓取的朝向。该节点为抓取阶段提供决策支持信息。

**主要功能：**
* 接收目标物体的检测结果
* 分析物体轮廓或点集，判断物体最大可抓取面（宽 / 窄）
* 输出抓取姿态建议或方向信息

**接口信息：**
* **输入：** 目标物体检测结果、姿态分析参数（来自 `object_pose_estimator_params.yaml`）
* **输出：** 物体姿态类别（例如宽面朝上 / 窄面朝上）、抓取方向建议信息

**运行方式：**
* 通过 `object_pose_estimation.launch.py` 启动
* 在 Scene 2 抓取阶段使用

---

## 4.5 TF Debug & Visualization (`tf_debug_publisher_node`)

### 功能说明
用于发布和调试 Vision 模块中的坐标系关系，确保各类空间信息在 RViz 中清晰可见。

**主要功能：**
* 发布相机、Marker、物体等坐标系
* 验证坐标系方向与位置是否正确，辅助演示和问题定位

**接口信息：**
* **输入：** Vision 模块内部的位姿数据
* **输出：** `/tf` 坐标变换信息

**运行方式：**
* 可单独启动或与其他节点同时运行，主要用于调试和 Final Demo 展示

---

## 4.6 运行流程总结

Vision 模块在 Final Project 中的典型运行流程如下：
1. 启动 `aruco_detector_node`，检测并定位 Marker。
2. 启动 `search_marker_node`，根据目标 ID 判断 Marker 状态。
3. 在 Scene 2 启动 `detect_object_node`，识别绿色立方体。
4. 启动 `object_pose_estimator_node`，判断抓取姿态。
5. 使用 `tf_debug_publisher_node` 在 RViz 中进行可视化与验证。

---
```
### HOW TO RUN 
``` bash
cd ~/Documents/codeLibrary/groupE_final/groupE_final
touch groupE_venv/COLCON_IGNORE
rm -rf build install log
source /opt/ros/jazzy/setup.bash

colcon build --packages-select vision_interfaces vision
source install/setup.bash

ros2 pkg executables vision

```


Vision 模块运行指南 (Bash版)

本指南总结了在 AINEX 机器人环境下配置、编译及运行 Vision 模块的标准流程。
1. 环境准备 (虚拟环境)

Vision 模块依赖特定的 Python 库，请确保已激活 groupE_venv。
Bash

# 进入工作空间
cd ~/Documents/codeLibrary/groupE_final/groupE_final

# 激活虚拟环境
source groupE_venv/bin/activate

# 激活 ROS2 环境
source /opt/ros/jazzy/setup.bash

2. 编译模块

如果修改了代码结构或接口文件，建议清除旧的构建文件并重新编译：
Bash

# 清除旧的编译文件 (可选，报错时必做)
rm -rf build/vision build/vision_interfaces install/vision install/vision_interfaces

# 编译接口和 Vision 模块 (使用 --symlink-install 以便快速调试 Python)
colcon build --symlink-install --packages-select vision_interfaces vision

# 刷新环境变量
source install/setup.bash

3. 核心节点启动命令
3.1 启动基础环境 (Bringup)

在运行视觉节点前，必须先启动机器人基础驱动：
Bash

ros2 launch bringup bringup.launch.py

3.2 启动 ArUco Marker 检测
Bash

ros2 launch vision marker_detection.launch.py

3.3 启动物体识别 (Scene 2)
Bash

ros2 launch vision object_detection.launch.py

3.4 启动位姿估算 (Scene 2)
Bash

ros2 launch vision object_pose_estimation.launch.py

3.5 启动 TF 调试工具

手动发布/查看坐标系关系：
Bash

ros2 run vision tf_debug_publisher_node

4. 调试常用命令

    查看 Topic 是否正常输出：
    Bash

ros2 topic list | grep vision

检查图像数据流：
Bash

# 查看是否有图像话题
ros2 topic list | grep image
# 检查频率（确保相机已开启）
ros2 topic hz /camera/image_raw

手动调用脚本 (Debug 用)： 直接运行二进制文件：
Bash

./install/vision/bin/tf_debug_publisher_node





colcon build vison_interfaces vision should together