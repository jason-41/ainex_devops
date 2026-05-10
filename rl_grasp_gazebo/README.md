# rl_grasp_gazebo — Gazebo Harmonic 训练脚手架（Phase 1）

PyBullet 版的 RL 抓取在 [`../rl_grasp/`](../rl_grasp/)。这里是把任务搬到 **Gazebo Harmonic** 的工作目录，目前**只搭了脚手架**：可以 spawn 出机器人，看看场景对不对。RL 训练循环还没接上。

## 当前能做的事

| 阶段 | 状态 | 工作量估算 |
|---|---|---|
| Phase 1: URDF 迁移 + spawn 进 Gazebo + 看到场景 | ✅ **本次完成的范围** | — |
| Phase 2: Python ↔ Gazebo 通讯（gz-transport / ros_gz_bridge） | ❌ 待办 | 1–2 天 |
| Phase 3: Gym env / reset 服务 / step 同步 / 训练 | ❌ 待办 | 2–3 天 |

## 目录

```
rl_grasp_gazebo/
├── urdf/
│   ├── ainex_gz.urdf.xacro       顶层 xacro：原 ainex 链 + Harmonic <gazebo>
│   └── gazebo_harmonic.xacro     从 classic gazebo.xacro 迁移过来的插件块
├── worlds/
│   └── grasp.sdf                 ground / sun / 桌台 / cube
├── launch/
│   └── spawn.launch.py           gz sim + robot_state_publisher + spawn + bridge
├── envs/
│   └── grasp_env_gz.py           Gym env 骨架（NotImplementedError，Phase 2 填）
├── scripts/                      训练/评估脚本（Phase 3 填）
├── config/                       超参 / 控制器增益（Phase 3 填）
└── README.md                     本文件
```

## URDF 迁移决策

原 [`ainex_description/urdf/gazebo.xacro`](../src/ainex/ainex_description/urdf/gazebo.xacro) 是 **Gazebo Classic** 的写法，迁移要点：

| Classic | Harmonic（这里用的） |
|---|---|
| `libgazebo_ros_control.so` | **没用** ros2_control。改用每关节 `gz-sim-joint-position-controller-system` 直发 cmd_pos topic |
| `libgazebo_ros_imu.so` | 删了，RL 用 kinematic state 不需要 IMU |
| `libgazebo_ros_camera.so` | 删了，RL 不用相机 |
| `<selfCollide>` | 改成 `<self_collide>`（snake_case） |
| `<material>Gazebo/Black</material>` | 删了，不影响仿真 |
| `<mu1>`, `<mu2>` | 改成 `<mu>`, `<mu2>` |
| `<kp>`, `<kd>`, `<fdir1>` 脚底 | 保留 |

每个关节会暴露一个 topic：`/ainex/<joint_name>/cmd_pos`，发布 `gz.msgs.Double`。
关节状态在 `/world/grasp_world/model/ainex/joint_state`。

## 跑起来看一眼（先停掉 PyBullet 训练再做）

```bash
# 1. source 你的 colcon workspace 让 $(find ainex_description) 能用
cd /home/jason/Desktop/TUM/backup_practice/ainex_devops
source /opt/ros/jazzy/setup.bash
source install/setup.bash

# 2. 启动 Gazebo + spawn ainex
ros2 launch rl_grasp_gazebo/launch/spawn.launch.py
# 应该看到 ainex 站在地面上，前方有棕色平台 + 绿色 cube
```

## 已知 / 待解决

1. **`ainex_gz.urdf.xacro` 用相对路径包含 `gazebo_harmonic.xacro`**——必须从 `urdf/` 目录运行 xacro，或者改成绝对路径。
2. **没装 ros2_control**——这是有意为之，避免 ~150MB apt install。如果将来想接 controller_manager / 各种 ros2_controllers，跑 `sudo apt install ros-jazzy-ros2-control ros-jazzy-ros2-controllers ros-jazzy-gz-ros2-control` 再回来重写 `gazebo_harmonic.xacro`。
3. **`ainex_description` 不在标准 ROS 包路径**——需要先在仓库根 `colcon build && source install/setup.bash` 让它注册。

## Phase 2 的两条路（择一）

### 路 A：gz-transport python（无 ROS）

```bash
pip install gz-transport13       # 在 rl_venv 里
```
优点：训练脚本干净，沿用 `~/rl_venv`，一切跟 PyBullet 训练一样。
缺点：Python 绑定文档少，需要自己摸 API。

### 路 B：rclpy + ros_gz_bridge（用 ROS）

把 sb3 装进 `groupE_venv`（有 numpy 1.26 / torch 兼容性风险），训练脚本是个标准 ROS2 节点。
优点：标准做法，工具齐全（rqt 看 topic 等）。
缺点：依赖冲突要小心处理。

**建议先 A**——保留环境干净。卡住再退路 B。
