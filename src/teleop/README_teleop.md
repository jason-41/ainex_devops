# About 
### Created by Jia Cao, afternoon, 20.01.2026
This is the main control unit of the robot. It revokes the vision functions and executes control(walking, hands following and grabbing).

# Env settings
see overall project settings in README.md

# Deactivate the Environment
```
deactivate
```

# Node Run:
## !!!!!!!!!!!!!!!!! Before using any walking functions!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  :
```bash
ros2 service call /activate_walking std_srvs/srv/Empty {}
```

## 1. Terminal:
```bash
# Don't activate GUI
ros2 launch ainex_description display.launch.py gui:=false
```

### or(if run on the real robot):
```bash
ros2 run teleop hands_control --ros-args -p mode:=1 -p sim:=False
```
For Exercise 2 (original tutorial 6) only:
```bash

```

## 2. New Teleop Nodes (Updates 2026-01)

### Turn Around (with Safety Backup)
First moves the robot backward for 10 seconds to ensure clearance, then rotates by the specified degree.
```bash
# Rotate 180 degrees (Default)
ros2 run teleop turn_around

# Specific angle:
ros2 run teleop turn_around --ros-args -p degrees:=180.0
```

### Walk to Aruco
Visual servoing to approach an Aruco marker and stop at 10cm distance.
```bash
ros2 run teleop walk_to_aruco
```

### Crouch / Stand Up
Control the robot posture to crouch (for grasping) or stand up.
```bash
# Crouch
ros2 run teleop crouch --ros-args -p action:=crouch

# Stand Up
ros2 run teleop crouch --ros-args -p action:=stand
```

### Hands Control (Arm Movement & Tracking)
**Note:** Verify joint torque is enabled (`/Lock_All_Joints`). The node attempts to lock joints automatically on real robot mode.

**Mode 1: Kinematics Test** (Default)
Performs a relative movement test (approx. 6cm trajectory) to verify arm kinematics.
- Default is Real Robot execution (`sim:=False`).
```bash
ros2 run teleop hands_control
```

**Mode 2: Aruco Tracking / Reaching**
Tracks an Aruco marker detected by the vision system.
```bash
# 1. Start Vision
ros2 launch ainex_vision object_detection.launch.py

# 2. Start Hands Control (Tracking Mode)
ros2 run teleop hands_control --ros-args -p mode:=2
```

### if you want to reset the robot to the initial position:
```bash
ros2 run ainex_motion joint_controller

ros2 service call /Lock_All_Joints std_srvs/srv/Empty {}
ros2 service call /Unlock_All_Joints std_srvs/srv/Empty {}
```

### Grasp
To run grasp testing for cube hardcoded position relativ to camera frame
```bash
ros2 run teleop grasp --ros-args -p sim:=false -p use_camera:=false -p hardcoded_cam_xyz:="[0.05,0.05,-0.05]"

q_init = np.array([
            0.18, -0.96, -0.01675516,  0.00418879,
            -0.87126839,  2.33315611,  1.47864294,  0.03769911,
            -0.29740411, -1.24191744,  0.02932153, -1.65457213,
            0.0,        -0.01675516,  0.00837758,  0.83775806,
            -2.22843647, -1.41162229, -0.03769911, -0.26808256,
            1.36758114, 0.10890855,  1.68389368,  0.74979347
        ], dtype=float)