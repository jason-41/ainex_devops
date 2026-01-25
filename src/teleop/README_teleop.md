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
## 1. Terminal:
```bash
# Don't activate GUI
ros2 launch ainex_description display.launch.py gui:=false
```

## 2. Terminal
For test only:
```bash
ros2 run ainex_controller new_ainex_hands_control_node --ros-args -p mode:=1 -p sim:=True
```
### or(if run on the real robot):
```bash
ros2 run ainex_controller new_ainex_hands_control_node --ros-args -p mode:=1 -p sim:=False
```
For Exercise 2 (original tutorial 6) only:
```bash
ros2 run ainex_vision aruco_detection_node
ros2 run ainex_controller new_ainex_hands_control_node --ros-args -p mode:=2 -p sim:=True
```

## 3. New Teleop Nodes (Updates 2026-01)

### Turn Around 180 degrees
Rotates the robot by a specified degree.
```bash
# Rotate 180 deg
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

### if you want to reset the robot to the initial position:
```bash
ros2 run ainex_motion joint_controller

ros2 service call /Lock_All_Joints std_srvs/srv/Empty {}
ros2 service call /Unlock_All_Joints std_srvs/srv/Empty {}
```