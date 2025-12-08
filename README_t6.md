# go in workspace, always hrs_groupE
```bash
cd ~/Documents/codeLibrary/hrs/hrs_groupE
```
# Env build
```bash
cd ~/Documents/codeLibrary/hrs/hrs_groupE
colcon build --symlink-install
```
# Env settings
```bash
## environment setup

python3 -m venv my_venv --system-site-packages --symlinks
source my_venv/bin/activate
pip install mediapipe

source my_venv/bin/activate
source /opt/ros/jazzy/setup.bash
source install/setup.bash
```

## Deactivate the Environment
```
deactivate
```

# node run:
## 1. Terminal:
```bash
# Don't activate GUI
ros2 launch ainex_description display.launch.py gui:=false
```

## 2. Terminal
I set one example, to do one motion
- It will reach the initial pose they provided (self.q)
![Initial Pose](images/initial_pose.png)
- And then I give it one example target pose (AiNexModel, line68)
![Example Pose](images/example_pose.png)
For Excercise 1 only:
```bash
ros2 run ainex_controller ainex_hands_control_node --ros-args -p mode:=1 -p sim:=True
or:
ros2 run ainex_controller ainex_hands_control_node --ros-args -p mode:=1 -p sim:=False
```
For Exercise 2 only:
```bash
ros2 run ainex_vision aruco_detection_node
ros2 run ainex_controller ainex_hands_control_node --ros-args -p mode:=2 -p sim:=True
```


# Further IDEA:
The updated target should propagate to the AiNexModel class, although it may still be triggered via ainex_hand → ainex_hands. We will probably need to compute the inverse kinematics using end-effector target points. According to the instructor’s guidance, we should use the detector to determine a reachable point and then pair it with a specified uncertainty radius to define the final end-effector target pose.


# Coorections:
An observed behavior ( only one time, ) is that right hand moves backward instead of forward
The issue might be due to coordinate frame orientations in case it was tested before the corrections
to the coordinate frame orientations. 
If not then maybe this ainex_robot.py should be readjusted from -1 to 1
iq_real[self.robot_model.get_joint_id('r_sho_pitch')] *= -1.0
