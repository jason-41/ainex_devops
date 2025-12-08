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
For Excercise 1 only:
```bash
ros2 run ainex_controller ainex_hands_control_node --ros-args -p mode:=1 -p sim:=True
```
### or(if run on the real robot):
```
ros2 run ainex_controller ainex_hands_control_node --ros-args -p mode:=1 -p sim:=False
```
For Exercise 2 only:
```bash
ros2 run ainex_vision aruco_detection_node
ros2 run ainex_controller ainex_hands_control_node --ros-args -p mode:=2 -p sim:=True
```

# Corrections:
An observed behavior ( only one time, ) is that right hand moves backward instead of forward
The issue might be due to coordinate frame orientations in case it was tested before the corrections
to the coordinate frame orientations. 
If not then maybe this ainex_robot.py should be readjusted from -1 to 1
iq_real[self.robot_model.get_joint_id('r_sho_pitch')] *= -1.0
