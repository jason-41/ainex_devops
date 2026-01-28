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

## Exercise1
For Excercise 1 only:

Moves the robot into a walking-ready posture and waits for velocity commands
```bash
ros2 service call /activate_walking std_srvs/srv/Empty {}
```
Teleoperate with keyboard

```
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

## Exercise2
For Exercise 2 only:

Open the first terminal and run:
```bash
ros2 run ainex_vision aruco_detection_node
```

Open the second terminal and run:
```bash
ros2 run ainex_controller ainex_walk_to_aruco_node
```
