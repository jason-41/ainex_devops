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
. my_venv/bin/activate
. /opt/ros/jazzy/setup.bash
. install/setup.bash
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
```bash
ros2 run ainex_controller ainex_hands_controller_node
```

# Further IDEA:
The updated target should propagate to the AiNexModel class, although it may still be triggered via ainex_hand → ainex_hands. We will probably need to compute the inverse kinematics using end-effector target points. According to the instructor’s guidance, we should use the detector to determine a reachable point and then pair it with a specified uncertainty radius to define the final end-effector target pose.