# Notation from Yuan Zhao, 20. Jan.
## topic list
```bash
/aruco markers  #demonstrated the info from aruco detecter, details check src/vision_part/vision_interface
/camera_image/compressed  # Original Image from robot
/camera/image_undistorted  #Undistorted Image msg
```





# Env settings
```bash
source groupE_venv/bin/activate
```

just source, don't recreate, or even you have to pip something new, remember numpy need to be under 2.
```bash
pip uninstall numpy
pip install "numpy<2"
```

# Build
```bash
cd groupE_final 
# afterward in our working computer pwd is: ~/Documents/codeLibrary/groupE_final/groupE_final$
colcon build --symlink-install
```

# Vision part

## Undistortion
Open one new terminal and run the following code:
```bash
. /opt/ros/jazzy/setup.bash
. install/setup.bash
ros2 run vision undistortion
```
It will open one window show the undistortion image and publish undistorted msg.

## Aruco detection
Open one new terminal and run the following code:
```bash
. /opt/ros/jazzy/setup.bash
. install/setup.bash
ros2 run vision aruco_detector_node
```
It will detect the Aruco Marker and publish some msg, so far I can only make sure the ID is correct, to check the output, open one new terminal and run the following code:
```bash
. /opt/ros/jazzy/setup.bash
. install/setup.bash
ros2 topic echo /aruco_markers
```

## 