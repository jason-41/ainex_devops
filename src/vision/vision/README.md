# Notation from Yuan Zhao, 20. Jan.



cd ~/Documents/codeLibrary/groupE_final/groupE_final

cd ~/Documents/codeLibrary/groupE_final/groupE_final
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select vision
source install/setup.bash



## topic list
```bash
## Topic List

/aruco_markers   # Structured ArUco detection results (marker_id, pose, distance, angles)
/camera_image/compressed   # Original compressed image stream from robot camera
/camera/image_undistorted   # Undistorted camera image after calibration
/aruco_vis     # Visualization image with detected marker boxes and IDs (debug only)
/marker_search/status      # Human-readable guidance text (turn left/right, stop, etc.)
/marker_search/target_pose  # Target marker pose in camera_optical_link frame (NOT base_link/map)

## Coordinate Frames

camera_optical_link            # Camera optical frame (after undistortion, still camera frame)
marker_<id>                    # ArUco marker frame published via TF
camera_optical_link -> marker_<id>  # TF transform for marker pose (use tf2 to convert to base_link/map)
```

# Build
```bash
cd groupE_final 
# afterward in our working computer pwd is: cd ~/Documents/codeLibrary/groupE_final/groupE_final
#ls  check if cd the right src
source /opt/ros/jazzy/setup.bash #build ROS2 Jazzy
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
``` bash
colcon build --symlink-install
source install/setup.bash
```

# if meet problem or want to clean(opticakl)
``` bash
rm -rf build install log
source install/setup.bash
colcon build --symlink-install
```

# Vision part

## Undistortion -Terminal A
Open one new terminal and run the following code:
```bash
. /opt/ros/jazzy/setup.bash
. install/setup.bash
ros2 run vision undistortion
```
It will open one window show the undistortion image and publish undistorted msg.

## Aruco detection - Terminal B
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