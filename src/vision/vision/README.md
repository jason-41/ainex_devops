# Environment
``` bash
cd ~/Documents/codeLibrary/groupE_final/groupE_final
source /opt/ros/jazzy/setup.bash

source groupE_venv/bin/activate

colcon build --symlink-install --packages-select vision
source install/setup.bash
```

# Build vision package
``` bash
colcon build --symlink-install --packages-select vision
source install/setup.bash
```
# Clean rebuild (optional)
``` bash
rm -rf build install log
colcon build --symlink-install --packages-select vision
source install/setup.bash
```

# HOW TO RUN-Run Launch Files
``` bash
#Run Launch Files
ros2 launch vision marker_detection.launch.py
ros2 launch vision object_detection.launch.py

#Run Nodes (console_scripts)
ros2 run vision undistortion
ros2 run vision aruco_detector_node
ros2 run vision object_detector
ros2 run vision target_object_detector_node
ros2 run vision tf_debug_publisher_node
```


## Topic List

### Camera & Image Pipeline
- `/camera_image/compressed` (`sensor_msgs/CompressedImage`)  
  Raw camera image stream from the robot.

- `/camera/image_undistorted` (`sensor_msgs/Image`)  
  Undistorted image published by `undistortion.py`.

---

### ArUco Detection
- `/aruco_pose` (`geometry_msgs/PoseStamped`)  
  6D pose of the detected ArUco marker in `camera_optical_link`.

- `/aruco_status` (`std_msgs/String`)  
  Status of ArUco detection (e.g. NO_MARKER, WRONG_ID, OK).

---

### Object Detection (2D)
- `/detected_objects_raw` (`std_msgs/String`)  
  JSON-formatted 2D detection results (shape, color, pixel geometry).

- `/circle_center` (`geometry_msgs/Point`)  
  Center position of detected circle in image pixel coordinates (debug / legacy).

---

### Target Object Pose (3D)
- `/detected_object_pose` (`geometry_msgs/PoseStamped`)  
  Estimated 3D pose of the selected target object in camera frame.

- `/picked_object_pose` (`geometry_msgs/PoseStamped`)  
  Same pose as above, provided as a dedicated output for control.

- `/detected_object/status` (`std_msgs/String`)  
  Human-readable perception state (MATCH, NO OBJECT, NO COLOR, POSE FAILED).



# Vision Function Description
## 1.undistortion.py — Camera Undistortion Node
### Function
Undistort the raw camera image using pre-calibrated parameters and publish a clean image stream for vision modules.

### Topics

**Subscribe**
- `camera_image/compressed` (`sensor_msgs/CompressedImage`)
**Publish**
- `/camera/image_undistorted` (`sensor_msgs/Image`)
### Core functions
- `load_calibration()`
- `image_callback(msg)`
### Calibration
- YAML file: `vision/config/camera_calibration.yaml`
- Parameters used: camera matrix **K**, distortion coefficients **D**
### Run
```bash
ros2 run vision undistortion
```



## 2.aruco_detector_node.py — ArUco Marker Detection Node
### Function
Detect ArUco markers in the undistorted camera image and estimate their 6D pose in the camera frame.
### Topics
**Subscribe**
- `/camera/image_undistorted` (`sensor_msgs/Image`)
**Publish**
- `/aruco_pose` (`geometry_msgs/PoseStamped`)
- `/aruco_status` (`std_msgs/String`)
### Core functions
- `detect_markers()`
- `estimate_pose()`
- `publish_pose()`
### Parameters
- `camera_frame` (default: `camera_optical_link`)
- `allowed_ids` (list of marker IDs)
- `marker_size_m` (physical marker size)
### Run
```bash
ros2 run vision aruco_detector_node
```


### 3.object_detector.py

```md
## object_detector.py — 2D Color & Shape Detection Node
### Function
Detect colored objects (cube / circle) in the undistorted image and publish structured 2D detection results.
### Topics
**Subscribe**
- `/camera/image_undistorted` (`sensor_msgs/Image`)
**Publish**
- `/detected_objects_raw` (`std_msgs/String`)
- `/circle_center` (`geometry_msgs/Point`)
### Core functions
- `detect_circles()`
- `detect_cubes_from_mask()`
- `object_detect()`
### Output format
- JSON string containing object shape, color, pixel geometry and timestamp.
### Run
```bash
ros2 run vision object_detector
```

### 4.target_object_detector_node.py

```md
## target_object_detector_node.py — Target Object 3D Pose Node

### Function
Select the target object from 2D detections and estimate its 3D pose in the camera coordinate frame.

### Topics

**Subscribe**
- `/camera_info` (`sensor_msgs/CameraInfo`)
- `/detected_objects_raw` (`std_msgs/String`)
- `/camera/image_undistorted` (`sensor_msgs/Image`)

**Publish**
- `/detected_object_pose` (`geometry_msgs/PoseStamped`)
- `/picked_object_pose` (`geometry_msgs/PoseStamped`)
- `/detected_object/status` (`std_msgs/String`)

### Core functions
- `pick_target()`
- `pose_from_cube()`
- `pose_from_circle()`
- `apply_z_linear_calib()`

### Parameters
- `camera_frame` (default: `camera_optical_link`)
- `target_shape`
- `target_color`
- `cube_size_m`
- `z_calib_a`, `z_calib_b`

### Run
```bash
ros2 run vision target_object_detector_node
