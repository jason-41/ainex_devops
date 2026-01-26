# vision_interfaces/msg/MarkerInfo.msg

# Standard ROS header
# - stamp: detection time
# - frame_id: reference frame (e.g. camera_optical_link)
std_msgs/Header header

# ArUco marker ID
int32 marker_id

# Estimated 3D pose of the marker in header.frame_id
geometry_msgs/Pose pose

# Distance from camera to marker center [m]
float64 distance

# Horizontal angle (yaw) relative to camera optical axis [rad]
float64 horizontal_angle

# Vertical angle (pitch) relative to camera optical axis [rad]
float64 vertical_angle

# Angle between marker plane normal and camera optical axis [rad]
# (0 if not estimated)
float64 normal_angle

# Detection confidence [0.0 - 1.0]
# (heuristic or reserved for future use)
float64 confidence
