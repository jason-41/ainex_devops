# vision_interfaces/msg/MarkerInfoArray.msg

# Header for the entire detection result
# All markers are expressed in this frame and time
std_msgs/Header header

# List of detected markers
MarkerInfo[] markers
