# vision/msg/DetectedObject.msg

geometry_msgs/PoseStamped pose

# 2D bbox size in pixels (from object_detector raw bbox [x, y, w, h])
float32 bbox_w_px
float32 bbox_h_px
