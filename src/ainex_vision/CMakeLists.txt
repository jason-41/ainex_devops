cmake_minimum_required(VERSION 3.8)
project(ainex_vision)

# 1. 基础依赖
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# 2. 生成 msg 接口
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/FaceBoundingBox.msg"
  "msg/FaceBoundingBoxArray.msg"
  DEPENDENCIES std_msgs sensor_msgs
)

# 3. 安装 Python 节点（注意：一条 install 只能 rename 一个程序）
install(
  PROGRAMS
    ainex_vision/face_detection_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME face_detection_node
)

install(
  PROGRAMS
    ainex_vision/camera_sub.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME camera_sub
)

install(
  PROGRAMS
    ainex_vision/aruco_tf_broadcaster.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME aruco_tf_broadcaster
)


install(
  PROGRAMS
    ainex_vision/aruco_detection_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME aruco_detection_node
)

# 4. 安装 resource（如有）
install(
  DIRECTORY models
  DESTINATION share/${PROJECT_NAME}
)



# 5. 导出依赖
ament_export_dependencies(rosidl_default_runtime)

ament_package()
