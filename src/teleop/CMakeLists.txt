cmake_minimum_required(VERSION 3.8)
project(teleop)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(PROGRAMS teleop/crouch.py DESTINATION lib/${PROJECT_NAME} RENAME crouch)
install(PROGRAMS teleop/hands_control.py DESTINATION lib/${PROJECT_NAME} RENAME hands_control)
install(PROGRAMS teleop/main_control.py DESTINATION lib/${PROJECT_NAME} RENAME main_control)
install(PROGRAMS teleop/turn_around.py DESTINATION lib/${PROJECT_NAME} RENAME turn_around)
install(PROGRAMS teleop/walk_to_Aruco.py DESTINATION lib/${PROJECT_NAME} RENAME walk_to_aruco)

ament_package()
