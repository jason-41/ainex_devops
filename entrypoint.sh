#!/bin/bash
set -e

# Source ROS
source /opt/ros/jazzy/setup.bash

# Source workspace
source /workspace/install/setup.bash

exec "$@"
