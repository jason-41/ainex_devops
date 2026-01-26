#!/usr/bin/env python3
"""
检查项目依赖关系
"""

import subprocess
import sys

# 需要检查的包列表
required_packages = [
    'sounddevice',
    'openai', 
    'face_recognition',
    'opencv-python',
    'numpy',
    'rospkg',
    'rclpy',
    'cv_bridge',
    'sensor_msgs',
    'geometry_msgs',
    'std_msgs'
]

print("=== 检查Python依赖包 ===")
for pkg in required_packages:
    try:
        __import__(pkg.replace('-', '_'))
        print(f"✓ {pkg}")
    except ImportError:
        print(f"✗ {pkg} - 未安装")

print("\n=== ROS2环境 ===")
try:
    result = subprocess.run(['ros2', '--version'], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"ROS2版本: {result.stdout.strip()}")
    else:
        print("ROS2未正确安装")
except FileNotFoundError:
    print("ROS2未找到，请先安装ROS2")

print("\n=== Python环境 ===")
print(f"Python路径: {sys.executable}")
print(f"Python版本: {sys.version}")

# 检查重要的系统依赖
print("\n=== 系统依赖 ===")
system_deps = ['portaudio19-dev', 'python3-dev', 'cmake', 'build-essential']
for dep in system_deps:
    result = subprocess.run(['dpkg', '-s', dep], capture_output=True, text=True)
    if 'install ok installed' in result.stdout:
        print(f"✓ {dep}")
    else:
        print(f"✗ {dep}")
