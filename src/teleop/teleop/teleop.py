"""
the main control logic unit
"""
import rclpy
from rclpy.node import Node
import math
import time

from ament_index_python.packages import get_package_share_directory

from ainex_controller.ainex_hand_controller import HandController
from ainex_controller.ainex_model import AiNexModel
from teleop.hands_control import hands_control
from teleop.hands_control import run_test
