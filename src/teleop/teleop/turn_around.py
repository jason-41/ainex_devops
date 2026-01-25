#!/usr/bin/env python3
# needs to be tested on real robot
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from ament_index_python.packages import get_package_share_directory

import numpy as np
import time
import math

from ainex_controller.ainex_model import AiNexModel
from ainex_controller.ainex_robot import AinexRobot


class TurnAroundNode(Node):
    def __init__(self):
        super().__init__("turn_around_node")

        self.declare_parameter("speed", 0.4)
        self.declare_parameter("degrees", 180.0)
        self.declare_parameter("sim", True)

        self.speed = self.get_parameter("speed").value
        self.target_degrees = self.get_parameter("degrees").value
        self.sim = self.get_parameter("sim").value

        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self.dt = 0.05

        # Initialize Robot Model for pose control (to keep safe posture)
        try:
            pkg = get_package_share_directory("ainex_description")
            urdf_path = pkg + "/urdf/ainex.urdf"
            self.robot_model = AiNexModel(self, urdf_path)
            self.ainex_robot = AinexRobot(
                self, self.robot_model, self.dt, sim=self.sim)
            self.has_robot_control = True
        except Exception as e:
            self.get_logger().warn(
                f"Could not initialize AinexRobot: {e}. Running in velocity-only mode.")
            self.has_robot_control = False

    def reset_posture(self):
        if not self.has_robot_control:
            return

        self.get_logger().info("Resetting posture before turning...")
        # Initial posture (Standard standing pose)
        q_init = np.zeros(self.robot_model.model.nq)

        # Set standard arm positions (similar to hands_control.py)
        # Assuming we want arms down or in a neutral safely tucked position
        # Using the same init as hands_control for consistency
        q_init[self.robot_model.get_joint_id('r_sho_roll')] = 1.4
        q_init[self.robot_model.get_joint_id('l_sho_roll')] = -1.4
        q_init[self.robot_model.get_joint_id('r_el_yaw')] = 1.58
        q_init[self.robot_model.get_joint_id('l_el_yaw')] = -1.58

        self.ainex_robot.move_to_initial_position(q_init)
        time.sleep(2.0)

    def execute_turn(self):
        # 1. Reset Posture
        self.reset_posture()

        # 2. Turn
        target_rad = math.radians(abs(self.target_degrees))
        speed = abs(self.speed)

        # Direction
        if self.target_degrees < 0:
            speed = -speed

        duration = target_rad / abs(speed)

        self.get_logger().info(
            f"Turning {self.target_degrees} degrees at {speed:.2f} rad/s. Duration: {duration:.2f}s")

        msg = Twist()
        msg.angular.z = float(speed)

        start_time = time.time()

        # Loop to publish velocity
        while rclpy.ok():
            elapsed = time.time() - start_time
            if elapsed > duration:
                break

            self.cmd_vel_pub.publish(msg)

            # Spin to keep node alive and processing potential callbacks
            rclpy.spin_once(self, timeout_sec=0.05)
            # time.sleep(0.05) # spin_once handles the delay/timeout

        # 3. Stop
        self.stop()
        self.get_logger().info("Turn complete.")

    def stop(self):
        msg = Twist()
        self.cmd_vel_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TurnAroundNode()

    try:
        node.execute_turn()
    except KeyboardInterrupt:
        node.stop()
    except Exception as e:
        node.get_logger().error(f"Error: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
