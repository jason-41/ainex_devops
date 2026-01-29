#!/usr/bin/env python3
# needs to be tested on real robot
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from servo_service.msg import SetPosture

import time
import math


class TurnAroundNode(Node):
    def __init__(self):
        super().__init__("turn_around_node")

        self.declare_parameter("speed", 2)
        self.declare_parameter("degrees", 180.0)
        self.declare_parameter("sim", True)

        self.speed = self.get_parameter("speed").value
        self.target_degrees = self.get_parameter("degrees").value
        self.sim = self.get_parameter("sim").value

        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.posture_pub = self.create_publisher(SetPosture, 'Set_Posture', 10)

        # Allow some time for connection
        time.sleep(0.5)

    def reset_posture(self):
        self.get_logger().info("Resetting posture before turning...")

        msg = SetPosture()
        msg.posture_name = 'stand'
        msg.duration = 1.5

        # Publish the command
        # We publish a few times just to be safe
        for i in range(3):
            self.posture_pub.publish(msg)
            time.sleep(0.1)

        # Wait for the motion to complete
        time.sleep(msg.duration)

    def execute_turn(self):
        # 1. Reset Posture
        self.reset_posture()

        # 2. Turn
        # Gait startup compensation: Legged robots take time to start moving.
        # We add a small buffer or rely on a tuning factor.
        # Increasing duration slightly to compensate for startup lag.
        startup_compensation = 1.0  # seconds estimated for gait init

        target_rad = math.radians(abs(self.target_degrees))
        speed = abs(self.speed)

        # Direction
        if self.target_degrees < 0:
            speed = -speed

        # Calculate pure motion duration
        motion_duration = target_rad / abs(speed)

        # Total duration strategy:
        # Simple open loop: just add startup time? Or assume effective motion starts late?
        # A simple heuristic: run for calculated time + compensation
        total_duration = motion_duration + startup_compensation
        total_duration = 60

        self.get_logger().info(
            f"Turning {self.target_degrees} degrees at {speed:.2f} rad/s.")
        self.get_logger().info(
            f"Motion Time: {motion_duration:.2f}s + Startup: {startup_compensation:.2f}s = Total: {total_duration:.2f}s")
        
        # NOTE: If pure rotation triggers translation, try to zero linear X/Y
        msg = Twist()
        msg.angular.z = 0
        msg.angular.z = float(speed)
        msg.linear.x = 0.0
        msg.linear.y = 0.0

        start_time = time.time()


        # Loop to publish velocity
        while rclpy.ok():
            elapsed = time.time() - start_time
            if elapsed > total_duration:
                break

            self.cmd_vel_pub.publish(msg)

            # Enforce loop rate more strictly
            # spin_once will return immediately if no events, so we need a manual sleep
            # to prevent flooding the network if there are no callbacks.
            time.sleep(0.05)
            rclpy.spin_once(self, timeout_sec=0)

        # 3. Stop
        self.stop()
        self.get_logger().info("Turn complete.")

    def stop(self):
        self.get_logger().info("Stopping robot...")
        msg = Twist()
        # Publish multiple times to ensure the robot receives the stop command
        # (UDP/Best-effort reliability fix)
        for _ in range(10):
            self.cmd_vel_pub.publish(msg)
            time.sleep(0.05)


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
