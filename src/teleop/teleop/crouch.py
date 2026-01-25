#!/usr/bin/env python3
"""
Crouch/Stand node using the built-in 'Set_Posture' service from servo_service.
This avoids manual joint calculation and uses pre-defined stable postures.
"""
import rclpy
from rclpy.node import Node
from servo_service.msg import SetPosture
import time


class CrouchNode(Node):
    def __init__(self):
        super().__init__("crouch_node")

        self.declare_parameter("action", "crouch")  # 'crouch' or 'stand'
        self.declare_parameter("duration", 1.5)    # seconds

        self.action = self.get_parameter("action").value
        self.duration = self.get_parameter("duration").value

        # Publisher for the built-in posture topic
        # Defined in src/ainex/ainex_motion/ainex_motion/joint_controller.py
        self.posture_pub = self.create_publisher(SetPosture, 'Set_Posture', 10)

        self.get_logger().info("CrouchNode Initialized (Passive Mode)")

    def run(self):
        # Allow some time for connection
        time.sleep(0.5)

        msg = SetPosture()
        msg.duration = float(self.duration)

        if self.action == "crouch":
            msg.posture_name = 'crouch'
            self.get_logger().info(
                f"Sending posture command: CROUCH ({self.duration}s)")
        elif self.action == "stand":
            msg.posture_name = 'stand'
            self.get_logger().info(
                f"Sending posture command: STAND ({self.duration}s)")
        else:
            self.get_logger().warn(
                f"Unknown action: {self.action}. Valid: crouch, stand")
            return

        # Publish the command
        # We publish a few times just to be safe (UDP-like behavior of ROS2 best effort sometimes drops first msg)
        for i in range(3):
            self.posture_pub.publish(msg)
            time.sleep(0.1)

        self.get_logger().info("Command sent.")

        # Wait for the duration to ensure action completes before killing node
        time.sleep(self.duration)


def main(args=None):
    rclpy.init(args=args)
    node = CrouchNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
