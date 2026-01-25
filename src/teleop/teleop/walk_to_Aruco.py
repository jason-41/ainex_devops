#!/usr/bin/env python3
"""
Refactored from ainex_walk-to-aruco_node.py
"""
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

import pinocchio as pin
import numpy as np
import math
import time
import sys
import termios
import tty
import threading
from ainex_controller.ainex_model import AiNexModel

from geometry_msgs.msg import PoseStamped, Twist
from scipy.spatial.transform import Rotation as R


class AinexWalkToAruco(Node):

    def __init__(self):
        super().__init__("ainex_walk_to_aruco_node")

        # --------------------------------------------------
        # Parameters
        # --------------------------------------------------
        self.dt = 0.05

        self.Kx = 1.2  # original 1.0
        self.Ky = 0.0
        self.Ktheta = 1.2

        self.stop_distance = 0.1   # meters
        self.max_vel = 1.0

        # --- ArUco timeout handling ---
        self.last_aruco_time = None
        self.aruco_timeout = 0.5   # seconds

        # --- Manual stop flag ---
        self.manual_stop = False

        # --------------------------------------------------
        # Robot model (Pinocchio)
        # --------------------------------------------------
        pkg = get_package_share_directory("ainex_description")
        urdf_path = pkg + "/urdf/ainex.urdf"

        self.robot_model = AiNexModel(self, urdf_path)

        try:
            self.cam_frame_id = self.robot_model.model.getFrameId(
                "camera_link")
        except Exception as e:
            self.get_logger().error(f"Cannot find camera_link: {e}")
            raise

        # camera_link -> camera_optical_link (ROS standard)
        R_clink_opt = R.from_euler(
            'xyz',
            [-np.pi / 2.0, 0.0, -np.pi / 2.0]
        ).as_matrix()
        self.T_clink_opt = pin.SE3(R_clink_opt, np.zeros(3))

        # --------------------------------------------------
        # ROS interfaces
        # --------------------------------------------------
        self.aruco_pose_msg = None

        self.create_subscription(
            PoseStamped,  # message type
            "/aruco_pose",  # topic name
            self.aruco_callback,  # callback
            10  # QoS depth
        )

        self.cmd_vel_pub = self.create_publisher(
            Twist,
            "/cmd_vel",
            10
        )

        # --------------------------------------------------
        # Control loop timer
        # --------------------------------------------------
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info("Ainex Walk-to-Aruco node started.")

        # keyboard listener for manual stop

        # keyboard_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        # keyboard_thread.start()

    # --------------------------------------------------
    # Callbacks
    # --------------------------------------------------
    def aruco_callback(self, msg: PoseStamped):
        self.aruco_pose_msg = msg
        self.last_aruco_time = self.get_clock().now()

    # def keyboard_listener(self):
    #     """
    #     Non-blocking keyboard listener.
    #     Press 'q' to stop the robot.
    #     """

    #     fd = sys.stdin.fileno()
    #     old_settings = termios.tcgetattr(fd)

    #     try:
    #         tty.setcbreak(fd)
    #         while rclpy.ok():
    #             ch = sys.stdin.read(1)
    #             if ch == 'q':
    #                 self.get_logger().warn("Manual stop triggered (q pressed).")
    #                 self.manual_stop = True
    #                 break
    #     finally:
    #         termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    # --------------------------------------------------
    # Main control loop
    # --------------------------------------------------

    def control_loop(self):

        # ----------------------------------------
        # Manual emergency stop
        # ----------------------------------------
        if self.manual_stop:
            self.cmd_vel_pub.publish(Twist())
            return

        twist = Twist()

        # --------------------------------------------------
        # No marker detected → stop
        # --------------------------------------------------
        if self.aruco_pose_msg is None:
            self.cmd_vel_pub.publish(twist)
            return

        now = self.get_clock().now()
        dt = (now - self.last_aruco_time).nanoseconds * 1e-9

        if dt > self.aruco_timeout:
            self.cmd_vel_pub.publish(Twist())
            return

        pose = self.aruco_pose_msg.pose

        # --------------------------------------------------
        # Marker pose in camera_optical_link
        # --------------------------------------------------
        p_opt = np.array([
            pose.position.x,
            pose.position.y,
            pose.position.z
        ], dtype=float)

        q_opt = np.array([
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w
        ], dtype=float)

        R_opt_m = R.from_quat(q_opt).as_matrix()
        T_opt_m = pin.SE3(R_opt_m, p_opt)

        # --------------------------------------------------
        # base_link -> marker
        # --------------------------------------------------
        T_b_clink = self.robot_model.data.oMf[self.cam_frame_id]
        T_b_opt = T_b_clink * self.T_clink_opt
        T_b_m = T_b_opt * T_opt_m

        x_m, y_m, _ = T_b_m.translation

        # --------------------------------------------------
        # Distance & heading
        # --------------------------------------------------
        distance = math.sqrt(x_m**2 + y_m**2)
        yaw_error = math.atan2(y_m, x_m)

        # --------------------------------------------------
        # Stop condition
        # --------------------------------------------------
        if distance < self.stop_distance:
            self.cmd_vel_pub.publish(Twist())
            return

        # # --------------------------------------------------
        # # Proportional walking controller （not used）
        # # --------------------------------------------------
        # vx = self.Kx * (x_m - self.stop_distance)
        # vy = self.Ky * y_m
        # wz = self.Ktheta * yaw_error

        # # Clamp velocities
        # vx = max(min(vx, self.max_vel), -self.max_vel)
        # vy = max(min(vy, self.max_vel), -self.max_vel)
        # wz = max(min(wz, self.max_vel), -self.max_vel)

        # twist.linear.x = vx
        # twist.linear.y = vy
        # twist.angular.z = wz

        # ----------------------------------------
        # Heading-first walking controller
        # ----------------------------------------
        angle_threshold = 0.174  # ~10 degrees

        twist = Twist()

        # 1) Rotate first if not facing marker
        if abs(yaw_error) > angle_threshold:
            twist.linear.x = 0.0
            twist.angular.z = self.Ktheta * yaw_error

        # 2) Otherwise, walk straight forward
        else:
            distance_error = x_m - self.stop_distance
            twist.linear.x = self.Kx * distance_error
            twist.angular.z = self.Ktheta * yaw_error

        # ----------------------------------------
        # Clamp
        # ----------------------------------------
        twist.linear.x = max(min(twist.linear.x, 0.6), -0.6)
        twist.angular.z = max(min(twist.angular.z, 1.0), -1.0)

        # IMPORTANT: no lateral motion
        twist.linear.y = 0.0

        self.cmd_vel_pub.publish(twist)


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    rclpy.init()
    node = AinexWalkToAruco()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down Ainex Walk-to-Aruco node.")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
