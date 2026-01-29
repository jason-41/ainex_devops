#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node

import cv2
import numpy as np
import yaml
import os

from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import Image


class UndistortImageNode(Node):
    def __init__(self):
        super().__init__("undistort_image_node")

        self.bridge = CvBridge()

        self.calibrated = False
        self.captured = 0
        self.last_capture_time = 0.0
        self.declare_parameter("imshowEnable", True)

        self.K = None
        self.D = None
        # ==========================
        # Load calibration config
        # ==========================
        # dir_path = os.path.dirname(os.path.realpath(__file__))
        # self.yaml_path = os.path.join(dir_path, "camera_calibration.yaml")
        pkg_share = get_package_share_directory("vision")
        self.yaml_path = os.path.join(
            pkg_share,
            "config",
            "camera_calibration.yaml"
        )


        if not os.path.exists(self.yaml_path):
            self.get_logger().error(
                f"[Undistort] Calibration file not found: {self.yaml_path}"
            )
            raise RuntimeError("Missing camera_calibration.yaml")

        self.load_calibration()

        # ==========================
        # ROS interfaces
        # ==========================
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribe raw image
        self.sub = self.create_subscription(
            CompressedImage,
            "camera_image/compressed",              
            self.image_callback,                
            qos
        )

        # Publish undistorted image
        self.pub = self.create_publisher(
            Image,
            "/camera/image_undistorted",
            10
        )

        self.get_logger().info("[Undistort] Node started.")
        self.get_logger().info("[Undistort] Press 'q' to quit.")

    # ==========================================================
    def load_calibration(self):
        with open(self.yaml_path, "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        cam = data["camera_matrix"]
        self.K = np.array(cam["data"]).reshape(
            cam["rows"], cam["cols"]
        )

        dist = data["distortion_coefficients"]
        self.D = np.array(dist["data"]).reshape(
            dist["rows"], dist["cols"]
        )

        self.get_logger().info("[Undistort] Calibration loaded successfully.")

    # ==========================================================
    def image_callback(self, msg):
        # ROS Image -> OpenCV
        frame = self.bridge.compressed_imgmsg_to_cv2(
            msg, desired_encoding="bgr8"
        )
        if frame is None:
            self.get_logger().info("[Undistort] No image input")
            return

        # Undistort
        undistorted = cv2.undistort(frame, self.K, self.D)

        # Show
        if self.get_parameter("imshowEnable").value:
            cv2.imshow("Undistorted View", undistorted)
            # Key handling
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.get_logger().info("[Undistort] Quit requested.")
                self.destroy_node()

        # Publish
        undist_msg = self.bridge.cv2_to_imgmsg(
            undistorted, encoding="bgr8"
        )
        self.pub.publish(undist_msg)



def main(args=None):
    rclpy.init(args=args)
    node = UndistortImageNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
