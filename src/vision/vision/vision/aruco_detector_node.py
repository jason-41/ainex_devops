#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
import numpy as np
import cv2
import cv2.aruco as aruco

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Int32


import tf_transformations



def rvec_to_quat_and_rpy(rvec):
    """rvec -> quaternion + (roll,pitch,yaw) in radians."""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    qx, qy, qz, qw = tf_transformations.quaternion_from_matrix(T)
    roll, pitch, yaw = tf_transformations.euler_from_matrix(T, axes="sxyz")
    return (qx, qy, qz, qw), (roll, pitch, yaw)


class ArucoDetectionNode(Node):
    def __init__(self):
        super().__init__("aruco_detector_node")
        self.bridge = CvBridge()

        # ================== parameter =================
        self.target_aruco_id = None  # Dynamic target from /vision_target
        self.marker_length = 0.0485  # meter

        self.camera_frame = "camera_optical_link"

        # EMA filter
        self.alpha = 0.2
        self.tvec_filt = None

        self.print_interval = 0.5
        self.last_print_time = 0.0
        self.last_state = None

        self.overlay_interval = 0.5
        self.last_overlay_time = 0.0
        self.overlay_lines = ["msg = [--, [0.000, 0.000, 0.000], [0.0, 0.0, 0.0]]"]

        self.text_color = (255, 255, 0)
        self.text_outline = (0, 0, 0)

        # ================== 2. ROS interface ==================
        self.pub_pose = self.create_publisher(PoseStamped, "/aruco_pose", 10)
        self.pub_status = self.create_publisher(String, "/aruco_status", 10)

        self.create_subscription(
            Int32,
            "/aruco_target",
            self.aruco_target_callback,
            10
        )

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.create_subscription(
            Image,
            "/camera/image_undistorted",
            self.image_callback,
            qos
        )
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

        try:
            self.parameters = aruco.DetectorParameters_create()
        except AttributeError:
            self.parameters = aruco.DetectorParameters()



        # ================== camera parameter ==================
        self.camera_matrix = np.array([
            [943.5696359603829, 0.0, 324.1555436980887],
            [0.0, 913.4473828616333, 244.9907944719079],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)


        self.dist_coeffs = np.zeros((1, 5), dtype=np.float64)

        self.get_logger().info("[Aruco] Node started. Sub=/camera/image_undistorted")

    def aruco_target_callback(self, msg):
        """Update the target ID to look for based on main_control request."""
        self.target_aruco_id = msg.data
        self.get_logger().info(f"[Aruco] Set target ID to: {self.target_aruco_id}")

    # ==========================================================
    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        if frame is None:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

        now = time.time()

        if ids is None or len(ids) == 0:
            self.publish_once("No marker detected.", state="NO_MARKER")
            # cv2.rectangle(frame, (10, 10), (620, 130), (0, 0, 255), 2)
            self.overlay_lines = ["STATE: NO_MARKER", "ID: --", "POS: --", "ORI: --"]
            if self.target_aruco_id is not None:
                 self.overlay_lines.append(f"Target: {self.target_aruco_id}")

            for i, line in enumerate(self.overlay_lines):
                self.draw_text(frame, line, 20, 40 + i * 30)

            cv2.imshow("Aruco Detection", frame)
            cv2.waitKey(1)
            return

        ids_list = [int(i) for i in ids.flatten().tolist()]
        # Determine which ID to track
        target_id = None
        
        # 1. If we have a specific target request from Main Control
        if self.target_aruco_id is not None:
            if self.target_aruco_id in ids_list:
                target_id = self.target_aruco_id
            else:
                # Target not found in this frame
                error_msg = f"LOOKING FOR {self.target_aruco_id}, FOUND {ids_list}"
                self.publish_once(error_msg, state="WRONG_ID")
                
                # cv2.rectangle(frame, (10, 10), (620, 130), (0, 0, 255), 2)
                self.overlay_lines = [
                    "TARGET NOT FOUND",
                    f"Target: {self.target_aruco_id}",
                    f"Visible: {ids_list}"
                ]
                for i, line in enumerate(self.overlay_lines):
                    self.draw_text(frame, line, 20, 40 + i * 40)
                    
                aruco.drawDetectedMarkers(frame, corners, ids)
                cv2.imshow("Aruco Detection", frame)
                cv2.waitKey(1)
                return

        # 2. If no specific target, accept the first detected marker
        else:
            target_id = ids_list[0]

        # Found a target_id to process
        idx = ids_list.index(target_id)


        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners, self.marker_length, self.camera_matrix, self.dist_coeffs
        )

        rvec = np.asarray(rvecs[idx][0], dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(tvecs[idx][0], dtype=np.float64).reshape(3, 1)

        if self.tvec_filt is None:
            self.tvec_filt = tvec.copy()
        else:
            self.tvec_filt = (1.0 - self.alpha) * self.tvec_filt + self.alpha * tvec

        x, y, z = self.tvec_filt.flatten().tolist()

        # ================== MANUAL CALIBRATION (scale + bias) ==================
        #   x_cal = sx*x + bx
        #   y_cal = sy*y + by
        #   z_cal = sz*z + bz
        # ================== FINAL Z CALIBRATION ==================
        sx, sy, sz = 1.0, 0.962, 0.573
        bx, by, bz = 0.0, 0.052, 0.002

        x = sx * x + bx
        y = sy * y + by
        z = sz * z + bz
        # =========================================================

                        # Convert rvec -> quaternion + rpy
        (qx, qy, qz, qw), (roll, pitch, yaw) = rvec_to_quat_and_rpy(rvec)

        self.publish_pose_msg(x, y, z, qx, qy, qz, qw)

        cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_length * 1.5)

        if now - self.last_overlay_time >= self.overlay_interval:
        
            # ===== nicer overlay (3 lines) =====
            roll_d  = math.degrees(roll)
            pitch_d = math.degrees(pitch)
            yaw_d   = math.degrees(yaw)

            self.overlay_lines = [
                f"ID: {target_id}",
                f"POS  x:{x:+.3f}  y:{y:+.3f}  z:{z:+.3f}  [m]",
                f"ORI  r:{roll_d:+.1f}  p:{pitch_d:+.1f}  y:{yaw_d:+.1f}  [deg]",
            ]

            self.last_overlay_time = now

        for i, line in enumerate(self.overlay_lines):
            self.draw_text(frame, line, 20, 40 + i * 40)

        cv2.imshow("Aruco Detection", frame)
        cv2.waitKey(1)

    def publish_pose_msg(self, x, y, z, qx, qy, qz, qw):
        p = PoseStamped()
        p.header.frame_id = self.camera_frame
        p.header.stamp = self.get_clock().now().to_msg()
        p.pose.position.x = float(x)
        p.pose.position.y = float(y)
        p.pose.position.z = float(z)
        p.pose.orientation.x = float(qx)
        p.pose.orientation.y = float(qy)
        p.pose.orientation.z = float(qz)
        p.pose.orientation.w = float(qw)
        
        # Log as [x,y,z], [qx,qy,qz,qw] using pose object
        # pos = p.pose.position
        # ori = p.pose.orientation
        # self.get_logger().info(f"[{pos.x}, {pos.y}, {pos.z}], [{ori.x}, {ori.y}, {ori.z}, {ori.w}]")
        
        self.pub_pose.publish(p)

    def draw_text(self, img, text, x, y):
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.text_outline, 3, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.text_color, 1, cv2.LINE_AA)

    def publish_once(self, text, state):
        if self.last_state != state:
            self.get_logger().info(text)
            self.pub_status.publish(String(data=text))
            self.last_state = state


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
