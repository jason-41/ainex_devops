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

from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

import tf_transformations

def rvec_to_quat_and_rpy(rvec):
    """
    Convert OpenCV rotation vector (rvec) to:
      - quaternion (x, y, z, w)
      - Euler angles (roll, pitch, yaw) in radians

    This is a pure math helper and can be moved to a TF / utils module later.
    """
    # rvec -> rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # build 4x4 homogeneous matrix for tf_transformations
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R

    # quaternion from rotation matrix
    qx, qy, qz, qw = tf_transformations.quaternion_from_matrix(T)

    # Euler angles (roll, pitch, yaw) in radians
    roll, pitch, yaw = tf_transformations.euler_from_matrix(T, axes="sxyz")

    return (qx, qy, qz, qw), (roll, pitch, yaw)

class ArucoDetectionNode(Node):
    def __init__(self):
        super().__init__("aruco_detector_node")

        # ================== 1. 参数配置 ==================
        self.allowed_ids = {18, 25}
        self.marker_length = 0.0485  # 物理边长 4.85cm

        # 视觉标准坐标系名称
        self.camera_frame = "camera_optical_link"

        # 平滑滤波 (EMA)
        self.alpha = 0.2
        self.tvec_filt = None

        # 状态控制
        self.print_interval = 0.5
        self.last_print_time = 0.0
        self.last_state = None
        self.overlay_interval = 0.5
        self.last_overlay_time = 0.0
        self.overlay_lines = ["msg = [--, [0.000, 0.000, 0.000], [0.0, 0.0, 0.0]]"]

        # UI 样式
        self.text_color = (255, 255, 0)  # 黄色
        self.text_outline = (0, 0, 0)    # 黑色轮廓

        # ================== 2. ROS 接口 ==================
        self.pub_pose = self.create_publisher(PoseStamped, "/aruco_pose", 10)
        self.pub_status = self.create_publisher(String, "/aruco_status", 10)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(CompressedImage, "/camera_image/compressed", self.image_callback, qos)

        # ================== 3. 视觉对象初始化 ==================
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        # 注意：这里使用了兼容新版 OpenCV 的参数创建方式
        try:
            self.parameters = aruco.DetectorParameters_create()
        except AttributeError:
            self.parameters = aruco.DetectorParameters()
            self.detector = aruco.ArucoDetector(self.aruco_dict, self.parameters)

        # ✅ 固化你的相机标定参数 (Camera Intrinsics)
        self.camera_matrix = np.array([
            [943.5696359603829, 0.0,               324.15554369880887],
            [0.0,               913.4473828616333, 244.9907944719079],
            [0.0,               0.0,               1.0],
        ], dtype=np.float64)

        self.dist_coeffs = np.array([
            -1.657449531248904, 5.644383618488399, -0.0232250230303879,
            -0.04489190371732941, -11.87213346365638
        ], dtype=np.float64)

    def image_callback(self, msg: CompressedImage):
        # 解码
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None: return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 检测 Marker
        if hasattr(self, 'detector'):
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

        now = time.time()

        # 1. 完全没检测到任何 Marker
        if ids is None or len(ids) == 0:
            self.publish_once("No marker detected.", state="NO_MARKER")
            # 在画面上提示
            cv2.putText(frame, "NO MARKER", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.imshow("Aruco Detection", frame)
            cv2.waitKey(1)
            return

        # 2. 提取所有检测到的 ID
        ids_list = [int(i) for i in ids.flatten().tolist()]
        valid_ids = [i for i in ids_list if i in self.allowed_ids]

        # 3. 如果检测到了 Marker，但 ID 是错误的（不在 allowed_ids 里）
        if not valid_ids:
            error_msg = f"WRONG ID = {ids_list}"
            self.publish_once(error_msg, state="WRONG_ID")
            
            # --- 保持画面输出 ---
            # 画出左上角红框 (Final project 风格)
            cv2.rectangle(frame, (10, 10), (620, 60), (0, 0, 255), 2)
            # 在框内写出错误的 ID 
            error_display = f"msg = [WRONG ID: {ids_list}, [0,0,0], [0,0,0]]"
            cv2.putText(frame, error_display, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # 可选：也把错误的 Marker 框出来，让人知道哪错了
            aruco.drawDetectedMarkers(frame, corners, ids)
            
            cv2.imshow("Aruco Detection", frame)
            cv2.waitKey(1)
            return

        target_id = valid_ids[0]
        idx = ids_list.index(target_id)

        # 关键：计算 3D 位姿 (PnP 算法)
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners, self.marker_length, self.camera_matrix, self.dist_coeffs
        )
        rvec, tvec = rvecs[idx][0], tvecs[idx][0]

        # 滤波
        if self.tvec_filt is None: self.tvec_filt = tvec
        else: self.tvec_filt = (1.0 - self.alpha) * self.tvec_filt + self.alpha * tvec
        x, y, z = self.tvec_filt

        # Convert OpenCV rotation vector to quaternion + Euler angles
        (qx, qy, qz, qw), (roll, pitch, yaw) = rvec_to_quat_and_rpy(rvec)

        # 发布消息
        self.publish_pose_msg(x, y, z, qx, qy, qz, qw)
        
        # ✅ 可视化：根据官方定义画出 3D 坐标轴
        self.draw_marker_axes(frame, rvec, tvec)

        # 更新 UI 文字
        if now - self.last_overlay_time >= self.overlay_interval:
            self.overlay_lines = [f"msg = [{target_id}, [{x:.3f},{y:.3f},{z:.3f}], [{math.degrees(roll):.1f},{math.degrees(pitch):.1f},{math.degrees(yaw):.1f}]]"]
            self.last_overlay_time = now

        for i, line in enumerate(self.overlay_lines):
            self.draw_text(frame, line, 20, 40 + i * 40)

        cv2.imshow("Aruco Detection", frame)
        cv2.waitKey(1)

    def draw_marker_axes(self, frame, rvec, tvec):
        """
        核心数学原理：重投影 (Re-projection)
        依照官方定义：X-红, Y-绿, Z-蓝
        """
        # 轴长度 L 建议设为 marker_length * 1.5 或 0.5
        L = self.marker_length * 1.5

        # 官方定义：原点在中心，X右(L,0,0)，Y下(0,L,0)，Z前(0,0,L)
        pts_3d = np.float32([
            [0, 0, 0],    # 原点
            [L, 0, 0],    # X 轴端点 (红色)
            [0, L, 0],    # Y 轴端点 (绿色)
            [0, 0, L]     # Z 轴端点 (蓝色)
        ]).reshape(-1, 1, 3)

        # 将 3D 点投影到 2D 像素平面
        pts_2d, _ = cv2.projectPoints(pts_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        pts = pts_2d.reshape(-1, 2).astype(int)

        o, x_axis, y_axis, z_axis = pts[0], pts[1], pts[2], pts[3]

        # 画线 (BGR 格式)
        cv2.line(frame, tuple(o), tuple(x_axis), (0, 0, 255), 3) # X - 红色
        cv2.line(frame, tuple(o), tuple(y_axis), (0, 255, 0), 3) # Y - 绿色
        cv2.line(frame, tuple(o), tuple(z_axis), (255, 0, 0), 3) # Z - 蓝色

    def publish_pose_msg(self, x, y, z, qx, qy, qz, qw):
        p = PoseStamped()
        p.header.frame_id = self.camera_frame
        p.header.stamp = self.get_clock().now().to_msg()
        p.pose.position.x, p.pose.position.y, p.pose.position.z = float(x), float(y), float(z)
        p.pose.orientation.x, p.pose.orientation.y, p.pose.orientation.z, p.pose.orientation.w = qx, qy, qz, qw
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
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()