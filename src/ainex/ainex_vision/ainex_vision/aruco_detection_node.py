#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CompressedImage
import cv2
import cv2.aruco as aruco
import numpy as np


class ArucoDetectionNode(Node):
    def __init__(self):
        super().__init__('aruco_detection_node')

        # 发布 ArUco 位姿
        self.pub = self.create_publisher(PoseStamped, '/aruco_pose', 10)

        # 订阅机器人摄像头（压缩 JPEG）
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.create_subscription(
            CompressedImage,
            "/camera_image/compressed",
            self.image_callback,
            qos
        )

        # ArUco 字典
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.parameters = aruco.DetectorParameters_create()

        # 你的相机内参（从你的原文件复制）
        self.camera_matrix = np.array([
            [943.5696359603829, 0.0, 324.1555436980887],
            [0.0, 913.4473828616333, 244.9907944719079],
            [0.0, 0.0, 1.0]
        ], dtype=float)

        self.dist_coeffs = np.array([
            -1.657449531248904,
            5.644383618483999,
            -0.023225023033879,
            -0.04498190371732941,
            -11.87213346365638
        ], dtype=float)

        self.get_logger().info("ArucoDetectionNode started (using /camera_image/compressed).")

    # ==========================================================
    # 图像回调：从压缩 JPEG 解码并进行 ArUco 检测
    # ==========================================================
    def image_callback(self, msg: CompressedImage):

        # 解码 JPEG → OpenCV BGR 图像
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            self.get_logger().warn("Failed to decode image.")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.parameters
        )

        if ids is None:
            cv2.imshow("Aruco Detection", frame)
            cv2.waitKey(1)
            return

        # 使用第一个 marker
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
            corners, 0.0485, self.camera_matrix, self.dist_coeffs
        )

        # 发布 PoseStamped
        msg_pose = PoseStamped()
        msg_pose.header.frame_id = "camera_optical_link"
        msg_pose.header.stamp = self.get_clock().now().to_msg()

        msg_pose.pose.position.x = float(tvec[0][0][0])
        msg_pose.pose.position.y = float(tvec[0][0][1])
        msg_pose.pose.position.z = float(tvec[0][0][2])

        # rvec → rotation matrix → quaternion
        R, _ = cv2.Rodrigues(rvec[0][0])
        qw = np.sqrt(1.0 + np.trace(R)) / 2
        qx = (R[2,1] - R[1,2]) / (4 * qw)
        qy = (R[0,2] - R[2,0]) / (4 * qw)
        qz = (R[1,0] - R[0,1]) / (4 * qw)

        msg_pose.pose.orientation.w = float(qw)
        msg_pose.pose.orientation.x = float(qx)
        msg_pose.pose.orientation.y = float(qy)
        msg_pose.pose.orientation.z = float(qz)

        self.pub.publish(msg_pose)

        # 绘制 marker
        aruco.drawDetectedMarkers(frame, corners, ids)

        # 显示画面
        cv2.imshow("Aruco Detection", frame)
        cv2.waitKey(1)


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


if __name__ == '__main__':
    main()