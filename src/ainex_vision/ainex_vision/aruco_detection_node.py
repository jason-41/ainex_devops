#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import cv2
import cv2.aruco as aruco
import numpy as np

class ArucoDetectionNode(Node):
    def __init__(self):
        super().__init__('aruco_detection_node')

        self.pub = self.create_publisher(PoseStamped, '/aruco_pose', 10)

        # 摄像头
        self.cap = cv2.VideoCapture(0)

        # 使用 OpenCV ArUco 字典
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.parameters = aruco.DetectorParameters_create()

        # 假设已经有相机内参（可根据你的摄像头修改）
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


        self.timer = self.create_timer(0.05, self.detect_marker)

        self.get_logger().info("ArucoDetectionNode started.")

    def detect_marker(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.parameters
        )

        if ids is not None:
            # 估计姿态（marker 长度 0.05 m，可以根据实际修改）
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                corners, 0.05, self.camera_matrix, self.dist_coeffs
            )

            # 发布 PoseStamped
            msg = PoseStamped()
            msg.header.frame_id = "camera_optical_link"
            msg.header.stamp = self.get_clock().now().to_msg()

            msg.pose.position.x = float(tvec[0][0][0])
            msg.pose.position.y = float(tvec[0][0][1])
            msg.pose.position.z = float(tvec[0][0][2])

            # 旋转矩阵 → 四元数
            R, _ = cv2.Rodrigues(rvec[0][0])
            qw = np.sqrt(1.0 + R[0,0] + R[1,1] + R[2,2]) / 2
            qx = (R[2,1] - R[1,2]) / (4 * qw)
            qy = (R[0,2] - R[2,0]) / (4 * qw)
            qz = (R[1,0] - R[0,1]) / (4 * qw)

            msg.pose.orientation.w = float(qw)
            msg.pose.orientation.x = float(qx)
            msg.pose.orientation.y = float(qy)
            msg.pose.orientation.z = float(qz)

            self.pub.publish(msg)

            # 画出 marker
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
        node.cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
