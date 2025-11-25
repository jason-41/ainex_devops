#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import CompressedImage, Image
from ainex_vision.msg import FaceBoundingBox, FaceBoundingBoxArray
from cv_bridge import CvBridge

from ament_index_python.packages import get_package_share_directory

from std_msgs.msg import Bool  # for face_detected topic

import mediapipe as mp
import numpy as np
import cv2
import os
import time

'''
modified from t4_ex3_face_detection_node.py
'''


class FaceDetectionNode(Node):
    def __init__(self):
        super().__init__("face_detection_node")
        self.bridge = CvBridge()

        # QoS settings
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribe to camera image topic
        # (For Bo's laptop: if others want to reproduce it, change the topic name)
        self.sub = self.create_subscription(
            CompressedImage,
            'camera_image/compressed',
            self.image_callback,
            qos
        )

        # Publisher for face detection results
        self.pub = self.create_publisher(
            FaceBoundingBoxArray,
            "/mediapipe/face_bbox",
            10
        )

        self.face_detected_pub = self.create_publisher(
            Bool,
            "face_detected",
            10
        )

        # Load MediaPipe FaceDetector
        BaseOptions = mp.tasks.BaseOptions
        FaceDetector = mp.tasks.vision.FaceDetector
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        package_share = get_package_share_directory('ainex_vision')
        model_path = os.path.join(package_share, 'models', 'face_detector.task')

        self.detector = FaceDetector.create_from_options(
            FaceDetectorOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=VisionRunningMode.IMAGE
            )
        )

        self.get_logger().info("FaceDetectionNode: detector created OK")

        # Directory for automatic screenshot saving
        self.save_dir = os.path.join(
            os.path.dirname(__file__),
            "face_detection_outputs"
        )
        os.makedirs(self.save_dir, exist_ok=True)
        self.last_save = 0

        self.get_logger().info("[Task 3] FaceDetectionNode started (Wayland headless mode).")

# ------------------------------------------------------------------------------

    def image_callback(self, msg):
        """Process camera image → detect faces → publish → visualize → save screenshot"""

        # Convert ROS Image → OpenCV format
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        h, w, _ = frame.shape

        # Convert to MediaPipe input format
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

        result = self.detector.detect(mp_image)

        out_msg = FaceBoundingBoxArray()
        output_vis = frame.copy()

        has_face = len(result.detections) > 0
        self.face_detected_msg = Bool()
        self.face_detected_msg.data = has_face
        self.face_detected_pub.publish(self.face_detected_msg)
        has_face = False  # reset for next frame

        if result.detections:
            for det in result.detections:
                bbox = det.bounding_box

                x_min = bbox.origin_x / w
                y_min = bbox.origin_y / h
                width = bbox.width / w
                height = bbox.height / h
                score = det.categories[0].score

                out_msg.faces.append(FaceBoundingBox(
                    x_min=x_min,
                    y_min=y_min,
                    width=width,
                    height=height,
                    score=score
                ))

                # Draw detection bounding box
                x1 = int(bbox.origin_x)
                y1 = int(bbox.origin_y)
                x2 = int(bbox.origin_x + bbox.width)
                y2 = int(bbox.origin_y + bbox.height)
                cv2.rectangle(output_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Publish detection result message
        self.pub.publish(out_msg)
        result = None  # free memory

        # Visualize detection output
        try:
            cv2.imshow("Face Detection", output_vis)
            cv2.waitKey(1)
        except:
            pass

        # Automatically save a screenshot every 1 second
        # now = time.time()
        # if now - self.last_save > 1.0:
        #     self.last_save = now
        #     save_path = os.path.join(self.save_dir, f"face_{int(now)}.jpg")
        #     cv2.imwrite(save_path, frame)
        #     self.get_logger().info(f"[Task 3] Screenshot saved: {save_path}")


# -----------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = FaceDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
