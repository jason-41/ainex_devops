#!/usr/bin/env python3

"""
File: face_detection_node.py

Purpose:
This file implements a ROS2 node for real-time face detection and identity-based
authentication using camera input. The node combines MediaPipe face detection
with face recognition to identify a known user and publish authentication state
to the rest of the system.

Detected faces are visualized, bounding boxes are published, and authentication
is confirmed only after consistent recognition over multiple frames.

Structure:
- FaceDetectionNode class:
  - Subscribes to camera images
  - Detects faces using MediaPipe
  - Performs face recognition against a reference encoding
  - Publishes bounding boxes, detection status, and authentication state
- main function:
  - Initializes and spins the ROS2 node
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import CompressedImage, Image
from ainex_vision.msg import FaceBoundingBox, FaceBoundingBoxArray
from cv_bridge import CvBridge

from ament_index_python.packages import get_package_share_directory

from std_msgs.msg import Bool  # for face_detected topic
import face_recognition
from auth_msgs.msg import AuthState # custom message for authentication state


import mediapipe as mp
import numpy as np
import cv2
import os
import time


class FaceDetectionNode(Node):
    """
    ROS2 node for face detection and face-based user authentication.
    """
    def __init__(self):
        """
        Initialize the face detection and authentication node.

        Purpose:
        - Subscribe to the camera image topic
        - Initialize the MediaPipe face detector
        - Load a reference face for identity verification
        - Create publishers for detection and authentication results

        Inputs:
        - None

        Outputs:
        - None
        """
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
        # TODO: Adapt the topic name to match your camera setup and adpt to CompressedImage if needed
        self.sub = self.create_subscription(
            Image,
            'image_raw',
            self.image_callback,
            qos
        )

        # Publisher for face detection results
        self.pub = self.create_publisher(
            FaceBoundingBoxArray,
            "/mediapipe/face_bbox",
            10
        )
        # Publish a boolean indicating whether a face is detected
        self.face_detected_pub = self.create_publisher(
            Bool,
            "face_detected",
            10
        )

        # Publisher for face authorization state
        self.auth_pub = self.create_publisher(
            AuthState,
            "/auth/face_state",
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
                running_mode=VisionRunningMode.IMAGE,
                min_detection_confidence=0.9  #every time you change, colcon build again
                # Lower confidence threshold (default is 0.5)
            )
        )

        # ---- load reference face (Bo) ----
        #TODO: Adapt the path to the reference image as needed, or use your own reference image, out of data privacy concerns i don't upload Bo's image
        bo_image_path = os.path.join(package_share, 'models', 'bo.jpg')
        bo_image = face_recognition.load_image_file(bo_image_path)

        bo_encodings = face_recognition.face_encodings(bo_image)
        if len(bo_encodings) == 0:
            raise RuntimeError("No face found in reference image!")

        self.bo_face_encoding = bo_encodings[0]
        self.get_logger().info("Reference face (Bo) loaded.")


        self.get_logger().info("FaceDetectionNode: detector created OK")

        self.get_logger().info("[Task 3] FaceDetectionNode started (Wayland headless mode).")


        # ---- authentication state ----
        self.known_face_counter = 0
        self.REQUIRED_CONFIRMATIONS = 5
        self.authenticated = False


# ------------------------------------------------------------------------------

    def image_callback(self, msg):
        """
        Process incoming camera images for face detection and authentication.

        Purpose:
        - Convert ROS image messages to OpenCV format
        - Detect faces using MediaPipe
        - Perform face recognition against a reference encoding
        - Publish bounding boxes and authentication state
        - Shut down the node after successful authentication

        Inputs:
        - msg (Image): Incoming camera image message

        Outputs:
        - None
        """
       
        # If already authenticated, do nothing
        if self.authenticated:
            return

        # Face recognition authorization variables
        authorized = False
        user_name = "Unknown"


        


        # Convert ROS Image → OpenCV format
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        h, w, _ = frame.shape

        # resize for faster detection
        small_frame = cv2.resize(frame, (320, 240))
        sh, sw, _ = small_frame.shape

        # Convert to MediaPipe input format
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
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
            self.get_logger().info(f">>> detections count: {len(result.detections)}")
            for det in result.detections:
                self.get_logger().info(">>> entered detection loop")
                bbox = det.bounding_box

                # Face Recognition (Bo vs Stranger) 

                # scale bbox back to original frame
                scale_x = w / sw
                scale_y = h / sh

                x1 = max(0, int(bbox.origin_x * scale_x))
                y1 = max(0, int(bbox.origin_y * scale_y))
                x2 = min(w, int((bbox.origin_x + bbox.width) * scale_x))
                y2 = min(h, int((bbox.origin_y + bbox.height) * scale_y))

                face_crop = frame[y1:y2, x1:x2]

                # skip invalid crop
                if face_crop.size == 0:
                    continue

                # convert to RGB for face_recognition
                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

                # encodings = face_recognition.face_encodings(face_rgb)
                h_fc, w_fc, _ = face_rgb.shape
                encodings = face_recognition.face_encodings(
                    face_rgb,
                    known_face_locations=[(0, w_fc, h_fc, 0)]
                )



                if len(encodings) == 0:
                    self.get_logger().info(f"face_crop size: {face_crop.shape}")
                    self.get_logger().warning("face_recognition: no face encoding found")
                    continue


                if len(encodings) > 0:
                    face_encoding = encodings[0]
                    distance = np.linalg.norm(face_encoding - self.bo_face_encoding)
                    
                    # threshold for recognition
                    if distance < 0.45: 
                        self.known_face_counter += 1
                        self.get_logger().info(
                            f"Known face detected ({self.known_face_counter}/"
                            f"{self.REQUIRED_CONFIRMATIONS})"
                        )
                    else:
                        # reset counter if not consistently Bo
                        self.known_face_counter = 0

                x_min = (bbox.origin_x * scale_x) / w
                y_min = (bbox.origin_y * scale_y) / h
                width = (bbox.width * scale_x) / w
                height = (bbox.height * scale_y) / h

                score = det.categories[0].score

                out_msg.faces.append(FaceBoundingBox(
                    x_min=x_min,
                    y_min=y_min,
                    width=width,
                    height=height,
                    score=score
                ))

                pad = 0.35  

                bw = bbox.width
                bh = bbox.height

                x1 = max(0, int((bbox.origin_x - pad * bw) * scale_x))
                y1 = max(0, int((bbox.origin_y - pad * bh) * scale_y))
                x2 = min(w, int((bbox.origin_x + bw * (1 + pad)) * scale_x))
                y2 = min(h, int((bbox.origin_y + bh * (1 + pad)) * scale_y))

                cv2.rectangle(output_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)


        # ---- check authentication threshold ----
        if self.known_face_counter >= self.REQUIRED_CONFIRMATIONS:
            self.get_logger().info("Identity confirmed: Bo")

            auth_msg = AuthState()
            auth_msg.authorized = True
            auth_msg.user_name = "Bo"
            self.auth_pub.publish(auth_msg)

            self.authenticated = True

            self.get_logger().info(
                "Authentication complete. Shutting down face_detection_node."
            )

            # shutdown node after successful authentication
            rclpy.shutdown()
            return

        # Publish detection result message
        self.pub.publish(out_msg)
        #publish authorization state
        if not self.authenticated:
            auth_msg = AuthState()
            auth_msg.authorized = authorized
            auth_msg.user_name = user_name
            self.auth_pub.publish(auth_msg)
        result = None  # free memory

        # Visualize detection output
        try:
            cv2.imshow("Face Recognition", output_vis)
            cv2.waitKey(1)
        except:
            pass


# -----------------------------------------------------------------------------
def main(args=None):
    """
    Entry point for the face detection and authentication node.

    Purpose:
    - Initialize ROS2
    - Create and spin the FaceDetectionNode
    - Ensure clean shutdown on exit

    Inputs:
    - args: Optional command-line arguments

    Outputs:
    - None
    """
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


