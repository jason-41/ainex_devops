#!/usr/bin/env python3
"""
TUM - ICS AiNex CameraSubscriberCompressed Demo for ROS 2 Jazzy
----------------------------------------
Subscribes to JPEG-compressed images and raw images on /camera_image/compressed and /camera_image,
shows frames with OpenCV, and displays CameraInfo.

Requires:
  sudo apt install python3-numpy python3-opencv

Msgs:
    sensor_msgs/CompressedImage
    sensor_msgs/CameraInfo
"""

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, CameraInfo
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import Point

class CameraSubscriber(Node):
    def __init__(self):
        super().__init__('camera_subscriber')
        self.publisher_blob = self.create_publisher(Point, 'blob_center', 10) #q9
        self.publisher_circle = self.create_publisher(Point, 'circle_center', 10)

        self.cb_group = ReentrantCallbackGroup()

        # QoS: Reliable to ensure camera_info is received
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Subscribe compressed images
        self.sub_compressed = self.create_subscription(
            CompressedImage,
            'camera_image/compressed',
            self.image_callback_compressed,
            sensor_qos,
            callback_group=self.cb_group,
        )
        self.sub_compressed

        # Subscribe camera info
        self.sub_camerainfo = self.create_subscription(
            CameraInfo,
            'camera_info',
            self.camera_info_callback,
            sensor_qos,
            callback_group=self.cb_group
        )
        self.sub_camerainfo
        # self.sub_camerainfo_grey = cv2.cvtColor(self.sub_camerainfo, cv2.COLOR_RBG2GRAY)
        # State variables
        self.camera_info_received = False

        self.frame = None
        #initialize circle image to avoid error if not detected
        self.frame_circular_display = np.zeros((480, 640, 3), dtype=np.uint8)


    def camera_info_callback(self, msg: CameraInfo):
        if not self.camera_info_received:
            self.get_logger().info(
                f'Camera Info received: {msg.width}x{msg.height}\n'
                f'K: {msg.k}\n'
                f'D: {msg.d}'
            )
            print(f'Camera Info received: {msg.width}x{msg.height}')
            print(f'Intrinsic matrix K: {msg.k}')
            print(f'Distortion coeffs D: {msg.d}')
            self.camera_info_received = True

    def image_callback_compressed(self, msg: CompressedImage):
        
        # Define lower range for red
        lower_red1 = np.array([0, 120, 120])
        upper_red1 = np.array([10, 255, 255])

        # Define upper range for red
        lower_red2 = np.array([160, 120, 120])
        upper_red2 = np.array([180, 255, 255])

        # Define lower range for blue, question 7
        lower_blue = np.array([90, 120, 120])
        upper_blue = np.array([130, 255, 255])

        # Define lower range for green
        lower_green = np.array([35, 80, 80])
        upper_green = np.array([85, 255, 255])

        try:
            
            # Decode the compressed image
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            frame_cvt = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # q2 trun bgr/rbg to grey image
            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            _, frame_binary = cv2.threshold(frame_cvt, 127,255, cv2.THRESH_BINARY) #q5

            mask1_red = cv2.inRange(frame_hsv, lower_red1, upper_red1)
            mask2_red = cv2.inRange(frame_hsv, lower_red2, upper_red2)
            mask_green = cv2.inRange(frame_hsv, lower_green, upper_green)
            mask_blue = cv2.inRange(frame_hsv, lower_blue, upper_blue)

            frame_red = cv2.bitwise_or(mask1_red, mask2_red) #q6
            frame_green = mask_green #q7
            frame_blue = mask_blue #q7
            
            if frame is None:
                self.get_logger().warn('JPEG decode returned None')
                return
            if frame_cvt is None:
                self.get_logger().warn('GRAY decode returned None')
            self.frame = frame
            self.frame_cvt = frame_cvt
            self.frame_hsv = frame_hsv
            self.frame_binary = frame_binary
            self.frame_red = frame_red
            self.frame_green = frame_green
            self.frame_blue = frame_blue

            # blob extraction
            color_mask = frame_red  #q8
            kernel = np.ones((7, 7), np.uint8)
            eroded = cv2.erode(color_mask, kernel, iterations=1)
            dilated = cv2.dilate(eroded, kernel, iterations=2)
            self.frame_blob = dilated

            # === Q9: Find largest blob and its mass center ===
            contours, _ = cv2.findContours(self.frame_blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            self.center = None
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    self.center = (cx, cy)

            # Draw the blob center
            blob_with_center = cv2.cvtColor(self.frame_blob, cv2.COLOR_GRAY2BGR)
            if self.center:
                cv2.circle(blob_with_center, self.center, 8, (0, 0, 255), -1)

            # Save for display
            self.frame_blob_display = blob_with_center

            # Publish the blob center as a ROS message
            if self.center:
                msg = Point()
                msg.x = float(self.center[0])
                msg.y = float(self.center[1])
                msg.z = 0.0
                self.publisher_blob.publish(msg)
                self.get_logger().info(f'Published blob center: ({msg.x}, {msg.y})')


            #Q10: Detect circular shapes using HoughCircles
            frame_gray_blur = cv2.GaussianBlur(self.frame_cvt, (9, 9), 2)

            # HoughCircle Detection
            circles = cv2.HoughCircles(
                frame_gray_blur,
                cv2.HOUGH_GRADIENT,
                dp=1.2,           # Inverse ratio of the accumulator resolution to the image resolution
                minDist=50,       # Minimum distance between the centers of detected circles
                param1=100,       # Higher threshold for the Canny edge detector
                param2=50,        # Accumulator threshold for circle detection (lower values detect more false circles)
                minRadius=20,     # Minimum circle radius to be detected
                maxRadius=100      # Maximum circle radius to be detected

            )

            # copy raw image for visulization
            circular_display = frame.copy()

            if circles is not None:
                circles = np.uint16(np.around(circles))
                for (x, y, r) in circles[0, :]:
                    # draw circle
                    cv2.circle(circular_display, (x, y), r, (0, 255, 0), 2)
                    # draw center
                    cv2.circle(circular_display, (x, y), 4, (0, 0, 255), -1)

                    msg_circle = Point()
                    msg_circle.x = float(x)
                    msg_circle.y = float(y)
                    msg_circle.z = float(r)  # z = radius
                    self.publisher_circle.publish(msg_circle)

                    self.get_logger().info(f'Published circle center: ({x}, {y}), radius: {r}')

            # storage for display
            self.frame_circular_display = circular_display

        except Exception as exc:
            self.get_logger().error(f'Decode error in compressed image: {exc}')


    def process_key(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False  # Quit
        if key == ord('c'):
            self.show_compressed = True
            self.get_logger().info('Switched to compressed image')
        return True

    def display_loop(self):
        while rclpy.ok():
            if self.frame is not None:
                # Display the compressed image
                cv2.imshow('Camera Subscriber', self.frame)
                # cv2.imshow('Greyscale image', self.frame_cvt)
                # cv2.imshow('Binary image', self.frame_binary)
                # #cv2.imshow('HSV image', self.frame_hsv)
                # cv2.imshow('RED image', self.frame_red)
                # cv2.imshow('GREEN image', self.frame_green)
                # cv2.imshow('BLUE image', self.frame_blue)
                # cv2.imshow('Blob extraction', self.frame_blob)
                # cv2.imshow('Camera', self.frame)
                # cv2.imshow('Color extraction', self.frame_red)
                # cv2.imshow('Blob extraction', self.frame_blob_display)
                cv2.imshow('Camera', self.frame)
                cv2.imshow('Circular shapes', self.frame_circular_display)

            if not self.process_key():
                break

            rclpy.spin_once(self, timeout_sec=0.01)

        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = CameraSubscriber()
    node.get_logger().info('CameraSubscriber node started')

    try:
        node.display_loop()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
