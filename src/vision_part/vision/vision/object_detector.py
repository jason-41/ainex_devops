#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import json
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point            # optional: keep /circle_center
from std_msgs.msg import String                # for /detected_objects_raw JSON


class ObjectDetectorNode(Node):
    def __init__(self):
        super().__init__("object_detector_node")

        # Subscribe to compressed camera images
        self.create_subscription(
            CompressedImage,
            "/camera_image/compressed",
            self.image_callback,
            qos_profile_sensor_data
        )

        # Optional: keep the old circle_center topic (if someone uses it)
        self.publisher_circle = self.create_publisher(Point, "/circle_center", 10)

        # New: raw detection publisher (shape + color + 2D geometry) as JSON string
        self.pub_raw = self.create_publisher(String, "/detected_objects_raw", 10)

        self.get_logger().info("[object_detector_node] started.")

    # ---------------- Color masks ----------------

    def get_red_mask(self, hsv):
        # Red has two hue ranges
        m1 = cv2.inRange(hsv, np.array([0, 50, 120]), np.array([10, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([160, 70, 130]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(m1, m2)
        return self.clean(mask)

    def get_green_mask(self, hsv):
        mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
        return self.clean(mask)

    def get_blue_mask(self, hsv):
        mask = cv2.inRange(hsv, np.array([95, 25, 35]), np.array([125, 255, 255]))
        return self.clean(mask)

    def get_purple_mask(self, hsv):
        mask = cv2.inRange(hsv, np.array([130, 30, 20]), np.array([160, 255, 255]))
        return self.clean(mask)

    @staticmethod
    def clean(mask):
        # Small morphological clean-up for color masks
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)
        return mask

    # ---------------- Raw detection publisher (JSON) ----------------

    def publish_raw_detection(
        self,
        shape: str,
        color: str,
        center_uv=None,
        quad_points=None,
        bbox=None,
        radius: float = 0.0,
    ):
        """
        Publish one raw detection as JSON on /detected_objects_raw.

        JSON structure:
        {
          "shape": "cube" / "circle" / ...,
          "color": "red" / "green" / "blue" / "purple",
          "center_uv": [u, v] or [],
          "quad": [u0, v0, u1, v1, u2, v2, u3, v3] or [],
          "bbox": [x, y, w, h] or [],
          "radius": float,
          "stamp": float (seconds, ROS time)
        }
        """
        data = {
            "shape": shape,
            "color": color,
            "center_uv": [],
            "quad": [],
            "bbox": [],
            "radius": float(radius),
            "stamp": self.get_clock().now().nanoseconds / 1e9,
        }

        if center_uv is not None:
            cu, cv = float(center_uv[0]), float(center_uv[1])
            data["center_uv"] = [cu, cv]

        if quad_points is not None:
            quad_points = np.asarray(quad_points, dtype=np.float32).reshape(-1, 2)
            flat = []
            for (u, v) in quad_points:
                flat.append(float(u))
                flat.append(float(v))
            data["quad"] = flat

        if bbox is not None:
            x, y, w, h = bbox
            data["bbox"] = [float(x), float(y), float(w), float(h)]

        msg = String()
        msg.data = json.dumps(data)
        self.pub_raw.publish(msg)

    # ---------------- Circle detector ----------------

    def detect_circles(self, frame, display, mask=None):
        """
        If mask is not None, restrict detection to masked region.
        Here we assume purple circles only, so color='purple'.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if mask is not None:
            gray = cv2.bitwise_and(gray, gray, mask=mask)

        gray_blur = cv2.GaussianBlur(gray, (9, 9), 2)

        circles = cv2.HoughCircles(
            gray_blur,
            cv2.HOUGH_GRADIENT,
            dp=1.3,
            minDist=80,
            param1=120,
            param2=35,
            minRadius=25,
            maxRadius=100
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for (x, y, r) in circles[0, :]:
                # Visualization
                cv2.circle(display, (x, y), r, (0, 255, 255), 2)
                cv2.circle(display, (x, y), 3, (0, 0, 255), -1)

                # Optional: keep old /circle_center topic
                point_msg = Point()
                point_msg.x = float(x)
                point_msg.y = float(y)
                point_msg.z = float(r)
                self.publisher_circle.publish(point_msg)

                # Publish raw detection
                bbox = (x - r, y - r, 2 * r, 2 * r)
                self.publish_raw_detection(
                    shape="circle",
                    color="purple",
                    center_uv=(x, y),
                    quad_points=None,
                    bbox=bbox,
                    radius=float(r),
                )

    # ---------------- Cube detector (all cube-like blobs get a rotated box) ----------------

    def detect_cubes_from_mask(self, frame, mask, display, color_name: str, label="Cube"):
        """
        From a single color mask, find all blobs with roughly square shape
        using minAreaRect. Draw rotated boxes and publish raw detections.

        color_name: "red" / "green" / "blue"
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)
            # Area threshold: ignore tiny noise
            if area < 400:
                continue

            rect = cv2.minAreaRect(c)
            (cx, cy), (w, h), angle = rect
            if w <= 0 or h <= 0:
                continue

            longer = max(w, h)
            shorter = min(w, h)
            ar = longer / shorter

            # Filter out very elongated shapes
            if ar > 2.0:
                continue

            box = cv2.boxPoints(rect)
            box = np.intp(box)

            # Visualization
            cv2.drawContours(display, [box], 0, (0, 255, 255), 2)
            cv2.circle(display, (int(cx), int(cy)), 4, (0, 0, 255), -1)
            cv2.putText(
                display,
                f"{label}_{color_name}",
                (int(cx) + 5, int(cy) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            # Axis-aligned bounding box
            x, y, w_bb, h_bb = cv2.boundingRect(c)

            # Publish raw detection (quad = cube face corners)
            self.publish_raw_detection(
                shape="cube",
                color=color_name,
                center_uv=(cx, cy),
                quad_points=box,
                bbox=(x, y, w_bb, h_bb),
                radius=0.0,
            )

    # ---------------- Main callback ----------------

    def image_callback(self, msg: CompressedImage):
        try:
            frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                return

            # ========== ROI (central 80%) ==========
            H, W = frame.shape[:2]
            roi_scale = 0.8  # central 80%

            roi_w = int(W * roi_scale)
            roi_h = int(H * roi_scale)

            x1 = (W - roi_w) // 2
            y1 = (H - roi_h) // 2
            x2 = x1 + roi_w
            y2 = y1 + roi_h

            # ROI mask: 255 inside ROI, 0 outside
            roi_mask = np.zeros((H, W), dtype=np.uint8)
            roi_mask[y1:y2, x1:x2] = 255

            # ========== base images for debug ==========
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(
                gray, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            cv2.imshow("Greyscale image", gray)
            cv2.imshow("Binary image", binary)
            cv2.imshow("HSV image", hsv)

            # ========== color masks (then apply ROI) ==========
            mask_r = self.get_red_mask(hsv)
            mask_g = self.get_green_mask(hsv)
            mask_b = self.get_blue_mask(hsv)
            mask_p = self.get_purple_mask(hsv)

            # Keep only central ROI
            mask_r = cv2.bitwise_and(mask_r, roi_mask)
            mask_g = cv2.bitwise_and(mask_g, roi_mask)
            mask_b = cv2.bitwise_and(mask_b, roi_mask)
            mask_p = cv2.bitwise_and(mask_p, roi_mask)

            cv2.imshow("RED Mask", mask_r)
            cv2.imshow("GREEN Mask", mask_g)
            cv2.imshow("BLUE Mask", mask_b)
            cv2.imshow("PURPLE Mask", mask_p)
            cv2.imshow("ROI Mask", roi_mask)

            # ========== shape detection ==========
            win_circle = frame.copy()
            win_cube = frame.copy()

            # Draw ROI rectangle (dark gray) on debug images
            cv2.rectangle(win_circle, (x1, y1), (x2, y2), (50, 50, 50), 2)
            cv2.rectangle(win_cube,   (x1, y1), (x2, y2), (50, 50, 50), 2)

            # Circles: only purple region + ROI
            self.detect_circles(frame, win_circle, mask_p)

            # Cubes: per-color masks (already ROI-limited)
            self.detect_cubes_from_mask(frame, mask_r, win_cube, color_name="red",   label="Cube")
            self.detect_cubes_from_mask(frame, mask_g, win_cube, color_name="green", label="Cube")
            self.detect_cubes_from_mask(frame, mask_b, win_cube, color_name="blue",  label="Cube")
            # If you ever have purple cubes, also call detect_cubes_from_mask(frame, mask_p, ...)

            cv2.imshow("Circle Detector", win_circle)
            cv2.imshow("Cube Detector", win_cube)

            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error in image_callback: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectorNode()
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
