#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/vision/vision/object_detector_node.py

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf_transformations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

# ==============================================================================
# Algorithm Logic: HSV Detection + 6D Pose Estimation
# ==============================================================================

@dataclass
class DetectedObject:
    color: str
    shape: str
    bbox: Tuple[int, int, int, int]     # x,y,w,h
    center: Tuple[int, int]            # cx,cy
    area: float
    yaw_deg: float                     # Principal direction in plane
    contour: np.ndarray

def _shape_from_contour(cnt: np.ndarray) -> str:
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    v = len(approx)
    if v == 4:
        return "quadrilateral"
    elif v > 4:
        return "circle_like"
    return "unknown"

def _yaw_from_contour(cnt: np.ndarray) -> float:
    rect = cv2.minAreaRect(cnt)  # ((cx,cy),(w,h),angle)
    (w, h) = rect[1]
    angle = rect[2]
    # OpenCV angle normalization for minAreaRect
    if w < h:
        yaw = angle
    else:
        yaw = angle + 90.0
    return float(yaw)

def detect_by_hsv(
    bgr: np.ndarray,
    hsv_lower: Tuple[int, int, int],
    hsv_upper: Tuple[int, int, int],
    color_name: str = "green",
    min_area: float = 800.0,
    morph_kernel: int = 5,
) -> Tuple[List[DetectedObject], np.ndarray]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array(hsv_lower, dtype=np.uint8)
    upper = np.array(hsv_upper, dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)
    # Morphology to remove noise
    k = np.ones((morph_kernel, morph_kernel), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: List[DetectedObject] = []
    
    for cnt in cnts:
        area = float(cv2.contourArea(cnt))
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cx = int(x + w / 2)
        cy = int(y + h / 2)
        shape = _shape_from_contour(cnt)
        yaw = _yaw_from_contour(cnt)

        out.append(DetectedObject(
            color=color_name, shape=shape, bbox=(x, y, w, h),
            center=(cx, cy), area=area, yaw_deg=yaw, contour=cnt
        ))
    return out, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

def estimate_grasp_face(obj: DetectedObject) -> Dict[str, Any]:
    """Determine the largest graspable face"""
    x, y, w, h = obj.bbox
    if w >= h:
        face = "wide"
    else:
        face = "tall"
    return {
        "face": face,          
        "yaw_deg": obj.yaw_deg,
        "shape": obj.shape
    }

# ==============================================================================
# ROS Node
# ==============================================================================

class ObjectDetectorNode(Node):
    def __init__(self):
        super().__init__("object_detector")

        # Parameters
        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("hsv_lower", [35, 80, 60])   
        self.declare_parameter("hsv_upper", [85, 255, 255])
        self.declare_parameter("min_area", 800.0)
        self.declare_parameter("camera_frame", "camera_optical_link")
        # For 6D estimation (approximate)
        self.declare_parameter("real_width_cm", 5.0)  # Known object width
        self.declare_parameter("focal_length_px", 600.0) # Approximate focal length

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.hsv_lower = tuple(int(x) for x in self.get_parameter("hsv_lower").value)
        self.hsv_upper = tuple(int(x) for x in self.get_parameter("hsv_upper").value)
        self.min_area = float(self.get_parameter("min_area").value)
        self.camera_frame = str(self.get_parameter("camera_frame").value)
        self.real_width_cm = float(self.get_parameter("real_width_cm").value)
        self.focal_length_px = float(self.get_parameter("focal_length_px").value)

        self.bridge = CvBridge()

        # Topic publishers
        self.pub_status = self.create_publisher(String, "/detect_object/status", 10)
        self.pub_grasp = self.create_publisher(String, "/detect_object/grasp_hint", 10)
        self.pub_vis = self.create_publisher(Image, "/detect_object/vis", 10)
        # 6D Pose Publisher
        self.pub_pose = self.create_publisher(PoseStamped, "/detect_object/target_pose_hint", 10)

        self.sub = self.create_subscription(Image, self.image_topic, self.cb, qos_profile_sensor_data)
        self.get_logger().info(f"[object_detector] Started. sub={self.image_topic}")

    def cb(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            return

        # 1. Core HSV detection
        objects, mask_bgr = detect_by_hsv(
            frame, self.hsv_lower, self.hsv_upper, "green", self.min_area
        )

        out_img = frame.copy()

        if not objects:
            self.pub_status.publish(String(data="NO_OBJECT"))
            self.pub_vis.publish(self.bridge.cv2_to_imgmsg(mask_bgr, "bgr8"))
            return

        # 2. Select the largest object
        target = max(objects, key=lambda o: o.area)

        # 3. Pose estimation (Approximation for 6D)
        # Z = (f * real_size) / pixel_size
        bbox_size_px = max(target.bbox[2], target.bbox[3]) # Use max dimension
        if bbox_size_px > 0:
            z_cm = (self.focal_length_px * self.real_width_cm) / bbox_size_px
            z_m = z_cm / 100.0
        else:
            z_m = 0.5 # Default fallback

        # X, Y relative to camera center
        h, w_img = frame.shape[:2]
        cx_img = w_img / 2.0
        cy_img = h / 2.0
        u, v = target.center
        
        # Pinhole model: x = (u - cx) * Z / f
        x_m = (u - cx_img) * z_m / self.focal_length_px
        y_m = (v - cy_img) * z_m / self.focal_length_px

        # 4. Grasp Analysis
        grasp = estimate_grasp_face(target)
        
        # 5. Publish Results
        
        # Status
        info_str = f"DETECT: {target.color} {target.shape} area={target.area:.0f} yaw={target.yaw_deg:.1f}"
        self.pub_status.publish(String(data=info_str))

        # Grasp Hint
        grasp_txt = f"GRASP_HINT face={grasp['face']} yaw_deg={grasp['yaw_deg']:.1f} shape={grasp['shape']}"
        self.pub_grasp.publish(String(data=grasp_txt))

        # 6D Pose
        ps = PoseStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = self.camera_frame
        ps.pose.position.x = float(x_m)
        ps.pose.position.y = float(y_m)
        ps.pose.position.z = float(z_m)
        
        # Convert yaw (planar rotation) to quaternion
        # Assuming table plane is perpendicular to Z-axis? 
        # Usually standard camera frame: Z forward, X right, Y down.
        # Object flat on table -> rotation around Z (standard assumption for top-down or angled view)
        q = tf_transformations.quaternion_from_euler(0, 0, math.radians(target.yaw_deg))
        ps.pose.orientation.x = q[0]
        ps.pose.orientation.y = q[1]
        ps.pose.orientation.z = q[2]
        ps.pose.orientation.w = q[3]
        
        self.pub_pose.publish(ps)

        # 6. Visualization
        x, y, w, h = target.bbox
        cv2.rectangle(out_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{target.shape} Z:{z_m:.2f}m"
        cv2.putText(out_img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(out_img, grasp_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        self.pub_vis.publish(self.bridge.cv2_to_imgmsg(out_img, "bgr8"))


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
