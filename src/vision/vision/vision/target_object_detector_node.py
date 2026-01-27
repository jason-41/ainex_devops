#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from std_msgs.msg import String
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import tf_transformations


# ============================ data model ============================

@dataclass
class RawDetection:
    shape: str
    color: str
    stamp: float
    center_uv: Optional[Tuple[float, float]] = None
    quad: Optional[np.ndarray] = None          # (4,2)
    bbox: Optional[Tuple[float, float, float, float]] = None
    radius_px: float = 0.0


# ============================ helpers ============================

def order_quad_points(pts_4x2: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts_4x2, dtype=np.float64).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float64)


def rvec_to_quat(rvec: np.ndarray):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    return tf_transformations.quaternion_from_matrix(T)


# ============================ node ============================

class TargetObjectDetectorNode(Node):

    def __init__(self):
        super().__init__("target_object_detector_node")

        # ---------------- params ----------------
        self.declare_parameter("camera_frame", "camera_optical_link")
        self.declare_parameter("target_shape", "cube")
        self.declare_parameter("target_color", "red")
        self.declare_parameter("cube_size_m", 0.035)

        self.camera_frame = self.get_parameter("camera_frame").value

        # ---------------- state ----------------
        self.K = None
        self.D = None
        self.have_caminfo = False

        self.detections: List[RawDetection] = []

        self.bridge = CvBridge()
        self.last_frame = None

        self.last_status = ""

        # ---------------- pubs ----------------
        self.pub_pose = self.create_publisher(
            PoseStamped, "/detected_object_pose", 10
        )
        self.pub_status = self.create_publisher(
            String, "/detected_object/status", 10
        )

        # ---------------- subs ----------------
        self.create_subscription(
            CameraInfo, "/camera_info", self.cb_caminfo, qos_profile_sensor_data
        )
        self.create_subscription(
            String, "/detected_objects_raw", self.cb_det_raw, 10
        )
        self.create_subscription(
            Image, "/camera/image_undistorted", self.cb_image, qos_profile_sensor_data
        )

        self.timer = self.create_timer(0.1, self.on_timer)

        cv2.namedWindow("TargetObjectDetector", cv2.WINDOW_NORMAL)

        self.get_logger().info("TargetObjectDetectorNode started")

    # ============================ callbacks ============================

    def cb_caminfo(self, msg: CameraInfo):
        if self.have_caminfo:
            return
        self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.D = np.array(msg.d, dtype=np.float64).reshape(-1, 1) if msg.d else np.zeros((5, 1))
        self.have_caminfo = True
        self.publish_status_once("CAMERA INFO OK")

    def cb_image(self, msg: Image):
        try:
            self.last_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            pass

    def cb_det_raw(self, msg: String):
        try:
            data = json.loads(msg.data)
        except Exception:
            return

        det = RawDetection(
            shape=str(data.get("shape", "")).lower(),
            color=str(data.get("color", "")).lower(),
            stamp=float(data.get("stamp", time.time())),
            radius_px=float(data.get("radius", 0.0)),
        )

        if "center_uv" in data and len(data["center_uv"]) == 2:
            det.center_uv = tuple(data["center_uv"])

        if "quad" in data and len(data["quad"]) == 8:
            det.quad = np.array(data["quad"], dtype=np.float64).reshape(4, 2)

        if "bbox" in data and len(data["bbox"]) == 4:
            det.bbox = tuple(data["bbox"])

        self.detections.append(det)
        self.detections = self.detections[-50:]

    # ============================ main loop ============================

    def on_timer(self):
        if not self.have_caminfo:
            self.publish_status_once("NO CAMERA")
            return

        target_shape = self.get_parameter("target_shape").value.lower()
        target_color = self.get_parameter("target_color").value.lower()

        fresh = [d for d in self.detections if time.time() - d.stamp < 0.3]

        if not fresh:
            self.publish_status_once("NO OBJECT")
            self.draw_debug(None, "NO OBJECT")
            return

        target = self.pick_target(fresh, target_shape, target_color)
        if target is None:
            self.publish_status_once("NO MATCH")
            self.draw_debug(None, "NO MATCH")
            return

        pose, rvec, tvec = self.pose_from_cube(target)
        if pose is None:
            self.publish_status_once("POSE FAILED")
            return

        self.pub_pose.publish(pose)
        self.publish_status_once("FOUND")

        self.draw_debug(target, None, rvec, tvec)

    # ============================ logic ============================

    def pick_target(self, dets, want_shape, want_color):
        for d in dets:
            if d.shape == want_shape and d.color == want_color:
                return d
        return None

    def pose_from_cube(self, det: RawDetection):
        if det.quad is None:
            return None, None, None

        img_pts = order_quad_points(det.quad)

        size = self.get_parameter("cube_size_m").value
        h = size / 2.0
        obj_pts = np.array([
            [-h, -h, 0],
            [ h, -h, 0],
            [ h,  h, 0],
            [-h,  h, 0],
        ], dtype=np.float64)

        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, self.K, self.D,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if not ok:
            return None, None, None

        qx, qy, qz, qw = rvec_to_quat(rvec)

        ps = PoseStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = self.camera_frame
        ps.pose.position.x = float(tvec[0])
        ps.pose.position.y = float(tvec[1])
        ps.pose.position.z = float(tvec[2])
        ps.pose.orientation.x = qx
        ps.pose.orientation.y = qy
        ps.pose.orientation.z = qz
        ps.pose.orientation.w = qw

        return ps, rvec, tvec

    # ============================ debug ============================

    def draw_debug(self, det, note, rvec=None, tvec=None):
        if self.last_frame is None:
            return

        frame = self.last_frame.copy()

        if note:
            cv2.putText(frame, note, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        if det and det.bbox:
            x, y, w, h = det.bbox
            cv2.rectangle(frame, (int(x), int(y)),
                          (int(x+w), int(y+h)), (0, 255, 0), 2)

        if rvec is not None and tvec is not None:
            self.draw_axes(frame, rvec, tvec)

        cv2.imshow("TargetObjectDetector", frame)
        cv2.waitKey(1)

    def draw_axes(self, frame, rvec, tvec):
        L = self.get_parameter("cube_size_m").value * 1.5
        pts_3d = np.float32([
            [0, 0, 0],
            [L, 0, 0],
            [0, L, 0],
            [0, 0, L],
        ]).reshape(-1, 1, 3)

        pts_2d, _ = cv2.projectPoints(pts_3d, rvec, tvec, self.K, self.D)
        pts = pts_2d.reshape(-1, 2).astype(int)

        o, x, y, z = pts
        cv2.line(frame, tuple(o), tuple(x), (0, 0, 255), 3)
        cv2.line(frame, tuple(o), tuple(y), (0, 255, 0), 3)
        cv2.line(frame, tuple(o), tuple(z), (255, 0, 0), 3)

    # ============================ utils ============================

    def publish_status_once(self, text):
        if text != self.last_status:
            self.pub_status.publish(String(data=text))
            self.last_status = text


def main(args=None):
    rclpy.init(args=args)
    node = TargetObjectDetectorNode()
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
