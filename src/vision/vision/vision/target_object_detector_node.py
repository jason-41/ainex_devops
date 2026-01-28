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
        self.declare_parameter("target_shape", "circle")
        self.declare_parameter("target_color", "purple")
        self.declare_parameter("cube_size_m", 0.035)

        self.declare_parameter("z_calib_a", 1.0)
        self.declare_parameter("z_calib_b", 0.0)


        self.camera_frame = self.get_parameter("camera_frame").value
        

        # ---------------- state ----------------
        self.K = None
        self.D = None
        self.have_caminfo = False

        self.detections: List[RawDetection] = []

        self.bridge = CvBridge()
        self.last_frame = None
        self.last_image_stamp = None
        self.last_center_tvec = None  

        self.last_status = ""
        self.last_pose = None
        self.target_shape = None
        self.target_color = None

        # ---------------- pubs ----------------
        self.pub_pose = self.create_publisher(
            PoseStamped, "/detected_object_pose", 10
        )
        self.pub_status = self.create_publisher(
            String, "/detected_object/status", 10
        )
        self.pub_picked_pose = self.create_publisher(
            PoseStamped, "/picked_object_pose", 10
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
            self.last_image_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
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
            self.draw_debug(None, "NO CAMERA")
            return

        self.target_shape = self.get_parameter("target_shape").value.lower()
        self.target_color = self.get_parameter("target_color").value.lower()

        fresh = [d for d in self.detections if time.time() - d.stamp < 0.3]

        status = None
        target = None
        rvec = None
        tvec = None
        pose = None

        if not fresh:
            status = "NO OBJECT"

        else:
            has_shape = any(d.shape == self.target_shape for d in fresh)
            has_color = any(d.color == self.target_color for d in fresh)

            target = self.pick_target(fresh, self.target_shape, self.target_color)

            if target is not None:
                status = "MATCH"

                if self.target_shape == "cube":
                    pose, rvec, tvec = self.pose_from_cube(target)
                elif self.target_shape == "circle":
                    pose, rvec, tvec = self.pose_from_circle(target)

                if pose is not None:
                    pose = self.apply_z_linear_calib(pose)
                    self.pub_pose.publish(pose)  # /detected_object_pose
                    self.pub_picked_pose.publish(pose)  # /picked_object_pose
                    self.last_pose = pose
                else:
                    status = "POSE FAILED"


            elif has_color and not has_shape:
                status = "NO SHAPE"

            elif has_shape and not has_color:
                status = "NO COLOR"

            else:
                status = "NO MATCH"

            # ---------- 统一出口 ----------
        self.publish_status_once(status)
        self.draw_debug(target, status, rvec, tvec, fresh=fresh)

    # ============================ logic ============================

    def pick_target(self, dets, want_shape, want_color):
        # 1) 先筛出所有 match
        matches = [d for d in dets if d.shape == want_shape and d.color == want_color]
        if not matches:
            return None

        # 2) circle：选半径最大的（最近）
        if want_shape == "circle":
            # radius_px 可能为 0 的要排掉
            matches = [d for d in matches if d.radius_px is not None and d.radius_px > 1.0]
            if not matches:
                return None
            return max(matches, key=lambda d: d.radius_px)

        # 3) cube：选 bbox 面积最大的（最近）
        if want_shape == "cube":
            def bbox_area(d):
                if d.bbox is None:
                    return -1.0
                _, _, w, h = d.bbox
                return float(w) * float(h)

            # bbox 没有的话就退化：按 quad 面积（鞋带公式）估
            def quad_area(d):
                if d.quad is None:
                    return -1.0
                pts = np.asarray(d.quad, dtype=np.float64).reshape(4, 2)
                x = pts[:, 0]; y = pts[:, 1]
                return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

            # 优先 bbox
            best = max(matches, key=lambda d: bbox_area(d))
            if bbox_area(best) > 0:
                return best

            # bbox 全没，就用 quad
            return max(matches, key=lambda d: quad_area(d))

        # 4) 其他 shape：先随便返回一个（或你以后再扩展）
        return matches[0]


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
        
        # ---- 把“面中心” -> “立方体中心（稳定版）” ----
        size = float(self.get_parameter("cube_size_m").value)
        h = size / 2.0

        R, _ = cv2.Rodrigues(rvec)                 # 3x3
        n_cam = R @ np.array([0.0, 0.0, 1.0])      # 面法向在相机坐标系

        # 1) 归一化（非常重要）
        norm = np.linalg.norm(n_cam)
        if norm > 1e-9:
            n_cam = n_cam / norm

        t_center_plus  = tvec.reshape(3) + n_cam * h
        t_center_minus = tvec.reshape(3) - n_cam * h

        # 2) 连续性选择（防止旋转时 +h / -h 来回跳）
        if self.last_center_tvec is None:
            # 第一帧：选 Z 更大的（更远离相机）
            t_center = t_center_plus if t_center_plus[2] > t_center_minus[2] else t_center_minus
        else:
            # 后续：选“离上一帧更近”的那个
            dp = np.linalg.norm(t_center_plus  - self.last_center_tvec)
            dm = np.linalg.norm(t_center_minus - self.last_center_tvec)
            t_center = t_center_plus if dp < dm else t_center_minus

        self.last_center_tvec = t_center.copy()

        ps = PoseStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = self.camera_frame
        ps.pose.position.x = float(t_center[0])
        ps.pose.position.y = float(t_center[1])
        ps.pose.position.z = float(t_center[2])

        # 你现在只关心中心，不关心姿态 → 固定姿态（非常对）
        ps.pose.orientation.x = 0.0
        ps.pose.orientation.y = 0.0
        ps.pose.orientation.z = 0.0
        ps.pose.orientation.w = 1.0

        return ps, rvec, t_center.reshape(3, 1)



    def pose_from_circle(self, det: RawDetection):
        #center + radius
        if det.center_uv is None or det.radius_px <= 1.0:
            return None, None, None

        u, v = det.center_uv
        r_px = float(det.radius_px)

        R_m = 0.0315

        fx = float(self.K[0, 0])
        fy = float(self.K[1, 1])
        cx = float(self.K[0, 2])
        cy = float(self.K[1, 2])

    
        Z = fx * R_m / r_px
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        ps = PoseStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = self.camera_frame
        ps.pose.position.x = float(X)
        ps.pose.position.y = float(Y)
        ps.pose.position.z = float(Z)

        ps.pose.orientation.x = 0.0
        ps.pose.orientation.y = 0.0
        ps.pose.orientation.z = 0.0
        ps.pose.orientation.w = 1.0

        tvec = np.array([[X], [Y], [Z]], dtype=np.float64)
        return ps, None, tvec


    # ============================ debug ============================
    def same_det(self, a: RawDetection, b: RawDetection) -> bool:
        """
        判断两个 detection 是否是同一个目标，用于避免黄/绿叠加。
        circle: 用 center 距离判断
        cube  : 用 bbox IOU 判断（简化版：中心距离 + 尺寸相近）
        """
        if a is None or b is None:
            return False
        if a.shape != b.shape:
            return False

        # ----- circle -----
        if a.shape == "circle":
            if a.center_uv is None or b.center_uv is None:
                return False
            ax, ay = a.center_uv
            bx, by = b.center_uv
            dist2 = (ax - bx) ** 2 + (ay - by) ** 2
            # 阈值：像素距离小于 15 px 认为同一个（你可调大一点如 25）
            return dist2 < 15.0 ** 2

        # ----- cube -----
        if a.shape == "cube":
            if a.bbox is None or b.bbox is None:
                return False
            ax, ay, aw, ah = a.bbox
            bx, by, bw, bh = b.bbox

            acx, acy = ax + aw / 2.0, ay + ah / 2.0
            bcx, bcy = bx + bw / 2.0, by + bh / 2.0
            dist2 = (acx - bcx) ** 2 + (acy - bcy) ** 2

            # 中心距离阈值 + 尺寸相近（防止误判）
            size_ok = (abs(aw - bw) < 0.4 * max(aw, bw)) and (abs(ah - bh) < 0.4 * max(ah, bh))
            return (dist2 < 25.0 ** 2) and size_ok

        return False


    def draw_debug(self, det, note, rvec=None, tvec=None, fresh=None):
        if self.last_frame is None:
            return

        frame = self.last_frame.copy()

        # ===== Target definition (top-right, single line) =====
        text = f"target object = {self.target_color}, {self.target_shape}"
        (font_w, font_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        margin = 20
        tx = frame.shape[1] - margin - font_w
        ty = margin + font_h
        cv2.putText(frame, text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # ===== status color =====
        if note == "MATCH":
            c = (0, 255, 0)          # green
        elif note in ["NO OBJECT", "NO COLOR", "NO SHAPE"]:
            c = (0, 255, 255)        # yellow
        else:
            c = (0, 0, 255)          # red  (NO MATCH / POSE FAILED / fallback)

        cv2.putText(frame, str(note), (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, c, 2)

        # =====================================================================
        # 1) Yellow: draw ALL candidates that match target (except the picked one)
        # =====================================================================
        if fresh is not None:
            candidates = [
                d for d in fresh
                if d.shape == self.target_shape and d.color == self.target_color
            ]

            # ===== 去重：同一个物体只保留一个 detection =====
            uniq = []
            for d in candidates:
                merged = False
                for i, u in enumerate(uniq):
                    if self.same_det(d, u):
                        # 同一个物体：保留“更好”的那条（circle 用半径更大；cube 用面积更大）
                        if d.shape == "circle":
                            if d.radius_px > u.radius_px:
                                uniq[i] = d
                        elif d.shape == "cube" and d.bbox and u.bbox:
                            if d.bbox[2] * d.bbox[3] > u.bbox[2] * u.bbox[3]:
                                uniq[i] = d
                        merged = True
                        break
                if not merged:
                    uniq.append(d)

            for d in uniq:
                if det is not None and self.same_det(d, det):
                    continue
    
                # --- draw yellow candidate ---
                if d.shape == "circle" and d.center_uv is not None and d.radius_px > 1.0:
                    cx, cy = int(d.center_uv[0]), int(d.center_uv[1])
                    r = int(d.radius_px)
                    cv2.circle(frame, (cx, cy), r, (0, 255, 255), 2)   # yellow outer
                    cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1)    # small red dot

                elif d.shape == "cube" and d.bbox is not None:
                    x, y, w, h = d.bbox
                    cv2.rectangle(frame, (int(x), int(y)),
                                  (int(x + w), int(y + h)), (0, 255, 255), 2)  # yellow bbox

        # ======================================================
        # 2) Green: draw ONLY the selected target (pick_target)
        # ======================================================
        if det is not None:
            if det.shape == "circle" and det.center_uv is not None and det.radius_px > 1.0:
                cx, cy = int(det.center_uv[0]), int(det.center_uv[1])
                r = int(det.radius_px)
                cv2.circle(frame, (cx, cy), r, (0, 255, 0), 3)   # green outer
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)  # red center

            elif det.shape == "cube" and det.bbox is not None:
                x, y, w, h = det.bbox
                cv2.rectangle(frame, (int(x), int(y)),
                              (int(x + w), int(y + h)), (0, 255, 0), 3)  # green bbox

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
        # Helper to convert numpy points to python int tuples for cv2.line
        # OpenCV handles python ints better than numpy scalars in some versions
        pt_o = (int(o[0]), int(o[1]))
        pt_x = (int(x[0]), int(x[1]))
        pt_y = (int(y[0]), int(y[1]))
        pt_z = (int(z[0]), int(z[1]))

        cv2.line(frame, pt_o, pt_x, (0, 0, 255), 3)
        cv2.line(frame, pt_o, pt_y, (0, 255, 0), 3)
        cv2.line(frame, pt_o, pt_z, (255, 0, 0), 3)

    # ============================ utils ============================

    def publish_status_once(self, text):
        if text != self.last_status:
            self.pub_status.publish(String(data=text))
            self.last_status = text

     # ============================ calibration ============================

    def apply_z_linear_calib(self, ps: PoseStamped) -> PoseStamped:
        a = float(self.get_parameter("z_calib_a").value)
        b = float(self.get_parameter("z_calib_b").value)

        X = float(ps.pose.position.x)
        Y = float(ps.pose.position.y)
        Z = float(ps.pose.position.z)


        if abs(Z) < 1e-6:
            return ps

        Zc = a * Z + b

        scale = Zc / Z
        ps.pose.position.x = X * scale
        ps.pose.position.y = Y * scale
        ps.pose.position.z = Zc
        return ps


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
