#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from std_msgs.msg import String
from servo_service.msg import InstructionAfterLLM
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import tf_transformations


# ============================ data model ============================

@dataclass
class RawDetection:
    shape: str
    color: str
    stamp: float  # seconds (ROS time)
    center_uv: Optional[Tuple[float, float]] = None
    quad: Optional[np.ndarray] = None          # (4,2)
    bbox: Optional[Tuple[float, float, float, float]] = None
    radius_px: float = 0.0


# ============================ helpers ============================
# 定义这个顺序：左上、右上、右下、左下（为了和 obj_pts 对应）
def order_quad_points(pts_4x2: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts_4x2, dtype=np.float64).reshape(4, 2)

    # 常见排序法：通过 (x+y) 和 (y-x) 的极值来区分四角
    s = pts.sum(axis=1)               # x+y
    d = (pts[:, 1] - pts[:, 0])       # y-x

    tl = pts[np.argmin(s)]            # 最小 x+y
    br = pts[np.argmax(s)]            # 最大 x+y
    tr = pts[np.argmin(d)]            # 最小 y-x  -> 右上（注意：依赖坐标系定义）
    bl = pts[np.argmax(d)]            # 最大 y-x  -> 左下

    return np.array([tl, tr, br, bl], dtype=np.float64)


def rvec_to_quat(rvec: np.ndarray) -> Tuple[float, float, float, float]:
    """solvePnP 的 rvec（可见面朝向）转成四元数 (x,y,z,w)，用于发布 cube 的 orientation。"""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    q = tf_transformations.quaternion_from_matrix(T)
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


# ============================ node ============================

class TargetObjectDetectorNode(Node):

    def __init__(self):
        super().__init__("target_object_detector_node")

        # ---------------- params ----------------
        self.declare_parameter("camera_frame", "camera_optical_link")
        self.declare_parameter("target_shape", "cube")
        self.declare_parameter("target_color", "green")

        # IMPORTANT: must match the real cube side length (meters)
        self.declare_parameter("cube_size_m", 0.035)

        # ---------- timing + filtering ----------
        # 只保留0.3s以内检测结果；然后在最新那批里只用最新50ms窗口
        self.declare_parameter("latest_window_s", 0.05)   # 50ms
        self.declare_parameter("fresh_timeout_s", 0.30)

        # EMA low-pass (position)
        # x_ema = (1-a)*x_ema + a*x_raw
        self.declare_parameter("ema_alpha", 0.10)         # 0.15~0.30 更灵敏但更抖

        # optional outlier gate
        self.declare_parameter("enable_outlier_gate", False)
        self.declare_parameter("max_jump_m", 0.05)        # 单次跳变>5cm就拒绝

        self.camera_frame = self.get_parameter("camera_frame").value

        # ---------------- state ----------------
        self.K = None            # 3x3 intrinsics
        self.D = None            # distortion coeffs (we will set to ZERO for undistorted image)
        self.have_caminfo = False

        self.detections: List[RawDetection] = []

        self.bridge = CvBridge()
        self.last_frame = None
        self.last_image_stamp = None

        self.last_status = ""
        self.last_pose = None

        self.target_shape = None
        self.target_color = None
        self.llm_shape = None
        self.llm_color = None

        # cube center continuity
        self.last_center_tvec = None

        # pick hysteresis
        self.last_picked_det: Optional[RawDetection] = None

        # EMA state
        self.ema_t = None
        self.last_raw_t = None

        # ---------------- pubs ----------------
        self.pub_pose = self.create_publisher(PoseStamped, "/detected_object_pose", 10)
        self.pub_status = self.create_publisher(String, "/detected_object/status", 10)
        self.pub_picked_pose = self.create_publisher(PoseStamped, "/picked_object_pose", 10)

        # ---------------- subs ----------------
        # camera_info 用 sensor_data QoS 是合理的（相机数据允许丢帧，追求低延迟）
        self.create_subscription(CameraInfo, "/camera_info", self.cb_caminfo, qos_profile_sensor_data)
        self.create_subscription(String, "/detected_objects_raw", self.cb_det_raw, 10)
        self.create_subscription(Image, "/camera/image_undistorted", self.cb_image, qos_profile_sensor_data)
        self.create_subscription(InstructionAfterLLM, "instruction_after_llm", self.cb_llm, 10)

        self.timer = self.create_timer(0.1, self.on_timer)
        cv2.namedWindow("TargetObjectDetector", cv2.WINDOW_NORMAL)

        self.get_logger().info("TargetObjectDetectorNode started (latest-window + uniq + EMA)")

    # ============================ time helpers ============================

    def ros_now_s(self) -> float:
        # 当前 ROS time（秒）
        return self.get_clock().now().nanoseconds / 1e9

    # ============================ callbacks ============================

    def cb_caminfo(self, msg: CameraInfo):
        if self.have_caminfo:
            return

        # K: 3x3 camera matrix
        self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)

        # IMPORTANT:
        # 你的输入图像是 /camera/image_undistorted（已经去畸变）所以 solvePnP 不应该再使用畸变参数，否则会“重复补偿”导致Z崩
        self.D = np.zeros((5, 1), dtype=np.float64)
        self.have_caminfo = True
        self.publish_status_once("CAMERA INFO OK ")


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

        stamp = float(data.get("stamp", self.ros_now_s()))

        det = RawDetection(
            shape=str(data.get("shape", "")).lower(),
            color=str(data.get("color", "")).lower(),
            stamp=stamp,
            radius_px=float(data.get("radius", 0.0)),
        )

        if "center_uv" in data and isinstance(data["center_uv"], list) and len(data["center_uv"]) == 2:
            det.center_uv = (float(data["center_uv"][0]), float(data["center_uv"][1]))

        if "quad" in data and isinstance(data["quad"], list) and len(data["quad"]) == 8:
            det.quad = np.array(data["quad"], dtype=np.float64).reshape(4, 2)

        if "bbox" in data and isinstance(data["bbox"], list) and len(data["bbox"]) == 4:
            det.bbox = (float(data["bbox"][0]), float(data["bbox"][1]),
                        float(data["bbox"][2]), float(data["bbox"][3]))

        self.detections.append(det)
        self.detections = self.detections[-200:]

    def cb_llm(self, msg: InstructionAfterLLM):
        self.llm_color = str(msg.object_color).lower()
        self.llm_shape = str(msg.object_shape).lower()
        self.get_logger().info(
            f"[LLM] target updated -> color={self.llm_color}, shape={self.llm_shape}"
        )

    # ============================ main loop ============================

    def on_timer(self):
        if not self.have_caminfo:
            self.publish_status_once("NO CAMERA")
            self.draw_debug(None, "NO CAMERA")
            return

        # prefer LLM instruction if available; fallback to params
        self.target_shape = self.llm_shape or self.get_parameter("target_shape").value.lower()
        self.target_color = self.llm_color or self.get_parameter("target_color").value.lower()

        now = self.ros_now_s()
        fresh_timeout = float(self.get_parameter("fresh_timeout_s").value)
        latest_window = float(self.get_parameter("latest_window_s").value)

        dets_fresh = [d for d in self.detections if (now - d.stamp) < fresh_timeout]
        self.detections = dets_fresh

        if not dets_fresh:
            self.publish_status_once("NO OBJECT")
            self.draw_debug(None, "NO OBJECT", fresh=[])
            return

        t_latest = max(d.stamp for d in dets_fresh)
        dets_latest = [d for d in dets_fresh if (t_latest - d.stamp) < latest_window]

        candidates = [d for d in dets_latest if d.shape == self.target_shape and d.color == self.target_color]
        uniq_candidates = self.merge_detections(candidates)

        status = None
        target = None
        rvec = None
        tvec = None
        pose = None

        if not dets_latest:
            status = "NO OBJECT"
        else:
            has_shape = any(d.shape == self.target_shape for d in dets_latest)
            has_color = any(d.color == self.target_color for d in dets_latest)

            target = self.pick_target_from_uniq(uniq_candidates, self.target_shape)

            if target is not None:
                status = "MATCH"

                if self.target_shape == "cube":
                    pose, rvec, tvec = self.pose_from_cube(target)
                elif self.target_shape == "circle":
                    pose, rvec, tvec = self.pose_from_circle(target)
                else:
                    pose = None

                if pose is not None and tvec is not None:
                    # EMA smoothing
                    t_filtered = self.filter_tvec(tvec.reshape(3))
                    pose.pose.position.x = float(t_filtered[0])
                    pose.pose.position.y = float(t_filtered[1])
                    pose.pose.position.z = float(t_filtered[2])   # <<< 不要再做 20/35 这种硬缩放
                    if self.target_shape == "cube":
                        pose.pose.position.z = (pose.pose.position.z-0.01)*2/3+0.01
                    self.pub_pose.publish(pose)
                    self.pub_picked_pose.publish(pose)
                    self.last_pose = pose
                else:
                    status = "POSE FAILED"

            elif has_color and not has_shape:
                status = "NO SHAPE"
            elif has_shape and not has_color:
                status = "NO COLOR"
            else:
                status = "NO MATCH"

        self.publish_status_once(status)
        self.draw_debug(target, status, rvec, tvec, fresh=dets_latest, uniq_candidates=uniq_candidates)

    # ============================ uniq merge ============================

    def merge_detections(self, dets: List[RawDetection]) -> List[RawDetection]:
        uniq: List[RawDetection] = []
        for d in dets:
            merged = False
            for i, u in enumerate(uniq):
                if self.same_det(d, u):
                    if d.shape == "circle":
                        if d.radius_px > u.radius_px:
                            uniq[i] = d
                    elif d.shape == "cube":
                        if self.bbox_area(d) > self.bbox_area(u):
                            uniq[i] = d
                    merged = True
                    break
            if not merged:
                uniq.append(d)
        return uniq

    # ============================ pick ============================

    def bbox_area(self, d: RawDetection) -> float:
        if d.bbox is None:
            return -1.0
        _, _, w, h = d.bbox
        return float(w) * float(h)

    def pick_target_from_uniq(self, uniq_matches: List[RawDetection], want_shape: str) -> Optional[RawDetection]:
        if not uniq_matches:
            self.last_picked_det = None
            return None

        hysteresis_gain = 1.2

        if want_shape == "circle":
            uniq_matches = [d for d in uniq_matches if d.radius_px is not None and d.radius_px > 1.0]
            if not uniq_matches:
                self.last_picked_det = None
                return None

            best = max(uniq_matches, key=lambda d: d.radius_px)

            prev = None
            if self.last_picked_det is not None:
                for d in uniq_matches:
                    if self.same_det(d, self.last_picked_det):
                        prev = d
                        break

            if prev is None:
                chosen = best
            else:
                chosen = prev if (best is prev or best.radius_px <= prev.radius_px * hysteresis_gain) else best

            self.last_picked_det = chosen
            return chosen

        if want_shape == "cube":
            best = max(uniq_matches, key=lambda d: self.bbox_area(d))

            prev = None
            if self.last_picked_det is not None:
                for d in uniq_matches:
                    if self.same_det(d, self.last_picked_det):
                        prev = d
                        break

            if prev is None:
                chosen = best
            else:
                best_area = self.bbox_area(best)
                prev_area = self.bbox_area(prev)
                chosen = best if (best is not prev and best_area > prev_area * hysteresis_gain) else prev

            self.last_picked_det = chosen
            return chosen

        chosen = uniq_matches[0]
        self.last_picked_det = chosen
        return chosen

    # ============================ pose estimation ============================
    def pose_from_cube(self, det: RawDetection):
        if det.bbox is None or det.center_uv is None:
            return None, None, None

        x, y, w, h = det.bbox
        u, v = det.center_uv

        # 使用 bbox 的“较短边”作为可见正方形边长（更稳）
        s_px = min(w, h)
        if s_px < 5:
            return None, None, None

        # cube 实际边长（你说这个是对的）
        S = float(self.get_parameter("cube_size_m").value)

        fx = float(self.K[0, 0])
        fy = float(self.K[1, 1])
        cx = float(self.K[0, 2])
        cy = float(self.K[1, 2])

        # 深度（核心）
        Z = fx * S / s_px

        # 反投影
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        # 合理性检查
        if Z < 0.05 or Z > 2.0:
            self.get_logger().warn(f"[cube] Unreasonable Z = {Z:.3f} m")
            return None, None, None

        ps = PoseStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = self.camera_frame

        ps.pose.position.x = float(X)
        ps.pose.position.y = float(Y)
        ps.pose.position.z = float(Z)

        # 不算姿态，给单位四元数
        ps.pose.orientation.x = 0.0
        ps.pose.orientation.y = 0.0
        ps.pose.orientation.z = 0.0
        ps.pose.orientation.w = 1.0

        tvec = np.array([[X], [Y], [Z]], dtype=np.float64)
        return ps, None, tvec


    def pose_from_circle(self, det: RawDetection):
        if det.center_uv is None or det.radius_px <= 1.0:
            return None, None, None

        u, v = det.center_uv
        r_px = float(det.radius_px)

        # physical radius (m)
        R_m = 0.0315

        fx = float(self.K[0, 0])
        fy = float(self.K[1, 1])
        cx = float(self.K[0, 2])
        cy = float(self.K[1, 2])

        Z = fx * R_m / r_px
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        # 合理性检查：Z 应该在 0.10m ~ 2.0m 范围内
        if Z < 0.10 or Z > 2.0:
            self.get_logger().warn(f"[pose_from_circle] Unreasonable Z = {Z:.3f} m (should be 0.10~2.0)")

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

    # ============================ filtering ============================

    def filter_tvec(self, t_raw: np.ndarray) -> np.ndarray:
        alpha = float(self.get_parameter("ema_alpha").value)
        enable_gate = bool(self.get_parameter("enable_outlier_gate").value)
        max_jump = float(self.get_parameter("max_jump_m").value)

        t_raw = np.asarray(t_raw, dtype=np.float64).reshape(3)

        if self.ema_t is None:
            self.ema_t = t_raw.copy()

        if self.last_raw_t is None:
            self.last_raw_t = t_raw.copy()

        if enable_gate:
            jump = np.linalg.norm(t_raw - self.last_raw_t)
            if jump > max_jump:
                self.get_logger().warn(f"[filter] outlier rejected: jump={jump:.3f} m > {max_jump:.3f} m")
                return self.ema_t

        self.last_raw_t = t_raw.copy()
        a = float(alpha)
        self.ema_t = (1.0 - a) * self.ema_t + a * t_raw
        return self.ema_t

    # ============================ debug / visualization ============================

    def same_det(self, a: RawDetection, b: RawDetection) -> bool:
        if a is None or b is None:
            return False
        if a.shape != b.shape:
            return False

        if a.shape == "circle":
            if a.center_uv is None or b.center_uv is None:
                return False
            ax, ay = a.center_uv
            bx, by = b.center_uv
            dist2 = (ax - bx) ** 2 + (ay - by) ** 2
            return dist2 < 20.0 ** 2

        if a.shape == "cube":
            if a.bbox is None or b.bbox is None:
                return False
            ax, ay, aw, ah = a.bbox
            bx, by, bw, bh = b.bbox
            acx, acy = ax + aw / 2.0, ay + ah / 2.0
            bcx, bcy = bx + bw / 2.0, by + bh / 2.0
            dist2 = (acx - bcx) ** 2 + (acy - bcy) ** 2
            size_ok = (abs(aw - bw) < 0.5 * max(aw, bw)) and (abs(ah - bh) < 0.5 * max(ah, bh))
            return (dist2 < 30.0 ** 2) and size_ok

        return False

    def draw_debug(self, det, note, rvec=None, tvec=None, fresh=None, uniq_candidates=None):
        if self.last_frame is None:
            return

        frame = self.last_frame.copy()

        text = f"target object = {self.target_color}, {self.target_shape}"
        (font_w, font_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        margin = 20
        tx = frame.shape[1] - margin - font_w
        ty = margin + font_h
        cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if note == "MATCH":
            c = (0, 255, 0)
        elif note in ["NO OBJECT", "NO COLOR", "NO SHAPE"]:
            c = (0, 255, 255)
        else:
            c = (0, 0, 255)

        cv2.putText(frame, str(note), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, c, 2)

        if uniq_candidates is not None:
            for d in uniq_candidates:
                if det is not None and self.same_det(d, det):
                    continue
                if d.shape == "circle" and d.center_uv is not None and d.radius_px > 1.0:
                    cx, cy = int(d.center_uv[0]), int(d.center_uv[1])
                    r = int(d.radius_px)
                    cv2.circle(frame, (cx, cy), r, (0, 255, 255), 2)
                    cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1)
                elif d.shape == "cube" and d.bbox is not None:
                    x, y, w, h = d.bbox
                    cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 255), 2)

        if det is not None:
            if det.shape == "circle" and det.center_uv is not None and det.radius_px > 1.0:
                cx, cy = int(det.center_uv[0]), int(det.center_uv[1])
                r = int(det.radius_px)
                cv2.circle(frame, (cx, cy), r, (0, 255, 0), 3)
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                # Circle 的 3D 中心投影
                if tvec is not None and self.K is not None and tvec.size >= 3:
                    X, Y, Z = float(tvec[0]), float(tvec[1]), float(tvec[2])
                    if Z > 1e-6:
                        fx, fy = self.K[0, 0], self.K[1, 1]
                        cx_k, cy_k = self.K[0, 2], self.K[1, 2]
                        u3 = int(fx * X / Z + cx_k)
                        v3 = int(fy * Y / Z + cy_k)
                        if 0 <= u3 < frame.shape[1] and 0 <= v3 < frame.shape[0]:
                            cv2.circle(frame, (u3, v3), 6, (255, 255, 0), 2)  # 青色圆环
                            cv2.putText(frame, "3D center", (u3 + 6, v3 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            elif det.shape == "cube" and det.bbox is not None:
                x, y, w, h = det.bbox
                # 最大外框（绿框）
                cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 3)
                # 发布用的 cube 真实 3D 中心投影到图像（绿点）
                if tvec is not None and self.K is not None and tvec.size >= 3:
                    X, Y, Z = float(tvec[0]), float(tvec[1]), float(tvec[2])
                    if Z > 1e-6:
                        fx, fy = self.K[0, 0], self.K[1, 1]
                        cx_k, cy_k = self.K[0, 2], self.K[1, 2]
                        u3 = int(fx * X / Z + cx_k)
                        v3 = int(fy * Y / Z + cy_k)
                        if 0 <= u3 < frame.shape[1] and 0 <= v3 < frame.shape[0]:
                            cv2.circle(frame, (u3, v3), 6, (0, 255, 0), 2)
                            cv2.putText(frame, "3D center", (u3 + 6, v3 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        if self.ema_t is not None:
            xyz_txt = f"XYZ = [{self.ema_t[0]:.3f}, {self.ema_t[1]:.3f}, {self.ema_t[2]:.3f}] m"
            cv2.putText(frame, xyz_txt, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            # 单独显示深度 Z（更醒目）
            z_txt = f"Depth Z = {self.ema_t[2]:.3f} m"
            z_color = (0, 255, 0) if 0.1 < self.ema_t[2] < 1.0 else (0, 0, 255)  # 绿色=正常，红色=异常
            cv2.putText(frame, z_txt, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, z_color, 2)

        cv2.imshow("TargetObjectDetector", frame)
        cv2.waitKey(1)

    # ============================ utils ============================

    def publish_status_once(self, text: str):
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
