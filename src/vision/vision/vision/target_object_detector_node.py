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

        # Z linear calibration (optional)
        self.declare_parameter("z_calib_a", 1.0)
        self.declare_parameter("z_calib_b", 0.0)

        # ---------- NEW: timing + filtering ----------
        # Use only detections close to the latest frame (avoid mixing ~0.3s history)
        self.declare_parameter("latest_window_s", 0.05)   # 50ms
        self.declare_parameter("fresh_timeout_s", 0.30)   # Maximum lifetime of raw detections (for dropping too-old data)

        # EMA low-pass filter (only for position)
        self.declare_parameter("ema_alpha", 0.10)         # 0.15~0.30 
        self.declare_parameter("enable_outlier_gate", False)
        self.declare_parameter("max_jump_m", 0.05)        # Max allowed jump per update (meters), tune by distance/FPS

        self.camera_frame = self.get_parameter("camera_frame").value

        # ---------------- state ----------------
        self.K = None
        self.D = None
        self.have_caminfo = False

        self.detections: List[RawDetection] = []

        self.bridge = CvBridge()
        self.last_frame = None
        self.last_image_stamp = None

        self.last_status = ""
        self.last_pose = None

        self.target_shape = None
        self.target_color = None

        # cube center continuity (existing logic)
        self.last_center_tvec = None  # np.array shape (3,)

        # MOD: target pick hysteresis (stickiness to previous target)
        self.last_picked_det: Optional[RawDetection] = None

        # EMA state
        self.ema_t = None  # np.array shape (3,)
        self.last_raw_t = None  # for outlier gate

        # ---------------- pubs ----------------
        self.pub_pose = self.create_publisher(PoseStamped, "/detected_object_pose", 10)
        self.pub_status = self.create_publisher(String, "/detected_object/status", 10)
        self.pub_picked_pose = self.create_publisher(PoseStamped, "/picked_object_pose", 10)

        # ---------------- subs ----------------
        self.create_subscription(CameraInfo, "/camera_info", self.cb_caminfo, qos_profile_sensor_data)
        self.create_subscription(String, "/detected_objects_raw", self.cb_det_raw, 10)
        self.create_subscription(Image, "/camera/image_undistorted", self.cb_image, qos_profile_sensor_data)

        self.timer = self.create_timer(0.1, self.on_timer)
        cv2.namedWindow("TargetObjectDetector", cv2.WINDOW_NORMAL)

        self.get_logger().info("TargetObjectDetectorNode started (Scheme B: latest-window + uniq + EMA)")

    # ============================ time helpers ============================

    def ros_now_s(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

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

        # IMPORTANT: use ROS time base only
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
        # keep a bit more; we will filter by time anyway
        self.detections = self.detections[-200:]

    # ============================ main loop ============================

    def on_timer(self):
        if not self.have_caminfo:
            self.publish_status_once("NO CAMERA")
            self.draw_debug(None, "NO CAMERA")
            return

        self.target_shape = self.get_parameter("target_shape").value.lower()
        self.target_color = self.get_parameter("target_color").value.lower()

        now = self.ros_now_s()
        fresh_timeout = float(self.get_parameter("fresh_timeout_s").value)
        latest_window = float(self.get_parameter("latest_window_s").value)

        dets_fresh = [d for d in self.detections if (now - d.stamp) < fresh_timeout]
        self.detections = dets_fresh  # shrink in-place

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

            # NOTE: pick only among uniq candidates
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
                    # apply Z calib first (optional)
                    pose = self.apply_z_linear_calib(pose)

                    # EMA + outlier gate on position (tvec)
                    t_filtered = self.filter_tvec(tvec.reshape(3))
                    pose.pose.position.x = float(t_filtered[0])
                    pose.pose.position.y = float(t_filtered[1])
                    pose.pose.position.z = float(t_filtered[2])*20/35

                    # publish
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


        # debug: show only latest-window dets + uniq candidates 
        self.draw_debug(target, status, rvec, tvec, fresh=dets_latest, uniq_candidates=uniq_candidates)

    # ============================ uniq merge ============================

    def merge_detections(self, dets: List[RawDetection]) -> List[RawDetection]:
        """
        Merge duplicates that likely correspond to the same physical object.
        circle: merge by center distance, keep larger radius
        cube  : merge by bbox center distance+size similarity, keep larger area
        """
        uniq: List[RawDetection] = []

        for d in dets:
            merged = False
            for i, u in enumerate(uniq):
                if self.same_det(d, u):
                    # keep the "better" one
                    if d.shape == "circle":
                        if d.radius_px > u.radius_px:
                            uniq[i] = d
                    elif d.shape == "cube":
                        # prefer larger bbox area if available
                        da = self.bbox_area(d)
                        ua = self.bbox_area(u)
                        if da > ua:
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
            # Reset hysteresis state when there is no candidate
            self.last_picked_det = None
            return None

        # MOD: simple hysteresis factor – new target must be 20% better to switch
        hysteresis_gain = 1.2

        if want_shape == "circle":
            # remove invalid radius
            uniq_matches = [d for d in uniq_matches if d.radius_px is not None and d.radius_px > 1.0]
            if not uniq_matches:
                # No valid circle detections this frame
                self.last_picked_det = None
                return None

            best = max(uniq_matches, key=lambda d: d.radius_px)

            # If we had a previous target and it still exists in this frame, apply hysteresis
            prev = None
            if self.last_picked_det is not None:
                for d in uniq_matches:
                    if self.same_det(d, self.last_picked_det):
                        prev = d
                        break

            if prev is None:
                # No previous target in current candidates -> just use best
                chosen = best
            else:
                # Only switch if best is significantly larger (radius > 1.2x)
                if best is prev or best.radius_px <= prev.radius_px * hysteresis_gain:
                    chosen = prev
                else:
                    chosen = best

            self.last_picked_det = chosen
            return chosen

        if want_shape == "cube":
            # prefer bbox area
            best = max(uniq_matches, key=lambda d: self.bbox_area(d))
            best_area = self.bbox_area(best)
            if best_area <= 0:
                # Fallback: will use quad area below
                prev = None
            else:
                # Hysteresis logic based on bbox area
                prev = None
                if self.last_picked_det is not None:
                    for d in uniq_matches:
                        if self.same_det(d, self.last_picked_det):
                            prev = d
                            break

                if prev is not None:
                    prev_area = self.bbox_area(prev)
                    if prev_area > 0 and best is not prev and best_area > prev_area * hysteresis_gain:
                        chosen = best
                    else:
                        chosen = prev
                    self.last_picked_det = chosen
                    return chosen
                else:
                    # No history – directly use best
                    self.last_picked_det = best
                    return best

            # fallback: quad polygon area
            def quad_area(d):
                if d.quad is None:
                    return -1.0
                pts = np.asarray(d.quad, dtype=np.float64).reshape(4, 2)
                x = pts[:, 0]; y = pts[:, 1]
                return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

            chosen = max(uniq_matches, key=lambda d: quad_area(d))
            self.last_picked_det = chosen
            return chosen

        # Other shapes: no hysteresis, just pick the first
        chosen = uniq_matches[0]
        self.last_picked_det = chosen
        return chosen

    # ============================ pose estimation ============================

    def pose_from_cube(self, det: RawDetection):
        if det.quad is None:
            return None, None, None

        img_pts = order_quad_points(det.quad)

        size = float(self.get_parameter("cube_size_m").value)
        h = size / 2.0
        obj_pts = np.array([
            [-h, -h, 0],
            [ h, -h, 0],
            [ h,  h, 0],
            [-h,  h, 0],
        ], dtype=np.float64)

        ok, rvec, tvec_face = cv2.solvePnP(
            obj_pts, img_pts, self.K, self.D,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if not ok:
            return None, None, None

        # Convert face center -> cube center with continuity (+h/-h)
        R, _ = cv2.Rodrigues(rvec)
        n_cam = R @ np.array([0.0, 0.0, 1.0])

        norm = np.linalg.norm(n_cam)
        if norm > 1e-9:
            n_cam = n_cam / norm

        t_face = tvec_face.reshape(3)
        t_center_plus  = t_face + n_cam * h
        t_center_minus = t_face - n_cam * h

        if self.last_center_tvec is None:
            # first frame: choose farther z
            t_center = t_center_plus if t_center_plus[2] > t_center_minus[2] else t_center_minus
        else:
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

        # you don't care orientation -> identity
        ps.pose.orientation.x = 0.0
        ps.pose.orientation.y = 0.0
        ps.pose.orientation.z = 0.0
        ps.pose.orientation.w = 1.0

        tvec_center = t_center.reshape(3, 1)
        return ps, rvec, tvec_center

    def pose_from_circle(self, det: RawDetection):
        if det.center_uv is None or det.radius_px <= 1.0:
            return None, None, None

        u, v = det.center_uv
        r_px = float(det.radius_px)

        # physical radius (m) of the ball/circle
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

    # ============================ filtering (EMA + outlier gate) ============================

    def filter_tvec(self, t_raw: np.ndarray) -> np.ndarray:
        """
        t_raw: shape (3,)
        returns filtered tvec (3,)
        """
        alpha = float(self.get_parameter("ema_alpha").value)
        enable_gate = bool(self.get_parameter("enable_outlier_gate").value)
        # enable_gate = False
        max_jump = float(self.get_parameter("max_jump_m").value)
        t_raw = np.asarray(t_raw, dtype=np.float64).reshape(3)
        if self.ema_t is None:
            self.ema_t = t_raw

        # Outlier gate (compare raw jumps)
        if self.last_raw_t is None:
            self.last_raw_t = t_raw
        if enable_gate and self.last_raw_t is not None:
            jump = np.linalg.norm(t_raw - self.last_raw_t)
            if jump > max_jump:
                # reject this update: return previous EMA if exists, else last raw
                self.get_logger().warn(f"[filter] outlier rejected: jump={jump:.3f} m > {max_jump:.3f} m")
                if self.ema_t is not None:
                    return self.ema_t
                return self.last_raw_t
        else:
            self.last_raw_t = t_raw

            a = float(alpha)
            self.ema_t = (1.0 - a) * self.ema_t + a * t_raw

        return self.ema_t
    # ============================ debug / visualization ============================

    def same_det(self, a: RawDetection, b: RawDetection) -> bool:
        if a is None or b is None:
            return False
        if a.shape != b.shape:
            return False

        # circle: center distance
        if a.shape == "circle":
            if a.center_uv is None or b.center_uv is None:
                return False
            ax, ay = a.center_uv
            bx, by = b.center_uv
            dist2 = (ax - bx) ** 2 + (ay - by) ** 2
            return dist2 < 20.0 ** 2  # 20 px

        # cube: bbox center distance + size similarity
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

        # target definition
        text = f"target object = {self.target_color}, {self.target_shape}"
        (font_w, font_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        margin = 20
        tx = frame.shape[1] - margin - font_w
        ty = margin + font_h
        cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # status color
        if note == "MATCH":
            c = (0, 255, 0)
        elif note in ["NO OBJECT", "NO COLOR", "NO SHAPE"]:
            c = (0, 255, 255)
        else:
            c = (0, 0, 255)

        cv2.putText(frame, str(note), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, c, 2)

        # Yellow: draw uniq candidates (same shape/color) except picked one
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
                    cv2.rectangle(frame, (int(x), int(y)),
                                  (int(x + w), int(y + h)), (0, 255, 255), 2)

        # Green: draw picked one
        if det is not None:
            if det.shape == "circle" and det.center_uv is not None and det.radius_px > 1.0:
                cx, cy = int(det.center_uv[0]), int(det.center_uv[1])
                r = int(det.radius_px)
                cv2.circle(frame, (cx, cy), r, (0, 255, 0), 3)
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            elif det.shape == "cube" and det.bbox is not None:
                x, y, w, h = det.bbox
                cv2.rectangle(frame, (int(x), int(y)),
                              (int(x + w), int(y + h)), (0, 255, 0), 3)

        # Optional overlay: show filtered XYZ
        if self.ema_t is not None:
            xyz_txt = f"EMA XYZ = [{self.ema_t[0]:.3f}, {self.ema_t[1]:.3f}, {self.ema_t[2]:.3f}] m"
            cv2.putText(frame, xyz_txt, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow("TargetObjectDetector", frame)
        cv2.waitKey(1)

    # ============================ utils ============================

    def publish_status_once(self, text: str):
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