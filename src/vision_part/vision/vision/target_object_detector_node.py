#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scene 2: Detect target cube of given color and output 6D pose.

- 使用 camera_optical_frame / camera_optical_link 作为姿态坐标系
- 立方体边长 s = 3.5cm，用正方形面做 solvePnP
- RPY = intrinsic XYZ（与 aruco_detector_node / tf_debug_publisher_node 一致）
- Z = camera → object（tvec[2]）

Topics:
  输入
    /camera_image          (sensor_msgs/Image)
    /camera_info           (sensor_msgs/CameraInfo)

  输出
    /detected_object_pose          (geometry_msgs/PoseStamped)
    /detect_object/target_pose_hint(geometry_msgs/PoseStamped, 兼容原接口)
    /detect_object/status          (std_msgs/String, 文本状态)
    /detect_object/grasp_hint      (std_msgs/String, 简单提示)
    /detect_object/shape           (std_msgs/String, object_shape)
    /detect_object/color           (std_msgs/String, object_color)
"""

import math
import time
from typing import Optional, List, Dict, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from cv_bridge import CvBridge

import tf_transformations


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class TargetObjectDetectorNode(Node):
    def __init__(self):
        super().__init__("target_object_detector_node")

        # ===== Target config（上层可改）=====
        # target_shape: "cube" / "circle" / "rectangle" / "any"
        # target_color: "green" / "red" / "blue" / "any"
        self.target_shape: str = "cube"
        self.target_color: str = "green"

        # ===== Parameters =====
        self.declare_parameter("image_topic", "/camera_image")
        self.declare_parameter("camera_info_topic", "/camera_info")
        self.declare_parameter("camera_frame", "camera_optical_link")

        # 立方体面边长（m）：3.5 cm
        self.declare_parameter("square_size_m", 0.035)

        self.declare_parameter("ema_alpha", 0.1)
        self.declare_parameter("publish_hz", 5.0)

        # 显示频率（可关闭窗口时影响小）
        self.declare_parameter("show_hz", 10.0)
        self.declare_parameter("mask_show_hz", 10.0)

        # HSV 阈值（假设是 target_color，用 YAML / param 调整）
        self.declare_parameter("h_low", 35)
        self.declare_parameter("s_low", 60)
        self.declare_parameter("v_low", 40)
        self.declare_parameter("h_high", 85)
        self.declare_parameter("s_high", 255)
        self.declare_parameter("v_high", 255)

        # ROI：从某个 y 比例往下开始（比如 0.25）
        self.declare_parameter("use_roi", True)
        self.declare_parameter("roi_y_start_ratio", 0.25)

        # 过滤参数
        self.declare_parameter("min_mask_area_px", 800.0)
        self.declare_parameter("min_rect_side_px", 10.0)

        # 坐标轴长度
        self.declare_parameter("axis_len_m", 0.05)

        # ===== Load params =====
        self.image_topic = str(self.get_parameter("image_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self.camera_frame = str(self.get_parameter("camera_frame").value)

        self.target_size_m = float(self.get_parameter("square_size_m").value)

        self.alpha = float(self.get_parameter("ema_alpha").value)
        self.publish_hz = float(self.get_parameter("publish_hz").value)
        self.publish_period = 1.0 / max(self.publish_hz, 1e-6)

        self.show_hz = float(self.get_parameter("show_hz").value)
        self.show_period = 1.0 / max(self.show_hz, 1e-6)
        self.mask_show_hz = float(self.get_parameter("mask_show_hz").value)
        self.mask_show_period = 1.0 / max(self.mask_show_hz, 1e-6)

        self.use_roi = bool(self.get_parameter("use_roi").value)
        self.roi_y_start_ratio = float(self.get_parameter("roi_y_start_ratio").value)

        self.min_mask_area_px = float(self.get_parameter("min_mask_area_px").value)
        self.min_rect_side_px = float(self.get_parameter("min_rect_side_px").value)

        self.axis_len_m = float(self.get_parameter("axis_len_m").value)

        # ===== Camera intrinsics =====
        self.fx: Optional[float] = None
        self.fy: Optional[float] = None
        self.cx: Optional[float] = None
        self.cy: Optional[float] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.have_caminfo: bool = False

        # ===== State =====
        self.bridge = CvBridge()
        self.pos_filt: Optional[np.ndarray] = None
        self.roll_filt_deg: Optional[float] = None
        self.pitch_filt_deg: Optional[float] = None
        self.yaw_filt_deg: Optional[float] = None

        self.last_pub = 0.0
        self.last_show = 0.0
        self.last_mask_show = 0.0
        self.last_state: Optional[str] = None

        # ===== Publishers =====
        self.pub_status = self.create_publisher(String, "/detect_object/status", 10)
        self.pub_grasp = self.create_publisher(String, "/detect_object/grasp_hint", 10)

        # 原有接口（兼容）
        self.pub_pose_hint = self.create_publisher(PoseStamped, "/detect_object/target_pose_hint", 10)

        # 新接口：真正给控制的 6D pose
        self.pub_pose = self.create_publisher(PoseStamped, "/detected_object_pose", 10)

        # 直接把 object_shape / object_color 也发出去
        self.pub_shape = self.create_publisher(String, "/detect_object/shape", 10)
        self.pub_color = self.create_publisher(String, "/detect_object/color", 10)

        # ===== Subscribers =====
        self.create_subscription(CameraInfo, self.camera_info_topic, self.cb_info, qos_profile_sensor_data)
        self.create_subscription(Image, self.image_topic, self.cb_img, qos_profile_sensor_data)

        # ===== Windows =====
        cv2.namedWindow("Object Detection (Target)", cv2.WINDOW_NORMAL)
        cv2.namedWindow("HSV MASK (Target Color)", cv2.WINDOW_NORMAL)

        self.get_logger().info(
            f"[target_object_detector_node] Started. image={self.image_topic}, info={self.camera_info_topic}"
        )

    # ------------------------------------------------------------------
    # Helpers: target shape 名字统一
    # ------------------------------------------------------------------
    def normalize_target_shape(self) -> str:
        ts = (self.target_shape or "").lower()
        if ts in ("cube", "square", "block"):
            return "square"
        if ts in ("rect", "rectangle", "cuboid"):
            return "rectangle"
        if ts in ("circle", "ball", "sphere"):
            return "circle"
        if ts in ("any", ""):
            return "any"
        return ts

    # ------------------------------------------------------------------
    # CameraInfo callback
    # ------------------------------------------------------------------
    def cb_info(self, msg: CameraInfo):
        if self.have_caminfo:
            return

        K = msg.k  # 3x3 row-major
        self.fx = float(K[0])
        self.fy = float(K[4])
        self.cx = float(K[2])
        self.cy = float(K[5])

        if msg.d:
            self.dist_coeffs = np.array(msg.d, dtype=np.float64).reshape(-1, 1)
        else:
            self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)

        self.have_caminfo = True
        self.get_logger().info(
            f"[target_object_detector_node] Got camera intrinsics: fx={self.fx:.1f}, fy={self.fy:.1f}, "
            f"cx={self.cx:.1f}, cy={self.cy:.1f}"
        )

    # ------------------------------------------------------------------
    # HSV mask for target_color
    # ------------------------------------------------------------------
    def get_color_mask(self, hsv: np.ndarray) -> np.ndarray:
        h_low = int(self.get_parameter("h_low").value)
        s_low = int(self.get_parameter("s_low").value)
        v_low = int(self.get_parameter("v_low").value)
        h_high = int(self.get_parameter("h_high").value)
        s_high = int(self.get_parameter("s_high").value)
        v_high = int(self.get_parameter("v_high").value)

        lower = np.array([h_low, s_low, v_low], dtype=np.uint8)
        upper = np.array([h_high, s_high, v_high], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

        return mask

    # ------------------------------------------------------------------
    # Shape detection in ROI
    # ------------------------------------------------------------------
    def detect_shapes(self, roi_bgr: np.ndarray, out_img: np.ndarray, roi_y0: int) -> List[Dict]:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shapes: List[Dict] = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_mask_area_px:
                continue

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            x, y, w, h = cv2.boundingRect(approx)
            if w < self.min_rect_side_px or h < self.min_rect_side_px:
                continue

            # Circle / square / rectangle
            num_vertices = len(approx)
            shape_type = "unknown"
            if num_vertices >= 8:
                shape_type = "circle"
            elif 4 <= num_vertices <= 7:
                ar = float(w) / float(h)
                if 0.8 <= ar <= 1.25:
                    shape_type = "square"
                else:
                    shape_type = "rectangle"

            # 最小外接旋转矩形，用于 PnP
            rect = cv2.minAreaRect(c)
            (cx, cy), (rw, rh), angle = rect
            box = cv2.boxPoints(rect)
            box = np.intp(box)

            # 在 out_img 上画出来（加上 ROI 的偏移）
            box_global = box.copy()
            box_global[:, 1] += roi_y0
            cv2.drawContours(out_img, [box_global], 0, (0, 255, 255), 2)
            cv2.putText(
                out_img,
                shape_type,
                (int(cx), int(cy) + roi_y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

            shapes.append(
                {
                    "shape": shape_type,
                    "contour": c,
                    "area": area,
                    "rect": rect,
                    "box": box,  # 还没加 y 偏移的 ROI 内坐标
                    "roi_y0": roi_y0,
                }
            )

        return shapes

    # ------------------------------------------------------------------
    # 从所有 shapes 里挑出 target shape（只挑一个）
    # ------------------------------------------------------------------
    def pick_target_shape(self, shapes: List[Dict]) -> Optional[Dict]:
        tshape = self.normalize_target_shape()
        if tshape == "any":
            if not shapes:
                return None
            # 选择面积最大的
            return max(shapes, key=lambda s: s["area"])

        candidates = [s for s in shapes if s["shape"] == tshape]
        if not candidates:
            return None
        return max(candidates, key=lambda s: s["area"])

    def draw_cube_axes(self, frame, rvec, tvec):
        """
        用 3D -> 2D 重投影画出 cube 在 camera_optical_link 下的坐标轴。
        约定：
        X - 红色
        Y - 绿色
        Z - 蓝色
        """

        # 轴长度：用 cube 边长的 1.5 倍（单位：米）
        L = self.target_size_m * 1.5

        # === 3D 轴端点（相机坐标系） ===
        axis_3d = np.float32([
            [0.0, 0.0, 0.0],   # 原点
            [L,   0.0, 0.0],   # X 轴
            [0.0, L,   0.0],   # Y 轴
            [0.0, 0.0, L],     # Z 轴
        ]).reshape(-1, 1, 3)

        # === 相机内参矩阵 K（直接用 fx, fy, cx, cy 重新拼） ===
        K = np.array([
            [self.fx, 0.0,      self.cx],
            [0.0,     self.fy,  self.cy],
            [0.0,     0.0,      1.0    ],
        ], dtype=np.float32)

        # === 重投影到像素平面 ===
        pts_2d, _ = cv2.projectPoints(
            axis_3d,
            rvec,
            tvec,
            K,
            self.dist_coeffs
        )

        pts = pts_2d.reshape(-1, 2).astype(int)
        o = tuple(pts[0])
        x = tuple(pts[1])
        y = tuple(pts[2])
        z = tuple(pts[3])

        # BGR：X 红, Y 绿, Z 蓝（和 aruco 一样）
        cv2.line(frame, o, x, (0,   0, 255), 3)  # X - red
        cv2.line(frame, o, y, (0, 255,   0), 3)  # Y - green
        cv2.line(frame, o, z, (255, 0,   0), 3)  # Z - blue

    # ------------------------------------------------------------------
    # solvePnP + 6D pose publishing
    # ------------------------------------------------------------------
    def estimate_pose_and_publish(
        self,
        out: np.ndarray,
        target: Dict,
    ) -> None:
        if not self.have_caminfo:
            self.publish_once("NO_CAMERA_INFO", state="NO_CAMINFO")
            self._show(out, None)
            return

        box = target["box"]  # ROI 内坐标
        roi_y0 = target["roi_y0"]

        # 把 box 加上 ROI 偏移，变成全图坐标系
        box_global = box.astype(np.float64)
        box_global[:, 1] += float(roi_y0)

        # 排序：顺序为 (-x,-y), (+x,-y), (+x,+y), (-x,+y)
        # 简单做法：按 y，再按 x
        idx = np.lexsort((box_global[:, 0], box_global[:, 1]))
        sorted_pts = box_global[idx]

        # top-left, top-right, bottom-right, bottom-left
        img_pts = np.array(
            [
                sorted_pts[0],
                sorted_pts[1],
                sorted_pts[3],
                sorted_pts[2],
            ],
            dtype=np.float64,
        )

        half = self.target_size_m / 2.0
        obj_pts = np.array(
            [
                [-half, -half, 0.0],
                [half, -half, 0.0],
                [half, half, 0.0],
                [-half, half, 0.0],
            ],
            dtype=np.float64,
        )

        K = np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        dist = self.dist_coeffs if self.dist_coeffs is not None else np.zeros((5, 1), dtype=np.float64)

        try:
            ok, rvec, tvec = cv2.solvePnP(
            obj_pts,
            img_pts,
            K,
            dist,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )


        except Exception as e:
            self.publish_once(f"SOLVEPNP_EXCEPTION: {e}", state="PNP_FAIL")
            self.pos_filt = None
            self.yaw_filt_deg = None
            self._show(out, None)
            return

        if not ok:
            self.publish_once("SOLVEPNP_FAILED", state="PNP_FAIL")
            self.pos_filt = None
            self.yaw_filt_deg = None
            self._show(out, None)
            return

        # tvec: [x,y,z] in camera frame
        tvec = tvec.reshape(3)

        # 平移 EMA 滤波
        if self.pos_filt is None:
            self.pos_filt = tvec
        else:
            self.pos_filt = (1.0 - self.alpha) * self.pos_filt + self.alpha * tvec

        x, y, z = float(self.pos_filt[0]), float(self.pos_filt[1]), float(self.pos_filt[2])

        # 旋转矩阵 -> 欧拉角 (roll, pitch, yaw)，使用 intrinsic XYZ（和原来保持一致）
        R, _ = cv2.Rodrigues(rvec)
        M = np.eye(4, dtype=np.float64)
        M[:3, :3] = R

        roll_rad, pitch_rad, yaw_rad = tf_transformations.euler_from_matrix(M, "sxyz")
        roll_raw_deg = math.degrees(roll_rad)
        pitch_raw_deg = math.degrees(pitch_rad)
        yaw_raw_deg = math.degrees(yaw_rad)

        # --- 对 roll / pitch / yaw 全部做 EMA 滤波 ---
        if self.roll_filt_deg is None:
            self.roll_filt_deg = roll_raw_deg
        else:
            self.roll_filt_deg = (1.0 - self.alpha) * self.roll_filt_deg + self.alpha * roll_raw_deg

        if self.pitch_filt_deg is None:
            self.pitch_filt_deg = pitch_raw_deg
        else:
            self.pitch_filt_deg = (1.0 - self.alpha) * self.pitch_filt_deg + self.alpha * pitch_raw_deg

        if self.yaw_filt_deg is None:
            self.yaw_filt_deg = yaw_raw_deg
        else:
            self.yaw_filt_deg = (1.0 - self.alpha) * self.yaw_filt_deg + self.alpha * yaw_raw_deg

        roll_deg = float(self.roll_filt_deg)
        pitch_deg = float(self.pitch_filt_deg)
        yaw_deg = float(self.yaw_filt_deg)

        # quaternion（仍然用 intrinsic XYZ）
        q = tf_transformations.quaternion_from_euler(
            math.radians(roll_deg),
            math.radians(pitch_deg),
            math.radians(yaw_deg),
        )


        # ===== Publish 6D pose =====
        now = time.time()
        label_prefix = f"{self.target_color.upper()}_{self.target_shape.upper()}_6D"

        if now - self.last_pub >= self.publish_period:
            status = (
                f"{label_prefix} | "
                f"x={x:.3f} y={y:.3f} z={z:.3f} m | "
                f"roll={roll_deg:.1f} pitch={pitch_deg:.1f} yaw={yaw_deg:.1f} deg | "
                f"frame={self.camera_frame}"
            )
            self.get_logger().info(status)
            self.pub_status.publish(String(data=status))

            # 提示用的 grasp hint
            self.pub_grasp.publish(String(data="GRASP_HINT: use detected_object_pose"))

            # PoseStamped（相机光学坐标系）
            ps = PoseStamped()
            ps.header.stamp = self.get_clock().now().to_msg()
            ps.header.frame_id = self.camera_frame
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.position.z = z
            ps.pose.orientation.x = float(q[0])
            ps.pose.orientation.y = float(q[1])
            ps.pose.orientation.z = float(q[2])
            ps.pose.orientation.w = float(q[3])

            # 兼容老 topic
            self.pub_pose_hint.publish(ps)
            # 新 topic：控制真正订阅这个
            self.pub_pose.publish(ps)

            # 同时输出 object_shape / object_color
            self.pub_shape.publish(String(data=self.target_shape))
            self.pub_color.publish(String(data=self.target_color))

            self.last_pub = now
            self.last_state = "FOUND"

        # 画坐标轴
        self.draw_cube_axes(out, rvec, tvec)


        # overlay 一点信息
        cv2.putText(
            out,
            f"{label_prefix} ({self.camera_frame})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            f"x={x:.3f}  y={y:.3f}  z={z:.3f}  (m)",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            f"r={roll_deg:.1f}  p={pitch_deg:.1f}  y={yaw_deg:.1f}  (deg)",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        self._show(out, None)

    # ------------------------------------------------------------------
    # draw axes
    # ------------------------------------------------------------------
    # def _project_and_draw_axes(self, img: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> None:
    #     axis_len = self.axis_len_m
    #     axis = np.float32(
    #         [
    #             [0, 0, 0],
    #             [axis_len, 0, 0],
    #             [0, axis_len, 0],
    #             [0, 0, axis_len],
    #         ]
    #     )

    #     K = np.array(
    #         [
    #             [self.fx, 0.0, self.cx],
    #             [0.0, self.fy, self.cy],
    #             [0.0, 0.0, 1.0],
    #         ],
    #         dtype=np.float64,
    #     )
    #     dist = self.dist_coeffs if self.dist_coeffs is not None else np.zeros((5, 1), dtype=np.float64)

    #     try:
    #         imgpts, _ = cv2.projectPoints(axis, rvec, tvec, K, dist)
    #     except Exception:
    #         return

    #     imgpts = imgpts.reshape(-1, 2).astype(int)
    #     origin = tuple(imgpts[0])

    #     # X: red, Y: green, Z: blue
    #     cv2.line(img, origin, tuple(imgpts[1]), (0, 0, 255), 2)
    #     cv2.line(img, origin, tuple(imgpts[2]), (0, 255, 0), 2)
    #     cv2.line(img, origin, tuple(imgpts[3]), (255, 0, 0), 2)

    # ------------------------------------------------------------------
    # Status helper
    # ------------------------------------------------------------------
    def publish_once(self, msg: str, state: str) -> None:
        if state != self.last_state:
            self.get_logger().info(msg)
            self.pub_status.publish(String(data=f"{state}: {msg}"))
            self.last_state = state

    # ------------------------------------------------------------------
    # Show windows
    # ------------------------------------------------------------------
    def _show(self, out: np.ndarray, hsv_mask: Optional[np.ndarray]) -> None:
        now = time.time()
        if now - self.last_show >= self.show_period:
            cv2.imshow("Object Detection (Target)", out)
            self.last_show = now

        if hsv_mask is not None and now - self.last_mask_show >= self.mask_show_period:
            cv2.imshow("HSV MASK (Target Color)", hsv_mask)
            self.last_mask_show = now

        cv2.waitKey(1)

    # ------------------------------------------------------------------
    # Image callback
    # ------------------------------------------------------------------
    def cb_img(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.publish_once(f"CV_BRIDGE_ERROR: {e}", state="BRIDGE_FAIL")
            return

        if frame is None:
            return

        H, W, _ = frame.shape
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_color = self.get_color_mask(hsv)

        # ROI（一般下面是桌子）
        roi_y0 = 0
        roi = frame
        mask_roi = mask_color
        if self.use_roi:
            roi_y0 = int(self.roi_y_start_ratio * H)
            roi = frame[roi_y0:, :]
            mask_roi = mask_color[roi_y0:, :]

        # 只在颜色 mask 区域里做 shape 检测
        roi_color = cv2.bitwise_and(roi, roi, mask=mask_roi)

        out = frame.copy()

        shapes = self.detect_shapes(roi_color, out, roi_y0)
        target = self.pick_target_shape(shapes)

        if target is None:
            self.publish_once("NO_TARGET_SHAPE_FOUND", state="NO_TARGET")
            self._show(out, mask_color)
            self.pos_filt = None
            self.yaw_filt_deg = None
            return

        # 只针对这个 target 做 PnP + 发布 6D pose
        self.estimate_pose_and_publish(out, target)

    # ------------------------------------------------------------------
    # Spin
    # ------------------------------------------------------------------


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
