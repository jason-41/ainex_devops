#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
from vision_interfaces.msg import MarkerInfo, MarkerInfoArray
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster
import multiprocessing as mp

# -------------------------
# Helper function for guidance
# -------------------------
def marker_guidance_text(m, close_cm: float = 10.0) -> str:
    dist_cm = float(m.distance) * 100.0
    h = float(m.horizontal_angle)
    if abs(h) < math.radians(5):
        direction = "Front"
    elif h > 0:
        direction = f"Right Front ({math.degrees(h):.1f} deg)"
    else:
        direction = f"Left Front ({math.degrees(abs(h)):.1f} deg)"

    lines = [f"Found Marker {m.marker_id}",
             f"Dist: {m.distance:.3f}m ({dist_cm:.1f}cm) Dir: {direction}"]
    
    if dist_cm <= close_cm:
        lines.append(f"Warning: Dist < {close_cm:.0f}cm: Slow down/Stop")
    else:
        if abs(h) > math.radians(10):
            turn = "Left" if h > 0 else "Right"
            lines.append(f"Suggestion: Turn {turn}")
        else:
            lines.append("Suggestion: Go Straight")
    return " | ".join(lines)

# -------------------------
# Worker process for detection
# -------------------------
def _detector_worker(in_q: mp.Queue, out_q: mp.Queue, dict_name: str):
    cv2.setNumThreads(1)
    dict_id = getattr(cv2.aruco, dict_name, cv2.aruco.DICT_4X4_50)
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    params = cv2.aruco.DetectorParameters_create()

    while True:
        item = in_q.get()
        if item is None:
            break
        t_sec, frame = item
        if frame is None:
            out_q.put((t_sec, [], None, None))
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)

        ids_list, corners_list = [], []
        if ids is not None:
            ids_list = ids.flatten().tolist()
            corners_list = [c.tolist() for c in corners]
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        out_q.put((t_sec, ids_list, corners_list, frame))

# -------------------------
# ArUco Detector Node
# -------------------------
class ArucoDetectorNode(Node):
    def __init__(self):
        super().__init__("aruco_detector")
        self.bridge = CvBridge()

        # Topic to subscribe: the undistorted image
        self.image_topic = "/camera/image_undistorted"
        self.dict_name = "DICT_4X4_50"
        self.camera_frame = "camera_optical_link"
        self.focal_length_px = 600.0
        self.marker_size_cm = 10.0
        self.target_id = 1

        # Publishers
        self.pub_markers = self.create_publisher(MarkerInfoArray, "/aruco_markers", 10)
        self.pub_vis = self.create_publisher(Image, "/aruco_vis", 10)
        self.pub_search_status = self.create_publisher(String, "/marker_search/status", 10)
        self.pub_search_pose = self.create_publisher(PoseStamped, "/marker_search/target_pose", 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Multiprocessing queues
        self.in_q = mp.Queue()
        self.out_q = mp.Queue()
        self.p = mp.Process(target=_detector_worker, args=(self.in_q, self.out_q, self.dict_name))
        self.p.daemon = True
        self.p.start()

        # Subscribe undistorted image
        self.create_subscription(Image, self.image_topic, self.feed_worker, 10)

        # Timer to check results
        self.timer = self.create_timer(0.01, self.check_results)

    # Feed worker queue
    def feed_worker(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        if self.in_q.qsize() < 2:
            t_sec = self.get_clock().now().nanoseconds / 1e9
            self.in_q.put((t_sec, frame))

    # Process detection results
    def check_results(self):
        while not self.out_q.empty():
            t_sec, ids, corners_list, frame_img = self.out_q.get()
            if frame_img is not None:
                self.pub_vis.publish(self.bridge.cv2_to_imgmsg(frame_img, "bgr8"))

            if not ids:
                continue

            msg_array = MarkerInfoArray()
            msg_array.header.stamp = self.get_clock().now().to_msg()
            msg_array.header.frame_id = self.camera_frame
            found_target = None

            for i, mid in enumerate(ids):
                c = np.array(corners_list[i][0])
                avg_side_px = np.mean([np.linalg.norm(c[j]-c[(j+1)%4]) for j in range(4)])
                z_m = (self.focal_length_px * self.marker_size_cm / avg_side_px)/100 if avg_side_px>0 else 0.5
                center = np.mean(c, axis=0)
                x_m = (center[0]-640/2)*z_m/self.focal_length_px
                y_m = (center[1]-480/2)*z_m/self.focal_length_px
                horizontal_angle = -math.atan2(x_m, z_m)

                m_info = MarkerInfo()
                m_info.marker_id = mid
                m_info.horizontal_angle = horizontal_angle
                m_info.distance = z_m
                m_info.pose.position.x = x_m
                m_info.pose.position.y = y_m
                m_info.pose.position.z = z_m
                m_info.pose.orientation.w = 1.0

                msg_array.markers.append(m_info)
                if mid == self.target_id:
                    found_target = m_info

            self.pub_markers.publish(msg_array)

            if found_target:
                guidance = marker_guidance_text(found_target)
                self.pub_search_status.publish(String(data=guidance))
                ps = PoseStamped()
                ps.header.frame_id = self.camera_frame
                ps.header.stamp = self.get_clock().now().to_msg()
                ps.pose = found_target.pose
                self.pub_search_pose.publish(ps)

                t = TransformStamped()
                t.header = ps.header
                t.child_frame_id = f"marker_{self.target_id}"
                t.transform.translation.x = ps.pose.position.x
                t.transform.translation.y = ps.pose.position.y
                t.transform.translation.z = ps.pose.position.z
                t.transform.rotation = ps.pose.orientation
                self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
