#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
from datetime import datetime

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class DatasetCollectorNode(Node):
    """
    Subscribe /camera/image_undistorted (sensor_msgs/Image)
    Press keys to save frames into datasets/shape_detection/images/{train,val,test}

    Keys:
      s : save to split (default train)
      1 : set split=train
      2 : set split=val
      3 : set split=test
      q : quit
    """

    def __init__(self):
        super().__init__("dataset_collector_node")

        # ---------- params ----------
        self.declare_parameter("image_topic", "/camera/image_undistorted")
        self.declare_parameter("window_name", "DatasetCollector")
        self.declare_parameter("base_dir", "datasets/shape_detection/images")
        self.declare_parameter("split", "train")  # train/val/test
        self.declare_parameter("save_every_n_frames", 0)  # 0 means only manual save
        self.declare_parameter("min_save_interval_s", 0.15)  # avoid double click
        self.declare_parameter("jpeg_quality", 95)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.window_name = str(self.get_parameter("window_name").value)
        self.base_dir = str(self.get_parameter("base_dir").value)
        self.split = str(self.get_parameter("split").value).lower()

        self.save_every_n = int(self.get_parameter("save_every_n_frames").value)
        self.min_interval = float(self.get_parameter("min_save_interval_s").value)
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)

        # ---------- state ----------
        self.bridge = CvBridge()
        self.last_frame = None
        self.frame_count = 0
        self.last_save_time = 0.0
        self.saved_counts = {"train": 0, "val": 0, "test": 0}

        # ---------- ensure dirs ----------
        for sp in ["train", "val", "test"]:
            os.makedirs(os.path.join(self.base_dir, sp), exist_ok=True)

        # ---------- sub ----------
        self.create_subscription(Image, self.image_topic, self.cb_image, qos_profile_sensor_data)

        # ---------- timer for UI ----------
        self.timer = self.create_timer(0.03, self.on_timer)  # ~33 FPS UI refresh

        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        except Exception as e:
            self.get_logger().error(f"Cannot create OpenCV window (GUI missing?): {e}")

        self.get_logger().info(
            f"[DatasetCollector] started. topic={self.image_topic}, save_dir={self.base_dir}, split={self.split}\n"
            "Keys: [1]=train [2]=val [3]=test  [s]=save  [q]=quit"
        )

    def cb_image(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return
        self.last_frame = frame
        self.frame_count += 1

        # auto-save mode
        if self.save_every_n > 0 and (self.frame_count % self.save_every_n == 0):
            self.save_current_frame()

    def save_current_frame(self):
        if self.last_frame is None:
            return
        now = time.time()
        if (now - self.last_save_time) < self.min_interval:
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{self.split}_{ts}.jpg"
        out_path = os.path.join(self.base_dir, self.split, filename)

        ok = cv2.imwrite(
            out_path,
            self.last_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if ok:
            self.saved_counts[self.split] += 1
            self.last_save_time = now
            self.get_logger().info(f"[SAVE] {out_path}")
        else:
            self.get_logger().error(f"[SAVE FAILED] {out_path}")

    def on_timer(self):
        if self.last_frame is None:
            return

        vis = self.last_frame.copy()

        # overlay text
        h, w = vis.shape[:2]
        line1 = f"split={self.split} | saved(train/val/test)=({self.saved_counts['train']}/{self.saved_counts['val']}/{self.saved_counts['test']})"
        line2 = "Keys: 1=train 2=val 3=test | s=save | q=quit"
        cv2.putText(vis, line1, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(vis, line2, (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow(self.window_name, vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            self.get_logger().info("[DatasetCollector] Quit.")
            rclpy.shutdown()
            return

        if key == ord('1'):
            self.split = "train"
            self.get_logger().info("[DatasetCollector] split=train")
        elif key == ord('2'):
            self.split = "val"
            self.get_logger().info("[DatasetCollector] split=val")
        elif key == ord('3'):
            self.split = "test"
            self.get_logger().info("[DatasetCollector] split=test")
        elif key == ord('s'):
            self.save_current_frame()


def main(args=None):
    rclpy.init(args=args)
    node = DatasetCollectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
