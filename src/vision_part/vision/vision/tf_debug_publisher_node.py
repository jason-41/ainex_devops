# -*- coding: utf-8 -*-
# src/vision/vision/tf_debug_publisher_node.py

# Description: tf_debug_publisher_node allows publishing and debugging coordinate systems in the Vision module.

import threading
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import tf_transformations as tf_trans
from dataclasses import dataclass
from typing import Optional, Tuple


def euler_to_quaternion(roll: float, pitch: float, yaw: float):
    """
    Small helper to convert Euler angles (roll, pitch, yaw in radians)
    into a quaternion (x, y, z, w) using tf_transformations.

    This keeps the TF math in one place so it can be reused by other nodes.
    """
    qx, qy, qz, qw = tf_trans.quaternion_from_euler(roll, pitch, yaw)
    return qx, qy, qz, qw


# ==============================================================================
# Core Logic
# ==============================================================================

@dataclass
class TFCommand:
    op: str
    name: str = ""
    parent: str = ""
    xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)


def parse_tf_cmd(line: str) -> Optional[TFCommand]:
    """
    Supports:
      add <name> <parent> <x> <y> <z> <roll> <pitch> <yaw>
      remove <name>
      list
      exit
    """
    if not line:
        return None
    parts = line.strip().split()
    if not parts:
        return None

    op = parts[0].lower()
    if op == "add" and len(parts) >= 9:
        name = parts[1]
        parent = parts[2]
        x, y, z = map(float, parts[3:6])
        r, p, yaw = map(float, parts[6:9])
        return TFCommand(op="add", name=name, parent=parent, xyz=(x, y, z), rpy=(r, p, yaw))

    if op == "remove" and len(parts) >= 2:
        return TFCommand(op="remove", name=parts[1])

    if op == "list":
        return TFCommand(op="list")

    if op == "exit":
        return TFCommand(op="exit")

    return None


# ==============================================================================
# ROS Node
# ==============================================================================

class TFDebugPublisherNode(Node):
    def __init__(self):
        super().__init__("tf_debug_publisher")
        self.broadcaster = TransformBroadcaster(self)
        self.transforms = {}  # name -> TransformStamped

        self.get_logger().info("TF Debug Publisher Started")
        self.get_logger().info(
            "Commands: add <name> <parent> <x> <y> <z> <roll> <pitch> <yaw> | list | remove <name> | exit"
        )

        self.timer = self.create_timer(0.1, self.publish_all)
        self._stop = False
        threading.Thread(target=self.cli_loop, daemon=True).start()

    def cli_loop(self):
        while rclpy.ok() and not self._stop:
            try:
                line = input("tf> ")
                cmd = parse_tf_cmd(line)
                if cmd is None:
                    print("Unknown command")
                    continue
                if cmd.op == "exit":
                    self._stop = True
                    break
                if cmd.op == "list":
                    self.cmd_list()
                elif cmd.op == "remove":
                    self.cmd_remove(cmd.name)
                elif cmd.op == "add":
                    self.cmd_add(cmd.name, cmd.parent, cmd.xyz, cmd.rpy)
            except Exception as e:
                print(f"Error: {e}")

    def cmd_add(self, name, parent, xyz, rpy):
        t = TransformStamped()
        t.header.frame_id = parent
        t.child_frame_id = name

        # Translation
        t.transform.translation.x = float(xyz[0])
        t.transform.translation.y = float(xyz[1])
        t.transform.translation.z = float(xyz[2])

        # Rotation: use the helper function so TF math is centralized
        qx, qy, qz, qw = euler_to_quaternion(
            float(rpy[0]),
            float(rpy[1]),
            float(rpy[2]),
        )
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        self.transforms[name] = t
        print(f"Added TF: {name} -> {parent}")

    def cmd_list(self):
        if not self.transforms:
            print("No active TF")
            return
        for name, t in self.transforms.items():
            tr = t.transform.translation
            print(f"{name} -> {t.header.frame_id}: [{tr.x:.2f}, {tr.y:.2f}, {tr.z:.2f}]")

    def cmd_remove(self, name):
        if name in self.transforms:
            del self.transforms[name]
            print(f"Removed TF: {name}")
        else:
            print(f"TF not found: {name}")

    def publish_all(self):
        now = self.get_clock().now().to_msg()
        for t in self.transforms.values():
            t.header.stamp = now
            self.broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = TFDebugPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
