#!/usr/bin/env python3
"""
Degrasp node (ROS2 Humble) for AiNex

Behavior:
- DOES NOT move to any initial posture (does not reset robot state).
- Waits for /active_hand ("left"/"right") from grasp node.
- Optionally waits for /grasp_done (Bool True).
- Moves the active hand forward +8 cm in base_link (relative motion).
- Opens ONLY the active gripper.
"""

import time
import numpy as np
import pinocchio as pin
import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Bool
from ament_index_python.packages import get_package_share_directory

from ainex_controller.ainex_model import AiNexModel
from ainex_controller.ainex_robot import AinexRobot
from ainex_controller.ainex_hand_controller import HandController


class AinexDegraspNode(Node):
    def __init__(self):
        super().__init__("ainex_degrasp_node")

        # ----------------------------
        # PARAMETERS
        # ----------------------------
        self.declare_parameter("sim", False)
        self.declare_parameter("dt_cmd", 0.05)

        # Topics from grasp node
        self.declare_parameter("active_hand_topic", "/active_hand")
        self.declare_parameter("grasp_done_topic", "/grasp_done")

        # Behavior
        self.declare_parameter("wait_for_grasp_done", True)  # wait until grasp finished
        self.declare_parameter("forward_x", 0.1)            # move forward 8 cm
        self.declare_parameter("move_duration", 2.0)         # seconds for the forward motion
        self.declare_parameter("settle_s", 0.15)             # small pause before opening

        # Gripper control
        self.declare_parameter("gripper_kp", 6.0)
        self.declare_parameter("gripper_vel_max", 2.0)
        self.declare_parameter("gripper_eps", 0.02)
        self.declare_parameter("open_hold_time", 0.5)

        # ----------------------------
        # STATE
        # ----------------------------
        self.active_hand = None            # "left" or "right"
        self.grasp_done = False

        self.active_hand_topic = str(self.get_parameter("active_hand_topic").value)
        self.grasp_done_topic = str(self.get_parameter("grasp_done_topic").value)

        self.create_subscription(String, self.active_hand_topic, self._active_hand_cb, 10)
        self.create_subscription(Bool, self.grasp_done_topic, self._grasp_done_cb, 10)

        self.get_logger().info(f"Subscribed to {self.active_hand_topic} and {self.grasp_done_topic}")

        # ----------------------------
        # LOAD MODEL + ROBOT (NO reset posture)
        # ----------------------------
        pkg = get_package_share_directory("ainex_description")
        urdf_path = pkg + "/urdf/ainex.urdf"
        self.robot_model = AiNexModel(self, urdf_path)

        self.sim = bool(self.get_parameter("sim").value)
        self.dt_cmd = float(self.get_parameter("dt_cmd").value)

        self.robot = AinexRobot(self, self.robot_model, self.dt_cmd, sim=self.sim)

        # NOTE: We intentionally do NOT call move_to_initial_position().
        # We assume the robot is already in the post-grasp state.

        # Precompute gripper joint indices
        self.gripper_indices = {}
        for jname in ["l_gripper", "r_gripper"]:
            jid = self.robot_model.model.getJointId(jname)
            q_idx = self.robot_model.model.joints[jid].idx_q
            v_idx = self.robot_model.model.joints[jid].idx_v
            self.gripper_indices[jname] = (q_idx, v_idx)

        # Gripper controller params
        self.gripper_kp = float(self.get_parameter("gripper_kp").value)
        self.gripper_vel_max = float(self.get_parameter("gripper_vel_max").value)
        self.gripper_eps = float(self.get_parameter("gripper_eps").value)

        # Run one-shot degrasp (Manual call required if integrated)
        self.run()

    def _active_hand_cb(self, msg: String):
        val = (msg.data or "").strip().lower()
        if val in ("left", "right"):
            self.active_hand = val

    def _grasp_done_cb(self, msg: Bool):
        self.grasp_done = bool(msg.data)

    def _sleep_to_rate(self, t_cycle_start: float):
        rem = self.dt_cmd - (time.monotonic() - t_cycle_start)
        if rem > 0.0:
            time.sleep(rem)

    def _compute_open_q(self, arm_side: str) -> float:
        """Compute gripper open position from joint limits (consistent with your grasp script)."""
        gripper_joint = "l_gripper" if arm_side == "left" else "r_gripper"
        q_idx, _ = self.gripper_indices[gripper_joint]

        q_lo = float(self.robot_model.model.lowerPositionLimit[q_idx])
        q_hi = float(self.robot_model.model.upperPositionLimit[q_idx])
        margin = 0.15 * (q_hi - q_lo)

        if arm_side == "left":
            # left: low=open, high=closed
            return q_lo + margin
        else:
            # right: high=open, low=closed
            return q_hi - margin

    def _drive_active_gripper(self, arm_side: str, q_target: float):
        """Velocity P-controller on ONLY the selected gripper joint."""
        gripper_joint = "l_gripper" if arm_side == "left" else "r_gripper"
        qg_idx, vg_idx = self.gripper_indices[gripper_joint]

        qg = float(self.robot.q[qg_idx])
        vg = self.gripper_kp * (q_target - qg)
        vg = float(np.clip(vg, -self.gripper_vel_max, self.gripper_vel_max))
        if abs(q_target - qg) < self.gripper_eps:
            vg = 0.0

        self.robot.v[vg_idx] = vg

    def run(self):
        wait_for_grasp_done = bool(self.get_parameter("wait_for_grasp_done").value)
        forward_x = float(self.get_parameter("forward_x").value)
        move_duration = float(self.get_parameter("move_duration").value)
        settle_s = float(self.get_parameter("settle_s").value)
        open_hold_time = float(self.get_parameter("open_hold_time").value)

        # ----------------------------
        # Wait for active_hand (+ optional grasp_done)
        # ----------------------------
        self.get_logger().info("Waiting for /active_hand ...")
        t0 = time.time()
        while rclpy.ok() and self.active_hand is None:
            rclpy.spin_once(self, timeout_sec=0.05)
            if time.time() - t0 > 10.0:
                self.get_logger().error("Timeout waiting for /active_hand.")
                return

        self.get_logger().info(f"Active hand = {self.active_hand}")

        if wait_for_grasp_done:
            self.get_logger().info("Waiting for /grasp_done == True ...")
            t1 = time.time()
            while rclpy.ok() and not self.grasp_done:
                rclpy.spin_once(self, timeout_sec=0.05)
                if time.time() - t1 > 20.0:
                    self.get_logger().warn("Timeout waiting for /grasp_done; proceeding anyway.")
                    break

        # ----------------------------
        # Create controller for selected arm (no posture reset)
        # ----------------------------
        arm_side = self.active_hand
        hand_ctrl = HandController(self, self.robot_model, arm_side=arm_side)

        # Conservative motion for safety (tune if needed)
        hand_ctrl.linear_vel_limit = 0.06
        hand_ctrl.joint_vel_limit = np.array([2.5, 2.5, 2.5, 2.5])

        # If your HandController supports these (from your newer version), set them safely:
        if hasattr(hand_ctrl, "dls_lambda"):
            hand_ctrl.dls_lambda = 0.03
        if hasattr(hand_ctrl, "u_alpha"):
            hand_ctrl.u_alpha = 0.25

        # Relative motion: +X forward in base_link
        T_rel = pin.SE3(np.eye(3), np.array([forward_x, 0.0, 0.0], dtype=float))
        hand_ctrl.set_target_pose(T_rel, duration=move_duration, type="rel")
        self.get_logger().info(f"Moving {arm_side} hand forward by {forward_x:.3f} m ...")

        # ----------------------------
        # Execute motion
        # ----------------------------
        while rclpy.ok():
            t_cycle_start = time.monotonic()
            rclpy.spin_once(self, timeout_sec=0.0)

            v_hand = hand_ctrl.update(self.dt_cmd)

            if arm_side == "right":
                self.robot.update(None, v_hand, self.dt_cmd)
            else:
                self.robot.update(v_hand, None, self.dt_cmd)

            if hand_ctrl.is_finished():
                break

            self._sleep_to_rate(t_cycle_start)

        time.sleep(settle_s)

        # ----------------------------
        # Open ONLY the active gripper
        # ----------------------------
        q_open = self._compute_open_q(arm_side)
        self.get_logger().info(f"Opening {arm_side} gripper to q={q_open:.3f} ...")

        t_open = time.time()
        while rclpy.ok():
            t_cycle_start = time.monotonic()
            rclpy.spin_once(self, timeout_sec=0.0)

            # hold arm still while opening
            if arm_side == "right":
                self.robot.update(None, np.zeros(4), self.dt_cmd)
            else:
                self.robot.update(np.zeros(4), None, self.dt_cmd)

            self._drive_active_gripper(arm_side, q_open)

            # stop after it reaches target (or after a small hold time)
            # Note: we check joint position error using robot.q
            gripper_joint = "l_gripper" if arm_side == "left" else "r_gripper"
            qg_idx, _ = self.gripper_indices[gripper_joint]
            if abs(float(self.robot.q[qg_idx]) - float(q_open)) < self.gripper_eps:
                if time.time() - t_open >= open_hold_time:
                    break

            if time.time() - t_open > 3.0:
                self.get_logger().warn("Opening timeout; stopping.")
                break

            self._sleep_to_rate(t_cycle_start)

        self.get_logger().info("Degrasp done.")


def main():
    rclpy.init()
    node = None
    try:
        node = AinexDegraspNode()
        node.run()
    except Exception as e:
        print(f"[ERROR] degrasp failed: {e}")
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
