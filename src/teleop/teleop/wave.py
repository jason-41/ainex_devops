#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node

from ainex_motion.joint_controller import JointController


class Wave(Node):
    def __init__(self):
        super().__init__("ainex_wave_node")

        # Joint controller
        self.jc = JointController(self)

        # ----------------------------
        # Parameters
        # ----------------------------
        self.declare_parameter("arm", "right")          # "right" or "left"
        self.declare_parameter("repeat", 3)            # number of left-right swings per wave cycle
        self.declare_parameter("swing_duration", 0.5)  # seconds per swing command
        self.declare_parameter("auto_wave", True)      # if True -> wave periodically
        self.declare_parameter("period", 6.0)          # seconds between wave cycles

        self.arm = str(self.get_parameter("arm").value).strip().lower()
        self.repeat = int(self.get_parameter("repeat").value)
        self.swing_duration = float(self.get_parameter("swing_duration").value)
        self.auto_wave = bool(self.get_parameter("auto_wave").value)
        self.period = float(self.get_parameter("period").value)

        # Prevent overlapping waves
        self._busy = False

        # Optional periodic timer
        if self.auto_wave:
            self.timer = self.create_timer(self.period, self._timer_cb)
            self.get_logger().info(
                f"WaveNode started. Auto-wave ON every {self.period:.1f}s (arm={self.arm})."
            )
        else:
            self.get_logger().info(
                f"WaveNode started. Auto-wave OFF (arm={self.arm}). Call wave_arm() manually."
            )

    def _timer_cb(self):
        if self._busy:
            return
        self._busy = True
        try:
            self.wave_arm()
        finally:
            self._busy = False

    def wave_arm(self):
        """
        Simple wave using JointController.
        Uses the same right-arm joint set you had.
        """
        if self.arm not in ("right", "left"):
            self.get_logger().warn(f"Invalid arm='{self.arm}', defaulting to 'right'")
            self.arm = "right"

        if self.arm == "right":
            joints = ["r_sho_roll", "r_sho_pitch", "r_el_pitch", "r_el_yaw"]

            standby_pos_down = [-0.017, 0.18, -0.084, 1.257]  # hand down
            standby_pos_up   = [1.515, 1.595, 0.0,   1.55]    # hand up
            far_right_pos    = [1.515, 1.595, 0.0,   0.95]
            far_left_pos     = [1.515, 1.595, 0.0,   2.0]
        else:
            # Mirror for left arm (you may need to adjust yaw directions depending on your robot's conventions)
            joints = ["l_sho_roll", "l_sho_pitch", "l_el_pitch", "l_el_yaw"]

            standby_pos_down = [0.017, 0.18, -0.084, -1.257]
            standby_pos_up   = [-1.515, 1.595, 0.0,   -1.55]
            far_right_pos    = [-1.515, 1.595, 0.0,   -2.0]
            far_left_pos     = [-1.515, 1.595, 0.0,   -0.95]

        # Move to wave-up
        self.jc.setJointPositions(joints, standby_pos_up, duration=self.swing_duration)
        time.sleep(self.swing_duration)

        # Swing left-right repeat times
        for _ in range(max(1, self.repeat)):
            self.jc.setJointPositions(joints, far_right_pos, duration=self.swing_duration)
            time.sleep(self.swing_duration)
            self.jc.setJointPositions(joints, far_left_pos, duration=self.swing_duration)
            time.sleep(self.swing_duration)

        # Return down
        self.jc.setJointPositions(joints, standby_pos_down, duration=self.swing_duration)
        time.sleep(self.swing_duration)

        self.get_logger().info("Wave finished.")


def main(args=None):
    rclpy.init(args=args)
    node = WaveNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
