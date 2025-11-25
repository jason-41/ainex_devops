#!/usr/bin/env python3
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

from ainex_motion.joint_controller import JointController


class FaceTriggeredWaveNode(Node):
    """
    Node that waits for the perception module to detect a face
    and then performs ONE right-arm waving gesture each time
    face_detected goes from False -> True.
    """

    def _init_(self):
        super()._init_('face_triggered_wave_node')

        # Motion SDK
        self.jc = JointController(self)

        # Face detection state
        self.face_detected = False
        self.prev_face_detected = False

        # Are we currently executing a wave?
        self.waving = False

        # Subscribe to the output of the face detector team
        # They should publish std_msgs/Bool on 'face_detected'
        self.create_subscription(
            Bool,
            'face_detected',
            self.face_callback,
            10
        )

        self.get_logger().info(
            'FaceTriggeredWaveNode initialized. '
            'Waiting for /face_detected (std_msgs/Bool)...'
        )

        # Go to an initial posture
        self.jc.setPosture('stand', duration=1.5)
        time.sleep(1.7)

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------
    def face_callback(self, msg: Bool):
        """
        Called whenever perception publishes a new Bool.
        We trigger a wave on the rising edge: False -> True.
        """
        self.face_detected = msg.data

        # Detect rising edge: previously False, now True
        if (not self.prev_face_detected) and self.face_detected:
            self.get_logger().info('Face detected (rising edge).')

            if not self.waving:
                self.get_logger().info('Starting wave gesture.')
                self.waving = True
                threading.Thread(
                    target=self.wave_routine,
                    daemon=True
                ).start()

        # Update previous state
        self.prev_face_detected = self.face_detected

    # -------------------------------------------------------------------------
    # Right-arm wave routine
    # -------------------------------------------------------------------------
    def wave_routine(self):
        """
        Execute one complete wave gesture with the right arm.
        Runs in a background thread to avoid blocking callbacks.
        """
        try:
            # Right-arm joint names from JointController.joint_id
            r_sho_pitch = 'r_sho_pitch'
            r_sho_roll  = 'r_sho_roll'
            r_el_pitch  = 'r_el_pitch'
            r_el_yaw    = 'r_el_yaw'

            # 1) Ensure home posture
            self.get_logger().info('Wave: going to home posture.')
            self.jc.setPosture('stand', duration=1.5)
            time.sleep(1.7)

            # 2) Raise right arm into waving pose
            self.get_logger().info('Wave: raising right arm.')
            self.jc.setJointPositions(
                [r_sho_pitch, r_sho_roll, r_el_pitch, r_el_yaw],
                positions=[-0.3, -1.0, -1.0, 0.0],
                duration=1.0,
                unit='rad',
            )
            time.sleep(1.2)

            # 3) Wave: oscillate elbow yaw a few times
            self.get_logger().info('Wave: waving with right hand...')
            n_cycles = 4
            delta_yaw = 0.4  # rad

            for _ in range(n_cycles):
                # yaw outward
                self.jc.changeJointPositions(
                    [r_el_yaw],
                    [delta_yaw],
                    duration=0.4,
                    unit='rad',
                )
                time.sleep(0.5)

                # yaw back
                self.jc.changeJointPositions(
                    [r_el_yaw],
                    [-delta_yaw],
                    duration=0.4,
                    unit='rad',
                )
                time.sleep(0.5)

            # 4) Return to home posture
            self.get_logger().info('Wave: returning to home posture.')
            self.jc.setPosture('stand', duration=1.5)
            time.sleep(1.7)

            self.get_logger().info('Wave gesture finished.')
        finally:
            self.waving = False


def main(args=None):
    rclpy.init(args=args)
    node = FaceTriggeredWaveNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()