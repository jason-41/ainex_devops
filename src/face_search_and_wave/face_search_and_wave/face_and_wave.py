import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool  # for face_detected topic
from ainex_motion.joint_controller import JointController  # import JointController

import time


class FaceSearchAndWaveNode(Node):
    def __init__(self):
        super().__init__("face_search_and_wave")

        # Joint controller
        self.jc = JointController(self)

        # Subscribe to face_detected message
        self.sub = self.create_subscription(
            Bool,
            "face_detected",     # topic name
            self.face_callback,
            10
        )

        self.get_logger().info("FaceSearchAndWaveNode started.")

    # --------------------------------------------------------
    def face_callback(self, msg: Bool):
        if msg.data == 0:
            # no face detected → do nothing (simple version)
            return

        # Face detected = 1
        self.get_logger().info("Face detected! Stopping head & waving arm.")

        # 1. Stop head movement (most basic version)
        self.stop_head()

        # 2. Do waving motion
        # self.wave_arm()

    # --------------------------------------------------------
    def stop_head(self):
        # 头部两个关节
        head_joints = ["head_pan", "head_tilt"]

        # 一版最简单：全部设为0（home）
        self.jc.setJointPositions(head_joints, [0.0, 0.0], duration=0.5)
        time.sleep(1.0)

    # --------------------------------------------------------
    def wave_arm(self):
        """
        最简单的挥手：让肩关节上下摆几次
        你说了现在不关注动作是否正确
        """

        # 用左手：l_sho_pitch, l_sho_roll
        joints = ["l_sho_pitch", "l_sho_roll"]

        # 简单两三次（角度只是示意）
        for _ in range(3):
            self.jc.setJointPositions(
                joints, [0.5, 0.0], duration=0.5)  # raise
            time.sleep(0.6)
            self.jc.setJointPositions(
                joints, [-0.5, 0.0], duration=0.5)  # lower
            time.sleep(0.6)

        # 回 home
        self.jc.setJointPositions(joints, [0.0, 0.0], duration=0.5)
        time.sleep(0.5)


# --------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)

    node = FaceSearchAndWaveNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
