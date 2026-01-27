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

        self.search_positions = [
            {"joints": ["head_pan", "head_tilt"], "pos": [-0.8, 0.1]},
            {"joints": ["head_pan", "head_tilt"], "pos": [-0.4, 0.1]},
            {"joints": ["head_pan", "head_tilt"], "pos": [0.0,  0.1]},
            {"joints": ["head_pan", "head_tilt"], "pos": [0.4,  0.1]},
            {"joints": ["head_pan", "head_tilt"], "pos": [0.8,  0.1]},
            {"joints": ["head_pan", "head_tilt"], "pos": [0.4,  0.1]},
            {"joints": ["head_pan", "head_tilt"], "pos": [0.0,  0.1]},
            {"joints": ["head_pan", "head_tilt"], "pos": [-0.4, 0.1]},
        ]
        self.face_already_detected = False   # 防止重复触发 prevent multiple triggers
        self.current_search_index = 0
        self.tilt_flag = 0
        self.search_direction = 1

        # Subscribe to face_detected message
        self.sub = self.create_subscription(
            Bool,
            "face_detected",     # topic name
            self.face_callback,
            10
        )

        # 创建定时器：周期性转头“找人”  --- English: create a timer to periodically turn head to "search for people"
        # 例如每 1.0 s 换一个角度 (e.g., change angle every 1.0 s)
        self.search_timer = self.create_timer(1.0, self.search_head_timer_cb)
        self.get_logger().info("FaceSearchAndWaveNode started. Searching for faces...")

    # 定时器：搜索头部关节  English: Timer: search head joints
    def search_head_timer_cb(self):
        """周期性改变头部关节角度，模拟机器人在左右张望找人。"""

        if self.face_already_detected:
            # 如果已经检测到人脸，就不用再搜索了→ 直接返回 If a face has already been detected, no need to search anymore → just return
            return

        # 取当前要设置的姿态 Get the current posture to set
        cfg = self.search_positions[self.current_search_index]
        joints = cfg["joints"]
        pos = cfg["pos"]
        
        if self.search_direction == 1 and self.tilt_flag == 1:
            pos[1] += 0.3 # slightly tilt up 0.3 each time
            if pos[1] >= 1.1:  # prevent over-tilting
                self.search_direction = -1
                pos[1] -= 0.3
        if self.search_direction == -1 and self.tilt_flag == 1:
            pos[1] -= 0.3 # slightly tilt down 0.3 each time
            if pos[1] <= -0.1:  # prevent over-tilting
                self.search_direction = 1
                pos[1] += 0.3

        # 发送关节指令 Send joint command
        self.jc.setJointPositions(joints, pos, duration=0.8)

        # 移动到下一个搜索姿态（循环）Move to the next search posture (loop). that's why use modulo
        if self.current_search_index + 1 >= len(self.search_positions):
            self.tilt_flag = 1

        self.get_logger().info(
            f"Searching... set head to {pos}, index = {self.current_search_index}"
        )

        self.current_search_index = (self.current_search_index + 1) % len(self.search_positions)

    # --------------------------------------------------------
    def face_callback(self, msg: Bool):
        if not msg.data:
            # no face detected → do nothing (simple version)
            return

        if self.face_already_detected:
            # already handled → do nothing
            return
        self.face_already_detected = True
        
        self.get_logger().info("Face detected! Stopping head & waving arm.")

        # Do waving motion
        self.wave_arm()
        self.face_already_detected = False

    def wave_arm(self):
        """
        最简单的挥手 the simplest wave
        """

        joints = ["r_sho_roll", "r_sho_pitch", "r_el_pitch", "r_el_yaw"]
        
        # Waving positions based on README.md
        standby_pos_down = [-0.017 , 0.18, -0.084, 1.257]  # Standby positions(hand down)
        standby_pos_up = [1.515, 1.595, 0., 1.55]    # Standby positions(hand up) 
        far_right_pos = [1.515, 1.595, 0., 0.95]     # Far Right positions
        far_left_pos = [1.515, 1.595, 0., 2.0]      # Far Left positions
        front_pos = [1.515, 1.595, -0.75, 1.55]      # Front positions(optional)
        back_pos = [1.515, 1.595, 0.75, 1.55]        # Back positions(optional)

        self.jc.setJointPositions(joints, standby_pos_up, duration=0.5)
        time.sleep(0.5)

        # 简单两三次 2-3 simple waves
        for _ in range(3):
            self.jc.setJointPositions(joints, far_right_pos, duration=0.5)
            time.sleep(0.5)
            self.jc.setJointPositions(joints, far_left_pos, duration=0.5)
            time.sleep(0.5)

        # 回 home
        self.jc.setJointPositions(joints, standby_pos_down, duration=0.5)
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
