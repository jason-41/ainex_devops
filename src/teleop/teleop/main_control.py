#!/usr/bin/env python3
"""
Main Control Node for Group E Final Project
Integrates LLM, Vision, Walking, Hands Control, Grasping, and Turning.
"""
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import time
import threading

# Import messages
from servo_service.msg import InstructionAfterLLM, VisionTarget, GraspFeedback

from ament_index_python.packages import get_package_share_directory
import numpy as np

# Import other control modules
from teleop.walk_to_Aruco import AinexWalkToAruco
from teleop.turn_around import TurnAroundNode
from teleop.hands_control import hands_control
from ainex_controller.ainex_model import AiNexModel
from ainex_controller.ainex_robot import AinexRobot
from ainex_controller.ainex_hand_controller import HandController


class MainControlNode(Node):
    def __init__(self):
        super().__init__("main_control_node")

        # --- Subscribers ---
        self.llm_sub = self.create_subscription(
            InstructionAfterLLM,
            'instruction_after_llm',
            self.llm_callback,
            10
        )

        self.grasp_sub = self.create_subscription(
            GraspFeedback,
            'grasp_ok',
            self.grasp_callback,
            10
        )

        # --- Publishers ---
        self.vision_pub = self.create_publisher(
            VisionTarget,
            'vision_target',
            10
        )

        # --- State Variables ---
        self.current_instruction = None
        self.grasp_success = False
        self.state = "IDLE"

        self.get_logger().info("MainControlNode Initialized. Waiting for LLM instructions...")

    def llm_callback(self, msg):
        self.get_logger().info(
            f"Received LLM Instruction: Pickup {msg.object_color} {msg.object_shape} at Loc {msg.pickup_location}")
        self.current_instruction = msg
        # Trigger the workflow if idle
        if self.state == "IDLE":
            self.start_workflow()

    def grasp_callback(self, msg):
        if msg.grasp_ok:
            self.get_logger().info("Received Grasp Success Feedback!")
            self.grasp_success = True

    def publish_vision_target(self, aruco_id):
        if not self.current_instruction:
            return

        msg = VisionTarget()
        msg.object_color = self.current_instruction.object_color
        msg.object_shape = self.current_instruction.object_shape
        msg.current_aruco_id = int(aruco_id)

        self.vision_pub.publish(msg)
        self.get_logger().info(f"Published Vision Target: ID {aruco_id}")

    def start_workflow(self):
        # Run in a separate thread to avoid blocking callbacks
        threading.Thread(target=self._workflow_logic, daemon=True).start()

    def _workflow_logic(self):
        try:
            self.state = "WORKING"
            pickup_loc = self.current_instruction.pickup_location
            destination_loc = self.current_instruction.destination_location

            # ---------------------------------------------------
            # 1. Walk to Pickup Location (e.g. ArUco 1)
            # ---------------------------------------------------
            self.get_logger().info(
                f"--- Step 1: Walking to Pickup Location {pickup_loc} ---")

            # Publish target for Vision (so it detects the correct ArUco)
            for _ in range(5):
                self.publish_vision_target(pickup_loc)
                time.sleep(0.1)

            # Execute Walk
            self.run_walk_to_aruco()

            # ---------------------------------------------------
            # 2. Hands Control (Approach)
            # ---------------------------------------------------
            self.get_logger().info("--- Step 2: Hands Control (Approach) ---")
            # Run for 10s or until stable
            self.run_hands_control(duration=10.0)

            # ---------------------------------------------------
            # 3. Grasping
            # ---------------------------------------------------
            self.get_logger().info("--- Step 3: Triggering Grasp ---")
            self.grasp_success = False
            timeout = 30  # seconds
            start_wait = time.time()

            # Here we assume the external Grasp Node is active and will send 'True' eventually
            while not self.grasp_success:
                if time.time() - start_wait > timeout:
                    self.get_logger().warn("Grasp timed out!")
                    break
                time.sleep(0.5)

            self.get_logger().info("Grasp Check passed (or timed out).")

            # ---------------------------------------------------
            # 4. Turn Around (180 deg)
            # ---------------------------------------------------
            self.get_logger().info("--- Step 4: Turning Around ---")
            self.run_turn_around()

            # ---------------------------------------------------
            # 5. Walk to Destination (e.g. ArUco 0)
            # ---------------------------------------------------
            self.get_logger().info(
                f"--- Step 5: Returning to Destination {destination_loc} ---")

            # Update Vision Target
            for _ in range(5):
                self.publish_vision_target(destination_loc)
                time.sleep(0.1)

            self.run_walk_to_aruco()

            # ---------------------------------------------------
            # 6. De-Grasp
            # ---------------------------------------------------
            self.get_logger().info("--- Step 6: De-Grasp ---")
            time.sleep(2.0)  # Simulate drop / call degrasp node

            # ---------------------------------------------------
            # 7. Re-orienting (Face forward for next round)
            # ---------------------------------------------------
            self.get_logger().info("--- Step 7: Re-orienting ---")
            self.run_turn_around()

            self.state = "IDLE"
            self.current_instruction = None
            self.get_logger().info("Workflow Complete. Waiting for next instruction.")

        except Exception as e:
            self.get_logger().error(f"Workflow failed: {e}")
            self.state = "IDLE"

    def run_walk_to_aruco(self):
        # Instantiate the node
        walk_node = AinexWalkToAruco()
        # Spin it until finished
        try:
            while rclpy.ok() and not walk_node.finished:
                rclpy.spin_once(walk_node, timeout_sec=0.05)
        finally:
            walk_node.destroy_node()

    def run_turn_around(self):
        turn_node = TurnAroundNode()
        try:
            turn_node.execute_turn()
        finally:
            turn_node.destroy_node()

    def run_hands_control(self, duration=10.0):
        temp_node = rclpy.create_node("hands_control_temp")
        try:
            pkg = get_package_share_directory("ainex_description")
            urdf_path = pkg + "/urdf/ainex.urdf"
            dt = 0.05

            robot_model = AiNexModel(temp_node, urdf_path)
            # Note: sim=True is safer for testing. Change to False for real robot.
            ainex_robot = AinexRobot(temp_node, robot_model, dt, sim=True)

            left_hand_controller = HandController(
                temp_node, robot_model, arm_side='left')
            right_hand_controller = HandController(
                temp_node, robot_model, arm_side='right')

            hands_control(
                temp_node, robot_model, ainex_robot,
                left_hand_controller, right_hand_controller, dt,
                duration=duration
            )
        finally:
            temp_node.destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MainControlNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Main Control Node")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
