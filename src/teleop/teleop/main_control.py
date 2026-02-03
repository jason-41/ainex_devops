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
from servo_service.msg import InstructionAfterLLM #, GraspFeedback
from std_msgs.msg import Int32, Bool

from ament_index_python.packages import get_package_share_directory
import numpy as np
from geometry_msgs.msg import PoseStamped

# Import other control modules
from teleop.walk_to_Aruco import AinexWalkToAruco
from teleop.turn_around import TurnAroundNode
from teleop.grasp import AinexGraspNode
from teleop.degrasp import AinexDegraspNode
from teleop.crouch import CrouchNode
# from ainex_controller.ainex_model import AiNexModel
# from ainex_controller.ainex_robot import AinexRobot
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

        # self.grasp_sub = self.create_subscription(
        #     GraspFeedback,
        #     'grasp_ok',
        #     self.grasp_callback,
        #     10
        # )

        self.aruco_pose_sub = self.create_subscription(
            PoseStamped,
            '/aruco_pose',
            self.aruco_pose_callback,
            10
        )

        # --- Publishers ---
        self.vision_pub = self.create_publisher(
            Int32,
            '/aruco_target',
            10
        )

        # --- State Variables ---
        self.current_instruction = None
        self.grasp_success = False
        self.latest_aruco_pose = None
        self.state = "IDLE"
        self.stop_requested = False

        # --- LLM instruction fields ---
        self.object_color = "green"
        self.object_shape = "cube"
        self.pickup_location = 25
        self.destination_location = 33

        self.get_logger().info("MainControlNode Initialized. Waiting for LLM instructions...")



    def llm_callback(self, msg):
        self.get_logger().info(
            f"Received LLM Instruction: Pickup {msg.object_color} {msg.object_shape} at Loc {msg.pickup_location}")
        self.current_instruction = msg
        self.object_color = msg.object_color
        self.object_shape = msg.object_shape
        self.pickup_location = msg.pickup_location
        self.destination_location = msg.destination_location
        # Trigger the workflow if idle
        if self.state == "IDLE":
            self.start_workflow()

    # def grasp_callback(self, msg):
    #     if msg.grasp_ok:
    #         self.get_logger().info("Received Grasp Success Feedback!")
    #         self.grasp_success = True

    def aruco_pose_callback(self, msg):
        # print(msg.data.pose)
        self.latest_aruco_pose = msg

    def call_aruco(self, aruco_id):
        self.get_logger().info(f"Calling ArUco ID: {aruco_id}")
        
        msg = Int32()
        msg.data = int(aruco_id)
        self.vision_pub.publish(msg)
        
        self.latest_aruco_pose = None
        timeout = 5.0
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.latest_aruco_pose:
                self.get_logger().info(f"Received pose for ID {aruco_id}")
                return self.latest_aruco_pose
            time.sleep(0.1)
            
        self.get_logger().warn(f"Timeout waiting for pose: {aruco_id}")
        return None

    def publish_vision_target(self, aruco_id):
        if not self.current_instruction:
            return

        msg = Int32()
        msg.data = int(aruco_id)
        self.vision_pub.publish(msg)
        self.get_logger().info(f"Published Vision Target (Int32): ID {aruco_id}")

    def start_workflow(self):
        # Run in a separate thread to avoid blocking callbacks
        threading.Thread(target=self._workflow_logic, daemon=True).start()

    def _workflow_logic(self):
        try:
            self.state = "WORKING"
            # Sequence requested for workflow logic
            self.call_aruco(self.pickup_location)
            self.run_walk_to_aruco()
            # self.run_crouch()
            active_side = self.run_grasp()
            time.sleep(2)
            self.run_stand()
            time.sleep(2)
            self.call_aruco(self.destination_location)
            self.run_turn_around(330.0)

            self.run_walk_to_aruco()
            
            # Run Degrasp
            self.run_degrasp(active_side)
            time.sleep(5)

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
            while rclpy.ok() and not walk_node.finished and not self.stop_requested:
                rclpy.spin_once(walk_node, timeout_sec=0.05)
        finally:
            # if self.stop_requested:
            walk_node.manual_stop = True
            walk_node.destroy_node()

    def run_turn_around(self, degrees=180.0):
        """Turn robot by specified degrees."""
        turn_node = TurnAroundNode()
        # Set the degrees parameter
        turn_node.set_parameters([rclpy.parameter.Parameter('degrees', rclpy.Parameter.Type.DOUBLE, degrees)])
        try:
            turn_node.execute_turn()
        finally:
            turn_node.destroy_node()

    def run_grasp(self):
        """Run grasp node directly (executes complete grasp sequence)."""
        active_side = None
        try:
            grasp_node = AinexGraspNode()
            # AinexGraspNode.__init__ calls self.run() automatically,
            # so the grasp sequence completes during initialization
            if hasattr(grasp_node, 'active_side'):
                active_side = grasp_node.active_side
            grasp_node.destroy_node()
            self.get_logger().info(f"Grasp completed successfully. Side: {active_side}")
        except Exception as e:
            self.get_logger().error(f"Grasp failed: {e}")
        return active_side

    def run_degrasp(self, active_side):
        """Run degrasp node directly."""
        if not active_side:
            self.get_logger().warn("Skipping degrasp: no active side.")
            return

        try:
            self.get_logger().info(f"Starting Degrasp for side: {active_side}")
            degrasp_node = AinexDegraspNode()
            degrasp_node.active_hand = active_side
            # We skip waiting for grasp_done since we are sequencing it manually
            degrasp_node.set_parameters([
                rclpy.parameter.Parameter('wait_for_grasp_done', rclpy.Parameter.Type.BOOL, False)
            ])
            degrasp_node.run()
            degrasp_node.destroy_node()
            self.get_logger().info("Degrasp completed.")
        except Exception as e:
            self.get_logger().error(f"Degrasp failed: {e}")

    def run_stand(self, duration=3):
        """Make robot stand up."""
        try:
            # Create a temporary context for this node
            stand_node = CrouchNode()
            stand_node.set_parameters([rclpy.parameter.Parameter('action', rclpy.Parameter.Type.STRING, 'stand'),
                                       rclpy.parameter.Parameter('duration', rclpy.Parameter.Type.DOUBLE, duration)])
            stand_node.run()
            stand_node.destroy_node()
            self.get_logger().info("Stand completed.")
        except Exception as e:
            self.get_logger().error(f"Stand failed: {e}")

    def run_crouch(self, duration=1.5):
        """Make robot crouch down."""
        try:
            crouch_node = CrouchNode()
            crouch_node.set_parameters([rclpy.parameter.Parameter('action', rclpy.Parameter.Type.STRING, 'crouch'),
                                        rclpy.parameter.Parameter('duration', rclpy.Parameter.Type.DOUBLE, duration)])
            crouch_node.run()
            crouch_node.destroy_node()
            self.get_logger().info("Crouch completed.")
        except Exception as e:
            self.get_logger().error(f"Crouch failed: {e}")

    def stop_all(self):
        """Emergency stop - stop all robot motion."""
        try:
            self.stop_requested = True
            # Signal walk_to_aruco to stop immediately
            if not hasattr(self, 'walk_stop_pub'):
                self.walk_stop_pub = self.create_publisher(Bool, '/walk_stop', 10)
            self.walk_stop_pub.publish(Bool(data=True))

            from geometry_msgs.msg import Twist
            # Publish zero velocity to stop walking
            stop_twist = Twist()
            # Create a temporary publisher if needed
            if not hasattr(self, 'emergency_stop_pub'):
                self.emergency_stop_pub = self.create_publisher(Twist, '/cmd_vel', 10)
            
            # Send stop command multiple times to ensure it's received
            for _ in range(5):
                self.emergency_stop_pub.publish(stop_twist)
                time.sleep(0.05)
            
            self.get_logger().warn("Emergency stop executed - all motion halted.")
        except Exception as e:
            self.get_logger().error(f"Emergency stop failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = MainControlNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    # Test sequence
    # node.run_stand()

    
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.stop_all()  # Emergency stop all motion
        node.get_logger().info("Shutting down Main Control Node")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
