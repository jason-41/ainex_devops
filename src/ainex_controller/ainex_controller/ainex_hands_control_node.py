#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

import pinocchio as pin
import numpy as np
import time

from ainex_controller.ainex_model import AiNexModel
from ainex_controller.ainex_robot import AinexRobot
from ainex_controller.ainex_hand_controller import HandController

from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R   # <-- NEW


# ============================================================
# Exercise 1 — Fixed 6cm movement
# ============================================================
def run_exercise_1(node, robot_model, ainex_robot,
                   left_hand_controller, right_hand_controller, dt):

    node.get_logger().info("Running Exercise 1 (Fixed 6cm movement)…")

    # Left hand initial → target
    left_current_matrix = robot_model.left_hand_pose()
    left_cur = pin.SE3(left_current_matrix[:3, :3],
                       left_current_matrix[:3, 3])

    left_target = pin.SE3(left_cur.rotation, left_cur.translation.copy())
    left_target.translation[2] += 0.06
    left_hand_controller.set_target_pose(left_target, duration=3.0, type='abs')

    # Right hand initial → target
    right_current_matrix = robot_model.right_hand_pose()
    right_cur = pin.SE3(right_current_matrix[:3, :3],
                        right_current_matrix[:3, 3])

    right_target = pin.SE3(right_cur.rotation, right_cur.translation.copy())
    right_target.translation[2] += 0.06
    right_hand_controller.set_target_pose(right_target, duration=3.0, type='abs')

    node.get_logger().info("Both hands moving upward 6cm…")

    # Control loop
    start_time = time.time()

    while rclpy.ok():
        vL = left_hand_controller.update(dt)
        vR = right_hand_controller.update(dt)

        ainex_robot.update(vL, vR, dt)
        rclpy.spin_once(node, timeout_sec=dt)

        if left_hand_controller.is_finished() and right_hand_controller.is_finished():
            node.get_logger().info("Exercise 1 finished!")
            break

        if time.time() - start_time > 10.0:
            node.get_logger().warn("Timeout reached in Exercise 1")
            break


# ============================================================
# Exercise 2 — Aruco Tracking with proper frame transform
# ============================================================
def run_exercise_2(node, robot_model, ainex_robot,
                   left_hand_controller, right_hand_controller, dt):

    node.get_logger().info("Running Exercise 2 (Aruco Tracking)…")

    # Latest ArUco message
    aruco_pose_msg = {"msg": None}

    def aruco_callback(msg: PoseStamped):
        aruco_pose_msg["msg"] = msg

    node.create_subscription(PoseStamped, "/aruco_pose", aruco_callback, 10)

    # --- Get camera frame in Pinocchio model ---
    # URDF has "camera_link", not "camera_optical_link".
    # We treat /aruco_pose as being in camera_optical_link and use
    # the standard fixed rotation between camera_link and camera_optical_link.
    try:
        cam_frame_id = robot_model.model.getFrameId("camera_link")
    except Exception as e:
        node.get_logger().error(f"Could not find 'camera_link' in Pinocchio model: {e}")
        return

    # Standard ROS transform: camera_link -> camera_optical_link
    # (x-forward,y-left,z-up) -> (x-right,y-down,z-forward)
    R_clink_opt = R.from_euler('xyz', [-np.pi / 2.0, 0.0, -np.pi / 2.0]).as_matrix()
    T_clink_opt = pin.SE3(R_clink_opt, np.zeros(3))

    node.get_logger().info(
        f"Using Pinocchio frame 'camera_link' (id={cam_frame_id}) "
        "and standard camera_link->camera_optical_link rotation."
    )

    # Main tracking loop
    while rclpy.ok():
        # Let ROS process callbacks (Aruco detections)
        rclpy.spin_once(node, timeout_sec=0.01)

        if aruco_pose_msg["msg"] is None:
            continue

        pose = aruco_pose_msg["msg"].pose

        # Pose of marker in camera_optical_link frame (OpenCV convention)
        p_opt = np.array([pose.position.x,
                          pose.position.y,
                          pose.position.z],
                         dtype=float)
        q_opt = np.array([pose.orientation.x,
                          pose.orientation.y,
                          pose.orientation.z,
                          pose.orientation.w],
                         dtype=float)

        R_opt_m = R.from_quat(q_opt).as_matrix()
        T_opt_m = pin.SE3(R_opt_m, p_opt)     # camera_optical_link -> marker

        # base_link -> camera_link from Pinocchio
        T_b_clink = robot_model.data.oMf[cam_frame_id]

        # base_link -> camera_optical_link
        T_b_opt = T_b_clink * T_clink_opt

        # base_link -> marker
        T_b_m = T_b_opt * T_opt_m

        marker_pos_base = T_b_m.translation

        # Depth correction (necessary due to ArUco detection inaccuracies or camera calibration mistakes)
        marker_pos_base[0] *= 0.6   # Move marker 3cm closer to robot
        x_b, y_b, z_b = marker_pos_base

        # Build target SE3 in base_link frame
        target = pin.SE3.Identity()
        target.translation = marker_pos_base

        # Decide which arm(s) to use based on marker y in base_link
        # (y_b > 0 → robot's left side, y_b < 0 → right side)
        if y_b > 0.05:
            # Use left arm
            left_hand_controller.set_target_pose(target, duration=0.2, type='abs')

        elif y_b < -0.05:
            # Use right arm
            right_hand_controller.set_target_pose(target, duration=0.2, type='abs')

        else:
            # Marker roughly in front: use both arms, slightly shifted in y
            left_target = pin.SE3(target.rotation,
                                  marker_pos_base + np.array([0.0, +0.08, 0.0]))
            right_target = pin.SE3(target.rotation,
                                   marker_pos_base + np.array([0.0, -0.08, 0.0]))
            left_hand_controller.set_target_pose(left_target, duration=0.2, type='abs')
            right_hand_controller.set_target_pose(right_target, duration=0.2, type='abs')

        # Update robot
        vL = left_hand_controller.update(dt)
        vR = right_hand_controller.update(dt)
        ainex_robot.update(vL, vR, dt)


# ============================================================
# Main
# ============================================================
def main():
    rclpy.init()
    node = rclpy.create_node("ainex_hands_control_node")

    node.declare_parameter("mode", 1)
    mode = int(node.get_parameter("mode").value)

    dt = 0.05

    try:
        pkg = get_package_share_directory("ainex_description")
        urdf_path = pkg + "/urdf/ainex.urdf"

        robot_model = AiNexModel(node, urdf_path)
        ainex_robot = AinexRobot(node, robot_model, dt, sim=False)

        # Initial posture (same as your working Exercise 1)
        q_init = np.zeros(robot_model.model.nq)
        q_init[robot_model.get_joint_id('r_sho_roll')] = 1.4
        q_init[robot_model.get_joint_id('l_sho_roll')] = -1.4
        q_init[robot_model.get_joint_id('r_el_yaw')] = 1.58
        q_init[robot_model.get_joint_id('l_el_yaw')] = -1.58
        ainex_robot.move_to_initial_position(q_init)
        time.sleep(2.0)

        # Controllers
        left_hand_controller = HandController(node, robot_model, arm_side='left')
        right_hand_controller = HandController(node, robot_model, arm_side='right')

        # Choose mode
        if mode == 1:
            node.get_logger().info("MODE 1: Exercise 1")
            run_exercise_1(node, robot_model, ainex_robot,
                           left_hand_controller, right_hand_controller, dt)
        else:
            node.get_logger().info("MODE 2: Exercise 2 (Aruco Tracking)")
            run_exercise_2(node, robot_model, ainex_robot,
                           left_hand_controller, right_hand_controller, dt)

    except Exception as e:
        node.get_logger().error(f"Initialization error: {e}")

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
