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
from std_msgs.msg import String     # <-- NEW
from scipy.spatial.transform import Rotation as R

"""
This is a refactored version of the ainex_hands_control_node.py
"""

# ============================================================
# Test — Fixed 6cm movement
# ============================================================


def run_test(node, robot_model, ainex_robot,
             left_hand_controller, right_hand_controller, dt):

    node.get_logger().info("Running Test (Fixed 6cm movement)…")

    # Left hand initial → target
    left_current_matrix = robot_model.left_hand_pose()
    left_cur = pin.SE3(left_current_matrix[:3, :3],  # rotation matrix
                       left_current_matrix[:3, 3])  # translation vector

    left_target = pin.SE3(left_cur.rotation, left_cur.translation)
    left_target.translation[0] += 0.06
    left_target.translation[1] -= 0.08
    left_target.translation[2] -= 0.06
    left_hand_controller.set_target_pose(left_target, duration=3.0, type='abs')

    # Right hand initial → target
    # right_current_matrix = robot_model.right_hand_pose()
    # right_cur = pin.SE3(right_current_matrix[:3, :3],
    #                     right_current_matrix[:3, 3])

    right_target = pin.SE3.Identity()
    # right_target = pin.SE3(right_cur.rotation, right_cur.translation)
    right_target.translation = np.array([0.06, -0.08, 0.06])
    right_hand_controller.set_target_pose(
        right_target, duration=3.0, type='rel')

    # node.get_logger().info("Both hands moving upward 6cm…")

    # Control loop
    start_time = time.time()

    while rclpy.ok():
        vL = left_hand_controller.update(dt)
        vR = right_hand_controller.update(dt)

        ainex_robot.update(vL, vR, dt)
        rclpy.spin_once(node, timeout_sec=dt)

        if left_hand_controller.is_finished() and right_hand_controller.is_finished():
            node.get_logger().info("Test finished!")
            break

        if time.time() - start_time > 10.0:
            node.get_logger().warn("Timeout reached in Test")
            break


# ============================================================
# Object Tracking Control. Not decided yet: Grab or only track?
# ============================================================
def hands_control(node, robot_model, ainex_robot,
                  left_hand_controller, right_hand_controller, dt, duration=None):

    node.get_logger().info("Running hands_control (Aruco Tracking)…")

    if duration is not None:
        node.get_logger().info(
            f"hands_control will run for {duration} seconds.")

    # Latest ArUco message
    aruco_pose_msg = {"msg": None}

    def aruco_callback(msg: PoseStamped):
        aruco_pose_msg["msg"] = msg

    node.create_subscription(PoseStamped, "/aruco_pose", aruco_callback, 10)

    # --- Store "home" poses for both hands (current pose after q_init) ---
    left_home_matrix = robot_model.left_hand_pose()
    right_home_matrix = robot_model.right_hand_pose()
    left_home_se3 = pin.SE3(left_home_matrix[:3, :3], left_home_matrix[:3, 3])
    right_home_se3 = pin.SE3(
        right_home_matrix[:3, :3], right_home_matrix[:3, 3])

    # Track if an arm is currently following the marker
    left_tracking = False
    right_tracking = False

    # --- Get camera frame in Pinocchio model ---
    # User requested kinematic chain: BASE -> HEAD-TILT -> CAMERA LINK -> CAMERA OPTICAL LINK
    # We use head_tilt_link because URDF camera_link is fixed to body.
    try:
        head_tilt_id = robot_model.model.getFrameId("head_tilt_link")
    except Exception as e:
        node.get_logger().error(
            f"Could not find 'head_tilt_link' in Pinocchio model: {e}")
        return

    # Approximate offset from head_tilt_link to camera_link based on URDF
    # P_tilt_cam = P_body_cam - (P_body_pan + P_pan_tilt)
    # Calculated as [0.038068, 0.018573, 0.016398]
    offset_pos = np.array([0.038068, 0.018573, 0.016398])
    T_tilt_cam = pin.SE3(np.eye(3), offset_pos)

    # Standard ROS transform: camera_link -> camera_optical_link
    R_clink_opt = R.from_euler(
        'xyz', [-np.pi / 2.0, 0.0, -np.pi / 2.0]).as_matrix()
    T_clink_opt = pin.SE3(R_clink_opt, np.zeros(3))

    node.get_logger().info(
        f"Using Pinocchio frame 'head_tilt_link' (id={head_tilt_id}) + fixed offset for camera_link."
    )

    # Main tracking loop

    # Publisher for active hand side (for grasping node)
    active_hand_pub = node.create_publisher(String, '/active_hand', 10)

    start_time = time.time()

    while rclpy.ok():
        # Check timeout
        if duration is not None and (time.time() - start_time > duration):
            node.get_logger().info("hands_control duration finished.")
            break

        # Let ROS process callbacks (Aruco detections)
        rclpy.spin_once(node, timeout_sec=0.01)

        if aruco_pose_msg["msg"] is None:
            # No new detection: just keep updating controllers/robot
            vL = left_hand_controller.update(dt)
            vR = right_hand_controller.update(dt)
            ainex_robot.update(vL, vR, dt)
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

        # base_link -> head_tilt -> camera_link
        T_b_head = robot_model.data.oMf[head_tilt_id]
        T_b_clink = T_b_head * T_tilt_cam

        # base_link -> camera_optical_link
        T_b_opt = T_b_clink * T_clink_opt

        """
        Warning: The following code might affect the precision of the ArUco tracking.
        """
        ###################################################################
        # base_link -> marker
        T_b_m = T_b_opt * T_opt_m
        marker_pos_base = T_b_m.translation

        # Depth correction
        # marker_pos_base[0] *= 0.6
        marker_pos_base[0] *= 1.0
        x_b, y_b, z_b = marker_pos_base

        ####################################################################

        # Build target SE3 in base_link frame
        target = pin.SE3.Identity()
        target.translation = marker_pos_base

        # Decide which arm(s) to use
        desired_left = False
        desired_right = False
        left_target = None
        right_target = None

        if y_b > 0:
            # marker on robot's left side -> left arm only
            desired_left = True
            left_target = target
            active_hand_pub.publish(String(data="left"))
        else:
            # marker on robot's right side -> right arm only
            desired_right = True
            right_target = target
            active_hand_pub.publish(String(data="right"))

        # else:
        #     # marker roughly in front -> both arms, slightly offset
        #     desired_left = True
        #     desired_right = True
        #     left_target = pin.SE3(target.rotation,
        #                           marker_pos_base + np.array([0.0, +0.08, 0.0]))
        #     right_target = pin.SE3(target.rotation,
        #                            marker_pos_base + np.array([0.0, -0.08, 0.0]))

        # ----- CONTROL LEFT ARM -----
        if desired_left:
            # Track marker with left arm
            left_hand_controller.set_target_pose(
                left_target, duration=0.2, type='abs')
            left_tracking = True
        else:
            # Marker no longer on left side/front -> send left arm home once
            if left_tracking:
                node.get_logger().info("Marker moved away from left side -> left arm going home.")
                left_hand_controller.set_target_pose(
                    left_home_se3, duration=2.0, type='abs')
                left_tracking = False

        # ----- CONTROL RIGHT ARM -----
        if desired_right:
            right_hand_controller.set_target_pose(
                right_target, duration=0.2, type='abs')
            right_tracking = True
        else:
            if right_tracking:
                node.get_logger().info("Marker moved away from right side -> right arm going home.")
                right_hand_controller.set_target_pose(
                    right_home_se3, duration=2.0, type='abs')
                right_tracking = False

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

    node.declare_parameter("sim", True)
    sim = bool(node.get_parameter("sim").value)

    dt = 0.05

    try:
        pkg = get_package_share_directory("ainex_description")
        urdf_path = pkg + "/urdf/ainex.urdf"

        robot_model = AiNexModel(node, urdf_path)
        ainex_robot = AinexRobot(node, robot_model, dt, sim=sim)

        # Initial posture (same as working Exercise 1)
        q_init = np.zeros(robot_model.model.nq)
        q_init[robot_model.get_joint_id('r_sho_roll')] = 1.51
        q_init[robot_model.get_joint_id('l_sho_roll')] = -1.51
        q_init[robot_model.get_joint_id('r_el_yaw')] = 1.58
        q_init[robot_model.get_joint_id('l_el_yaw')] = -1.58
        ainex_robot.move_to_initial_position(q_init)
        time.sleep(2.0)

        # Controllers
        left_hand_controller = HandController(
            node, robot_model, arm_side='left')
        right_hand_controller = HandController(
            node, robot_model, arm_side='right')

        # Choose mode
        if mode == 1:
            node.get_logger().info("MODE 1: Test (Fixed 6cm movement)")
            run_test(node, robot_model, ainex_robot,
                     left_hand_controller, right_hand_controller, dt)
        else:
            node.get_logger().info("MODE 2: hands_control")
            hands_control(node, robot_model, ainex_robot,
                          left_hand_controller, right_hand_controller, dt)

    except Exception as e:
        node.get_logger().error(f"Initialization error: {e}")

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
