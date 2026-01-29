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

    # --- Left Hand ---
    left_current_matrix = robot_model.left_hand_pose()
    left_cur = pin.SE3(left_current_matrix[:3, :3], left_current_matrix[:3, 3])

    # Target: 5cm Forward (+X), 5cm Up (+Z) in ABSOLUTE BASE FRAME
    # Maintains current rotation
    left_target = pin.SE3(left_cur.rotation, left_cur.translation + np.array([0.05, 0.0, 0.05]))
    node.get_logger().info(f"Left Start: {left_cur.translation.T} | Target: {left_target.translation.T}")
    left_hand_controller.set_target_pose(left_target, duration=3.0, type='abs')

    # --- Right Hand ---
    right_current_matrix = robot_model.right_hand_pose()
    right_cur = pin.SE3(right_current_matrix[:3, :3], right_current_matrix[:3, 3])

    # Target: 5cm Forward (+X), 5cm Up (+Z) in ABSOLUTE BASE FRAME
    right_target = pin.SE3(right_cur.rotation, right_cur.translation + np.array([0.05, 0.0, 0.05]))
    node.get_logger().info(f"Right Start: {right_cur.translation.T} | Target: {right_target.translation.T}")
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

    node.get_logger().info("Running hands_control (Reaching out totarget object)")

    if duration is not None:
        node.get_logger().info(
            f"hands_control will run for {duration} seconds.")

    # Latest ArUco message
    aruco_pose_msg = {"msg": None}

    def aruco_callback(msg: PoseStamped):
        aruco_pose_msg["msg"] = msg

    node.create_subscription(PoseStamped, "detected_object_pose", aruco_callback, 10)

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
    # We want:
    # X_opt (Right)   -> -Y_link (Right)
    # Y_opt (Down)    -> -Z_link (Down)
    # Z_opt (Forward) ->  X_link (Forward)
    #
    # Matrix columns are X_opt, Y_opt, Z_opt in link frame:
    # [ 0,  0,  1]  <- X comp
    # [-1,  0,  0]  <- Y comp
    # [ 0, -1,  0]  <- Z comp

    R_clink_opt = np.array([
        [0.0,  0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0]
    ])
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

        # Get target position from vision
        target_pos = marker_pos_base

        # --- LOGIC: Select Hand based on Distance to Object ---
        # Get current hand positions
        # Note: robot_model is updated inside ainex_robot.update(), so these are fresh
        left_pos = robot_model.left_hand_pose()[:3, 3]
        right_pos = robot_model.right_hand_pose()[:3, 3]
        
        dist_l = np.linalg.norm(left_pos - target_pos)
        dist_r = np.linalg.norm(right_pos - target_pos)
        
        # Hysteresis buffer to prevent jitter
        # To switch from Left -> Right, Right must be significantly closer
        HYSTERESIS = 0.05 
        
        desired_left = False
        desired_right = False
        
        # Decision Logic
        if left_tracking:
            # Currently using Left. Switch only if Right is MUCH closer
            if dist_r < (dist_l - HYSTERESIS):
                desired_right = True
                desired_left = False
                node.get_logger().info(f"Switching to RIGHT hand (dL={dist_l:.3f}, dR={dist_r:.3f})")
            else:
                desired_left = True
        elif right_tracking:
            # Currently using Right. Switch only if Left is MUCH closer
            if dist_l < (dist_r - HYSTERESIS):
                desired_left = True
                desired_right = False
                node.get_logger().info(f"Switching to LEFT hand (dL={dist_l:.3f}, dR={dist_r:.3f})")
            else:
                desired_right = True
        else:
            # No hand tracking yet. Pick the closest one.
            if dist_l < dist_r:
                desired_left = True
                node.get_logger().info(f"Initializing with LEFT hand (dL={dist_l:.3f} < dR={dist_r:.3f})")
            else:
                desired_right = True
                node.get_logger().info(f"Initializing with RIGHT hand (dR={dist_r:.3f} < dL={dist_l:.3f})")
        
        # Publish active hand
        if desired_left:
            active_hand_pub.publish(String(data="left"))
        else:
            active_hand_pub.publish(String(data="right"))
            
        # --------------------------------------------------------

        # ----- CONTROL LEFT ARM -----
        if desired_left:
            # Track marker with left arm
            # Use HOME rotation + Target Position
            left_target = pin.SE3(left_home_se3.rotation, target_pos)
            
            left_hand_controller.set_target_pose(
                left_target, duration=0.2, type='abs')
            left_tracking = True
        else:
            # Marker no longer on left side/front -> send left arm home once
            if left_tracking:
                node.get_logger().info("Left arm going home.")
                left_hand_controller.set_target_pose(
                    left_home_se3, duration=2.0, type='abs')
                left_tracking = False

        # ----- CONTROL RIGHT ARM -----
        if desired_right:
            # Use HOME rotation + Target Position
            right_target = pin.SE3(right_home_se3.rotation, target_pos)

            right_hand_controller.set_target_pose(
                right_target, duration=0.2, type='abs')
            right_tracking = True
        else:
            if right_tracking:
                node.get_logger().info("Right arm going home.")
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

    # Sim parameter (default to False for real robot usage now)
    node.declare_parameter("sim", False)
    sim = bool(node.get_parameter("sim").value)
    
    if sim:
        node.get_logger().warn(">>> RUNNING IN SIMULATION MODE (No hardware commands will be sent) <<<")
    else:
        node.get_logger().info(">>> RUNNING IN REAL ROBOT MODE <<<")

    dt = 0.05

    try:
        pkg = get_package_share_directory("ainex_description")
        urdf_path = pkg + "/urdf/ainex.urdf"

        robot_model = AiNexModel(node, urdf_path)
        ainex_robot = AinexRobot(node, robot_model, dt, sim=sim)

        # Initial posture (using current robot state)
        if sim:
            q_init = np.zeros(robot_model.model.nq)
        else:
            q_init = ainex_robot.q.copy()
            
        # Set arm joints to a safe, bent configuration to avoid singularities (stretching)
        # r_sho_roll / l_sho_roll: Arms slightly out
        q_init[robot_model.get_joint_id('r_sho_roll')] = -0.2  # Negative is out for right? Checking logic
        q_init[robot_model.get_joint_id('l_sho_roll')] = 0.2
        
        # r_sho_pitch / l_sho_pitch: Arms slightly forward
        q_init[robot_model.get_joint_id('r_sho_pitch')] = -0.0
        q_init[robot_model.get_joint_id('l_sho_pitch')] = 0.0

        # r_el_pitch / l_el_pitch: Elbows bent (important!)
        # Assuming 0 is straight, we want -1.57 (90 deg) or similar
        # Based on commented code: r_el_yaw was used? Checking URDF...
        # The URDF has r_el_pitch and r_el_yaw. Pitch usually bends the elbow.
        
        # Let's try forcing a known "Ready" pose for the arms using ainex_robot logic compatibility
        # We need to be careful about the sign.
        # Safe bet: Move them to a "Crouch-like" or "Boxer" pose
        
        # From previous commented code:
        # q_init[robot_model.get_joint_id('r_sho_roll')] = 1.51  <- This looks like max limit?
        # q_init[robot_model.get_joint_id('l_sho_roll')] = -1.51
        
        node.get_logger().info("Moving to Safe Initial Posture (Arms Bent)...")
        # Define a safe "Home" pose for arms
        # Right Arm
        q_init[robot_model.get_joint_id('r_sho_pitch')] = 0.0
        q_init[robot_model.get_joint_id('r_sho_roll')] = -0.2
        q_init[robot_model.get_joint_id('r_el_pitch')] = -1.0 # Bent
        
        # Left Arm
        q_init[robot_model.get_joint_id('l_sho_pitch')] = 0.0
        q_init[robot_model.get_joint_id('l_sho_roll')] = 0.2
        q_init[robot_model.get_joint_id('l_el_pitch')] = -1.0 # Bent
        
        ainex_robot.move_to_initial_position(q_init)
        
        # Update model with this new pose so controllers start correctly
        robot_model.update_model(q_init, np.zeros(robot_model.model.nv))
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
