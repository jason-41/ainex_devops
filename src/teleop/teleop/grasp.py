#!/usr/bin/env python3
import time
import numpy as np
import pinocchio as pin
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster
from scipy.spatial.transform import Rotation as R

from ament_index_python.packages import get_package_share_directory

from ainex_controller.ainex_model import AiNexModel
from ainex_controller.ainex_robot import AinexRobot
from ainex_controller.ainex_hand_controller import HandController


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def pose_to_se3_from_pose_msg(pose) -> pin.SE3:
    t = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=float)
    q = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w], dtype=float)
    Rot = R.from_quat(q).as_matrix()
    return pin.SE3(Rot, t)


def send_tf(br: TransformBroadcaster, node: Node, parent: str, child: str, T: pin.SE3):
    msg = TransformStamped()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = parent
    msg.child_frame_id = child
    msg.transform.translation.x = float(T.translation[0])
    msg.transform.translation.y = float(T.translation[1])
    msg.transform.translation.z = float(T.translation[2])
    q = R.from_matrix(T.rotation).as_quat()  # xyzw
    msg.transform.rotation.x = float(q[0])
    msg.transform.rotation.y = float(q[1])
    msg.transform.rotation.z = float(q[2])
    msg.transform.rotation.w = float(q[3])
    br.sendTransform(msg)


class AinexGraspNode(Node):
    def __init__(self):
        super().__init__("ainex_grasp_node")
        self.br = TransformBroadcaster(self)

        # ----------------------------
        # PARAMETERS
        # ----------------------------
        self.declare_parameter("sim", False)  # False on real robot, True for local sim testing
        self.declare_parameter("use_camera", True)
        self.declare_parameter("pose_topic", "/cube_pose")  # from cube_pose_node.py (PoseStamped in camera frame)
        self.declare_parameter("lock_target_once", True)    # lock first valid detection for the whole grasp
        self.declare_parameter("camera_frame_fallback", "camera_optical_link")  # used if msg.header.frame_id is empty

        # Hardcoded pose in CAMERA frame when use_camera:=false
        self.declare_parameter("hardcoded_cam_xyz", [0.05, 0.05, 0.05])
        self.declare_parameter("hardcoded_cam_rpy", [0.0, 0.0, 0.0])  # not used for control, but kept for TF completeness

        # Grasp offsets (position-only), in BASE frame after transform
        self.declare_parameter("pre_x_off", 0.02)     # meters (pregrasp: -x)
        self.declare_parameter("pre_z_off", 0.015)    # meters (+z)
        self.declare_parameter("approach_x_off", 0.01)  # meters (approach: -x)
        self.declare_parameter("lift_z", 0.10)        # meters

        # Speeds / durations
        self.declare_parameter("dt", 0.05)
        self.declare_parameter("pre_duration", 3.0)
        self.declare_parameter("approach_duration", 3.0)
        self.declare_parameter("lift_duration", 2.0)

        # Gripper behavior (position via velocity)
        self.declare_parameter("close_fraction", 0.70)
        self.declare_parameter("gripper_kp", 6.0)
        self.declare_parameter("gripper_vel_max", 2.0)
        self.declare_parameter("gripper_eps", 0.02)
        self.declare_parameter("squeeze_time", 0.5)

        # ----------------------------
        # LOAD MODEL + ROBOT
        # ----------------------------
        pkg = get_package_share_directory("ainex_description")
        urdf_path = pkg + "/urdf/ainex.urdf"
        self.robot_model = AiNexModel(self, urdf_path)

        self.dt = float(self.get_parameter("dt").value)
        self.sim = bool(self.get_parameter("sim").value)

        self.robot = AinexRobot(self, self.robot_model, self.dt, sim=self.sim)

        # Home posture (same as your scripts)
        q_init = np.zeros(self.robot_model.model.nq)
        q_init[self.robot_model.get_joint_id("r_sho_roll")] = 1.4
        q_init[self.robot_model.get_joint_id("l_sho_roll")] = -1.4
        q_init[self.robot_model.get_joint_id("r_el_yaw")] = 1.58
        q_init[self.robot_model.get_joint_id("l_el_yaw")] = -1.58
        self.robot.move_to_initial_position(q_init)
        time.sleep(1.0)

        # Hand controller (position-only)
        self.hand_ctrl = HandController(self, self.robot_model, arm_side="right")
        self.hand_ctrl.set_yaw_control(False)  # initial deployment: no orientation control

        # ----------------------------
        # CAMERA FRAME RESOLUTION (Pinocchio)
        # ----------------------------
        self.cam_frame_id = None
        self.cam_frame_name = None

        for name in ["camera_optical_link", "camera_link"]:
            fid = self.robot_model.model.getFrameId(name)
            if fid < self.robot_model.model.nframes:
                self.cam_frame_id = fid
                self.cam_frame_name = name
                self.get_logger().info(f"Using camera frame '{self.cam_frame_name}' (frame id={self.cam_frame_id})")
                break

        if self.cam_frame_id is None:
            self.get_logger().warn("Could not find 'camera_optical_link' or 'camera_link' in URDF. "
                                   "Will still run, but camera->base transform will fail unless msg.header.frame_id matches a frame.")

        # ----------------------------
        # TARGET POSE INPUT
        # ----------------------------
        self.use_camera = bool(self.get_parameter("use_camera").value)
        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.lock_target_once = bool(self.get_parameter("lock_target_once").value)
        self.latest_pose_msg = None
        self.latest_pose_t = None
        self.locked_T_b_obj = None

        if self.use_camera:
            self.create_subscription(PoseStamped, self.pose_topic, self._pose_cb, 10)
            self.get_logger().info(f"Listening for object pose on {self.pose_topic} (PoseStamped, camera frame).")
        else:
            self.get_logger().warn("use_camera:=false -> using hardcoded pose in camera frame.")

        # ----------------------------
        # GRIPPER SETUP
        # ----------------------------
        gripper_joint_name = "r_gripper"
        jid = self.robot_model.model.getJointId(gripper_joint_name)
        self.qg_idx = self.robot_model.model.joints[jid].idx_q
        self.vg_idx = self.robot_model.model.joints[jid].idx_v

        q_lo = float(self.robot_model.model.lowerPositionLimit[self.qg_idx])
        q_hi = float(self.robot_model.model.upperPositionLimit[self.qg_idx])
        margin = 0.15 * (q_hi - q_lo)

        # Your earlier discovery: +q opens → open near q_hi, close toward q_lo
        self.gripper_open_q = q_hi - margin
        close_fraction = float(self.get_parameter("close_fraction").value)
        self.gripper_close_q = self.gripper_open_q + close_fraction * ((q_lo + margin) - self.gripper_open_q)

        self.get_logger().info(f"Gripper limits [{q_lo:.2f},{q_hi:.2f}] open={self.gripper_open_q:.3f} close={self.gripper_close_q:.3f}")

        self.gripper_kp = float(self.get_parameter("gripper_kp").value)
        self.gripper_vel_max = float(self.get_parameter("gripper_vel_max").value)
        self.gripper_eps = float(self.get_parameter("gripper_eps").value)
        self.squeeze_time = float(self.get_parameter("squeeze_time").value)

        # ----------------------------
        # RUN STATE MACHINE (blocking)
        # ----------------------------
        self.run()

    def _pose_cb(self, msg: PoseStamped):
        self.latest_pose_msg = msg
        self.latest_pose_t = time.time()

    def _get_T_b_c(self, frame_name_hint: str = None) -> pin.SE3:
        """
        Get base_link -> camera transform from Pinocchio using current robot state.
        If frame_name_hint matches a Pinocchio frame, use that.
        """
        if frame_name_hint:
            fid = self.robot_model.model.getFrameId(frame_name_hint)
            if fid < self.robot_model.model.nframes:
                return self.robot_model.data.oMf[fid]

        if self.cam_frame_id is None:
            raise RuntimeError("No camera frame in model and frame_name_hint not found.")
        return self.robot_model.data.oMf[self.cam_frame_id]

    def _get_object_T_b(self) -> pin.SE3:
        """
        Returns base_link -> object transform.
        - If use_camera: uses latest PoseStamped in camera frame + URDF camera extrinsics.
        - Else: uses hardcoded pose in camera frame.
        """
        if self.lock_target_once and self.locked_T_b_obj is not None:
            return self.locked_T_b_obj

        # Determine T_c_obj
        if self.use_camera:
            if self.latest_pose_msg is None:
                raise RuntimeError("No pose received yet.")
            # Optional freshness check
            if self.latest_pose_t is None or (time.time() - self.latest_pose_t) > 1.0:
                raise RuntimeError("Pose is stale (>1.0s).")

            T_c_obj = pose_to_se3_from_pose_msg(self.latest_pose_msg.pose)
            cam_frame_hint = self.latest_pose_msg.header.frame_id if self.latest_pose_msg.header.frame_id else None
        else:
            xyz = self.get_parameter("hardcoded_cam_xyz").value
            rpy = self.get_parameter("hardcoded_cam_rpy").value
            t = np.array([xyz[0], xyz[1], xyz[2]], dtype=float)
            Rot = R.from_euler("xyz", [rpy[0], rpy[1], rpy[2]]).as_matrix()
            T_c_obj = pin.SE3(Rot, t)
            cam_frame_hint = str(self.get_parameter("camera_frame_fallback").value)

        # Get T_b_c from Pinocchio (depends on current q)
        T_b_c = self._get_T_b_c(cam_frame_hint)
        T_b_obj = T_b_c * T_c_obj

        # cache
        if self.lock_target_once:
            self.locked_T_b_obj = T_b_obj

        # TF for RViz/debug
        try:
            send_tf(self.br, self, "base_link", "camera_frame_used", pin.SE3(T_b_c.rotation, T_b_c.translation))
            send_tf(self.br, self, "base_link", "object_in_base", pin.SE3(T_b_obj.rotation, T_b_obj.translation))
        except Exception:
            pass

        return T_b_obj

    def _drive_gripper(self, q_target: float):
        qg = float(self.robot.q[self.qg_idx])
        vg = self.gripper_kp * (q_target - qg)
        vg = float(np.clip(vg, -self.gripper_vel_max, self.gripper_vel_max))
        if abs(q_target - qg) < self.gripper_eps:
            vg = 0.0
        self.robot.v[self.vg_idx] = vg

    def run(self):
        # ----------------------------
        # WAIT FOR TARGET
        # ----------------------------
        start_wait = time.time()
        T_b_obj = None
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)

            # For real robot, refresh model with measured joints if possible (optional improvement)
            if (not self.sim) and hasattr(self.robot, "read_joint_positions_from_robot"):
                try:
                    q_real = self.robot.read_joint_positions_from_robot()
                    self.robot.q = q_real
                    self.robot_model.update_model(self.robot.q, self.robot.v)
                except Exception:
                    pass

            try:
                T_b_obj = self._get_object_T_b()
                break
            except Exception as e:
                if time.time() - start_wait > 10.0:
                    raise RuntimeError(f"Timeout waiting for object pose: {e}")

        obj_pos = T_b_obj.translation.copy()
        self.get_logger().info(f"[TARGET] object_base={obj_pos.tolist()}")

        # ----------------------------
        # BUILD POSITION TARGETS (POSITION ONLY)
        # ----------------------------
        pre_x = float(self.get_parameter("pre_x_off").value)
        pre_z = float(self.get_parameter("pre_z_off").value)
        approach_x = float(self.get_parameter("approach_x_off").value)
        lift_z = float(self.get_parameter("lift_z").value)

        pre_pos = obj_pos + np.array([-pre_x, 0.0, pre_z], dtype=float)
        approach_pos = obj_pos + np.array([-approach_x, 0.0, 0.0], dtype=float)
        lift_pos = approach_pos + np.array([0.0, 0.0, lift_z], dtype=float)

        # keep current hand rotation (we ignore orientation for initial deployment)
        H = self.robot_model.right_hand_pose()
        R_hand = H[:3, :3]

        T_pre = pin.SE3(R_hand, pre_pos)
        T_approach = pin.SE3(R_hand, approach_pos)
        T_lift = pin.SE3(R_hand, lift_pos)

        self.get_logger().info(f"[PREGRASP] target={pre_pos.tolist()}")
        self.get_logger().info(f"[APPROACH] target={approach_pos.tolist()}")
        self.get_logger().info(f"[LIFT] target={lift_pos.tolist()}")

        # ----------------------------
        # PHASE 1: PREGRASP (open gripper)
        # ----------------------------
        self._drive_gripper(self.gripper_open_q)
        self.hand_ctrl.set_yaw_control(False)
        self.hand_ctrl.set_target_pose(T_pre, duration=float(self.get_parameter("pre_duration").value), type="abs")

        phase = "PREGRASP"
        t0 = time.time()

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=self.dt)

            # Real robot feedback refresh (optional)
            if (not self.sim) and hasattr(self.robot, "read_joint_positions_from_robot"):
                try:
                    q_real = self.robot.read_joint_positions_from_robot()
                    self.robot.q = q_real
                    self.robot_model.update_model(self.robot.q, self.robot.v)
                except Exception:
                    pass

            if phase in ("PREGRASP", "APPROACH"):
                self._drive_gripper(self.gripper_open_q)

            if phase == "PREGRASP":
                v_hand = self.hand_ctrl.update(self.dt)
                self.robot.update(None, v_hand, self.dt)

                if self.hand_ctrl.is_finished():
                    phase = "APPROACH"
                    self.get_logger().info("Phase: APPROACH")
                    self.hand_ctrl.set_target_pose(T_approach, duration=float(self.get_parameter("approach_duration").value), type="abs")

            elif phase == "APPROACH":
                # slow down for approach on real robot
                self.hand_ctrl.linear_vel_limit = 0.02
                v_hand = self.hand_ctrl.update(self.dt)
                self.robot.update(None, v_hand, self.dt)

                if self.hand_ctrl.is_finished():
                    phase = "CLOSE"
                    self.get_logger().info("Phase: CLOSE")

            elif phase == "CLOSE":
                # freeze arm during close
                self.robot.update(None, np.zeros(4), self.dt)
                self._drive_gripper(self.gripper_close_q)
                self.robot.update(None, np.zeros(4), self.dt)

                if abs(float(self.robot.q[self.qg_idx]) - float(self.gripper_close_q)) < self.gripper_eps:
                    phase = "SQUEEZE"
                    t_s = time.time()
                    self.get_logger().info("Phase: SQUEEZE")

            elif phase == "SQUEEZE":
                self.robot.update(None, np.zeros(4), self.dt)
                self._drive_gripper(self.gripper_close_q)
                self.robot.update(None, np.zeros(4), self.dt)

                if time.time() - t_s >= self.squeeze_time:
                    phase = "LIFT"
                    self.get_logger().info("Phase: LIFT")
                    self.hand_ctrl.linear_vel_limit = 0.08
                    self.hand_ctrl.set_target_pose(T_lift, duration=float(self.get_parameter("lift_duration").value), type="abs")

            elif phase == "LIFT":
                self._drive_gripper(self.gripper_close_q)
                v_hand = self.hand_ctrl.update(self.dt)
                self.robot.update(None, v_hand, self.dt)

                if self.hand_ctrl.is_finished():
                    self.get_logger().info("Done.")
                    break

            # basic safety timeout
            if time.time() - t0 > 60.0:
                self.get_logger().warn("Timeout, stopping.")
                break


def main():
    rclpy.init()
    node = None
    try:
        node = AinexGraspNode()
    except Exception as e:
        if node is not None:
            node.get_logger().error(f"ainex_grasp_node failed: {e}")
        else:
            print(f"ainex_grasp_node failed: {e}")
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
