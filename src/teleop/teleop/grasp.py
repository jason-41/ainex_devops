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
        self.declare_parameter("sim", False)
        self.declare_parameter("use_camera", True)

        # IMPORTANT: match hands_control topic by default
        self.declare_parameter("pose_topic", "detected_object_pose")

        self.declare_parameter("lock_target_once", False)

        # Hardcoded pose in CAMERA frame when use_camera:=false
        self.declare_parameter("hardcoded_cam_xyz", [0.0, 0.0, 0.05])
        self.declare_parameter("hardcoded_cam_rpy", [0.0, 0.0, 0.0])

        # Grasp offsets (position-only), in BASE frame after transform
        self.declare_parameter("pre_x_off", 0.02)
        self.declare_parameter("pre_z_off", 0.015)
        self.declare_parameter("approach_x_off", 0.0)
        self.declare_parameter("lift_z", 0.10)

        # Durations
        self.declare_parameter("pre_duration", 5.0)
        self.declare_parameter("approach_duration", 5.0)
        self.declare_parameter("lift_duration", 3.0)

        # Real-robot control loop period (fixed)
        self.declare_parameter("dt_cmd", 0.01)          # 100 Hz fixed command loop
        self.declare_parameter("feedback_hz", 0.0)      # default OFF

        # Gripper
        self.declare_parameter("close_fraction", 0.4)
        self.declare_parameter("gripper_kp", 6.0)
        self.declare_parameter("gripper_vel_max", 2.0)
        self.declare_parameter("gripper_eps", 0.02)
        self.declare_parameter("squeeze_time", 0.5)

        self.declare_parameter("phase_settle_s", 0.25)

        # ----------------------------
        # LOAD MODEL + ROBOT
        # ----------------------------
        pkg = get_package_share_directory("ainex_description")
        urdf_path = pkg + "/urdf/ainex.urdf"
        self.robot_model = AiNexModel(self, urdf_path)

        self.sim = bool(self.get_parameter("sim").value)
        dt_cmd = float(self.get_parameter("dt_cmd").value)

        self.robot = AinexRobot(self, self.robot_model, dt_cmd, sim=self.sim)

        # ----------------------------
        # Starting posture (exactly as you requested)
        # ----------------------------
        q_init = np.array([
            0.06702064, -0.47752208, -0.01675516,  0.00418879,
            -0.87126839,  2.33315611,  1.47864294,  0.03769911,
            -0.29740411, -1.24191744,  0.02932153, -1.65457213,
            0.0,        -0.01675516,  0.00837758,  0.83775806,
            -2.22843647, -1.41162229, -0.03769911, -0.26808256,
            1.36758114, 0.10890855,  1.68389368,  0.74979347
        ], dtype=float)
        self.get_logger().info(f"initial pose {q_init}")
        self.robot.move_to_initial_position(q_init)
        time.sleep(1.0)

        self.hand_ctrl = None

        # ----------------------------
        # MATCH hands_control camera chain:
        # base -> head_tilt_link -> (fixed offset) -> camera_link -> (fixed rot) -> camera_optical_link
        # ----------------------------
        try:
            self.head_tilt_id = self.robot_model.model.getFrameId("head_tilt_link")
            if self.head_tilt_id >= self.robot_model.model.nframes:
                raise RuntimeError("head_tilt_link frame id out of range")
        except Exception as e:
            raise RuntimeError(f"Could not find 'head_tilt_link' in Pinocchio model: {e}")

        # Same offset as hands_control
        offset_pos = np.array([0.038068, 0.018573, 0.016398], dtype=float)
        self.T_tilt_cam = pin.SE3(np.eye(3), offset_pos)

        # Same camera_link -> camera_optical_link rotation as hands_control
        R_clink_opt = np.array([
            [0.0,  0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0]
        ], dtype=float)
        self.T_clink_opt = pin.SE3(R_clink_opt, np.zeros(3))

        self.get_logger().info(
            f"Using camera chain via head_tilt_link (id={self.head_tilt_id}) + fixed offset + fixed optical rotation."
        )

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
            self.get_logger().info(f"Listening for object pose on {self.pose_topic} (PoseStamped, expected camera_optical_link).")
        else:
            self.get_logger().warn("use_camera:=false -> using hardcoded pose in camera_optical_link frame.")

        # ----------------------------
        # GRIPPER INDICES (both sides)
        # ----------------------------
        self.gripper_indices = {}
        for jname in ["l_gripper", "r_gripper"]:
            jid = self.robot_model.model.getJointId(jname)
            q_idx = self.robot_model.model.joints[jid].idx_q
            v_idx = self.robot_model.model.joints[jid].idx_v
            self.gripper_indices[jname] = (q_idx, v_idx)

        self.qg_idx = None
        self.vg_idx = None
        self.gripper_open_q = None
        self.gripper_close_q = None

        self.gripper_kp = float(self.get_parameter("gripper_kp").value)
        self.gripper_vel_max = float(self.get_parameter("gripper_vel_max").value)
        self.gripper_eps = float(self.get_parameter("gripper_eps").value)
        self.squeeze_time = float(self.get_parameter("squeeze_time").value)

        self._t_last_feedback = 0.0
        
        self.run()

    def _pose_cb(self, msg: PoseStamped):
        self.latest_pose_msg = msg
        self.latest_pose_t = time.time()

    def _get_T_b_opt(self) -> pin.SE3:
        """
        EXACTLY like hands_control:
          T_b_opt = T_b_head * T_tilt_cam * T_clink_opt
        """
        T_b_head = self.robot_model.data.oMf[self.head_tilt_id]
        T_b_clink = T_b_head * self.T_tilt_cam
        T_b_opt = T_b_clink * self.T_clink_opt
        return T_b_opt

    def _get_object_T_b(self) -> pin.SE3:
        if self.lock_target_once and self.locked_T_b_obj is not None:
            return self.locked_T_b_obj

        # T_opt_obj (camera_optical_link -> object)
        if self.use_camera:
            if self.latest_pose_msg is None:
                raise RuntimeError("No pose received yet.")
            if self.latest_pose_t is None or (time.time() - self.latest_pose_t) > 1.0:
                raise RuntimeError("Pose is stale (>1.0s).")

            # We assume the incoming PoseStamped is in camera_optical_link, same as hands_control.
            # If frame_id is present and differs, we warn (but do not guess transforms).
            fid = self.latest_pose_msg.header.frame_id.strip() if self.latest_pose_msg.header.frame_id else ""
            if fid and fid != "camera_optical_link":
                self.get_logger().warn(f"Pose frame_id='{fid}' (expected 'camera_optical_link'). Using it as optical anyway.")

            T_opt_obj = pose_to_se3_from_pose_msg(self.latest_pose_msg.pose)

        else:
            xyz = self.get_parameter("hardcoded_cam_xyz").value
            rpy = self.get_parameter("hardcoded_cam_rpy").value
            t = np.array([xyz[0], xyz[1], xyz[2]], dtype=float)
            Rot = R.from_euler("xyz", [rpy[0], rpy[1], rpy[2]]).as_matrix()
            T_opt_obj = pin.SE3(Rot, t)

        # base_link -> camera_optical_link (hands_control chain)
        T_b_opt = self._get_T_b_opt()

        # base_link -> object
        T_b_obj = T_b_opt * T_opt_obj

        if self.lock_target_once:
            self.locked_T_b_obj = T_b_obj

        # Debug TF
        try:
            send_tf(self.br, self, "base_link", "camera_optical_used", pin.SE3(T_b_opt.rotation, T_b_opt.translation))
            send_tf(self.br, self, "base_link", "object_in_base", pin.SE3(T_b_obj.rotation, T_b_obj.translation))
        except Exception:
            pass

        return T_b_obj

    def _maybe_refresh_from_robot(self):
        if self.sim:
            return
        if not hasattr(self.robot, "read_joint_positions_from_robot"):
            return

        hz = float(self.get_parameter("feedback_hz").value)
        if hz <= 0.0:
            return

        period = 1.0 / hz
        now = time.monotonic()
        if (now - self._t_last_feedback) < period:
            return

        self._t_last_feedback = now
        try:
            q_real = self.robot.read_joint_positions_from_robot()
            self.robot.q = q_real
            self.robot_model.update_model(self.robot.q, self.robot.v)
        except Exception:
            pass

    def _drive_gripper(self, q_target: float):
        qg = float(self.robot.q[self.qg_idx])
        vg = self.gripper_kp * (q_target - qg)
        vg = float(np.clip(vg, -self.gripper_vel_max, self.gripper_vel_max))
        if abs(q_target - qg) < self.gripper_eps:
            vg = 0.0
        self.robot.v[self.vg_idx] = vg

    def _sleep_to_rate(self, t_cycle_start: float, dt_cmd: float):
        remaining = dt_cmd - (time.monotonic() - t_cycle_start)
        if remaining > 0.0:
            time.sleep(remaining)

    def run(self):
        # WAIT FOR TARGET
        start_wait = time.time()
        T_b_obj = None
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            self._maybe_refresh_from_robot()
            try:
                T_b_obj = self._get_object_T_b()
                break
            except Exception as e:
                if time.time() - start_wait > 10.0:
                    raise RuntimeError(f"Timeout waiting for object pose: {e}")

        obj_pos = T_b_obj.translation.copy()
        self.get_logger().info(f"[TARGET] object_base={obj_pos.tolist()}")

        # ARM SELECTION
        arm_side = "left" if float(obj_pos[1]) > 0.0 else "right"
        self.get_logger().info(f"[ARM_SELECT] y={float(obj_pos[1]):.3f} -> arm_side={arm_side}")

        self.hand_ctrl = HandController(self, self.robot_model, arm_side=arm_side)
        # Set a base speed limit
        self.hand_ctrl.linear_vel_limit = 0.04

        # GRIPPER CONFIG
        gripper_joint_name = "l_gripper" if arm_side == "left" else "r_gripper"
        self.qg_idx, self.vg_idx = self.gripper_indices[gripper_joint_name]

        q_lo = float(self.robot_model.model.lowerPositionLimit[self.qg_idx])
        q_hi = float(self.robot_model.model.upperPositionLimit[self.qg_idx])
        margin = 0.15 * (q_hi - q_lo)
        self.gripper_open_q = q_hi - margin
        close_fraction = float(self.get_parameter("close_fraction").value)
        self.gripper_close_q = self.gripper_open_q + close_fraction * ((q_lo + margin) - self.gripper_open_q)

        self.get_logger().info(
            f"[GRIP] {gripper_joint_name} limits [{q_lo:.2f},{q_hi:.2f}] "
            f"open={self.gripper_open_q:.3f} close={self.gripper_close_q:.3f}"
        )

        # BUILD TARGETS
        pre_x = float(self.get_parameter("pre_x_off").value)
        pre_z = float(self.get_parameter("pre_z_off").value)
        approach_x = float(self.get_parameter("approach_x_off").value)
        lift_z = float(self.get_parameter("lift_z").value)

        pre_pos = obj_pos + np.array([-pre_x, 0.0, pre_z], dtype=float)
        approach_pos = obj_pos + np.array([-approach_x, 0.0, 0.0], dtype=float)
        lift_pos = approach_pos + np.array([0.0, 0.0, lift_z], dtype=float)

        H = (self.robot_model.right_hand_pose() if arm_side == "right" else self.robot_model.left_hand_pose())
        R_hand = H[:3, :3]

        T_pre = pin.SE3(R_hand, pre_pos)
        T_approach = pin.SE3(R_hand, approach_pos)
        T_lift = pin.SE3(R_hand, lift_pos)

        self.get_logger().info(f"[PREGRASP] target={pre_pos.tolist()}")
        self.get_logger().info(f"[APPROACH] target={approach_pos.tolist()}")
        self.get_logger().info(f"[LIFT] target={lift_pos.tolist()}")

        self.hand_ctrl.set_target_pose(T_pre, duration=float(self.get_parameter("pre_duration").value), type="abs")

        phase = "PREGRASP"
        t0 = time.time()
        t_s = None

        approach_lin_limit = 0.02
        lift_lin_limit = 0.08

        dt_cmd = float(self.get_parameter("dt_cmd").value)
        phase_settle_s = float(self.get_parameter("phase_settle_s").value)

        dbg_t = time.monotonic()

        while rclpy.ok():
            t_cycle_start = time.monotonic()
            rclpy.spin_once(self, timeout_sec=0.0)
            self._maybe_refresh_from_robot()

            # --- DYNAMIC TRACKING UPDATES ---
            if phase in ("PREGRASP", "APPROACH"):
                try:
                    self.locked_T_b_obj = None  # Force fresh pose
                    T_current_obj = self._get_object_T_b()
                    current_obj_pos = T_current_obj.translation.copy()

                    # Recalculate targets based on fresh object position
                    # We keep the orientation R_hand fixed to avoid jitter
                    pre_pos_new = current_obj_pos + np.array([-pre_x, 0.0, pre_z], dtype=float)
                    approach_pos_new = current_obj_pos + np.array([-approach_x, 0.0, 0.0], dtype=float)
                    
                    T_pre = pin.SE3(R_hand, pre_pos_new)
                    T_approach = pin.SE3(R_hand, approach_pos_new)
                    
                except Exception:
                    pass

            if phase in ("PREGRASP", "APPROACH"):
                self._drive_gripper(self.gripper_open_q)

            if phase == "PREGRASP":
                # VISUAL SERVOING: Update target continuously with short duration
                self.hand_ctrl.set_target_pose(T_pre, duration=0.2, type="abs")

                v_hand = self.hand_ctrl.update(dt_cmd)
                if arm_side == "right":
                    self.robot.update(None, v_hand, dt_cmd)
                else:
                    self.robot.update(v_hand, None, dt_cmd)

                # Check completion by DISTANCE, not time
                dist_to_target = np.linalg.norm(self.hand_ctrl.x_cur.translation - T_pre.translation)
                if dist_to_target < 0.015:  # 1.5cm threshold
                    time.sleep(phase_settle_s)
                    phase = "APPROACH"
                    self.get_logger().info("Phase: APPROACH")
                    self.hand_ctrl.linear_vel_limit = approach_lin_limit
                    # Don't need to set long duration here, loop will update it
 
            elif phase == "APPROACH":
                # VISUAL SERVOING: Update target continuously with short duration
                self.hand_ctrl.set_target_pose(T_approach, duration=0.2, type="abs")

                v_hand = self.hand_ctrl.update(dt_cmd)
                if arm_side == "right":
                    self.robot.update(None, v_hand, dt_cmd)
                else:
                    self.robot.update(v_hand, None, dt_cmd)

                # Check completion by DISTANCE
                dist_to_target = np.linalg.norm(self.hand_ctrl.x_cur.translation - T_approach.translation)
                if dist_to_target < 0.01:  # 1cm threshold
                    time.sleep(phase_settle_s)
                    phase = "CLOSE"
                    self.get_logger().info("Phase: CLOSE")

            elif phase == "CLOSE":
                self._drive_gripper(self.gripper_close_q)
                self.robot.update(np.zeros(4), np.zeros(4), dt_cmd)

                if abs(float(self.robot.q[self.qg_idx]) - float(self.gripper_close_q)) < self.gripper_eps:
                    phase = "SQUEEZE"
                    t_s = time.time()
                    self.get_logger().info("Phase: SQUEEZE")

            elif phase == "SQUEEZE":
                self._drive_gripper(self.gripper_close_q)
                self.robot.update(np.zeros(4), np.zeros(4), dt_cmd)

                if t_s is not None and (time.time() - t_s) >= self.squeeze_time:
                    time.sleep(phase_settle_s)
                    phase = "LIFT"
                    self.get_logger().info(f"Phase: LIFT with target {lift_pos.tolist()}")
                    self.hand_ctrl.linear_vel_limit = lift_lin_limit
                    # Re-calculate lift target based on WHERE WE ENDED UP, not original simple offset?
                    # Or just use the original fixed Lift target derived from latest Approach?
                    # Let's use the T_lift derived from latest valid Approach (which is T_approach.translation + [0,0,lift_z])
                    # We should probably update T_lift one last time based on where we actually grasped.
                    
                    final_grasp_pos = self.hand_ctrl.x_cur.translation.copy()
                    lift_pos_final = final_grasp_pos + np.array([0.0, 0.0, lift_z], dtype=float)
                    T_lift.translation = lift_pos_final
                    
                    self.hand_ctrl.set_target_pose(T_lift, duration=float(self.get_parameter("lift_duration").value), type="abs")

            elif phase == "LIFT":
                self._drive_gripper(self.gripper_close_q)
                v_hand = self.hand_ctrl.update(dt_cmd)
                if arm_side == "right":
                    self.robot.update(None, v_hand, dt_cmd)
                else:
                    self.robot.update(v_hand, None, dt_cmd)

                if self.hand_ctrl.is_finished():
                    self.get_logger().info("Done.")
                    break

            if (time.monotonic() - dbg_t) > 1.0:
                dbg_t = time.monotonic()
                self.get_logger().info(f"[LOOP] phase={phase} dt_cmd={dt_cmd:.3f} feedback_hz={float(self.get_parameter('feedback_hz').value):.1f}")

            if time.time() - t0 > 60.0:
                self.get_logger().warn("Timeout, stopping.")
                break

            self._sleep_to_rate(t_cycle_start, dt_cmd)


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
