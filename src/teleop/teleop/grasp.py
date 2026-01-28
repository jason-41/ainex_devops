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
        self.declare_parameter("sim", False)  # False on real robot, True for local sim testing
        self.declare_parameter("use_camera", True)
        self.declare_parameter("pose_topic", "/cube_pose")
        self.declare_parameter("lock_target_once", True)
        self.declare_parameter("camera_frame_fallback", "camera_optical_link")

        # Hardcoded pose in CAMERA frame when use_camera:=false
        self.declare_parameter("hardcoded_cam_xyz", [0.0, 0.0, 0.05])
        self.declare_parameter("hardcoded_cam_rpy", [0.0, 0.0, 0.0])

        # Grasp offsets (position-only), in BASE frame after transform
        self.declare_parameter("pre_x_off", 0.02)
        self.declare_parameter("pre_z_off", 0.015)
        self.declare_parameter("approach_x_off", 0.01)
        self.declare_parameter("lift_z", 0.10)

        # Speeds / durations
        self.declare_parameter("pre_duration", 3.0)
        self.declare_parameter("approach_duration", 3.0)
        self.declare_parameter("lift_duration", 2.0)

        # Control timing (IMPORTANT on real robot)
        self.declare_parameter("dt_min", 0.01)   # 100 Hz clamp
        self.declare_parameter("dt_max", 0.05)   # 20 Hz clamp
        self.declare_parameter("feedback_hz", 1.0)  # throttle robot joint read (0 to disable)

        # Gripper behavior (velocity command)
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

        self.sim = bool(self.get_parameter("sim").value)

        # AinexRobot still needs a nominal dt for internal init; we will pass real dt each update()
        nominal_dt = float(self.get_parameter("dt_max").value)
        self.robot = AinexRobot(self, self.robot_model, nominal_dt, sim=self.sim)

        # Home posture
        q_init = np.zeros(self.robot_model.model.nq)
        q_init[self.robot_model.get_joint_id("r_sho_roll")] = 1.4
        q_init[self.robot_model.get_joint_id("l_sho_roll")] = -1.4
        q_init[self.robot_model.get_joint_id("r_el_yaw")] = 1.58
        q_init[self.robot_model.get_joint_id("l_el_yaw")] = -1.58
        self.robot.move_to_initial_position(q_init)
        time.sleep(1.0)

        self.hand_ctrl = None

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
            self.get_logger().warn(
                "Could not find 'camera_optical_link' or 'camera_link' in URDF. "
                "Will still run, but camera->base transform will fail unless msg.header.frame_id matches a frame."
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
            self.get_logger().info(f"Listening for object pose on {self.pose_topic} (PoseStamped, camera frame).")
        else:
            self.get_logger().warn("use_camera:=false -> using hardcoded pose in camera frame.")

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

        # Timing helpers (wall time to avoid sim_time pitfalls + service stalls)
        self._t_last = time.monotonic()
        self._t_last_feedback = 0.0

        # ----------------------------
        # RUN STATE MACHINE (blocking)
        # ----------------------------
        self.run()

    def _pose_cb(self, msg: PoseStamped):
        self.latest_pose_msg = msg
        self.latest_pose_t = time.time()

    def _get_T_b_c(self, frame_name_hint: str = None) -> pin.SE3:
        if frame_name_hint:
            fid = self.robot_model.model.getFrameId(frame_name_hint)
            if fid < self.robot_model.model.nframes:
                return self.robot_model.data.oMf[fid]
        if self.cam_frame_id is None:
            raise RuntimeError("No camera frame in model and frame_name_hint not found.")
        return self.robot_model.data.oMf[self.cam_frame_id]

    def _get_object_T_b(self) -> pin.SE3:
        if self.lock_target_once and self.locked_T_b_obj is not None:
            return self.locked_T_b_obj

        if self.use_camera:
            if self.latest_pose_msg is None:
                raise RuntimeError("No pose received yet.")
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

        T_b_c = self._get_T_b_c(cam_frame_hint)
        T_b_obj = T_b_c * T_c_obj

        if self.lock_target_once:
            self.locked_T_b_obj = T_b_obj

        try:
            send_tf(self.br, self, "base_link", "camera_frame_used", pin.SE3(T_b_c.rotation, T_b_c.translation))
            send_tf(self.br, self, "base_link", "object_in_base", pin.SE3(T_b_obj.rotation, T_b_obj.translation))
        except Exception:
            pass

        return T_b_obj

    def _compute_dt(self) -> float:
        dt_min = float(self.get_parameter("dt_min").value)
        dt_max = float(self.get_parameter("dt_max").value)

        now = time.monotonic()
        dt = now - self._t_last
        self._t_last = now
        dt = max(dt_min, min(dt, dt_max))
        return dt

    def _maybe_refresh_from_robot(self):
        """
        Throttled joint feedback read (service call). This can block, so do it rarely.
        Set feedback_hz:=0 to disable.
        """
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

    def run(self):
        # ----------------------------
        # WAIT FOR TARGET
        # ----------------------------
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

        # ----------------------------
        # ARM SELECTION
        # ----------------------------
        arm_side = "left" if float(obj_pos[1]) > 0.0 else "right"
        self.get_logger().info(f"[ARM_SELECT] y={float(obj_pos[1]):.3f} -> arm_side={arm_side}")

        self.hand_ctrl = HandController(self, self.robot_model, arm_side=arm_side)
        #self.hand_ctrl.set_yaw_control(False)

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

        H = (self.robot_model.right_hand_pose() if arm_side == "right" else self.robot_model.left_hand_pose())
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
        #self.hand_ctrl.set_yaw_control(False)
        self.hand_ctrl.set_target_pose(
            T_pre,
            duration=float(self.get_parameter("pre_duration").value),
            type="abs",
        )

        phase = "PREGRASP"
        t0 = time.time()
        t_s = None

        # (optional) approach speed settings
        approach_lin_limit = 0.02
        lift_lin_limit = 0.08

        # loop debug throttling
        dbg_t = time.monotonic()

        while rclpy.ok():
            dt = self._compute_dt()
            rclpy.spin_once(self, timeout_sec=0.0)  # do not tie loop to dt; we already have dt from wall time
            self._maybe_refresh_from_robot()

            if phase in ("PREGRASP", "APPROACH"):
                self._drive_gripper(self.gripper_open_q)

            if phase == "PREGRASP":
                v_hand = self.hand_ctrl.update(dt)
                if arm_side == "right":
                    self.robot.update(None, v_hand, dt)
                else:
                    self.robot.update(v_hand, None, dt)

                if self.hand_ctrl.is_finished():
                    phase = "APPROACH"
                    self.get_logger().info("Phase: APPROACH")
                    self.hand_ctrl.linear_vel_limit = approach_lin_limit
                    self.hand_ctrl.set_target_pose(
                        T_approach,
                        duration=float(self.get_parameter("approach_duration").value),
                        type="abs",
                    )

            elif phase == "APPROACH":
                v_hand = self.hand_ctrl.update(dt)
                if arm_side == "right":
                    self.robot.update(None, v_hand, dt)
                else:
                    self.robot.update(v_hand, None, dt)

                if self.hand_ctrl.is_finished():
                    phase = "CLOSE"
                    self.get_logger().info("Phase: CLOSE")

            elif phase == "CLOSE":
                # Freeze arms; drive gripper closed
                self._drive_gripper(self.gripper_close_q)
                self.robot.update(np.zeros(4), np.zeros(4), dt)

                if abs(float(self.robot.q[self.qg_idx]) - float(self.gripper_close_q)) < self.gripper_eps:
                    phase = "SQUEEZE"
                    t_s = time.time()
                    self.get_logger().info("Phase: SQUEEZE")

            elif phase == "SQUEEZE":
                self._drive_gripper(self.gripper_close_q)
                self.robot.update(np.zeros(4), np.zeros(4), dt)

                if t_s is not None and (time.time() - t_s) >= self.squeeze_time:
                    phase = "LIFT"
                    self.get_logger().info("Phase: LIFT")
                    self.hand_ctrl.linear_vel_limit = lift_lin_limit
                    self.hand_ctrl.set_target_pose(
                        T_lift,
                        duration=float(self.get_parameter("lift_duration").value),
                        type="abs",
                    )

            elif phase == "LIFT":
                self._drive_gripper(self.gripper_close_q)
                v_hand = self.hand_ctrl.update(dt)
                if arm_side == "right":
                    self.robot.update(None, v_hand, dt)
                else:
                    self.robot.update(v_hand, None, dt)

                if self.hand_ctrl.is_finished():
                    self.get_logger().info("Done.")
                    break

            # Debug once per second: confirms dt isn't tiny and loop is alive
            if (time.monotonic() - dbg_t) > 1.0:
                dbg_t = time.monotonic()
                self.get_logger().info(f"[LOOP] phase={phase} dt={dt:.3f} sim={self.sim}")

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
