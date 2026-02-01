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
    q = R.from_matrix(T.rotation).as_quat()
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
        self.declare_parameter("pose_topic", "detected_object_pose")
        self.declare_parameter("lock_target_once", True)

        self.declare_parameter("hardcoded_cam_xyz", [0.056, -0.008, 0.20])
        self.declare_parameter("hardcoded_cam_rpy", [0.0, 0.0, 0.0])

        # Offsets in base frame
        self.declare_parameter("pre_x_off", 0.02)
        self.declare_parameter("pre_y_off", 0.02)          # NEW
        self.declare_parameter("pre_z_off", 0.00)

        self.declare_parameter("approach_x_off", 0.005)
        self.declare_parameter("approach_y_off", 0.0005)      # NEW
        self.declare_parameter("lift_z", 0.10)

        # Durations
        self.declare_parameter("pre_duration", 2.0)
        self.declare_parameter("approach_duration", 2.0)
        self.declare_parameter("lift_duration", 2.0)

        # Loop timing
        self.declare_parameter("dt_cmd", 0.05)

        # Gripper
        self.declare_parameter("close_fraction", 0.4)
        self.declare_parameter("gripper_kp", 6.0)
        self.declare_parameter("gripper_vel_max", 2.0)
        self.declare_parameter("gripper_eps", 0.02)
        self.declare_parameter("squeeze_time", 0.5)

        self.declare_parameter("phase_settle_s", 0.20)

        # ----------------------------
        # LOAD MODEL + ROBOT
        # ----------------------------
        pkg = get_package_share_directory("ainex_description")
        urdf_path = pkg + "/urdf/ainex.urdf"
        self.robot_model = AiNexModel(self, urdf_path)

        self.sim = bool(self.get_parameter("sim").value)
        self.dt_cmd = float(self.get_parameter("dt_cmd").value)

        self.robot = AinexRobot(self, self.robot_model, self.dt_cmd, sim=self.sim)

        # ----------------------------
        # INITIAL POSTURE
        # ----------------------------
        q_init = np.array([
            -0.3, -0.96,
            0.031416, -0.004189, -0.879646, 2.280796, 1.451416, 0.033510,
            -0.064926, -0.942419, -0.129853, -1.625251, 0.222,
            -0.031416, 0.004189, 0.879646, -2.280796, -1.451416, -0.033510,
            -0.064926, 0.942419, -0.129853, 1.625251, 0.222
        ], dtype=float)

        self.get_logger().info("Moving to initial posture...")
        self.robot.move_to_initial_position(q_init)
        time.sleep(1.0)

        # ----------------------------
        # CAMERA CHAIN (same as hands_control)
        # base -> head_tilt -> (offset) -> camera_link -> (rot) -> camera_optical
        # ----------------------------
        self.head_tilt_id = self.robot_model.model.getFrameId("head_tilt_link")
        offset = np.array([0.038068, 0.018573, 0.016398], dtype=float)
        self.T_tilt_cam = pin.SE3(np.eye(3), offset)

        R_clink_opt = np.array([
            [0.0,  0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0]
        ], dtype=float)
        self.T_clink_opt = pin.SE3(R_clink_opt, np.zeros(3))

        # ----------------------------
        # OBJECT POSE INPUT
        # ----------------------------
        self.use_camera = bool(self.get_parameter("use_camera").value)
        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.lock_target_once = bool(self.get_parameter("lock_target_once").value)

        self.latest_pose_msg = None
        self.latest_pose_t_wall = None
        self.locked_T_b_obj = None

        if self.use_camera:
            self.create_subscription(PoseStamped, self.pose_topic, self._pose_cb, 10)
            self.get_logger().info(f"Subscribed to '{self.pose_topic}'")
        else:
            self.get_logger().warn("use_camera=False: using hardcoded pose")

        # ----------------------------
        # GRIPPER JOINT INDICES
        # ----------------------------
        self.gripper_indices = {}
        for jname in ["l_gripper", "r_gripper"]:
            jid = self.robot_model.model.getJointId(jname)
            q_idx = self.robot_model.model.joints[jid].idx_q
            v_idx = self.robot_model.model.joints[jid].idx_v
            self.gripper_indices[jname] = (q_idx, v_idx)

        self.gripper_kp = float(self.get_parameter("gripper_kp").value)
        self.gripper_vel_max = float(self.get_parameter("gripper_vel_max").value)
        self.gripper_eps = float(self.get_parameter("gripper_eps").value)
        self.squeeze_time = float(self.get_parameter("squeeze_time").value)

        self.qg_idx = None
        self.vg_idx = None
        self.gripper_open_q = None
        self.gripper_close_q = None

        self.hand_ctrl = None

        self.run()

    def _pose_cb(self, msg: PoseStamped):
        self.latest_pose_msg = msg
        self.latest_pose_t_wall = time.time()

    def _get_T_b_opt(self) -> pin.SE3:
        T_b_head = self.robot_model.data.oMf[self.head_tilt_id]
        T_b_clink = T_b_head * self.T_tilt_cam
        return T_b_clink * self.T_clink_opt

    def _get_object_T_b(self) -> pin.SE3:
        if self.lock_target_once and self.locked_T_b_obj is not None:
            return self.locked_T_b_obj

        if self.use_camera:
            if self.latest_pose_msg is None:
                raise RuntimeError("No object pose received yet.")
            if self.latest_pose_t_wall is None or (time.time() - self.latest_pose_t_wall) > 1.0:
                raise RuntimeError("Object pose is stale (>1s).")

            fid = self.latest_pose_msg.header.frame_id.strip() if self.latest_pose_msg.header.frame_id else ""
            if fid and fid != "camera_optical_link":
                self.get_logger().warn(f"Pose frame_id='{fid}' (expected 'camera_optical_link')")

            T_opt_obj = pose_to_se3_from_pose_msg(self.latest_pose_msg.pose)
        else:
            xyz = self.get_parameter("hardcoded_cam_xyz").value
            rpy = self.get_parameter("hardcoded_cam_rpy").value
            t = np.array([xyz[0], xyz[1], xyz[2]], dtype=float)
            Rot = R.from_euler("xyz", [rpy[0], rpy[1], rpy[2]]).as_matrix()
            T_opt_obj = pin.SE3(Rot, t)

        T_b_opt = self._get_T_b_opt()
        T_b_obj = T_b_opt * T_opt_obj

        if self.lock_target_once:
            self.locked_T_b_obj = T_b_obj

        # Debug TF
        try:
            send_tf(self.br, self, "base_link", "camera_optical_used", T_b_opt)
            send_tf(self.br, self, "base_link", "object_in_base", T_b_obj)
        except Exception:
            pass

        return T_b_obj

    def _sleep_to_rate(self, t_cycle_start: float):
        rem = self.dt_cmd - (time.monotonic() - t_cycle_start)
        if rem > 0.0:
            time.sleep(rem)

    def _drive_active_gripper(self, q_target: float):
        """Command ONLY the selected arm gripper (physical), using velocity on that joint."""
        qg = float(self.robot.q[self.qg_idx])
        vg = self.gripper_kp * (q_target - qg)
        vg = float(np.clip(vg, -self.gripper_vel_max, self.gripper_vel_max))
        if abs(q_target - qg) < self.gripper_eps:
            vg = 0.0
        self.robot.v[self.vg_idx] = vg

    def run(self):
        phase_settle_s = float(self.get_parameter("phase_settle_s").value)

        # ---------------------------------------------------------
        # 1) Robust target estimate (median of N samples)
        # ---------------------------------------------------------
        self.get_logger().info("Collecting samples for robust target...")
        samples = []
        last_wall = None
        t_start = time.time()

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)

            if time.time() - t_start > 5.0:
                break
            if len(samples) >= 20:
                break

            try:
                if self.use_camera:
                    if self.latest_pose_t_wall is None:
                        continue
                    if last_wall is not None and self.latest_pose_t_wall == last_wall:
                        continue
                    if (time.time() - self.latest_pose_t_wall) > 1.0:
                        continue
                    last_wall = self.latest_pose_t_wall

                self.locked_T_b_obj = None
                T_sample = self._get_object_T_b()
                samples.append(T_sample.translation.copy())
            except Exception:
                continue

        if len(samples) == 0:
            obj_pos = self._get_object_T_b().translation.copy()
        else:
            data = np.array(samples)
            obj_pos = np.median(data, axis=0)
            self.get_logger().info(f"Target (median of {len(samples)}): {obj_pos}")
            self.locked_T_b_obj = pin.SE3(np.eye(3), obj_pos.copy())
            self.lock_target_once = True

        # ---------------------------------------------------------
        # 2) Select arm
        # ---------------------------------------------------------
        arm_side = "left" if float(obj_pos[1]) > 0.0 else "right"
        self.active_side = arm_side
        self.get_logger().info(f"Selected arm: {arm_side} (y={float(obj_pos[1]):.3f})")

        # ---------------------------------------------------------
        # 3) Setup controller
        # ---------------------------------------------------------
        Kp = np.array([5.0, 5.0, 5.0])
        Kd = np.array([2.0, 2.0, 2.0])

        self.hand_ctrl = HandController(self, self.robot_model, arm_side=arm_side, Kp=Kp, Kd=Kd)

        self.hand_ctrl.linear_vel_limit = 0.25
        self.hand_ctrl.joint_vel_limit = np.array([3.0, 3.0, 3.0, 3.0])
        self.hand_ctrl.dls_lambda = 0.03
        self.hand_ctrl.u_alpha = 0.25

        # ---------------------------------------------------------
        # 4) Setup active gripper indices + open/close values
        # ---------------------------------------------------------
        gripper_joint = "l_gripper" if arm_side == "left" else "r_gripper"
        self.qg_idx, self.vg_idx = self.gripper_indices[gripper_joint]

        q_lo = float(self.robot_model.model.lowerPositionLimit[self.qg_idx])
        q_hi = float(self.robot_model.model.upperPositionLimit[self.qg_idx])
        margin = 0.15 * (q_hi - q_lo)
        close_frac = float(self.get_parameter("close_fraction").value)

        if arm_side == "left":
            self.gripper_open_q = q_lo + margin
            self.gripper_close_q = self.gripper_open_q + close_frac * ((q_hi - margin) - self.gripper_open_q)
        else:
            self.gripper_open_q = q_hi - margin
            self.gripper_close_q = self.gripper_open_q + close_frac * ((q_lo + margin) - self.gripper_open_q)

        self.get_logger().info(
            f"Active gripper={gripper_joint} open={self.gripper_open_q:.3f} close={self.gripper_close_q:.3f}"
        )

        # ---------------------------------------------------------
        # 5) Build targets (use current hand orientation)
        # ---------------------------------------------------------
        pre_x = float(self.get_parameter("pre_x_off").value)
        pre_y = float(self.get_parameter("pre_y_off").value)               # NEW
        pre_z = float(self.get_parameter("pre_z_off").value)

        # ----------------------------
        # Safety shaping for LEFT arm near center
        # ----------------------------
        
        approach_x = float(self.get_parameter("approach_x_off").value)
        approach_y = float(self.get_parameter("approach_y_off").value)     # NEW

        # obj_y = float(obj_pos[1])
        # obj_z = float(obj_pos[2])
        # if arm_side == "left":
        #     # If object is too close to centerline, force a more "outside" lateral path
        #     Y_MIN_LEFT = 0.05     # 5 cm outward
        #     if obj_y < Y_MIN_LEFT:
        #         # Ensure PRE and APPROACH are not near the center
        #         pre_y = max(pre_y, Y_MIN_LEFT - obj_y + 0.02)         # add extra margin
        #         approach_y = max(approach_y, Y_MIN_LEFT - obj_y)      # keep approach outside

        #     # Prevent very low approach that tends to hit the knee region
        #     Z_MIN_APPROACH = -0.01  # don't go too low (tune)
        #     # raise PRE more than APPROACH
        #     pre_z = max(pre_z, (Z_MIN_APPROACH - obj_z) + 0.03)

        

        lift_z = float(self.get_parameter("lift_z").value)

        # If you prefer "magnitudes" only, uncomment this to auto-apply side sign:
        # side_sign = 1.0 if arm_side == "left" else -1.0
        # pre_y = side_sign * abs(pre_y)
        # approach_y = side_sign * abs(approach_y)

        pre_pos = obj_pos + np.array([-pre_x, pre_y, pre_z], dtype=float)                 # UPDATED
        approach_pos = obj_pos + np.array([-approach_x , approach_y, 0.0], dtype=float)    # UPDATED
        lift_pos = approach_pos + np.array([0.0, 0.0, lift_z], dtype=float)

        H_current = self.robot_model.right_hand_pose() if arm_side == "right" else self.robot_model.left_hand_pose()
        R_hand = H_current[:3, :3]

        T_pre = pin.SE3(R_hand, pre_pos)
        T_app = pin.SE3(R_hand, approach_pos)
        T_lift = pin.SE3(R_hand, lift_pos)

        self.get_logger().info(f"PRE: {pre_pos.tolist()}")
        self.get_logger().info(f"APP: {approach_pos.tolist()}")
        self.get_logger().info(f"LIFT:{lift_pos.tolist()}")

        # Phase limits
        pre_lin_limit = 0.20
        app_lin_limit = 0.03
        lift_lin_limit = 0.20

        pre_duration = float(self.get_parameter("pre_duration").value)
        app_duration = float(self.get_parameter("approach_duration").value)
        lift_duration = float(self.get_parameter("lift_duration").value)

        # thresholds
        pre_th = 0.0050
        app_th = 0.0001
        lift_th = 0.015

        # ---------------------------------------------------------
        # 6) State machine
        # ---------------------------------------------------------
        phase = "PREGRASP"
        self.hand_ctrl.linear_vel_limit = pre_lin_limit
        self.hand_ctrl.set_target_pose(T_pre, duration=pre_duration, type="abs")
        t_squeeze_start = None

        self.get_logger().info("Starting state machine...")

        while rclpy.ok():
            t_cycle_start = time.monotonic()

            rclpy.spin_once(self, timeout_sec=0.0)

            v_hand = self.hand_ctrl.update(self.dt_cmd)

            # Only active gripper is commanded
            if phase in ("PREGRASP", "APPROACH"):
                self._drive_active_gripper(self.gripper_open_q)
            elif phase in ("CLOSE", "SQUEEZE", "LIFT"):
                self._drive_active_gripper(self.gripper_close_q)

            # Send to robot
            if arm_side == "right":
                self.robot.update(None, v_hand, self.dt_cmd)
            else:
                self.robot.update(v_hand, None, self.dt_cmd)

            # Phase transitions
            if phase == "PREGRASP":
                dist = float(np.linalg.norm(self.hand_ctrl.x_cur.translation - T_pre.translation))
                if dist < pre_th or self.hand_ctrl.is_finished():
                    time.sleep(phase_settle_s)
                    phase = "APPROACH"
                    self.get_logger().info("Phase: APPROACH")

                    # Slower + softer for final approach
                    self.hand_ctrl.linear_vel_limit = app_lin_limit
                    self.hand_ctrl.Kp = np.array([3.0, 3.0, 3.0])   # softer stiffness
                    self.hand_ctrl.Kd = np.array([2.5, 2.5, 2.5])   # a bit more damping
                    self.hand_ctrl.u_alpha = 0.18                   # more smoothing in approach

                    self.hand_ctrl.set_target_pose(T_app, duration=app_duration, type="abs")


            elif phase == "APPROACH":
                dist = float(np.linalg.norm(self.hand_ctrl.x_cur.translation - T_app.translation))
                if dist < app_th or self.hand_ctrl.is_finished():
                    time.sleep(phase_settle_s)
                    phase = "CLOSE"
                    self.get_logger().info("Phase: CLOSE")

            elif phase == "CLOSE":
                # Hold arm still while closing
                if arm_side == "right":
                    self.robot.update(None, np.zeros(4), self.dt_cmd)
                else:
                    self.robot.update(np.zeros(4), None, self.dt_cmd)

                if abs(float(self.robot.q[self.qg_idx]) - float(self.gripper_close_q)) < self.gripper_eps:
                    phase = "SQUEEZE"
                    t_squeeze_start = time.monotonic()
                    self.get_logger().info("Phase: SQUEEZE")

            elif phase == "SQUEEZE":
                if arm_side == "right":
                    self.robot.update(None, np.zeros(4), self.dt_cmd)
                else:
                    self.robot.update(np.zeros(4), None, self.dt_cmd)

                if t_squeeze_start is not None and (time.monotonic() - t_squeeze_start) >= self.squeeze_time:
                    time.sleep(phase_settle_s)
                    phase = "LIFT"
                    self.get_logger().info("Phase: LIFT")
                    self.hand_ctrl.linear_vel_limit = lift_lin_limit
                    grasp_pos = self.hand_ctrl.x_cur.translation.copy()
                    T_lift.translation = grasp_pos + np.array([0.0, 0.0, lift_z], dtype=float)
                    self.hand_ctrl.set_target_pose(T_lift, duration=lift_duration, type="abs")

            elif phase == "LIFT":
                dist = float(np.linalg.norm(self.hand_ctrl.x_cur.translation - T_lift.translation))
                if dist < lift_th or self.hand_ctrl.is_finished():
                    time.sleep(phase_settle_s)
                    self.get_logger().info("DONE")
                    break

            self._sleep_to_rate(t_cycle_start)


def main():
    rclpy.init()
    node = None
    try:
        node = AinexGraspNode()
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
