#!/usr/bin/env python3
import time
import numpy as np
import pinocchio as pin

from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation

from ainex_controller.ainex_model import AiNexModel
from ainex_controller.spline_trajectory import LinearSplineTrajectory


class HandController:
    """
    Cartesian POSITION-only controller:
      - spline generates desired position x_des and feedforward v_des
      - PD law produces desired linear velocity xdot_des
      - map xdot_des -> joint velocities via Damped Least Squares (DLS)
    """

    def __init__(self, node: Node, model: AiNexModel, arm_side: str,
                 Kp: np.ndarray = None,
                 Kd: np.ndarray = None):
        self.node = node
        self.br = TransformBroadcaster(node)

        self.robot_model = model
        self.arm_side = arm_side

        self.x_cur = pin.SE3.Identity()
        self.x_des = pin.SE3.Identity()
        self.x_init = pin.SE3.Identity()
        self.x_target = pin.SE3.Identity()

        self.v_cur = pin.Motion.Zero()

        self.spline = None
        self.spline_duration = 0.0
        self.start_time = None  # monotonic time

        # Manipulability threshold (tuned for AiNex)
        self.w_threshold = 1e-5

        # Limits
        self.joint_vel_limit = np.array([2.0, 2.0, 2.0, 2.0], dtype=float)  # rad/s
        self.linear_vel_limit = 0.10  # m/s

        # Gains
        self.Kp = np.array([5.0, 5.0, 5.0], dtype=float) if Kp is None else Kp.astype(float)
        self.Kd = np.array([2.0, 2.0, 2.0], dtype=float) if Kd is None else Kd.astype(float)

        # DLS damping (prevents blow-up near singularities)
        self.dls_lambda = 0.003  # try 0.02–0.06

        # Smooth command (low-pass joint velocity command to remove micro-shake)
        self.u_prev = np.zeros(4, dtype=float)
        self.u_alpha = 0.25  # 0.15–0.35 (higher = less smoothing)

        # Throttled warnings (avoid jitter from log spam)
        self._last_warn_t = 0.0
        self._warn_period = 1.0  # seconds

        # ------------------------------------------------------------
        # NEW: Nullspace quadratic potential for singularity avoidance
        # Default OFF (as you requested)
        # ------------------------------------------------------------
        self.enable_nullspace_potential = False  # default false
        self.ks_null = 0.0                       # gain for potential (set > 0 to activate if enabled)
        self.mth_null = 5.0e-5                   # threshold where avoidance activates (tune)
        self.null_eps = 1.0e-3                   # finite-diff step for grad(m) (rad)
        self.null_max = 0.6                      # cap for ||u0|| (rad/s), optional safety
        # ------------------------------------------------------------

    def _now(self) -> float:
        return time.monotonic()

    def _warn_throttle(self, msg: str):
        t = self._now()
        if (t - self._last_warn_t) >= self._warn_period:
            self.node.get_logger().warn(msg)
            self._last_warn_t = t

    # ------------------------------------------------------------
    # NEW: helper to compute manipulability m = sqrt(det(JJ^T))
    # ------------------------------------------------------------
    @staticmethod
    def _manipulability_from_Jpos(J_pos: np.ndarray) -> float:
        JJT = J_pos @ J_pos.T
        det_val = float(np.linalg.det(JJT))
        return float(np.sqrt(max(det_val, 0.0)))
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # NEW: numerical gradient of manipulability wrt the 4 arm joints
    # IMPORTANT: we do NOT need a reference pose for the potential.
    # The potential is in terms of m(q) only; the gradient is computed
    # around the current q (finite differences).
    #
    # Assumption (kept local): robot_model.update_model(q, v) exists and
    # updates J_left/J_right for the current full q.
    # ------------------------------------------------------------
    def _grad_manipulability_numeric(self, q_full: np.ndarray, arm_ids: np.ndarray) -> np.ndarray:
        eps = float(self.null_eps)

        # Save current model state so we can restore after finite-diff updates
        q_save = q_full.copy()
        v_save = getattr(self.robot_model, "v", None)
        if v_save is None:
            # fallback if AiNexModel stores current velocity elsewhere
            v_save = np.zeros(self.robot_model.model.nv, dtype=float)
        else:
            v_save = np.array(v_save, dtype=float).copy()

        grad = np.zeros(len(arm_ids), dtype=float)

        # Evaluate m at current q (using current J_full already computed in update)
        # We'll recompute anyway for safety.
        self.robot_model.update_model(q_save, v_save)
        if self.arm_side == "left":
            J_full = self.robot_model.J_left
        else:
            J_full = self.robot_model.J_right
        J_pos = J_full[:3, arm_ids]
        m0 = self._manipulability_from_Jpos(J_pos)

        for i, idx in enumerate(arm_ids):
            q_p = q_save.copy()
            q_m = q_save.copy()
            q_p[idx] += eps
            q_m[idx] -= eps

            self.robot_model.update_model(q_p, v_save)
            J_full_p = self.robot_model.J_left if self.arm_side == "left" else self.robot_model.J_right
            mp = self._manipulability_from_Jpos(J_full_p[:3, arm_ids])

            self.robot_model.update_model(q_m, v_save)
            J_full_m = self.robot_model.J_left if self.arm_side == "left" else self.robot_model.J_right
            mm = self._manipulability_from_Jpos(J_full_m[:3, arm_ids])

            grad[i] = (mp - mm) / (2.0 * eps)

        # Restore state
        self.robot_model.update_model(q_save, v_save)

        # If something went numerically weird, fall back to zeros
        if not np.all(np.isfinite(grad)):
            return np.zeros_like(grad)

        return grad
    # ------------------------------------------------------------

    def set_target_pose(self, pose: pin.SE3, duration: float, type: str = 'abs'):
        # Get current end-effector pose from model
        if self.arm_side == 'left':
            current_matrix = self.robot_model.left_hand_pose()
        else:
            current_matrix = self.robot_model.right_hand_pose()

        self.x_cur = pin.SE3(current_matrix[:3, :3], current_matrix[:3, 3])
        self.x_init = self.x_cur.copy()

        # Target
        if type == 'abs':
            self.x_target = pose.copy()
        elif type == 'rel':
            self.x_target = self.x_cur * pose
        else:
            raise ValueError("type must be 'abs' or 'rel'")

        # Linear spline for translation
        self.spline = LinearSplineTrajectory(
            x_init=self.x_init.translation,
            x_final=self.x_target.translation,
            duration=float(duration)
        )
        self.spline_duration = float(duration)
        self.start_time = self._now()

        # Reset filter each new motion
        self.u_prev[:] = 0.0

    def update(self, dt: float) -> np.ndarray:
        """Update the arm controller with new joint states.
        
        Args:
            joint_states (np.ndarray): Current joint positions.
            dt (float): Time step for the update.

        Returns:
            np.ndarray: Desired joint velocities for the arm.(4,)
        """
        # --- Broadcast TF of target pose (debug/visualization)
        t = TransformStamped()
        t.header.stamp = self.node.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = f'{self.arm_side}_hand_target'
        t.transform.translation.x = float(self.x_target.translation[0])
        t.transform.translation.y = float(self.x_target.translation[1])
        t.transform.translation.z = float(self.x_target.translation[2])
        r = Rotation.from_matrix(self.x_target.rotation)
        q = r.as_quat()  # x,y,z,w
        t.transform.rotation.x = float(q[0])
        t.transform.rotation.y = float(q[1])
        t.transform.rotation.z = float(q[2])
        t.transform.rotation.w = float(q[3])
        self.br.sendTransform(t)

        # --- Current pose + velocity
        if self.arm_side == 'left':
            current_matrix = self.robot_model.left_hand_pose()
            self.v_cur = self.robot_model.left_hand_velocity()
            J_full = self.robot_model.J_left
        else:
            current_matrix = self.robot_model.right_hand_pose()
            self.v_cur = self.robot_model.right_hand_velocity()
            J_full = self.robot_model.J_right

        self.x_cur = pin.SE3(current_matrix[:3, :3], current_matrix[:3, 3])

        # --- Desired pose from spline
        if self.start_time is not None and self.spline is not None:
            t_elapsed = self._now() - self.start_time
            t_elapsed = min(t_elapsed, self.spline_duration)
            x_des_pos, v_des_pos = self.spline.update(t_elapsed)
            self.x_des = pin.SE3(self.x_target.rotation, x_des_pos)
        else:
            self.x_des = self.x_cur.copy()
            v_des_pos = np.zeros(3, dtype=float)

        # --- Cartesian PD in position only
        pos_error = self.x_des.translation - self.x_cur.translation
        vel_error = v_des_pos - self.v_cur[:3]  # linear component
        xdot_des = self.Kp * pos_error + self.Kd * vel_error

        # --- Clamp Cartesian linear speed
        n = float(np.linalg.norm(xdot_des))
        if n > float(self.linear_vel_limit) and n > 1e-12:
            xdot_des *= (float(self.linear_vel_limit) / n)
            # throttle only if needed during tuning
            # self._warn_throttle(f"{self.arm_side} EE vel limited to {self.linear_vel_limit:.2f} m/s")

        # --- Jacobian (linear rows, arm columns)
        arm_ids = self.robot_model.get_arm_ids(self.arm_side)
        J_pos = J_full[:3, arm_ids]  # (3,4)

        # --- Damped Least Squares: u = J^T (J J^T + λ^2 I)^-1 xdot
        lam = float(self.dls_lambda)
        JJT = J_pos @ J_pos.T
        A = JJT + (lam * lam) * np.eye(3, dtype=float)

        try:
            y = np.linalg.solve(A, xdot_des)
            u = (J_pos.T @ y).astype(float)  # (4,)
        except np.linalg.LinAlgError:
            u = np.zeros(4, dtype=float)

        # ------------------------------------------------------------
        # NEW: Nullspace quadratic potential for singularity avoidance
        # u <- u + N * u0, where u0 = ks*(mth - m)*grad(m) if m < mth
        # ------------------------------------------------------------
        if bool(self.enable_nullspace_potential) and float(self.ks_null) > 0.0:
            m = self._manipulability_from_Jpos(J_pos)

            mth = float(self.mth_null)
            if m < mth:
                # gradient of manipulability wrt the 4 arm joints
                # we need the FULL q from model (already maintained by robot_model)
                q_full = getattr(self.robot_model, "q", None)
                if q_full is None:
                    # if AiNexModel doesn't expose q, we cannot do FD safely -> skip
                    grad_m = np.zeros(4, dtype=float)
                else:
                    grad_m = self._grad_manipulability_numeric(np.array(q_full, dtype=float), arm_ids)

                # Quadratic potential: U = 0.5 * ks * (mth - m)^2
                # -> descent direction in joint space: u0 = +ks*(mth - m)*grad(m)
                u0 = float(self.ks_null) * (mth - m) * grad_m

                # Cap nullspace term
                u0_norm = float(np.linalg.norm(u0))
                if u0_norm > float(self.null_max) and u0_norm > 1e-12:
                    u0 *= float(self.null_max) / u0_norm

                # Nullspace projector in velocity form:
                # N = I - J# J with SAME DLS inverse used above
                # J# = J^T (J J^T + λ^2 I)^-1
                J_pinv_dls = J_pos.T @ np.linalg.inv(A)
                N = np.eye(len(arm_ids), dtype=float) - (J_pinv_dls @ J_pos)

                u = u + (N @ u0)
        # ------------------------------------------------------------

        # --- Joint velocity clamp (no spam)
        for i in range(len(u)):
            lim = float(self.joint_vel_limit[i])
            if abs(u[i]) > lim:
                u[i] = np.sign(u[i]) * lim
                # self._warn_throttle(f"{self.arm_side} joint {i} vel limited to {lim:.2f} rad/s")

        # --- Smooth singularity handling (NO hard stop)
        det_val = float(np.linalg.det(JJT))
        w = float(np.sqrt(max(det_val, 0.0)))

        # Scale down commands when approaching singularity
        w0 = float(self.w_threshold)
        w_safe = 5.0 * w0  # scaling zone
        if w < w_safe:
            if w <= w0:
                scale = 0.0
            else:
                scale = (w - w0) / (w_safe - w0)
                scale = float(np.clip(scale, 0.0, 1.0))
            u *= scale

        # --- Low-pass filter on joint velocity command
        a = float(self.u_alpha)
        u = (1.0 - a) * self.u_prev + a * u
        self.u_prev = u.copy()

        return u

    def is_finished(self) -> bool:
        """Check if the trajectory has finished executing.
        
        Returns:
            bool: True if trajectory is complete, False otherwise.
        """
        if self.start_time is None or self.spline is None:
            return True
        return (self._now() - self.start_time) >= self.spline_duration
