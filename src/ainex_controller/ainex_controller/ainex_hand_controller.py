import pinocchio as pin
from ainex_controller.ainex_model import AiNexModel
from ainex_controller.spline_trajectory import LinearSplineTrajectory
import numpy as np  
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation
import time

class HandController():
    def __init__(self, node: Node, model: AiNexModel, arm_side: str,
                 Kp: np.ndarray = None,
                 Kd: np.ndarray = None):
        """Initialize the Ainex Arm Controller.
        
        Args:
            model: Pinocchio robot model.
            arm_side: 'left' or 'right' arm.
        """
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
        self.start_time = None

        self.w_threshold = 5e-4  # manipulability threshold
        
        # Add velocity limits
        self.joint_vel_limit = np.array([2.0, 2.0, 2.0, 2.0])  # Joint velocity limits (rad/s)
        self.linear_vel_limit = 0.1  # End-effector linear velocity limit (m/s)
        
        if Kp is not None:
            self.Kp = Kp
        else:
            self.Kp = np.array([5.0, 5.0, 5.0])
        
        if Kd is not None:
            self.Kd = Kd
        else:
            self.Kd = np.array([0.5, 0.5, 0.5])


    def set_target_pose(self, pose: pin.SE3, duration: float, type: str = 'abs'):
        """Set the desired target pose for the specified arm.
        
        Args:
            pose (pin.SE3): Desired end-effector pose.
            type (str): 'abs' or 'rel' pose setting.
        """

        # TODO: get current pose from the robot model
        if self.arm_side == 'left':
            current_matrix = self.robot_model.left_hand_pose()
        else:
            current_matrix = self.robot_model.right_hand_pose()
        
        self.x_cur = pin.SE3(current_matrix[:3, :3], current_matrix[:3, 3])
        self.x_init = self.x_cur.copy()

        # TODO: set target pose based on type 
        # abs: absolute pose w.r.t. base_link
        # rel: relative pose w.r.t. current pose

        if type == 'abs':
            self.x_target = pose.copy()
        elif type == 'rel':
            self.x_target = self.x_cur * pose  # SE3 composition for relative transform

        
        # TODO: Plan a spline trajectory from current to target pose using the
        # class LinearSplineTrajectory defined in spline_trajectory.py
        self.spline = LinearSplineTrajectory(
            x_init=self.x_init.translation,
            x_final=self.x_target.translation,
            duration=duration
        )
        self.spline_duration = duration
        # TODO: save start time
        self.start_time = time.time()

    def update(self, dt: float) -> np.ndarray:
        """Update the arm controller with new joint states.
        
        Args:
            joint_states (np.ndarray): Current joint positions.
            dt (float): Time step for the update.

        Returns:
            np.ndarray: Desired joint velocities for the arm.(4,)
        """

        # TODO: Broadcast target pose as TF for visualization
        t = TransformStamped()
        t.header.stamp = self.node.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = f'{self.arm_side}_hand_target'
        
        # Extract translation
        t.transform.translation.x = float(self.x_target.translation[0])
        t.transform.translation.y = float(self.x_target.translation[1])
        t.transform.translation.z = float(self.x_target.translation[2])
        
        # Extract rotation and convert to quaternion
        r = Rotation.from_matrix(self.x_target.rotation)
        quat = r.as_quat()  # [x, y, z, w]
        t.transform.rotation.x = float(quat[0])
        t.transform.rotation.y = float(quat[1])
        t.transform.rotation.z = float(quat[2])
        t.transform.rotation.w = float(quat[3])
        
        self.br.sendTransform(t)

        # TODO: get current pose and velocity from the robot model
        # We assume the model is already updated with the latest joint states
        if self.arm_side == 'left':
            current_matrix = self.robot_model.left_hand_pose()
            self.v_cur = self.robot_model.left_hand_velocity()
        else:
            current_matrix = self.robot_model.right_hand_pose()
            self.v_cur = self.robot_model.right_hand_velocity()
            
        self.x_cur = pin.SE3(current_matrix[:3, :3], current_matrix[:3, 3])

        # TODO: Calculate the time elapsed since the start of the trajectory
        # Then update the spline to get desired position at current time
        if self.start_time is not None and self.spline is not None:
            t_elapsed = time.time() - self.start_time
            t_elapsed = min(t_elapsed, self.spline_duration)  # Clamp to trajectory duration
            
            x_des_pos, v_des_pos = self.spline.update(t_elapsed)
            
            # Create desired SE3 pose (keep target orientation, update position from spline)
            self.x_des = pin.SE3(self.x_target.rotation, x_des_pos)
        else:
            self.x_des = self.x_cur.copy()
            v_des_pos = np.zeros(3)

        # TODO: Implement the Cartesian PD control law for end-effector POSITION only (no orientation part)
        # compute desired end-effector velocity
        pos_error = self.x_des.translation - self.x_cur.translation
        vel_error = v_des_pos - self.v_cur[:3]  # Take only linear velocity part
        xdot_des = self.Kp * pos_error + self.Kd * vel_error
        
        xdot_des_norm = np.linalg.norm(xdot_des)
        if (xdot_des_norm > self.linear_vel_limit):
            xdot_des = xdot_des * (self.linear_vel_limit / xdot_des_norm)
            self.node.get_logger().warn(f"{self.arm_side} arm linear velocity limited to {self.linear_vel_limit} m/s")
        
        # TODO: Retrieve the end-effector Jacibian that relates 
        # the end-effector LINEAR velocity and the ARM JOINTS.
        # Hint: Extract the linear part of the full Jacobian by taking its first three rows, 
        # and keep only the columns corresponding to the arm joints.
        # You can obtain the arm joint indices using AinexModel.get_arm_ids().
        arm_ids = self.robot_model.get_arm_ids(self.arm_side)
        
        if self.arm_side == 'left':
            J = self.robot_model.J_left
        else:
            J = self.robot_model.J_right
            
        # Extract linear part (first 3 rows) and arm joint columns only
        J_pos = J[:3, arm_ids]

        # TODO: compute the control command (velocities for the arm joints)
        # by mapping the desired end-effector velocity to arm joint velocities 
        # using the Jacobian pseudoinverse
        J_pos_pinv = np.linalg.pinv(J_pos)
        u = J_pos_pinv @ xdot_des
        
        for i in range(len(u)):
            if abs(u[i]) > self.joint_vel_limit[i]:
                u[i] = np.sign(u[i]) * self.joint_vel_limit[i]
                self.node.get_logger().warn(f"{self.arm_side} arm joint {i} velocity limited to {self.joint_vel_limit[i]} rad/s")
       
        ## Check manipulability to prevent singularities
        # TODO: calculate the manipulability index with the task Jacobian J_pos.
        # Hint: w = sqrt(det(J * J^T))
        # If the manipulability is below the threshold self.w_threshold,
        # stop the robot by setting u to zero.
        JJT = J_pos @ J_pos.T
        w = np.sqrt(np.linalg.det(JJT))
        
        if w < self.w_threshold:
            self.node.get_logger().warn(f"{self.arm_side} arm near singularity (w={w:.6f}), stopping motion")
            u = np.zeros_like(u)

        return u

    def is_finished(self) -> bool:
        """Check if the trajectory has finished executing.
        
        Returns:
            bool: True if trajectory is complete, False otherwise.
        """
        if self.start_time is None or self.spline is None:
            return True
            
        t_elapsed = time.time() - self.start_time
        return t_elapsed >= self.spline_duration