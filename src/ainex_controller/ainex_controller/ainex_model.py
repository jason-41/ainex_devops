import pinocchio as pin
import numpy as np
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation

class AiNexModel:
    """AiNex Robot Model using Pinocchio"""
    def __init__(self, node: Node, urdf_path: str, q_init=None, v_init=None):
        self.node = node
        self.br = TransformBroadcaster(node)

        # Load Pinocchio model from URDF
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        self.q = q_init if q_init is not None else np.zeros(self.model.nq)
        self.v = v_init if v_init is not None else np.zeros(self.model.nv)

        # Add additional frames for left and right hands
        self.add_additional_frames(
            name="l_hand_link",
            parent_frame="l_gripper_link",
            translation=np.array([-0.02, 0.025, 0.0]),
            rotation=np.eye(3)
        )
        self.add_additional_frames(
            name="r_hand_link",
            parent_frame="r_gripper_link",
            translation=np.array([-0.02, -0.025, 0.0]),
            rotation=np.eye(3)
        )

        # Retrieve frame IDs for hands for later use
        self.left_hand_id = self.model.getFrameId("l_hand_link")
        self.right_hand_id = self.model.getFrameId("r_hand_link")

        self.left_arm_id = self.get_arm_ids('left')
        self.right_arm_id = self.get_arm_ids('right')
        # Initialize end-effector poses and velocities
        self.x_left = pin.SE3.Identity()
        self.x_right = pin.SE3.Identity()

        self.v_left = pin.Motion.Zero()
        self.v_right = pin.Motion.Zero()

        # Initialize Jacobians for left and right hands
        self.J_left = np.zeros((6, self.model.nv))
        self.J_right = np.zeros((6, self.model.nv))

        # Store joint names in Pinocchio order
        self.joint_names = list(self.model.names[1:])
        print("Node names in Pinocchio model:", self.joint_names)


#####################################################################################
    def update_target(self, target_q):
        self.q = target_q
        # Just add the command from outside(exercise 02) like: AiNexModel.update_target([0.23, -1.0, -0.7, 0.7, 0.23, 1.0, 0.7, 0.7])  
        
    def update_model(self, q, v):
        """Update the model with new joint positions and velocities."""
        self.q = q
        self.v = v
        # update pinocchio model with new q, v
        # print("ori q:",self.q)
        target_q_left = [0.23, -1.0, -0.7, 0.7]
        target_q_right = [0.23, 1.0, 0.7, 0.7]
        check_i = 0
        for i in self.left_arm_id:
            self.q[i] = target_q_left[check_i]
            check_i += 1
        check_i = 0
        for i in self.right_arm_id:
            self.q[i] = target_q_right[check_i]
            check_i += 1
        # print("new q:",self.q)
#####################################################################################

        pin.forwardKinematics(self.model, self.data, self.q, self.v)
        pin.updateFramePlacements(self.model, self.data)

        # Done--TODO: retrieve end-effector poses from updated pinocchio data
        # Hint: take a look at the init function for hand ids
        self.x_left = self.data.oMf[self.left_hand_id].homogeneous
        self.x_right = self.data.oMf[self.right_hand_id].homogeneous
        print("Left hand pose:\n", self.x_left)
        print("Right hand pose:\n", self.x_right)

        # TODO: get end-effector Jacobians in pin.LOCAL_WORLD_ALIGNED frame
        self.J_left = pin.computeFrameJacobian(self.model, self.data, self.q, self.left_hand_id, pin.LOCAL_WORLD_ALIGNED)
        self.J_right = pin.computeFrameJacobian(self.model, self.data, self.q, self.right_hand_id, pin.LOCAL_WORLD_ALIGNED)

        # TODO: calculate end-effector velocities using the Jacobians
        # Hint: v_cartesian = J * v_joint
        self.v_left = self.J_left @ self.v
        self.v_right = self.J_right @ self.v

        # TODO: broadcast tf transformation of hand links w.r.t. base_link for visualization in RViz
        # Hint: take a look at the tf2_ros documentation for examples
        # https://docs.ros.org/en/jazzy/Tutorials/Intermediate/Tf2/Writing-A-Tf2-Broadcaster-Py.html
        
        # Broadcast left hand transform
        t_left = TransformStamped()
        t_left.header.stamp = self.node.get_clock().now().to_msg()
        t_left.header.frame_id = 'base_link'
        t_left.child_frame_id = 'l_hand_link'
        
        # Extract translation from 4x4 matrix
        t_left.transform.translation.x = float(self.x_left[0, 3])
        t_left.transform.translation.y = float(self.x_left[1, 3])
        t_left.transform.translation.z = float(self.x_left[2, 3])
        
        # Extract rotation matrix and convert to quaternion
        rotation_matrix_left = self.x_left[:3, :3]
        r_left = Rotation.from_matrix(rotation_matrix_left)
        quat_left = r_left.as_quat()  # returns [x, y, z, w]
        
        t_left.transform.rotation.x = float(quat_left[0])
        t_left.transform.rotation.y = float(quat_left[1])
        t_left.transform.rotation.z = float(quat_left[2])
        t_left.transform.rotation.w = float(quat_left[3])
        
        # Broadcast right hand transform
        t_right = TransformStamped()
        t_right.header.stamp = self.node.get_clock().now().to_msg()
        t_right.header.frame_id = 'base_link'
        t_right.child_frame_id = 'r_hand_link'
        
        # Extract translation from 4x4 matrix
        t_right.transform.translation.x = float(self.x_right[0, 3])
        t_right.transform.translation.y = float(self.x_right[1, 3])
        t_right.transform.translation.z = float(self.x_right[2, 3])
        
        # Extract rotation matrix and convert to quaternion
        rotation_matrix_right = self.x_right[:3, :3]
        r_right = Rotation.from_matrix(rotation_matrix_right)
        quat_right = r_right.as_quat()  # returns [x, y, z, w]
        
        t_right.transform.rotation.x = float(quat_right[0])
        t_right.transform.rotation.y = float(quat_right[1])
        t_right.transform.rotation.z = float(quat_right[2])
        t_right.transform.rotation.w = float(quat_right[3])
        
        # Send both transforms
        self.br.sendTransform([t_left, t_right])

    def add_additional_frames(self, name, parent_frame, translation, rotation):
        # parent_frame: a frame name in the existing model (string)
        # compute the parent joint index from the existing frame
        parent_frame_id = self.model.getFrameId(parent_frame)
        parent_joint = self.model.frames[parent_frame_id].parent

        # ensure numpy arrays and create SE3 placement
        translation = np.asarray(translation, dtype=float)
        rotation = np.asarray(rotation, dtype=float)
        placement = pin.SE3(rotation, translation)

        # create Frame using parent joint id and SE3 placement
        frame = pin.Frame(name, parent_joint, placement, pin.FrameType.OP_FRAME)

        # add to model and recreate data so sizes match
        self.model.addFrame(frame)
        self.data = self.model.createData()

    def left_hand_pose(self):
        """Return the left hand pose in base_link frame."""
        return self.x_left
    
    def right_hand_pose(self):
        """Return the right hand pose in base_link frame."""
        return self.x_right
    
    def left_hand_velocity(self):
        """Return the left hand velocity in base_link frame."""
        return self.v_left
    
    def right_hand_velocity(self):
        """Return the right hand velocity in base_link frame."""
        return self.v_right
    
    def pin_joint_names(self):
        """Return the joint names in Pinocchio model order."""
        return self.joint_names
    
    def get_arm_ids(self, arm_side: str):
        """Get joint ids for the specified arm side ('left' or 'right')."""
        # l/r_sho_pitch, l/r_sho_roll, l/r_el_yaw, l/r_el_pitch

        arm_joint_names = ['sho_pitch', 'sho_roll', 'el_yaw', 'el_pitch']
        if arm_side == 'left':
            prefix = 'l_'
        elif arm_side == 'right':
            prefix = 'r_'
        else:
            raise ValueError("arm_side must be 'left' or 'right'")
        
        arm_joint_names = [prefix + 'sho_pitch', #0.23
                           prefix + 'sho_roll',  #-1
                           prefix + 'el_yaw', #-0.7
                           prefix + 'el_pitch']  #0.7

        arm_ids = []
        for name in arm_joint_names:
            jid = self.model.getJointId(name)
            q_idx = self.model.joints[jid].idx_q
            arm_ids.append(q_idx)

        return arm_ids

    def get_joint_id(self, joint_name: str) -> int:
        """Get the joint id from the pinocchio model."""
        jid = self.model.getJointId(joint_name)
        q_idx = self.model.joints[jid].idx_q
        return q_idx
