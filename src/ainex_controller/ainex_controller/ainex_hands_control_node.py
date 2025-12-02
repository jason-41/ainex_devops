import rclpy
from ament_index_python.packages import get_package_share_directory

import pinocchio as pin
import numpy as np
import time

from ainex_controller.ainex_model import AiNexModel
from ainex_controller.ainex_robot import AinexRobot
from ainex_controller.ainex_hand_controller import HandController

def main():
    rclpy.init()
    node = rclpy.create_node('ainex_hands_control_node')

    dt = 0.05

    try:
        # Initialize robot model
        # get package path
        pkg = get_package_share_directory('ainex_description')
        urdf_path = pkg + '/urdf/ainex.urdf'
        robot_model = AiNexModel(node, urdf_path)

        # Create AinexRobot instance
        # TODO: set sim=False when interfacing with real robot
        # IMPORTANT !!!: Always test first in simulation to avoid damage to the real robot!!!
        ainex_robot = AinexRobot(node, robot_model, dt, sim=True)

        q_init = np.zeros(robot_model.model.nq)
        # Home position defined in urdf/pinocchio model
        # TODO: feel free to change to other initial positions away from singularities
        q_init[robot_model.get_joint_id('r_sho_roll')] = 1.4
        q_init[robot_model.get_joint_id('l_sho_roll')] = -1.4
        q_init[robot_model.get_joint_id('r_el_yaw')] = 1.58
        q_init[robot_model.get_joint_id('l_el_yaw')] = -1.58
        
        # Move robot to initial position
        ainex_robot.move_to_initial_position(q_init)
        
        # Wait for initial position to be reached
        time.sleep(2.0)

        # Create HandController instances for left and right hands
        left_hand_controller = HandController(node, robot_model, arm_side='left')
        right_hand_controller = HandController(node, robot_model, arm_side='right')

        # TODO: Feel free to change to other target poses for testing

        # left hand target pose (relative movement)
        left_target = pin.SE3.Identity()
        left_target.translation = np.array([0.0, 0.03, 0.0])  # Move 3 cm forward in Y direction
        left_hand_controller.set_target_pose(left_target, duration=3.0, type='rel')

        # right hand target pose (absolute movement)
        # Get current pose directly from robot model (it's already updated in move_to_initial_position)
        right_current_matrix = robot_model.right_hand_pose()
        
        # Create SE3 object from current pose matrix
        right_current_se3 = pin.SE3(right_current_matrix[:3, :3], right_current_matrix[:3, 3])
        
        # Create target as SE3 object
        right_target = pin.SE3(right_current_se3.rotation, right_current_se3.translation.copy())
        right_target.translation[2] += 0.02  # Move up by 2 cm in Z direction
        right_hand_controller.set_target_pose(right_target, duration=3.0, type='abs')

        node.get_logger().info("Starting hand control...")
        node.get_logger().info(f"Left hand moving 3cm forward (Y), Right hand moving 2cm up (Z)")

        # Control loop
        v_cmd_left = None
        v_cmd_right = None
        start_time = time.time()
        
        while rclpy.ok():
            try:
                # Update controllers
                v_cmd_left = left_hand_controller.update(dt)
                v_cmd_right = right_hand_controller.update(dt)
                
                # Update robot with velocity commands
                ainex_robot.update(v_cmd_left, v_cmd_right, dt)
                
                # Spin ROS node
                rclpy.spin_once(node, timeout_sec=dt)
                
                # Check if both controllers are done (optional timeout)
                if (left_hand_controller.is_finished() and right_hand_controller.is_finished()):
                    node.get_logger().info("Both hand movements completed!")
                    break
                    
                # Safety timeout (10 seconds)
                if time.time() - start_time > 10.0:
                    node.get_logger().warn("Control timeout reached, stopping...")
                    break
                    
            except KeyboardInterrupt:
                node.get_logger().info("Received interrupt signal, stopping...")
                break
            except Exception as e:
                node.get_logger().error(f"Error in control loop: {e}")
                break

    except Exception as e:
        node.get_logger().error(f"Failed to initialize robot: {e}")
    finally:
        # Clean shutdown
        try:
            # Send zero velocities to stop robot motion (if not in simulation)
            if 'ainex_robot' in locals() and not ainex_robot.sim:
                # Stop motion by sending current position as command
                ainex_robot.send_cmd(ainex_robot.q, 0.1)
        except:
            pass
            
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()