#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
import time
import numpy as np
import pinocchio as pin
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation

from ainex_controller.ainex_model import AiNexModel
from ainex_controller.ainex_robot import AinexRobot

def main():
    rclpy.init()
    node = rclpy.create_node("monitor_real_robot_node")
    
    # Force sim=False because we want to connect to real robot
    sim = False
    
    dt = 0.05 # 20Hz refresh rate
    
    # Publisher for camera pose
    camera_pose_pub = node.create_publisher(PoseStamped, '/ainex_camera_pose', 10)
    target_frame = "head_tilt"

    try:
        node.get_logger().info("Initializing connection to REAL robot...")
        node.get_logger().info("Note: Ensure your ROS 2 network is configured to reach 192.168.50.201")
        
        # Load URDF model
        pkg = get_package_share_directory("ainex_description")
        urdf_path = pkg + "/urdf/ainex.urdf"
        robot_model = AiNexModel(node, urdf_path)
        
        # Check if frame exists
        if robot_model.model.existFrame(target_frame):
            frame_id = robot_model.model.getFrameId(target_frame)
            node.get_logger().info(f"Found frame '{target_frame}' with ID {frame_id}")
        else:
            node.get_logger().error(f"Frame '{target_frame}' not found in URDF! Camera pose will not be published.")
            frame_id = None
        
        # Initialize AinexRobot (sim=False will create JointController and connect to services)
        # This blocks until services like 'Get_Joint' are available!
        ainex_robot = AinexRobot(node, robot_model, dt, sim=sim)
        
        node.get_logger().info("Successfully connected! Starting to publish joint states...")
        node.get_logger().info("Publishing to topic: /ainex_joint_states")
        node.get_logger().info(f"Publishing camera pose to: /ainex_camera_pose (Frame: {target_frame})")

        while rclpy.ok():
            start_time = time.time()
            
            # 1. Read real positions from robot (uses Service calls)
            # This calls joint_controller.getJointPositions and applies offset fixes
            q_real = ainex_robot.read_joint_positions_from_robot()
            
            if q_real is None:
                # Skip this iteration if read failed
                time.sleep(1.0)
                continue

            # Print head_tilt (debug)
            # if 'head_tilt' in ainex_robot.joint_names:
            #     idx = ainex_robot.joint_names.index('head_tilt')
            #     node.get_logger().info(f"head_tilt: {q_real[idx]}")

            # if 'head_tilt' in ainex_robot.joint_names:
            #     idx = ainex_robot.joint_names.index('head_tilt')
            #     node.get_logger().info(f"head_tilt: {q_real[idx]}")
            # 2. Update the internal state of the AinexRobot wrapper
            # (We only update position 'q', we assume velocity 'v' is 0 or we could try to estimate it)
            ainex_robot.q = q_real
            ainex_robot.v = np.zeros_like(q_real) # We don't have real velocity feedback readily available in this method
            
            # 3. Publish to ROS 2 topic /ainex_joint_states
            ainex_robot.publish_joint_states()
            
            # 4. Calculate and Publish Camera Pose
            if frame_id is not None:
                # Update Pinocchio model (Forward Kinematics)
                robot_model.update_model(q_real, ainex_robot.v)
                
                # Get frame pose (SE3 object)
                # update_model does pin.updateFramePlacements, so we can access data.oMf
                pose_se3 = robot_model.data.oMf[frame_id]
                
                # Convert to PoseStamped
                msg = PoseStamped()
                msg.header.stamp = node.get_clock().now().to_msg()
                msg.header.frame_id = "base_link" # Pose relative to base_link
                
                # Translation
                msg.pose.position.x = float(pose_se3.translation[0])
                msg.pose.position.y = float(pose_se3.translation[1])
                msg.pose.position.z = float(pose_se3.translation[2])
                
                # Rotation (Matrix -> Quaternion)
                # Pinocchio rotation is a 3x3 matrix
                r = Rotation.from_matrix(pose_se3.rotation)
                quat = r.as_quat() # [x, y, z, w]
                
                msg.pose.orientation.x = float(quat[0])
                msg.pose.orientation.y = float(quat[1])
                msg.pose.orientation.z = float(quat[2])
                msg.pose.orientation.w = float(quat[3])
                
                camera_pose_pub.publish(msg)
            
            # Sleep to maintain loop rate
            process_duration = time.time() - start_time
            sleep_time = max(0.0, dt - process_duration)
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        node.get_logger().info("Monitor stopped by user.")
    except Exception as e:
        node.get_logger().error(f"Error in monitor: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
