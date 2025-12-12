#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster, Buffer, TransformListener

#import tf_transformations  # 用于矩阵/四元数计算

class ArucoTFBroadcaster(Node):
    def __init__(self):
        super().__init__('aruco_tf_broadcaster')

        # ---- TF Broadcaster ----
        self.br = TransformBroadcaster(self)

        # ---- TF Buffer + Listener（用于 base_link 转换） ----
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---- 订阅摄像头检测到的 ArUco Pose ----
        self.subscription = self.create_subscription(
            PoseStamped,
            '/aruco_pose',
            self.handle_aruco_pose,
            10
        )

        # 记录 marker 丢失次数
        self.missed_frames = 0
        self.missed_limit = 5   # 连续 5 帧未检测到则机械臂回到初始位姿

        self.get_logger().info("ArucoTFBroadcaster started.")

    # ----------------------------------------------------------------------
    #                (1) 发布 TF：camera_optical_link → aruco_marker
    # ----------------------------------------------------------------------
    def handle_aruco_pose(self, msg):

        # 发布 TF transform
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = "camera_optical_link"
        t.child_frame_id = "aruco_raw"

        t.transform.translation = msg.pose.position
        t.transform.rotation = msg.pose.orientation

        self.br.sendTransform(t)

        # 成功检测到 → 计数归零
        self.missed_frames = 0

        # ---- 现在计算 ArUco 在 base_link 下的位置 ----
        try:
            tf_base_to_marker = self.tf_buffer.transform(
                t,
                "base_link",
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
        except Exception as e:
            self.get_logger().warn(f"TF transform failed: {e}")
            return

        # 解析坐标
        px = tf_base_to_marker.transform.translation.x
        py = tf_base_to_marker.transform.translation.y
        pz = tf_base_to_marker.transform.translation.z

        q = tf_base_to_marker.transform.rotation
        qx, qy, qz, qw = q.x, q.y, q.z, q.w

        self.get_logger().info(
            f"[base_link] Marker pos = ({px:.3f}, {py:.3f}, {pz:.3f})"
        )

        # ---- (3) 控制机械臂移动到 marker ----
        self.move_arm_to_target(px, py, pz, qx, qy, qz, qw)

    # ----------------------------------------------------------------------
    #                Timer：没有检测到 marker → 回到初始位姿
    # ----------------------------------------------------------------------
    def timer_callback(self):
        if self.missed_frames >= self.missed_limit:
            self.get_logger().warn("Marker lost. Returning robot arm to home pose.")
            self.move_arm_home()
        else:
            self.missed_frames += 1

    # ----------------------------------------------------------------------
    #                 机械臂控制接口（你需要替换为自己的实现）
    # ----------------------------------------------------------------------
    def move_arm_to_target(self, x, y, z, qx, qy, qz, qw):
        """
        TODO: 你需要填入 exercise 1 的机械臂控制逻辑。
        接收 base_link 下的目标位置与姿态。
        """
        self.get_logger().info(
            f"[Arm] Moving to ({x:.3f}, {y:.3f}, {z:.3f}) ..."
        )
        # Example (伪代码):
        # self.arm.move_to_pose(x, y, z, qx, qy, qz, qw)
        pass

    def move_arm_home(self):
        """
        TODO: 回到初始位姿（exercise 1 的函数）
        """
        self.get_logger().info("[Arm] Going home pose.")
        pass


def main(args=None):
    rclpy.init(args=args)
    node = ArucoTFBroadcaster()

    # Timer 用于监测 marker 丢失情况
    node.create_timer(0.2, node.timer_callback)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
