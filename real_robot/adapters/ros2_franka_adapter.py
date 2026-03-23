from __future__ import annotations

"""
ROS2 adapter (template) for lab machine connected to Franka.

Expected external stack:
- A perception node publishing gripper and scene point clouds.
- A controller node subscribing to target EE poses.

This adapter publishes pose targets and reads observations from ROS topics.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import Float64
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs_py import point_cloud2
except Exception:  # pragma: no cover
    rclpy = None
    Node = object

from real_robot.interfaces import Observation, ObservationProvider, RobotCommander


def _pc2_to_xyz(msg: PointCloud2) -> np.ndarray:
    pts = []
    for p in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
        pts.append([float(p[0]), float(p[1]), float(p[2])])
    if not pts:
        return np.zeros((1, 3), dtype=np.float64)
    return np.asarray(pts, dtype=np.float64)


@dataclass
class Ros2Topics:
    pc_gripper_topic: str
    pc_scene_topic: str
    ee_pose_topic: str
    target_pose_topic: str
    gripper_width_topic: str


class _RosNode(Node):
    def __init__(self, topics: Ros2Topics):
        super().__init__("adpro_bridge")
        self.topics = topics

        self.pc_gripper_msg = None
        self.pc_scene_msg = None
        self.ee_pose_msg = None

        self.create_subscription(PointCloud2, topics.pc_gripper_topic, self._on_pc_gripper, 10)
        self.create_subscription(PointCloud2, topics.pc_scene_topic, self._on_pc_scene, 10)
        self.create_subscription(PoseStamped, topics.ee_pose_topic, self._on_ee_pose, 10)

        self.pub_target = self.create_publisher(PoseStamped, topics.target_pose_topic, 10)
        self.pub_gripper = self.create_publisher(Float64, topics.gripper_width_topic, 10)

    def _on_pc_gripper(self, msg):
        self.pc_gripper_msg = msg

    def _on_pc_scene(self, msg):
        self.pc_scene_msg = msg

    def _on_ee_pose(self, msg):
        self.ee_pose_msg = msg


class Ros2ObservationProvider(ObservationProvider):
    def __init__(self, node: _RosNode):
        self.node = node

    def get_observation(self, timeout_s: float = 0.0) -> Optional[Observation]:
        if rclpy is None:
            raise RuntimeError("ROS2 non disponible (rclpy non installé).")

        deadline = self.node.get_clock().now().nanoseconds + int(timeout_s * 1e9)
        while rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.02)
            if self.node.pc_gripper_msg and self.node.pc_scene_msg and self.node.ee_pose_msg:
                break
            if timeout_s > 0 and self.node.get_clock().now().nanoseconds > deadline:
                return None

        if not (self.node.pc_gripper_msg and self.node.pc_scene_msg and self.node.ee_pose_msg):
            return None

        pc_gripper = _pc2_to_xyz(self.node.pc_gripper_msg)
        pc_scene = _pc2_to_xyz(self.node.pc_scene_msg)

        p = self.node.ee_pose_msg.pose.position
        q = self.node.ee_pose_msg.pose.orientation
        ee_pos = np.array([p.x, p.y, p.z], dtype=np.float64)
        quat = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)

        # Rotation not strictly needed by current policy path; keep identity placeholder.
        ee_rot = np.eye(3, dtype=np.float64)

        return Observation(
            pc_gripper=pc_gripper,
            pc_scene=pc_scene,
            ee_pos=ee_pos,
            ee_rot=ee_rot,
            gripper_width=0.04,
            obj_pos=None,
            T_ee=None,
        )

    def close(self):
        pass


class Ros2FrankaCommander(RobotCommander):
    def __init__(self, node: _RosNode):
        self.node = node

    def send_pose_target(self, action7: np.ndarray, gripper_width: Optional[float] = None) -> None:
        msg = PoseStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "panda_link0"

        msg.pose.position.x = float(action7[0])
        msg.pose.position.y = float(action7[1])
        msg.pose.position.z = float(action7[2])

        msg.pose.orientation.x = float(action7[3])
        msg.pose.orientation.y = float(action7[4])
        msg.pose.orientation.z = float(action7[5])
        msg.pose.orientation.w = float(action7[6])

        self.node.pub_target.publish(msg)

        if gripper_width is not None:
            g = Float64()
            g.data = float(gripper_width)
            self.node.pub_gripper.publish(g)

    def hold_position(self) -> None:
        # Could re-publish latest EE pose if desired.
        pass

    def close(self) -> None:
        pass


def build_ros2_stack(topics: Ros2Topics):
    if rclpy is None:
        raise RuntimeError("ROS2 non disponible (rclpy non installé).")
    rclpy.init(args=None)
    node = _RosNode(topics)
    provider = Ros2ObservationProvider(node)
    commander = Ros2FrankaCommander(node)
    return node, provider, commander
