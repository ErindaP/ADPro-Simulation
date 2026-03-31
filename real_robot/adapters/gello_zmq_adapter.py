from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any
import os
import sys

import numpy as np

from real_robot.interfaces import Observation, ObservationProvider, RobotCommander
from utils.geometry import (
    FRANKA_JOINT_LIMITS,
    action_to_se3,
    forward_kinematics,
    rot_to_quat,
)
from utils.point_cloud import sample_gripper_pointcloud, sample_scene_pointcloud


MAX_OPEN = 0.09


@dataclass
class GelloZmqConfig:
    host: str
    robot_port: int
    obj_pos: np.ndarray
    table_height: float
    n_pc_points: int
    ik_iters: int


class GelloObservationProvider(ObservationProvider):
    def __init__(self, client: Any, cfg: GelloZmqConfig):
        self.client = client
        self.cfg = cfg
        self._rng = np.random.default_rng(42)

    def get_observation(self, timeout_s: float = 0.0) -> Optional[Observation]:
        _ = timeout_s  # ZMQ client in this repo is request/reply without timeout control.
        robot_obs = self.client.get_observations()
        if robot_obs is None or "joint_positions" not in robot_obs:
            return None

        joints = np.asarray(robot_obs["joint_positions"], dtype=np.float64)
        if joints.shape[0] < 7:
            return None

        q = joints[:7]
        grip_norm = float(joints[7]) if joints.shape[0] >= 8 else 0.5
        gripper_width = float(np.clip(grip_norm, 0.0, 1.0) * MAX_OPEN)

        T_ee = forward_kinematics(q)
        ee_pos = T_ee[:3, 3].copy()
        ee_rot = T_ee[:3, :3].copy()

        pc_gripper = sample_gripper_pointcloud(
            T_ee,
            gripper_width=np.clip(gripper_width, 0.0, MAX_OPEN),
            n_points=self.cfg.n_pc_points,
            rng=self._rng,
        )
        pc_scene = sample_scene_pointcloud(
            self.cfg.obj_pos,
            table_height=self.cfg.table_height,
            n_obj_points=self.cfg.n_pc_points // 2,
            n_table_points=self.cfg.n_pc_points // 2,
            rng=self._rng,
        )

        return Observation(
            pc_gripper=pc_gripper,
            pc_scene=pc_scene,
            ee_pos=ee_pos,
            ee_rot=ee_rot,
            gripper_width=gripper_width,
            obj_pos=self.cfg.obj_pos.copy(),
            T_ee=T_ee,
        )

    def close(self):
        pass


class GelloZmqCommander(RobotCommander):
    def __init__(self, client: Any, cfg: GelloZmqConfig):
        self.client = client
        self.cfg = cfg
        self._q_last = None
        self._grip_norm_last = 0.5

    def _solve_ik(self, T_target: np.ndarray, q_init: np.ndarray, n_iter: int) -> np.ndarray:
        q = np.asarray(q_init, dtype=np.float64).copy()
        lambda_dls = 1e-2
        eps = 1e-4

        for _ in range(n_iter):
            T_cur = forward_kinematics(q)
            pos_err = T_target[:3, 3] - T_cur[:3, 3]
            R_err = T_target[:3, :3] @ T_cur[:3, :3].T
            q_err = rot_to_quat(R_err)
            rot_vec_err = 2.0 * q_err[:3] * np.sign(q_err[3])
            err = np.concatenate([pos_err, 0.25 * rot_vec_err])

            if np.linalg.norm(err[:3]) < 1e-4 and np.linalg.norm(err[3:]) < 1e-3:
                break

            J = np.zeros((6, 7), dtype=np.float64)
            for j in range(7):
                q_p = q.copy()
                q_p[j] += eps
                T_p = forward_kinematics(q_p)
                dp = (T_p[:3, 3] - T_cur[:3, 3]) / eps

                R_delta = T_p[:3, :3] @ T_cur[:3, :3].T
                q_delta = rot_to_quat(R_delta)
                drot = (2.0 * q_delta[:3] * np.sign(q_delta[3])) / eps

                J[:3, j] = dp
                J[3:, j] = drot

            A = J @ J.T + lambda_dls * np.eye(6)
            dq = J.T @ np.linalg.solve(A, err)
            dq = np.clip(dq, -0.08, 0.08)
            q = np.clip(q + dq, FRANKA_JOINT_LIMITS[:, 0], FRANKA_JOINT_LIMITS[:, 1])

        return q

    def _current_joint_state(self) -> np.ndarray:
        js = np.asarray(self.client.get_joint_state(), dtype=np.float64)
        if js.shape[0] < 8:
            js = np.pad(js, (0, max(0, 8 - js.shape[0])), mode="constant")
        return js

    def send_pose_target(self, action7: np.ndarray, gripper_width: Optional[float] = None) -> None:
        js = self._current_joint_state()
        q_cur = js[:7]
        grip_norm_cur = float(np.clip(js[7], 0.0, 1.0))
        self._grip_norm_last = grip_norm_cur

        q_seed = self._q_last if self._q_last is not None else q_cur
        T_target = action_to_se3(action7)
        q_cmd = self._solve_ik(T_target, q_seed, n_iter=self.cfg.ik_iters)
        self._q_last = q_cmd.copy()

        if gripper_width is None:
            grip_norm_cmd = grip_norm_cur
        else:
            width = float(np.clip(gripper_width, 0.0, MAX_OPEN))
            # PandaRobot.command_joint_state uses 1-width/MAX_OPEN.
            grip_norm_cmd = float(np.clip(1.0 - (width / MAX_OPEN), 0.0, 1.0))

        cmd = np.concatenate([q_cmd, [grip_norm_cmd]], dtype=np.float64)
        self.client.command_joint_state(cmd)

    def hold_position(self) -> None:
        js = self._current_joint_state()
        self.client.command_joint_state(js)

    def close(self) -> None:
        pass


def build_gello_stack(cfg: GelloZmqConfig):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from gello.zmq_core.robot_node import ZMQClientRobot

    client = ZMQClientRobot(port=int(cfg.robot_port), host=cfg.host)
    provider = GelloObservationProvider(client=client, cfg=cfg)
    commander = GelloZmqCommander(client=client, cfg=cfg)
    return provider, commander
