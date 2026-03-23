from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol
import numpy as np


@dataclass
class Observation:
    pc_gripper: np.ndarray
    pc_scene: np.ndarray
    ee_pos: np.ndarray
    ee_rot: np.ndarray
    gripper_width: float
    obj_pos: Optional[np.ndarray] = None
    T_ee: Optional[np.ndarray] = None

    def as_policy_dict(self) -> dict:
        data = {
            "pc_gripper": self.pc_gripper,
            "pc_scene": self.pc_scene,
            "ee_pos": self.ee_pos,
            "ee_rot": self.ee_rot,
            "gripper_width": float(self.gripper_width),
            "T_ee": self.T_ee,
        }
        if self.obj_pos is not None:
            data["obj_pos"] = self.obj_pos
        return data


class ObservationProvider(Protocol):
    def get_observation(self, timeout_s: float = 0.0) -> Optional[Observation]:
        ...

    def close(self) -> None:
        ...


class RobotCommander(Protocol):
    def send_pose_target(self, action7: np.ndarray, gripper_width: Optional[float] = None) -> None:
        ...

    def hold_position(self) -> None:
        ...

    def close(self) -> None:
        ...
