from __future__ import annotations

import numpy as np

from env.panda_env import PandaPickPlaceEnv
from real_robot.interfaces import Observation, ObservationProvider, RobotCommander


class MockObservationProvider(ObservationProvider):
    """Useful for dry-run on a non-robot machine."""

    def __init__(self, seed: int = 42):
        self.env = PandaPickPlaceEnv(seed=seed)
        self.obs = self.env.reset()

    def get_observation(self, timeout_s: float = 0.0):
        o = self.obs
        return Observation(
            pc_gripper=o["pc_gripper"],
            pc_scene=o["pc_scene"],
            ee_pos=o["ee_pos"],
            ee_rot=o["ee_rot"],
            gripper_width=float(o["gripper_width"]),
            obj_pos=o.get("obj_pos"),
            T_ee=o.get("T_ee"),
        )

    def apply_action(self, action7: np.ndarray):
        self.obs, _, _, _ = self.env.step(action7)

    def close(self):
        pass


class MockCommander(RobotCommander):
    def __init__(self, provider: MockObservationProvider):
        self.provider = provider

    def send_pose_target(self, action7: np.ndarray, gripper_width=None):
        self.provider.apply_action(action7)

    def hold_position(self):
        pass

    def close(self):
        pass
