from __future__ import annotations

import time
import numpy as np

from real_robot.interfaces import ObservationProvider, RobotCommander
from real_robot.policy_engine import PolicyEngine
from real_robot.safety import SafetyFilter


def run_control_loop(
    provider: ObservationProvider,
    commander: RobotCommander,
    policy: PolicyEngine,
    safety: SafetyFilter,
    policy_hz: float,
    run_seconds: float,
    dry_run: bool,
):
    period = 1.0 / max(policy_hz, 1e-6)
    t0 = time.time()

    while True:
        now = time.time()
        if run_seconds > 0 and (now - t0) >= run_seconds:
            break

        obs = provider.get_observation(timeout_s=0.1)
        if obs is None:
            commander.hold_position()
            time.sleep(period)
            continue

        obs_dict = obs.as_policy_dict()
        raw_action7 = policy.infer_action7(obs_dict)

        # Fallback current orientation from action if unavailable
        current_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        if "ee_rot" in obs_dict and obs_dict["ee_rot"] is not None:
            # Keep identity as conservative placeholder; real adapter can supply direct EE quaternion.
            pass

        safe_action7 = safety.filter_action(raw_action7, obs.ee_pos, current_quat)

        if not dry_run:
            commander.send_pose_target(safe_action7)

        dt = time.time() - now
        if dt < period:
            time.sleep(period - dt)

    commander.hold_position()
