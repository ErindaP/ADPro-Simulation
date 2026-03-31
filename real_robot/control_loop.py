from __future__ import annotations

import time
import numpy as np

from real_robot.interfaces import ObservationProvider, RobotCommander
from real_robot.policy_engine import PolicyEngine
from real_robot.safety import SafetyFilter
from utils.geometry import rot_to_quat


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
    step_idx = 0

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

        # Fallback current orientation if unavailable
        current_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        if "ee_rot" in obs_dict and obs_dict["ee_rot"] is not None:
            try:
                current_quat = rot_to_quat(np.asarray(obs_dict["ee_rot"], dtype=np.float64))
            except Exception:
                current_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

        safe_action7 = safety.filter_action(raw_action7, obs.ee_pos, current_quat)

        # Debug trace to verify target influence and safety clipping on real robot.
        if step_idx < 5 or (step_idx % 20 == 0):
            obj_txt = "None"
            if "obj_pos" in obs_dict and obs_dict["obj_pos"] is not None:
                obj_txt = np.array2string(np.asarray(obs_dict["obj_pos"]), precision=4)
            print(
                f"[control_loop] step={step_idx} "
                f"ee={np.array2string(np.asarray(obs.ee_pos), precision=4)} "
                f"obj={obj_txt}"
            )
            print(
                "[control_loop] "
                f"raw={np.array2string(np.asarray(raw_action7), precision=4)} "
                f"safe={np.array2string(np.asarray(safe_action7), precision=4)}"
            )

        if not dry_run:
            commander.send_pose_target(safe_action7)

        dt = time.time() - now
        if dt < period:
            time.sleep(period - dt)
        step_idx += 1

    commander.hold_position()
