from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class SafetyConfig:
    workspace_min: np.ndarray
    workspace_max: np.ndarray
    max_translation_step_m: float
    max_rotation_step_rad: float
    lowpass_alpha: float


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if n < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


def _quat_distance_rad(q1: np.ndarray, q2: np.ndarray) -> float:
    q1 = _normalize_quat(q1)
    q2 = _normalize_quat(q2)
    dot = np.clip(abs(np.dot(q1, q2)), 0.0, 1.0)
    return float(2.0 * np.arccos(dot))


class SafetyFilter:
    def __init__(self, cfg: SafetyConfig):
        self.cfg = cfg
        self._last_action7 = None

    def filter_action(self, action7: np.ndarray, current_ee_pos: np.ndarray, current_quat: np.ndarray) -> np.ndarray:
        a = action7.astype(np.float64).copy()

        # 1) Workspace clamp
        a[:3] = np.clip(a[:3], self.cfg.workspace_min, self.cfg.workspace_max)

        # 2) Translation step clamp wrt current EE pose
        dpos = a[:3] - current_ee_pos
        dnorm = np.linalg.norm(dpos)
        if dnorm > self.cfg.max_translation_step_m:
            a[:3] = current_ee_pos + dpos * (self.cfg.max_translation_step_m / (dnorm + 1e-12))

        # 3) Quaternion sanitize + step clamp
        q_cur = _normalize_quat(current_quat)
        q_cmd = _normalize_quat(a[3:7])
        if _quat_distance_rad(q_cur, q_cmd) > self.cfg.max_rotation_step_rad:
            # If orientation jump too large, hold current orientation for safety
            q_cmd = q_cur
        a[3:7] = q_cmd

        # 4) Low-pass smoothing wrt last sent command
        if self._last_action7 is not None:
            alpha = float(np.clip(self.cfg.lowpass_alpha, 0.0, 1.0))
            a = alpha * a + (1.0 - alpha) * self._last_action7
            a[3:7] = _normalize_quat(a[3:7])

        self._last_action7 = a.copy()
        return a
