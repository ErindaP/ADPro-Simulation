from __future__ import annotations

import os
import numpy as np

from models.adpro import ADPro
from train_baseline import load_policy


class PolicyEngine:
    def __init__(
        self,
        checkpoint_path: str,
        seed: int,
        adpro_impl: str,
        adpro_steps: int,
        adpro_M: int,
        adpro_spherical_scale: float,
        adpro_stochastic: bool,
    ):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint introuvable: {checkpoint_path}")

        self.policy = load_policy(checkpoint_path, seed=seed)
        self.adpro_steps = int(adpro_steps)

        self.adpro = ADPro(
            base_policy=self.policy,
            eta=0.08,
            M=adpro_M,
            implementation=adpro_impl,
            use_fgr=True,
            use_task_manifold=True,
            use_spherical=True,
            spherical_scale=adpro_spherical_scale,
            deterministic_denoising=(not adpro_stochastic),
            n_fgr_points=256,
            n_fgr_iter=8,
        )

    def infer_action7(self, obs_dict: dict) -> np.ndarray:
        return self.adpro.inference(obs_dict, n_steps=self.adpro_steps)
