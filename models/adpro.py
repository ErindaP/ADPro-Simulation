"""
adpro.py — Adaptive Diffusion Policy (ADPro)

Implémentation pratique d'ADPro (test-time adaptation, sans réentraînement):
1) Task-aware initialization (analogue Eq. 10)
2) Task manifold guidance (analogue Eq. 8)
3) Spherical manifold constraint (Eq. 9)
"""

import numpy as np
from models.diffusion_policy import DiffusionPolicy
from utils.point_cloud import subsample_pointcloud
from utils.geometry import (
    action_to_se3,
    se3_to_action,
    fast_global_registration,
    transform_pointcloud,
    chamfer_distance_grad,
    se3_inverse,
)


class ADPro:
    """
    Wrapper ADPro appliqué sur une politique de diffusion pré-entraînée.

    Cette version est orientée simulation robotique: elle exploite les nuages
    de points et la position objet pour guider la diffusion vers une pose de grasp.
    """

    def __init__(
        self,
        base_policy: DiffusionPolicy,
        eta: float = 0.08,
        M: int = None,
        implementation: str = "practical",
        use_fgr: bool = True,
        use_task_manifold: bool = True,
        use_spherical: bool = True,
        use_blend: bool = None,
        spherical_scale: float = 0.25,
        deterministic_denoising: bool = True,
        n_fgr_points: int = 512,
        n_fgr_iter: int = 8,
    ):
        self.policy = base_policy
        self.eta = eta
        self.M = M
        self.implementation = implementation
        self.use_fgr = use_fgr
        self.use_task_manifold = use_task_manifold
        self.use_spherical = use_spherical
        if use_blend is None:
            # "practical": robuste en simulation ; "paper": fidèle Eq. 5/7/9/10.
            self.use_blend = (implementation == "practical")
        else:
            self.use_blend = use_blend
        self.spherical_scale = spherical_scale
        self.deterministic_denoising = deterministic_denoising
        self.n_fgr_points = n_fgr_points
        self.n_fgr_iter = n_fgr_iter
        self.rng = base_policy.rng
        self.schedule = base_policy.schedule
        self.action_dim = base_policy.action_dim
        if self.implementation not in ("practical", "paper"):
            raise ValueError("implementation doit être 'practical' ou 'paper'")

    def _estimate_object_center(self, obs: dict) -> np.ndarray:
        """
        Estime le centre de l'objet cible depuis la scène.

        On privilégie les points au-dessus de la table (objet), sinon fallback
        sur `obj_pos` si disponible dans l'observation.
        """
        if 'obj_pos' in obs:
            return obs['obj_pos'].copy()

        pc_scene = obs['pc_scene']
        z_min = np.min(pc_scene[:, 2])
        obj_mask = pc_scene[:, 2] > (z_min + 0.01)
        if np.any(obj_mask):
            return pc_scene[obj_mask].mean(axis=0)
        return pc_scene.mean(axis=0)

    def _target_grasp_action(self, obs: dict) -> np.ndarray:
        """Pose de grasp cible (SE3) en format action 7D."""
        obj_center = self._estimate_object_center(obs)
        target_pos = obj_center + np.array([0.0, 0.0, 0.05])

        # Orientation "gripper vers le bas" (180 deg autour de X): q=[1,0,0,0]
        target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        return np.concatenate([target_pos, target_quat])

    def _task_aware_init(self, obs: dict, t_start: int) -> np.ndarray:
        """
        Initialisation structurée (analogue FGR/Eq.10):
        - translation approximée depuis les centroïdes gripper/objet
        - puis injection de bruit au niveau t_start
        """
        pc_gripper = subsample_pointcloud(obs['pc_gripper'], self.n_fgr_points, self.rng)
        pc_scene_all = subsample_pointcloud(obs['pc_scene'], self.n_fgr_points, self.rng)
        z_min = np.min(pc_scene_all[:, 2])
        obj_mask = pc_scene_all[:, 2] > (z_min + 0.01)
        pc_scene = pc_scene_all[obj_mask] if np.any(obj_mask) else pc_scene_all

        if self.implementation == "paper":
            # Version fidèle au papier: FGR rigide complet (R, t).
            T_fgr = fast_global_registration(
                pc_gripper,
                pc_scene,
                n_iter=self.n_fgr_iter,
                rng=self.rng,
            )
            a0_est = se3_to_action(T_fgr)[:self.action_dim]
        else:
            # Version pratique: translation coarse + cible de grasp explicite.
            target = self._target_grasp_action(obs)
            obj_pts = pc_scene

            gripper_center = pc_gripper.mean(axis=0)
            obj_center = obj_pts.mean(axis=0)

            a0_est = target.copy()
            a0_est[:3] = a0_est[:3] + 0.1 * (obj_center - gripper_center)

        sqrt_abar = self.schedule['sqrt_alpha_bars'][t_start]
        sqrt_1m = self.schedule['sqrt_one_minus_alpha_bars'][t_start]
        eps = self.rng.normal(0, 1, self.action_dim)
        a_M = sqrt_abar * a0_est + sqrt_1m * eps
        return a_M

    def _predict_clean_action(self, a_t: np.ndarray, eps_pred: np.ndarray, t: int) -> np.ndarray:
        """Estimation Tweedie de l'action propre a0 depuis at."""
        sqrt_abar = self.schedule['sqrt_alpha_bars'][t]
        sqrt_1m = self.schedule['sqrt_one_minus_alpha_bars'][t]
        a0_hat = (a_t - sqrt_1m * eps_pred) / (sqrt_abar + 1e-8)
        return a0_hat

    def _apply_task_manifold_guidance(
        self,
        a_tilde: np.ndarray,
        a0_hat: np.ndarray,
        obs: dict,
        t: int,
        return_stats: bool = False,
    ) -> np.ndarray:
        """
        Guidance géométrique vers la pose de grasp (task manifold).
        """
        if self.implementation == "paper":
            # Guidance "surface-à-surface" (Chamfer) comme Eq. 8.
            T_a0 = action_to_se3(a0_hat)
            # obs['pc_gripper'] est déjà dans le monde à la pose courante.
            # On repasse d'abord en repère local pince, puis on applique T_a0.
            T_cur = obs.get('T_ee', np.eye(4))
            pc_gripper_local = transform_pointcloud(obs['pc_gripper'], se3_inverse(T_cur))
            pc_gripper_transformed = transform_pointcloud(pc_gripper_local, T_a0)
            pc_scene_all = obs['pc_scene']
            z_min = np.min(pc_scene_all[:, 2])
            obj_mask = pc_scene_all[:, 2] > (z_min + 0.01)
            pc_scene = pc_scene_all[obj_mask] if np.any(obj_mask) else pc_scene_all
            grad_trans, grad_quat = chamfer_distance_grad(
                pc_gripper_transformed,
                pc_scene,
            )
            grad_action = np.concatenate([grad_trans, grad_quat])[:self.action_dim]
        else:
            # Guidance pratique vers une pose cible explicite.
            target = self._target_grasp_action(obs)
            grad_action = np.zeros(self.action_dim, dtype=np.float64)
            grad_action[:3] = a0_hat[:3] - target[:3]

            quat_err = a0_hat[3:7] - target[3:7]
            quat_err_alt = a0_hat[3:7] + target[3:7]  # q et -q même rotation
            if np.linalg.norm(quat_err_alt) < np.linalg.norm(quat_err):
                quat_err = quat_err_alt
            grad_action[3:7] = quat_err

        grad_norm = float(np.linalg.norm(grad_action))

        if self.use_spherical:
            d = self.action_dim
            sigma_t = np.sqrt(self.schedule['betas'][t])
            norm_grad = grad_norm + 1e-8
            step = self.spherical_scale * np.sqrt(d) * sigma_t * (grad_action / norm_grad)
        else:
            step = self.eta * grad_action

        a_corr = a_tilde - step
        if return_stats:
            return a_corr, grad_norm, float(np.linalg.norm(step))
        return a_corr

    def _adpro_step(self, a_t: np.ndarray, obs: dict, t: int, debug: dict = None) -> np.ndarray:
        eps_pred = self.policy.network.forward(obs, a_t, t)
        add_noise = (t > 0) and (not self.deterministic_denoising)
        a_tilde = self.policy.ddpm_step(a_t, eps_pred, t, add_noise=add_noise)

        if self.use_task_manifold:
            a0_hat = self._predict_clean_action(a_t, eps_pred, t)
            if debug is not None:
                a_out, grad_norm, guide_step_norm = self._apply_task_manifold_guidance(
                    a_tilde, a0_hat, obs, t, return_stats=True
                )
                debug['grad_norms'].append(grad_norm)
                debug['guide_step_norms'].append(guide_step_norm)
            else:
                a_out = self._apply_task_manifold_guidance(a_tilde, a0_hat, obs, t)
        else:
            a_out = a_tilde
            if debug is not None:
                debug['grad_norms'].append(0.0)
                debug['guide_step_norms'].append(0.0)

        # Option pratique: blend final vers la cible (désactivé en mode "paper").
        if self.use_blend:
            target = self._target_grasp_action(obs)
            blend = 0.15 + 0.75 * (1.0 - t / max(self.policy.T - 1, 1))
            a_out = (1.0 - blend) * a_out + blend * target
        else:
            blend = 0.0
        if debug is not None:
            debug['blend'].append(float(blend))
            debug['timesteps'].append(int(t))
            debug['deterministic_denoising'] = self.deterministic_denoising

        return a_out

    def inference(
        self,
        obs: dict,
        n_steps: int = None,
        return_trajectory: bool = False,
        return_debug: bool = False,
    ) -> np.ndarray:
        T = n_steps or self.policy.T

        target = self._target_grasp_action(obs)
        if self.use_fgr and self.M is not None:
            # Important: si n_steps < M (ex: 20 < 60), on garde quand même
            # l'initialisation task-aware en la projetant au pas maximal dispo.
            M = min(self.M, T - 1)
            a_t = self._task_aware_init(obs, t_start=M)
            t_start = M
            init_mode = 'task_aware'
        else:
            a_t = self.rng.normal(0, 1, self.action_dim)
            t_start = T - 1
            init_mode = 'gaussian'

        trajectory = [a_t.copy()] if return_trajectory else None
        debug = None
        if return_debug:
            debug = {
                'timesteps': [],
                'grad_norms': [],
                'guide_step_norms': [],
                'blend': [],
                't_start': int(t_start),
                'init_mode': init_mode,
                'implementation': self.implementation,
                'target_action': target.copy(),
            }

        for t in range(t_start, -1, -1):
            a_t = self._adpro_step(a_t, obs, t, debug=debug)
            if return_trajectory:
                trajectory.append(a_t.copy())

        if return_trajectory and return_debug:
            return a_t, trajectory, debug
        if return_trajectory:
            return a_t, trajectory
        if return_debug:
            return a_t, debug
        return a_t

    def evaluate_policy_improvement(self, obs: dict, gt_action: np.ndarray, n_steps_list: list = None) -> dict:
        if n_steps_list is None:
            n_steps_list = [5, 10, 20, 30, 50, 100]

        results = {'baseline_mse': [], 'adpro_mse': [], 'n_steps': n_steps_list}
        gt = gt_action[:self.action_dim]

        for n_steps in n_steps_list:
            self.rng = np.random.default_rng(42)
            self.policy.rng = self.rng
            a_base = self.policy.inference(obs, n_steps=n_steps)
            mse_base = float(np.mean((a_base - gt) ** 2))

            self.rng = np.random.default_rng(42)
            self.policy.rng = self.rng
            a_adpro = self.inference(obs, n_steps=n_steps)
            mse_adpro = float(np.mean((a_adpro - gt) ** 2))

            results['baseline_mse'].append(mse_base)
            results['adpro_mse'].append(mse_adpro)

        return results
