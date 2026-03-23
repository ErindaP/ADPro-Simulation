"""
simulate_realtime.py — Visualisation temps réel Panda (baseline vs ADPro)

Usage:
    /root/ENS/M2/ROBOT/.venv/bin/python simulate_realtime.py --no-train
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib.pyplot as plt

from env.panda_env import PandaPickPlaceEnv
from models.adpro import ADPro
from train_baseline import generate_expert_dataset, save_policy, load_policy
from models.diffusion_policy import train_diffusion_policy
from utils.geometry import forward_kinematics_chain


def get_or_train_policy(no_train: bool, seed: int):
    ckpt_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'baseline_policy.npz')
    if os.path.exists(ckpt_path):
        return load_policy(ckpt_path, seed=seed)
    if no_train:
        raise FileNotFoundError(
            f"Checkpoint introuvable: {ckpt_path}. Relance sans --no-train."
        )

    dataset = generate_expert_dataset(n_episodes=120, seed=seed)
    policy = train_diffusion_policy(dataset, n_epochs=70, lr=2e-3, T=100, seed=seed)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    save_policy(policy, ckpt_path)
    return policy


def _set_axes(ax):
    ax.set_xlim(0.2, 0.85)
    ax.set_ylim(-0.45, 0.45)
    ax.set_zlim(0.0, 0.75)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')


def _draw_robot(ax, env: PandaPickPlaceEnv, traj_xyz: list, title: str, color: str):
    ax.cla()
    _set_axes(ax)
    ax.set_title(title)

    # Table
    tx = np.array([0.2, 0.85, 0.85, 0.2, 0.2])
    ty = np.array([-0.45, -0.45, 0.45, 0.45, -0.45])
    tz = np.zeros_like(tx)
    ax.plot(tx, ty, tz, color='gray', alpha=0.7)

    # Chaîne cinématique
    chain = forward_kinematics_chain(env.q)
    joints = np.array([T[:3, 3] for T in chain])
    ax.plot(joints[:, 0], joints[:, 1], joints[:, 2], '-o', color='black', markersize=3, linewidth=1.5)

    # Objet
    ax.scatter(env.obj_pos[0], env.obj_pos[1], env.obj_pos[2], color='orange', s=100, marker='D', label='Objet')

    # End-effector + trajectoire
    ee = env.T_ee[:3, 3]
    if traj_xyz:
        t = np.array(traj_xyz)
        ax.plot(t[:, 0], t[:, 1], t[:, 2], color=color, linewidth=2.5, label='Trajectoire EE')
    ax.scatter(ee[0], ee[1], ee[2], color=color, s=80, marker='*', label='EE')

    dist = np.linalg.norm(ee - env.obj_pos)
    ax.text2D(0.02, 0.95, f"dist={dist:.3f} m", transform=ax.transAxes)
    ax.legend(loc='upper left', fontsize=8)


def _trajectory_xyz(action_trajectory: list) -> np.ndarray:
    return np.array([a[:3] for a in action_trajectory], dtype=np.float64)


def animate_denoising(
    obs: dict,
    policy,
    adpro: ADPro,
    baseline_steps: int,
    adpro_steps: int,
    frame_sleep: float,
    seed: int,
):
    """
    Anime pas à pas les trajectoires internes de dénoising:
    - baseline DDPM
    - ADPro (init + guidance + contrainte sphérique)
    """
    policy.rng = np.random.default_rng(seed)
    _, traj_base = policy.inference(obs, n_steps=baseline_steps, return_trajectory=True)

    adpro.rng = np.random.default_rng(seed)
    adpro.policy.rng = np.random.default_rng(seed)
    _, traj_adpro, dbg = adpro.inference(
        obs,
        n_steps=adpro_steps,
        return_trajectory=True,
        return_debug=True,
    )

    xyz_b = _trajectory_xyz(traj_base)
    xyz_a = _trajectory_xyz(traj_adpro)
    target = dbg['target_action'][:3]
    obj = obs['obj_pos']
    n_frames = max(len(xyz_b), len(xyz_a))
    print(n_frames, "frames à animer")  # Debug

    err_b = np.linalg.norm(xyz_b - target[None, :], axis=1)
    err_a = np.linalg.norm(xyz_a - target[None, :], axis=1)

    plt.ion()
    fig = plt.figure(figsize=(13, 6))
    ax3d = fig.add_subplot(1, 2, 1, projection='3d')
    ax2d = fig.add_subplot(1, 2, 2)

    for k in range(n_frames):
        kb = min(k + 1, len(xyz_b))
        ka = min(k + 1, len(xyz_a))

        ax3d.cla()
        ax3d.set_title("Denoising en espace action (position x,y,z)")
        ax3d.set_xlabel("x (m)")
        ax3d.set_ylabel("y (m)")
        ax3d.set_zlabel("z (m)")

        ax3d.plot(xyz_b[:kb, 0], xyz_b[:kb, 1], xyz_b[:kb, 2], color='tab:red', label='Baseline')
        ax3d.plot(xyz_a[:ka, 0], xyz_a[:ka, 1], xyz_a[:ka, 2], color='tab:green', label='ADPro')
        ax3d.scatter(target[0], target[1], target[2], color='tab:blue', marker='*', s=120, label='Target grasp')
        ax3d.scatter(xyz_a[0, 0], xyz_a[0, 1], xyz_a[0, 2], color='orange', marker='D', s=80, label='Init ADPro')
        ax3d.scatter(obj[0], obj[1], obj[2], color='black', marker='x', s=80, label='Objet')

        all_pts = np.vstack([xyz_b, xyz_a, target[None, :], obj[None, :]])
        margin = 0.05
        ax3d.set_xlim(all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
        ax3d.set_ylim(all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)
        ax3d.set_zlim(max(0.0, all_pts[:, 2].min() - margin), all_pts[:, 2].max() + margin)
        ax3d.legend(loc='upper left', fontsize=8)

        ax2d.cla()

        ax2d.set_title("Erreur vers la cible pendant le dénoising")
        ax2d.set_xlabel("Itération de dénoising")
        ax2d.set_ylabel("||a_t(xyz) - target||")
        ax2d.plot(np.arange(kb), err_b[:kb], color='tab:red', label='Baseline')
        ax2d.plot(np.arange(ka), err_a[:ka], color='tab:green', label='ADPro')
        ax2d.grid(alpha=0.3)
        ax2d.legend(loc='upper right')

        adpro_step = max(0, min(k, len(dbg['grad_norms']) - 1))
        g_norm = dbg['grad_norms'][adpro_step]
        s_norm = dbg['guide_step_norms'][adpro_step]
        blend = dbg['blend'][adpro_step]
        fig.suptitle(
            "Comparaison pas à pas du dénoising\n"
            f"ADPro init={dbg['init_mode']} (t_start={dbg['t_start']}) | "
            f"grad_norm={g_norm:.3f}, step_norm={s_norm:.3f}, blend={blend:.3f}"
        )
        plt.tight_layout()
        plt.pause(frame_sleep)

    plt.ioff()
    plt.show()


def run_live(
    seed: int,
    no_train: bool,
    baseline_steps: int,
    adpro_steps: int,
    max_env_steps: int,
    frame_sleep: float,
    show_denoise: bool,
    adpro_impl: str,
    adpro_spherical_scale: float,
    adpro_stochastic: bool,
):
    policy = get_or_train_policy(no_train=no_train, seed=seed)

    adpro = ADPro(
        base_policy=policy,
        eta=0.08,
        M=60,
        implementation=adpro_impl,
        use_fgr=True,
        use_task_manifold=True,
        use_spherical=True,
        spherical_scale=adpro_spherical_scale,
        deterministic_denoising=(not adpro_stochastic),
        n_fgr_points=256,
        n_fgr_iter=8,
    )

    env_baseline = PandaPickPlaceEnv(seed=seed)
    env_adpro = PandaPickPlaceEnv(seed=seed)

    # Même scène pour comparaison
    env_baseline.rng = np.random.default_rng(seed)
    obs_b = env_baseline.reset()
    obj_pos = obs_b['obj_pos'].copy()

    env_adpro.rng = np.random.default_rng(seed)
    obs_a = env_adpro.reset(obj_pos=obj_pos)

    if show_denoise:
        animate_denoising(
            obs=obs_a,
            policy=policy,
            adpro=adpro,
            baseline_steps=baseline_steps,
            adpro_steps=adpro_steps,
            frame_sleep=frame_sleep,
            seed=seed,
        )

    traj_b = [obs_b['ee_pos'].copy()]
    traj_a = [obs_a['ee_pos'].copy()]

    plt.ion()
    fig = plt.figure(figsize=(13, 6))
    ax_b = fig.add_subplot(1, 2, 1, projection='3d')
    ax_a = fig.add_subplot(1, 2, 2, projection='3d')

    done_b = False
    done_a = False

    for step in range(max_env_steps):
        if not done_b:
            action_b = policy.inference(obs_b, n_steps=baseline_steps)
            obs_b, _, done_b, info_b = env_baseline.step(action_b)
            traj_b.append(obs_b['ee_pos'].copy())
        else:
            info_b = {'success': True, 'dist_to_target': np.linalg.norm(env_baseline.T_ee[:3, 3] - env_baseline.obj_pos)}

        if not done_a:
            action_a = adpro.inference(obs_a, n_steps=adpro_steps)
            obs_a, _, done_a, info_a = env_adpro.step(action_a)
            traj_a.append(obs_a['ee_pos'].copy())
        else:
            info_a = {'success': True, 'dist_to_target': np.linalg.norm(env_adpro.T_ee[:3, 3] - env_adpro.obj_pos)}

        title_b = f"Baseline ({baseline_steps} denoise steps) | step env={step+1}"
        title_a = f"ADPro ({adpro_steps} denoise steps) | step env={step+1}"
        _draw_robot(ax_b, env_baseline, traj_b, title_b, color='tab:red')
        _draw_robot(ax_a, env_adpro, traj_a, title_a, color='tab:green')

        fig.suptitle(
            "Franka Panda Pick-and-Grasp en temps réel\n"
            f"Impl ADPro={adpro_impl} | Baseline success={info_b['success']} | ADPro success={info_a['success']}"
        )
        plt.tight_layout()
        plt.pause(frame_sleep)

        if done_b and done_a:
            break

    plt.ioff()
    plt.show()

    print("\nRésumé épisode:")
    print(f"  Baseline success: {done_b}")
    print(f"  ADPro success:    {done_a}")
    print(f"  Dist finale baseline: {np.linalg.norm(env_baseline.T_ee[:3, 3] - env_baseline.obj_pos):.4f} m")
    print(f"  Dist finale ADPro:    {np.linalg.norm(env_adpro.T_ee[:3, 3] - env_adpro.obj_pos):.4f} m")


def main():
    parser = argparse.ArgumentParser(description='Simulation Panda temps réel: Baseline vs ADPro')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-train', action='store_true')
    parser.add_argument('--baseline-steps', type=int, default=40)
    parser.add_argument('--adpro-steps', type=int, default=20)
    parser.add_argument('--max-env-steps', type=int, default=40)
    parser.add_argument('--frame-sleep', type=float, default=0.15)
    parser.add_argument('--show-denoise', action='store_true',
                        help='Afficher l évolution pas-à-pas du dénoising (a_t -> a_0).')
    parser.add_argument('--adpro-impl', type=str, default='practical', choices=['practical', 'paper'],
                        help='Version ADPro: practical (ancienne) ou paper (fidèle papier).')
    parser.add_argument('--adpro-spherical-scale', type=float, default=0.25,
                        help='Facteur de pas pour la contrainte sphérique ADPro.')
    parser.add_argument('--adpro-stochastic', action='store_true',
                        help='Active le bruit stochastique pendant le reverse process ADPro.')
    args = parser.parse_args()

    run_live(
        seed=args.seed,
        no_train=args.no_train,
        baseline_steps=args.baseline_steps,
        adpro_steps=args.adpro_steps,
        max_env_steps=args.max_env_steps,
        frame_sleep=args.frame_sleep,
        show_denoise=args.show_denoise,
        adpro_impl=args.adpro_impl,
        adpro_spherical_scale=args.adpro_spherical_scale,
        adpro_stochastic=args.adpro_stochastic,
    )


if __name__ == '__main__':
    main()
