"""
demo.py — Démonstration comparative : Politique de Diffusion Vanilla vs ADPro

Simule le Franka Panda sur une tâche de pick-and-place et compare :
1. La convergence (MSE vs nombre d'étapes de diffusion)
2. Les trajectoires 3D en espace cartésien
3. Le backtracking sur les composantes d'action (x, y, z, qx, qy, qz, qw)
4. Le taux de succès vs nombre d'étapes

Utilisation :
    python demo.py [--no-train]

Résultats sauvegardés dans results/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

from env.panda_env import PandaPickPlaceEnv
from models.diffusion_policy import (
    DiffusionPolicy, NoisePredictionNetwork, train_diffusion_policy
)
from models.adpro import ADPro
from train_baseline import generate_expert_dataset, save_policy, load_policy

# Style matplotlib
plt.rcParams.update({
    'figure.dpi': 150,
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'lines.linewidth': 2.0,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

COLORS = {
    'baseline': '#E74C3C',   # Rouge
    'adpro':    '#2ECC71',   # Vert
    'gt':       '#3498DB',   # Bleu
    'adpro_fgr': '#F39C12',  # Orange (FGR seul)
}


# ============================================================
# Utilitaires
# ============================================================

def get_or_train_policy(force_retrain: bool = False, seed: int = 42) -> DiffusionPolicy:
    """Charge ou entraîne la politique baseline."""
    ckpt_path = "checkpoints/baseline_policy.npz"
    if not force_retrain and os.path.exists(ckpt_path):
        print(f"Chargement de la politique pré-entraînée depuis {ckpt_path}")
        return load_policy(ckpt_path, seed=seed)

    print("Entraînement de la politique baseline...")
    dataset = generate_expert_dataset(n_episodes=120, seed=42)
    policy = train_diffusion_policy(dataset, n_epochs=60, lr=2e-3, T=100, seed=seed)
    os.makedirs("checkpoints", exist_ok=True)
    save_policy(policy, ckpt_path)
    return policy


def evaluate_n_steps(
    env: PandaPickPlaceEnv,
    policy: DiffusionPolicy,
    adpro: ADPro,
    n_steps_list: list,
    n_eval: int = 15,
    seed: int = 200,
) -> dict:
    """
    Évalue le taux de succès de baseline et ADPro pour différents n_steps.
    """
    results = {'n_steps': n_steps_list, 'baseline': [], 'adpro': []}

    max_env_steps = 12

    for n_steps in tqdm(n_steps_list, desc="Eval n_steps"):
        base_successes = 0
        adpro_successes = 0

        for i in range(n_eval):
            # Baseline
            env.rng = np.random.default_rng(seed + i)
            policy.rng = env.rng
            obs = env.reset()
            info_b = {'success': False}
            for _ in range(max_env_steps):
                action_b = policy.inference(obs, n_steps=n_steps)
                obs, _, done_b, info_b = env.step(action_b)
                if done_b:
                    break
            if info_b['success']:
                base_successes += 1

            # ADPro
            env.rng = np.random.default_rng(seed + i)
            adpro.rng = env.rng
            adpro.policy.rng = env.rng
            obs = env.reset()
            info_a = {'success': False}
            for _ in range(max_env_steps):
                action_a = adpro.inference(obs, n_steps=n_steps)
                obs, _, done_a, info_a = env.step(action_a)
                if done_a:
                    break
            if info_a['success']:
                adpro_successes += 1

        results['baseline'].append(100 * base_successes / n_eval)
        results['adpro'].append(100 * adpro_successes / n_eval)

    return results


def get_denoising_trajectories(
    obs: dict,
    policy: DiffusionPolicy,
    adpro: ADPro,
    gt_action: np.ndarray,
    n_steps: int = 100,
    seed: int = 42,
) -> dict:
    """Collecte les trajectoires complètes de dénoising des deux méthodes."""
    # Baseline
    policy.rng = np.random.default_rng(seed)
    _, traj_base = policy.inference(obs, n_steps=n_steps, return_trajectory=True)

    # ADPro
    adpro.rng = np.random.default_rng(seed)
    adpro.policy.rng = np.random.default_rng(seed)
    _, traj_adpro = adpro.inference(obs, n_steps=n_steps, return_trajectory=True)

    traj_base  = np.array(traj_base)   # (T+1, action_dim)
    traj_adpro = np.array(traj_adpro)  # (M+1, action_dim)

    gt = gt_action[:policy.action_dim]

    # Calculer MSE à chaque étape (en inversant l'axe temps)
    mse_base  = np.mean((traj_base  - gt) ** 2, axis=1)
    mse_adpro = np.mean((traj_adpro - gt) ** 2, axis=1)

    return {
        'traj_baseline':  traj_base,
        'traj_adpro':     traj_adpro,
        'mse_baseline':   mse_base,
        'mse_adpro':      mse_adpro,
        'gt_action':      gt,
    }


# ============================================================
# Figures
# ============================================================

def plot_convergence(traj_data: dict, save_path: str):
    """Fig 1 : Courbes de convergence MSE durant le dénoising."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "ADPro vs Baseline — Convergence du processus de dénoising\n"
        "Robot : Franka Panda Emika — Tâche : Pick-and-Place",
        fontsize=14, fontweight='bold'
    )

    mse_b = traj_data['mse_baseline']
    mse_a = traj_data['mse_adpro']

    # === Axe 1 : MSE durant le dénoising (ordre chronologique : T→0) ===
    ax = axes[0]
    steps_b = np.linspace(0, 1, len(mse_b))
    steps_a = np.linspace(0, 1, len(mse_a))
    ax.plot(steps_b, mse_b, color=COLORS['baseline'], label='Diffusion Baseline', alpha=0.85)
    ax.plot(steps_a, mse_a, color=COLORS['adpro'],    label='ADPro (le nôtre)',   alpha=0.85)
    ax.set_xlabel("Progression du dénoising (a_T → a_0)", fontsize=11)
    ax.set_ylabel("MSE avec l'action experte", fontsize=11)
    ax.set_title("Convergence durant le dénoising")
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)

    # Annotation de la convergence finale
    ax.annotate(
        f'MSE finale\nBaseline: {mse_b[-1]:.3f}\nADPro: {mse_a[-1]:.3f}',
        xy=(1.0, mse_b[-1]), xytext=(0.75, max(mse_b) * 0.6),
        arrowprops=dict(arrowstyle='->', color='gray'),
        fontsize=9, color='gray',
    )

    # === Axe 2 : Backtracking — oscillations de la composante x ===
    ax = axes[1]
    comp_labels = ['x', 'y', 'z', 'q_x', 'q_y', 'q_z', 'q_w']
    # Composante x (translation)
    comp_b = traj_data['traj_baseline'][:, 0]
    comp_a = traj_data['traj_adpro'][:, 0]
    gt_x   = traj_data['gt_action'][0]

    t_b = np.arange(len(comp_b))
    t_a = np.arange(len(comp_a))
    ax.plot(t_b, comp_b, color=COLORS['baseline'], label='Baseline (x)', alpha=0.85)
    ax.plot(t_a, comp_a, color=COLORS['adpro'],    label='ADPro (x)',    alpha=0.85)
    ax.axhline(gt_x, color=COLORS['gt'], linestyle='--', alpha=0.7, label='Expert (x)')
    ax.set_xlabel("Étape de dénoising (a_T → a_0)")
    ax.set_ylabel("Valeur de la composante x (m)")
    ax.set_title("Backtracking — composante de translation x\n(oscillations = instabilité)")
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Figure sauvegardée : {save_path}")


def plot_action_components(traj_data: dict, save_path: str):
    """
    Fig 2 : Comparaison de toutes les composantes de l'action (x,y,z,qx,qy,qz,qw).
    Reproduit la Fig. 4 du papier ADPro.
    """
    comp_names = ['x (m)', 'y (m)', 'z (m)', 'q_x', 'q_y', 'q_z', 'q_w']
    traj_b = traj_data['traj_baseline']
    traj_a = traj_data['traj_adpro']
    gt     = traj_data['gt_action']
    n_comp = min(7, traj_b.shape[1])

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    fig.suptitle(
        "Comparaison des composantes d'action — Baseline vs ADPro\n"
        "(Reproduction Fig. 4 du papier — Franka Panda 7-DOF)",
        fontsize=14, fontweight='bold'
    )

    for i in range(n_comp):
        ax = axes[i]
        t_b = np.arange(len(traj_b))
        t_a = np.arange(len(traj_a))

        ax.plot(t_b, traj_b[:, i], color=COLORS['baseline'],
                label='Baseline', alpha=0.85, linewidth=1.8)
        ax.plot(t_a, traj_a[:, i], color=COLORS['adpro'],
                label='ADPro',    alpha=0.85, linewidth=1.8)
        ax.axhline(gt[i], color=COLORS['gt'], linestyle='--',
                   alpha=0.7, label='Expert', linewidth=1.5)

        ax.set_title(f"Composante {comp_names[i]}")
        ax.set_xlabel("Étape de dénoising")
        ax.set_ylabel("Valeur")
        if i == 0:
            ax.legend(fontsize=9, loc='upper right')

        # Mettre en évidence le backtracking (changements de direction)
        d_b = np.diff(traj_b[:, i])
        sign_changes = np.where(np.diff(np.sign(d_b)) != 0)[0]
        for sc in sign_changes[:3]:  # Annoter les 3 premiers
            ax.axvline(sc + 1, color=COLORS['baseline'], alpha=0.15, linewidth=0.8)

    # Enlever le dernier subplot (vide)
    axes[-1].set_visible(False)

    # Légende globale
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc='lower right', fontsize=10, ncol=3,
    )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Figure sauvegardée : {save_path}")


def plot_success_vs_steps(results: dict, save_path: str):
    """Fig 3 : Taux de succès vs nombre d'étapes de diffusion."""
    fig, ax = plt.subplots(figsize=(9, 6))

    n_steps = results['n_steps']
    base_sr = results['baseline']
    adpro_sr = results['adpro']

    ax.plot(n_steps, base_sr, 'o-', color=COLORS['baseline'],
            label='Diffusion Baseline (DDPM)', markersize=7, alpha=0.9)
    ax.plot(n_steps, adpro_sr, 's-', color=COLORS['adpro'],
            label='ADPro (manifold + FGR)', markersize=7, alpha=0.9)

    # Ligne horizontale : niveau 70%
    ax.axhline(70, color='gray', linestyle=':', alpha=0.6, label='Seuil 70%')

    # Mettre en evidence le gain d'efficacité
    # Trouver les n_steps pour atteindre 70%
    base_70 = next((n_steps[i] for i, v in enumerate(base_sr) if v >= 70), None)
    adpro_70 = next((n_steps[i] for i, v in enumerate(adpro_sr) if v >= 70), None)
    if base_70 and adpro_70 and base_70 != adpro_70:
        ax.annotate(
            f'ADPro atteint 70% en {adpro_70} steps\nvs {base_70} steps (baseline)',
            xy=(adpro_70, 70), xytext=(adpro_70 + 5, 55),
            arrowprops=dict(arrowstyle='->', color=COLORS['adpro']),
            fontsize=10, color=COLORS['adpro'],
        )

    ax.set_xlabel("Nombre d'étapes de dénoising (DDPM steps)", fontsize=12)
    ax.set_ylabel("Taux de succès (%)", fontsize=12)
    ax.set_title(
        "Taux de succès vs Nombre d'étapes de diffusion\n"
        "Franka Panda — Tâche Pick-and-Place",
        fontsize=13, fontweight='bold'
    )
    ax.legend(fontsize=11)
    ax.set_ylim(-5, 105)
    ax.set_xlim(min(n_steps) - 2, max(n_steps) + 2)

    # Réduction de steps annotée
    ax.fill_betweenx([0, 105], 0, min(n_steps), alpha=0.05, color='green',
                      label='Zone faibles steps')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Figure sauvegardée : {save_path}")


def plot_3d_trajectories(
    env: PandaPickPlaceEnv,
    policy: DiffusionPolicy,
    adpro: ADPro,
    seed: int = 42,
    save_path: str = "results/trajectories_3d.png",
):
    """
    Fig 4 : Trajectoires 3D du gripper Franka Panda durant l'exécution.
    Montre le chemin effectif de l'end-effector en 3D.
    """
    def run_episode(use_adpro: bool, n_steps: int = 50) -> list:
        """Exécute un épisode complet et retourne la trajectoire EE."""
        env.rng = np.random.default_rng(seed)
        if use_adpro:
            adpro.rng = np.random.default_rng(seed)
            adpro.policy.rng = np.random.default_rng(seed)
        else:
            policy.rng = np.random.default_rng(seed)

        obs = env.reset()
        obj_pos = obs['obj_pos'].copy()
        traj = [obs['ee_pos'].copy()]

        for _ in range(8):
            if use_adpro:
                action = adpro.inference(obs, n_steps=n_steps)
            else:
                action = policy.inference(obs, n_steps=n_steps)

            obs, _, done, info = env.step(action)
            traj.append(obs['ee_pos'].copy())
            if done:
                break

        return np.array(traj), obj_pos

    traj_base, obj_pos = run_episode(False, n_steps=40)
    traj_adpro, _      = run_episode(True,  n_steps=20)  # ADPro avec moins de steps

    fig = plt.figure(figsize=(15, 6))
    fig.suptitle(
        "Trajectoires 3D du End-Effector Franka Panda\n"
        "Baseline (40 steps) vs ADPro (20 steps)",
        fontsize=14, fontweight='bold'
    )

    for idx, (traj, label, color, n_steps_label) in enumerate([
        (traj_base,  'Diffusion Baseline', COLORS['baseline'], '40 steps'),
        (traj_adpro, 'ADPro',              COLORS['adpro'],    '20 steps'),
    ]):
        ax = fig.add_subplot(1, 2, idx + 1, projection='3d')

        # Trajectoire
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                '-o', color=color, markersize=6, linewidth=2,
                label=f'{label} ({n_steps_label})')

        # Point de départ et d'arrivée
        ax.scatter(*traj[0],  color='purple', s=80, zorder=5, label='Départ')
        ax.scatter(*traj[-1], color='black',  s=80, zorder=5, marker='*', label='Arrivée')

        # Objet cible
        ax.scatter(*obj_pos, color='orange', s=150, zorder=5, marker='D', label='Objet cible')

        # Zone de workspace Franka
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'{label}\n(Franka Panda — {n_steps_label} de diffusion)')
        ax.legend(fontsize=8, loc='upper left')

        # Définir les mêmes axes pour comparaison
        margin = 0.1
        all_pts = np.vstack([traj, obj_pos.reshape(1, 3)])
        for dim, (getter, setter) in enumerate([
            (ax.get_xlim, ax.set_xlim),
            (ax.get_ylim, ax.set_ylim),
            (ax.get_zlim, ax.set_zlim),
        ]):
            lo = all_pts[:, dim].min() - margin
            hi = all_pts[:, dim].max() + margin
            setter((lo, hi))

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Figure sauvegardée : {save_path}")


def plot_mse_diffusion_steps(
    obs: dict,
    policy: DiffusionPolicy,
    adpro: ADPro,
    gt_action: np.ndarray,
    n_denoising_list: list,
    save_path: str,
    n_eval: int = 10,
):
    """
    Fig 5 : MSE finale vs nombre de steps de diffusion.
    Reproduit la Fig. 6 gauche du papier ADPro.
    """
    mse_baseline = []
    mse_adpro    = []

    for n_steps in tqdm(n_denoising_list, desc="MSE vs steps"):
        mse_b_list = []
        mse_a_list = []
        gt = gt_action[:policy.action_dim]

        for trial in range(n_eval):
            seed_t = 300 + trial

            # Baseline
            policy.rng = np.random.default_rng(seed_t)
            a_b = policy.inference(obs, n_steps=n_steps)
            mse_b_list.append(np.mean((a_b - gt) ** 2))

            # ADPro
            adpro.rng = np.random.default_rng(seed_t)
            adpro.policy.rng = np.random.default_rng(seed_t)
            a_a = adpro.inference(obs, n_steps=n_steps)
            mse_a_list.append(np.mean((a_a - gt) ** 2))

        mse_baseline.append(np.mean(mse_b_list))
        mse_adpro.append(np.mean(mse_a_list))

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.semilogy(n_denoising_list, mse_baseline, 'o-', color=COLORS['baseline'],
                label='Diffusion Baseline', markersize=7, alpha=0.9)
    ax.semilogy(n_denoising_list, mse_adpro, 's-', color=COLORS['adpro'],
                label='ADPro (le nôtre)', markersize=7, alpha=0.9)

    # Mettre en évidence le gain pour 20 steps
    if 20 in n_denoising_list:
        idx_20 = n_denoising_list.index(20)
        gain = mse_baseline[idx_20] / (mse_adpro[idx_20] + 1e-10)
        ax.annotate(
            f'×{gain:.1f} meilleur\nà 20 steps',
            xy=(20, mse_adpro[idx_20]),
            xytext=(25, mse_adpro[idx_20] * 3),
            arrowprops=dict(arrowstyle='->', color=COLORS['adpro']),
            color=COLORS['adpro'], fontsize=10,
        )

    ax.set_xlabel("Nombre d'étapes de dénoising", fontsize=12)
    ax.set_ylabel("MSE (log scale) avec l'action experte", fontsize=12)
    ax.set_title(
        "MSE Finale vs Nombre d'Étapes de Dénoising\n"
        "(Reproduction Fig. 6 du papier ADPro)",
        fontsize=13, fontweight='bold'
    )
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Figure sauvegardée : {save_path}")


# ============================================================
# Script principal
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Démonstration ADPro vs Baseline')
    parser.add_argument('--no-train', action='store_true',
                        help='Ne pas réentraîner, charger les checkpoints existants')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-eval', type=int, default=15,
                        help='Nombre d\'évaluations par configuration')
    parser.add_argument('--adpro-impl', type=str, default='practical',
                        choices=['practical', 'paper'],
                        help='Version ADPro: practical (ancienne) ou paper (fidèle papier).')
    args = parser.parse_args()

    print("=" * 65)
    print("ADPro — Démonstration Numérique")
    print("Robot : Franka Panda Emika (7 DOF, paramètres DH réels)")
    print("Tâche : Pick-and-Place 3D")
    print("=" * 65)
    print()

    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # =========================================================
    # 1. Politique baseline
    # =========================================================
    force_retrain = not args.no_train
    # Si checkpoint disponible, on charge; sinon on entraîne
    policy = get_or_train_policy(force_retrain=force_retrain, seed=args.seed)

    # =========================================================
    # 2. Créer ADPro (plug-and-play sur la baseline)
    # =========================================================
    print("\nConfiguration d'ADPro...")
    adpro = ADPro(
        base_policy=policy,
        eta=0.08,
        M=60,               # FGR init à partir de l'étape 60 (sur 100)
        implementation=args.adpro_impl,
        use_fgr=True,
        use_task_manifold=True,
        use_spherical=True,
        n_fgr_points=256,   # Rapidité pour la démo
        n_fgr_iter=8,
    )
    print("  ✓ ADPro configuré avec :")
    print(f"    - Implémentation : {args.adpro_impl}")
    print("    - Guidance task manifold (Eq. 8)")
    print("    - Contrainte sphérique de Gauss (Eq. 9)")
    print("    - Initialisation FGR à M=60 (Eq. 10)")
    print()

    # =========================================================
    # 3. Environnement et scénario de test
    # =========================================================
    env = PandaPickPlaceEnv(seed=args.seed)
    env.rng = np.random.default_rng(args.seed)
    obs = env.reset()
    gt_action = env.get_expert_action(noise_scale=0.0)

    print(f"Scénario de test :")
    print(f"  Position EE initiale : {obs['ee_pos'].round(3)} m")
    print(f"  Position objet cible : {obs['obj_pos'].round(3)} m")
    print(f"  Distance initiale    : {np.linalg.norm(obs['ee_pos'] - obs['obj_pos']):.3f} m")
    print()

    # =========================================================
    # 4. Fig 1 : Convergence et bactracking durant le dénoising
    # =========================================================
    print("Figure 1/5 : Convergence du processus de dénoising...")
    traj_data = get_denoising_trajectories(obs, policy, adpro, gt_action, n_steps=100)
    plot_convergence(traj_data, "results/fig1_convergence.png")

    # =========================================================
    # 5. Fig 2 : Toutes les composantes d'action (Fig. 4 du papier)
    # =========================================================
    print("Figure 2/5 : Composantes d'action (backtracking)...")
    plot_action_components(traj_data, "results/fig2_action_components.png")

    # =========================================================
    # 6. Fig 3 : Trajectoires 3D de l'end-effector
    # =========================================================
    print("Figure 3/5 : Trajectoires 3D du gripper Franka...")
    plot_3d_trajectories(env, policy, adpro, seed=args.seed,
                         save_path="results/fig3_trajectories_3d.png")

    # =========================================================
    # 7. Fig 4 : MSE finale vs nombre de steps (Fig. 6 papier)
    # =========================================================
    print("Figure 4/5 : MSE vs nombre de steps de diffusion...")
    n_denoising_list = [5, 10, 15, 20, 30, 50, 75, 100]
    plot_mse_diffusion_steps(
        obs, policy, adpro, gt_action,
        n_denoising_list=n_denoising_list,
        save_path="results/fig4_mse_vs_steps.png",
        n_eval=args.n_eval,
    )

    # =========================================================
    # 8. Fig 5 : Taux de succès vs nombre de steps
    # =========================================================
    print("Figure 5/5 : Taux de succès vs nombre de steps...")
    n_steps_list = [5, 10, 20, 30, 50, 75, 100]
    success_results = evaluate_n_steps(
        env, policy, adpro,
        n_steps_list=n_steps_list,
        n_eval=args.n_eval,
        seed=200,
    )
    plot_success_vs_steps(success_results, "results/fig5_success_vs_steps.png")

    # =========================================================
    # 9. Résumé
    # =========================================================
    print()
    print("=" * 65)
    print("RÉSULTATS — ADPro vs Baseline")
    print("=" * 65)
    print(f"{'N. Steps':>12} | {'Baseline SR':>12} | {'ADPro SR':>12} | {'Gain':>8}")
    print("-" * 55)
    for i, n in enumerate(success_results['n_steps']):
        b = success_results['baseline'][i]
        a = success_results['adpro'][i]
        gain = a - b
        print(f"{n:>12} | {b:>11.1f}% | {a:>11.1f}% | {gain:>+7.1f}%")

    # Backtracking
    mse_b = traj_data['mse_baseline']
    mse_a = traj_data['mse_adpro']
    backtracking_b = np.sum(np.abs(np.diff(mse_b)))
    backtracking_a = np.sum(np.abs(np.diff(mse_a)))
    print()
    print(f"Indice de backtracking (variation totale de MSE) :")
    print(f"  Baseline : {backtracking_b:.4f}")
    print(f"  ADPro    : {backtracking_a:.4f}")
    if backtracking_b > 0:
        print(f"  Réduction : {100*(1-backtracking_a/backtracking_b):.1f}%")
    print()
    print("Figures sauvegardées dans results/")
    print("  fig1_convergence.png")
    print("  fig2_action_components.png")
    print("  fig3_trajectories_3d.png")
    print("  fig4_mse_vs_steps.png")
    print("  fig5_success_vs_steps.png")
    print()
    print("✓ Démonstration ADPro terminée !")


if __name__ == "__main__":
    main()
