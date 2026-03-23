"""
train_baseline.py — Entraînement de la politique de diffusion baseline

Génère des données expertes pour le Franka Panda sur une tâche pick-and-place
et entraîne la politique de diffusion vanilla par behavior cloning.

Utilisation :
    python train_baseline.py

Le modèle entraîné est sauvegardé dans checkpoints/baseline_policy.npz
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pickle
from tqdm import tqdm

from env.panda_env import PandaPickPlaceEnv
from models.diffusion_policy import DiffusionPolicy, NoisePredictionNetwork, train_diffusion_policy


def generate_expert_dataset(n_episodes: int = 200, seed: int = 42):
    """
    Génère un dataset d'expert pour le Franka Panda pick-and-place.

    Chaque épisode : le robot se déplace de sa configuration
    initiale vers la pose de saisie avec des actions perturbées.
    """
    print(f"Génération du dataset expert ({n_episodes} épisodes)...")

    env = PandaPickPlaceEnv(seed=seed)
    dataset = []

    for ep in tqdm(range(n_episodes), desc="Episodes"):
        env.rng = np.random.default_rng(seed + ep)
        obs = env.reset()

        # Simulation de trajectoire d'expert vers la pose de saisie
        for step in range(8):
            # Action experte : se diriger vers la cible avec du bruit décroissant
            noise_scale = 0.02 * (1.0 - step / 8)
            action = env.get_expert_action(noise_scale=noise_scale)

            # Ne garder que les 7 premières composantes (pose SE3)
            action_7d = action[:7]

            dataset.append({
                'obs': {
                    'pc_gripper': obs['pc_gripper'].copy(),
                    'pc_scene':   obs['pc_scene'].copy(),
                },
                'action': action_7d.copy(),
            })

            next_obs, reward, done, info = env.step(action)
            obs = next_obs
            if done:
                break

    print(f"  Dataset généré : {len(dataset)} échantillons")
    return dataset


def save_policy(policy: DiffusionPolicy, path: str):
    """Sauvegarde les paramètres du réseau dans un fichier .npz."""
    parameters = []
    for p in policy.network.params():
        parameters.append(p)

    arrays = {f'param_{i}': p for i, p in enumerate(parameters)}
    arrays['action_dim'] = np.array([policy.action_dim])
    arrays['T'] = np.array([policy.T])
    np.savez(path, **arrays)
    print(f"  Politique sauvegardée dans : {path}")


def load_policy(path: str, seed: int = 0) -> DiffusionPolicy:
    """Charge une politique depuis un fichier .npz."""
    data = np.load(path)
    action_dim = int(data['action_dim'][0])
    T = int(data['T'][0])

    rng = np.random.default_rng(seed)
    policy = DiffusionPolicy(T=T, rng=rng)

    # Restaurer les paramètres
    all_params = policy.network.params()
    for i, p in enumerate(all_params):
        key = f'param_{i}'
        if key in data:
            p[:] = data[key]

    print(f"  Politique chargée depuis : {path}")
    return policy


def main():
    print("=" * 60)
    print("ADPro — Entraînement de la politique de diffusion baseline")
    print("Robot : Franka Panda Emika (7 DOF)")
    print("Tâche : Pick-and-Place")
    print("=" * 60)
    print()

    # === 1. Générer le dataset d'expert ===
    dataset = generate_expert_dataset(n_episodes=150, seed=42)

    # === 2. Entraîner la politique ===
    print("\nEntraînement...")
    policy = train_diffusion_policy(
        dataset,
        n_epochs=80,
        lr=2e-3,
        T=100,
        seed=0,
        verbose=True,
    )

    # === 3. Sauvegarder ===
    os.makedirs("checkpoints", exist_ok=True)
    save_policy(policy, "checkpoints/baseline_policy.npz")

    # Sauvegarder aussi le dataset pour les expériences
    with open("checkpoints/expert_dataset.pkl", "wb") as f:
        pickle.dump(dataset[:50], f)  # Garder 50 exemples pour l'éval
    print("  Dataset d'évaluation sauvegardé dans checkpoints/expert_dataset.pkl")

    # === 4. Évaluation rapide ===
    print("\nÉvaluation rapide...")
    env = PandaPickPlaceEnv(seed=99)
    n_success = 0
    n_eval = 20

    for i in range(n_eval):
        env.rng = np.random.default_rng(100 + i)
        obs = env.reset()
        action = policy.inference(obs, n_steps=100)
        _, _, done, info = env.step(action)
        if info['success']:
            n_success += 1

    print(f"\nRésultats baseline :")
    print(f"  Taux de succès : {n_success}/{n_eval} = {100*n_success/n_eval:.1f}%")
    print(f"  (Avec T=100 étapes DDPM)")
    print()
    print("Entraînement terminé ! Lancez demo.py pour comparer ADPro vs baseline.")


if __name__ == "__main__":
    main()
