"""
Visualisation offline de la trajectoire qu'un robot prendrait en suivant la politique.

Ce script ne commande pas le robot réel.
Il exécute un rollout dans PandaPickPlaceEnv avec:
- ADPro (par défaut) ou baseline,
- position objet venant de real_robot/config/lab_franka.json (gello.obj_pos),
- sauvegarde d'une figure 3D de la trajectoire EE.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from env.panda_env import PandaPickPlaceEnv
from models.adpro import ADPro
from train_baseline import load_policy


def main():
    parser = argparse.ArgumentParser(description="Offline rollout visualization (no real robot command).")
    parser.add_argument("--config", default="real_robot/config/lab_franka.json")
    parser.add_argument("--env-steps", type=int, default=20)
    parser.add_argument("--denoise-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline", action="store_true", help="Utiliser baseline au lieu d'ADPro.")
    parser.add_argument(
        "--stop-on-success",
        action="store_true",
        help="Arrêter la rollout dès que la cible est atteinte.",
    )
    parser.add_argument("--out", default="results/offline_rollout_trajectory.png")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    policy_cfg = cfg["policy"]
    gello_cfg = cfg.get("gello", {})
    obj_pos = np.asarray(gello_cfg.get("obj_pos", [0.55, 0.0, 0.05]), dtype=np.float64)
    ckpt = policy_cfg["checkpoint_path"]

    policy = load_policy(ckpt, seed=int(policy_cfg.get("seed", args.seed)))
    adpro = ADPro(
        base_policy=policy,
        eta=0.08,
        M=int(policy_cfg.get("adpro_M", 60)),
        implementation=policy_cfg.get("adpro_impl", "paper"),
        use_fgr=True,
        use_task_manifold=True,
        use_spherical=True,
        spherical_scale=float(policy_cfg.get("adpro_spherical_scale", 0.2)),
        deterministic_denoising=(not bool(policy_cfg.get("adpro_stochastic", False))),
        n_fgr_points=256,
        n_fgr_iter=8,
    )

    denoise_steps = args.denoise_steps
    if denoise_steps is None:
        denoise_steps = int(policy_cfg.get("adpro_steps", 20))

    env = PandaPickPlaceEnv(seed=args.seed)
    env.rng = np.random.default_rng(args.seed)
    obs = env.reset(obj_pos=obj_pos)

    traj = [obs["ee_pos"].copy()]
    info = {"success": False, "dist_to_target": np.linalg.norm(obs["ee_pos"] - obj_pos)}

    for _ in range(args.env_steps):
        if args.baseline:
            action = policy.inference(obs, n_steps=denoise_steps)
        else:
            action = adpro.inference(obs, n_steps=denoise_steps)
        obs, _, done, info = env.step(action)
        traj.append(obs["ee_pos"].copy())
        # PandaPickPlaceEnv coupe par défaut à 20 steps.
        # Pour visualisation longue, on ignore cette coupure, sauf succès si demandé.
        if bool(info.get("success", False)) and args.stop_on_success:
            break

    P = np.asarray(traj, dtype=np.float64)
    grasp_target = obj_pos + np.array([0.0, 0.0, 0.05], dtype=np.float64)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(P[:, 0], P[:, 1], P[:, 2], "b-", linewidth=2.0, label="EE trajectory (sim)")
    ax.scatter(P[0, 0], P[0, 1], P[0, 2], c="green", s=110, marker="o", label="EE start")
    ax.scatter(P[-1, 0], P[-1, 1], P[-1, 2], c="red", s=110, marker="o", label="EE end")
    ax.scatter(obj_pos[0], obj_pos[1], obj_pos[2], c="orange", s=130, marker="D", label="Object")
    ax.scatter(
        grasp_target[0],
        grasp_target[1],
        grasp_target[2],
        c="black",
        s=130,
        marker="*",
        label="Grasp target",
    )

    mode = "Baseline" if args.baseline else "ADPro"
    ax.set_title(f"Offline rollout ({mode}) | env_steps={len(P)-1}, denoise_steps={denoise_steps}")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    ax.text(obj_pos[0], obj_pos[1], obj_pos[2] + 0.01, f"obj {np.round(obj_pos, 3)}", color="darkorange")
    ax.text(P[0, 0], P[0, 1], P[0, 2] + 0.01, f"start {np.round(P[0], 3)}", color="green")
    ax.text(P[-1, 0], P[-1, 1], P[-1, 2] + 0.01, f"end {np.round(P[-1], 3)}", color="darkred")

    plt.tight_layout()
    plt.savefig(args.out, dpi=180)
    plt.close(fig)

    d0 = float(np.linalg.norm(P[0] - grasp_target))
    d1 = float(np.linalg.norm(P[-1] - grasp_target))
    print(f"[ok] Figure sauvée: {args.out}")
    print(f"[info] mode={mode} obj_pos={obj_pos} grasp_target={grasp_target}")
    print(f"[info] dist_to_grasp start={d0:.4f} end={d1:.4f} success={info.get('success', False)}")


if __name__ == "__main__":
    main()
