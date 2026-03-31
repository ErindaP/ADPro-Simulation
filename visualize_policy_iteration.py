"""
Visualisation offline d'une longue itération ADPro (sans commande robot).

Le script:
1) Lit la config real_robot/config/lab_franka.json (checkpoint + obj_pos).
2) Récupère la pose courante du robot via ZMQ (optionnel).
3) Construit une observation synthétique (pc_gripper + pc_scene).
4) Lance ADPro avec return_trajectory=True.
5) Sauvegarde une figure 3D avec:
   - position initiale EE,
   - trajectoire de dénoising (positions x,y,z),
   - position objet,
   - cible grasp (obj + [0,0,0.05]).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from models.adpro import ADPro
from train_baseline import load_policy
from utils.geometry import FRANKA_HOME_CONFIG, forward_kinematics
from utils.point_cloud import sample_gripper_pointcloud, sample_scene_pointcloud


def _try_get_current_ee_from_zmq(host: str, robot_port: int) -> np.ndarray | None:
    try:
        from gello.zmq_core.robot_node import ZMQClientRobot
    except Exception:
        return None

    try:
        c = ZMQClientRobot(port=robot_port, host=host)
        q = np.asarray(c.get_joint_state(), dtype=np.float64)[:7]
        return forward_kinematics(q)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Offline visualization of a long ADPro iteration.")
    parser.add_argument("--config", default="real_robot/config/lab_franka.json")
    parser.add_argument("--steps", type=int, default=120, help="Nombre de pas de dénoising ADPro.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--robot-port", type=int, default=6001)
    parser.add_argument("--out", type=str, default="results/policy_long_iteration.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--from-zmq",
        action="store_true",
        help="Lire la pose courante du robot via ZMQ (sinon départ FRANKA_HOME_CONFIG).",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    policy_cfg = cfg["policy"]
    gello_cfg = cfg.get("gello", {})
    obj_pos = np.asarray(gello_cfg.get("obj_pos", [-0.52, -0.16, 0.16]), dtype=np.float64)

    checkpoint_path = policy_cfg["checkpoint_path"]
    policy = load_policy(checkpoint_path, seed=int(policy_cfg.get("seed", args.seed)))
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

    used_zmq = False
    T_ee = None
    if args.from_zmq:
        T_ee = _try_get_current_ee_from_zmq(args.host, args.robot_port)
        used_zmq = T_ee is not None
    if T_ee is None:
        T_ee = forward_kinematics(FRANKA_HOME_CONFIG)

    ee_pos0 = T_ee[:3, 3].copy()
    pc_gripper = sample_gripper_pointcloud(
        T_ee,
        gripper_width=0.05,
        n_points=512,
        rng=np.random.default_rng(args.seed),
    )
    pc_scene = sample_scene_pointcloud(
        obj_pos,
        table_height=float(gello_cfg.get("table_height", 0.0)),
        n_obj_points=256,
        n_table_points=256,
        rng=np.random.default_rng(args.seed + 1),
    )
    obs = {
        "pc_gripper": pc_gripper,
        "pc_scene": pc_scene,
        "ee_pos": ee_pos0,
        "ee_rot": T_ee[:3, :3].copy(),
        "gripper_width": 0.05,
        "obj_pos": obj_pos.copy(),
        "T_ee": T_ee.copy(),
    }

    adpro.rng = np.random.default_rng(args.seed)
    adpro.policy.rng = np.random.default_rng(args.seed)
    _, traj = adpro.inference(obs, n_steps=args.steps, return_trajectory=True)
    traj = np.asarray(traj, dtype=np.float64)
    xyz = traj[:, :3]
    grasp_target = obj_pos + np.array([0.0, 0.0, 0.05], dtype=np.float64)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    # Trajectory colored by step index
    t = np.linspace(0.0, 1.0, len(xyz))
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=t, cmap="viridis", s=18, label="ADPro denoising path")
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="tab:blue", alpha=0.5, linewidth=1.2)

    ax.scatter(ee_pos0[0], ee_pos0[1], ee_pos0[2], c="red", s=100, marker="o", label="EE start")
    ax.scatter(obj_pos[0], obj_pos[1], obj_pos[2], c="orange", s=120, marker="D", label="Object")
    ax.scatter(
        grasp_target[0],
        grasp_target[1],
        grasp_target[2],
        c="black",
        s=120,
        marker="*",
        label="Grasp target (obj+z)",
    )

    ax.text(obj_pos[0], obj_pos[1], obj_pos[2] + 0.01, f"obj {np.round(obj_pos, 3)}", color="darkorange")
    ax.text(ee_pos0[0], ee_pos0[1], ee_pos0[2] + 0.01, f"start {np.round(ee_pos0, 3)}", color="darkred")

    ax.set_title(f"ADPro long iteration (n_steps={args.steps})")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=180)
    plt.close(fig)

    print(f"[ok] Figure sauvée: {args.out}")
    print(f"[info] obj_pos={obj_pos}")
    print(f"[info] ee_start={ee_pos0}")
    if used_zmq:
        print("[info] Départ depuis la pose robot courante (ZMQ).")
    else:
        print("[info] Départ depuis FRANKA_HOME_CONFIG (utilise --from-zmq pour la pose courante).")


if __name__ == "__main__":
    main()
