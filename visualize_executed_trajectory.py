"""
Visualisation de la trajectoire effectivement exécutée par le robot.

Ce script ne commande pas le robot.
Il lit la pose EE via ZMQ (launch_nodes.py) et trace:
- la trajectoire mesurée,
- le point de départ / d'arrivée,
- la position objet (depuis real_robot/config/lab_franka.json).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gello.zmq_core.robot_node import ZMQClientRobot
from utils.geometry import forward_kinematics


def main():
    parser = argparse.ArgumentParser(description="Visualize executed EE trajectory from ZMQ.")
    parser.add_argument("--config", default="real_robot/config/lab_franka.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--robot-port", type=int, default=6001)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--out", default="results/executed_trajectory.png")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    obj_pos = np.asarray(cfg.get("gello", {}).get("obj_pos", [-0.52, -0.16, 0.16]), dtype=np.float64)

    c = ZMQClientRobot(port=args.robot_port, host=args.host)
    dt = 1.0 / max(args.hz, 1e-6)
    n = max(1, int(args.seconds * args.hz))

    pts = []
    for _ in range(n):
        q = np.asarray(c.get_joint_state(), dtype=np.float64)[:7]
        p = forward_kinematics(q)[:3, 3]
        pts.append(p.copy())
        time.sleep(dt)

    P = np.asarray(pts, dtype=np.float64)
    if len(P) < 2:
        raise RuntimeError("Trajectoire trop courte: vérifie que le serveur ZMQ est actif.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(P[:, 0], P[:, 1], P[:, 2], "b-", linewidth=2.0, label="Executed EE trajectory")
    ax.scatter(P[0, 0], P[0, 1], P[0, 2], c="green", s=100, marker="o", label="EE start")
    ax.scatter(P[-1, 0], P[-1, 1], P[-1, 2], c="red", s=100, marker="o", label="EE end")

    ax.scatter(obj_pos[0], obj_pos[1], obj_pos[2], c="orange", s=120, marker="D", label="Object")
    grasp_target = obj_pos + np.array([0.0, 0.0, 0.05], dtype=np.float64)
    ax.scatter(
        grasp_target[0],
        grasp_target[1],
        grasp_target[2],
        c="black",
        s=120,
        marker="*",
        label="Grasp target",
    )

    ax.text(obj_pos[0], obj_pos[1], obj_pos[2] + 0.01, f"obj {np.round(obj_pos, 3)}", color="darkorange")
    ax.text(P[0, 0], P[0, 1], P[0, 2] + 0.01, f"start {np.round(P[0], 3)}", color="green")
    ax.text(P[-1, 0], P[-1, 1], P[-1, 2] + 0.01, f"end {np.round(P[-1], 3)}", color="darkred")

    ax.set_title("Executed EE trajectory (measured)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=180)
    plt.close(fig)

    d0 = float(np.linalg.norm(P[0] - grasp_target))
    d1 = float(np.linalg.norm(P[-1] - grasp_target))
    dmin = float(np.min(np.linalg.norm(P - grasp_target[None, :], axis=1)))
    print(f"[ok] Figure sauvée: {args.out}")
    print(f"[info] start={P[0]} end={P[-1]} obj={obj_pos} grasp={grasp_target}")
    print(f"[info] dist_to_grasp: start={d0:.4f} end={d1:.4f} min={dmin:.4f}")


if __name__ == "__main__":
    main()

