#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gello.zmq_core.robot_node import ZMQClientRobot
from utils.geometry import forward_kinematics


def solve_affine(robot_pts: np.ndarray, vis_pts: np.ndarray):
    """
    Fit affine map p_vis = A @ p_robot + b via least squares.
    robot_pts, vis_pts: (N,3)
    """
    n = robot_pts.shape[0]
    X = np.hstack([robot_pts, np.ones((n, 1), dtype=np.float64)])  # (N,4)
    W, *_ = np.linalg.lstsq(X, vis_pts, rcond=None)  # (4,3)
    A = W[:3, :].T
    b = W[3, :]
    pred = (A @ robot_pts.T).T + b
    rmse = float(np.sqrt(np.mean(np.sum((pred - vis_pts) ** 2, axis=1))))
    return A, b, rmse


def main():
    parser = argparse.ArgumentParser(description="Calibrate affine transform between robot and visualization frames.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6001)
    parser.add_argument("--out", default="results/visual_transform.json")
    args = parser.parse_args()

    print("Calibration affine: p_vis = A @ p_robot + b")
    print("Pour chaque point:")
    print("1) Place le robot sur un point repérable.")
    print("2) Appuie sur Entrée pour capturer la pose robot.")
    print("3) Entre la coordonnée de CE MÊME point dans ton repère de visualisation: x y z")
    print("Tape 'q' à tout moment pour terminer.")

    client = ZMQClientRobot(port=args.port, host=args.host)
    robot_points = []
    vis_points = []

    idx = 1
    while True:
        gate = input(f"\nPoint {idx} - Entrée pour capturer, q pour finir > ").strip().lower()
        if gate == "q":
            break
        q = np.asarray(client.get_joint_state(), dtype=np.float64)[:7]
        p_robot = forward_kinematics(q)[:3, 3]
        print(f"  p_robot = {np.round(p_robot, 6)}")

        vis_raw = input("  p_vis attendu (format: x y z) > ").strip()
        if vis_raw.lower() == "q":
            break
        try:
            vals = [float(v) for v in vis_raw.replace(",", " ").split()]
            if len(vals) != 3:
                raise ValueError("Need 3 values")
            p_vis = np.asarray(vals, dtype=np.float64)
        except Exception:
            print("  Entrée invalide, point ignoré.")
            continue

        robot_points.append(p_robot)
        vis_points.append(p_vis)
        print(f"  Pair OK #{idx}: robot={np.round(p_robot,4)} -> vis={np.round(p_vis,4)}")
        idx += 1

    if len(robot_points) < 4:
        raise RuntimeError("Au moins 4 paires sont nécessaires (6+ recommandé).")

    robot_pts = np.asarray(robot_points, dtype=np.float64)
    vis_pts = np.asarray(vis_points, dtype=np.float64)
    A, b, rmse = solve_affine(robot_pts, vis_pts)

    out = {
        "A": A.tolist(),
        "b": b.tolist(),
        "rmse": rmse,
        "num_pairs": int(len(robot_points)),
        "robot_points": robot_pts.tolist(),
        "vis_points": vis_pts.tolist(),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n=== Calibration terminée ===")
    print(f"pairs: {len(robot_points)} | rmse: {rmse:.6f}")
    print(f"A=\n{A}")
    print(f"b={b}")
    print(f"Fichier sauvegardé: {out_path}")


if __name__ == "__main__":
    main()

