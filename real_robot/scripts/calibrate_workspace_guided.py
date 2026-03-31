#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Optional

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from real_robot.adapters.gello_zmq_adapter import GelloZmqConfig, build_gello_stack
from utils.geometry import rot_to_quat


DIRS = [
    ("x+", np.array([1.0, 0.0, 0.0], dtype=np.float64)),
    ("x-", np.array([-1.0, 0.0, 0.0], dtype=np.float64)),
    ("y+", np.array([0.0, 1.0, 0.0], dtype=np.float64)),
    ("y-", np.array([0.0, -1.0, 0.0], dtype=np.float64)),
    ("z+", np.array([0.0, 0.0, 1.0], dtype=np.float64)),
    ("z-", np.array([0.0, 0.0, -1.0], dtype=np.float64)),
]


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_obs(provider):
    obs = provider.get_observation(timeout_s=0.5)
    if obs is None:
        raise RuntimeError("Observation indisponible (vérifie Terminal A/B/C).")
    return obs


def go_to_reference_pose(
    provider,
    commander,
    ref_pos: np.ndarray,
    ref_quat: np.ndarray,
    settle_s: float,
    return_step_m: float = 0.003,
    tol_m: float = 0.004,
    max_iters: int = 240,
):
    """Drive robot back to reference EE pose between direction scans."""
    return_step_m = max(float(return_step_m), 1e-4)
    for _ in range(max_iters):
        obs = get_obs(provider)
        cur = np.asarray(obs.ee_pos, dtype=np.float64)
        d = ref_pos - cur
        dn = float(np.linalg.norm(d))
        if dn <= tol_m:
            break
        step = d if dn <= return_step_m else (d / (dn + 1e-12) * return_step_m)
        target = cur + step
        action7 = np.concatenate([target, ref_quat], dtype=np.float64)
        commander.send_pose_target(action7)
        time.sleep(settle_s)


def go_to_reference_joints(
    commander,
    ref_joints8: np.ndarray,
    settle_s: float,
    joint_step_rad: float = 0.02,
    joint_tol_rad: float = 0.01,
    stagnation_tol_rad: float = 0.002,
    stagnation_patience: int = 12,
    max_iters: int = 300,
):
    """Return smoothly to exact initial joint configuration."""
    joint_step_rad = max(float(joint_step_rad), 1e-4)
    joint_tol_rad = max(float(joint_tol_rad), 1e-4)
    ref_joints8 = np.asarray(ref_joints8, dtype=np.float64).copy()
    best_dn = np.inf
    stagnant_count = 0

    for _ in range(max_iters):
        cur = np.asarray(commander.client.get_joint_state(), dtype=np.float64)
        if cur.shape[0] < 8:
            cur = np.pad(cur, (0, max(0, 8 - cur.shape[0])), mode="constant")
        d = ref_joints8 - cur
        dn = float(np.max(np.abs(d[:7])))
        if dn <= joint_tol_rad:
            break

        if dn + stagnation_tol_rad < best_dn:
            best_dn = dn
            stagnant_count = 0
        else:
            stagnant_count += 1
            if stagnant_count >= stagnation_patience:
                print(
                    f"[return_home] arrêt sur stagnation: err_joint={dn:.4f} rad "
                    f"(tol={joint_tol_rad:.4f})"
                )
                break

        step = np.clip(d[:7], -joint_step_rad, joint_step_rad)
        nxt = cur.copy()
        nxt[:7] = cur[:7] + step
        # keep smooth gripper return too
        nxt[7] = cur[7] + np.clip(d[7], -0.03, 0.03)
        commander.client.command_joint_state(nxt)
        time.sleep(settle_s)


def main():
    parser = argparse.ArgumentParser(description="Guided workspace calibration (gello backend).")
    parser.add_argument("--config", default="real_robot/config/lab_franka.json")
    parser.add_argument("--step", type=float, default=0.005, help="Pas cartésien en mètres.")
    parser.add_argument("--settle", type=float, default=0.25, help="Temps d'attente après commande (s).")
    parser.add_argument(
        "--return-step",
        type=float,
        default=0.003,
        help="Pas cartésien (m) pour le retour progressif vers la pose de référence.",
    )
    parser.add_argument(
        "--return-joint-step",
        type=float,
        default=0.02,
        help="Pas max (rad) par articulation pour le retour en configuration initiale.",
    )
    parser.add_argument(
        "--return-joint-tol",
        type=float,
        default=0.01,
        help="Tolérance (rad) max par articulation pour considérer le retour initial atteint.",
    )
    parser.add_argument("--max-steps-per-dir", type=int, default=80)
    parser.add_argument(
        "--unlock-orthogonal",
        action="store_true",
        help="Par défaut les axes orthogonaux sont verrouillés pendant le scan d'une direction.",
    )
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    g = cfg.get("gello", {})
    provider, commander = build_gello_stack(
        GelloZmqConfig(
            host=str(g.get("host", "127.0.0.1")),
            robot_port=int(g.get("robot_port", 6001)),
            obj_pos=np.array(g.get("obj_pos", [0.55, 0.0, 0.05]), dtype=np.float64),
            table_height=float(g.get("table_height", 0.0)),
            n_pc_points=int(g.get("n_pc_points", 512)),
            ik_iters=int(g.get("ik_iters", 24)),
        )
    )

    try:
        bounds: Dict[str, Optional[np.ndarray]] = {name: None for name, _ in DIRS}
        print("Calibration workspace interactive.")
        if not sys.stdin.isatty():
            print(
                "Attention: stdin non interactif detecte. "
                "Lance ce script dans un vrai terminal (pas via runner non interactif)."
            )
        print("Commandes: [Entrée]=pas suivant, ok=valider limite, q=quitter direction")
        print(f"Pas={args.step:.4f} m | max_steps_per_dir={args.max_steps_per_dir}")

        # Reference pose to restore between each direction.
        obs_ref = get_obs(provider)
        ref_pos = np.asarray(obs_ref.ee_pos, dtype=np.float64).copy()
        ref_quat = rot_to_quat(np.asarray(obs_ref.ee_rot, dtype=np.float64))
        ref_joints8 = np.asarray(commander.client.get_joint_state(), dtype=np.float64).copy()
        print(f"Pose de référence: {np.round(ref_pos, 4)}")

        for name, direction in DIRS:
            print()
            print(f"=== Direction {name} ===")
            print("Amène l'espace robot en sécurité, puis utilise Entrée/ok.")

            obs = get_obs(provider)
            print(f"EE départ {name}: {np.round(obs.ee_pos, 4)}")
            axis_idx = int(np.argmax(np.abs(direction)))
            anchor = np.asarray(obs.ee_pos, dtype=np.float64).copy()
            step_count = 0

            while step_count < args.max_steps_per_dir:
                try:
                    cmd = input(f"[{name}] Entrée=move, ok=save, q=next > ").strip().lower()
                except EOFError:
                    print(
                        f"[{name}] EOF sur stdin. "
                        "Terminal non interactif -> direction interrompue."
                    )
                    cmd = "q"
                if cmd == "ok":
                    obs_now = get_obs(provider)
                    bounds[name] = obs_now.ee_pos.copy()
                    print(f"[{name}] borne sauvegardée: {np.round(bounds[name], 4)}")
                    break
                if cmd == "q":
                    print(f"[{name}] ignorée.")
                    break
                if cmd not in ("",):
                    print("Commande non reconnue.")
                    continue

                obs_now = get_obs(provider)
                quat = rot_to_quat(np.asarray(obs_now.ee_rot, dtype=np.float64))
                target_pos = np.asarray(obs_now.ee_pos, dtype=np.float64) + args.step * direction
                if not args.unlock_orthogonal:
                    # Keep non-scanned axes fixed to reduce XY/Z coupling during calibration.
                    for j in (0, 1, 2):
                        if j != axis_idx:
                            target_pos[j] = anchor[j]
                action7 = np.concatenate([target_pos, quat], dtype=np.float64)
                commander.send_pose_target(action7)
                time.sleep(args.settle)
                obs_after = get_obs(provider)
                print(
                    f"[{name}] step {step_count+1:02d} "
                    f"ee={np.round(obs_after.ee_pos, 4)} "
                    f"(target={np.round(target_pos, 4)})"
                )
                step_count += 1

            if bounds[name] is None:
                obs_now = get_obs(provider)
                bounds[name] = obs_now.ee_pos.copy()
                print(f"[{name}] fallback borne courante: {np.round(bounds[name], 4)}")

            print(f"[{name}] retour à la pose de référence...")
            go_to_reference_joints(
                commander=commander,
                ref_joints8=ref_joints8,
                settle_s=args.settle,
                joint_step_rad=args.return_joint_step,
                joint_tol_rad=args.return_joint_tol,
            )
            obs_back = get_obs(provider)
            print(f"[{name}] pose après retour: {np.round(obs_back.ee_pos, 4)}")

        x_min = float(bounds["x-"][0])
        x_max = float(bounds["x+"][0])
        y_min = float(bounds["y-"][1])
        y_max = float(bounds["y+"][1])
        z_min = float(bounds["z-"][2])
        z_max = float(bounds["z+"][2])

        workspace_min = [x_min, y_min, z_min]
        workspace_max = [x_max, y_max, z_max]

        print()
        print("=== Résultat calibration ===")
        print(f"workspace_min: {workspace_min}")
        print(f"workspace_max: {workspace_max}")
        print()
        print("Snippet JSON à copier dans real_robot/config/lab_franka.json :")
        print("{")
        print(f'  "workspace_min": {workspace_min},')
        print(f'  "workspace_max": {workspace_max}')
        print("}")

    finally:
        provider.close()
        commander.close()


if __name__ == "__main__":
    main()
