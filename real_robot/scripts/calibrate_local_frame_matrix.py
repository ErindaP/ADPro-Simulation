#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from real_robot.adapters.gello_zmq_adapter import GelloZmqConfig, build_gello_stack
from real_robot.safety import SafetyConfig, SafetyFilter
from utils.geometry import rot_to_quat


AXES = [
    ("x", np.array([1.0, 0.0, 0.0], dtype=np.float64)),
    ("y", np.array([0.0, 1.0, 0.0], dtype=np.float64)),
    ("z", np.array([0.0, 0.0, 1.0], dtype=np.float64)),
]


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_obs(provider):
    obs = provider.get_observation(timeout_s=0.5)
    if obs is None:
        raise RuntimeError("Observation indisponible (vérifie Terminal A/B/C).")
    return obs


def move_towards_pose(provider, commander, safety, target_pos, target_quat, step_m, settle_s):
    """Move smoothly toward target ee position using small cartesian increments."""
    max_iters = 300
    step_m = max(float(step_m), 1e-4)
    for _ in range(max_iters):
        obs = get_obs(provider)
        cur = np.asarray(obs.ee_pos, dtype=np.float64)
        d = target_pos - cur
        dn = float(np.linalg.norm(d))
        if dn <= step_m:
            raw = np.concatenate([target_pos, target_quat], dtype=np.float64)
        else:
            raw = np.concatenate([cur + d / (dn + 1e-12) * step_m, target_quat], dtype=np.float64)
        safe = safety.filter_action(raw, obs.ee_pos, target_quat)
        commander.send_pose_target(safe)
        time.sleep(settle_s)
        if dn <= step_m:
            break


def return_to_reference_joints(
    commander,
    ref_joints8,
    settle_s=0.2,
    joint_step_rad=0.01,
    max_iters=400,
    timeout_s: float = 20.0,
):
    """Smoothly return to the exact initial joint snapshot."""
    ref_joints8 = np.asarray(ref_joints8, dtype=np.float64).copy()
    joint_step_rad = max(float(joint_step_rad), 1e-4)

    t0 = time.time()
    for _ in range(max_iters):
        if (time.time() - t0) > timeout_s:
            print("[calib][warn] return_to_reference_joints timeout")
            break
        cur = np.asarray(commander.client.get_joint_state(), dtype=np.float64)
        if cur.shape[0] < 8:
            cur = np.pad(cur, (0, max(0, 8 - cur.shape[0])), mode="constant")
        d = ref_joints8 - cur
        if float(np.max(np.abs(d[:7]))) <= joint_step_rad:
            commander.client.command_joint_state(ref_joints8)
            time.sleep(settle_s)
            break
        nxt = cur.copy()
        nxt[:7] = cur[:7] + np.clip(d[:7], -joint_step_rad, joint_step_rad)
        nxt[7] = cur[7] + np.clip(d[7], -0.03, 0.03)
        commander.client.command_joint_state(nxt)
        time.sleep(settle_s)


def main():
    parser = argparse.ArgumentParser(description="Calibrate local command->motion matrix around current pose.")
    parser.add_argument("--config", default="real_robot/config/lab_franka.json")
    parser.add_argument("--step", type=float, default=0.002, help="Cartesian command step (m).")
    parser.add_argument("--n-steps", type=int, default=10, help="Number of small steps per axis.")
    parser.add_argument("--settle", type=float, default=0.2, help="Sleep between commands (s).")
    parser.add_argument("--return-joint-step", type=float, default=0.01)
    parser.add_argument("--axis-timeout", type=float, default=45.0, help="Timeout max (s) for one axis.")
    parser.add_argument(
        "--min-motion",
        type=float,
        default=0.001,
        help="Alerte si le meilleur déplacement observé sur un axe est < min-motion (m).",
    )
    parser.add_argument("--out", default="results/local_frame_matrix.json")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    g = cfg.get("gello", {})
    s = cfg["safety"]

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
    safety = SafetyFilter(
        SafetyConfig(
            workspace_min=np.array(s["workspace_min"], dtype=np.float64),
            workspace_max=np.array(s["workspace_max"], dtype=np.float64),
            max_translation_step_m=float(s["max_translation_step_m"]),
            max_rotation_step_rad=float(s["max_rotation_step_rad"]),
            lowpass_alpha=float(s["lowpass_alpha"]),
        )
    )

    try:
        obs0 = get_obs(provider)
        p_ref = np.asarray(obs0.ee_pos, dtype=np.float64).copy()
        q_ref = rot_to_quat(np.asarray(obs0.ee_rot, dtype=np.float64))
        qj_ref = np.asarray(commander.client.get_joint_state(), dtype=np.float64).copy()
        print(f"[calib] reference ee={np.round(p_ref, 5)}")
        print(f"[calib] step={args.step:.4f} n_steps={args.n_steps}")

        total_cmd = args.step * args.n_steps
        M = np.zeros((3, 3), dtype=np.float64)  # columns: effect of cmd x/y/z
        samples = {}

        for j, (axis_name, axis_dir) in enumerate(AXES):
            print(f"\n[calib] axis {axis_name}: probing + and - ...")
            trial_results = []
            axis_t0 = time.time()
            for sign in (+1.0, -1.0):
                if (time.time() - axis_t0) > args.axis_timeout:
                    print(f"[calib][warn] axis {axis_name} timeout before trial {sign:+.0f}")
                    break
                # reset to exact reference in joint space before each trial
                return_to_reference_joints(
                    commander=commander,
                    ref_joints8=qj_ref,
                    settle_s=args.settle,
                    joint_step_rad=args.return_joint_step,
                    timeout_s=min(20.0, args.axis_timeout),
                )
                safety._last_action7 = None

                p_start = np.asarray(get_obs(provider).ee_pos, dtype=np.float64)
                direction = sign * axis_dir
                trial_name = f"{axis_name}{'+' if sign > 0 else '-'}"
                print(f"[calib] trial {trial_name}...")

                for _ in range(args.n_steps):
                    if (time.time() - axis_t0) > args.axis_timeout:
                        print(f"[calib][warn] axis {axis_name} timeout during {trial_name}")
                        break
                    obs = get_obs(provider)
                    cur = np.asarray(obs.ee_pos, dtype=np.float64)
                    raw = np.concatenate([cur + args.step * direction, q_ref], dtype=np.float64)
                    safe = safety.filter_action(raw, obs.ee_pos, q_ref)
                    commander.send_pose_target(safe)
                    time.sleep(args.settle)

                p_end = np.asarray(get_obs(provider).ee_pos, dtype=np.float64)
                dp = p_end - p_start
                amp = float(np.linalg.norm(dp))
                signed_total = total_cmd * sign
                gain_col = dp / max(abs(signed_total), 1e-9)
                if sign < 0:
                    gain_col = -gain_col

                trial_results.append(
                    {
                        "name": trial_name,
                        "sign": float(sign),
                        "p_start": p_start,
                        "p_end": p_end,
                        "delta": dp,
                        "amp": amp,
                        "gain_col": gain_col,
                    }
                )
                print(
                    f"[calib] {trial_name} dp={np.round(dp, 6)} "
                    f"amp={amp:.6f} gain={np.round(gain_col, 6)}"
                )

            if not trial_results:
                print(f"[calib][warn] axis {axis_name}: no trial result, set column to zeros.")
                M[:, j] = 0.0
                samples[axis_name] = {
                    "chosen_trial": None,
                    "chosen_amp": 0.0,
                    "chosen_delta": [0.0, 0.0, 0.0],
                    "p_start": None,
                    "p_end": None,
                    "all_trials": {},
                }
                continue

            best = max(trial_results, key=lambda t: t["amp"])
            M[:, j] = best["gain_col"]
            samples[axis_name] = {
                "chosen_trial": best["name"],
                "chosen_amp": float(best["amp"]),
                "chosen_delta": best["delta"].tolist(),
                "p_start": best["p_start"].tolist(),
                "p_end": best["p_end"].tolist(),
                "all_trials": {
                    t["name"]: {
                        "amp": float(t["amp"]),
                        "delta": t["delta"].tolist(),
                        "gain_col": t["gain_col"].tolist(),
                    }
                    for t in trial_results
                },
            }
            print(f"[calib] axis {axis_name} -> chosen {best['name']}")
            if best["amp"] < args.min_motion:
                print(
                    f"[calib][warn] axis {axis_name}: motion very small ({best['amp']:.6f} m). "
                    "Result may be poorly conditioned."
                )

        print("\n[calib] Local matrix M (observed_delta = M * commanded_delta):")
        print(M)

        Minv = None
        cond = float(np.linalg.cond(M))
        print(f"[calib] cond(M)={cond:.4f}")
        if np.isfinite(cond) and cond < 1e6:
            Minv = np.linalg.inv(M)
            print("[calib] M^-1:")
            print(Minv)
        else:
            print("[calib] Matrix near singular; skipping inverse.")

        out = {
            "reference_ee": p_ref.tolist(),
            "step": float(args.step),
            "n_steps": int(args.n_steps),
            "total_commanded_delta_per_axis": float(total_cmd),
            "M": M.tolist(),
            "Minv": Minv.tolist() if Minv is not None else None,
            "cond_M": cond,
            "samples": samples,
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"[calib] saved: {out_path}")
        print("[calib] Use M^-1 to convert desired world delta to corrected command delta.")

    finally:
        provider.close()
        commander.close()


if __name__ == "__main__":
    main()
