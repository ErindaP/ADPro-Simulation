import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from gello.zmq_core.robot_node import ZMQClientRobot
from utils.geometry import forward_kinematics


def get_transform(name: str):
    # p_vis = A @ p_robot + b
    A = np.eye(3, dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    if name == "identity":
        pass
    elif name == "invert_x":
        A = np.diag([-1.0, 1.0, 1.0])
    elif name == "invert_y":
        A = np.diag([1.0, -1.0, 1.0])
    elif name == "invert_z":
        A = np.diag([1.0, 1.0, -1.0])
    elif name == "swap_xy":
        A = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    elif name == "swap_xz":
        A = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    elif name == "swap_yz":
        A = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    elif name == "swap_xy_invert_x":
        A = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    else:
        raise ValueError(f"Unknown transform preset: {name}")
    return A, b


def load_transform_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    A = np.asarray(data["A"], dtype=np.float64)
    b = np.asarray(data["b"], dtype=np.float64)
    if A.shape != (3, 3) or b.shape != (3,):
        raise ValueError("Invalid transform file format: need A=(3,3), b=(3,)")
    return A, b


def apply_tf(p: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (A @ p.reshape(3, 1)).reshape(3) + b


def main():
    parser = argparse.ArgumentParser(description="Live EE trajectory visualization with optional axis transform.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6001)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument(
        "--transform",
        default="identity",
        choices=[
            "identity",
            "invert_x",
            "invert_y",
            "invert_z",
            "swap_xy",
            "swap_xz",
            "swap_yz",
            "swap_xy_invert_x",
        ],
    )
    parser.add_argument(
        "--transform-file",
        default=None,
        help="JSON file with affine transform fields: A (3x3), b (3). Overrides --transform.",
    )
    args = parser.parse_args()

    cfg_path = Path("real_robot/config/lab_franka.json")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    obj_pos = np.asarray(cfg.get("gello", {}).get("obj_pos", [-0.52, -0.16, 0.16]), dtype=np.float64)

    if args.transform_file:
        A, b = load_transform_file(args.transform_file)
        tf_name = f"file:{args.transform_file}"
    else:
        A, b = get_transform(args.transform)
        tf_name = args.transform
    obj_v = apply_tf(obj_pos, A, b)

    c = ZMQClientRobot(port=args.port, host=args.host)
    pts = []
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for _ in range(args.steps):
        q = np.asarray(c.get_joint_state(), dtype=np.float64)[:7]
        p = forward_kinematics(q)[:3, 3]
        p_v = apply_tf(p, A, b)
        pts.append(p_v.copy())
        P = np.array(pts)
        ax.cla()
        ax.plot(P[:, 0], P[:, 1], P[:, 2], "b-", label="EE trajectory")
        ax.scatter(P[-1, 0], P[-1, 1], P[-1, 2], c="r", s=60, label="EE current")
        ax.scatter(obj_v[0], obj_v[1], obj_v[2], c="orange", s=100, marker="D", label="Target object")
        ax.text(
            obj_v[0],
            obj_v[1],
            obj_v[2] + 0.01,
            f"obj=({obj_v[0]:.3f}, {obj_v[1]:.3f}, {obj_v[2]:.3f})",
            color="darkorange",
        )
        ax.set_title(f"Trajectoire EE live + cible objet | tf={tf_name}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend(loc="upper right")
        plt.pause(0.01)
        time.sleep(0.05)


if __name__ == "__main__":
    main()
