"""
point_cloud.py — Génération de nuages de points pour le Franka Panda

Génère les nuages de points du gripper et de la scène à partir
des paramètres géométriques réels du robot Franka Panda Emika.
"""

import numpy as np
from utils.geometry import (
    forward_kinematics, transform_pointcloud, quat_to_rot
)


# ============================================================
# Géométrie réelle du Franka Panda Hand (gripper)
# Source: Franka Emika technical documentation
# ============================================================
FRANKA_FINGER_LENGTH = 0.0595   # m
FRANKA_FINGER_WIDTH  = 0.012    # m
FRANKA_FINGER_DEPTH  = 0.010    # m
FRANKA_PALM_SIZE     = (0.055, 0.04, 0.03)   # (x, y, z) m

# Géométrie d'un objet cylindrique typique (ex. bouteille)
OBJECT_RADIUS = 0.025   # m
OBJECT_HEIGHT = 0.10    # m


def sample_gripper_pointcloud(
    T_ee: np.ndarray,
    gripper_width: float = 0.04,
    n_points: int = 512,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Génère le nuage de points du gripper Franka Panda à partir de sa pose SE(3).

    Le modèle géométrique du gripper comprend :
    - La paume (parallélépipède)
    - Les deux doigts (parallélépipèdes)

    Args:
        T_ee:          pose de l'end-effector (SE3, 4×4)
        gripper_width: ouverture du gripper en m (0..0.08)
        n_points:      nombre de points à générer
        rng:           générateur aléatoire

    Returns:
        pc: (n_points, 3) nuage de points dans le repère monde
    """
    if rng is None:
        rng = np.random.default_rng(0)

    points = []
    n_per_part = n_points // 3

    # --- Paume ---
    px, py, pz = FRANKA_PALM_SIZE
    palm_pts = rng.uniform(
        low=[-px/2, -py/2, -pz/2],
        high=[px/2,  py/2,  pz/2],
        size=(n_per_part, 3),
    )
    points.append(palm_pts)

    # --- Doigt gauche ---
    offset_y = gripper_width / 2
    left_pts = rng.uniform(
        low=[-FRANKA_FINGER_WIDTH/2, offset_y, 0],
        high=[FRANKA_FINGER_WIDTH/2, offset_y + FRANKA_FINGER_DEPTH, FRANKA_FINGER_LENGTH],
        size=(n_per_part, 3),
    )
    points.append(left_pts)

    # --- Doigt droit ---
    right_pts = rng.uniform(
        low=[-FRANKA_FINGER_WIDTH/2, -(offset_y + FRANKA_FINGER_DEPTH), 0],
        high=[FRANKA_FINGER_WIDTH/2, -offset_y, FRANKA_FINGER_LENGTH],
        size=(n_points - 2 * n_per_part, 3),
    )
    points.append(right_pts)

    pc_local = np.vstack(points)  # (n_points, 3)

    # Transformer dans le repère monde
    pc_world = transform_pointcloud(pc_local, T_ee)
    return pc_world


def sample_object_pointcloud(
    obj_pos: np.ndarray,
    obj_radius: float = OBJECT_RADIUS,
    obj_height: float = OBJECT_HEIGHT,
    n_points: int = 512,
    rng: np.random.Generator = None,
    shape: str = "cylinder",
) -> np.ndarray:
    """
    Génère le nuage de points d'un objet cible.

    Args:
        obj_pos:    (3,) centre de la base de l'objet (repère monde)
        obj_radius: rayon de l'objet (m)
        obj_height: hauteur de l'objet (m)
        n_points:   nombre de points
        rng:        générateur
        shape:      "cylinder" ou "box"

    Returns:
        pc: (n_points, 3) nuage de points dans le repère monde
    """
    if rng is None:
        rng = np.random.default_rng(0)

    if shape == "cylinder":
        # Surface latérale du cylindre
        theta = rng.uniform(0, 2 * np.pi, n_points)
        z = rng.uniform(0, obj_height, n_points)
        x = obj_radius * np.cos(theta)
        y = obj_radius * np.sin(theta)
        pc = np.stack([x, y, z], axis=1)
        # Ajouter bruit de surface
        pc += rng.normal(0, 0.001, pc.shape)
    elif shape == "box":
        side = obj_radius * 2
        pc = rng.uniform(
            low=[-side/2, -side/2, 0],
            high=[side/2, side/2, obj_height],
            size=(n_points, 3),
        )
    else:
        raise ValueError(f"Shape inconnu : {shape}")

    # Décaler à la position de l'objet
    pc += obj_pos
    return pc


def sample_scene_pointcloud(
    obj_pos: np.ndarray,
    table_height: float = 0.0,
    n_obj_points: int = 256,
    n_table_points: int = 256,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Génère le nuage de points de la scène complète :
    - Nuage de points de l'objet cible
    - Surface de la table

    Args:
        obj_pos:         (3,) position de l'objet
        table_height:    hauteur de la table en z
        n_obj_points:    points sur l'objet
        n_table_points:  points sur la table
        rng:             générateur

    Returns:
        pc_scene: (n, 3) nuage de points de la scène
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # Nuage de points de l'objet
    pc_obj = sample_object_pointcloud(obj_pos, n_points=n_obj_points, rng=rng)

    # Surface de la table (plan z=table_height)
    table_pts = rng.uniform(
        low=[obj_pos[0] - 0.3, obj_pos[1] - 0.3, table_height],
        high=[obj_pos[0] + 0.3, obj_pos[1] + 0.3, table_height],
        size=(n_table_points, 3),
    )

    return np.vstack([pc_obj, table_pts])


def subsample_pointcloud(pc: np.ndarray, n: int = 4096,
                          rng: np.random.Generator = None) -> np.ndarray:
    """Sous-échantillonne aléatoirement un nuage de points à n points."""
    if rng is None:
        rng = np.random.default_rng(0)
    if len(pc) <= n:
        return pc
    idx = rng.choice(len(pc), n, replace=False)
    return pc[idx]
