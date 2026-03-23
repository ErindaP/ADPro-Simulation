"""
geometry.py — Opérations de géométrie SE(3) pour le Franka Panda

Implémente les opérations nécessaires pour ADPro :
- Conversion vecteur 7D ↔ matrice SE(3) (position + quaternion + gripper width)
- Paramètres DH réels du Franka Panda Emika
- Cinématique directe du Franka Panda (7 DOF)
- Distance de Chamfer différenciable (numpy)
- Fast Global Registration (FGR) simplifié
"""

import numpy as np


# ============================================================
# Paramètres DH réels du Franka Panda Emika (Denavit-Hartenberg)
# Source: Franka Emika Panda Technical Specification
# Format : (d, a, alpha) en mètres et radians
# ============================================================
FRANKA_DH = [
    # d       a       alpha
    (0.333,  0.000,  0.000),   # Joint 1
    (0.000,  0.000, -np.pi/2), # Joint 2
    (0.316,  0.000,  np.pi/2), # Joint 3
    (0.000,  0.0825, np.pi/2), # Joint 4
    (0.384, -0.0825, -np.pi/2),# Joint 5
    (0.000,  0.000,  np.pi/2), # Joint 6
    (0.000,  0.088,  np.pi/2), # Joint 7
]

# Offset fixe vers le flanc outil (flange → TCP gripper)
FRANKA_FLANGE_TCP_OFFSET = np.array([0.0, 0.0, 0.1034])  # m

# Limites articulaires du Franka Panda (en radians)
FRANKA_JOINT_LIMITS = np.array([
    [-2.8973,  2.8973],  # q1
    [-1.7628,  1.7628],  # q2
    [-2.8973,  2.8973],  # q3
    [-3.0718, -0.0698],  # q4
    [-2.8973,  2.8973],  # q5
    [-0.0175,  3.7525],  # q6
    [-2.8973,  2.8973],  # q7
])

# Position de repos (configuration neutre)
FRANKA_HOME_CONFIG = np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4])


def dh_matrix(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """Matrice de transformation homogène DH standard 4×4."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,       sa,       ca,      d],
        [0.0,      0.0,      0.0,    1.0],
    ])


def forward_kinematics(q: np.ndarray) -> np.ndarray:
    """
    Cinématique directe du Franka Panda.

    Args:
        q: vecteur de configuration articulaire (7,)

    Returns:
        T_EE: matrice de transformation homogène 4×4 (monde → end-effector)
    """
    T = np.eye(4)
    for i, (d, a, alpha) in enumerate(FRANKA_DH):
        T = T @ dh_matrix(q[i], d, a, alpha)

    # Ajouter l'offset flange → TCP
    T_tcp = np.eye(4)
    T_tcp[:3, 3] = FRANKA_FLANGE_TCP_OFFSET
    T = T @ T_tcp
    return T


def forward_kinematics_chain(q: np.ndarray) -> list:
    """
    Retourne la chaîne de transformations du Franka Panda.

    Returns:
        chain: liste de matrices 4x4 [base, joint1, ..., joint7, tcp]
    """
    T = np.eye(4)
    chain = [T.copy()]
    for i, (d, a, alpha) in enumerate(FRANKA_DH):
        T = T @ dh_matrix(q[i], d, a, alpha)
        chain.append(T.copy())

    T_tcp = np.eye(4)
    T_tcp[:3, 3] = FRANKA_FLANGE_TCP_OFFSET
    chain.append((T @ T_tcp).copy())
    return chain


def action_to_se3(action: np.ndarray) -> np.ndarray:
    """
    Convertit un vecteur action 7D en matrice SE(3) 4×4.

    Format action : [x, y, z, r1, r2, r3, r4, w]
    où (r1..r4) est un quaternion (qx, qy, qz, qw) et w est l'ouverture du gripper.
    On ignore w pour SE(3).

    Args:
        action: vecteur (7,) ou (8,) — les 7 premières composantes sont utilisées

    Returns:
        T: matrice SE(3) 4×4
    """
    pos = action[:3]
    quat = action[3:7]  # [qx, qy, qz, qw]
    R = quat_to_rot(quat)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


def se3_to_action(T: np.ndarray, gripper_width: float = 0.08) -> np.ndarray:
    """
    Convertit une matrice SE(3) 4×4 en vecteur action 8D.

    Returns:
        action: [x, y, z, qx, qy, qz, qw, w]
    """
    pos = T[:3, 3]
    quat = rot_to_quat(T[:3, :3])
    return np.concatenate([pos, quat, [gripper_width]])


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Quaternion [qx, qy, qz, qw] → Matrice de rotation 3×3."""
    qx, qy, qz, qw = q / (np.linalg.norm(q) + 1e-8)
    return np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ])


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Matrice de rotation 3×3 → Quaternion [qx, qy, qz, qw]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    return q / (np.linalg.norm(q) + 1e-8)


def se3_compose(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
    """Composition de deux transformations SE(3)."""
    return T1 @ T2


def se3_inverse(T: np.ndarray) -> np.ndarray:
    """Inverse d'une transformation SE(3)."""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def chamfer_distance(pc1: np.ndarray, pc2: np.ndarray) -> float:
    """
    Distance de Chamfer entre deux nuages de points.

    Args:
        pc1: (N, 3) nuage de points 1
        pc2: (M, 3) nuage de points 2

    Returns:
        chamfer_dist: scalaire
    """
    # Distance de chaque point de pc1 vers le plus proche dans pc2
    diff = pc1[:, None, :] - pc2[None, :, :]   # (N, M, 3)
    dists = np.sqrt(np.sum(diff**2, axis=-1))    # (N, M)
    d12 = np.min(dists, axis=1).mean()           # pc1 → pc2
    d21 = np.min(dists, axis=0).mean()           # pc2 → pc1
    return d12 + d21


def chamfer_distance_grad(pc1: np.ndarray, pc2: np.ndarray) -> np.ndarray:
    """
    Gradient de la distance de Chamfer par rapport à la transformation appliquée à pc1.

    Retourne le gradient par rapport aux paramètres de déplacement (translation 3D).
    Pour les rotations, utilisé en approximation linéaire autour de la pose courante.

    Args:
        pc1: (N, 3) — nuage de points du gripper transformé
        pc2: (M, 3) — nuage de points de la scène cible

    Returns:
        grad_trans: (3,) gradient par rapport à la translation
        grad_quat:  (4,) gradient par rapport au quaternion (approx.)
    """
    diff = pc1[:, None, :] - pc2[None, :, :]   # (N, M, 3)
    dists = np.sqrt(np.sum(diff**2, axis=-1) + 1e-8)  # (N, M)

    # Indices du plus proche voisin dans pc2 pour chaque point de pc1
    nn_idx_12 = np.argmin(dists, axis=1)  # (N,)
    # Indices du plus proche voisin de pc2 dans pc1
    nn_idx_21 = np.argmin(dists, axis=0)  # (M,)

    # Gradient pc1 → pc2
    nn_pc2 = pc2[nn_idx_12]  # (N, 3)
    d_vals = dists[np.arange(len(pc1)), nn_idx_12]  # (N,)
    grad_12 = (pc1 - nn_pc2) / (d_vals[:, None] + 1e-8)  # (N, 3)
    grad_trans = grad_12.mean(axis=0) / len(pc1)

    # Gradient pc2 → pc1 (par rapport au déplacement de pc1)
    nn_pc1 = pc1[nn_idx_21]  # (M, 3)
    d_vals_21 = dists[nn_idx_21, np.arange(len(pc2))]  # (M,)
    grad_21 = -(pc2 - nn_pc1) / (d_vals_21[:, None] + 1e-8)  # (M, 3)
    grad_trans += grad_21.mean(axis=0) / len(pc2)

    # Gradient quaternion (approximation : torque induit par le gradient de translation)
    centroid_pc1 = pc1.mean(axis=0)
    # Moment ≈ centroid × grad_trans (approximation du gradient de rotation)
    torque = np.cross(centroid_pc1, grad_trans)
    grad_quat = np.concatenate([torque, [0.0]])  # [qx, qy, qz, 0] (composante qw nulle)

    return grad_trans, grad_quat


def fast_global_registration(pc_gripper: np.ndarray, pc_scene: np.ndarray,
                              n_iter: int = 8,
                              rng: np.random.Generator = None) -> np.ndarray:
    """
    Fast Global Registration (FGR) simplifié.

    Alignement coarse du nuage de points du gripper vers la scène cible.
    Retourne la matrice de transformation SE(3) 4×4.

    Args:
        pc_gripper: (N, 3) nuage de points du gripper
        pc_scene:   (M, 3) nuage de points de la scène
        n_iter:     nombre d'itérations (6-10 selon le papier)
        rng:        générateur aléatoire numpy

    Returns:
        T_fgr: matrice SE(3) 4×4 (transformation gripper → scène)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    pc_src = pc_gripper.copy()
    T_total = np.eye(4)

    for _ in range(n_iter):
        # Sous-échantillonnage (max 512 points pour l'efficacité)
        n_src = min(512, len(pc_src))
        n_dst = min(512, len(pc_scene))
        idx_src = rng.choice(len(pc_src), n_src, replace=False)
        idx_dst = rng.choice(len(pc_scene), n_dst, replace=False)
        src = pc_src[idx_src]
        dst = pc_scene[idx_dst]

        # Correspondances par plus proche voisin
        diff = src[:, None, :] - dst[None, :, :]  # (n_src, n_dst, 3)
        dists = np.sum(diff**2, axis=-1)           # (n_src, n_dst)
        nn_idx = np.argmin(dists, axis=1)          # (n_src,)
        nn_dst = dst[nn_idx]                       # (n_src, 3)

        # Estimation de la transformation par SVD (méthode Kabsch)
        mu_src = src.mean(axis=0)
        mu_dst = nn_dst.mean(axis=0)
        src_c = src - mu_src
        dst_c = nn_dst - mu_dst

        H = src_c.T @ dst_c  # (3, 3)
        U, S, Vt = np.linalg.svd(H)
        d_det = np.linalg.det(Vt.T @ U.T)
        D = np.diag([1.0, 1.0, d_det])
        R_step = Vt.T @ D @ U.T
        t_step = mu_dst - R_step @ mu_src

        # Mise à jour
        T_step = np.eye(4)
        T_step[:3, :3] = R_step
        T_step[:3, 3] = t_step
        T_total = T_step @ T_total
        pc_src = (R_step @ pc_src.T).T + t_step

    return T_total


def transform_pointcloud(pc: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Applique une transformation SE(3) à un nuage de points (N, 3)."""
    pc_h = np.hstack([pc, np.ones((len(pc), 1))])  # (N, 4)
    pc_t = (T @ pc_h.T).T                          # (N, 4)
    return pc_t[:, :3]
