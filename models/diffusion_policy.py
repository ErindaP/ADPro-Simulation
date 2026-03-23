"""
diffusion_policy.py — Politique de diffusion 3D basique pour le Franka Panda

Implémente une politique de diffusion simplifiée inspirée de 3D Diffuser Actor :
- Encodeur de nuages de points (PointNet simplifié avec max-pooling)
- Réseau de prédiction de bruit ε_θ(O, a_t, t)
- Dénoising DDPM standard
- Entraînement stable par Adam avec gradient clipping

Référence : Chi et al. (2023), "Diffusion Policy"
             Ze et al. (2024), "3D Diffusion Policy"
"""

import numpy as np
from tqdm import tqdm


# ============================================================
# Couche MLP numpy (réseau de neurones simple)
# ============================================================

class Linear:
    """Couche linéaire dense."""
    def __init__(self, in_dim: int, out_dim: int, rng: np.random.Generator):
        # Initialisation He (variance = 2/fan_in)
        scale = np.sqrt(2.0 / in_dim)
        self.W = rng.normal(0, scale, (out_dim, in_dim)).astype(np.float64)
        self.b = np.zeros(out_dim, dtype=np.float64)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.W.T + self.b

    def params(self):
        return [self.W, self.b]


class MLP:
    """Réseau de neurones multi-couches avec activation ReLU."""

    def __init__(self, dims: list, rng: np.random.Generator):
        self.layers = []
        for i in range(len(dims) - 1):
            self.layers.append(Linear(dims[i], dims[i+1], rng))

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers[:-1]:
            x = np.maximum(0, layer.forward(x))  # ReLU
        x = self.layers[-1].forward(x)
        return x

    def params(self):
        p = []
        for layer in self.layers:
            p.extend(layer.params())
        return p


# ============================================================
# Encodeur de nuage de points (PointNet simplifié)
# ============================================================

class PointNetEncoder:
    """
    Encodeur PointNet simplifié : MLP par point + max-pooling global.
    Produit un descripteur global de taille fixe.
    """

    def __init__(self, in_dim: int = 3, out_dim: int = 32,
                 rng: np.random.Generator = None):
        if rng is None:
            rng = np.random.default_rng(0)
        # Architecture réduite pour stabilité : 3→32→out_dim
        self.mlp = MLP([in_dim, 32, out_dim], rng)
        self.out_dim = out_dim

    def forward(self, pc: np.ndarray) -> np.ndarray:
        """
        Args:
            pc: (N, 3) nuage de points

        Returns:
            feat: (out_dim,) descripteur global
        """
        # Normalisation locale
        center = pc.mean(axis=0)
        scale = max(np.linalg.norm(pc - center, axis=1).max(), 1e-6)
        pc_norm = (pc - center) / scale

        # Sous-échantillonnage pour la vitesse (max 64 points)
        if len(pc_norm) > 64:
            step = len(pc_norm) // 64
            pc_norm = pc_norm[::step][:64]

        # MLP par point → max-pooling
        f = self.mlp.forward(pc_norm)  # (N, out_dim)
        return f.max(axis=0)           # (out_dim,)

    def params(self):
        return self.mlp.params()


# ============================================================
# Réseau de prédiction de bruit ε_θ
# ============================================================

ACTION_DIM = 7       # [x, y, z, qx, qy, qz, qw]
T_DIFFUSION = 100    # Nombre d'étapes de diffusion

class NoisePredictionNetwork:
    """
    Réseau ε_θ(O, a_t, t) qui prédit le bruit ajouté à l'action.

    Entrée : [encodeur_O0, encodeur_O1, a_t, embedding_t]
    Sortie : ε_t prédite (même dimension que a_t)

    Inspiré de la structure de 3D Diffuser Actor.
    """

    def __init__(self, pc_feat_dim: int = 32, action_dim: int = ACTION_DIM,
                 T: int = T_DIFFUSION, rng: np.random.Generator = None):
        if rng is None:
            rng = np.random.default_rng(0)
        self.rng = rng
        self.action_dim = action_dim
        self.T = T
        self.pc_feat_dim = pc_feat_dim

        # Encodeurs de nuages de points (gelés pendant l'entraînement)
        self.enc_gripper = PointNetEncoder(3, pc_feat_dim, rng)
        self.enc_scene   = PointNetEncoder(3, pc_feat_dim, rng)

        # Embedding de l'étape de diffusion t (sinusoïdal, fixe)
        self.t_embed_dim = 16

        # MLP principal (entraîné) : [O0_feat, O1_feat, a_t, t_embed] → ε_t
        total_in = pc_feat_dim + pc_feat_dim + action_dim + self.t_embed_dim
        # Architecture réduite pour stabilité
        self.noise_mlp = MLP([total_in, 128, 64, action_dim], rng)

    def _time_embedding(self, t: int) -> np.ndarray:
        """Embedding sinusoïdal normalisé de l'étape de diffusion."""
        d = self.t_embed_dim
        t_norm = t / self.T  # Normaliser dans [0, 1]
        freqs = np.exp(-np.log(1000) * np.arange(0, d, 2) / d)
        emb = np.zeros(d)
        emb[0::2] = np.sin(t_norm * freqs)
        emb[1::2] = np.cos(t_norm * freqs[:d//2])
        return emb

    def forward(self, obs: dict, a_t: np.ndarray, t: int) -> np.ndarray:
        """
        Prédit le bruit ε_t.

        Args:
            obs:  dict avec 'pc_gripper' et 'pc_scene'
            a_t:  action bruitée (action_dim,)
            t:    étape de diffusion (int)

        Returns:
            eps_pred: (action_dim,) bruit prédit
        """
        f0 = self.enc_gripper.forward(obs['pc_gripper'])
        f1 = self.enc_scene.forward(obs['pc_scene'])
        t_emb = self._time_embedding(t)

        # Normaliser a_t
        a_t_norm = np.clip(a_t, -10.0, 10.0)

        x = np.concatenate([f0, f1, a_t_norm, t_emb])
        eps_pred = self.noise_mlp.forward(x)
        return eps_pred

    def params(self):
        """Retourne tous les paramètres entraînables (seulement noise_mlp)."""
        return self.noise_mlp.params()


# ============================================================
# Schedule DDPM
# ============================================================

def make_ddpm_schedule(T: int = T_DIFFUSION, beta_start: float = 1e-4,
                        beta_end: float = 0.02) -> dict:
    """Calcule les coefficients du schedule DDPM linéaire."""
    betas = np.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    return {
        'betas': betas,
        'alphas': alphas,
        'alpha_bars': alpha_bars,
        'sqrt_alpha_bars': np.sqrt(alpha_bars),
        'sqrt_one_minus_alpha_bars': np.sqrt(1.0 - alpha_bars),
    }


# ============================================================
# Politique de diffusion (inférence DDPM standard)
# ============================================================

class DiffusionPolicy:
    """
    Politique de diffusion vanilla (DDPM).

    Algorithme de dénoising (Eq. 1 du papier ADPro) :
        a_{t-1} = 1/√α_t · (a_t - β_t/√(1-ᾱ_t) · ε_θ) + σ_t · z
    """

    def __init__(self, network: NoisePredictionNetwork = None,
                 T: int = T_DIFFUSION, rng: np.random.Generator = None):
        self.T = T
        self.rng = rng or np.random.default_rng(0)
        self.schedule = make_ddpm_schedule(T)
        self.network = network or NoisePredictionNetwork(rng=self.rng)
        self.action_dim = self.network.action_dim
        self.losses = []

    def add_noise(self, action: np.ndarray, t: int,
                  rng: np.random.Generator = None) -> tuple:
        """
        Ajoute du bruit selon le schedule DDPM.
        q(a_t | a_0) = N(√ᾱ_t a_0, (1-ᾱ_t) I)
        """
        if rng is None:
            rng = self.rng
        sqrt_abar = self.schedule['sqrt_alpha_bars'][t]
        sqrt_1m   = self.schedule['sqrt_one_minus_alpha_bars'][t]
        eps = rng.normal(0, 1, action.shape)
        a_t = sqrt_abar * action + sqrt_1m * eps
        return a_t, eps

    def ddpm_step(self, a_t: np.ndarray, eps_pred: np.ndarray,
                   t: int, add_noise: bool = True) -> np.ndarray:
        """
        Un pas de dénoising DDPM.
        a_{t-1} = (1/√α_t) * (a_t - β_t/√(1-ᾱ_t) * ε_θ) + σ_t * z
        """
        alpha_t = self.schedule['alphas'][t]
        beta_t  = self.schedule['betas'][t]
        sqrt_1m = self.schedule['sqrt_one_minus_alpha_bars'][t]

        # Clipper eps_pred pour éviter l'explosion
        eps_pred_c = np.clip(eps_pred, -5.0, 5.0)
        mean = (1 / np.sqrt(alpha_t)) * (a_t - (beta_t / (sqrt_1m + 1e-8)) * eps_pred_c)

        if add_noise and t > 0:
            sigma_t = np.sqrt(beta_t)
            z = self.rng.normal(0, 1, a_t.shape)
            return mean + sigma_t * z
        return mean

    def inference(self, obs: dict, n_steps: int = None,
                  return_trajectory: bool = False) -> np.ndarray:
        """
        Génère une action par dénoising DDPM standard.

        Args:
            obs:               dict d'observations (pc_gripper, pc_scene)
            n_steps:           nombre d'étapes (défaut : T)
            return_trajectory: si True, retourne aussi la trajectoire de dénoising

        Returns:
            a_0: action finale (action_dim,)
        """
        T = n_steps or self.T
        a_t = self.rng.normal(0, 1, self.action_dim)

        trajectory = [a_t.copy()] if return_trajectory else None

        for t in reversed(range(T)):
            eps_pred = self.network.forward(obs, a_t, t)
            a_t = self.ddpm_step(a_t, eps_pred, t, add_noise=(t > 0))
            if return_trajectory:
                trajectory.append(a_t.copy())

        return (a_t, trajectory) if return_trajectory else a_t


# ============================================================
# Optimiseur Adam pour numpy
# ============================================================

class AdamOptimizer:
    """Optimiseur Adam stable pour paramètres numpy."""

    def __init__(self, params: list, lr: float = 3e-4,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads: list):
        """Une étape de mise à jour Adam."""
        self.t += 1
        lr_t = self.lr * np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t + 1e-12)
        for i, (p, g) in enumerate(zip(self.params, grads)):
            if g is None or np.any(~np.isfinite(g)):
                continue
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g**2
            update = lr_t * self.m[i] / (np.sqrt(self.v[i]) + self.eps)
            p -= np.clip(update, -0.1, 0.1)  # Clip pour stabilité


# ============================================================
# Backpropagation manuelle sur le MLP
# ============================================================

def _forward_with_cache(mlp: MLP, x: np.ndarray) -> tuple:
    """Forward pass avec stockage des activations."""
    activations = [x.copy()]
    current = x
    for layer in mlp.layers[:-1]:
        current = layer.forward(current)
        current = np.maximum(0, current)
        activations.append(current.copy())
    out = mlp.layers[-1].forward(current)
    return out, activations


def _backprop_mlp(mlp: MLP, activations: list, delta: np.ndarray,
                   clip_norm: float = 1.0) -> list:
    """Rétropropagation sur le MLP. Retourne les gradients par paramètre."""
    delta = np.clip(delta, -5.0, 5.0)
    flat_grads = []

    # Couche finale (sortie)
    last = mlp.layers[-1]
    gW = np.outer(delta, activations[-1])
    gb = delta.copy()
    gW_norm = np.linalg.norm(gW)
    if gW_norm > clip_norm:
        gW *= clip_norm / gW_norm
    flat_grads = [(gW, gb)]
    delta = last.W.T @ delta

    # Couches intermédiaires
    for i in range(len(mlp.layers) - 2, -1, -1):
        layer = mlp.layers[i]
        relu_mask = (activations[i + 1] > 0).astype(float)
        delta = delta * relu_mask
        delta = np.clip(delta, -5.0, 5.0)

        gW = np.outer(delta, activations[i])
        gb = delta.copy()
        gW_norm = np.linalg.norm(gW)
        if gW_norm > clip_norm:
            gW *= clip_norm / gW_norm
        flat_grads.insert(0, (gW, gb))
        delta = layer.W.T @ delta

    # Aplatir en liste [W0, b0, W1, b1, ...]
    result = []
    for gW, gb in flat_grads:
        result.extend([gW, gb])
    return result


# ============================================================
# Entraînement stable
# ============================================================

def train_diffusion_policy(
    dataset: list,
    n_epochs: int = 50,
    lr: float = 3e-4,
    T: int = T_DIFFUSION,
    seed: int = 0,
    verbose: bool = True,
) -> DiffusionPolicy:
    """
    Entraîne la politique de diffusion par behavior cloning avec Adam.

    Approche stable :
    - Seulement le MLP principal est entraîné (encodeurs figés)
    - Normalisation des nuages de points dans PointNetEncoder
    - Adam optimizer avec gradient clipping (clip 1.0 par couche)
    - Loss clippée pour éviter les explosions numériques

    Args:
        dataset:  liste de {'obs': ..., 'action': np.ndarray(7)}
        n_epochs: nombre d'époques
        lr:       taux d'apprentissage
        T:        nombre d'étapes DDPM
        seed:     graine aléatoire

    Returns:
        policy: politique entraînée
    """
    rng = np.random.default_rng(seed)
    network = NoisePredictionNetwork(pc_feat_dim=32, rng=rng)
    policy = DiffusionPolicy(network=network, T=T, rng=rng)
    schedule = policy.schedule

    # Optimiser seulement le MLP principal
    mlp = network.noise_mlp
    opt_params = []
    for layer in mlp.layers:
        opt_params.extend([layer.W, layer.b])
    optimizer = AdamOptimizer(opt_params, lr=lr)

    losses = []
    indices = list(range(len(dataset)))

    if verbose:
        print(f"Entraînement de la politique de diffusion ({len(dataset)} échantillons)...")
        print(f"  Architecture : PointNet(32) + MLP[{network.noise_mlp.layers[0].W.shape[1]}→128→64→7]")
        print(f"  T_DDPM={T} | Adam lr={lr}")
        print()

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_valid = 0
        rng.shuffle(indices)

        for idx in indices:
            sample = dataset[idx]
            action = sample['action'][:network.action_dim].astype(np.float64)

            if not np.all(np.isfinite(action)):
                continue
            action = np.clip(action, -5.0, 5.0)

            # Timestep aléatoire
            t = int(rng.integers(0, T))

            # Ajouter du bruit (forward diffusion)
            sqrt_abar = schedule['sqrt_alpha_bars'][t]
            sqrt_1m   = schedule['sqrt_one_minus_alpha_bars'][t]
            eps_true  = rng.normal(0, 1, network.action_dim)
            a_t = sqrt_abar * action + sqrt_1m * eps_true

            # Encoder la scène (une fois, figé)
            obs = sample['obs']
            f0 = network.enc_gripper.forward(obs['pc_gripper'])
            f1 = network.enc_scene.forward(obs['pc_scene'])
            t_emb = network._time_embedding(t)
            x_in = np.concatenate([f0, f1, np.clip(a_t, -10.0, 10.0), t_emb])

            if not np.all(np.isfinite(x_in)):
                continue

            # Forward sur le MLP principal
            eps_pred, activations = _forward_with_cache(mlp, x_in)

            if not np.all(np.isfinite(eps_pred)):
                continue

            # Loss MSE (clippée)
            err = eps_pred - eps_true
            loss = float(np.mean(err**2))
            if loss > 100.0 or not np.isfinite(loss):
                continue

            epoch_loss += loss
            n_valid += 1

            # Backprop et mise à jour Adam
            grads = _backprop_mlp(mlp, activations, err * 2, clip_norm=1.0)
            optimizer.step(grads)

        avg_loss = epoch_loss / max(n_valid, 1)
        losses.append(avg_loss)

        if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
            print(f"  Epoch {epoch+1:3d}/{n_epochs} | Loss: {avg_loss:.4f} ({n_valid}/{len(dataset)} valides)")

    policy.losses = losses
    if verbose:
        print(f"\n  Entraînement terminé. Loss finale : {losses[-1]:.4f}")
    return policy
