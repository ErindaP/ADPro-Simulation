# Simulation Panda + ADPro

## Environnement Python
Utilise le venv du projet:

```bash
/root/ENS/M2/ROBOT/.venv/bin/python -V
```

## 1) Démo temps réel (fenêtre 3D)
Depuis le dossier `SIMUL`:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache /root/ENS/M2/ROBOT/.venv/bin/python simulate_realtime.py --no-train
```

Pour voir le dénoising pas-à-pas (`a_t -> a_0`) baseline vs ADPro:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache /root/ENS/M2/ROBOT/.venv/bin/python simulate_realtime.py --no-train --show-denoise --baseline-steps 40 --adpro-steps 20
```

Comparer les deux variantes ADPro:

```bash
# Version actuelle robuste (pratique)
MPLCONFIGDIR=/tmp/matplotlib-cache /root/ENS/M2/ROBOT/.venv/bin/python simulate_realtime.py --no-train --adpro-impl practical

# Version plus fidèle au papier (FGR + Chamfer, sans blend par défaut)
MPLCONFIGDIR=/tmp/matplotlib-cache /root/ENS/M2/ROBOT/.venv/bin/python simulate_realtime.py --no-train --adpro-impl paper
```

Options utiles:

```bash
--baseline-steps 40 --adpro-steps 20 --max-env-steps 12 --frame-sleep 0.15
```

## 2) (Optionnel) Réentraîner la baseline
Si le checkpoint n'existe pas:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache /root/ENS/M2/ROBOT/.venv/bin/python simulate_realtime.py
```

## 3) Démo figures offline

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache /root/ENS/M2/ROBOT/.venv/bin/python demo.py --no-train --n-eval 10
```

Les figures sont écrites dans `results/`.
