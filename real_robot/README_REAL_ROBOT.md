# Real Robot Layer (Franka Panda)

Ce dossier est une couche séparée de la simulation pour exécuter ADPro sur un robot réel.

## Architecture

- `policy_engine.py`: charge le checkpoint et exécute ADPro.
- `safety.py`: filtre de sécurité (workspace, pas max, lissage).
- `control_loop.py`: boucle de contrôle temps réel (policy -> safety -> commande).
- `adapters/ros2_franka_adapter.py`: bridge ROS2 (topics perception/commande).
- `adapters/mock_adapter.py`: backend local sans robot.
- `config/lab_franka.json`: configuration par défaut (sans dépendance externe).
- `config/lab_franka.yaml`: variante YAML (optionnelle).

## Lancement local (sans robot)

```bash
cd /root/ENS/M2/ROBOT/SIMUL
/root/ENS/M2/ROBOT/.venv/bin/python real_robot/scripts/run_policy.py \
  --backend mock \
  --config real_robot/config/lab_franka.json \
  --dry-run \
  --run-seconds 5
```

## Lancement labo (robot connecté)

1. Cloner le repo sur la machine labo.
2. Adapter `real_robot/config/lab_franka.json`:
- topics ROS2,
- workspace sécurité,
- fréquence de boucle,
- paramètres ADPro.
3. Vérifier d'abord en `--dry-run`.
4. Puis exécuter:

```bash
python3 real_robot/scripts/run_policy.py \
  --backend ros2 \
  --config real_robot/config/lab_franka.json
```

## Notes sécurité

- Cette couche n'est **pas** un remplacement du safety natif Franka.
- Toujours garder les limites collision/force strictes côté contrôleur bas niveau.
- Commencer par des vitesses faibles et espace libre.
- Tester d'abord `--run-seconds 5` avant un run long.
