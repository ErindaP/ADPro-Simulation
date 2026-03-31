# Panda + ADPro Simulation

## Python Environment
Use the project virtual environment:

```bash
/root/ENS/M2/ROBOT/.venv/bin/python -V
```

## 1) Real-time Demo (3D window)
From the `ADPro_Simulation` folder:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache /root/ENS/M2/ROBOT/.venv/bin/python simulate_realtime.py --no-train
```

To visualize denoising step by step (`a_t -> a_0`) for baseline vs ADPro:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache /root/ENS/M2/ROBOT/.venv/bin/python simulate_realtime.py --no-train --show-denoise --baseline-steps 40 --adpro-steps 20
```

Compare both ADPro variants:

```bash
# Current robust variant (practical)
MPLCONFIGDIR=/tmp/matplotlib-cache /root/ENS/M2/ROBOT/.venv/bin/python simulate_realtime.py --no-train --adpro-impl practical

# More paper-faithful variant (FGR + Chamfer, no blend by default)
MPLCONFIGDIR=/tmp/matplotlib-cache /root/ENS/M2/ROBOT/.venv/bin/python simulate_realtime.py --no-train --adpro-impl paper
```

Useful options:

```bash
--baseline-steps 40 --adpro-steps 20 --max-env-steps 12 --frame-sleep 0.15
```

## 2) (Optional) Retrain the Baseline
If no checkpoint exists:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache /root/ENS/M2/ROBOT/.venv/bin/python simulate_realtime.py
```

## 3) Offline Figures Demo

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache /root/ENS/M2/ROBOT/.venv/bin/python demo.py --no-train --n-eval 10
```

Figures are written to `results/`.

## 4) Real Robot Layer (Franka)

The deployment layer is separated under `real_robot/`.

Local dry-run (without robot):

```bash
/root/ENS/M2/ROBOT/.venv/bin/python real_robot/scripts/run_policy.py --backend mock --config real_robot/config/lab_franka.json --dry-run --run-seconds 5
```

Full guide:

- `doc/README_ROBOT_CONFIG.md`

### ROS2-free variant: GELLO backend (ZMQ + Polymetis)

If `ros2` is not available on the lab machine, ADPro can run on top of the existing GELLO stack.

1) In one terminal, start robot/GELLO servers:
```bash
source ~/miniconda3/bin/activate
conda activate gello
python experiments/launch_nodes.py --robot=panda --robot_ip=localhost
```

2) In a second terminal, run ADPro with the GELLO backend:
```bash
source ~/miniconda3/bin/activate
conda activate gello
python adpro_test/ADPro-Simulation/real_robot/scripts/run_policy.py \
  --backend gello \
  --config adpro_test/ADPro-Simulation/real_robot/config/lab_franka.json \
  --run-seconds 5
```
