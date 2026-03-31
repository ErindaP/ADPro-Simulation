# Real Robot Layer (Franka Panda)

This folder is a layer separated from simulation to run ADPro on a real robot.

## Architecture

- `policy_engine.py`: loads the checkpoint and runs ADPro.
- `safety.py`: safety filter (workspace, max step, smoothing).
- `control_loop.py`: real-time control loop (policy -> safety -> command).
- `adapters/ros2_franka_adapter.py`: ROS2 bridge (perception/command topics).
- `adapters/mock_adapter.py`: local backend without robot.
- `adapters/gello_zmq_adapter.py`: GELLO/ZMQ backend for lab deployment.
- `config/lab_franka.json`: default config (no external dependency).
- `config/lab_franka.yaml`: optional YAML variant.

## Local Run (without robot)

```bash
cd /root/ENS/M2/ROBOT/SIMUL
/root/ENS/M2/ROBOT/.venv/bin/python real_robot/scripts/run_policy.py \
  --backend mock \
  --config real_robot/config/lab_franka.json \
  --dry-run \
  --run-seconds 5
```

## Lab Run (robot connected)

1. Clone the repository on the lab machine.
2. Update `real_robot/config/lab_franka.json`:
   - ROS2 topics,
   - safety workspace,
   - control-loop frequency,
   - ADPro parameters.
3. Validate first with `--dry-run`.
4. Then run:

```bash
python3 real_robot/scripts/run_policy.py \
  --backend ros2 \
  --config real_robot/config/lab_franka.json
```

## Lab Run via GELLO (without ROS2)

This mode reuses the existing `experiments/launch_nodes.py` + ZMQ stack from the main project.

1. Start the GELLO robot server:
```bash
source ~/miniconda3/bin/activate
conda activate gello
python experiments/launch_nodes.py --robot=panda --robot_ip=localhost
```

2. In another terminal, run ADPro:
```bash
source ~/miniconda3/bin/activate
conda activate gello
python adpro_test/ADPro-Simulation/real_robot/scripts/run_policy.py \
  --backend gello \
  --config adpro_test/ADPro-Simulation/real_robot/config/lab_franka.json \
  --run-seconds 5
```

3. Parameters to tune in `config/lab_franka.json` under `gello`:
- `host`, `robot_port`: ZMQ robot-server endpoint.
- `obj_pos`: estimated target-object position (used to generate `pc_scene`).
- `ik_iters`: IK iterations for EE pose -> joints conversion.

## Safety Notes

- This layer is **not** a replacement for native Franka safety.
- Always keep strict collision/force limits at low-level control.
- Start with low speeds and free workspace.
- Test first with `--run-seconds 5` before long runs.
