# Repository Architecture

This document describes the global repository structure and the role of each file 

## Project Root
- [`README.md`](/root/ENS/M2/ROBOT/SIMUL/README.md): Main guide for setup, training, simulation, and ADPro evaluation.
- [`doc/README_LAUNCH_ROBOT_POLICY.md`](/root/ENS/M2/ROBOT/SIMUL/doc/README_LAUNCH_ROBOT_POLICY.md): Multi-terminal (A/B/C/D) operating procedure for lab experiments.
- [`requirements.txt`](/root/ENS/M2/ROBOT/SIMUL/requirements.txt): Python dependencies for simulation and visualization.
- [`demo.py`](/root/ENS/M2/ROBOT/SIMUL/demo.py): ADPro vs baseline benchmarking script (figures, MSE, success rate).
- [`simulate_realtime.py`](/root/ENS/M2/ROBOT/SIMUL/simulate_realtime.py): Real-time 3D Panda visualization for baseline vs ADPro.
- [`realtime.py`](/root/ENS/M2/ROBOT/SIMUL/realtime.py): Practical real-time execution loop (policy + display).
- [`train_baseline.py`](/root/ENS/M2/ROBOT/SIMUL/train_baseline.py): Expert dataset generation and baseline policy training/saving.
- [`calibrate_visual_transform.py`](/root/ENS/M2/ROBOT/SIMUL/calibrate_visual_transform.py): Visual transform calibration utility to align frames.
- [`visualize_executed_trajectory.py`](/root/ENS/M2/ROBOT/SIMUL/visualize_executed_trajectory.py): Plots and displays the actually executed trajectory.
- [`visualize_offline_rollout.py`](/root/ENS/M2/ROBOT/SIMUL/visualize_offline_rollout.py): Visualizes an offline policy rollout without a real robot.
- [`visualize_policy_iteration.py`](/root/ENS/M2/ROBOT/SIMUL/visualize_policy_iteration.py): Visualizes iterative policy evolution over an episode.

## Simulation Environment (`env/`)

- [`env/panda_env.py`](/root/ENS/M2/ROBOT/SIMUL/env/panda_env.py): Franka Panda pick-and-grasp environment (reset/step/obs/reward).

## Models (`models/`)
- [`models/diffusion_policy.py`](/root/ENS/M2/ROBOT/SIMUL/models/diffusion_policy.py): Baseline diffusion policy (training + DDPM inference).
- [`models/adpro.py`](/root/ENS/M2/ROBOT/SIMUL/models/adpro.py): ADPro wrapper (task-aware init, manifold guidance, inference constraints).

## Geometry and Point-Cloud Utilities (`utils/`)
- [`utils/geometry.py`](/root/ENS/M2/ROBOT/SIMUL/utils/geometry.py): SE(3) transforms, kinematics, simplified FGR, geometric metrics.
- [`utils/point_cloud.py`](/root/ENS/M2/ROBOT/SIMUL/utils/point_cloud.py): Gripper/scene point-cloud generation and subsampling.

## Real-Robot Deployment (`real_robot/`)
- [`doc/README_ROBOT_CONFIG.md`](/root/ENS/M2/ROBOT/SIMUL/doc/README_ROBOT_CONFIG.md): Safe deployment procedure for a real Franka robot.
- [`real_robot/__init__.py`](/root/ENS/M2/ROBOT/SIMUL/real_robot/__init__.py): Entry point for the real-robot package.
- [`real_robot/interfaces.py`](/root/ENS/M2/ROBOT/SIMUL/real_robot/interfaces.py): Abstract sensor/control interfaces for robot backends.
- [`real_robot/policy_engine.py`](/root/ENS/M2/ROBOT/SIMUL/real_robot/policy_engine.py): Bridge from real observations to policy/ADPro inference.
- [`real_robot/safety.py`](/root/ENS/M2/ROBOT/SIMUL/real_robot/safety.py): Safety guards (workspace limits, clipping, motion checks).
- [`real_robot/control_loop.py`](/root/ENS/M2/ROBOT/SIMUL/real_robot/control_loop.py): Real-time robot control loop (observe -> policy -> command).
- [`real_robot/requirements_real_robot.txt`](/root/ENS/M2/ROBOT/SIMUL/real_robot/requirements_real_robot.txt): Dependencies specific to real-robot deployment.

## Robot Adapters (`real_robot/adapters/`)
- [`real_robot/adapters/mock_adapter.py`](/root/ENS/M2/ROBOT/SIMUL/real_robot/adapters/mock_adapter.py): Simulated/mock backend to test the loop without hardware.
- [`real_robot/adapters/ros2_franka_adapter.py`](/root/ENS/M2/ROBOT/SIMUL/real_robot/adapters/ros2_franka_adapter.py): ROS2 backend for interfacing a real Franka robot.
- [`real_robot/adapters/gello_zmq_adapter.py`](/root/ENS/M2/ROBOT/SIMUL/real_robot/adapters/gello_zmq_adapter.py): GELLO-over-ZMQ backend for teleop/sensor integration.

## Robot Configuration (`real_robot/config/`)
- [`real_robot/config/lab_franka.json`](/root/ENS/M2/ROBOT/SIMUL/real_robot/config/lab_franka.json): Lab configuration (policy, limits, backend) in JSON.
- [`real_robot/config/lab_franka.yaml`](/root/ENS/M2/ROBOT/SIMUL/real_robot/config/lab_franka.yaml): Equivalent lab configuration in YAML.

## Real-Robot Operational Scripts (`real_robot/scripts/`)
- [`real_robot/scripts/run_policy.py`](/root/ENS/M2/ROBOT/SIMUL/real_robot/scripts/run_policy.py): Main policy runner (mock/ROS2, dry-run, safety).
- [`real_robot/scripts/calibrate_local_frame_matrix.py`](/root/ENS/M2/ROBOT/SIMUL/real_robot/scripts/calibrate_local_frame_matrix.py): Calibrates local-to-robot frame transform matrix.
- [`real_robot/scripts/calibrate_workspace_guided.py`](/root/ENS/M2/ROBOT/SIMUL/real_robot/scripts/calibrate_workspace_guided.py): Guided calibration of robot workspace bounds.

## Data and Artifacts
- [`checkpoints/baseline_policy.npz`](/root/ENS/M2/ROBOT/SIMUL/checkpoints/baseline_policy.npz): Trained diffusion-policy checkpoint.
- [`checkpoints/expert_dataset.pkl`](/root/ENS/M2/ROBOT/SIMUL/checkpoints/expert_dataset.pkl): Serialized expert dataset for training/experiments.
- [`results/fig1_convergence.png`](/root/ENS/M2/ROBOT/SIMUL/results/fig1_convergence.png): Denoising convergence figure.
- [`results/fig2_action_components.png`](/root/ENS/M2/ROBOT/SIMUL/results/fig2_action_components.png): Action-component and backtracking figure.
- [`results/fig3_trajectories_3d.png`](/root/ENS/M2/ROBOT/SIMUL/results/fig3_trajectories_3d.png): 3D trajectory comparison of baseline vs ADPro.
- [`results/fig4_mse_vs_steps.png`](/root/ENS/M2/ROBOT/SIMUL/results/fig4_mse_vs_steps.png): Final MSE curve vs diffusion steps.
- [`results/fig5_success_vs_steps.png`](/root/ENS/M2/ROBOT/SIMUL/results/fig5_success_vs_steps.png): Success-rate curve vs number of steps (current version).
- [`results/fig5_success_vs_steps_paper.png`](/root/ENS/M2/ROBOT/SIMUL/results/fig5_success_vs_steps_paper.png): Success-rate curve vs steps in ADPro `paper` mode.

## Imported Work Snapshot
- `ADPro-Simulation/ADPro-Simulation/`: Imported copy from a lab session used as a synchronization snapshot.

## Reading Notes
- `.git/`, `__pycache__/`, and `*:Zone.Identifier` files are technical artifacts and not part of the core project code.
