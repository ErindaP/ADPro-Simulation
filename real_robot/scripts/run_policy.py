#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import json
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from real_robot.adapters.mock_adapter import MockObservationProvider, MockCommander
from real_robot.adapters.ros2_franka_adapter import Ros2Topics, build_ros2_stack
from real_robot.control_loop import run_control_loop
from real_robot.policy_engine import PolicyEngine
from real_robot.safety import SafetyConfig, SafetyFilter


def load_cfg(path: str) -> dict:
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Optional YAML support if PyYAML is installed on lab machine
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Config YAML demandée mais PyYAML n'est pas installé. "
            "Utilise un fichier .json ou installe PyYAML."
        ) from exc

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Run ADPro policy on real robot bridge")
    parser.add_argument("--config", default="real_robot/config/lab_franka.json")
    parser.add_argument("--backend", choices=["mock", "ros2"], default="mock")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-seconds", type=float, default=None,
                        help="Override loop.run_seconds from config.")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    policy_cfg = cfg["policy"]
    loop_cfg = cfg["loop"]
    safety_cfg = cfg["safety"]

    policy = PolicyEngine(
        checkpoint_path=policy_cfg["checkpoint_path"],
        seed=int(policy_cfg.get("seed", 42)),
        adpro_impl=policy_cfg.get("adpro_impl", "paper"),
        adpro_steps=int(policy_cfg.get("adpro_steps", 20)),
        adpro_M=int(policy_cfg.get("adpro_M", 60)),
        adpro_spherical_scale=float(policy_cfg.get("adpro_spherical_scale", 0.2)),
        adpro_stochastic=bool(policy_cfg.get("adpro_stochastic", False)),
    )

    safety = SafetyFilter(
        SafetyConfig(
            workspace_min=np.array(safety_cfg["workspace_min"], dtype=np.float64),
            workspace_max=np.array(safety_cfg["workspace_max"], dtype=np.float64),
            max_translation_step_m=float(safety_cfg["max_translation_step_m"]),
            max_rotation_step_rad=float(safety_cfg["max_rotation_step_rad"]),
            lowpass_alpha=float(safety_cfg["lowpass_alpha"]),
        )
    )

    ros_node = None
    if args.backend == "mock":
        provider = MockObservationProvider(seed=int(policy_cfg.get("seed", 42)))
        commander = MockCommander(provider)
    else:
        topics_cfg = cfg["ros2_topics"]
        topics = Ros2Topics(
            pc_gripper_topic=topics_cfg["pc_gripper_topic"],
            pc_scene_topic=topics_cfg["pc_scene_topic"],
            ee_pose_topic=topics_cfg["ee_pose_topic"],
            target_pose_topic=topics_cfg["target_pose_topic"],
            gripper_width_topic=topics_cfg["gripper_width_topic"],
        )
        ros_node, provider, commander = build_ros2_stack(topics)

    try:
        run_seconds = float(loop_cfg.get("run_seconds", -1.0))
        if args.run_seconds is not None:
            run_seconds = float(args.run_seconds)

        run_control_loop(
            provider=provider,
            commander=commander,
            policy=policy,
            safety=safety,
            policy_hz=float(loop_cfg.get("policy_hz", 5.0)),
            run_seconds=run_seconds,
            dry_run=bool(args.dry_run),
        )
    finally:
        provider.close()
        commander.close()
        if ros_node is not None:
            import rclpy
            ros_node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
