# Running ADPro on Franka (Terminals A/B/C/D)

This guide describes exactly what to run in each terminal to execute ADPro with the `gello` backend (without ROS2).

## Prerequisites

- FCI enabled in Franka Desk.
- Robot unlocked and ready.
- Robotiq gripper connected.
- `gello` conda environment available.

---

## Terminal A — Polymetis Robot Server

```bash
source ~/miniconda3/bin/activate
conda activate gello
launch_robot.py robot_client=franka_hardware robot_client.executable_cfg.robot_ip=192.168.10.100
```

Keep this terminal open during the full run.

---

## Terminal B — Robotiq Gripper Server

If needed (serial-port permissions):
```bash
sudo chmod 666 /dev/ttyUSB0
sudo chmod 666 /dev/serial/by-id/usb-FTDI_USB_TO_RS-485_DAV0KVG-if00-port0
```

Then launch (recommended `by-id` port):
```bash
source ~/miniconda3/bin/activate
conda activate gello
launch_gripper.py gripper=robotiq_2f gripper.comport=/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_DAV0KVG-if00-port0
```

Keep this terminal open during the full run.

---

## Terminal C — GELLO ZMQ Bridge (port 6001)

```bash
source ~/miniconda3/bin/activate
conda activate gello
cd /home/panda/gello_software
python experiments/launch_nodes.py --robot=panda --robot_ip=localhost --robot-port=6001
```

Keep this terminal open during the full run.

---

## Terminal D — ADPro Execution

```bash
source ~/miniconda3/bin/activate
conda activate gello
cd /home/panda/gello_software/adpro_test/ADPro-Simulation
python real_robot/scripts/run_policy.py --backend gello --config real_robot/config/lab_franka.json --run-seconds 5
```

Optional dry-run command:

```bash
python real_robot/scripts/run_policy.py --backend gello --config real_robot/config/lab_franka.json --run-seconds 5 --dry-run
```

---

## Software Emergency Stop (if needed)

```bash
source ~/miniconda3/bin/activate
conda activate gello
sudo pkill -9 -f "real_robot/scripts/run_policy.py"
sudo pkill -9 -f "experiments/launch_nodes.py"
sudo pkill -9 -f "launch_gripper.py"
sudo pkill -9 -f "launch_robot.py"
sudo pkill -9 -f "franka_panda_client"
sudo pkill -9 run_server
```

Then verify:

```bash
ps -ef | egrep "run_policy.py|launch_nodes.py|launch_gripper.py|launch_robot.py|franka_panda_client|run_server" | grep -v grep
```

---

## Clean Restart Order

Always restart in this order:

1. Terminal A
2. Terminal B
3. Terminal C
4. Terminal D

---

## Real-time position monitor

```bash
cd /home/panda/gello_software/adpro_test/ADPro-Simulation
python realtime.py
```
