# ECHO Sim2Real Deployment

Deploy ECHO-generated motions to the Unitree G1 humanoid robot.

## Overview

The deployment pipeline has three components:

1. **Simulator** — `sim2sim.py` renders the robot in MuJoCo
2. **Controller** — `deploy.py` runs the ONNX tracking policy on the robot or simulator
3. **Motion Source** — `motion_select.py` (pre-recorded NPZ) or `text_to_motion.py` (cloud generator)

## Quick Start (Sim2Sim Test)

Terminal 1 — MuJoCo simulator:
```bash
python src/sim2sim.py
```

Terminal 2 — Controller (sim mode):
```bash
python src/deploy.py --sim2sim
```

Terminal 3 — Motion source (text-to-motion via cloud):
```bash
python src/text_to_motion.py
```

Key bindings: `s` = default pose, `a` = start tracking, `x` = exit.

## Real Robot Deployment

### Network Setup

Set static IP on the onboard computer:
```bash
sudo ifconfig enp4s0 192.168.123.100/24
```
Robot is at `192.168.123.161`.

### Startup

1. Turn on the robot; wait for standing pose
2. Press L2+R2 on remote controller until orange light (debug mode)
3. Run the controller:
```bash
python src/deploy.py --real
```
4. The controller loads the ONNX policy and waits for motion commands

### Text-to-Motion via Cloud

SSH tunnel to the cloud GPU server:
```bash
ssh -L 7000:127.0.0.1:8000 user@cloud-server
```

Then run:
```bash
python src/text_to_motion.py
```

Commands: type English text prompts, `up` for fall recovery, `default` for default pose, `last` to replay.

## Data Format

Deploy NPZ format:
- `joint_pos`: (T, 29) joint angles in Isaac order
- `root_pos`: (T, 3) root XYZ position (Z-up)
- `root_rot`: (T, 4) root quaternion (w, x, y, z)

Generator 38D format:
- `joint_pos(29) + root_vel_xy(2) + root_z(1) + root_rot_6d(6)`

Use `generator/utils/motion_npz_utils.py:reshape_generated_motion_38d()` to convert.

## Files

```
deploy/
├── config/
│   ├── controller.yaml    # PD gains, joint limits, motor config
│   └── tracking.yaml      # Policy path, motion list, action scales
├── src/
│   ├── deploy.py          # Main controller (real + sim)
│   ├── policy.py          # ONNX runtime inference
│   ├── observation.py     # Observation construction
│   ├── sim2sim.py         # MuJoCo simulator bridge
│   ├── motion_select.py   # Pre-recorded motion player
│   ├── text_to_motion.py  # Cloud generator WebSocket client
│   ├── paths.py           # Path resolution
│   └── common/            # Utilities (joint mapping, commands, math)
└── assets/
    └── ckpts/              # Place ONNX policy checkpoints here
```

## Dependencies

- `onnxruntime`
- `numpy`, `scipy`, `pyyaml`
- `unitree_sdk2py` (for real robot control)
- `mujoco` (for simulation)
