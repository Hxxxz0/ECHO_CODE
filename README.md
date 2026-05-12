# ECHO: Edge-Cloud Humanoid Orchestration for Language-to-Motion Control

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://echo-phi-eight.vercel.app)
[![ModelScope](https://img.shields.io/badge/Model-Weights-purple)](https://modelscope.cn/models/Hzzzz001/ECHO/summary)
[![Paper](https://img.shields.io/badge/arXiv-2603.16188-b31b1b)](https://arxiv.org/pdf/2603.16188)

**ECHO** is an edge–cloud framework for language-driven whole-body control of humanoid robots. A cloud-hosted diffusion-based text-to-motion generator synthesizes motion references from natural language, while an edge-deployed RL tracker executes them in closed loop on the **Unitree G1** humanoid.

## Overview

<p align="center">
  <img src="https://echo-phi-eight.vercel.app/static/images/carousel1.jpg" alt="ECHO Overview" width="100%">
</p>

**ECHO** processes natural language instructions through a CLIP-conditioned diffusion model on a cloud GPU, producing 38D robot-native motion sequences in ~1 second. The motion is streamed via WebSocket to an edge-deployed ONNX tracking policy that runs at 50 Hz on the G1 with PD control and autonomous fall recovery.

---

## Demo Videos

### Real Robot (Unitree G1)

<p align="center">
  <video src="https://echo-phi-eight.vercel.app/real-preview/walk%205%20step.mp4" muted autoplay loop playsinline width="22%"></video>
  <video src="https://echo-phi-eight.vercel.app/real-preview/do%20jumping%20jacks.mp4" muted autoplay loop playsinline width="22%"></video>
  <video src="https://echo-phi-eight.vercel.app/real-preview/wave%20right%20hand.mp4" muted autoplay loop playsinline width="22%"></video>
  <video src="https://echo-phi-eight.vercel.app/real-preview/walk%20in%20a%20circle.mp4" muted autoplay loop playsinline width="22%"></video>
</p>

### Simulation (MuJoCo)

<p align="center">
  <video src="https://echo-phi-eight.vercel.app/sim-preview/walk%205%20step.mp4" muted autoplay loop playsinline width="22%"></video>
  <video src="https://echo-phi-eight.vercel.app/sim-preview/fly%20kick.mp4" muted autoplay loop playsinline width="22%"></video>
  <video src="https://echo-phi-eight.vercel.app/sim-preview/a%20person%20is%20drinking%20water.mp4" muted autoplay loop playsinline width="22%"></video>
  <video src="https://echo-phi-eight.vercel.app/sim-preview/he%20is%20running%20straight%20and%20stopped.mp4" muted autoplay loop playsinline width="22%"></video>
</p>

> More videos on the [project page](https://echo-phi-eight.vercel.app).

---

## Key Features

- **Robot-native**: generates directly in G1 29-DOF joint space — no human body model, no retargeting
- **38D velocity-based representation**: joint angles + root velocity + root height + continuous 6D rotation
- **Classifier-free guidance**: DDIM sampling with 10 denoising steps produces motions in ~1 second on cloud GPU
- **Edge deployment**: ONNX tracking policy runs on CPU at 50 Hz with PD control and autonomous fall recovery

---

## Installation

```bash
conda create -n echo python=3.10 -y
conda activate echo
conda install pytorch pytorch-cuda=12.8 -c pytorch -c nvidia -y
pip install -r generator/requirements.txt

# For WebSocket server
pip install -r generator/requirements_server.txt
```

## Model Weights

Download from ModelScope ([Hzzzz001/ECHO](https://modelscope.cn/models/Hzzzz001/ECHO/summary)):

```bash
git clone https://www.modelscope.cn/Hzzzz001/ECHO.git checkpoints/
```

| Checkpoint | Backbone | Dim | Inference |
|-----------|----------|-----|-----------|
| `checkpoints/robotv2/robotv2_38d_lite` | UNet (small) | 128 | ~1.0s |
| `checkpoints/robotv2/robotv2_38d` | UNet (full) | 512 | ~1.5s |
| `checkpoints/robotv2/robotv2_38d_transformer` | Transformer | 768 | ~3.0s |

Each checkpoint: `opt.txt` (config), `model/latest.tar` (weights), `meta/{mean,std}.npy` (normalization).

Normalization stats for the dataset: `generator/data/Mean_38d.npy`, `generator/data/Std_38d.npy`.

---

## Usage

### Generate motion from text

```bash
cd generator
python scripts/generate_robot.py \
    --opt_path ../checkpoints/checkpoints/robotv2/robotv2_38d_lite/opt.txt \
    --text_prompt "a person walks forward" \
    --motion_length 4.0 \
    --output_dir ./output
```

Output: `output/npz/000000.npz` with `joint_pos (T,29)`, `root_pos (T,3)`, `root_rot (T,4)`.

### Start WebSocket server (cloud)

```bash
cd generator
python scripts/server_robot_ws.py \
    --opt_path ../checkpoints/checkpoints/robotv2/robotv2_38d_lite/opt.txt \
    --port 8000 --host 127.0.0.1
```

Health check: `curl http://127.0.0.1:8000/` → `{"status":"running","service":"ECHO Motion Generation Server"}`

WebSocket API: connect to `ws://127.0.0.1:8000/ws`, send JSON request, receive binary NPZ.

```json
{"text": "walk forward slowly", "motion_length": 4.0, "num_inference_steps": 10, "seed": 42}
```

Remote access via SSH tunnel: `ssh -L 8000:127.0.0.1:8000 user@cloud-server`

See [generator/docs/WEBSOCKET_QUICKSTART.md](generator/docs/WEBSOCKET_QUICKSTART.md) and [generator/docs/CLIENT_API.md](generator/docs/CLIENT_API.md) for details.

### Evaluate model

```bash
cd generator
python scripts/evaluation.py \
    --opt_path ../checkpoints/checkpoints/robotv2/robotv2_38d/opt.txt \
    --evaluator_dir ../checkpoints/checkpoints/robot_evaluator
```

Metrics: FID, R-Precision Top-1/2/3, Matching Score, Diversity, Multimodality, Motion Safety Score (MSS), Root Trajectory Consistency (RTC).

### Train generator

Requires preprocessed 38D robot motion data.

```bash
cd generator
accelerate launch scripts/train.py \
    --dataset_name robotv2 \
    --name robotv2_experiment \
    --batch_size 64 \
    --num_train_steps 500000 \
    --model_ema \
    --model_type unet \
    --base_dim 512 \
    --lr 2e-4
```

### Deploy to G1 robot

See [deploy/README.md](deploy/README.md) — sim2sim test, real robot setup, text-to-motion client, and ONNX policy inference.

---

## 38D Motion Representation

| Index | Field | Dims | Description |
|-------|-------|------|-------------|
| 0–28 | `joint_pos` | 29 | Joint angles (rad) in Isaac Gym order |
| 29–30 | `root_vel_xy` | 2 | Root planar velocity in body frame |
| 31 | `root_z` | 1 | Root height above ground (m) |
| 32–37 | `root_rot_6d` | 6 | Continuous 6D root rotation |

50 FPS, max 490 frames (~9.8s). Velocity-based root motion avoids global drift. 6D rotation prevents gimbal lock.

## Project Structure

```
ECHO_CODE/
├── generator/                  # Cloud diffusion generator
│   ├── models/                 # EchoUnet (1D Conv), Transformer
│   │   ├── unet.py             # CondUNet1D + AdaGN + cross-attention
│   │   ├── transformer.py      # Decoder-only diffusion transformer
│   │   └── gaussian_diffusion.py  # DDIM/DPMSolver inference pipeline
│   ├── datasets/               # 38D robot motion dataset loader
│   ├── trainers/               # DDPM training loop + EMA
│   ├── eval/                   # MoCLIP, MSS, RTC evaluation
│   ├── utils/                  # Motion processing, rotation, quaternion
│   ├── options/                # CLI argument parsers
│   ├── scripts/
│   │   ├── train.py            # Training entry point
│   │   ├── generate_robot.py   # Text-to-motion generation
│   │   ├── server_robot_ws.py  # WebSocket inference server
│   │   ├── evaluation.py       # Evaluation pipeline
│   │   └── compute_38d_stats.py
│   ├── tools/MoCLIP/           # MoCLIP evaluator training
│   ├── config/                 # Diffusion scheduler & evaluator YAML
│   ├── docs/                   # WebSocket API docs
│   ├── data/                   # Mean_38d.npy, Std_38d.npy
│   └── checkpoints/            # Pretrained weights (downloaded)
├── deploy/                     # Edge deployment (Sim2Real)
│   ├── src/
│   │   ├── deploy.py           # Main controller (real + sim)
│   │   ├── sim2sim.py          # MuJoCo simulator bridge
│   │   ├── text_to_motion.py   # Cloud generator WS client
│   │   ├── policy.py           # ONNX runtime inference
│   │   ├── observation.py      # Observation construction
│   │   └── common/             # Joint mapper, PD helpers, math
│   ├── config/                 # tracking.yaml, controller.yaml
│   └── assets/ckpts/           # ONNX policy checkpoint
└── scripts/                    # download_weights.sh, serve.sh
```

## Citation

```bibtex
@misc{jia2026echoedgecloudhumanoidorchestration,
      title={ECHO: Edge-Cloud Humanoid Orchestration for Language-to-Motion Control},
      author={Haozhe Jia and Jianfei Song and Yuan Zhang and Honglei Jin and Youcheng Fan and Wenshuo Chen and Wei Zhang and Yutao Yue},
      year={2026},
      eprint={2603.16188},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.16188},
}
```

## License

MIT — see [LICENSE](generator/LICENSE).
