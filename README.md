# ECHO: Edge-Cloud Humanoid Orchestration for Language-to-Motion Control

A cloud-hosted diffusion-based text-to-motion generator synthesizes motion references from natural language, while an edge-deployed RL tracker executes them in closed loop on the Unitree G1 humanoid robot.

```
┌──────────────────────────────────────────────────────┐
│  Cloud GPU                                           │
│  ┌────────────┐    ┌──────────────────────────┐      │
│  │ Text input │───▶│ Diffusion Generator (UNet)│      │
│  │ "walk fwd" │    │ DDIM 10 steps, CLIP cond │      │
│  └────────────┘    └──────────┬───────────────┘      │
│                               │ 38D motion            │
│                    WebSocket  │ (joint_pos, root_vel, │
│                    Server     │  root_z, root_rot_6d) │
└───────────────────────────────┼──────────────────────┘
                                │
     SSH Tunnel / Network       │
                                ▼
┌──────────────────────────────────────────────────────┐
│  Edge (Unitree G1)                                   │
│  ┌──────────────────────┐   ┌─────────────────┐     │
│  │ ONNX Tracking Policy │◀──│ 38D → NPZ        │     │
│  │ (PPO Teacher-Student)│   │ format converter │     │
│  └──────────┬───────────┘   └─────────────────┘     │
│             │ 29 joint targets                       │
│             ▼                                        │
│  ┌──────────────────────┐                            │
│  │ PD Controller (500Hz)│                            │
│  │ + Fall Recovery       │                            │
│  └──────────────────────┘                            │
└──────────────────────────────────────────────────────┘
```

## 38D Motion Representation

Each frame is a 38-dimensional vector:

| Component | Dims | Description |
|-----------|------|-------------|
| `joint_pos` | 29 | Joint angles (radians) in Isaac Gym order |
| `root_vel_xy` | 2 | Planar root velocity in body frame (m/frame) |
| `root_z` | 1 | Root height (m) |
| `root_rot_6d` | 6 | Continuous 6D root orientation |

At 50 FPS, this gives ~9.8s max motion (490 frames). No human body model retargeting needed.

## Installation

```bash
# Core dependencies
pip install -r requirements.txt

# WebSocket server (optional)
pip install -r requirements_server.txt
```

Core dependencies: `torch`, `diffusers`, `openai-clip`, `accelerate`, `einops`, `scipy`, `numpy`, `pyyaml`, `tensorboard`.

## Model Weights

Download pretrained checkpoints from ModelScope:

```bash
git clone https://www.modelscope.cn/Hzzzz001/ECHO.git checkpoints/
```

Or via ModelScope SDK:
```python
from modelscope import snapshot_download
model_dir = snapshot_download('Hzzzz001/ECHO')
```

Three checkpoints are provided:

| Checkpoint | Model | Description |
|-----------|-------|-------------|
| `robotv2_38d_lite` | UNet (small) | Fast inference, ~0.2s |
| `robotv2_38d` | UNet (full) | Best quality, ~1.0s |
| `robotv2_38d_transformer` | Transformer | Highest context, ~3.0s |

Each contains: `opt.txt` (training config), `model/latest.tar` (weights), `meta/mean.npy` + `meta/std.npy` (normalization).

## Usage

### Generate Motions from Text

```bash
cd generator
python scripts/generate.py \
    --opt_path checkpoints/robotv2/robotv2_38d_lite/opt.txt \
    --text_prompt "a person walks forward" \
    --motion_length 4.0 \
    --output_dir ./output
```

Output:
- `output/npz/000000.npz` — 38D reconstruction (joint_pos, root_pos, root_rot)
- `output/npy/000000.npy` — root position only

### Start Cloud WebSocket Server

```bash
cd generator
python scripts/server.py \
    --opt_path checkpoints/robotv2/robotv2_38d_lite/opt.txt \
    --port 8000 --host 127.0.0.1
```

Access remotely via SSH tunnel: `ssh -L 8000:127.0.0.1:8000 user@cloud-server`

WebSocket API: connect to `ws://localhost:8000/ws`, send JSON:
```json
{"text": "walk forward slowly", "motion_length": 4.0, "seed": 42, "num_inference_steps": 10}
```
Receive: binary NPZ data.

### Evaluate Model

```bash
cd generator
python scripts/evaluation.py \
    --opt_path checkpoints/robotv2/robotv2_38d/opt.txt \
    --evaluator_dir checkpoints/robot_evaluator
```

Metrics: FID, R-Precision, Matching Score, Diversity, Multimodality, MSS, RTC.

### Train Generator

```bash
# Requires preprocessed robot motion data in robot_humanml_data_v2/
cd generator
accelerate launch --config_file 1gpu.yaml scripts/train.py \
    --dataset_name robotv2 \
    --name robotv2_38d_lite \
    --batch_size 64 \
    --num_train_steps 500000
```

### Deploy to G1 Robot

See [deploy/README.md](deploy/README.md) for Sim2Real deployment instructions.

## Project Structure

```
ECHO_CODE/
├── generator/              # Cloud text-to-motion generator
│   ├── models/             # UNet, Transformer architectures
│   ├── datasets/           # Robot motion dataset (38D)
│   ├── trainers/           # DDPM training loop
│   ├── eval/               # MoCLIP, MSS, RTC metrics
│   ├── utils/              # Motion processing, rotation, EMA
│   ├── options/            # CLI argument parsers
│   ├── scripts/            # train, generate, server, evaluate
│   ├── config/             # Scheduler params, evaluator config
│   ├── tools/MoCLIP/       # MoCLIP evaluator training
│   ├── checkpoints/        # Model weights (download separately)
│   └── data/               # Normalization stats
├── deploy/                 # Edge deployment (Sim2Real)
│   ├── config/             # PD gains, tracking config
│   ├── src/                # Controller, policy, motion sources
│   └── assets/ckpts/       # ONNX policy checkpoints
└── scripts/                # Convenience scripts
```

## Citation

```bibtex
@article{echo2025,
  title={ECHO: Edge-Cloud Humanoid Orchestration for Language-to-Motion Control},
  author={Jensen},
  journal={arXiv preprint},
  year={2025}
}
```

## License

MIT License — see [LICENSE](generator/LICENSE) for details.
