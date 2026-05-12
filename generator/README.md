# ECHO Generator

Cloud-hosted diffusion-based text-to-motion generator for the ECHO framework. Synthesizes robot-native 38D motion references from natural language instructions, streaming to an edge-deployed RL tracker on the Unitree G1 humanoid.

## 38D Motion Representation

Compact velocity-based format at 50 FPS, directly compatible with low-level PD control — no retargeting needed.

| Dims | Field | Description |
|------|-------|-------------|
| 0-28 | `joint_pos` (29) | Joint angles for 29 DOFs in Isaac Gym order |
| 29-30 | `root_vel_xy` (2) | Root planar velocity in body frame |
| 31 | `root_z` (1) | Root height above ground |
| 32-37 | `root_rot_6d` (6) | Root rotation in continuous 6D representation |

## Installation

```bash
conda create -n echo python=3.10 -y
conda activate echo
conda install pytorch pytorch-cuda=12.8 -c pytorch -c nvidia -y
pip install -r requirements.txt
pip install -r requirements_server.txt  # for WebSocket server
```

## Model Weights

```bash
git clone https://www.modelscope.cn/Hzzzz001/ECHO.git checkpoints/
```

Three pretrained checkpoints:

| Checkpoint | Model | Size |
|-----------|-------|------|
| `robotv2/robotv2_38d_lite` | UNet (small) | ~1 GB |
| `robotv2/robotv2_38d` | UNet (full) | ~2 GB |
| `robotv2/robotv2_38d_transformer` | Transformer | ~2 GB |

## Usage

### Generate Motion from Text

```bash
python scripts/generate_robot.py \
  --opt_path checkpoints/robotv2/robotv2_38d_lite/opt.txt \
  --text_prompt "a person walks forward" \
  --motion_length 4.0 \
  --output_dir ./output
```

Output: `output/npz/000000.npz` with `joint_pos (T,29)`, `root_pos (T,3)`, `root_rot (T,4)`.

### Start Cloud WebSocket Server

```bash
python scripts/server_robot_ws.py \
  --opt_path checkpoints/robotv2/robotv2_38d_lite/opt.txt \
  --port 8000 --host 127.0.0.1
```

Remote access via SSH tunnel: `ssh -L 8000:127.0.0.1:8000 user@cloud-server`.

### Evaluate Model

```bash
python scripts/evaluation.py \
  --opt_path checkpoints/robotv2/robotv2_38d/opt.txt \
  --evaluator_dir checkpoints/robot_evaluator
```

Metrics: FID, R-Precision, Matching Score, Diversity, Multimodality, MSS, RTC.

### Train Generator

Requires preprocessed robot motion data in `robot_humanml_data_v2/`.

```bash
accelerate launch --config_file 1gpu.yaml scripts/train.py \
  --dataset_name robotv2 \
  --name robotv2_38d_lite \
  --batch_size 64 \
  --num_train_steps 500000
```

### Compute Normalization Stats

```bash
python -m scripts.compute_38d_stats --data_dir data/npz
```

## Project Structure

```
generator/
├── models/             # EchoUnet (1D Conv UNet), Transformer
├── datasets/           # 38D robot motion dataset
├── trainers/           # DDPM training loop with EMA
├── eval/               # MoCLIP, Motion Safety Score, Root Trajectory Consistency
├── utils/              # Motion processing, rotation, quaternion, metrics
├── options/            # CLI argument parsers
├── scripts/            # train, generate, server, evaluate, compute_stats
├── config/             # Diffusion scheduler + evaluator YAML
├── tools/MoCLIP/       # MoCLIP evaluator training
├── docs/               # WebSocket API documentation
├── data/               # Mean_38d.npy, Std_38d.npy
└── checkpoints/        # Pretrained model weights
```

## ECHO System Architecture

```
Cloud GPU                          Edge (Unitree G1)
┌──────────────────────┐           ┌──────────────────────┐
│ Text → CLIP → UNet   │  WebSocket│ ONNX Policy → PD     │
│ DDIM 10 steps, ~1s   │──────────▶│ 29 joint targets     │
│ 38D motion output    │  38D NPZ  │ + Fall Recovery      │
└──────────────────────┘           └──────────────────────┘
```

## Citation

```bibtex
@article{echo2025,
  title={ECHO: Edge-Cloud Humanoid Orchestration for Language-to-Motion Control},
  year={2025}
}
```

## License

MIT — see [LICENSE](LICENSE).
