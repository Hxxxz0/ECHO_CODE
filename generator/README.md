# SHELL: Semantic Hierarchical Embodied Language-to-Motion with Low-level Tracking

**Generator Training Code**

This repository contains the **generator component** of SHELL, an edge-cloud language-to-humanoid control framework. The generator runs on cloud servers to produce robot-native motion references, which are then streamed to an on-board tracker for real-time execution on the **Unitree G1 humanoid robot**.

Built upon [StableMoFusion](https://h-y1heng.github.io/StableMoFusion-page/), this diffusion-based text-to-motion generator creates semantically consistent whole-body motion sequences in a compact 38D velocity-based representation, designed to be **retargeting-free** and **tracker-friendly**.

## Key Features

- **Robot-native motion generation**: Direct generation in robot DOF space, avoiding human-to-robot retargeting
- **38D velocity-based representation**: 29 joint angles + 2D root velocity + 1D root height + 6D rotation
- **Diffusion-based architecture**: DDPM with CLIP text encoder for open-vocabulary instruction following
- **Two model options**: UNet (CondUNet1D) and Transformer backbones
- **Streaming-ready output**: Generates motion chunks up to 490 frames (~9.8s at 50 FPS) for low-latency deployment
- **MoCLIP evaluation**: Motion-CLIP alignment metrics (FID, R-precision, diversity, multimodality)
- **WebSocket server**: Persistent GPU inference server for production edge-cloud deployment
- **Classifier-free guidance**: Improved semantic consistency and motion quality

## Get Started

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- CUDA-compatible GPU (tested on NVIDIA A100)

### Installation

```bash
# Create conda environment
conda create -n stablerofusion python=3.8 -y
conda activate stablerofusion

# Install PyTorch (adjust for your CUDA version)
# See https://pytorch.org/ for instructions
pip install torch torchvision torchaudio

# Install dependencies
pip install -r requirements.txt

# (Optional) For WebSocket deployment server
pip install -r requirements_server.txt
```

## Motion Representation

SHELL uses a compact **38D velocity-based** robot motion representation at **50 FPS**, designed for tracker-friendly execution:

| Dimensions | Content | Description |
|-----------|---------|-------------|
| 0-28 | `joint_pos` (29) | Joint angles for 29 DOFs (robot-native) |
| 29-30 | `root_vel_xy` (2) | Root planar velocity in aligned global frame |
| 31 | `root_z` (1) | Root height above ground |
| 32-37 | `root_rot_6d` (6) | Root rotation in 6D continuous representation |

**Design rationale:**
- **Velocity-based root motion**: Avoids global position drift, improves streamability for real-time tracking
- **No global positions**: Relative motion is more robust for closed-loop execution under disturbances
- **6D rotation**: Continuous representation prevents gimbal lock and discontinuities
- **Robot-native DOF**: No human body model (SMPL/SMPL-X), eliminating retargeting overhead

### Data Directory Structure

```
robot_humanml_data/
  ├── npz/                  # Raw motion NPZ files
  ├── texts/                # Text descriptions (one .txt per motion)
  ├── train.txt             # Training split file IDs
  ├── test.txt              # Test split file IDs
  ├── Mean_38d.npy          # Per-dimension mean for normalization
  └── Std_38d.npy           # Per-dimension std for normalization
```

To compute normalization statistics for a new dataset:
```bash
python -m scripts.compute_38d_stats
```

## Training

### Train from Scratch

```bash
# UNet model (default)
bash scripts/train_robot.sh

# Or manually:
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file 1gpu.yaml \
  -m scripts.train \
  --name robot_38d_unet \
  --dataset_name robot \
  --batch_size 64 \
  --num_train_steps 50000 \
  --model-ema \
  --model_type unet \
  --base_dim 512 \
  --dim_mults 2 2 2 2 \
  --lr 2e-4
```

For Transformer model, see the commented section in `scripts/train_robot.sh`.

### Resume Training

```bash
bash scripts/train_robot_resume.sh
```

Checkpoints are saved to `./checkpoints/robot/<name>/model/`.

## Evaluation

### 1. Train MoCLIP Evaluator

The evaluation system uses MoCLIP (Motion-CLIP alignment model) to compute FID, R-precision, diversity, and multimodality metrics.

```bash
bash scripts/train_robot_evaluator.sh
```

This trains the MoCLIP model and saves checkpoints to `./checkpoints/robot_evaluator/`.

### 2. Run Evaluation

```bash
bash test_eval.sh

# Or manually:
python -m scripts.evaluation \
  --opt_path ./checkpoints/robot/robot_38d_unet/opt.txt \
  --evaluator_dir ./checkpoints/robot_evaluator \
  --which_ckpt latest \
  --num_inference_steps 10 \
  --gpu_id 0
```

Evaluation results are saved to `./checkpoints/robot/<name>/eval/`.

## Motion Generation

Generate robot-native motion references from text instructions:

```bash
# Generate from a text prompt
python -m scripts.generate_robot \
  --opt_path ./checkpoints/robot/robot_38d_unet/opt.txt \
  --which_ckpt latest \
  --text_prompt "a person walks forward" \
  --motion_length 4.0

# Generate from a file of prompts
python -m scripts.generate_robot \
  --opt_path ./checkpoints/robot/robot_38d_unet/opt.txt \
  --which_ckpt latest \
  --input_text prompts.txt \
  --motion_length 4.0
```

**Output format**: NPZ files contain `joint_pos` (29D), `root_pos` (3D integrated from velocity), and `root_rot` (quaternion wxyz). The 38D velocity-based representation is used internally during generation and streaming.

## Edge-Cloud Deployment

SHELL follows an **edge-cloud architecture** where the generator runs on a cloud server (GPU-accelerated) and streams motion references to the on-board tracker:

```bash
# Deploy generator server on cloud (with GPU)
python scripts/server_robot_ws.py \
  --opt_path ./checkpoints/robot/robot_38d_unet/opt.txt \
  --which_ckpt latest \
  --port 8000
```

**Architecture benefits:**
- **Low on-board compute**: Robot only runs lightweight tracking controller
- **High-rate feedback**: Tracker maintains 50+ Hz closed-loop control
- **Scalable semantics**: Complex language models run off-board without latency constraints
- **Modular design**: Generator and tracker can be developed/deployed independently

See [WebSocket Quickstart](docs/WEBSOCKET_QUICKSTART.md) and [Client API Documentation](docs/CLIENT_API.md) for integration with the tracker component.

## Project Structure

```
StableRofusion/
├── scripts/
│   ├── train.py                    # Training entry point
│   ├── train_robot.sh              # Training launch script
│   ├── train_robot_resume.sh       # Resume training
│   ├── evaluation.py               # Evaluation with MoCLIP
│   ├── generate_robot.py           # Motion generation
│   ├── server_robot_ws.py          # WebSocket inference server
│   ├── compute_38d_stats.py        # Compute normalization stats
│   └── train_robot_evaluator.sh    # Train MoCLIP evaluator
├── datasets/
│   └── robot_dataset.py            # Robot 38D dataset loader
├── models/
│   ├── unet.py                     # CondUNet1D architecture
│   ├── transformer.py              # Transformer architecture
│   ├── gaussian_diffusion.py       # Diffusion pipeline (eval)
│   └── gaussian_diffusion_inference.py  # Diffusion pipeline (inference/generation)
├── eval/
│   ├── evaluator_wrapper.py        # MoCLIP evaluator wrapper
│   └── eval_robot_moclip.py        # Robot evaluation metrics
├── trainers/
│   └── ddpm_trainer.py             # DDPM training loop
├── utils/
│   ├── robot_process.py            # Robot NPZ data processing
│   ├── robot_npz_utils.py          # Generation output post-processing
│   ├── rotation_utils.py           # 6D rotation utilities
│   ├── quaternion.py               # Quaternion operations
│   ├── model_load.py               # Checkpoint loading
│   ├── metrics.py                  # FID, diversity, R-precision
│   ├── ema.py                      # Exponential moving average
│   └── ...
├── tools/MoCLIP/                   # MoCLIP evaluator training
├── config/                         # Diffusion & evaluator configs
├── docs/                           # WebSocket API documentation
├── test_eval.sh                    # Evaluation test script
├── requirements.txt                # Python dependencies
└── requirements_server.txt         # Server dependencies
```

## SHELL System Overview

SHELL separates language-to-humanoid control into two concerns:

1. **Generator (this repository)**: Cloud-based diffusion model that produces semantically consistent, robot-native motion references from language instructions
2. **Tracker (separate component)**: On-board controller that executes motion references in real-time with physical feasibility and disturbance robustness

This separation addresses key deployment challenges:
- **Semantic expressivity vs. real-time control**: Generator handles complex language understanding off-board; tracker focuses on low-level stability
- **Retargeting overhead**: Direct robot-space generation eliminates human body model intermediate representations
- **Modularity**: Generator and tracker interfaces are decoupled, enabling independent development and platform portability

## Acknowledgments

This project is built upon [StableMoFusion](https://github.com/h-y1heng/StableMoFusion), with additional contributions from:

[text-to-motion](https://github.com/EricGuo5513/text-to-motion), [MDM](https://github.com/GuyTevet/motion-diffusion-model), [MotionDiffuse](https://github.com/mingyuan-zhang/MotionDiffuse), [MLD](https://github.com/ChenFengYe/motion-latent-diffusion), [OpenAI CLIP](https://github.com/openai/CLIP), and [Hugging Face Diffusers](https://github.com/huggingface/diffusers).

## License

This code is distributed under an [MIT LICENSE](LICENSE).

Note that this project depends on other libraries, including CLIP, Diffusers, and PyTorch, each with their own respective licenses.
