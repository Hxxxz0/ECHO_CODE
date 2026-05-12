"""Generate robot motions and save as complete NPZ files."""
import sys
import os
import torch
import numpy as np
from os.path import join as pjoin

# Add project root to path to fix imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accelerate.utils import set_seed
from models.gaussian_diffusion import DiffusePipeline
from options.generate_options import GenerateOptions
from utils.model_load import load_model_weights
from models import build_models
from utils.robot_npz_utils import reshape_generated_motion_38d


if __name__ == '__main__':
    parser = GenerateOptions()
    opt = parser.parse()
    set_seed(opt.seed)
    device = torch.device(f'cuda:{opt.gpu_id}' if torch.cuda.is_available() else 'cpu')
    opt.device = device
    
    print(f"Device: {device}")
    print(f"Dataset: {opt.dataset_name}")
    print(f"Dim pose: {opt.dim_pose}")
    print(f"Joints num: {opt.joints_num}")
    print(f"FPS: {opt.fps}")
    
    # Get text prompts
    if opt.text_prompt != '':
        texts = [opt.text_prompt]
        motion_lens = [int(opt.motion_length * opt.fps)]
    elif opt.input_text != '':
        with open(opt.input_text, 'r') as f:
            texts = [line.strip() for line in f.readlines() if line.strip()]
        if opt.input_lens != '':
            with open(opt.input_lens, 'r') as f:
                motion_lens = [int(line.strip()) for line in f.readlines()]
        else:
            motion_lens = [int(opt.motion_length * opt.fps)] * len(texts)
    else:
        texts = ["a person walks forward", "a person kicks with left leg"]
        motion_lens = [245, 245]  # 4.9 seconds at 50fps
    
    print(f"\nGenerating {len(texts)} motions...")
    for i, (text, mlen) in enumerate(zip(texts, motion_lens)):
        print(f"  {i+1}. '{text}' - {mlen} frames ({mlen/opt.fps:.2f}s)")
    
    # Load model
    print(f"\nLoading model from {opt.model_dir}...")
    model = build_models(opt)
    ckpt_path = pjoin(opt.model_dir, opt.which_ckpt + '.tar')
    niter = load_model_weights(model, ckpt_path, use_ema=not opt.no_ema)
    print(f"Loaded checkpoint at iteration {niter}")
    
    # Create pipeline (always use float32 for robot dataset to avoid CLIP dtype mismatch)
    pipeline = DiffusePipeline(
        opt=opt, 
        model=model, 
        diffuser_name=opt.diffuser_name,
        device=device, 
        num_inference_steps=opt.num_inference_steps,
        torch_dtype=torch.float32
    )
    
    # Generate motions
    print(f"\nGenerating motions with {opt.diffuser_name} sampler...")
    pred_motions = pipeline.generate(
        texts, 
        torch.LongTensor(motion_lens)
    )
    
    # Create output directory
    if opt.output_dir:
        out_path = opt.output_dir
    else:
        out_path = pjoin(opt.save_root, f'samples_iter{niter}_seed{opt.seed}')
    os.makedirs(out_path, exist_ok=True)
    
    # Load 38D normalization stats
    mean = np.load(pjoin(opt.data_root, 'Mean_38d.npy'))
    std = np.load(pjoin(opt.data_root, 'Std_38d.npy'))
    
    print(f"\nDenormalizing and saving results...")
    print(f"Output directory: {out_path}")
    
    # Save as complete NPZ files
    npz_dir = pjoin(out_path, 'npz')
    os.makedirs(npz_dir, exist_ok=True)
    
    # Also save as simple npy for backward compatibility
    npy_dir = pjoin(out_path, 'npy')
    os.makedirs(npy_dir, exist_ok=True)
    
    for i, motion in enumerate(pred_motions):
        # Denormalize
        motion_np = motion.cpu().numpy() * std + mean  # (frames, 38)

        # 38D format: velocity-based representation
        npz_data = reshape_generated_motion_38d(
            motion_np,
            fps=opt.fps,
            smooth=getattr(opt, 'enable_smooth', False),
            smooth_window=getattr(opt, 'smooth_window', 5),
            adaptive_smooth=getattr(opt, 'adaptive_smooth', False),
            static_start=getattr(opt, 'static_start', False),
            static_frames=getattr(opt, 'static_frames', 2),
            blend_frames=getattr(opt, 'blend_frames', 8)
        )
        npz_name = f'{i:06d}.npz'
        np.savez(pjoin(npz_dir, npz_name), **npz_data)
        npy_name = f'{i:06d}.npy'
        np.save(pjoin(npy_dir, npy_name), npz_data['root_pos'])
        print(f"  Saved sample {i}: {motion_lens[i]} frames ({motion_lens[i]/opt.fps:.2f}s)")
    
    # Save text descriptions
    with open(pjoin(out_path, 'prompts.txt'), 'w') as f:
        f.write('\n'.join(texts))
    
    # Save motion lengths
    with open(pjoin(out_path, 'lengths.txt'), 'w') as f:
        f.write('\n'.join([str(l) for l in motion_lens]))
    
    print(f"\n✓ Generation complete!")
    print(f"  NPZ files (complete): {npz_dir}/")
    print(f"  NPY files (positions): {npy_dir}/")
    print(f"  Prompts: {pjoin(out_path, 'prompts.txt')}")
    print(f"  Lengths: {pjoin(out_path, 'lengths.txt')}")

