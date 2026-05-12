"""Evaluation script for robot motion generation using MoCLIP metrics."""
import sys
import os
# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from motion_loader import get_robot_eval_loader, get_robot_motion_loader
from motion_loader.robot_eval_loader import RobotEvalDataset
from models import build_models
from eval import MoCLIPEvaluatorWrapper
from eval.eval_robot_moclip import evaluation_moclip
from utils.model_load import load_model_weights
from os.path import join as pjoin

from models.gaussian_diffusion import DiffusePipeline
from accelerate.utils import set_seed

from options.evaluate_options import TestOptions


if __name__ == '__main__':
    parser = TestOptions()
    opt = parser.parse()
    set_seed(0)

    device_id = opt.gpu_id
    device = torch.device('cuda:%d' % device_id if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device_id)
    opt.device = device

    # Load MoCLIP evaluator for robot dataset
    print("\n" + "="*80)
    print("Using MoCLIP Evaluator for Robot Dataset")
    print("="*80)
    eval_wrapper = MoCLIPEvaluatorWrapper(opt)

    # Load dataset
    gt_loader = get_robot_eval_loader(opt, opt.batch_size, split='test', mode='gt_eval')
    # Create RobotEvalDataset for MoCLIP (returns caption strings)
    gen_dataset = RobotEvalDataset(opt, split='test', mode='eval')

    # Load model
    model = build_models(opt)
    ckpt_path = pjoin(opt.model_dir, opt.which_ckpt + '.tar')  
    load_model_weights(model, ckpt_path, use_ema=not opt.no_ema, device=device)

    # Create a pipeline for generation in diffusion model framework
    # Use float32 to avoid dtype mismatch with CLIP
    pipeline = DiffusePipeline(
        opt = opt,
        model = model, 
        diffuser_name = opt.diffuser_name, 
        device=device,
        num_inference_steps=opt.num_inference_steps,
        torch_dtype=torch.float32)

    # Setup motion loaders for robot evaluation
    eval_motion_loaders = {
        'robot': lambda: get_robot_motion_loader(
            opt,
            opt.batch_size,
            pipeline,
            gen_dataset,
            opt.mm_num_samples,
            opt.mm_num_repeats,
        )
    }

    save_dir = pjoin(opt.save_root, 'eval') 
    os.makedirs(save_dir, exist_ok=True)
    if opt.no_ema:
        log_file = pjoin(save_dir, opt.diffuser_name) + f'_{str(opt.num_inference_steps)}steps.log'
    else:
        log_file = pjoin(save_dir, opt.diffuser_name) + f'_{str(opt.num_inference_steps)}steps_ema.log'
    
    if not os.path.exists(log_file):
        config_dict = dict(pipeline.scheduler.config)
        config_dict['no_ema'] = opt.no_ema
        with open(log_file, 'wt') as f:
            f.write('------------ Options -------------\n')
            for k, v in sorted(config_dict.items()):
                f.write('%s: %s\n' % (str(k), str(v)))
            f.write('-------------- End ----------------\n')

    # Run MoCLIP evaluation
    all_metrics = evaluation_moclip(
        eval_wrapper, gt_loader, eval_motion_loaders, log_file, 
        opt.replication_times, opt.diversity_times, opt.mm_num_times, 
        dataset=gen_dataset, run_mm=True
    )
