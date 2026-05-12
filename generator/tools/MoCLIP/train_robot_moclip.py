"""
Robot MoCLIP Training Script
Train a contrastive model to align robot motion and text descriptions
"""
import os
import sys
import argparse
import yaml
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Add current directory to path
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))
DEFAULT_CONFIG = current_dir / 'config_robot.yaml'

# Import local modules
from robot_moclip_model import ClipMotionAlignModel, clip_contrastive_loss, compute_retrieval_metrics
from datasets.robot_moclip_dataset import RobotMoCLIPDataset

# CLIP imports
try:
    import clip
    OPENAI_CLIP_AVAILABLE = True
except ImportError:
    OPENAI_CLIP_AVAILABLE = False

try:
    from transformers import CLIPTokenizer
    HF_CLIP_AVAILABLE = True
except ImportError:
    HF_CLIP_AVAILABLE = False


def cosine_scheduler(optimizer, num_epochs, warmup_epochs=5, min_lr=1e-5):
    """Create a cosine learning rate scheduler with warmup."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, dataloader, optimizer, scheduler, tokenizer, device, epoch, use_openai_clip=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (captions, motions, m_lengths) in enumerate(pbar):
        motions = motions.to(device)
        m_lengths = m_lengths.to(device)
        
        if use_openai_clip:
            # OpenAI CLIP: pass raw text directly
            motion_emb, text_emb = model(motions, m_lengths, raw_text=captions)
        else:
            # Hugging Face CLIP: tokenize text first
            text_inputs = tokenizer(
                captions,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors='pt'
            )
            input_ids = text_inputs['input_ids'].to(device)
            attention_mask = text_inputs['attention_mask'].to(device)
            motion_emb, text_emb = model(motions, m_lengths, input_ids=input_ids, attention_mask=attention_mask)
        
        loss = clip_contrastive_loss(motion_emb, text_emb, model.logit_scale)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def evaluate(model, dataloader, tokenizer, device, sample_size=32, use_openai_clip=False):
    """Evaluate model on validation set."""
    model.eval()
    all_motion_emb = []
    all_text_emb = []
    
    with torch.no_grad():
        for captions, motions, m_lengths in tqdm(dataloader, desc='Evaluating'):
            motions = motions.to(device)
            m_lengths = m_lengths.to(device)
            
            if use_openai_clip:
                # OpenAI CLIP: pass raw text directly
                motion_emb, text_emb = model(motions, m_lengths, raw_text=captions)
            else:
                # Hugging Face CLIP: tokenize text first
                text_inputs = tokenizer(
                    captions,
                    padding=True,
                    truncation=True,
                    max_length=77,
                    return_tensors='pt'
                )
                input_ids = text_inputs['input_ids'].to(device)
                attention_mask = text_inputs['attention_mask'].to(device)
                motion_emb, text_emb = model(motions, m_lengths, input_ids=input_ids, attention_mask=attention_mask)
            
            all_motion_emb.append(motion_emb.cpu())
            all_text_emb.append(text_emb.cpu())
    
    all_motion_emb = torch.cat(all_motion_emb, dim=0)
    all_text_emb = torch.cat(all_text_emb, dim=0)
    metrics = compute_retrieval_metrics(all_motion_emb, all_text_emb, sample_size=sample_size)
    return metrics


# compute_retrieval_metrics is imported from robot_moclip_model


def main():
    parser = argparse.ArgumentParser(description='Train Robot MoCLIP Model')
    parser.add_argument('--config', type=str, default=str(DEFAULT_CONFIG), help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 80)
    print("Robot MoCLIP Training")
    print("=" * 80)
    print(f"Configuration: {args.config}")
    print(f"Device: {config['device']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['lr']}")
    print(f"Num epochs: {config['num_epochs']}")
    
    # Determine which CLIP implementation to use
    use_openai_clip = config.get('use_openai_clip', False)
    
    if use_openai_clip:
        if not OPENAI_CLIP_AVAILABLE:
            raise ImportError("OpenAI CLIP is not available. Install with: pip install git+https://github.com/openai/CLIP.git")
        print(f"Using OpenAI CLIP: {config['clip_model_name']}")
        
        # Load OpenAI CLIP model and tokenizer
        device_for_clip = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, clip_preprocess = clip.load(config['clip_model_name'], device=device_for_clip)
        tokenizer = clip.tokenize  # OpenAI CLIP tokenizer function
        print(f"✓ Loaded OpenAI CLIP model: {config['clip_model_name']}")
    else:
        if not HF_CLIP_AVAILABLE:
            raise ImportError("Hugging Face transformers is not available. Install with: pip install transformers")
        print(f"Using Hugging Face CLIP: {config['clip_model_name']}")
        
        # Load Hugging Face CLIP tokenizer
        local_clip_path = os.environ.get('LOCAL_CLIP_PATH', None)
        if local_clip_path and os.path.exists(local_clip_path):
            print(f"Loading CLIP from local path: {local_clip_path}")
            tokenizer = CLIPTokenizer.from_pretrained(local_clip_path, local_files_only=True)
        else:
            print(f"Loading CLIP from Hugging Face: {config['clip_model_name']}")
            tokenizer = CLIPTokenizer.from_pretrained(config['clip_model_name'])
        
        print(f"✓ Loaded Hugging Face CLIP tokenizer")
    
    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = RobotMoCLIPDataset(
        data_root=config['data_root'],
        split='train',
        max_motion_length=config['max_seq_length'],
        min_motion_len=config.get('min_motion_len', 100)
    )
    
    val_dataset = RobotMoCLIPDataset(
        data_root=config['data_root'],
        split='test',  # Use test.txt as validation set
        max_motion_length=config['max_seq_length'],
        min_motion_len=config.get('min_motion_len', 100)
    )
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )
    
    # Create model
    print("\nCreating model...")
    model = ClipMotionAlignModel(config).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    # Create scheduler
    scheduler = cosine_scheduler(
        optimizer,
        config['num_epochs'],
        warmup_epochs=config['warmup_epochs'],
        min_lr=config['min_lr']
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_metric = 0.0
    
    if args.resume or config.get('resume_from'):
        checkpoint_path = args.resume or config['resume_from']
        if os.path.exists(checkpoint_path):
            print(f"\nResuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_metric = checkpoint.get('best_metric', 0.0)
            print(f"Resumed from epoch {start_epoch}, best metric: {best_metric:.2f}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(config['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCheckpoint directory: {checkpoint_dir}")
    
    # Training loop
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    
    for epoch in range(start_epoch, config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        print("-" * 80)
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, tokenizer, device, epoch + 1, use_openai_clip
        )
        print(f"Train Loss: {train_loss:.4f}")
        
        # Evaluate
        if (epoch + 1) % config['eval_interval'] == 0:
            print("\nEvaluating...")
            metrics = evaluate(
                model, val_loader, tokenizer, device, 
                sample_size=config['eval_sample_size'], 
                use_openai_clip=use_openai_clip
            )
            
            print(f"M2T R@1: {metrics['M2T_R1']:.2f}, R@2: {metrics['M2T_R2']:.2f}, R@3: {metrics['M2T_R3']:.2f}")
            print(f"T2M R@1: {metrics['T2M_R1']:.2f}, R@2: {metrics['T2M_R2']:.2f}, R@3: {metrics['T2M_R3']:.2f}")
            
            # Save best model
            avg_r1 = (metrics['M2T_R1'] + metrics['T2M_R1']) / 2
            if avg_r1 > best_metric:
                best_metric = avg_r1
                checkpoint_path = checkpoint_dir / 'best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': metrics,
                    'best_metric': best_metric,
                    'config': config
                }, checkpoint_path)
                print(f"✓ Saved best model (Avg R@1: {best_metric:.2f})")
        
        # Save checkpoint periodically
        if (epoch + 1) % config['save_interval'] == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metric': best_metric,
                'config': config
            }, checkpoint_path)
            print(f"✓ Saved checkpoint: {checkpoint_path.name}")
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best Avg R@1: {best_metric:.2f}")
    print("=" * 80)


if __name__ == '__main__':
    main()

