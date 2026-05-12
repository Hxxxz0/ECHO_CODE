"""
Robot evaluation data loader for MoCLIP.
Simplified version that returns captions directly instead of GloVe embeddings.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from os.path import join as pjoin


class RobotEvalDataset(Dataset):
    """
    Robot motion dataset for evaluation with MoCLIP.
    
    Returns caption strings directly (not GloVe embeddings) for MoCLIP processing.
    """
    
    def __init__(self, opt, split='test', mode='gt_eval'):
        """
        Args:
            opt: Options object
            split: 'train', 'test', or 'eval'
            mode: 'gt_eval' for ground truth evaluation
        """
        from datasets.robot_dataset import RobotMotionDataset
        
        # Use existing RobotMotionDataset
        self.dataset = RobotMotionDataset(opt, split=split, mode=mode)
        self.opt = opt
        self.mode = mode
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Returns:
            caption: str - Text description
            motion: np.ndarray (T, 38) - Motion data
            m_length: int - Motion length
        """
        # Get data from underlying dataset
        caption, motion, m_length = self.dataset[idx]
        
        return caption, motion, m_length


class RobotGeneratedEvalDataset(Dataset):
    """
    Dataset for evaluating generated robot motions with MoCLIP.
    """
    
    def __init__(self, opt, pipeline, ground_truth_dataset, mm_num_samples, mm_num_repeats):
        """
        Args:
            opt: Options object
            pipeline: Generation pipeline
            ground_truth_dataset: Ground truth dataset (RobotEvalDataset)
            mm_num_samples: Number of samples for multimodality evaluation
            mm_num_repeats: Number of repeats for each sample
        """
        self.opt = opt
        self.dataset = ground_truth_dataset
        self.generated_motion = []
        self.mm_generated_motion = []
        
        # Prepare data loader for ground truth
        dataloader = DataLoader(
            ground_truth_dataset, 
            batch_size=1, 
            num_workers=1, 
            shuffle=True
        )
        
        min_mov_length = 100  # Robot: minimum 2 seconds at 50 FPS
        
        # Select samples for multimodality evaluation
        mm_idxs = []
        if mm_num_samples > 0:
            mm_idxs = np.random.choice(len(ground_truth_dataset), mm_num_samples, replace=False)
            mm_idxs = np.sort(mm_idxs)
        
        # Collect all captions and lengths for batch generation
        all_captions = []
        all_m_lens = []
        all_data_indices = []
        
        from tqdm import tqdm
        
        print("\nPreparing captions for generation...")
        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
                caption, motion, m_lens = data
                caption = caption[0] if isinstance(caption, (list, tuple)) else caption
                
                # Determine if this is a multimodality sample
                mm_num_now = len([x for x in all_data_indices if 'mm' in str(x)])
                is_mm = (mm_num_now < mm_num_samples) and (i in mm_idxs)
                repeat_times = mm_num_repeats if is_mm else 1
                
                # Adjust motion length
                m_lens = max(
                    torch.div(m_lens, opt.unit_length, rounding_mode='trunc') * opt.unit_length,
                    min_mov_length * opt.unit_length
                )
                m_lens = min(m_lens, opt.max_motion_length)
                
                if not isinstance(m_lens, torch.Tensor):
                    m_lens = torch.LongTensor([m_lens])
                
                # Add to generation lists
                for t in range(repeat_times):
                    all_captions.append(caption)
                    all_m_lens.append(m_lens.to(opt.device))
                    all_data_indices.append((i, is_mm, t))
        
        all_m_lens = torch.cat(all_m_lens)
        
        # Generate all motions in batch
        print(f"Generating {len(all_captions)} motions...")
        with torch.no_grad():
            all_pred_motions, t_eval = pipeline.generate(all_captions, all_m_lens)
        
        self.eval_generate_time = t_eval
        
        # Organize generated motions
        mm_motion_buffer = {}
        for idx, (data_idx, is_mm, repeat_idx) in enumerate(all_data_indices):
            caption = all_captions[idx]
            motion = all_pred_motions[idx].cpu().numpy()
            m_length = all_m_lens[idx].item()
            
            motion_data = {
                'motion': motion,
                'length': m_length,
                'caption': caption
            }
            
            if is_mm:
                # Store multimodality samples
                if data_idx not in mm_motion_buffer:
                    mm_motion_buffer[data_idx] = []
                mm_motion_buffer[data_idx].append(motion_data)
            else:
                # Regular samples
                self.generated_motion.append(motion_data)
        
        # Convert multimodality buffer to list
        self.mm_generated_motion = [
            {'mm_motions': mm_motion_buffer[idx]}
            for idx in sorted(mm_motion_buffer.keys())
        ]
        
        print(f"Generated {len(self.generated_motion)} regular samples and "
              f"{len(self.mm_generated_motion)} multimodality samples")
    
    def __len__(self):
        return len(self.generated_motion)
    
    def __getitem__(self, item):
        """
        Returns:
            caption: str - Text description
            motion: np.ndarray (T, 38) - Generated motion
            m_length: int - Motion length
        """
        data = self.generated_motion[item]
        motion = data['motion']
        m_length = data['length']
        caption = data['caption']
        
        # Denormalize and renormalize for evaluation
        # (This step maintains compatibility with evaluation metrics)
        denormed_motion = self.dataset.dataset.inv_transform(motion)
        renormed_motion = (denormed_motion - self.dataset.dataset.mean) / self.dataset.dataset.std
        motion = renormed_motion
        
        # Pad motion if needed
        if len(motion) < self.opt.max_motion_length:
            motion = np.concatenate([
                motion,
                np.zeros((self.opt.max_motion_length - len(motion), motion.shape[1]))
            ], axis=0)
        
        return caption, motion, m_length


class RobotMMGeneratedDataset(Dataset):
    """Dataset for multimodality evaluation."""
    
    def __init__(self, opt, motion_dataset):
        self.opt = opt
        self.dataset = motion_dataset.mm_generated_motion
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, item):
        data = self.dataset[item]
        mm_motions = data['mm_motions']
        
        m_lens = []
        motions = []
        
        for mm_motion in mm_motions:
            m_lens.append(mm_motion['length'])
            motion = mm_motion['motion']
            
            # Pad if needed
            if len(motion) < self.opt.max_motion_length:
                motion = np.concatenate([
                    motion,
                    np.zeros((self.opt.max_motion_length - len(motion), motion.shape[1]))
                ], axis=0)
            
            motion = motion[None, :]
            motions.append(motion)
        
        m_lens = np.array(m_lens, dtype=np.int32)
        motions = np.concatenate(motions, axis=0)
        
        # Sort by length (descending)
        sort_indx = np.argsort(m_lens)[::-1].copy()
        m_lens = m_lens[sort_indx]
        motions = motions[sort_indx]
        
        return motions, m_lens


def robot_collate_fn(batch):
    """
    Collate function for robot evaluation data.
    
    Args:
        batch: List of (caption, motion, m_length)
    
    Returns:
        captions: List[str] - Batch of captions
        motions: torch.Tensor (B, T, 38) - Batch of motions
        m_lengths: torch.Tensor (B,) - Batch of lengths
    """
    captions = [item[0] for item in batch]
    motions = torch.stack([torch.from_numpy(item[1]).float() for item in batch])
    m_lengths = torch.LongTensor([item[2] for item in batch])
    
    return captions, motions, m_lengths


def get_robot_eval_loader(opt, batch_size, split='test', mode='gt_eval'):
    """
    Get data loader for robot evaluation.
    
    Args:
        opt: Options object
        batch_size: Batch size
        split: 'train', 'test', or 'eval'
        mode: 'gt_eval' for ground truth evaluation
    
    Returns:
        DataLoader for robot evaluation
    """
    dataset = RobotEvalDataset(opt, split=split, mode=mode)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        collate_fn=robot_collate_fn
    )
    
    return dataloader


def get_robot_motion_loader(opt, batch_size, pipeline, ground_truth_dataset, 
                             mm_num_samples, mm_num_repeats):
    """
    Get motion loaders for generated robot motions.
    
    Args:
        opt: Options object
        batch_size: Batch size
        pipeline: Generation pipeline
        ground_truth_dataset: Ground truth dataset
        mm_num_samples: Number of multimodality samples
        mm_num_repeats: Number of repeats per sample
    
    Returns:
        motion_loader: DataLoader for regular evaluation
        mm_motion_loader: DataLoader for multimodality evaluation
        eval_generate_time: Generation time
    """
    dataset = RobotGeneratedEvalDataset(
        opt, pipeline, ground_truth_dataset, mm_num_samples, mm_num_repeats
    )
    
    mm_dataset = RobotMMGeneratedDataset(opt, dataset)
    
    motion_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=robot_collate_fn,
        drop_last=True,
        num_workers=4
    )
    
    mm_motion_loader = DataLoader(
        mm_dataset,
        batch_size=1,
        num_workers=1
    )
    
    return motion_loader, mm_motion_loader, dataset.eval_generate_time



