"""
Robot Motion Dataset for MoCLIP Training
Loads 38-dim robot motion features and text descriptions for contrastive learning

NOTE: This dataset uses lazy loading - data is loaded on-demand in __getitem__,
not during initialization. This saves memory for large datasets.

Uses the same preprocessing as robot_dataset.py to ensure consistency.
"""
import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
import random
import codecs as cs
from tqdm.auto import tqdm
import sys
from pathlib import Path

# Add project root to path to import utils
# This file is at: tools/MoCLIP/datasets/robot_moclip_dataset.py
# Project root is: tools/MoCLIP/../../ (two levels up from MoCLIP)
current_file = Path(__file__).resolve()
# Go up 4 levels: datasets -> MoCLIP -> tools -> project_root
project_root = current_file.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.robot_process import process_robot_npz


class RobotMoCLIPDataset(data.Dataset):
    """
    Robot motion dataset for MoCLIP training.
    
    Uses lazy loading - data is loaded on-demand, not during initialization.
    This allows training on large datasets without loading everything into memory.
    
    Motion representation: 38-dim (joint_pos[29] + root_vel_xy[2] + root_z[1] + root_rot_6d[6])
    Max length: 490 frames (~9.8 seconds at 50 FPS)
    
    Uses the same preprocessing pipeline as robot_dataset.py:
    - Floor alignment (Z normalization)
    - Root XY alignment to origin
    - Facing direction normalization (aligned to +X)
    - Velocity computation in aligned global frame
    
    Returns:
        caption: str - Text description
        motion: torch.Tensor (T, 38) - Normalized motion features
        m_length: int - Actual motion length
    """
    
    def __init__(self, data_root='./robot_humanml_data', split='train', 
                 max_motion_length=490, min_motion_len=100):
        """
        Args:
            data_root: Path to robot_humanml_data directory
            split: 'train', 'eval', or 'test'
            max_motion_length: Maximum motion sequence length
            min_motion_len: Minimum motion sequence length
        """
        self.data_root = data_root
        self.max_motion_length = max_motion_length
        self.min_motion_len = min_motion_len
        self.split = split
        
        self.npz_dir = pjoin(data_root, 'npz')
        self.text_dir = pjoin(data_root, 'texts')
        
        # Load normalization stats
        mean_path = pjoin(data_root, 'Mean_38d.npy')
        std_path = pjoin(data_root, 'Std_38d.npy')
        
        if not os.path.exists(mean_path) or not os.path.exists(std_path):
            raise FileNotFoundError(f"Normalization stats not found at {data_root}. "
                                    f"Please ensure Mean_38d.npy and Std_38d.npy exist.")
        
        self.mean = np.load(mean_path)
        self.std = np.load(std_path)
        
        # Verify dimensions
        if self.mean.shape[0] != 38 or self.std.shape[0] != 38:
            raise ValueError(f"Expected 38-dim stats, got mean.shape={self.mean.shape}, "
                           f"std.shape={self.std.shape}")
        
        # Load split file - only store file names, not data
        split_file = pjoin(data_root, f'{split}.txt')
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        print(f'\nInitializing MoCLIP Robot {split} dataset from {data_root} ...')
        print(f'  Using lazy loading - data will be loaded on-demand')
        
        # Only load file names, not actual data
        self.name_list = []
        with open(split_file, 'r') as f:
            for line in f.readlines():
                name = line.strip()
                # Pre-validate file existence
                npz_path = pjoin(self.npz_dir, name + '.npz')
                text_path = pjoin(self.text_dir, name + '.txt')
                if os.path.exists(npz_path) and os.path.exists(text_path):
                    self.name_list.append(name)
        
        print(f'\nCompleted initialization:')
        print(f'  - Valid samples: {len(self.name_list)}')
        print(f'  - Motion dim: 38 (joint_pos[29] + root_vel_xy[2] + root_z[1] + root_rot_6d[6])')
        print(f'  - Max length: {max_motion_length} frames')
        print(f'  - Data loading: Lazy (on-demand)')
    
    def __len__(self):
        return len(self.name_list)
    
    
    def _load_sample(self, name):
        """
        Load a single sample from disk (lazy loading).
        
        Args:
            name: Sample name (without extension)
        
        Returns:
            motion: np.ndarray (T, 38) - Motion features
            m_length: int - Motion length
            text_list: List[str] - Text descriptions
        """
        # Load NPZ
        npz_path = pjoin(self.npz_dir, name + '.npz')
        npz_data = np.load(npz_path, allow_pickle=True)
        
        # Extract 38-dim features using the same preprocessing as robot_dataset.py
        # This ensures consistency: floor alignment, root XY alignment, facing normalization
        motion = process_robot_npz(npz_data, root_idx=0)
        m_length = len(motion)
        
        # Filter by length
        if m_length < self.min_motion_len or m_length >= 500:
            return None, None, None
        
        # Load text descriptions
        text_list = []
        text_path = pjoin(self.text_dir, name + '.txt')
        with cs.open(text_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split('#')
                if len(parts) < 2:
                    continue
                caption = parts[0].strip()
                if caption:
                    text_list.append(caption)
        
        if len(text_list) == 0:
            return None, None, None
        
        return motion, m_length, text_list
    
    def __getitem__(self, idx):
        """
        Load and return a sample (lazy loading).
        
        Returns:
            caption: str - Text description
            motion: np.ndarray (max_motion_length, 38) - Normalized, padded motion
            m_length: int - Actual motion length (before padding)
        """
        name = self.name_list[idx]
        
        # Load data on-demand
        motion, m_length, text_list = self._load_sample(name)
        
        # If loading failed, try next sample (with wrap-around)
        if motion is None:
            # Fallback: try next sample
            next_idx = (idx + 1) % len(self.name_list)
            name = self.name_list[next_idx]
            motion, m_length, text_list = self._load_sample(name)
            if motion is None:
                # Last resort: return zeros
                motion = np.zeros((self.min_motion_len, 38), dtype=np.float32)
                m_length = self.min_motion_len
                text_list = [""]
        
        # Randomly select a caption
        caption = random.choice(text_list)
        
        # Z-score normalization
        motion = (motion - self.mean) / self.std
        
        # Crop if too long
        if m_length >= self.max_motion_length:
            start_idx = random.randint(0, m_length - self.max_motion_length)
            motion = motion[start_idx:start_idx + self.max_motion_length]
            m_length = self.max_motion_length
        
        # Pad if too short
        if m_length < self.max_motion_length:
            motion = np.concatenate([
                motion,
                np.zeros((self.max_motion_length - m_length, 38), dtype=np.float32)
            ], axis=0)
        
        return caption, motion.astype(np.float32), m_length


def collate_fn(batch):
    """
    Collate function for DataLoader.
    
    Args:
        batch: List of (caption, motion, m_length)
    
    Returns:
        captions: List[str] - Batch of captions
        motions: torch.Tensor (B, T, 38) - Batch of motions
        m_lengths: torch.Tensor (B,) - Batch of lengths
    """
    captions = [item[0] for item in batch]
    motions = torch.stack([torch.from_numpy(item[1]) for item in batch])
    m_lengths = torch.LongTensor([item[2] for item in batch])
    
    return captions, motions, m_lengths

