"""Robot Motion Dataset for 30-joint motion data at 50 FPS."""
import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
import random
import codecs as cs
from tqdm.auto import tqdm
from utils.rotation_utils import quat_wxyz_to_6d
from utils.robot_process import process_robot_npz
from multiprocessing import Pool, cpu_count


def _load_single_sample(args):
    """Helper function to load a single sample (for multiprocessing)."""
    name, npz_dir, text_dir, min_motion_len, dataset_name, max_motion_len = args
    try:
        # Load NPZ
        npz_data = np.load(pjoin(npz_dir, name + '.npz'), allow_pickle=True)
        
        # Extract 38-dim features (auto-detects full vs simple format)
        motion = process_robot_npz(npz_data, root_idx=0)
        
        # Filter by minimum length; keep long sequences and let __getitem__ crop.
        # Optional max_motion_len is kept for compatibility if caller needs it.
        if len(motion) < min_motion_len:
            return None
        if max_motion_len is not None and len(motion) > max_motion_len * 3:
            return None
        
        # Load text - format depends on dataset
        text_data = []
        with cs.open(pjoin(text_dir, name + '.txt')) as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                    
                text_dict = {}
                
                if dataset_name in ('robot', 'robotv2', 'robotv2_hard'):
                    # robot_humanml_data format: caption#tokens#0.0#0.0
                    parts = line.split('#')
                    if len(parts) < 4:
                        continue
                    caption = parts[0]
                    tokens = parts[1].split(' ')
                elif dataset_name == 'kungfu':
                    # kungfu format: plain sentence per line
                    caption = line
                    # Simple tokenization by space
                    tokens = caption.split(' ')
                else:
                    raise ValueError(f"Unknown dataset_name: {dataset_name}")
                
                text_dict['caption'] = caption
                text_dict['tokens'] = tokens
                text_data.append(text_dict)
        
        if len(text_data) == 0:
            return None
        
        return {
            'name': name,
            'motion': motion,
            'length': len(motion),
            'text': text_data
        }
    except Exception as e:
        return {'error': str(e), 'name': name}


class RobotMotionDataset(data.Dataset):
    """Robot motion dataset with 30 joints at 50 FPS.
    
    Motion representation: 38-dim (joint_pos[29] + root_vel_xy[2] + root_z[1] + root_rot_6d[6])
    Max length: 490 frames (~9.8 seconds at 50 FPS)
    
    Uses velocity-based representation:
    - joint_pos: 29D (joint angles)
    - root_vel_xy: 2D (root velocity in XY plane, per-frame displacement in aligned global frame)
    - root_z: 1D (root height)
    - root_rot_6d: 6D (root rotation in continuous 6D representation)
    """
    
    def __init__(self, opt, split, mode='train', accelerator=None):
        # Determine dataset name and data root
        self.dataset_name = getattr(opt, 'dataset_name', 'robot')
        
        if self.dataset_name == 'robot':
            self.data_root = './robot_humanml_data'
            text_subdir = 'texts'
        elif self.dataset_name == 'robotv2':
            self.data_root = './robot_humanml_data_v2'
            text_subdir = 'texts'
        elif self.dataset_name == 'robotv2_hard':
            self.data_root = './robot_humanml_data_v2_hard'
            text_subdir = 'texts'
        elif self.dataset_name == 'kungfu':
            self.data_root = './MotionMillion_kungfu'
            text_subdir = 'txt'
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        self.joints_num = 30
        self.dim_pose = 38  # 29 joint_pos + 2 root_vel_xy + 1 root_z + 6 root_rot_6d
        self.max_motion_length = getattr(opt, 'max_motion_length', 490)  # Get from opt (dataset-specific)
        self.min_motion_len = 100     # 2s at 50fps
        self.max_text_len = getattr(opt, 'max_text_len', 20)
        self.unit_length = getattr(opt, 'unit_length', 4)
        self.mode = mode
        
        npz_dir = pjoin(self.data_root, 'npz')
        text_dir = pjoin(self.data_root, text_subdir)
        
        # Load normalization stats (use 38d for robot dataset)
        if mode == 'train':
            mean = np.load(pjoin(self.data_root, 'Mean_38d.npy'))
            std = np.load(pjoin(self.data_root, 'Std_38d.npy'))
        else:
            # For eval/test, try to use 38d stats from data_root, fallback to meta_dir
            mean_38d_path = pjoin(self.data_root, 'Mean_38d.npy')
            std_38d_path = pjoin(self.data_root, 'Std_38d.npy')
            if os.path.exists(mean_38d_path) and os.path.exists(std_38d_path):
                mean = np.load(mean_38d_path)
                std = np.load(std_38d_path)
            else:
                # Fallback to meta_dir (should be 38d if training was done correctly)
                mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
                std = np.load(pjoin(opt.meta_dir, 'std.npy'))
                # Verify dimension
                if mean.shape[0] != 38 or std.shape[0] != 38:
                    raise ValueError(f"Expected 38-dim stats, but got mean.shape={mean.shape}, std.shape={std.shape}")
        
        self.mean = mean
        self.std = std
        
        # Load split file
        split_file = pjoin(self.data_root, f'{split}.txt')
        id_list = []
        with open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        
        if accelerator:
            accelerator.print(f'\nLoading {mode} mode {self.dataset_name} {split} dataset ...')
        else:
            print(f'\nLoading {mode} mode {self.dataset_name} dataset ...')
        
        # Build data dictionary with multiprocessing for faster loading
        data_dict = {}
        name_list = []
        length_list = []
        
        # Prepare arguments for multiprocessing
        # Disable multiprocessing in eval mode to avoid deadlock issues
        use_multiprocessing = (mode == 'train' and accelerator is not None)
        
        if use_multiprocessing:
            num_workers = min(cpu_count(), 8)  # Use up to 8 workers
            load_args = [
                (name, npz_dir, text_dir, self.min_motion_len, self.dataset_name, self.max_motion_length)
                for name in id_list
            ]
            
            # Use multiprocessing for parallel loading
            with Pool(processes=num_workers) as pool:
                results = list(tqdm(
                    pool.imap(_load_single_sample, load_args),
                    total=len(id_list),
                    disable=not accelerator.is_local_main_process if accelerator else False
                ))
        else:
            # Sequential loading for eval mode (avoid multiprocessing deadlock)
            results = []
            for name in tqdm(id_list, desc=f'Loading {mode} mode {self.dataset_name} {split} dataset', disable=False):
                result = _load_single_sample((
                    name, npz_dir, text_dir, self.min_motion_len, self.dataset_name, self.max_motion_length
                ))
                results.append(result)
        
        # Process results
        for result in results:
            if result is None:
                continue
            if 'error' in result:
                if accelerator and accelerator.is_local_main_process:
                    print(f"Warning: Failed to load {result['name']}: {result['error']}")
                continue
            
            name = result['name']
            data_dict[name] = {
                'motion': result['motion'],
                'length': result['length'],
                'text': result['text']
            }
            name_list.append(name)
            length_list.append(result['length'])
        
        # Sort by length
        name_list, length_list = zip(*sorted(zip(name_list, length_list), key=lambda x: x[1]))
        
        # Save mean/std to meta_dir during training
        if mode == 'train' and accelerator is not None and accelerator.is_main_process:
            np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(opt.meta_dir, 'std.npy'), std)
        
        self.data_dict = data_dict
        self.name_list = name_list
        
        if accelerator:
            accelerator.print(f'Completed loading {self.dataset_name} dataset: {len(self.name_list)} samples')
        else:
            print(f'Completed loading {self.dataset_name} dataset: {len(self.name_list)} samples')
    
    def inv_transform(self, data):
        """Denormalize motion data."""
        return data * self.std + self.mean
    
    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, idx):
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption = text_data['caption']
        
        # Z-score normalize
        motion = (motion - self.mean) / self.std
        
        # Crop if too long
        if m_length >= self.max_motion_length:
            idx = random.randint(0, len(motion) - self.max_motion_length)
            motion = motion[idx: idx + self.max_motion_length]
            m_length = self.max_motion_length
        
        # Pad if too short
        if m_length < self.max_motion_length:
            motion = np.concatenate([
                motion,
                np.zeros((self.max_motion_length - m_length, motion.shape[1]))
            ], axis=0)
        
        assert len(motion) == self.max_motion_length
        
        return caption, motion, m_length

