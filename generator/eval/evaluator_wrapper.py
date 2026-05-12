"""MoCLIP-based evaluator wrapper for robot motion evaluation."""
import torch
import os
from os.path import join as pjoin
import numpy as np
import torch.nn.functional as F


class MoCLIPEvaluatorWrapper(object):
    """
    MoCLIP-based evaluator for Robot dataset.
    Uses CLIP for text encoding and Transformer for motion encoding.
    """
    
    def __init__(self, opt):
        """
        Initialize MoCLIP evaluator.
        
        Args:
            opt: Options object with the following attributes:
                - evaluator_dir: Directory containing MoCLIP checkpoint
                - device: Device to use for inference
                - dataset_name: Dataset name (should be 'robot')
        """
        import sys
        
        # Add tools/MoCLIP to path
        moclip_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tools', 'MoCLIP'))
        if moclip_path not in sys.path:
            sys.path.insert(0, moclip_path)
        
        # Try OpenAI CLIP first (preferred for offline usage)
        try:
            import clip
            use_openai_clip = True
            print('Using OpenAI CLIP (offline-compatible)')
        except ImportError:
            use_openai_clip = False
            print('OpenAI CLIP not available, falling back to Hugging Face CLIP')
        
        self.opt = opt
        self.device = opt.device
        self.use_openai_clip = use_openai_clip
        
        # Load checkpoint to get configuration
        checkpoint_path = self._find_checkpoint(opt.evaluator_dir)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        config = checkpoint.get('config', {})
        
        # Get CLIP model name from checkpoint config or opt
        clip_model_name = config.get('clip_model_name', getattr(opt, 'clip_model_name', 'ViT-L/14'))
        self.max_length = getattr(opt, 'max_length', 77)
        
        if use_openai_clip:
            # Use OpenAI CLIP (no network required after first download)
            # Map HuggingFace names to OpenAI names
            clip_name_map = {
                "openai/clip-vit-large-patch14": "ViT-L/14",
                "openai/clip-vit-base-patch32": "ViT-B/32",
                "openai/clip-vit-base-patch16": "ViT-B/16"
            }
            openai_clip_name = clip_name_map.get(clip_model_name, clip_model_name)
            print(f'Loading OpenAI CLIP: {openai_clip_name}')
            
            self.tokenizer = clip.tokenize
            self.clip_model_name = openai_clip_name
        else:
            # Fall back to Hugging Face CLIP
            from transformers import CLIPTokenizer
            
            # Try to load CLIP tokenizer (with local path support)
            local_clip_path = os.environ.get('LOCAL_CLIP_PATH', None)
            if local_clip_path and os.path.exists(local_clip_path):
                print(f'Loading CLIP tokenizer from local path: {local_clip_path}')
                self.tokenizer = CLIPTokenizer.from_pretrained(local_clip_path, local_files_only=True)
            else:
                try:
                    # Try loading from cache first
                    self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name, local_files_only=True)
                    print(f'Loaded CLIP tokenizer from cache: {clip_model_name}')
                except Exception:
                    # Fall back to downloading (requires network)
                    print(f'Loading CLIP tokenizer from Hugging Face: {clip_model_name}')
                    self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
            
            self.clip_model_name = clip_model_name
        
        # Initialize MoCLIP model
        from robot_moclip_model import ClipMotionAlignModel
        
        # Create model with configuration from checkpoint
        model_config = {
            'clip_model_name': self.clip_model_name,
            'use_openai_clip': use_openai_clip,
            'input_dim': config.get('input_dim', 38),
            'embed_dim': config.get('embed_dim', 768),
            'num_heads': config.get('num_heads', 8),
            'num_layers': config.get('num_layers', 4),
            'max_seq_length': config.get('max_seq_length', 490),
            'freeze_clip': config.get('freeze_clip', True)
        }
        
        self.model = ClipMotionAlignModel(model_config)
        
        # Load model weights
        print(f'\nLoading MoCLIP model weights from {checkpoint_path} ...')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        epoch = checkpoint.get('epoch', 'unknown')
        best_r1 = checkpoint.get('best_r1', checkpoint.get('best_metric', 'unknown'))
        print(f'MoCLIP Model loaded (Epoch: {epoch}, Best R@1: {best_r1})')
        
        # Move to device and set to eval mode
        self.model.to(opt.device)
        self.model.eval()
    
    def _find_checkpoint(self, evaluator_dir):
        """Find the best or latest MoCLIP checkpoint."""
        # Support multiple checkpoint naming conventions
        checkpoint_names = [
            'best_model.pth',      # Standard naming from training script
            'moclip_best.pth',     # Alternative naming
            'moclip_latest.pth'    # Latest checkpoint
        ]
        
        for name in checkpoint_names:
            path = pjoin(evaluator_dir, name)
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError(
            f"No MoCLIP checkpoint found in {evaluator_dir}. "
            f"Tried: {', '.join(checkpoint_names)}. "
            f"Please train the MoCLIP model first using tools/MoCLIP/train_robot_moclip.py"
        )
    
    def get_co_embeddings(self, captions, motions, m_lens):
        """
        Get co-embeddings for captions and motions.
        
        Args:
            captions: List[str] or torch.Tensor - Text captions
            motions: torch.Tensor (B, T, 38) - Motion sequences
            m_lens: torch.Tensor (B,) - Actual motion lengths
        
        Returns:
            text_embedding: torch.Tensor (B, 768) - Text embeddings
            motion_embedding: torch.Tensor (B, 768) - Motion embeddings
        """
        with torch.no_grad():
            motions = motions.detach().to(self.device).float()
            m_lens = m_lens.to(self.device)
            
            # Handle captions input
            if isinstance(captions, (list, tuple)):
                caption_list = captions
            else:
                raise ValueError(
                    "MoCLIP requires text captions as input. "
                    "Please use get_robot_eval_loader which returns caption strings."
                )
            
            # Sort by length for efficient processing (descending order)
            align_idx = np.argsort(m_lens.cpu().numpy())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]
            
            # Also sort captions to match motion order
            sorted_captions = [caption_list[i] for i in align_idx]
            
            if self.use_openai_clip:
                # OpenAI CLIP: pass raw text directly to model
                motion_embedding, text_embedding = self.model(
                    motions, m_lens, raw_text=sorted_captions
                )
            else:
                # Hugging Face CLIP: tokenize text first
                text_enc = self.tokenizer(
                    sorted_captions,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                input_ids = text_enc["input_ids"].to(self.device)
                attention_mask = text_enc["attention_mask"].to(self.device)
                
                # Get embeddings from MoCLIP model
                motion_embedding, text_embedding = self.model(
                    motions, m_lens, input_ids, attention_mask
                )
            
            # Restore original order for both embeddings
            restore_idx = np.argsort(align_idx)
            text_embedding = text_embedding[restore_idx]
            motion_embedding = motion_embedding[restore_idx]
            
            # Normalize embeddings for evaluation (critical for distance-based metrics)
            text_embedding = F.normalize(text_embedding, dim=-1)
            motion_embedding = F.normalize(motion_embedding, dim=-1)
            
        return text_embedding, motion_embedding
    
    def get_motion_embeddings(self, motions, m_lens):
        """
        Get motion embeddings only.
        
        Args:
            motions: torch.Tensor (B, T, 38) - Motion sequences
            m_lens: torch.Tensor (B,) - Actual motion lengths
        
        Returns:
            motion_embedding: torch.Tensor (B, 768) - Motion embeddings
        """
        with torch.no_grad():
            motions = motions.detach().to(self.device).float()
            m_lens = m_lens.to(self.device)
            
            # Sort by length for efficient processing
            align_idx = np.argsort(m_lens.cpu().numpy())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]
            
            # Get motion embeddings
            motion_embedding = self.model.encode_motion(motions, m_lens)
            
            # Restore original order
            restore_idx = np.argsort(align_idx)
            motion_embedding = motion_embedding[restore_idx]
            
            # Normalize embeddings
            motion_embedding = F.normalize(motion_embedding, dim=-1)
            
        return motion_embedding
