"""
MoCLIP Model for Robot Motion Dataset
Contrastive learning between motion and text using CLIP
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
try:
    import clip  # OpenAI CLIP
    OPENAI_CLIP_AVAILABLE = True
except ImportError:
    OPENAI_CLIP_AVAILABLE = False

try:
    from transformers import CLIPModel, CLIPTokenizer  # Hugging Face CLIP
    HF_CLIP_AVAILABLE = True
except ImportError:
    HF_CLIP_AVAILABLE = False


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model, max_len=5000, dropout=0.2):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (T, B, D)
        Returns:
            (T, B, D)
        """
        seq_len = x.size(0)
        x = x + self.pe[:seq_len, :]
        return self.dropout(x)


class MotionEncoder(nn.Module):
    """
    Transformer-based motion encoder for robot motion.
    
    Encodes variable-length motion sequences into fixed-dimensional embeddings.
    """
    
    def __init__(self, input_dim=38, embed_dim=768, num_heads=8, num_layers=4,
                 dim_feedforward=2048, dropout=0.2, max_seq_length=490):
        """
        Args:
            input_dim: Input feature dimension (38 for robot)
            embed_dim: Embedding dimension (768 to match CLIP-ViT-Large)
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
        """
        super(MotionEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # Project input to embedding dimension
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=embed_dim, 
            max_len=max_seq_length, 
            dropout=dropout
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False  # (T, B, D) format
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, motion, lengths):
        """
        Args:
            motion: (B, T, input_dim) - Motion sequences
            lengths: (B,) - Actual lengths of sequences
        
        Returns:
            motion_emb: (B, embed_dim) - Motion embeddings
        """
        B, T, D = motion.shape
        device = motion.device
        
        # Project to embedding dimension
        x = self.input_proj(motion)  # (B, T, embed_dim)
        x = x.transpose(0, 1)  # (T, B, embed_dim)
        
        # Add positional encoding
        x = self.pos_encoder(x)  # (T, B, embed_dim)
        
        # Create padding mask (True for padded positions)
        pad_mask = torch.zeros((B, T), dtype=torch.bool, device=device)
        for i, length in enumerate(lengths):
            if length < T:
                pad_mask[i, length:] = True
        
        # Apply transformer
        x = self.transformer_encoder(x, src_key_padding_mask=pad_mask)  # (T, B, embed_dim)
        x = x.transpose(0, 1)  # (B, T, embed_dim)
        
        # Average pooling over valid frames
        pooled_list = []
        for i in range(B):
            valid_len = lengths[i]
            if valid_len > 0:
                pooled_list.append(x[i, :valid_len].mean(dim=0))
            else:
                pooled_list.append(torch.zeros(self.embed_dim, device=device))
        
        pooled = torch.stack(pooled_list, dim=0)  # (B, embed_dim)
        
        # Final projection
        pooled = self.dropout(pooled)
        motion_emb = self.fc(pooled)  # (B, embed_dim)
        
        return motion_emb


class ClipMotionAlignModel(nn.Module):
    """
    MoCLIP model for motion-text alignment.
    
    Uses CLIP for text encoding and custom transformer for motion encoding.
    """
    
    def __init__(self, config_or_clip_name="openai/clip-vit-large-patch14", 
                 motion_encoder=None, temperature=0.07, freeze_clip=True,
                 use_openai_clip=None):
        """
        Args:
            config_or_clip_name: Either a config dict or CLIP model name string
                - If dict: should contain 'clip_model_name', 'temperature', 'freeze_clip', etc.
                - If string: Pretrained CLIP model name
                    * For OpenAI CLIP: "ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50", etc.
                    * For Hugging Face CLIP: "openai/clip-vit-large-patch14", etc.
            motion_encoder: MotionEncoder instance (if None, will create default)
            temperature: Temperature parameter for contrastive loss
            freeze_clip: Whether to freeze CLIP weights
            use_openai_clip: Whether to use OpenAI CLIP (None = auto-detect)
        """
        super(ClipMotionAlignModel, self).__init__()
        
        # Handle config dict or direct parameters
        if isinstance(config_or_clip_name, dict):
            config = config_or_clip_name
            clip_model_name = config.get('clip_model_name', "openai/clip-vit-large-patch14")
            temperature = config.get('temperature', 0.07)
            freeze_clip = config.get('freeze_clip', True)
            use_openai_clip = config.get('use_openai_clip', None)
            
            # Create motion encoder from config if not provided
            if motion_encoder is None:
                motion_encoder = MotionEncoder(
                    input_dim=config.get('input_dim', 38),
                    embed_dim=config.get('embed_dim', 768),
                    num_heads=config.get('num_heads', 8),
                    num_layers=config.get('num_layers', 4),
                    dim_feedforward=config.get('dim_feedforward', 2048),
                    dropout=config.get('dropout', 0.2),
                    max_seq_length=config.get('max_seq_length', 490)
                )
        else:
            clip_model_name = config_or_clip_name
        
        # Auto-detect which CLIP to use
        if use_openai_clip is None:
            # Prefer OpenAI CLIP if available (matches project's unet.py)
            use_openai_clip = OPENAI_CLIP_AVAILABLE
        
        self.use_openai_clip = use_openai_clip
        
        if use_openai_clip and OPENAI_CLIP_AVAILABLE:
            # Use OpenAI CLIP (same as models/unet.py)
            print(f'Using OpenAI CLIP: {clip_model_name}')
            # Map Hugging Face names to OpenAI names if needed
            clip_name_map = {
                "openai/clip-vit-large-patch14": "ViT-L/14",
                "openai/clip-vit-base-patch32": "ViT-B/32",
                "openai/clip-vit-base-patch16": "ViT-B/16",
            }
            openai_clip_name = clip_name_map.get(clip_model_name, clip_model_name)
            
            self.clip_model, _ = clip.load(
                openai_clip_name, 
                device='cpu',  # Will move to device later
                jit=False  # Must set jit=False for training
            )
            self.clip_dim = self.clip_model.visual.output_dim
        elif HF_CLIP_AVAILABLE:
            # Use Hugging Face CLIP
            print(f'Using Hugging Face CLIP: {clip_model_name}')
            import os
            local_model_path = os.environ.get('LOCAL_CLIP_PATH', None)
            
            if local_model_path and os.path.exists(local_model_path):
                print(f'Loading CLIP model from local path: {local_model_path}')
                self.clip_model = CLIPModel.from_pretrained(local_model_path, local_files_only=True)
            else:
                # Try to use cache first, then fallback to online
                try:
                    self.clip_model = CLIPModel.from_pretrained(clip_model_name, local_files_only=True)
                    print(f'Loaded CLIP model from cache: {clip_model_name}')
                except Exception as e:
                    print(f'Warning: Could not load from cache: {e}')
                    print('Attempting to download from Hugging Face...')
                    self.clip_model = CLIPModel.from_pretrained(clip_model_name)
            self.clip_dim = self.clip_model.config.projection_dim
        else:
            raise ImportError("Neither OpenAI CLIP nor Hugging Face CLIP is available. "
                            "Please install one: pip install git+https://github.com/openai/CLIP.git "
                            "or pip install transformers")
        
        # Freeze CLIP if requested
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False
            self.clip_model.eval()
        
        # Store CLIP dimension for motion encoder compatibility
        if not hasattr(self, 'clip_dim'):
            if self.use_openai_clip:
                self.clip_dim = self.clip_model.visual.output_dim
            else:
                self.clip_dim = self.clip_model.config.projection_dim
        
        # Motion encoder
        if motion_encoder is None:
            self.motion_encoder = MotionEncoder(
                input_dim=38,
                embed_dim=768,  # Match CLIP-ViT-Large
                num_heads=8,
                num_layers=4,
                max_seq_length=490
            )
        else:
            self.motion_encoder = motion_encoder
        
        # Learnable temperature parameter
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / temperature)))
    
    def encode_motion(self, motion, lengths):
        """
        Encode motion sequences.
        
        Args:
            motion: (B, T, 38) - Motion sequences
            lengths: (B,) - Actual lengths
        
        Returns:
            motion_emb: (B, 768) - Motion embeddings
        """
        return self.motion_encoder(motion, lengths)
    
    def encode_text(self, input_ids=None, attention_mask=None, raw_text=None):
        """
        Encode text using CLIP.
        
        Args:
            input_ids: (B, seq_len) - Tokenized text (for Hugging Face CLIP)
            attention_mask: (B, seq_len) - Attention mask (for Hugging Face CLIP)
            raw_text: List[str] - Raw text strings (for OpenAI CLIP)
        
        Returns:
            text_emb: (B, D) - Text embeddings
        """
        with torch.set_grad_enabled(self.training and not all(not p.requires_grad for p in self.clip_model.parameters())):
            if self.use_openai_clip:
                # OpenAI CLIP: use raw text
                if raw_text is None:
                    raise ValueError("raw_text is required when using OpenAI CLIP")
                device = next(self.clip_model.parameters()).device
                text_tokens = clip.tokenize(raw_text, truncate=True).to(device)
                text_emb = self.clip_model.encode_text(text_tokens).float()
            else:
                # Hugging Face CLIP: use tokenized input
                if input_ids is None:
                    raise ValueError("input_ids is required when using Hugging Face CLIP")
                text_emb = self.clip_model.get_text_features(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
        return text_emb
    
    def forward(self, motion, lengths, input_ids=None, attention_mask=None, raw_text=None):
        """
        Forward pass.
        
        Args:
            motion: (B, T, 38) - Motion sequences
            lengths: (B,) - Actual lengths
            input_ids: (B, seq_len) - Tokenized text (for Hugging Face CLIP)
            attention_mask: (B, seq_len) - Attention mask (for Hugging Face CLIP)
            raw_text: List[str] - Raw text strings (for OpenAI CLIP)
        
        Returns:
            motion_emb: (B, D) - Motion embeddings
            text_emb: (B, D) - Text embeddings
        """
        motion_emb = self.encode_motion(motion, lengths)
        text_emb = self.encode_text(input_ids=input_ids, attention_mask=attention_mask, raw_text=raw_text)
        
        return motion_emb, text_emb


def clip_contrastive_loss(motion_emb, text_emb, logit_scale):
    """
    CLIP-style contrastive loss.
    
    Args:
        motion_emb: (B, D) - Motion embeddings
        text_emb: (B, D) - Text embeddings
        logit_scale: Scalar - Learnable temperature parameter
    
    Returns:
        loss: Scalar - Contrastive loss
    """
    # Clamp logit_scale to prevent numerical issues
    logit_scale = logit_scale.exp().clamp(max=100)
    
    # Normalize embeddings
    motion_emb = F.normalize(motion_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)
    
    # Compute similarity matrix
    logits_per_motion = motion_emb @ text_emb.t() * logit_scale
    logits_per_text = text_emb @ motion_emb.t() * logit_scale
    
    # Ground truth: diagonal elements are positive pairs
    B = motion_emb.size(0)
    ground_truth = torch.arange(B, device=motion_emb.device)
    
    # Cross-entropy loss in both directions
    loss_m2t = F.cross_entropy(logits_per_motion, ground_truth)
    loss_t2m = F.cross_entropy(logits_per_text, ground_truth)
    
    return (loss_m2t + loss_t2m) * 0.5


def compute_retrieval_metrics(motion_emb, text_emb, sample_size=32):
    """
    Compute retrieval metrics (R@1, R@2, R@3) for motion-text matching.
    
    Args:
        motion_emb: (N, D) - Motion embeddings
        text_emb: (N, D) - Text embeddings
        sample_size: Size of each batch for computing metrics
    
    Returns:
        metrics: dict - Dictionary containing:
            - m2t_r1, m2t_r2, m2t_r3: Motion-to-text retrieval
            - t2m_r1, t2m_r2, t2m_r3: Text-to-motion retrieval
    """
    # Normalize embeddings
    motion_emb = F.normalize(motion_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)
    
    num_samples = motion_emb.shape[0]
    
    # Adjust for sample_size
    if num_samples >= sample_size:
        num_full_samples = (num_samples // sample_size) * sample_size
        motion_emb = motion_emb[:num_full_samples]
        text_emb = text_emb[:num_full_samples]
        num_batches = num_full_samples // sample_size
    else:
        num_batches = 1
    
    m2t_r1_list, m2t_r2_list, m2t_r3_list = [], [], []
    t2m_r1_list, t2m_r2_list, t2m_r3_list = [], [], []
    
    for i in range(num_batches):
        start_idx = i * sample_size
        end_idx = (i + 1) * sample_size
        batch_motion = motion_emb[start_idx:end_idx]
        batch_text = text_emb[start_idx:end_idx]
        
        # Similarity matrix
        sim_matrix = batch_motion @ batch_text.t()
        N = sim_matrix.size(0)
        
        # Motion -> Text retrieval
        ranks = []
        for j in range(N):
            sim_row = sim_matrix[j]
            sorted_idx = torch.argsort(sim_row, descending=True)
            rank = (sorted_idx == j).nonzero(as_tuple=True)[0].item()
            ranks.append(rank)
        ranks = torch.tensor(ranks)
        m2t_r1_list.append((ranks < 1).float().mean().item())
        m2t_r2_list.append((ranks < 2).float().mean().item())
        m2t_r3_list.append((ranks < 3).float().mean().item())
        
        # Text -> Motion retrieval
        ranks_t2m = []
        for j in range(N):
            sim_col = sim_matrix[:, j]
            sorted_idx = torch.argsort(sim_col, descending=True)
            rank = (sorted_idx == j).nonzero(as_tuple=True)[0].item()
            ranks_t2m.append(rank)
        ranks_t2m = torch.tensor(ranks_t2m)
        t2m_r1_list.append((ranks_t2m < 1).float().mean().item())
        t2m_r2_list.append((ranks_t2m < 2).float().mean().item())
        t2m_r3_list.append((ranks_t2m < 3).float().mean().item())
    
    # Average across batches
    metrics = {
        'm2t_r1': sum(m2t_r1_list) / len(m2t_r1_list),
        'm2t_r2': sum(m2t_r2_list) / len(m2t_r2_list),
        'm2t_r3': sum(m2t_r3_list) / len(m2t_r3_list),
        't2m_r1': sum(t2m_r1_list) / len(t2m_r1_list),
        't2m_r2': sum(t2m_r2_list) / len(t2m_r2_list),
        't2m_r3': sum(t2m_r3_list) / len(t2m_r3_list),
    }
    
    return metrics

