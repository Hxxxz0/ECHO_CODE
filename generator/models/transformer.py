from typing import Union, Optional, Tuple
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import numpy as np

logger = logging.getLogger(__name__)

class TimestepEmbedder(nn.Module):
    """Sinusoidal positional embedding for timesteps"""
    def __init__(self, d_model, max_len=5000):
        super(TimestepEmbedder, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[x]

def make_key_padding_mask(lengths: torch.Tensor, max_len: int, device) -> torch.Tensor:
    """
    Create key padding mask for variable-length sequences.
    
    Args:
        lengths: (B,) actual lengths of each sequence
        max_len: maximum sequence length (T)
        device: torch device
    
    Returns:
        mask: (B, max_len), True means padding (should be ignored in attention)
    """
    batch_size = lengths.shape[0]
    idx = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, -1)  # (B, max_len)
    mask = idx >= lengths.unsqueeze(1)  # (B, max_len), True = padding
    return mask

class TransformerForDiffusion(nn.Module):
    def __init__(self,
            input_dim: int,
            output_dim: int,
            horizon: int,
            n_obs_steps: int = None,
            cond_dim: int = 0,
            n_layer: int = 12,
            n_head: int = 12,
            n_emb: int = 768,
            p_drop_emb: float = 0.1,
            p_drop_attn: float = 0.1,
            causal_attn: bool=False,
            time_as_cond: bool=True,
            obs_as_cond: bool=False,
            n_cond_layers: int = 0,
            # Text conditioning parameters
            clip_dim: int = 512,
            clip_version: str = 'ViT-B/32',
            text_latent_dim: int = 256,
            text_ff_size: int = 2048,
            text_num_heads: int = 4,
            num_text_layers: int = 4,
            cond_mask_prob: float = 0.1
        ) -> None:
        super().__init__()

        # Update cond_dim to use text_latent_dim if using text conditioning
        # This needs to be done before T_cond calculation
        if cond_dim == 0:
            cond_dim = text_latent_dim
            obs_as_cond = True
        
        obs_as_cond = cond_dim > 0

        # compute number of tokens for main trunk and condition encoder
        if n_obs_steps is None:
            n_obs_steps = 77  # Default to CLIP token length for text conditioning
        
        T = horizon
        T_cond = 1
        if not time_as_cond:
            T += 1
            T_cond -= 1
        
        if obs_as_cond:
            assert time_as_cond
            # Ensure T_cond is large enough for text conditioning (CLIP uses 77 tokens)
            cond_len = n_obs_steps
            if text_latent_dim > 0:
                 cond_len = max(cond_len, 77)
            T_cond += cond_len

        # input embedding stem
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        # cond encoder
        self.time_emb = TimestepEmbedder(n_emb)
        self.cond_obs_emb = None
        
        # Text encoder setup
        self.text_latent_dim = text_latent_dim
        self.cond_mask_prob = cond_mask_prob
        self.embed_text = nn.Linear(clip_dim, text_latent_dim)
        self.clip_version = clip_version
        self.clip_model = self.load_and_freeze_clip(clip_version)
        
        textTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=text_latent_dim,
            nhead=text_num_heads,
            dim_feedforward=text_ff_size,
            dropout=p_drop_attn,
            activation='gelu',
            batch_first=True
        )
        self.textTransEncoder = nn.TransformerEncoder(
            textTransEncoderLayer,
            num_layers=num_text_layers
        )
        self.text_ln = nn.LayerNorm(text_latent_dim)
        
        if obs_as_cond:
            self.cond_obs_emb = nn.Linear(cond_dim, n_emb)

        self.cond_pos_emb = None
        self.encoder = None
        self.decoder = None
        encoder_only = False
        if T_cond > 0:
            self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))
            if n_cond_layers > 0:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=n_emb,
                    nhead=n_head,
                    dim_feedforward=4*n_emb,
                    dropout=p_drop_attn,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer=encoder_layer,
                    num_layers=n_cond_layers
                )
            else:
                self.encoder = nn.Sequential(
                    nn.Linear(n_emb, 4 * n_emb),
                    nn.Mish(),
                    nn.Linear(4 * n_emb, n_emb)
                )
            # decoder
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4*n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True # important for stability
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=n_layer
            )
        else:
            # encoder only BERT
            encoder_only = True

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4*n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=n_layer
            )

        # attention mask
        self.causal_attn = causal_attn
        if causal_attn:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # torch.nn.Transformer uses additive mask as opposed to multiplicative mask in minGPT
            # therefore, the upper triangle should be -inf and others (including diag) should be 0.
            # Create a large enough mask that can be sliced for different sequence lengths
            max_len = 512  # Support sequences up to 512 frames
            mask = (torch.triu(torch.ones(max_len, max_len)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            self.register_buffer("mask", mask)
        else:
            self.mask = None
        
        # memory_mask will be computed dynamically in forward() based on actual condition length
        self.memory_mask = None

        # decoder head
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)
        
        # Add local temporal smoothing via depthwise convolution
        # This helps reduce frame-to-frame discontinuities
        self.use_local_smoothing = True
        if self.use_local_smoothing:
            self.dwconv = nn.Conv1d(
                n_emb, n_emb,
                kernel_size=5,
                padding=2,
                groups=n_emb,  # depthwise: each channel convolves independently
                bias=False
            )
            # Initialize as identity mapping (residual style)
            # We use dirac initialization here to ensure the conv layer produces non-zero output
            # which allows the smooth_alpha gate to receive gradients.
            nn.init.dirac_(self.dwconv.weight.data)
            
            # Learnable gate initialized to 0
            # This ensures the layer starts as an identity mapping (x + 0*conv(x) = x)
            # while allowing the model to learn the optimal smoothing factor
            self.smooth_alpha = nn.Parameter(torch.tensor(0.0))
        else:
            self.dwconv = None
            
        # constants
        self.T = T
        self.T_cond = T_cond
        self.horizon = horizon
        self.time_as_cond = time_as_cond
        self.obs_as_cond = obs_as_cond
        self.encoder_only = encoder_only
        self.n_head = n_head  # Save for mask expansion in forward()
        self.input_feats = input_dim  # For compatibility with DiffusePipeline
        self.output_feats = output_dim

        # init
        self.apply(self._init_weights)
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def _init_weights(self, module):
        ignore_types = (nn.Dropout, 
            TimestepEmbedder, 
            nn.TransformerEncoderLayer, 
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential,
            nn.GELU,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d)
        
        # Skip initialization for modules without trainable parameters (like frozen CLIP)
        if not any(p.requires_grad for p in module.parameters(recurse=False)):
            return
            
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            
            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerForDiffusion):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_obs_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            # Skip unknown types (e.g., from CLIP model) instead of raising error
            pass
    
    def get_optim_groups(self, weight_decay: float=1e-3):
        """
        Separate all trainable parameters into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, layernorm/embedding weights,
        positional embeddings, etc.). Frozen parameters (e.g., CLIP) are excluded.
        """

        # Only consider trainable parameters
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                if not p.requires_grad:
                    continue  # Skip frozen parameters (e.g., CLIP model)
                
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, (torch.nn.Conv1d,)):
                    # depthwise conv weights should not be decayed
                    no_decay.add(fpn)

        # special case: position embeddings not decayed
        no_decay.add("pos_emb")
        if self.cond_pos_emb is not None:
            no_decay.add("cond_pos_emb")
        
        # special case: learnable gate for temporal smoothing
        if hasattr(self, 'smooth_alpha') and self.smooth_alpha is not None:
            no_decay.add("smooth_alpha")

        # validate that we considered every trainable parameter
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer parameter groups
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay)) if pn in param_dict],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay)) if pn in param_dict],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups


    def configure_optimizers(self, 
            learning_rate: float=1e-4, 
            weight_decay: float=1e-3,
            betas: Tuple[float, float]=(0.9,0.95)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def load_and_freeze_clip(self, clip_version):
        """Load CLIP model and freeze its parameters"""
        clip_model, _ = clip.load(
            clip_version, device='cpu',
            jit=False)  # Must set jit=False for training

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model
    
    def encode_text(self, raw_text, device):
        """Encode text using CLIP and text transformer"""
        with torch.no_grad():
            texts = clip.tokenize(raw_text, truncate=True).to(device)
            x = self.clip_model.token_embedding(texts).type(self.clip_model.dtype)
            x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.clip_model.transformer(x)
            x = self.clip_model.ln_final(x).type(self.clip_model.dtype)

        # LND -> NLD (B, T, clip_dim) for batch_first text transformer
        x = x.permute(1, 0, 2).float()  # Ensure float32 for downstream learnable layers
        x = self.embed_text(x)  # [batch_size, len, text_latent_dim]
        x = self.textTransEncoder(x)  # batch_first=True
        x = self.text_ln(x)
        # Already (B, T, D)
        return x
    
    def mask_cond(self, bs, force_mask=False, device=None):
        """Mask text condition for classifier-free guidance"""
        if device is None:
            device = next(self.parameters()).device
        if force_mask:
            cond_indices = torch.empty(0, dtype=torch.long, device=device)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=device) * self.cond_mask_prob)
            mask = (1. - mask)
            cond_indices = torch.nonzero(mask).squeeze(-1)
        else:
            cond_indices = torch.arange(bs, device=device)
        
        return cond_indices

    def forward(self, 
        sample: torch.Tensor, 
        timestep: Union[torch.Tensor, float, int], 
        text: Optional[list]=None,
        uncond: bool=False,
        enc_text: Optional[torch.Tensor]=None,
        cond: Optional[torch.Tensor]=None,
        lengths: Optional[torch.Tensor]=None,
        **kwargs):
        """
        sample: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        text: list of strings with input text prompts
        uncond: whether to use text condition
        enc_text: pre-encoded text features (B, N, text_latent_dim)
        cond: (B,T',cond_dim) - kept for backward compatibility
        lengths: (B,) actual sequence lengths for padding mask (optional)
        output: (B,T,input_dim)
        """
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)
        # (B,1,n_emb)

        # 2. Encode text condition
        if text is not None and enc_text is None:
            enc_text = self.encode_text(text, sample.device)
        
        # Use encoded text as cond if provided
        if enc_text is not None and cond is None:
            cond = enc_text
        
        # 3. Apply condition masking for classifier-free guidance
        if not uncond:
            cond_indices = self.mask_cond(sample.shape[0], force_mask=False, device=sample.device)
        else:
            cond_indices = torch.empty(0, dtype=torch.long, device=sample.device)
        
        use_null_mask = None
        if cond is not None:
            # Create null condition (zeros) for masked samples
            null_cond = torch.zeros_like(cond[:1])  # (1, N, text_latent_dim)
            
            # Detect zero conditions (for CFG: samples with zero condition should be treated as masked)
            # Check if any sample has all-zero condition
            cond_norms = cond.norm(dim=(1, 2))  # (B,)
            zero_cond_mask = cond_norms < 1e-6  # Samples with zero condition
            
            # Determine which samples should use null condition
            # A sample should use null condition if:
            # (1) It's not in cond_indices (masked by mask_cond), OR
            # (2) It has zero condition (from CFG)
            use_null_mask = torch.ones(sample.shape[0], dtype=torch.bool, device=sample.device)
            if cond_indices.numel() > 0:
                # Mark samples in cond_indices as should use condition (unless they have zero condition)
                use_null_mask[cond_indices] = False
            
            # Override: samples with zero condition should always use null condition
            use_null_mask[zero_cond_mask] = True
            
            # Apply null condition to masked samples
            if use_null_mask.any():
                cond_masked = cond.clone()
                cond_masked[use_null_mask] = null_cond.expand(use_null_mask.sum().item(), -1, -1)
                cond = cond_masked

        # process input
        input_emb = self.input_emb(sample)

        if self.encoder_only:
            # BERT
            token_embeddings = torch.cat([time_emb, input_emb], dim=1)
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[
                :, :t, :
            ]  # each position maps to a (learnable) vector
            x = self.drop(token_embeddings + position_embeddings)
            # (B,T+1,n_emb)
            
            # Construct key padding mask for variable-length sequences
            src_key_padding_mask = None
            if lengths is not None:
                lengths = lengths.to(sample.device)
                # +1 for time token at the beginning
                src_key_padding_mask = make_key_padding_mask(lengths + 1, t, sample.device)
                # Zero out padding positions in input to minimize their influence on attention
                x = x.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)
            
            # Use causal mask slice for actual sequence length
            mask = self.mask[:t, :t] if self.mask is not None else None
            x = self.encoder(src=x, mask=mask)
            # (B,T+1,n_emb)
            x = x[:,1:,:]
            # (B,T,n_emb)
            
            # Zero out padding positions in output
            if src_key_padding_mask is not None:
                output_padding_mask = src_key_padding_mask[:, 1:]  # Remove time token column
                x = x.masked_fill(output_padding_mask.unsqueeze(-1), 0.0)
        else:
            # encoder
            cond_embeddings = time_emb
            if self.obs_as_cond:
                cond_obs_emb = self.cond_obs_emb(cond)
                # Ensure masked samples do not receive bias-only conditioning
                if use_null_mask is not None and use_null_mask.any():
                    cond_obs_emb = cond_obs_emb.clone()
                    cond_obs_emb[use_null_mask] = 0.0
                # (B,To,n_emb)
                cond_embeddings = torch.cat([cond_embeddings, cond_obs_emb], dim=1)
            tc = cond_embeddings.shape[1]
            position_embeddings = self.cond_pos_emb[
                :, :tc, :
            ]  # each position maps to a (learnable) vector
            x = self.drop(cond_embeddings + position_embeddings)
            x = self.encoder(x)
            memory = x
            # (B,T_cond,n_emb)
            
            # decoder
            token_embeddings = input_emb
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[
                :, :t, :
            ]  # each position maps to a (learnable) vector
            x = self.drop(token_embeddings + position_embeddings)
            # (B,T,n_emb)
            
            # Construct key padding mask for variable-length sequences
            tgt_key_padding_mask = None
            if lengths is not None:
                lengths = lengths.to(sample.device)
                tgt_key_padding_mask = make_key_padding_mask(lengths, t, sample.device)  # (B, T), bool
                # Zero out padding positions in input to minimize their influence on attention
                # This avoids the memory cost of expanding padding mask to (B*n_head, T, T)
                x = x.masked_fill(tgt_key_padding_mask.unsqueeze(-1), 0.0)
            
            # Get causal mask slice for actual sequence length
            if self.mask is not None:
                if t > self.mask.size(0):
                    raise ValueError(
                        f"Sequence length {t} exceeds maximum supported length {self.mask.size(0)}. "
                        f"Increase max_len in transformer initialization."
                    )
                tgt_mask = self.mask[:t, :t]  # (T, T)
            else:
                tgt_mask = None
            
            x = self.decoder(
                tgt=x,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,  # Mask padding tokens in self-attention
                memory_mask=None,  # Text is global - all frames see all text tokens
            )
            # (B,T,n_emb)
            
            # Zero out padding positions in output
            if tgt_key_padding_mask is not None:
                x = x.masked_fill(tgt_key_padding_mask.unsqueeze(-1), 0.0)
        
        # Apply local temporal smoothing to reduce frame-to-frame discontinuities
        # Placed before LayerNorm so the head sees the smoothed, normalized features
        if hasattr(self, 'dwconv') and self.dwconv is not None:
            # x: (B, T, n_emb) -> transpose -> conv -> transpose back
            x_smooth = self.dwconv(x.transpose(1, 2)).transpose(1, 2)  # (B, T, n_emb)
            x = x + self.smooth_alpha * x_smooth  # residual connection with learnable gate
        
        # head
        x = self.ln_f(x)
        x = self.head(x)
        # (B,T,n_out)
        return x
    
    def forward_with_cfg(
        self, 
        sample: torch.Tensor, 
        timestep: Union[torch.Tensor, float, int], 
        text: Optional[list]=None,
        enc_text: Optional[torch.Tensor]=None,
        cfg_scale: float=2.5,
        lengths: Optional[torch.Tensor]=None
    ):
        """
        Classifier-free guidance forward pass.
        
        Args:
            sample: (B,T,input_dim)
            timestep: (B,) or int, diffusion step
            text: list of strings with input text prompts
            enc_text: pre-encoded text features (B, N, text_latent_dim)
            cfg_scale: classifier-free guidance scale
            lengths: (B,) actual sequence lengths for padding mask (optional)
        Returns:
            (B,T,input_dim)
        """
        B = sample.shape[0]
        
        # Encode text if not provided
        if enc_text is None and text is not None:
            enc_text = self.encode_text(text, sample.device)
        
        # Duplicate inputs for conditional and unconditional passes
        combined_sample = torch.cat([sample, sample], dim=0)
        
        # Handle timestep
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif len(timestep.shape) == 0:
            timestep = timestep.unsqueeze(0)
        combined_timestep = torch.cat([timestep.expand(B), timestep.expand(B)], dim=0)
        
        # Duplicate text encoding: first B samples use condition, last B samples use null condition
        if enc_text is not None:
            null_cond = torch.zeros_like(enc_text[:1])  # (1, N, text_latent_dim)
            combined_enc_text = torch.cat([enc_text, null_cond.expand(B, -1, -1)], dim=0)
        else:
            combined_enc_text = None
        
        # Duplicate lengths for conditional and unconditional passes
        combined_lengths = None
        if lengths is not None:
            combined_lengths = torch.cat([lengths, lengths], dim=0)
        
        # Temporarily store original training state and set to eval to avoid random masking
        original_training = self.training
        self.eval()
        
        # Forward pass: first B samples will use condition, last B samples use null condition (zeros)
        # The forward method will detect that cond_indices in eval mode returns all indices,
        # but we've already set last B samples to zero condition, so they'll be treated as masked
        out = self.forward(
            sample=combined_sample,
            timestep=combined_timestep,
            enc_text=combined_enc_text,
            uncond=False,
            lengths=combined_lengths
        )
        
        # Restore training state
        self.train(original_training)
        
        # Split conditional and unconditional outputs
        out_cond, out_uncond = torch.split(out, B, dim=0)
        
        # Apply CFG: out_uncond + cfg_scale * (out_cond - out_uncond)
        return out_uncond + cfg_scale * (out_cond - out_uncond)


def test():
    # GPT with time embedding
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        # cond_dim=10,
        causal_attn=True,
        # time_as_cond=False,
        # n_cond_layers=4
    )
    opt = transformer.configure_optimizers()

    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    out = transformer(sample, timestep)
    

    # GPT with time embedding and obs cond
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        cond_dim=10,
        causal_attn=True,
        # time_as_cond=False,
        # n_cond_layers=4
    )
    opt = transformer.configure_optimizers()
    
    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    cond = torch.zeros((4,4,10))
    out = transformer(sample, timestep, cond)

    # GPT with time embedding and obs cond and encoder
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        cond_dim=10,
        causal_attn=True,
        # time_as_cond=False,
        n_cond_layers=4
    )
    opt = transformer.configure_optimizers()
    
    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    cond = torch.zeros((4,4,10))
    out = transformer(sample, timestep, cond)

    # BERT with time embedding token
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        # cond_dim=10,
        # causal_attn=True,
        time_as_cond=False,
        # n_cond_layers=4
    )
    opt = transformer.configure_optimizers()

    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    out = transformer(sample, timestep)
