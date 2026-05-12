

from .unet import EchoUnet
from .transformer import TransformerForDiffusion


__all__ = ['EchoUnet', 'TransformerForDiffusion', 'build_models']


def build_models(opt):
    print('Initializing model ...')

    model_type = getattr(opt, 'model_type', 'unet')

    if model_type == 'unet':
        model = EchoUnet(
            input_feats=opt.dim_pose, 
            text_latent_dim=opt.text_latent_dim,
            base_dim = opt.base_dim,
            dim_mults = opt.dim_mults,
            time_dim=opt.time_dim,
            adagn = not opt.no_adagn,
            zero = True,
            no_eff = opt.no_eff,
            cond_mask_prob = getattr(opt, 'cond_mask_prob', 0.)
        )
    elif model_type == 'transformer':
        model = TransformerForDiffusion(
            input_dim=opt.dim_pose,
            output_dim=opt.dim_pose,
            horizon=opt.max_motion_length,
            n_obs_steps=77,  # CLIP always produces 77 tokens
            clip_dim=512,
            text_latent_dim=opt.text_latent_dim,
            n_layer=getattr(opt, 'n_layer', 12),
            n_head=getattr(opt, 'n_head', 12),
            n_emb=getattr(opt, 'n_emb', 768),
            causal_attn=getattr(opt, 'causal_attn', False),
            n_cond_layers=getattr(opt, 'n_cond_layers', 0),
            cond_mask_prob=getattr(opt, 'cond_mask_prob', 0.1),
            time_as_cond=True,
            p_drop_emb=0.1,
            p_drop_attn=0.1
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'unet' or 'transformer'.")

    return model

