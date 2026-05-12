import torch
from .ema import ExponentialMovingAverage
def load_model_weights(model, ckpt_path, use_ema=True, device='cuda:0'):
    """
    Load weights of a model from a checkpoint file.

    Args:
        model (torch.nn.Module): The model to load weights into.
        ckpt_path (str): Path to the checkpoint file.
        use_ema (bool): Whether to use Exponential Moving Average (EMA) weights if available.
    """
    checkpoint = torch.load(ckpt_path,map_location={'cuda:0': str(device)})
    total_iter = checkpoint.get('total_it', 0)

    if "model_ema" in checkpoint and use_ema:
        ema_key = next(iter(checkpoint["model_ema"]))
        is_ema_wrapped = ('module' in ema_key) or ('n_averaged' in ema_key)
        
        if is_ema_wrapped:
            model = ExponentialMovingAverage(model, decay=1.0)
        
        # Remove 'mask' from state_dict if it exists (for backward compatibility)
        # The mask will be regenerated with the correct size in the model
        state_dict = checkpoint["model_ema"]
        if 'mask' in state_dict:
            old_mask_shape = state_dict['mask'].shape
            print(f"Note: Removing 'mask' (shape {old_mask_shape}) from checkpoint - will use new mask with larger size")
            state_dict = {k: v for k, v in state_dict.items() if k != 'mask'}
        
        model.load_state_dict(state_dict, strict=False)
        
        if is_ema_wrapped:
            model = model.module
            print(f'Loading EMA module model from {ckpt_path} with {total_iter} iterations')
        else:
            print(f'Loading EMA model from {ckpt_path} with {total_iter} iterations')
    else:
        # Remove 'mask' from state_dict if it exists
        state_dict = checkpoint['encoder']
        if 'mask' in state_dict:
            old_mask_shape = state_dict['mask'].shape
            print(f"Note: Removing 'mask' (shape {old_mask_shape}) from checkpoint - will use new mask with larger size")
            state_dict = {k: v for k, v in state_dict.items() if k != 'mask'}
        
        model.load_state_dict(state_dict, strict=False)
        print(f'Loading model from {ckpt_path} with {total_iter} iterations')

    return total_iter