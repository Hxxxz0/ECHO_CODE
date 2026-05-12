"""WebSocket server for robot motion generation inference.

This server loads the StableMoFusion model once at startup and continuously
serves motion generation requests via WebSocket. Designed to be accessed
via SSH tunnel for secure remote access.

Usage:
    python scripts/server_robot_ws.py \
        --opt_path ./checkpoints/robot/robot_38d_new/opt.txt \
        --which_ckpt latest \
        --port 8000 \
        --host 127.0.0.1
"""

import sys
import os
import io
import json
import logging
import argparse
from typing import Optional, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn
from os.path import join as pjoin

from accelerate.utils import set_seed
from models.gaussian_diffusion import DiffusePipeline
from utils.model_load import load_model_weights
from models import build_models
from utils.robot_npz_utils import reshape_generated_motion_38d
from options.get_opt import get_opt
from argparse import Namespace

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MotionGenerationServer:
    """WebSocket server for motion generation."""
    
    def __init__(self, opt_path: str, which_ckpt: str = 'latest', 
                 gpu_id: int = 0, no_ema: bool = False, no_fp16: bool = False):
        """Initialize server with model loading.

        Args:
            opt_path: Path to training options file
            which_ckpt: Checkpoint name to load
            gpu_id: GPU device ID
            no_ema: Whether to disable EMA model
            no_fp16: Whether to disable FP16 (default False = FP16 enabled)
        """
        logger.info("=" * 80)
        logger.info("Initializing Motion Generation Server")
        logger.info("=" * 80)
        
        # Load options
        self.opt = Namespace()
        self.opt.no_ema = no_ema
        self.opt.no_fp16 = no_fp16
        self.opt.which_ckpt = which_ckpt
        self.opt.gpu_id = gpu_id
        get_opt(self.opt, opt_path)
        
        # Set device
        self.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        self.opt.device = self.device
        
        logger.info(f"Device: {self.device}")
        logger.info(f"Dataset: {self.opt.dataset_name}")
        logger.info(f"Dim pose: {self.opt.dim_pose}")
        logger.info(f"Joints num: {self.opt.joints_num}")
        logger.info(f"FPS: {self.opt.fps}")
        logger.info(f"Model directory: {self.opt.model_dir}")
        
        # Load 38D normalization stats
        self.mean = np.load(pjoin(self.opt.data_root, 'Mean_38d.npy'))
        self.std = np.load(pjoin(self.opt.data_root, 'Std_38d.npy'))
        logger.info(f"Loaded 38D stats from {self.opt.data_root}")
        
        # Load model
        logger.info(f"Loading model from {self.opt.model_dir}...")
        self.model = build_models(self.opt)
        ckpt_path = pjoin(self.opt.model_dir, which_ckpt + '.tar')
        self.niter = load_model_weights(self.model, ckpt_path, use_ema=not no_ema)
        logger.info(f"✓ Loaded checkpoint at iteration {self.niter}")
        
        # Create pipeline (will be recreated per request with different settings)
        self.torch_dtype = torch.float32 if no_fp16 else torch.float16
        logger.info(f"Using dtype: {self.torch_dtype}")
        
        logger.info("=" * 80)
        logger.info("Server initialization complete!")
        logger.info("=" * 80)
    
    def validate_request(self, request: Dict[str, Any]) -> Optional[str]:
        """Validate incoming request.
        
        Args:
            request: Request dictionary
            
        Returns:
            Error message if validation fails, None otherwise
        """
        # Check required fields
        if 'text' not in request:
            return "Missing required field: 'text'"
        
        if not isinstance(request['text'], str) or not request['text'].strip():
            return "Field 'text' must be a non-empty string"
        
        # Validate motion_length
        motion_length = request.get('motion_length', 4.0)
        if not isinstance(motion_length, (int, float)) or motion_length <= 0:
            return "Field 'motion_length' must be a positive number"
        
        # Use dataset-specific max_motion_length
        max_seconds = self.opt.max_motion_length / self.opt.fps
        if motion_length > max_seconds:
            return f"Field 'motion_length' must be <= {max_seconds:.1f} seconds (max {self.opt.max_motion_length} frames at {self.opt.fps}fps)"
        
        # Validate num_inference_steps
        num_steps = request.get('num_inference_steps', 10)
        if not isinstance(num_steps, int) or num_steps < 1:
            return "Field 'num_inference_steps' must be a positive integer"
        
        # Validate seed
        seed = request.get('seed', 0)
        if not isinstance(seed, int):
            return "Field 'seed' must be an integer"
        
        return None
    
    def generate_motion(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate motion from request.
        
        Args:
            request: Request dictionary with fields:
                - text (str): Motion description
                - motion_length (float): Length in seconds (default: 4.0)
                - num_inference_steps (int): Sampling steps (default: 10)
                - seed (int): Random seed (default: 0)
                - smooth (bool): Enable Savitzky-Golay smoothing (default: False)
                - smooth_window (int): Smoothing window size (default: 5)
                - adaptive_smooth (bool): Enable adaptive smoothing (default: False)
                - static_start (bool): Force static start (default: False)
                - static_frames (int): Number of static frames (default: 2)
                - blend_frames (int): Number of blend frames (default: 8)
        
        Returns:
            Dictionary containing NPZ data or error
        """
        try:
            # Extract parameters
            text = request['text'].strip()
            motion_length = request.get('motion_length', 4.0)
            num_inference_steps = request.get('num_inference_steps', 10)
            seed = request.get('seed', 0)
            
            # Smoothing parameters
            smooth = request.get('smooth', False)
            smooth_window = request.get('smooth_window', 5)
            adaptive_smooth = request.get('adaptive_smooth', False)
            static_start = request.get('static_start', False)
            static_frames = request.get('static_frames', 2)
            blend_frames = request.get('blend_frames', 8)
            
            # Calculate motion length in frames
            motion_frames = int(motion_length * self.opt.fps)
            
            logger.info(f"Generating motion: '{text}' - {motion_frames} frames ({motion_length:.2f}s)")
            logger.info(f"  Steps: {num_inference_steps}, Seed: {seed}, Smooth: {smooth}")
            
            # Set seed
            set_seed(seed)
            
            # Create pipeline with current settings
            pipeline = DiffusePipeline(
                opt=self.opt,
                model=self.model,
                diffuser_name='dpmsolver',
                device=self.device,
                num_inference_steps=num_inference_steps,
                torch_dtype=self.torch_dtype
            )
            
            # Generate motion
            pred_motions = pipeline.generate(
                [text],
                torch.LongTensor([motion_frames])
            )
            
            # Process result
            motion = pred_motions[0]
            motion_np = motion.cpu().numpy() * self.std + self.mean
            
            # Reshape 38D output
            npz_data = reshape_generated_motion_38d(
                motion_np,
                fps=self.opt.fps,
                smooth=smooth,
                smooth_window=smooth_window,
                adaptive_smooth=adaptive_smooth,
                static_start=static_start,
                static_frames=static_frames,
                blend_frames=blend_frames
            )
            logger.info(f"  Generated 38D: joint_pos {npz_data['joint_pos'].shape}, "
                      f"root_pos {npz_data['root_pos'].shape}")
            
            return npz_data
            
        except Exception as e:
            logger.error(f"Error generating motion: {str(e)}", exc_info=True)
            return {
                'error': f'Generation failed: {str(e)}',
                'code': 'GENERATION_ERROR'
            }


# Global server instance
server_instance: Optional[MotionGenerationServer] = None

# FastAPI app
app = FastAPI(title="Robot Motion Generation Server")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "service": "Robot Motion Generation Server",
        "model_iteration": server_instance.niter if server_instance else None,
        "dim_pose": server_instance.opt.dim_pose if server_instance else None,
        "fps": server_instance.opt.fps if server_instance else None
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for motion generation.
    
    Protocol:
        Client sends JSON request with motion parameters
        Server responds with binary NPZ data or JSON error
    """
    await websocket.accept()
    logger.info(f"Client connected: {websocket.client}")
    
    try:
        while True:
            # Receive JSON request
            data = await websocket.receive_text()
            
            try:
                request = json.loads(data)
                logger.info(f"Received request: {request.get('text', 'N/A')[:50]}...")
            except json.JSONDecodeError as e:
                error_response = {
                    'error': f'Invalid JSON: {str(e)}',
                    'code': 'INVALID_JSON'
                }
                await websocket.send_text(json.dumps(error_response))
                continue
            
            # Validate request
            error = server_instance.validate_request(request)
            if error:
                error_response = {
                    'error': error,
                    'code': 'INVALID_REQUEST'
                }
                await websocket.send_text(json.dumps(error_response))
                continue
            
            # Generate motion
            result = server_instance.generate_motion(request)
            
            # Check if error occurred
            if 'error' in result:
                await websocket.send_text(json.dumps(result))
                continue
            
            # Convert to NPZ binary
            bio = io.BytesIO()
            np.savez_compressed(bio, **result)
            payload = bio.getvalue()
            
            # Send binary response
            await websocket.send_bytes(payload)
            logger.info(f"Sent NPZ response: {len(payload)} bytes")
            
    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {websocket.client}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}", exc_info=True)
        try:
            error_response = {
                'error': f'Server error: {str(e)}',
                'code': 'SERVER_ERROR'
            }
            await websocket.send_text(json.dumps(error_response))
        except:
            pass


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='WebSocket server for robot motion generation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--opt_path', type=str, required=True,
                       help='Path to training options file (opt.txt)')
    parser.add_argument('--which_ckpt', type=str, default='latest',
                       help='Checkpoint name to load')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--no_ema', action='store_true',
                       help='Disable EMA model')
    parser.add_argument('--no_fp16', action='store_true', default=False,
                       help='Disable FP16 (disabled by default for stability)')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                       help='Host to bind to (use 127.0.0.1 for SSH tunnel)')
    parser.add_argument('--port', type=int, default=8000,
                       help='Port to bind to')
    
    args = parser.parse_args()
    
    # Initialize server
    global server_instance
    server_instance = MotionGenerationServer(
        opt_path=args.opt_path,
        which_ckpt=args.which_ckpt,
        gpu_id=args.gpu_id,
        no_ema=args.no_ema,
        no_fp16=args.no_fp16
    )
    
    # Run server
    logger.info(f"\nStarting WebSocket server on {args.host}:{args.port}")
    logger.info(f"Connect via: ws://{args.host}:{args.port}/ws")
    if args.host == '127.0.0.1':
        logger.info("Note: Server is bound to localhost. Use SSH tunnel for remote access:")
        logger.info("  ssh -L 8000:127.0.0.1:8000 user@server_ip")
    logger.info("")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == '__main__':
    main()

