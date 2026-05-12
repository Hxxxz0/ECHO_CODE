# WebSocket Server Quick Start Guide

This guide will help you quickly set up and test the Robot Motion Generation WebSocket server.

## Prerequisites

1. **Install server dependencies:**
```bash
pip install -r requirements_server.txt
```

2. **Verify model checkpoint exists:**
```bash
ls checkpoints/robot/robot_38d_new/model/latest.tar
```

## Step 1: Start the Server

On your GPU server, run:

```bash
python scripts/server_robot_ws.py \
  --opt_path ./checkpoints/robot/robot_38d_new/opt.txt \
  --which_ckpt latest \
  --gpu_id 0 \
  --port 8000 \
  --host 127.0.0.1
```

**Expected output:**
```
============================================================
Initializing Motion Generation Server
============================================================
Device: cuda:0
Dataset: robot
Dim pose: 38
Joints num: 30
FPS: 50
...
✓ Loaded checkpoint at iteration 50304
============================================================
Server initialization complete!
============================================================

Starting WebSocket server on 127.0.0.1:8000
Connect via: ws://127.0.0.1:8000/ws
```

The server is now running and ready to accept connections!

## Step 2: Set Up SSH Tunnel (if remote)

If the server is on a remote machine, open a **new terminal** on your local machine:

```bash
ssh -L 8000:127.0.0.1:8000 your_username@your_server_ip
```

Keep this terminal open. Now `localhost:8000` on your local machine connects to the server.

## Step 3: Test with a Simple Client

Create a test file `test_client.py`:

```python
#!/usr/bin/env python3
import asyncio
import json
import io
import numpy as np
import websockets

async def test_generation():
    uri = "ws://127.0.0.1:8000/ws"
    
    print("Connecting to server...")
    async with websockets.connect(uri, max_size=50*1024*1024) as ws:
        print("✓ Connected!")
        
        # Test request
        request = {
            "text": "a person walks forward",
            "motion_length": 4.0,
            "num_inference_steps": 10,
            "seed": 0,
            "adaptive_smooth": True,
            "static_start": True
        }
        
        print(f"\nSending request: {request['text']}")
        await ws.send(json.dumps(request))
        
        print("Waiting for response...")
        response = await ws.recv()
        
        # Check if error
        if isinstance(response, str):
            error = json.loads(response)
            print(f"❌ Error: {error['error']}")
            return
        
        # Parse NPZ
        print("✓ Received binary NPZ data")
        data = np.load(io.BytesIO(response))
        
        print(f"\n=== Motion Data ===")
        print(f"Fields: {list(data.keys())}")
        print(f"FPS: {data['fps'][0]}")
        print(f"Frames: {len(data['root_pos'])}")
        print(f"Duration: {len(data['root_pos']) / data['fps'][0]:.2f}s")
        print(f"Root position shape: {data['root_pos'].shape}")
        print(f"Joint position shape: {data['joint_pos'].shape}")
        
        # Save to file
        filename = "test_motion.npz"
        np.savez(filename, **data)
        print(f"\n✓ Saved to {filename}")

if __name__ == '__main__':
    try:
        asyncio.run(test_generation())
    except KeyboardInterrupt:
        print("\nAborted")
    except Exception as e:
        print(f"Error: {e}")
```

Run the test:

```bash
python test_client.py
```

**Expected output:**
```
Connecting to server...
✓ Connected!

Sending request: a person walks forward
Waiting for response...
✓ Received binary NPZ data

=== Motion Data ===
Fields: ['fps', 'joint_pos', 'root_pos', 'root_rot']
FPS: 50
Frames: 200
Duration: 4.00s
Root position shape: (200, 3)
Joint position shape: (200, 29)

✓ Saved to test_motion.npz
```

## Step 4: Verify Server Logs

Check the server terminal for logs:

```
Received request: a person walks forward...
Generating motion: 'a person walks forward' - 200 frames (4.00s)
  Steps: 10, Seed: 0, Smooth: False
  Generated 38D format: joint_pos (200, 29), root_pos (200, 3)
Sent NPZ response: 45123 bytes
```

## Troubleshooting

### Connection Refused

1. Check server is running: `curl http://127.0.0.1:8000/`
2. Verify SSH tunnel is active (if remote)
3. Check firewall settings

### CUDA Out of Memory

- Reduce `num_inference_steps` to 5
- Use `--gpu_id` to select a different GPU
- Restart the server

### Import Errors

Install missing dependencies:
```bash
pip install fastapi uvicorn websockets
```

## Next Steps

1. **Read the full API documentation:** [docs/CLIENT_API.md](CLIENT_API.md)
2. **Integrate with your robot control system**
3. **Experiment with different prompts and parameters**
4. **Build a visualization client** (see examples in CLIENT_API.md)

## Common Parameters

Adjust these in your requests for different results:

- **Quality vs Speed:**
  - Fast: `"num_inference_steps": 10` (1-2 seconds)
  - Balanced: `"num_inference_steps": 20` (2-3 seconds)
  - High quality: `"num_inference_steps": 50` (5-8 seconds)

- **Smoothing:**
  - For 38D: Add `"smooth": true, "adaptive_smooth": true`
  - For 239D: Already smooth by default

- **Reduce initial jitter:**
  - Add `"static_start": true, "static_frames": 2, "blend_frames": 8`

## Server Management

**Stop the server:**
- Press `Ctrl+C` in the server terminal

**Run server in background (Linux):**
```bash
nohup python scripts/server_robot_ws.py \
  --opt_path ./checkpoints/robot/robot_38d_new/opt.txt \
  --which_ckpt latest \
  > server.log 2>&1 &

# Check logs
tail -f server.log

# Stop server
pkill -f server_robot_ws.py
```

**Monitor GPU usage:**
```bash
watch -n 1 nvidia-smi
```

---

For more details, see:
- **[Client API Documentation](CLIENT_API.md)** - Complete integration guide
- **[README.md](../README.md)** - Project overview and setup



