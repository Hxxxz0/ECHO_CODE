# ECHO WebSocket Server Quick Start

Start the ECHO cloud generator server and test motion generation.

## Prerequisites

```bash
pip install -r requirements_server.txt
```

Verify model checkpoint:
```bash
ls checkpoints/robotv2/robotv2_38d_lite/model/latest.tar
```

## Step 1: Start the Server

```bash
python scripts/server_robot_ws.py \
  --opt_path checkpoints/robotv2/robotv2_38d_lite/opt.txt \
  --which_ckpt latest \
  --gpu_id 0 \
  --port 8000 \
  --host 127.0.0.1
```

Expected output:
```
============================================================
Initializing Motion Generation Server
============================================================
Device: cuda:0
Dim pose: 38
FPS: 50
...
Loaded checkpoint at iteration 50175
Server initialization complete!
Starting WebSocket server on 127.0.0.1:8000
```

## Step 2: SSH Tunnel (if remote)

```bash
ssh -L 8000:127.0.0.1:8000 user@cloud-server
```

## Step 3: Test Client

```python
import asyncio, json, io
import numpy as np
import websockets

async def test():
    uri = "ws://127.0.0.1:8000/ws"
    async with websockets.connect(uri, max_size=50*1024*1024) as ws:
        request = {
            "text": "a person walks forward",
            "motion_length": 4.0,
            "num_inference_steps": 10,
            "seed": 0,
            "adaptive_smooth": True,
            "static_start": True
        }
        await ws.send(json.dumps(request))
        response = await ws.recv()
        if isinstance(response, str):
            print(f"Error: {json.loads(response)['error']}")
            return
        data = np.load(io.BytesIO(response))
        print(f"Fields: {list(data.keys())}")
        print(f"Frames: {len(data['root_pos'])} ({len(data['root_pos'])/50:.2f}s)")
        np.savez("test_motion.npz", **data)
        print("Saved to test_motion.npz")

asyncio.run(test())
```

Run: `python test_client.py`

## Common Parameters

| Parameter | Fast | Balanced | High Quality |
|-----------|------|----------|-------------|
| `num_inference_steps` | 10 (~1s) | 20 (~2s) | 50 (~5s) |
| `smooth` | false | true | true |
| `adaptive_smooth` | false | true | true |
| `static_start` | false | true | true |

## Troubleshooting

- **Connection refused**: check `curl http://127.0.0.1:8000/`
- **CUDA OOM**: reduce `num_inference_steps` or select different GPU with `--gpu_id`
- **Large response timeout**: set `max_size=50*1024*1024` in websocket connect

See [CLIENT_API.md](CLIENT_API.md) for full API documentation.
