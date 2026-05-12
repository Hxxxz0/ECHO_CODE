# ECHO Motion Generation Client API

Integration guide for the ECHO cloud generator WebSocket server. Text-to-motion generation returning 38D NPZ data for robot control.

---

## Table of Contents

1. [Connection Setup](#1-connection-setup)
2. [Request Protocol](#2-request-protocol)
3. [Response Protocol](#3-response-protocol)
4. [Data Format Details](#4-data-format-details)
5. [Visualization Guidelines](#5-visualization-guidelines)
6. [Error Handling](#6-error-handling)
7. [Examples](#7-examples)

---

## 1. Connection Setup

### SSH Tunnel for Remote Access

The server binds to `127.0.0.1:8000` by default for security. To access it remotely, establish an SSH tunnel:

```bash
# On your local machine
ssh -L 8000:127.0.0.1:8000 username@server_hostname

# Keep this terminal open while using the service
```

Now you can connect to `ws://127.0.0.1:8000/ws` from your local machine as if the server were running locally.

### WebSocket Connection

**Endpoint:** `ws://127.0.0.1:8000/ws`

**Recommended Libraries:**
- **Python:** `websockets` or `websocket-client`
- **JavaScript/Node.js:** `ws`
- **C++:** `websocketpp` or `Boost.Beast`
- **Rust:** `tokio-tungstenite`

### Health Check

Before connecting to WebSocket, you can check server status:

```bash
curl http://127.0.0.1:8000/
```

Response:
```json
{
  "status": "running",
  "service": "ECHO Motion Generation Server",
  "model_iteration": 50304,
  "dim_pose": 38,
  "fps": 50
}
```

---

## 2. Request Protocol

### Message Format

Clients send JSON-formatted text messages to the WebSocket endpoint.

### Request Schema

```json
{
  "text": "a person walks forward",
  "motion_length": 4.0,
  "num_inference_steps": 10,
  "seed": 0,
  
  "smooth": false,
  "smooth_window": 5,
  "adaptive_smooth": true,
  "static_start": true,
  "static_frames": 2,
  "blend_frames": 8
}
```

### Field Descriptions

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `text` | string | **Yes** | - | Natural language description of the desired motion |
| `motion_length` | float | No | 4.0 | Duration of motion in seconds (max: 9.8) |
| `num_inference_steps` | int | No | 10 | Number of denoising steps (10=fast, 50=high quality) |
| `seed` | int | No | 0 | Random seed for reproducibility |
| `smooth` | bool | No | See below | Enable basic Savitzky-Golay smoothing |
| `smooth_window` | int | No | 5 | Smoothing window size (must be odd) |
| `adaptive_smooth` | bool | No | false | Enable adaptive smoothing (stronger on initial frames) |
| `static_start` | bool | No | false | Force static start to reduce initial velocity |
| `static_frames` | int | No | 2 | Number of completely static frames at start |
| `blend_frames` | int | No | 8 | Number of frames to blend from static to motion |

### Smoothing Default

The `smooth` parameter defaults to `false`. Enable `adaptive_smooth` and `static_start` for smoother output with reduced initial jitter.

### Valid Value Ranges

- `motion_length`: 0.1 to 9.8 seconds (5 to 490 frames at 50 FPS)
- `num_inference_steps`: 1 to 1000 (recommended: 10-50)
- `seed`: any integer
- `smooth_window`: odd integers ≥ 3 (recommended: 5, 7, 9, 11)

---

## 3. Response Protocol

The server responds with either **binary NPZ data** (success) or **JSON error message** (failure).

### Success Response: Binary NPZ

**Content Type:** Binary (application/octet-stream)

The response is a compressed NumPy NPZ file containing `joint_pos`, `root_pos`, and `root_rot` arrays in 38D format.

### Error Response: JSON

```json
{
  "error": "Missing required field: 'text'",
  "code": "INVALID_REQUEST"
}
```

**Error Codes:**
- `INVALID_JSON`: Malformed JSON in request
- `INVALID_REQUEST`: Missing or invalid required fields
- `GENERATION_ERROR`: Model inference failed
- `UNSUPPORTED_FORMAT`: Unknown dim_pose format
- `SERVER_ERROR`: Unexpected server error

---

## 4. Data Format Details

### Parsing NPZ Files

**Python:**
```python
import io
import numpy as np

# Receive binary data from WebSocket
npz_bytes = await websocket.recv()

# Parse NPZ
data = np.load(io.BytesIO(npz_bytes))

# Access fields
fps = int(data['fps'][0])
joint_pos = data['joint_pos']  # (T, 29)
root_pos = data['root_pos']    # (T, 3)
root_rot = data['root_rot']    # (T, 4)
```

**C++ (using cnpy library):**
```cpp
#include <cnpy.h>
#include <sstream>

// Receive binary data into std::vector<uint8_t> npz_bytes
std::stringstream ss;
ss.write(reinterpret_cast<const char*>(npz_bytes.data()), npz_bytes.size());

cnpy::npz_t npz = cnpy::npz_load(ss);

// Access fields
auto joint_pos = npz["joint_pos"];  // shape: (T, 29)
auto root_pos = npz["root_pos"];    // shape: (T, 3)
```

**JavaScript (Node.js with zarr.js):**
```javascript
const { openArray } = require('zarr');
const pako = require('pako');

// Receive binary data
const npzBytes = await websocket.recv();

// Note: NPZ parsing in JavaScript is more complex
// Consider using a Python microservice or C++ addon
```

### 38D Format (Minimal Representation)

Used for efficient robot control with joint angles and root state.

**Fields:**

| Field | Shape | Type | Description |
|-------|-------|------|-------------|
| `fps` | (1,) | int32 | Frame rate (always 50) |
| `joint_pos` | (T, 29) | float32 | Joint angles in radians |
| `root_pos` | (T, 3) | float32 | Root position [x, y, z] in meters |
| `root_rot` | (T, 4) | float32 | Root rotation quaternion [w, x, y, z] |

**Coordinate System:**
- World frame, right-handed coordinate system
- +X: forward, +Y: left, +Z: up
- Quaternion format: [w, x, y, z] (scalar-first)

**Example Data Shapes:**
```
For 4.0 second motion at 50 FPS (200 frames):
  fps: (1,) = [50]
  joint_pos: (200, 29) = 200 frames × 29 joints
  root_pos: (200, 3) = 200 frames × XYZ
  root_rot: (200, 4) = 200 frames × quaternion
```

## 5. Visualization

```python
def animate_motion(npz_data):
    fps = int(npz_data['fps'][0])
    dt = 1.0 / fps
    root_pos = npz_data['root_pos']   # (T, 3)
    root_rot = npz_data['root_rot']   # (T, 4)
    joint_pos = npz_data['joint_pos'] # (T, 29)

    for t in range(len(root_pos)):
        set_root_position(root_pos[t])
        set_root_rotation(root_rot[t])
        for j in range(29):
            set_joint_angle(j, joint_pos[t, j])
        render_frame()
        sleep(dt)
```

---

## 6. Error Handling

### Connection Errors

```python
import websockets
import asyncio

async def connect_with_retry(uri, max_retries=5):
    for attempt in range(max_retries):
        try:
            ws = await websockets.connect(uri, max_size=50*1024*1024)
            return ws
        except (ConnectionRefusedError, OSError) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Connection failed, retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                raise
```

### Request Validation

Always validate your request locally before sending:

```python
def validate_request(req):
    if not req.get('text') or not isinstance(req['text'], str):
        raise ValueError("'text' is required and must be a string")
    
    if req.get('motion_length', 4.0) > 9.8:
        raise ValueError("'motion_length' must be <= 9.8 seconds")
    
    if req.get('num_inference_steps', 10) < 1:
        raise ValueError("'num_inference_steps' must be >= 1")
    
    return True
```

### Handling Server Errors

```python
async def send_request(ws, request):
    await ws.send(json.dumps(request))
    response = await ws.recv()
    
    # Check if response is error (JSON) or success (binary)
    if isinstance(response, str):
        error = json.loads(response)
        if 'error' in error:
            raise RuntimeError(f"Server error [{error['code']}]: {error['error']}")
    
    # Response is binary NPZ
    return response
```

---

## 7. Examples

### Python Client (Minimal Example)

```python
#!/usr/bin/env python3
"""Minimal client example for robot motion generation."""

import asyncio
import json
import io
import numpy as np
import websockets

async def generate_motion(text, motion_length=4.0):
    """Generate a single motion from text description."""
    uri = "ws://127.0.0.1:8000/ws"
    
    async with websockets.connect(uri, max_size=50*1024*1024) as ws:
        # Prepare request
        request = {
            "text": text,
            "motion_length": motion_length,
            "num_inference_steps": 10,
            "seed": 0,
            "adaptive_smooth": True,
            "static_start": True
        }
        
        print(f"Requesting: {text}")
        
        # Send request
        await ws.send(json.dumps(request))
        
        # Receive response
        response = await ws.recv()
        
        # Check for error
        if isinstance(response, str):
            error = json.loads(response)
            print(f"Error: {error['error']}")
            return None
        
        # Parse NPZ
        data = np.load(io.BytesIO(response))
        print(f"Received motion: {len(data['root_pos'])} frames")
        
        return data

if __name__ == '__main__':
    # Example usage
    data = asyncio.run(generate_motion("a person walks forward", 4.0))
    
    if data:
        print(f"Motion data fields: {list(data.keys())}")
        print(f"FPS: {data['fps'][0]}")
        print(f"Duration: {len(data['root_pos']) / data['fps'][0]:.2f}s")
```

### Interactive Client Example

```python
#!/usr/bin/env python3
"""Interactive client with continuous connection."""

import asyncio
import json
import io
import numpy as np
import websockets

async def interactive_client():
    """Run interactive client that accepts text prompts."""
    uri = "ws://127.0.0.1:8000/ws"
    
    print("Connecting to server...")
    async with websockets.connect(uri, max_size=50*1024*1024) as ws:
        print("Connected! Enter text prompts (or 'quit' to exit)")
        
        while True:
            # Get user input
            text = input("\nPrompt: ").strip()
            if text.lower() in ['quit', 'exit', 'q']:
                break
            
            if not text:
                continue
            
            # Prepare request
            request = {
                "text": text,
                "motion_length": 4.0,
                "num_inference_steps": 10,
                "seed": 0
            }
            
            try:
                # Send request
                await ws.send(json.dumps(request))
                print("Generating motion...")
                
                # Receive response
                response = await ws.recv()
                
                # Check for error
                if isinstance(response, str):
                    error = json.loads(response)
                    print(f"❌ Error: {error['error']}")
                    continue
                
                # Parse NPZ
                data = np.load(io.BytesIO(response))
                fps = int(data['fps'][0])
                frames = len(data['root_pos']) if 'root_pos' in data else len(data['body_pos_w'])
                
                print(f"✓ Received: {frames} frames ({frames/fps:.2f}s)")
                print(f"  Fields: {', '.join(data.keys())}")
                
                # Save to file
                filename = f"motion_{text[:30].replace(' ', '_')}.npz"
                np.savez(filename, **data)
                print(f"  Saved to: {filename}")
                
            except Exception as e:
                print(f"Error: {e}")

if __name__ == '__main__':
    try:
        asyncio.run(interactive_client())
    except KeyboardInterrupt:
        print("\nGoodbye!")
```

### C++ Client Skeleton (websocketpp)

```cpp
#include <websocketpp/config/asio_client.hpp>
#include <websocketpp/client.hpp>
#include <json/json.h>
#include <cnpy.h>

typedef websocketpp::client<websocketpp::config::asio_tls_client> client;

class MotionClient {
public:
    MotionClient() {
        m_client.init_asio();
        m_client.set_message_handler([this](auto hdl, auto msg) {
            this->on_message(hdl, msg);
        });
    }
    
    void generate_motion(const std::string& text, float motion_length) {
        // Create JSON request
        Json::Value request;
        request["text"] = text;
        request["motion_length"] = motion_length;
        request["num_inference_steps"] = 10;
        request["seed"] = 0;
        
        // Send request
        m_client.send(m_hdl, Json::writeString(request), 
                     websocketpp::frame::opcode::text);
    }
    
private:
    void on_message(websocketpp::connection_hdl hdl, client::message_ptr msg) {
        if (msg->get_opcode() == websocketpp::frame::opcode::binary) {
            // Parse NPZ
            auto payload = msg->get_payload();
            std::stringstream ss(payload);
            auto npz = cnpy::npz_load(ss);
            
            // Access motion data
            auto root_pos = npz["root_pos"];
            // ... process motion data
        }
    }
    
    client m_client;
    websocketpp::connection_hdl m_hdl;
};
```

---

## Troubleshooting

### Common Issues

1. **Connection Refused**
   - Ensure SSH tunnel is active
   - Check server is running: `curl http://127.0.0.1:8000/`
   - Verify port numbers match

2. **"Invalid JSON" Error**
   - Check JSON formatting with online validator
   - Ensure all string values are properly quoted
   - Use `json.dumps()` in Python, not manual string concatenation

3. **"GENERATION_ERROR"**
   - Try reducing `num_inference_steps`
   - Check GPU memory on server
   - Verify model checkpoint exists

4. **Large Response Timeout**
   - Increase WebSocket `max_size` parameter (default 1MB is too small)
   - Set to at least 50MB: `websockets.connect(uri, max_size=50*1024*1024)`

5. **Unexpected Motion Quality**
   - Try enabling smoothing: `"adaptive_smooth": true, "static_start": true`
   - Increase inference steps: `"num_inference_steps": 50`
   - Adjust motion length (very short/long motions may have artifacts)

---

## Performance Notes

- **Latency:** Typical generation time is 1-5 seconds depending on:
  - `num_inference_steps` (10=fast, 50=slow)
  - Motion length (longer = more computation)
  - GPU model and load
  
- **Throughput:** Server handles one request at a time per connection
  - For parallel generation, open multiple WebSocket connections
  - Server is thread-safe and can handle concurrent connections

- **Data Size:** NPZ response ~50-500 KB for 4-second motion

---

## Support

For issues or questions:
1. Check server logs for detailed error messages
2. Verify your request against the examples in this document
3. Test with the minimal Python client example first
4. Check network connectivity and SSH tunnel

---

**Last Updated:** 2026-02-04  
**Server Version:** ECHO 38D robot motion generation

