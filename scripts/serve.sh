#!/bin/bash
# Start ECHO WebSocket server
# Usage: bash scripts/serve.sh [checkpoint_name] [port]

CKPT="${1:-checkpoints/robotv2/robotv2_38d_lite}"
PORT="${2:-8000}"

cd "$(dirname "$0")/../generator"
python scripts/server.py \
    --opt_path "${CKPT}/opt.txt" \
    --which_ckpt latest \
    --port "$PORT" \
    --host 127.0.0.1
