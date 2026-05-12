#!/bin/bash
# Download ECHO model weights from ModelScope
# Usage: bash scripts/download_weights.sh

set -e
DEST="${1:-checkpoints}"
echo "Downloading ECHO model weights to $DEST..."
git clone https://www.modelscope.cn/Hzzzz001/ECHO.git "$DEST"
echo "Done. Weights saved to $DEST/"
