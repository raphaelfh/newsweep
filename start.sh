#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== newsweep Autocomplete Server ==="
echo "Model: sweep-next-edit-v2-7B (MLX 4-bit + speculative decoding)"
echo "Device: Apple Silicon (MLX)"
echo "Port: 8741"
echo ""

# Activate venv if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Start the server
exec python -m uvicorn sweep_local.server:app --host 0.0.0.0 --port 8741
