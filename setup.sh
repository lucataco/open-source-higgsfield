#!/usr/bin/env bash
# Body Double setup: the app venv, model files (size + sha256 checked) and a node/model check against ComfyUI.
# Never downloads MiniMax H3. Usage: ./setup.sh [--accept-sam-license] [--comfyui=/path/to/ComfyUI]
set -euo pipefail
cd "$(dirname "$0")"
COMFYUI_DIR=${COMFYUI_DIR:-$HOME/ComfyUI}; COMFYUI_COMMIT=7cebca52; CHECK_PORT=${CHECK_PORT:-8190}
for a in "$@"; do case $a in --accept-sam-license) export ACCEPT_SAM=1;; --comfyui=*) COMFYUI_DIR=${a#*=};; esac; done
command -v uv >/dev/null || { echo "missing: uv (curl -LsSf https://astral.sh/uv/install.sh | sh)"; exit 1; }
for c in git ffmpeg ffprobe curl sha256sum; do command -v $c >/dev/null || { echo "missing: $c"; exit 1; }; done
if [ ! -d "$COMFYUI_DIR" ]; then
  git clone https://github.com/Comfy-Org/ComfyUI "$COMFYUI_DIR" && git -C "$COMFYUI_DIR" checkout "$COMFYUI_COMMIT"
  uv venv -p 3.12 "$COMFYUI_DIR/.venv"
  # tested: torch 2.14.1+cu130 on aarch64 (DGX Spark); use the index that matches your CUDA
  uv pip install -p "$COMFYUI_DIR/.venv" torch==2.14.1 torchvision==0.29.1 --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
  uv pip install -p "$COMFYUI_DIR/.venv" -r "$COMFYUI_DIR/requirements.txt"
elif [ "$(git -C "$COMFYUI_DIR" rev-parse --short=8 HEAD)" != "$COMFYUI_COMMIT" ]; then
  echo "note: ComfyUI is not at the tested commit $COMFYUI_COMMIT"
fi
[ -d .venv ] || uv venv -p 3.12 .venv
uv pip install -p .venv -e .
COMFYUI_DIR="$COMFYUI_DIR" scripts/download_models.sh core
# Check nodes and models against a temporary CPU-only ComfyUI on a spare port.
mkdir -p runs
"$COMFYUI_DIR/.venv/bin/python" "$COMFYUI_DIR/main.py" --cpu --disable-all-custom-nodes --listen 127.0.0.1 --port "$CHECK_PORT" > runs/setup_check.log 2>&1 &
CHECK_PID=$!; trap 'kill $CHECK_PID 2>/dev/null || true' EXIT
for _ in $(seq 180); do curl -sf "http://127.0.0.1:$CHECK_PORT/object_info/LoadImage" >/dev/null && break; sleep 1; done
.venv/bin/python scripts/check_env.py --comfyui "http://127.0.0.1:$CHECK_PORT"
echo "Ready. Start with ./run.sh"
