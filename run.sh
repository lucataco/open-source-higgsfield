#!/usr/bin/env bash
# Start ComfyUI (if needed) and comfy-api-proxy, both on loopback. The app talks to ComfyUI only through the proxy.
set -euo pipefail
cd "$(dirname "$0")"
COMFYUI_DIR=${COMFYUI_DIR:-$HOME/ComfyUI}
mkdir -p runs
if ! curl -sf http://127.0.0.1:8188/system_stats >/dev/null; then
  (cd "$COMFYUI_DIR" && nohup .venv/bin/python main.py --listen 127.0.0.1 --port 8188 > "$OLDPWD/runs/comfyui.log" 2>&1 &)
  for _ in $(seq 300); do curl -sf http://127.0.0.1:8188/system_stats >/dev/null && break; sleep 1; done
fi
if ! curl -sf -o /dev/null http://127.0.0.1:8189/api/v2/jobs; then
  nohup .venv/bin/comfy-api-proxy run --comfyui http://127.0.0.1:8188 --host 127.0.0.1 --port 8189 > runs/proxy.log 2>&1 &
  for _ in $(seq 30); do curl -sf -o /dev/null http://127.0.0.1:8189/api/v2/jobs && break; sleep 1; done
fi
echo "ComfyUI :8188 and comfy-api-proxy :8189 are up."
if [ "${1:-}" = "--check" ]; then
  .venv/bin/python scripts/check_env.py --comfyui http://127.0.0.1:8188
  exit $?
fi
echo "Recast: .venv/bin/python -m bodydouble.runner JOB --src clip.mp4 --cast character.png --prompt \"...\""
