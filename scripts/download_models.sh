#!/usr/bin/env bash
# Download every model in scripts/models.tsv into $COMFYUI_DIR/models, skipping files that already match.
# Set ACCEPT_SAM=1 to accept Meta's SAM License for SAM 3.1. FLUX 3 Image Klein is gated: accept its license on
# huggingface.co/black-forest-labs/flux-3-image-klein-comfy and set HF_TOKEN.
set -euo pipefail
COMFYUI_DIR=${COMFYUI_DIR:-$HOME/ComfyUI}
PROFILE=${1:-core}
tail -n +2 "$(dirname "$0")/models.tsv" | while IFS=$'\t' read -r folder file url bytes sha license profile; do
  [ "$profile" = "$PROFILE" ] || continue
  out="$COMFYUI_DIR/models/$folder/$file"
  if [ -f "$out" ] && [ "$(stat -c %s "$out")" = "$bytes" ]; then echo "ok    $file"; continue; fi
  if [ "$license" = "SAM-License" ] && [ "${ACCEPT_SAM:-0}" != 1 ]; then
    echo "skip  $file: run setup.sh --accept-sam-license to accept Meta's SAM License"; exit 1
  fi
  if [ "$license" = "FLUX-Kommunity-1.0" ] && [ -z "${HF_TOKEN:-}" ]; then
    echo "skip  $file: accept the FLUX 3 license on Hugging Face and set HF_TOKEN"; exit 1
  fi
  echo "get   $file ($license, $bytes bytes)"
  mkdir -p "$(dirname "$out")"
  curl -L -C - --fail --retry 5 ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} -o "$out.part" "$url"
  echo "$sha  $out.part" | sha256sum -c --quiet
  mv "$out.part" "$out"
done
