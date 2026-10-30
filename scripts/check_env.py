"""Check a ComfyUI server has every node, input name and model file Body Double's workflows use."""
import argparse
import csv
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOADERS = {"UNETLoader": "unet_name", "LoraLoaderModelOnly": "lora_name", "CLIPLoader": "clip_name", "VAELoader": "vae_name",
           "CLIPVisionLoader": "clip_name", "CheckpointLoaderSimple": "ckpt_name"}


def get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--comfyui", default="http://127.0.0.1:8188")
    a = p.parse_args()
    version = get(f"{a.comfyui}/system_stats")["system"]["comfyui_version"]
    errors = []
    available = {}
    for wf in sorted((ROOT / "workflows/api").glob("*.api.json")):
        for nid, node in json.loads(wf.read_text()).items():
            cls = node["class_type"]
            info = get(f"{a.comfyui}/object_info/{cls}").get(cls)
            if info is None:
                errors.append(f"{wf.name}: node {cls} missing (ComfyUI {version})")
                continue
            names = set(info["input"].get("required", {})) | set(info["input"].get("optional", {}))
            errors += [f"{wf.name}: {cls} has no input {k}" for k in node["inputs"] if k.split(".")[0] not in names]
            if cls in LOADERS:
                spec = (info["input"]["required"].get(LOADERS[cls]) or [[]])[0]
                available.setdefault(cls, set(spec if isinstance(spec, list) else spec_options(info, LOADERS[cls])))
    models = [r for r in csv.DictReader(open(ROOT / "scripts/models.tsv"), delimiter="\t") if r["profile"] == "core"]
    every = set().union(*available.values()) if available else set()
    errors += [f"model {r['folder']}/{r['filename']} not visible to ComfyUI" for r in models if r["filename"] not in every]
    print(f"ComfyUI {version}: {'OK' if not errors else str(len(errors)) + ' problem(s)'}")
    for e in errors:
        print("  " + e)
    sys.exit(1 if errors else 0)


def spec_options(info, name):
    spec = info["input"]["required"][name]
    return spec[1].get("options", []) if len(spec) > 1 else []


if __name__ == "__main__":
    main()
