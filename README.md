# Body Double

**Same take. New cast. Open weights only.**

[![Tests](https://github.com/lucataco/open-source-higgsfield/actions/workflows/tests.yml/badge.svg)](https://github.com/lucataco/open-source-higgsfield/actions/workflows/tests.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Body Double recasts the performer in a video as a new character. Their motion, the camera move, the rest of the shot
and the original audio stay. It runs locally on stock ComfyUI with open-weights models (SAM 3.1, FLUX 3 Image Klein
and SCAIL-2), and it's an open alternative to Higgsfield Genjutsu.

[![Body Double demo: the original take left of the line, the recast right of it](media/body_double_demo.webp)](media/body_double_demo.mp4)

_Left of the line: the original take, a [Mixkit stock clip](https://mixkit.co/free-stock-video/man-breakdancing-452/).
Right: the same take recast as an original fictional character (AI-generated). This is a 15 s preview; click it for
the full 30 s at 1080p._

## Highlights

- **Follow-cam recasting.** SCAIL-2 does badly on small people, and its replacement mode re-synthesises the whole
  frame. Body Double tracks the performer with SAM 3.1 and renders a virtual camera that keeps them large at the
  model's resolution. Only the recast performer is pasted back into the untouched full-resolution plate, so the set,
  the crowd and the grain stay pixel-identical.
- **Re-identification.** If the track loses the performer (a turn, an occlusion, a hard cut), Body Double re-detects
  everyone and picks the performer again by appearance.
- **Open first frame.** FLUX 3 Image Klein paints the new character into the real first frame (four options), and
  that frame anchors identity for the whole take.
- **Long takes.** Each 81-frame segment is its own ComfyUI job, conditioned on the previous segment's last five
  frames, with a per-segment seed and colour match. QC (seam ratio and background drift) runs before the next segment
  starts, and a flagged segment is re-rolled. Runs resume after a crash.
- **Core nodes only.** No custom node packs, partner nodes or closed APIs.

## How it works

| Stage | What runs | Models |
|---|---|---|
| conform | ffmpeg: full-res frames, audio, 960-wide tracking proxy (HDR tone-mapped) | — |
| detect, track | SAM3_Detect picks the performer, then windowed SAM3_VideoTrack with re-identification | SAM 3.1 |
| crops | A smoothed follow-cam crop per frame, at the model's size | — |
| anchor | Two-image edit of the first crop frame: the performer becomes the cast | FLUX 3 Image Klein |
| segments | One `WanSCAILToVideo` replacement job per 81 frames, with 5-frame overlap and QC | SCAIL-2 int8 + DPO LoRA + lightx2v, UMT5, CLIP-H, Wan VAE |
| assemble | Paste the recast performer into the plate inside the dilated source ∪ cast masks; mux audio | — |

Chunk k starts at frame 76k and is at most 81 frames. The last chunk is padded to 4n+1 frames and trimmed after
assembly, so F frames in gives F frames out (`tests/test_plan.py`).

The runner (`bodydouble/runner.py`) builds each graph in `bodydouble/graphs.py` and queues it through comfy-sdk and
comfy-api-proxy. Example graphs are in `workflows/api/` (API format) and `workflows/ui/` (open them in the ComfyUI
frontend); `scripts/export_workflows.py` and `scripts/export_ui_workflows.py` regenerate them.

## Performance

The 30 s, 720-frame demo at 1280×704 (follow-cam) and 1920×1080 (composite), 6 steps per segment:

| | DGX Spark (GB10) | H100 PCIe 80 GB |
|---|---|---|
| One 81-frame segment | ~8 min | ~2.5 min |
| 10 segments | 80 min | 24 min |
| Assemble | 11 min | 18 min |
| Whole take, including tracking and the first frame | ~100 min | 48 min |

Segments use comfy-kitchen's int8 matmuls and attention, plus block-sparse attention after the first 20% of steps.
They need about 50 GB of GPU memory. Assembly runs on the CPU.

## Quick start

Tested with ComfyUI at commit `7cebca52` on a DGX Spark (aarch64) and an H100 PCIe (x86_64). You need:

- `uv`, `git`, `ffmpeg`, `ffprobe`, `curl` and `sha256sum`, and about 42 GB of disk for models.
- A Hugging Face token from an account that has accepted the
  [FLUX 3 Image Klein license](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy) (the repo is gated).
- To accept Meta's SAM License for SAM 3.1 (`--accept-sam-license`).

```bash
export HF_TOKEN=hf_...
./setup.sh --accept-sam-license    # app venv, models (sha256-checked), node/model check
./run.sh                           # ComfyUI :8188 + comfy-api-proxy :8189 on loopback
./run.sh --check                   # verify nodes and core models against the running server
.venv/bin/python -m bodydouble.runner myjob --size 1280x704 --src clip.mp4 --cast character.png \
  --prompt "who the new cast is: face, hair, outfit"
```

ComfyUI is expected at `$HOME/ComfyUI` (set `COMFYUI_DIR` or pass `--comfyui=/path`); `setup.sh` clones it at the
tested commit if it's missing. comfy-kitchen's fast kernels need torch built for CUDA 13. On NVIDIA drivers older than
580, install `cuda-compat-13-0` and start ComfyUI with `LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat`.

Outputs go to `runs/<job>/<take>/`: the composite at source resolution, plus the follow-cam crop. Runs resume where
they stopped; `--redo <stage>` runs a stage again.

## Usage

| Option | What it does |
|---|---|
| `--src`, `--seconds` | The source clip and how much of it to recast (default 30 s) |
| `--cast`, `--prompt` | The cast image (full body on a plain background works best) and a description of the cast |
| `--size WxH` | Follow-cam and model resolution, e.g. `1280x704` (default `512x896`) |
| `--pick N` | Which person detected in frame 0 to recast (default: the largest; see `detect` in `runs/<job>/state.json`) |
| `--anchor N` | Which of the four first-frame options to use (`runs/<job>/anchors_<size>/sheet.jpg`) |
| `--anchor-prompt` | The FLUX 3 instruction for the first frame; with several people in frame, say which one to replace |
| `--relight FILE` | Add the SCAIL-2 relight LoRA so the cast takes on the scene's light |
| `--cast-edit`, `--props` | Clean the cast image with FLUX 3, and the held props to check for (below) |
| `--take`, `--seed` | Name a take and set its seed; takes share the earlier stages |
| `--stages`, `--redo` | Run only some stages, or run them again |

`--help` lists every option, including model files, LoRA strengths and step counts.

**Held props.** Leave a cigarette, phone or microphone out of the cast image and the prompt. SCAIL-2 draws held props
from the source video wherever the performer holds one; a prop in the references pops in where the performer has
none. Before a take's first segment the runner stops if SAM 3.1 finds one on the cast or the prompt names one;
`--cast-edit "Remove the cigarette. Keep everything else the same."` cleans the cast locally.

**Two or more people.** Pick the performer with `--pick`, and name them in `--anchor-prompt` (for example "Replace
only the man on the right in image 1 ..."); otherwise the first-frame edit can change everyone. The final composite
only replaces the tracked performer.

## Limitations

- When a take mixes close-ups and wide shots, the follow-cam can stay at full frame, so the performer is small in the
  wide shots and keeps more of the source's clothing.
- Upside-down poses can borrow the source performer's hair.
- On bright, plain backgrounds SCAIL-2 can leave a faint halo around the silhouette.
- One performer per run.

## Models and licenses

See [MODELS.md](MODELS.md). `setup.sh` downloads each file from its official upstream; nothing is redistributed here,
and each model keeps its own license. FLUX 3 Image Klein is under the FLUX Kommunity License: non-commercial use, or
commercial use by qualifying users under its revenue cap. MiniMax H3 is not used (MODELS.md explains why).

The repository license, [Apache-2.0](LICENSE), covers code and workflow files only (see [NOTICE](NOTICE)).

## AI-generated content

Body Double's outputs are AI-generated. Keep an AI-generated label on them wherever they are published, and only
recast footage and characters you have the rights to. The demo's sources are recorded in [RIGHTS.md](RIGHTS.md).

## Acknowledgements

Built for Comfy's [Open Source Higgsfield challenge](https://comfy.org/open-source-higgsfield) (#OpenSourceHiggsfield).
Thanks to the authors of SCAIL-2, SAM 3.1, FLUX 3 Image Klein, Wan 2.1 and lightx2v, and to Comfy-Org for ComfyUI,
comfy-kitchen, comfy-sdk and comfy-api-proxy.
