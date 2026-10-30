# Models and licenses

`setup.sh` downloads each file from its official upstream and checks its sha256 (`scripts/models.tsv`); weights are
never redistributed here. The repository license (Apache-2.0) covers code and workflow files only.

| Role | File (ComfyUI folder) | Source | License |
|---|---|---|---|
| Video core | `wan2.1_14B_SCAIL_2_int8_convrot` (diffusion_models) | Comfy-Org/SCAIL-2 | MIT |
| Identity DPO LoRA | `wan2.1_SCAIL_2_DPO_lora_bf16` (loras) | Comfy-Org/SCAIL-2 | MIT |
| Relight LoRA (optional) | `wan2.1_SCAIL_2_relight_lora_bf16` (loras) | Comfy-Org/SCAIL-2 | MIT |
| Step-distill LoRA | `Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64` or the T2V variant (loras) | lightx2v | Apache-2.0 |
| Text encoder | `umt5_xxl_fp8_e4m3fn_scaled` (text_encoders) | Comfy-Org/Wan_2.1_ComfyUI_repackaged | Apache-2.0 |
| CLIP vision | `clip_vision_h` (clip_vision) | Comfy-Org/Wan_2.1_ComfyUI_repackaged | Apache-2.0 |
| Video VAE | `wan_2.1_vae` (vae) | Comfy-Org/Wan_2.1_ComfyUI_repackaged | Apache-2.0 |
| Masks and tracking | `sam3.1_multiplex_fp16` (checkpoints) | Comfy-Org/sam3.1 | SAM License (Meta) |
| First-frame anchor | `flux3-image-klein-int8-convrot` (diffusion_models) | black-forest-labs/flux-3-image-klein-comfy (gated) | FLUX Kommunity License v1.0 |
| Klein text encoder | `qwen3vl_4b_int8_convrot` (text_encoders) | black-forest-labs/flux-3-image-klein-comfy (gated) | FLUX Kommunity License v1.0 |
| Klein VAE | `flux2-vae` (vae) | Comfy-Org/flux2-klein | Apache-2.0 |

FLUX 3 Image Klein is gated: accept its license on Hugging Face and set `HF_TOKEN` before running `setup.sh`. The
FLUX Kommunity License allows non-commercial use, and commercial use only for qualifying users under its revenue cap;
read it on the model page before using outputs commercially.

Tools: comfy-sdk and comfy-api-proxy (MIT), ComfyUI (GPL-3.0, called over HTTP, not vendored), ffmpeg, PyAV, numpy, Pillow.

## Not used, and why

- MiniMax H3 and its derivatives (FastH3, swap LoRAs, Fun ControlNet, turbo LoRAs): the H3 Community License excludes
  several territories, including the US, and restricts display of outputs there.
- Partner/API nodes and closed models: the stack is open weights only.
- Custom node packs and third-party LoRAs without a license file.
