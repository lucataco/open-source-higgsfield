"""API-format graphs for Body Double. Core ComfyUI nodes only.

Input files are paths relative to ComfyUI/input/, or annotated '... [output]' paths for files a previous job saved.
"""
import json

SAM = "sam3.1_multiplex_fp16.safetensors"

SCAIL = {
    "dit": "wan2.1_14B_SCAIL_2_int8_convrot.safetensors",
    "dpo": "wan2.1_SCAIL_2_DPO_lora_bf16.safetensors",
    "dpo_strength": 1.0,
    "lightx2v": "Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors",
    "lightx2v_strength": 0.8,
    "relight": None,  # "wan2.1_SCAIL_2_relight_lora_bf16.safetensors"
    "relight_strength": 1.0,
    "shift": 5.0,
    "steps": 6,
    "attention": "comfy kitchen attention",  # INT8; ~1.45x per step at 1280x704 on GB10
    "sparse_tau": 1.3,  # training-free SOL block-sparse attention after the first 20%; 0 = dense. With the above, ~1.9x per segment
    "umt5": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    "clip_vision": "clip_vision_h.safetensors",
    "vae": "wan_2.1_vae.safetensors",
}

KLEIN = {
    "unet": "flux-2-klein-4b-fp8.safetensors",
    "te": "qwen_3_4b.safetensors",
    "vae": "flux2-vae.safetensors",
    "steps": 4,
}

ANCHOR_PROMPT = ("Replace the person in image 1 with the character from image 2, including their face, hair, body and "
                 "complete outfit exactly as shown in image 2. Do not keep any of the original person's clothing. "
                 "Keep image 1's pose, framing, background and lighting.")


class Graph(dict):
    def add(self, cls, **inputs):
        nid = str(len(self) + 1)
        self[nid] = {"class_type": cls, "inputs": inputs}
        return nid


def _batch(g, cls, key, nodes):
    if len(nodes) == 1:
        return nodes[0]
    return [g.add(cls, **{f"{key}s.{key}{i}": n for i, n in enumerate(nodes)}), 0]


def track(video, init_mask, prefix):
    """Track one person through a whole (low-res) video from a first-frame mask; save one mask PNG per frame."""
    g = Graph()
    comp = g.add("GetVideoComponents", video=[g.add("LoadVideo", file=video), 0])
    sam = g.add("CheckpointLoaderSimple", ckpt_name=SAM)
    init = g.add("LoadImageMask", image=init_mask, channel="red")
    tr = g.add("SAM3_VideoTrack", images=[comp, 0], model=[sam, 0], initial_mask=[init, 0],
               detection_threshold=0.5, max_objects=1, detect_interval=1)
    masks = g.add("SAM3_TrackToMask", track_data=[tr, 0], object_indices="0")
    g.add("SaveImage", images=[g.add("MaskToImage", mask=[masks, 0]), 0], filename_prefix=prefix)
    return g


def detect(image, prefix, text="person", points=None):
    """Per-object masks for one image, from text or from positive points [{"x":..,"y":..}]."""
    g = Graph()
    img = g.add("LoadImage", image=image)
    sam = g.add("CheckpointLoaderSimple", ckpt_name=SAM)
    kw = {}
    if points:
        kw["positive_coords"] = json.dumps(points)
    else:
        kw["conditioning"] = [g.add("CLIPTextEncode", clip=[sam, 1], text=text), 0]
    det = g.add("SAM3_Detect", model=[sam, 0], image=[img, 0], threshold=0.5, refine_iterations=2, individual_masks=True, **kw)
    g.add("SaveImage", images=[g.add("MaskToImage", mask=[det, 0]), 0], filename_prefix=prefix)
    return g


def detect_who(image, prefix, who, text="person"):
    """Per-person masks plus masks for a description of who to pick ("white shirt"); the app matches them up."""
    g = Graph()
    img = [g.add("LoadImage", image=image), 0]
    sam = g.add("CheckpointLoaderSimple", ckpt_name=SAM)
    for name, t in (("p", text), ("w", who)):
        det = g.add("SAM3_Detect", model=[sam, 0], image=img, threshold=0.4, refine_iterations=2, individual_masks=True,
                    conditioning=[g.add("CLIPTextEncode", clip=[sam, 1], text=t), 0])
        g.add("SaveImage", images=[g.add("MaskToImage", mask=[det, 0]), 0], filename_prefix=f"{prefix}/{name}")
    return g


def props(images, names, prefix, threshold=0.7):
    """One SAM 3.1 mask (all instances, empty if none) of each prop in `names` per image, saved as prefix/<image>_<name>."""
    g = Graph()
    sam = g.add("CheckpointLoaderSimple", ckpt_name=SAM)
    texts = [[g.add("CLIPTextEncode", clip=[sam, 1], text=n), 0] for n in names]
    for i, image in enumerate(images):
        img = [g.add("LoadImage", image=image), 0]
        for j, text in enumerate(texts):
            det = g.add("SAM3_Detect", model=[sam, 0], image=img, threshold=threshold, refine_iterations=2, individual_masks=False,
                        conditioning=text)
            g.add("SaveImage", images=[g.add("MaskToImage", mask=[det, 0]), 0], filename_prefix=f"{prefix}/{i}_{j}")
    return g


def anchor(crop, cast, width, height, seed, prefix, prompt=ANCHOR_PROMPT, batch=4):
    """FLUX.2 Klein 4B two-image edit: the person in `crop` becomes the character in `cast`. Sizes are multiples of 16."""
    return edit([crop, cast], prompt, width, height, seed, prefix, batch)


def edit(images, prompt, width, height, seed, prefix, batch=1):
    """FLUX.2 Klein 4B edit; `prompt` calls the inputs image 1, image 2, ... Sizes are multiples of 16."""
    g = Graph()
    unet = g.add("UNETLoader", unet_name=KLEIN["unet"], weight_dtype="default")
    clip = g.add("CLIPLoader", clip_name=KLEIN["te"], type="flux2", device="default")
    vae = g.add("VAELoader", vae_name=KLEIN["vae"])
    text = g.add("CLIPTextEncode", clip=[clip, 0], text=prompt)
    pos, neg = [text, 0], [g.add("ConditioningZeroOut", conditioning=[text, 0]), 0]
    for f in images:
        lat = [g.add("VAEEncode", pixels=[g.add("LoadImage", image=f), 0], vae=[vae, 0]), 0]
        pos = [g.add("ReferenceLatent", conditioning=pos, latent=lat), 0]
        neg = [g.add("ReferenceLatent", conditioning=neg, latent=lat), 0]
    guider = g.add("CFGGuider", model=[unet, 0], positive=pos, negative=neg, cfg=1.0)
    sample = g.add("SamplerCustomAdvanced", noise=[g.add("RandomNoise", noise_seed=seed), 0], guider=[guider, 0],
                   sampler=[g.add("KSamplerSelect", sampler_name="euler"), 0],
                   sigmas=[g.add("Flux2Scheduler", steps=KLEIN["steps"], width=width, height=height), 0],
                   latent_image=[g.add("EmptyFlux2LatentImage", width=width, height=height, batch_size=batch), 0])
    g.add("SaveImage", images=[g.add("VAEDecode", samples=[sample, 0], vae=[vae, 0]), 0], filename_prefix=prefix)
    return g


def segment(chunk, length, width, height, refs, ref_masks, clip_vision_ref, init_mask, prev_tail, prompt, seed, prefix,
            save_handoff, save_ref_mask=False, cfg=SCAIL):
    """One SCAIL-2 replacement segment. Saves final frames, per-frame source and cast masks, and the next handoff mask.

    prev_tail: the previous segment's last 5 saved frames (annotated paths) or None for segment 0.
    """
    g = Graph()
    model = [g.add("UNETLoader", unet_name=cfg["dit"], weight_dtype="default"), 0]
    for lora, strength in ((cfg["dpo"], cfg["dpo_strength"]), (cfg["relight"], cfg["relight_strength"]),
                           (cfg["lightx2v"], cfg["lightx2v_strength"])):
        if lora:
            model = [g.add("LoraLoaderModelOnly", model=model, lora_name=lora, strength_model=strength), 0]
    model = [g.add("ModelSamplingSD3", model=model, shift=cfg["shift"]), 0]
    model = [g.add("ModelAttentionBackend", model=model, attention=cfg["attention"]), 0]
    if cfg["sparse_tau"]:
        model = [g.add("BlockSparseAttention", model=model, selection="sol-attn", **{"selection.tau": cfg["sparse_tau"]},
                       start_percent=0.2, end_percent=1.0, dense_blocks="", min_tokens=12288, extra_tokens=256,
                       sink_conditioning="exact_kv_and_rows", verbose=False), 0]
    clip = g.add("CLIPLoader", clip_name=cfg["umt5"], type="wan", device="default")
    pos = g.add("CLIPTextEncode", clip=[clip, 0], text=prompt)
    neg = g.add("CLIPTextEncode", clip=[clip, 0], text="")
    vae = g.add("VAELoader", vae_name=cfg["vae"])
    cv = g.add("CLIPVisionEncode", clip_vision=[g.add("CLIPVisionLoader", clip_name=cfg["clip_vision"]), 0],
               image=[g.add("LoadImage", image=clip_vision_ref), 0], crop="none")

    frames = [g.add("GetVideoComponents", video=[g.add("LoadVideo", file=chunk), 0]), 0]
    ref_img = _batch(g, "BatchImagesNode", "image", [[g.add("LoadImage", image=f), 0] for f in refs])
    ref_mask = _batch(g, "BatchMasksNode", "mask", [[g.add("LoadImageMask", image=f, channel="red"), 0] for f in ref_masks])

    sam = g.add("CheckpointLoaderSimple", ckpt_name=SAM)
    init = [g.add("LoadImageMask", image=init_mask, channel="red"), 0]
    tr = g.add("SAM3_VideoTrack", images=frames, model=[sam, 0], initial_mask=init, detection_threshold=0.5,
               max_objects=1, detect_interval=1)
    colored = g.add("SCAIL2ColoredMask", driving_track_data=[tr, 0], object_indices="", sort_by="none",
                    replacement_mode=True, ref_track_data=ref_mask)
    extra = {}
    if prev_tail:
        extra["previous_frames"] = _batch(g, "BatchImagesNode", "image", [[g.add("LoadImage", image=f), 0] for f in prev_tail])
    scail = g.add("WanSCAILToVideo", positive=[pos, 0], negative=[neg, 0], vae=[vae, 0], width=width, height=height,
                  length=length, batch_size=1, pose_strength=1.0, pose_start=0.0, pose_end=1.0, video_frame_offset=0,
                  previous_frame_count=5, pose_video=frames, pose_video_mask=[colored, 0], replacement_mode=True,
                  reference_image=ref_img, reference_image_mask=[colored, 1], clip_vision_output=[cv, 0], **extra)
    sample = g.add("SamplerCustom", model=model, add_noise=True, noise_seed=seed, cfg=1.0, positive=[scail, 0],
                   negative=[scail, 1], sampler=[g.add("KSamplerSelect", sampler_name="euler"), 0],
                   sigmas=[g.add("BasicScheduler", model=model, scheduler="simple", steps=cfg["steps"], denoise=1.0), 0],
                   latent_image=[scail, 2])
    decoded = [g.add("VAEDecode", samples=[sample, 0], vae=[vae, 0]), 0]

    # The cast is tracked on the output from the source's first-frame mask, for the composite back into the plate.
    cast_tr = g.add("SAM3_VideoTrack", images=decoded, model=[sam, 0], initial_mask=init, detection_threshold=0.5,
                    max_objects=1, detect_interval=1)
    src_masks = [g.add("MaskToImage", mask=[g.add("SAM3_TrackToMask", track_data=[tr, 0], object_indices="0"), 0]), 0]
    cast_masks = [g.add("MaskToImage", mask=[g.add("SAM3_TrackToMask", track_data=[cast_tr, 0], object_indices="0"), 0]), 0]
    final = decoded
    if prev_tail:
        # Decoded frame 4 re-renders the previous tail's last frame, so match on it and keep relative lighting.
        last = [g.add("LoadImage", image=prev_tail[-1]), 0]
        matched = [g.add("ColorTransfer", image_target=decoded, image_ref=last, method="reinhard_lab",
                         source_stats="target_frame", **{"source_stats.target_index": 4}, strength=1.0), 0]
        final = [g.add("ImageFromBatch", image=matched, batch_index=5, length=4096), 0]
        src_masks = [g.add("ImageFromBatch", image=src_masks, batch_index=5, length=4096), 0]
        cast_masks = [g.add("ImageFromBatch", image=cast_masks, batch_index=5, length=4096), 0]
    g.add("SaveImage", images=final, filename_prefix=f"{prefix}/f")
    g.add("SaveImage", images=src_masks, filename_prefix=f"{prefix}/src_mask")
    g.add("SaveImage", images=cast_masks, filename_prefix=f"{prefix}/cast_mask")
    if save_handoff:
        handoff = g.add("ImageFromBatch", image=[g.add("MaskToImage", mask=[g.add("SAM3_TrackToMask", track_data=[tr, 0], object_indices="0"), 0]), 0],
                        batch_index=76, length=1)
        g.add("SaveImage", images=[handoff, 0], filename_prefix=f"{prefix}/handoff")
    if save_ref_mask:
        g.add("SaveImage", images=[colored, 1], filename_prefix=f"{prefix}/ref_mask")
    return g
