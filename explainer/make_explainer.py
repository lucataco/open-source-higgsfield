"""Render the Body Double explainer video from the storm_noir run's real intermediates.

Video tooling only, not part of the pipeline. It reads a finished runs/storm_noir and runs/storm_purz, their files in
$COMFYUI_DIR/input and output, and the cast images in $BODYDOUBLE_CASTS; none of these are in the repository.

  .venv/bin/python explainer/make_vo.py                   # narration clips (Replicate), once per script edit
  .venv/bin/python explainer/make_music.py 2              # background music (Replicate), once
  .venv/bin/python explainer/make_explainer.py            # full render
  .venv/bin/python explainer/make_explainer.py --stills   # one still per scene, for review
"""
import itertools
import json
import math
import os
import subprocess
import sys
import wave
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS = 1920, 1080, 25
OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
RUN = ROOT / "runs/storm_noir"
COMFYUI_DIR = Path(os.environ.get("COMFYUI_DIR", Path.home() / "ComfyUI"))
CASTS = Path(os.environ["BODYDOUBLE_CASTS"])
REFS = COMFYUI_DIR / "input/bodydouble/storm_noir/1280x704"
PURZ_IN = COMFYUI_DIR / "input/bodydouble/storm_purz/1280x704"

S = json.loads((RUN / "state.json").read_text())
RECTS = S["crops"]["1280x704"]["rects"]
SEGS = S["takes"]["t1"]["segments"]
RAW = [f for s in SEGS for f in s["frames"]]
TRACK = S["track_masks"]
GRAPH = json.loads((RUN / "graphs/t1_seg00_0.api.json").read_text())

# narration: one clip per caption; scene length and cue times follow the voice
VO = json.loads((OUT / "vo/vo.json").read_text())["scenes"]
LEAD, GAP, TAIL = 0.4, 0.3, 0.45
STARTS = {k: list(itertools.accumulate([LEAD] + [c["dur"] + GAP for c in v[:-1]])) for k, v in VO.items()}

BG = (13, 14, 18)
PANEL = (27, 29, 36)
LINE = (60, 64, 78)
TEXT = (238, 240, 244)
MUTED = (150, 157, 172)
ORANGE = (255, 122, 26)
TEAL = (64, 210, 196)
GREEN = (92, 204, 128)

UB = "/usr/share/fonts/truetype/ubuntu/UbuntuSans[wdth,wght].ttf"
DJ = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
DJB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


# ---------------------------------------------------------------- frames

def src(i): return Image.open(RUN / f"src/frames/{i:05d}.png").convert("RGB")
def plate(i): return Image.open(RUN / f"t1/plate/{i:05d}.png").convert("RGB")
def src_crop(i): return Image.open(ROOT / f"runs/storm_purz/crop_1280x704/frames/{i:05d}.png").convert("RGB")
def out_crop(i): return Image.open(RUN / f"t1/crop/{i:05d}.png").convert("RGB")
def matte(i): return Image.open(RUN / f"t1/cast_mask/{i:05d}.png").convert("L")
def raw(i): return Image.open(RAW[i]).convert("RGB")
def track(i): return Image.open(TRACK[i]).convert("L")


# ---------------------------------------------------------------- drawing helpers

def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def ease(x): x = clamp(x); return x * x * (3 - 2 * x)
def ramp(t, t0, dur=0.5): return ease((t - t0) / dur)
def lerp(a, b, k): return a + (b - a) * k


@lru_cache(None)
def font(size, weight="Regular"):
    f = ImageFont.truetype(UB, size)
    f.set_variation_by_name(weight)
    return f


@lru_cache(None)
def fallback(size, weight):
    return ImageFont.truetype(DJB if weight in ("SemiBold", "Bold", "ExtraBold") else DJ, size)


@lru_cache(None)
def mono(size): return ImageFont.truetype(MONO, size)


@lru_cache(None)
def has(ch):
    f = font(40)
    miss = f.getmask("")
    m = f.getmask(ch)
    return ch == " " or m.size != miss.size or m.getbbox() != miss.getbbox()


def runs(s, size, weight):
    out = []
    for ch in s:
        f = font(size, weight) if has(ch) else fallback(size, weight)
        if out and out[-1][0] is f:
            out[-1][1] += ch
        else:
            out.append([f, ch])
    return out


def tw(s, size, weight="Regular", fixed=False):
    return mono(size).getlength(s) if fixed else sum(f.getlength(t) for f, t in runs(s, size, weight))


def fade(layer, a):
    if a < 1:
        layer.putalpha(layer.getchannel("A").point(lambda v: int(v * a)))
    return layer


def text(c, xy, s, size, weight="Regular", fill=TEXT, alpha=1.0, anchor="l", fixed=False):
    """Draw s with its top at xy[1]; anchor l/m/r is horizontal. Falls back to DejaVu per glyph."""
    if alpha <= 0.01 or not s:
        return
    w = tw(s, size, weight, fixed)
    x, y = xy
    x -= {"l": 0, "m": w / 2, "r": w}[anchor]
    pad = size // 2
    layer = Image.new("RGBA", (int(w) + 2 * pad, int(size * 1.5) + pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    base = pad // 2 + font(size, weight).getmetrics()[0]
    cx = pad
    for f, t in ([[mono(size), s]] if fixed else runs(s, size, weight)):
        d.text((cx, base), t, font=f, fill=fill + (255,), anchor="ls")
        cx += f.getlength(t)
    x0, y0 = int(x) - pad, int(y) - pad // 2
    c.alpha_composite(fade(layer, alpha), (max(0, x0), max(0, y0)), (max(0, -x0), max(0, -y0)))


def wrap(s, size, maxw, weight="Regular"):
    lines, cur = [], ""
    for word in s.split(" "):
        t = (cur + " " + word).strip()
        if tw(t, size, weight) <= maxw or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = word
    return lines + [cur]


def rgba(col, a=255): return col if len(col) == 4 else col + (a,)


def box(c, xy, fill=None, r=12, outline=None, width=2, alpha=1.0):
    x0, y0, x1, y1 = (int(round(v)) for v in xy)
    if alpha <= 0.01 or x1 <= x0 or y1 <= y0:
        return
    layer = Image.new("RGBA", (x1 - x0, y1 - y0), (0, 0, 0, 0))
    ImageDraw.Draw(layer).rounded_rectangle((0, 0, x1 - x0 - 1, y1 - y0 - 1), r, fill=fill and rgba(fill),
                                            outline=outline and rgba(outline), width=width)
    c.alpha_composite(fade(layer, alpha), (x0, y0))


def cover(im, w, h):
    iw, ih = im.size
    s = max(w / iw, h / ih)
    cw, ch = w / s, h / s
    x0, y0 = (iw - cw) / 2, (ih - ch) / 2
    return im.resize((w, h), Image.BILINEAR, box=(x0, y0, x0 + cw, y0 + ch))


def contain(im, w, h, bg=PANEL):
    iw, ih = im.size
    s = min(w / iw, h / ih)
    out = Image.new("RGB", (w, h), bg)
    sm = im.convert("RGB").resize((max(1, round(iw * s)), max(1, round(ih * s))), Image.LANCZOS)
    out.paste(sm, ((w - sm.width) // 2, (h - sm.height) // 2))
    return out


@lru_cache(None)
def round_mask(w, h, r):
    m = Image.new("L", (w * 2, h * 2), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, w * 2 - 1, h * 2 - 1), r * 2, fill=255)
    return m.resize((w, h), Image.LANCZOS)


def panel(c, im, xy, size, r=14, alpha=1.0, fit=cover, border=None, bw=3):
    if alpha <= 0.01:
        return
    w, h = size
    im = (im if im.size == (w, h) else fit(im, w, h)).convert("RGBA")
    m = round_mask(w, h, r)
    im.putalpha(m if alpha >= 1 else m.point(lambda v: int(v * alpha)))
    c.alpha_composite(im, (int(xy[0]), int(xy[1])))
    if border:
        box(c, (xy[0] - bw, xy[1] - bw, xy[0] + w + bw, xy[1] + h + bw), None, r + bw, border, bw, alpha)


def tag(c, xy, s, size=20, fg=TEXT, bg=(0, 0, 0, 175), anchor="l", alpha=1.0, weight="SemiBold"):
    w = tw(s, size, weight)
    x, y = xy
    if anchor == "r":
        x -= w + 24
    elif anchor == "m":
        x -= (w + 24) / 2
    box(c, (x, y, x + w + 24, y + size + 16), bg, 8, alpha=alpha)
    text(c, (x + 12, y + 7), s, size, weight, fg, alpha)


def ai_tag(c, x1, y1, alpha=1.0):
    tag(c, (x1 - 10, y1 - 40), "AI-generated", 17, bg=(0, 0, 0, 160), anchor="r", alpha=alpha)


def arrow(c, p0, p1, col=MUTED, width=4, head=14, alpha=1.0):
    if alpha <= 0.01:
        return
    x0, y0, x1, y1 = min(p0[0], p1[0]) - head, min(p0[1], p1[1]) - head, max(p0[0], p1[0]) + head, max(p0[1], p1[1]) + head
    layer = Image.new("RGBA", (int(x1 - x0) + 1, int(y1 - y0) + 1), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    a, b = (p0[0] - x0, p0[1] - y0), (p1[0] - x0, p1[1] - y0)
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    tip = b
    back = (b[0] - head * 1.2 * math.cos(ang), b[1] - head * 1.2 * math.sin(ang))
    d.line([a, back], fill=rgba(col), width=width)
    d.polygon([tip, (back[0] + head * 0.7 * math.sin(ang), back[1] - head * 0.7 * math.cos(ang)),
               (back[0] - head * 0.7 * math.sin(ang), back[1] + head * 0.7 * math.cos(ang))], fill=rgba(col))
    c.alpha_composite(fade(layer, alpha), (int(x0), int(y0)))


def gradient(h, top, bottom):
    a = np.linspace(top, bottom, h)[:, None].repeat(W, 1)
    g = np.zeros((h, W, 4), np.uint8)
    g[..., 3] = a.astype(np.uint8)
    return Image.fromarray(g, "RGBA")


TOP_SHADE = gradient(260, 225, 0)
BOTTOM_SHADE = gradient(300, 0, 210)


def canvas(): return Image.new("RGBA", (W, H), BG + (255,))


def header(c, step, title, sub, t):
    a = ramp(t, 0.0, 0.5)
    x, y = 64, 50
    if step:
        box(c, (x, y, x + 62, y + 62), ORANGE, 14, alpha=a)
        text(c, (x + 31, y + 6), str(step), 40, "Bold", BG, a, "m")
        x += 84
    text(c, (x, y - 2), title, 46, "Bold", TEXT, a)
    text(c, (x + 2, y + 56), sub, 26, "Regular", MUTED, a)


@lru_cache(None)
def balanced(s, maxw):
    """Wrap s into as few lines as fit maxw, then narrow the measure so the lines come out even."""
    s = s.replace("SAM 3.1", "SAM\u00a03.1")
    n = len(wrap(s, 31, maxw))
    while n > 1 and maxw > 400 and len(wrap(s, 31, maxw - 20)) == n:
        maxw -= 20
    return wrap(s, 31, maxw)


def caption(c, cues, t, x=64, maxw=1700, bottom=H - 52):
    """cues: [(t0, t1, text)]; each fades in at t0 and out at t1."""
    for t0, t1, s in cues:
        a = min(ramp(t, t0, 0.45), 1 - ramp(t, t1 - 0.35, 0.35))
        if a <= 0.01:
            continue
        lines = balanced(s, maxw)
        h = len(lines) * 42 + 30
        w = max(tw(l, 31) for l in lines) + 56
        y = bottom - h + (1 - a) * 16
        box(c, (x, y, x + w, y + h), (8, 9, 12, 220), 14, alpha=a)
        box(c, (x, y, x + 6, y + h), ORANGE, 3, alpha=a)
        for k, l in enumerate(lines):
            text(c, (x + 30, y + 14 + k * 42), l, 31, "Regular", TEXT, a)


def cues(name, d):
    st = STARTS[name]
    return [(a - 0.15, b, c["cap"]) for a, b, c in zip(st, [x - 0.1 for x in st[1:]] + [d], VO[name])]


def overlay_mask(im, m, col, fill=0.38, edge=0.95, ring=7):
    """Tint mask m (L, any size) onto im with an outline."""
    big = np.asarray(m.resize(im.size, Image.BILINEAR), np.float32) / 255
    dil = np.asarray(m.filter(ImageFilter.MaxFilter(ring)).resize(im.size, Image.BILINEAR), np.float32) / 255
    a = (fill * big + edge * np.clip(dil - big, 0, 1))[..., None]
    arr = np.asarray(im, np.float32)
    return Image.fromarray((arr * (1 - a) + np.array(col, np.float32) * a).astype(np.uint8))


def dim_outside(im, rect, k=0.38):
    arr = np.asarray(im, np.float32) * k
    x0, y0, x1, y1 = (int(round(v)) for v in rect)
    arr[y0:y1, x0:x1] /= k
    return Image.fromarray(arr.astype(np.uint8))


def rect_box(r): return (r[0], r[1], r[0] + r[2], r[1] + r[3])


# ---------------------------------------------------------------- static assets (loaded before the pool forks)

CAST_PORTRAIT = Image.open(CASTS / "noir_orange_portrait.png").convert("RGB")
CAST_FULL = Image.open(CASTS / "noir_orange_full.png").convert("RGB")
ANCH = [Image.open(RUN / f"anchors_1280x704/a{k}.png").convert("RGB") for k in range(4)]
ANCHOR = Image.open(RUN / "anchors_1280x704/a2_clean.png").convert("RGB")
HAND_BOX = (400, 420, 720, 700)
REF_VIEWS = []
for name in ("ref0", "full", "portrait"):
    im = np.asarray(Image.open(REFS / f"{name}.png").convert("RGB"), np.float32)
    m = np.asarray(Image.open(REFS / f"{name}_mask.png").convert("L"), np.float32)[..., None] / 255
    REF_VIEWS.append(Image.fromarray((im * (0.22 + 0.78 * m)).astype(np.uint8)))
FIRST_CROP = src_crop(0)

ZOOM = np.array([1080 / r[3] for r in RECTS])


def thumb_track(i, w, h):
    return overlay_mask(src(i), track(i), TEAL).resize((w, h), Image.LANCZOS)


def thumb_follow(i, w, h):
    im = dim_outside(src(i), rect_box(RECTS[i]))
    d = ImageDraw.Draw(im)
    d.rectangle(rect_box(RECTS[i]), outline=ORANGE, width=8)
    return im.resize((w, h), Image.LANCZOS)


CARD_W, CARD_TH = 270, 152
CARDS = [
    ("Conform", "ffmpeg: full-res frames, audio and a 960-wide tracking proxy", src(0).resize((CARD_W, CARD_TH), Image.LANCZOS)),
    ("Track", "SAM 3.1 finds the performer, tracks them, re-identifies if lost", thumb_track(200, CARD_W, CARD_TH)),
    ("Follow-cam", "A smoothed virtual camera keeps them large at 1280×704", thumb_follow(400, CARD_W, CARD_TH)),
    ("Anchor", "FLUX.2 Klein 4B paints the cast into the first frame", cover(ANCHOR, CARD_W, CARD_TH)),
    ("Segments", "SCAIL-2 14B: one ComfyUI job per 81 frames, QC between", cover(raw(400), CARD_W, CARD_TH)),
    ("Assemble", "Align, matte, paste into the untouched plate, mux audio", plate(400).resize((CARD_W, CARD_TH), Image.LANCZOS)),
]


# ---------------------------------------------------------------- the ComfyUI graph (real segment graph, ComfyUI-style)

TITLES = {
    "UNETLoader": "Load Diffusion Model", "LoraLoaderModelOnly": "Load LoRA", "CLIPLoader": "Load CLIP",
    "CLIPTextEncode": "CLIP Text Encode (Prompt)", "VAELoader": "Load VAE", "CLIPVisionLoader": "Load CLIP Vision",
    "LoadImage": "Load Image", "CLIPVisionEncode": "CLIP Vision Encode", "LoadImageMask": "Load Image (as Mask)",
    "CheckpointLoaderSimple": "Load Checkpoint", "VAEDecode": "VAE Decode", "SaveImage": "Save Image",
    "ModelAttentionBackend": "Model Attention Backend", "BlockSparseAttention": "Model Sparse Attention",
    "LoadVideo": "Load Video", "GetVideoComponents": "Get Video Components", "BatchImagesNode": "Batch Images",
    "BatchMasksNode": "Batch Masks", "SAM3_VideoTrack": "Run SAM3 Video Track", "SCAIL2ColoredMask": "Create SCAIL-2 Colored Mask",
    "SAM3_TrackToMask": "SAM3 Track to Mask", "MaskToImage": "Convert Mask to Image",
}
OUTS = {
    "UNETLoader": [("MODEL", "MODEL")], "LoraLoaderModelOnly": [("MODEL", "MODEL")], "ModelSamplingSD3": [("MODEL", "MODEL")],
    "ModelAttentionBackend": [("MODEL", "MODEL")], "BlockSparseAttention": [("MODEL", "MODEL")], "CLIPLoader": [("CLIP", "CLIP")],
    "CLIPTextEncode": [("CONDITIONING", "CONDITIONING")], "VAELoader": [("VAE", "VAE")], "CLIPVisionLoader": [("CLIP_VISION", "CLIP_VISION")],
    "LoadImage": [("IMAGE", "IMAGE"), ("MASK", "MASK")], "CLIPVisionEncode": [("CLIP_VISION_OUTPUT", "CLIP_VISION_OUTPUT")],
    "LoadVideo": [("VIDEO", "VIDEO")], "GetVideoComponents": [("images", "IMAGE"), ("audio", "AUDIO"), ("fps", "FLOAT"), ("bit_depth", "COMBO"), ("color_space", "COMBO")],
    "BatchImagesNode": [("IMAGE", "IMAGE")], "LoadImageMask": [("MASK", "MASK")], "BatchMasksNode": [("MASK", "MASK")],
    "CheckpointLoaderSimple": [("MODEL", "MODEL"), ("CLIP", "CLIP"), ("VAE", "VAE")], "SAM3_VideoTrack": [("track_data", "SAM3_TRACK_DATA")],
    "SCAIL2ColoredMask": [("pose_video_mask", "IMAGE"), ("reference_image_mask", "IMAGE")],
    "WanSCAILToVideo": [("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("latent", "LATENT"), ("video_frame_offset", "INT")],
    "KSamplerSelect": [("SAMPLER", "SAMPLER")], "BasicScheduler": [("SIGMAS", "SIGMAS")],
    "SamplerCustom": [("output", "LATENT"), ("denoised_output", "LATENT")], "VAEDecode": [("IMAGE", "IMAGE")],
    "SAM3_TrackToMask": [("masks", "MASK")], "MaskToImage": [("IMAGE", "IMAGE")], "SaveImage": [],
}
LINK = {
    "MODEL": (179, 157, 219), "CLIP": (255, 213, 0), "VAE": (255, 110, 110), "CONDITIONING": (255, 169, 49),
    "IMAGE": (100, 181, 246), "LATENT": (255, 156, 249), "MASK": (129, 199, 132), "CLIP_VISION": (168, 218, 220),
    "CLIP_VISION_OUTPUT": (173, 116, 82), "SAMPLER": (236, 180, 180), "SIGMAS": (205, 255, 205), "SAM3_TRACK_DATA": (77, 208, 225),
    "VIDEO": (160, 170, 190),
}
SHOW = {
    "BlockSparseAttention": ["selection", "selection.tau", "start_percent", "end_percent"],
    "WanSCAILToVideo": ["width", "height", "length", "previous_frame_count", "replacement_mode"],
    "SAM3_VideoTrack": ["max_objects", "detect_interval"], "SamplerCustom": ["noise_seed", "cfg"],
    "SCAIL2ColoredMask": ["replacement_mode"], "CLIPLoader": ["clip_name", "type"], "UNETLoader": ["unet_name"],
    "LoadImageMask": ["image"], "BasicScheduler": ["scheduler", "steps"], "CLIPVisionEncode": ["crop"],
}
SEG0 = COMFYUI_DIR / "output/bodydouble/storm_noir/t1/seg00_0"
PREVIEW = {
    "12": REFS / "portrait_cv.png", "16": REFS / "ref0.png", "17": REFS / "full.png", "18": REFS / "portrait.png",
    "20": REFS / "ref0_mask.png", "21": REFS / "full_mask.png", "22": REFS / "portrait_mask.png", "25": PURZ_IN / "init_00.png",
    "14": ROOT / "runs/storm_purz/crop_1280x704/frames/00000.png", "38": Path(SEGS[0]["frames"][40]),
    "39": Path(SEGS[0]["src_masks"][40]), "40": Path(SEGS[0]["cast_masks"][40]), "41": sorted(SEG0.glob("ref_mask_*.png"))[0],
}
NODE_W, TITLE_H, ROW, WROW = 270, 32, 22, 24
COLS = [  # (x, y, node ids stacked top to bottom)
    *[(60 + k * 310, 90, [str(n)]) for k, n in enumerate(range(1, 7))],
    (60, 600, ["7", "11", "12"]), (370, 600, ["8", "9", "13"]),
    (720, 600, ["16", "17", "18"]), (1020, 600, ["20", "21", "22"]), (1320, 600, ["19", "23"]),
    (1720, 600, ["14", "15", "24", "25"]), (2030, 600, ["26", "27"]),
    (2440, 90, ["10", "29", "30", "28"]), (2750, 90, ["31", "32", "38"]), (3060, 90, ["33", "36", "37", "40"]),
    (2750, 960, ["34", "35"]), (3060, 960, ["39", "41"]),
]
GROUPS = {  # name: (title, color, node ids)
    "model": ("Model · SCAIL-2 14B int8 + LoRAs + attention speed-ups", (110, 90, 170), [str(n) for n in range(1, 7)]),
    "prompt": ("Prompt + CLIP-Vision", (150, 120, 40), ["7", "8", "9", "11", "12", "13"]),
    "refs": ("Cast references + masks", (50, 110, 160), ["16", "17", "18", "19", "20", "21", "22", "23"]),
    "drive": ("Driving chunk · SAM 3.1 track", (60, 130, 80), ["14", "15", "24", "25", "26", "27"]),
    "gen": ("Generate · decode · track the new cast", (170, 90, 40), ["10", "28", "29", "30", "31", "32", "38", "33", "36", "37", "40"]),
    "debug": ("Masks saved for assemble", (90, 90, 100), ["34", "35", "39", "41"]),
}


def node_spec(nid):
    n = GRAPH[nid]
    cls = n["class_type"]
    links = [(k, v) for k, v in n["inputs"].items() if isinstance(v, list)]
    wid = [(k, v) for k, v in n["inputs"].items() if not isinstance(v, list)]
    if cls in SHOW:
        wid = [(k, v) for k, v in wid if k in SHOW[cls]]
    outs = OUTS[cls]
    prev = PREVIEW.get(nid)
    th = round((NODE_W - 20) * 704 / 1280) if prev else 0
    h = TITLE_H + 8 + max(len(links), len(outs)) * ROW + len(wid) * WROW + (th + 10 if prev else 0) + 8
    return dict(cls=cls, title=TITLES.get(cls, cls), links=links, wid=wid, outs=outs, prev=prev, th=th, h=h)


def short(v):
    v = str(v)
    if "/" in v:
        v = v.rsplit("/", 1)[1]
    return {"True": "true", "False": "false"}.get(v, v)


def build_graph(sc=2):
    specs = {nid: node_spec(nid) for nid in GRAPH}
    pos = {}
    M = 320  # margin, so zoomed views near the edges stay on the canvas
    for x, y, ids in COLS:
        y += M
        for nid in ids:
            pos[nid] = (x + M, y)
            y += specs[nid]["h"] + 34
    gw = max(x for x, _ in pos.values()) + NODE_W + 70 + M
    gh = max(pos[n][1] + specs[n]["h"] for n in pos) + 60 + M
    img = Image.new("RGBA", (gw * sc, gh * sc), (24, 24, 26, 255))
    d = ImageDraw.Draw(img)
    for gx in range(0, gw, 40):
        for gy in range(0, gh, 40):
            d.ellipse(((gx - 1.2) * sc, (gy - 1.2) * sc, (gx + 1.2) * sc, (gy + 1.2) * sc), fill=(44, 44, 48, 255))
    rects = {}
    for key, (title, col, ids) in GROUPS.items():
        x0 = min(pos[n][0] for n in ids) - 26
        y0 = min(pos[n][1] for n in ids) - 70
        x1 = max(pos[n][0] for n in ids) + NODE_W + 26
        y1 = max(pos[n][1] + specs[n]["h"] for n in ids) + 26
        rects[key] = (x0, y0, x1, y1)
        box(img, (x0 * sc, y0 * sc, x1 * sc, y1 * sc), col + (70,), 10 * sc, col + (200,), 2 * sc)
        text(img, (x0 * sc + 18 * sc, y0 * sc + 12 * sc), title, 24 * sc, "Bold", (235, 235, 240))

    def slot_y(nid, k): return pos[nid][1] + TITLE_H + 8 + k * ROW + ROW / 2

    for nid, sp in specs.items():  # links under the nodes
        for k, (name, (srcid, slot)) in enumerate(sp["links"]):
            typ = OUTS[GRAPH[srcid]["class_type"]][slot][1]
            p0 = (pos[srcid][0] + NODE_W - 10, slot_y(srcid, slot))
            p1 = (pos[nid][0] + 10, slot_y(nid, k))
            dx = max(abs(p1[0] - p0[0]) * 0.45, 80)
            pts = []
            for j in range(48):
                u = j / 47
                bx = (1 - u) ** 3 * p0[0] + 3 * (1 - u) ** 2 * u * (p0[0] + dx) + 3 * (1 - u) * u * u * (p1[0] - dx) + u ** 3 * p1[0]
                by = (1 - u) ** 3 * p0[1] + 3 * (1 - u) ** 2 * u * p0[1] + 3 * (1 - u) * u * u * p1[1] + u ** 3 * p1[1]
                pts.append((bx * sc, by * sc))
            d.line(pts, fill=LINK.get(typ, (150, 150, 150)) + (235,), width=3 * sc, joint="curve")
    for nid, sp in specs.items():
        x, y = pos[nid]
        X = lambda v: v * sc
        box(img, (X(x), X(y), X(x + NODE_W), X(y + sp["h"])), (53, 53, 56), 8 * sc, (20, 20, 22), sc)
        box(img, (X(x), X(y), X(x + NODE_W), X(y + TITLE_H)), (38, 38, 41), 8 * sc)
        d.ellipse((X(x + 11), X(y + 11), X(x + 21), X(y + 21)), fill=(120, 120, 126, 255))
        text(img, (X(x + 30), X(y + 7)), sp["title"], 15 * sc, "SemiBold", (222, 222, 226))
        text(img, (X(x + NODE_W - 8), X(y + 9)), f"#{nid}", 11 * sc, "Regular", (120, 120, 128), anchor="r")
        for k, (name, (srcid, slot)) in enumerate(sp["links"]):
            col = LINK.get(OUTS[GRAPH[srcid]["class_type"]][slot][1], (150, 150, 150))
            cy = slot_y(nid, k)
            d.ellipse((X(x + 5), X(cy - 5), X(x + 15), X(cy + 5)), fill=col + (255,))
            text(img, (X(x + 22), X(cy - 9)), name, 13 * sc, "Regular", (170, 170, 176))
        for k, (name, typ) in enumerate(sp["outs"]):
            col = LINK.get(typ, (150, 150, 150))
            cy = slot_y(nid, k)
            d.ellipse((X(x + NODE_W - 15), X(cy - 5), X(x + NODE_W - 5), X(cy + 5)), fill=col + (255,))
            text(img, (X(x + NODE_W - 22), X(cy - 9)), name, 13 * sc, "Regular", (170, 170, 176), anchor="r")
        wy = y + TITLE_H + 8 + max(len(sp["links"]), len(sp["outs"])) * ROW
        for name, val in sp["wid"]:
            box(img, (X(x + 10), X(wy + 2), X(x + NODE_W - 10), X(wy + WROW - 2)), (34, 34, 36), 9 * sc)
            text(img, (X(x + 20), X(wy + 4)), name, 12 * sc, "Regular", (140, 140, 148))
            v = short(val)
            room = NODE_W - 50 - tw(name, 12)
            while tw(v, 12) > room and len(v) > 2:
                v = v[:-2] + "…" if not v.endswith("…") else v[:-2] + "…"
            text(img, (X(x + NODE_W - 20), X(wy + 4)), v, 12 * sc, "Regular", (225, 225, 230), anchor="r")
            wy += WROW
        if sp["prev"]:
            im = contain(Image.open(sp["prev"]), (NODE_W - 20) * sc, sp["th"] * sc, (30, 30, 32))
            img.paste(im, (X(x + 10), X(wy + 6)))
    return img.convert("RGB"), rects, (M - 30, M - 50, gw - M + 30, gh - M + 30)


GRAPH_IMG, GROUP_RECTS, GRAPH_BOX = build_graph()
GSC = 2


# ---------------------------------------------------------------- scenes; each takes local time t and duration d

def s_title(t, d):
    i = min(int(t * FPS), 749)
    bg = plate(i).resize((384, 216), Image.BILINEAR).filter(ImageFilter.GaussianBlur(5)).resize((W, H), Image.BILINEAR)
    c = Image.fromarray((np.asarray(bg, np.float32) * 0.32).astype(np.uint8)).convert("RGBA")
    a = ramp(t, 0.3, 0.8)
    text(c, (W / 2, 300 + (1 - a) * 20), "Body Double", 140, "Bold", TEXT, a, "m")
    box(c, (W / 2 - 90 * ramp(t, 0.7, 0.6), 492, W / 2 + 90 * ramp(t, 0.7, 0.6), 498), ORANGE, 3)
    text(c, (W / 2, 530), "Same take. New cast. Open weights only.", 50, "Regular", TEXT, ramp(t, 0.9, 0.7), "m")
    text(c, (W / 2, 660), "How the workflow is set up", 36, "SemiBold", ORANGE, ramp(t, 1.6, 0.7), "m")
    text(c, (W / 2, 720), "SAM 3.1  ·  FLUX.2 Klein 4B  ·  SCAIL-2  —  stock ComfyUI nodes, run locally on one DGX Spark", 28,
         "Regular", MUTED, ramp(t, 2.0, 0.7), "m")
    ai_tag(c, W - 30, H - 24)
    return c


def s_result(t, d):
    i = min(int(t * FPS), 749)
    x = int(W / 2 + W * 0.30 * math.sin(2 * math.pi * t / 6.5))
    comp = plate(i)
    comp.paste(src(i).crop((0, 0, x, H)), (0, 0))
    c = comp.convert("RGBA")
    c.alpha_composite(TOP_SHADE)
    box(c, (x - 2, 0, x + 2, H), (255, 255, 255), 0)
    box(c, (x - 26, H / 2 - 26, x + 26, H / 2 + 26), (255, 255, 255), 26)
    arrow(c, (x - 4, H / 2), (x - 16, H / 2), BG, 3, 8)
    arrow(c, (x + 4, H / 2), (x + 16, H / 2), BG, 3, 8)
    tag(c, (40, 40), "SOURCE", 24)
    tag(c, (W - 40, 40), "RECAST  ·  AI-generated", 24, BG, ORANGE + (235,), "r")
    c.alpha_composite(BOTTOM_SHADE, (0, H - 300))
    caption(c, cues("s_result", d), t)
    return c


def s_overview(t, d):
    c = canvas()
    header(c, None, "The pipeline", "Six stages, one command. The model stages are stock ComfyUI graphs that the runner queues for you.", t)
    x0, y0, gap = 64, 290, 36
    for k, (name, desc, th) in enumerate(CARDS):
        a = ramp(t, 0.5 + 0.3 * k, 0.5)
        x = x0 + k * (CARD_W + gap)
        y = y0 + (1 - a) * 30
        box(c, (x, y, x + CARD_W, y + 340), PANEL, 16, LINE, 2, a)
        panel(c, th, (x, y), (CARD_W, CARD_TH), 16, a)
        box(c, (x + 18, y + 172, x + 54, y + 208), ORANGE, 8, alpha=a)
        text(c, (x + 36, y + 175), str(k + 1), 24, "Bold", BG, a, "m")
        text(c, (x + 66, y + 172), name, 30, "Bold", TEXT, a)
        for j, l in enumerate(wrap(desc, 22, CARD_W - 36)):
            text(c, (x + 18, y + 226 + j * 31), l, 22, "Regular", MUTED, a)
        if k < 5:
            arrow(c, (x + CARD_W + 6, y0 + 76), (x + CARD_W + gap - 6, y0 + 76), ORANGE, 4, 10, ramp(t, 0.8 + 0.3 * k, 0.4))
    k2 = STARTS["s_overview"][1]
    a = ramp(t, k2, 0.6)
    chain = ["bodydouble.runner", "comfy-sdk", "comfy-api-proxy  :8189", "ComfyUI  :8188  (stock nodes)", "DGX Spark  GB10"]
    widths = [tw(s, 26, "SemiBold") + 48 for s in chain]
    x = (W - (sum(widths) + 56 * (len(chain) - 1))) / 2
    y = 760
    text(c, (W / 2, y - 52), "How a model stage runs", 24, "SemiBold", MUTED, a, "m")
    for k, (s, w) in enumerate(zip(chain, widths)):
        box(c, (x, y, x + w, y + 60), (34, 37, 46), 12, LINE, 2, a)
        text(c, (x + 24, y + 15), s, 26, "SemiBold", TEXT, a)
        if k < len(chain) - 1:
            arrow(c, (x + w + 8, y + 30), (x + w + 48, y + 30), MUTED, 3, 9, a)
        x += w + 56
    text(c, (W / 2, y + 92), "Everything stays on 127.0.0.1. state.json records each finished stage, so a crashed run resumes where it stopped.",
         24, "Regular", MUTED, ramp(t, k2 + 0.6, 0.6), "m")
    caption(c, cues("s_overview", d), t)
    return c


def s_track(t, d):
    i = 175 + min(int(t * FPS), 574)
    m = track(i)
    c = overlay_mask(src(i), m, TEAL).convert("RGBA")
    ys, xs = np.nonzero(np.asarray(m) > 127)
    if len(xs):
        sx = W / m.width
        bx = (xs.min() * sx - 10, ys.min() * sx - 10, xs.max() * sx + 10, ys.max() * sx + 10)
        bx = (bx[0], bx[1], bx[2], min(bx[3], H - 8))
        box(c, bx, None, 10, TEAL, 3)
        tag(c, (bx[0], max(0, bx[1] - 42)), "performer  ·  SAM 3.1 track", 19, BG, TEAL + (235,))
    c.alpha_composite(TOP_SHADE)
    c.alpha_composite(BOTTOM_SHADE, (0, H - 300))
    header(c, 2, "Detect and track", "SAM3_Detect + SAM3_VideoTrack (SAM 3.1), on a 960-wide proxy", t)
    tag(c, (W - 40, 56), f"frame {i} / 750", 22, anchor="r")
    caption(c, cues("s_track", d), t)
    return c


def s_follow(t, d):
    i = 300 + min(int(t * FPS), 449)
    c = canvas()
    header(c, 3, "Follow-cam", "A smoothed virtual camera, rendered at the model's size", t)
    fx, fy, fw, fh = 64, 196, 1080, 608
    s = fw / W
    r = rect_box(RECTS[i])
    full = dim_outside(src(i), r).resize((fw, fh), Image.BILINEAR)
    panel(c, full, (fx, fy), (fw, fh), 14)
    rr = (fx + r[0] * s, fy + r[1] * s, fx + r[2] * s, fy + r[3] * s)
    px, py, pw, ph = 1196, 196, 660, 363
    d_ = ImageDraw.Draw(c)
    d_.line([(rr[2], rr[1]), (px, py)], fill=(150, 80, 30, 255), width=2)
    d_.line([(rr[2], rr[3]), (px, py + ph)], fill=(150, 80, 30, 255), width=2)
    box(c, rr, None, 6, ORANGE, 4)
    tag(c, (rr[0] + 6, rr[1] + 8), "virtual camera", 18, BG, ORANGE + (235,))
    tag(c, (fx + 14, fy + fh - 50), "source  1920×1080", 18)
    panel(c, src_crop(i), (px, py), (pw, ph), 14, border=ORANGE)
    tag(c, (px + 14, py + 14), "model input  1280×704", 18)
    x, y, w, h = (round(v) for v in RECTS[i])
    text(c, (px, 590), f"window  x {x:4d}  y {y:4d}  {w}×{h}", 24, fill=MUTED, fixed=True)
    text(c, (px, 632), f"{ZOOM[i]:.1f}×", 76, "Bold", ORANGE)
    for j, l in enumerate(["larger performer than rendering", "the whole frame at 1280×704"]):
        text(c, (px + 170, 650 + j * 32), l, 24, "Regular", TEXT)
    sx, sy, sw, sh = px, 760, pw, 120
    box(c, (sx, sy - 8, sx + sw, sy + sh + 8), PANEL, 10)
    zmax = ZOOM.max()
    pts = [(sx + 10 + k / 749 * (sw - 20), sy + sh - 6 - (ZOOM[k] - 1) / (zmax - 1) * (sh - 20)) for k in range(0, 750, 3)]
    d_.line(pts, fill=ORANGE + (255,), width=3)
    mx = sx + 10 + i / 749 * (sw - 20)
    my = sy + sh - 6 - (ZOOM[i] - 1) / (zmax - 1) * (sh - 20)
    d_.ellipse((mx - 7, my - 7, mx + 7, my + 7), fill=TEXT + (255,))
    text(c, (sx + 12, sy - 2), "zoom across the 30 s take", 18, "Regular", MUTED)
    caption(c, cues("s_follow", d), t)
    return c


def s_anchor(t, d):
    c = canvas()
    header(c, 4, "Cast and anchor", "FLUX.2 Klein 4B two-image edit of the first follow-cam frame", t)
    k2, k3 = STARTS["s_anchor"][1:]
    # cast input
    a = ramp(t, 0.3)
    text(c, (64, 186), "Cast input", 26, "SemiBold", TEXT, a)
    panel(c, contain(CAST_PORTRAIT, 214, 266), (64, 226), (214, 266), 12, a)
    panel(c, contain(CAST_FULL, 214, 266), (290, 226), (214, 266), 12, a)
    ai_tag(c, 504, 492, a)
    text(c, (64, 504), "fictional character from a text prompt", 19, "Regular", MUTED, a)
    text(c, (64, 528), "(Nano Banana 2.1, made outside the app)", 19, "Regular", MUTED, a)
    a = ramp(t, 1.3)
    text(c, (64, 586), "First follow-cam frame", 26, "SemiBold", TEXT, a)
    panel(c, FIRST_CROP, (64, 626), (440, 242), 12, a)
    # four options
    a = ramp(t, 2.4)
    arrow(c, (520, 470), (590, 470), ORANGE, 5, 13, a)
    text(c, (608, 186), "FLUX.2 Klein 4B  ·  4 options", 26, "SemiBold", TEXT, a)
    for k in range(4):
        x, y = 608 + (k % 2) * 290, 226 + (k // 2) * 166
        ak = ramp(t, 2.6 + 0.2 * k)
        panel(c, ANCH[k], (x, y), (274, 151), 10, ak, border=ORANGE if k == 2 and t > k2 - 0.6 else None)
    a = ramp(t, k2 - 0.6)
    tag(c, (608 + 14, 392 + 12), "picked", 18, BG, ORANGE + (235,), alpha=a)
    ai_tag(c, 608 + 564, 226 + 317, ramp(t, 3.2))
    a = ramp(t, k2)
    text(c, (608, 586), "Fix pass: remove the source performer's tattoos", 26, "SemiBold", TEXT, a)
    panel(c, ANCH[2].crop(HAND_BOX), (608, 626), (274, 240), 10, a)
    panel(c, ANCHOR.crop(HAND_BOX), (898, 626), (274, 240), 10, ramp(t, k2 + 1.2), border=ORANGE)
    tag(c, (618, 634), "a2", 17, alpha=a)
    tag(c, (908, 634), "a2, second Klein edit", 17, alpha=ramp(t, k2 + 1.2))
    # anchor + references
    a = ramp(t, k3)
    arrow(c, (1188, 470), (1252, 470), ORANGE, 5, 13, a)
    text(c, (1268, 186), "Anchor: frame 1 of the take", 26, "SemiBold", TEXT, a)
    panel(c, ANCHOR, (1268, 226), (588, 323), 12, a, border=ORANGE)
    ai_tag(c, 1856, 549, a)
    a = ramp(t, k3 + 2.0)
    text(c, (1268, 586), "SCAIL-2 reference views (masked)", 26, "SemiBold", TEXT, a)
    for k, (im, name) in enumerate(zip(REF_VIEWS, ["anchor", "full body", "portrait"])):
        x = 1268 + k * 200
        panel(c, im, (x, 626), (188, 104), 8, ramp(t, k3 + 2.0 + 0.2 * k), fit=cover)
        text(c, (x + 94, 738), name, 19, "Regular", MUTED, ramp(t, k3 + 2.0 + 0.2 * k), "m")
    text(c, (1268, 790), "+ CLIP-Vision of the portrait", 21, "Regular", MUTED, ramp(t, k3 + 2.8))
    text(c, (1268, 822), "+ the text prompt", 21, "Regular", MUTED, ramp(t, k3 + 2.8))
    caption(c, cues("s_anchor", d), t)
    return c


GRAPH_GROUPS = [None, "model", "drive", ("prompt", "refs"), ("gen",)]  # framed during each graph cue; None = whole graph
VIEW = (64, 176, 1792, 716)


def group_rect(g):
    if g is None:
        return GRAPH_BOX
    gs = (g,) if isinstance(g, str) else g
    rs = [GROUP_RECTS[k] for k in gs]
    return (min(r[0] for r in rs) - 20, min(r[1] for r in rs) - 20, max(r[2] for r in rs) + 20, max(r[3] for r in rs) + 20)


def fit_view(r):
    x0, y0, x1, y1 = r
    vw, vh = VIEW[2], VIEW[3]
    s = min(vw / (x1 - x0), vh / (y1 - y0))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return cx, cy, s


def s_graph(t, d):
    c = canvas()
    header(c, 5, "One segment = one ComfyUI graph", "runs/storm_noir/graphs/t1_seg00_0.api.json  ·  41 stock nodes, exactly as queued", t)
    keys = [(0.0, None)]
    for s, g0, g1 in zip(STARTS["s_graph"][1:], GRAPH_GROUPS, GRAPH_GROUPS[1:]):
        keys += [(s - 0.5, g0), (s + 0.5, g1)]
    keys += [(d - 1.8, GRAPH_GROUPS[-1]), (d - 0.9, None), (d, None)]
    for (t0, g0), (t1, g1) in zip(keys, keys[1:]):
        if t0 <= t <= t1:
            break
    k = ease((t - t0) / (t1 - t0))
    a0, a1 = fit_view(group_rect(g0)), fit_view(group_rect(g1))
    cx, cy = lerp(a0[0], a1[0], k), lerp(a0[1], a1[1], k)
    s = math.exp(lerp(math.log(a0[2]), math.log(a1[2]), k))
    vx, vy, vw, vh = VIEW
    bw, bh = vw / s, vh / s
    x0, y0, x1, y1 = (cx - bw / 2) * GSC, (cy - bh / 2) * GSC, (cx + bw / 2) * GSC, (cy + bh / 2) * GSC
    k0, l0, k1, l1 = max(0, x0), max(0, y0), min(GRAPH_IMG.width, x1), min(GRAPH_IMG.height, y1)
    view = Image.new("RGB", (vw, vh), (24, 24, 26))
    dx0, dy0 = round((k0 - x0) / (x1 - x0) * vw), round((l0 - y0) / (y1 - y0) * vh)
    dx1, dy1 = round((k1 - x0) / (x1 - x0) * vw), round((l1 - y0) / (y1 - y0) * vh)
    view.paste(GRAPH_IMG.resize((dx1 - dx0, dy1 - dy0), Image.BILINEAR, box=(k0, l0, k1, l1)), (dx0, dy0))
    panel(c, view, (vx, vy), (vw, vh), 16)
    box(c, (vx, vy, vx + vw, vy + vh), None, 16, LINE, 2)
    k1, k2 = STARTS["s_graph"][1:3]
    tag(c, (vx + vw - 20, vy + 20), "optional  --relight  adds the SCAIL-2 relight LoRA after DPO", 21, BG, ORANGE + (235,), "r",
        min(ramp(t, k1 + 2.6), 1 - ramp(t, k2 - 0.5)))
    caption(c, cues("s_graph", d), t)
    return c


CHUNKS = [(76 * k, min(76 * k + 81, 750)) for k in range(10)]


def s_segments(t, d):
    c = canvas()
    header(c, 5, "Segments and QC", "10 ComfyUI jobs  ·  ~8.7 min each on one DGX Spark (15 min before the attention speed-ups)  ·  85 min for 30 s", t)
    f = int(clamp((t - 0.8) / (d - 1.8)) * 749)
    panel(c, src_crop(f), (64, 168), (880, 484), 14)
    panel(c, raw(f), (976, 168), (880, 484), 14)
    tag(c, (78, 182), "driving chunk  (follow-cam crop)", 19)
    tag(c, (990, 182), "SCAIL-2 output", 19, BG, ORANGE + (235,))
    ai_tag(c, 1856, 652)
    tag(c, (1856 - 14, 182), "3× speed", 17, anchor="r")
    X0, sc = 64, 1792 / 750
    d_ = ImageDraw.Draw(c)
    for k, (a, b) in enumerate(CHUNKS):
        y = 690 + (k % 2) * 62
        x0, x1 = X0 + a * sc, X0 + b * sc
        box(c, (x0, y, x1, y + 50), PANEL, 8, LINE, 2)
        done = clamp((f - a) / (b - a))
        if done > 0:
            box(c, (x0, y, x0 + (x1 - x0) * done, y + 50), (120, 60, 20), 8)
        if k:
            box(c, (x0, y, x0 + 5 * sc, y + 50), (255, 255, 255, 70), 3)
        seg = SEGS[k]
        text(c, (x0 + 12, y + 12), f"seg {k}  ·  {seg['secs']} s", 19, "SemiBold", TEXT)
        if f >= b - 1:
            q = seg["qc"]
            label = f"✓ seam {q['seam']:.2f}" if "seam" in q else "✓ first"
            text(c, (x0 + 12, 826 + (k % 2) * 30), label, 18, "SemiBold", GREEN)
    px = X0 + f * sc
    d_.line([(px, 676), (px, 820)], fill=TEXT + (255,), width=3)
    a = ramp(t, 1.2) * (1 - ramp(t, STARTS["s_segments"][1]))
    if a > 0:
        ox = X0 + 76 * sc + 2
        arrow(c, (ox + 70, 668), (ox + 4, 742), TEXT, 3, 9, a)
        text(c, (ox + 76, 640), "5-frame overlap: the last frames of one job seed the next", 19, "Regular", TEXT, a)
    caption(c, cues("s_segments", d), t)
    return c


def diff_map(a, b):
    return np.clip(np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32)).max(2) * 2.5, 0, 255)


def s_assemble(t, d):
    i = 300 + min(int(t * FPS), 449)
    c = canvas()
    header(c, 6, "Assemble", "align  ·  colour-match  ·  matte  ·  composite  ·  mux audio", t)
    pw, ph = 440, 242
    sc_ = src_crop(i).resize((pw, ph), Image.BILINEAR)
    rw = raw(i).resize((pw, ph), Image.BILINEAR)
    oc = out_crop(i).resize((pw, ph), Image.BILINEAR)
    labels = ["1  SCAIL-2 output", "2  difference vs. source: before | after align", "3  cast matte"]
    before, after = diff_map(rw, sc_), diff_map(oc, sc_)
    dm = np.concatenate([before[:, :pw // 2], after[:, pw // 2:]], 1)
    heat = Image.fromarray(np.stack([dm, dm * 0.55, dm * 0.15], 2).astype(np.uint8))
    m = matte(i).resize((pw, ph), Image.BILINEAR)
    ma = np.asarray(m, np.float32)[..., None] / 255
    mt = Image.fromarray((np.asarray(oc, np.float32) * (0.22 + 0.78 * ma)).astype(np.uint8))
    mt = overlay_mask(mt, m, ORANGE, 0.0, 0.95, 5)
    for k, im in enumerate([rw, heat, mt]):
        y = 196 + k * 284
        a = ramp(t, 0.3 + 0.25 * k)
        text(c, (64, y - 4), labels[k], 21, "SemiBold", TEXT, a)
        panel(c, im, (64, y + 28), (pw, ph), 10, a)
        if k == 1:
            box(c, (64 + pw / 2 - 1, y + 28, 64 + pw / 2 + 1, y + 28 + ph), (255, 255, 255), 0, alpha=a)
        arrow(c, (64 + pw + 10, y + 28 + ph / 2), (64 + pw + 50, y + 28 + ph / 2), ORANGE, 4, 10, a)
    ai_tag(c, 64 + pw, 196 + 28 + ph)
    bx, by, bw, bh = 616, 224, 1200, 675
    s = bw / W
    panel(c, plate(i), (bx, by), (bw, bh), 14)
    r = rect_box(RECTS[i])
    box(c, (bx + r[0] * s, by + r[1] * s, bx + r[2] * s, by + r[3] * s), None, 4, (255, 122, 26, 160), 2)
    text(c, (bx, 192), "4  pasted into the untouched 1920×1080 plate", 21, "SemiBold", TEXT)
    ai_tag(c, bx + bw, by + bh)
    caption(c, cues("s_assemble", d), t, x=616, maxw=1150)
    return c


def s_final(t, d):
    i = 450 + min(int(t * FPS), 299)
    c = plate(i).convert("RGBA")
    c.alpha_composite(BOTTOM_SHADE, (0, H - 300))
    pw, ph = 448, 252
    panel(c, src(i), (W - 64 - pw, H - 64 - ph), (pw, ph), 10, border=(255, 255, 255))
    tag(c, (W - 64 - pw + 12, H - 64 - ph + 12), "source", 18)
    tag(c, (W - 40, 40), "RECAST  ·  AI-generated", 24, BG, ORANGE + (235,), "r")
    caption(c, cues("s_final", d), t, maxw=1180)
    return c


def s_end(t, d):
    bg = plate(700).resize((384, 216), Image.BILINEAR).filter(ImageFilter.GaussianBlur(5)).resize((W, H), Image.BILINEAR)
    c = Image.fromarray((np.asarray(bg, np.float32) * 0.22).astype(np.uint8)).convert("RGBA")
    a = ramp(t, 0.2, 0.6)
    text(c, (W / 2, 210), "Body Double", 110, "Bold", TEXT, a, "m")
    text(c, (W / 2, 352), "Same take. New cast. Open weights only.", 42, "Regular", TEXT, a, "m")
    lines = [
        ("SAM 3.1  ·  FLUX.2 Klein 4B  ·  SCAIL-2 (Wan 2.1 14B) with DPO, lightx2v and optional relight LoRAs", TEXT),
        ("Stock ComfyUI nodes, driven through comfy-sdk and comfy-api-proxy  ·  no partner nodes or closed APIs in the pipeline", MUTED),
        ("This 30 s take on one DGX Spark: 85 min to render, 17 min to assemble", MUTED),
    ]
    for k, (l, col) in enumerate(lines):
        text(c, (W / 2, 480 + k * 56), l, 30, "Regular", col, ramp(t, 0.8 + 0.3 * k, 0.6), "m")
    text(c, (W / 2, 730), "#OpenSourceHiggsfield   ·   Apache-2.0", 38, "SemiBold", ORANGE, ramp(t, 1.8, 0.6), "m")
    text(c, (W / 2, 860), "This video's narration and music: Gemini 3.8 Flash TTS and Lyria 3 Pro (AI-generated, not part of the pipeline)",
         22, "Regular", MUTED, ramp(t, 2.2, 0.6), "m")
    ai_tag(c, W - 30, H - 24)
    return c


SCENES = [(fn, round(max(m, STARTS[fn.__name__][-1] + VO[fn.__name__][-1]["dur"] + TAIL) * FPS) / FPS)
          for fn, m in [(s_title, 4.5), (s_result, 8.5), (s_overview, 8.0), (s_track, 0), (s_follow, 0), (s_anchor, 0),
                        (s_graph, 0), (s_segments, 0), (s_assemble, 0), (s_final, 7.5), (s_end, 6.0)]]
TIMELINE = []
for n, (fn, dur) in enumerate(SCENES):
    TIMELINE += [(n, k / FPS) for k in range(round(dur * FPS))]


def render(g):
    n, t = TIMELINE[g]
    fn, d = SCENES[n]
    im = np.asarray(fn(t, d).convert("RGB"))
    k = min(1.0, (t + 1 / FPS) / 0.35 if n else 1.0, (d - t) / 0.35 if n < len(SCENES) - 1 else (d - t) / 0.8)
    if k < 1:
        im = (im.astype(np.float32) * ease(k)).astype(np.uint8)
    return im.tobytes()


def narration(path, sr=48000):
    out = np.zeros(round(len(TIMELINE) / FPS * sr), np.float32)
    start = 0.0
    for fn, d in SCENES:
        for t0, c in zip(STARTS[fn.__name__], VO[fn.__name__]):
            with wave.open(str(OUT / "vo" / c["file"])) as w:
                a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32)
            i = round((start + t0) * sr)
            out[i:i + len(a)] += a[:len(out) - i]
        start += round(d * FPS) / FPS
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(np.clip(out, -32768, 32767).astype(np.int16).tobytes())


def lufs(path):
    err = subprocess.run(["ffmpeg", "-nostats", "-i", str(path), "-af", "ebur128", "-f", "null", "-"], capture_output=True, text=True).stderr
    return float(err.rsplit("I:", 1)[1].split()[0])


def mix(narr, music, T):
    """filter_complex for inputs 1 (narration) and 2 (music): the music sits 10 LU under the voice, ~20 dB under while she speaks."""
    g = lufs(narr) - 10 - lufs(music)
    return (f"[1:a]pan=stereo|c0=c0|c1=c0,asplit[vo][key];[2:a]atrim=0:{T:.3f},volume={g:.1f}dB,afade=t=out:st={T - 3:.3f}:d=3[bed];"
            "[bed][key]sidechaincompress=threshold=0.06:ratio=3:attack=20:release=400[duck];"
            "[vo][duck]amix=inputs=2:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=11[a]")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if "--stills" in sys.argv:
        sd = OUT / "stills"
        sd.mkdir(exist_ok=True)
        start = 0
        for n, (fn, dur) in enumerate(SCENES):
            for frac in (0.45, 0.9):
                g = start + int(dur * FPS * frac)
                Image.frombytes("RGB", (W, H), render(g)).save(sd / f"{n:02d}_{fn.__name__}_{int(frac * 100)}.jpg", quality=88)
            start += round(dur * FPS)
        print(sd)
        return
    path = OUT / "body_double_explainer.mp4"
    narr, music = OUT / "vo/narration.wav", OUT / "music/music.wav"
    narration(narr)
    T = len(TIMELINE) / FPS
    print(f"{T:.2f} s")
    enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                            "-i", "-", "-i", str(narr), "-i", str(music), "-filter_complex", mix(narr, music, T), "-map", "0:v", "-map", "[a]",
                            "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-ar", "48000", "-c:a", "aac",
                            "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
    with Pool(14) as pool:
        for k, buf in enumerate(pool.imap(render, range(len(TIMELINE)), chunksize=4)):
            enc.stdin.write(buf)
            if k % 250 == 0:
                print(f"{k}/{len(TIMELINE)}", flush=True)
    enc.stdin.close()
    enc.wait()
    print(path)


if __name__ == "__main__":
    main()
