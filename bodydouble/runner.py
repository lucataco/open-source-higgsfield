"""Recast one performer in a clip, resumably.

conform -> detect -> track -> crops -> anchor -> refs -> segments (QC, re-roll) -> assemble

Every stage records its result in runs/<job>/state.json, so a crashed or interrupted run resumes where it stopped.
Usage: python -m bodydouble.runner JOB --src clip.mp4 --cast cast.png --prompt "..." [--stages all]
"""
import argparse
import fcntl
import json
import re
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from . import graphs, media, qc
from .client import COMFYUI_DIR, Client, RunError, annotated
from .plan import chunks

ROOT = Path(__file__).resolve().parent.parent
INPUT, OUTPUT = COMFYUI_DIR / "input", COMFYUI_DIR / "output"
STAGES = ["conform", "detect", "track", "crops", "anchor", "refs", "segments", "assemble"]


def out_path(o):
    return (OUTPUT if o["type"] == "output" else INPUT) / o["subfolder"] / o["filename"]


def files(outs, prefix):
    return sorted([o for o in outs if o["filename"].startswith(prefix + "_")], key=lambda o: o["filename"])


class Job:
    def __init__(self, name, transport="sdk"):
        self.name = name
        self.dir = ROOT / "runs" / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.rel = f"bodydouble/{name}"
        self.inp = INPUT / self.rel
        self.inp.mkdir(parents=True, exist_ok=True)
        self.lock = open(self.dir / ".lock", "w")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"job {name} is already running in another process") from None
        self.state_path = self.dir / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.client = Client(self.dir / "sdk.log", transport)

    def save(self):
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1))
        tmp.replace(self.state_path)

    def stage(self, rel_name, src):
        """Copy or save a file into ComfyUI/input/bodydouble/<job>/ and return its input-relative name."""
        dst = self.inp / rel_name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(src, Image.Image):
            src.save(dst)
        else:
            shutil.copyfile(src, dst)
        return f"{self.rel}/{rel_name}"

    def run(self, graph, tag, stack):
        (self.dir / "graphs").mkdir(exist_ok=True)
        (self.dir / "graphs" / f"{tag.replace('/', '_')}.api.json").write_text(json.dumps(graph, indent=1))
        return self.client.run(graph, tag, stack)


def size_key(size):
    return f"{size[0]}x{size[1]}"


# -- stages ------------------------------------------------------------------------------------------------------------
def conform(job, a):
    info = media.conform(a.src, job.dir / "src", a.seconds)
    job.state["src"] = info
    job.state["src_file"] = str(Path(a.src).resolve())


def detect(job, a):
    src = job.state["src"]
    pw, ph = src["proxy"]
    frame0 = Image.open(job.dir / "src/frames/00000.png").convert("RGB").resize((pw, ph), Image.LANCZOS)
    name = job.stage("frame0_proxy.png", frame0)
    outs = job.run(graphs.detect(name, f"{job.rel}/detect/m", text=a.detect_text), "detect", "sam")
    masks = [media.mask_array(out_path(o)) for o in files(outs, "m")]
    if not masks:
        raise RunError(f"SAM 3.1 found no '{a.detect_text}' in frame 0")
    areas = [float((m > 0.5).sum()) for m in masks]
    pick = a.pick if a.pick is not None else max(range(len(masks)), key=lambda i: areas[i])
    job.state["init_mask"] = job.stage("init_mask_proxy.png", Image.fromarray(((masks[pick] > 0.5) * 255).astype("uint8")))
    job.state["detect"] = {"count": len(masks), "areas": areas, "pick": pick}


def _proxy_window(job, start, frames):
    src = job.state["src"]
    out = job.dir / "src/track_windows" / f"w{start:05d}.mp4"
    out.parent.mkdir(exist_ok=True)
    pw, ph = src["proxy"]
    media.encode(job.dir / "src/frames/%05d.png", src["fps"], out, vf=f"scale={pw}:{ph}:flags=area", crf=8, start=start, frames=frames)
    return job.stage(f"track/w{start:05d}.mp4", out)


def _reacquire(job, a, i, ref_hist, last_box):
    """Re-detect the performer at frame i: the person whose colours best match the performer, near where they were."""
    pw, ph = job.state["src"]["proxy"]
    frame = Image.open(job.dir / f"src/frames/{i:05d}.png").convert("RGB").resize((pw, ph), Image.LANCZOS)
    name = job.stage(f"track/f{i:05d}.png", frame)
    outs = job.run(graphs.detect(name, f"{job.rel}/reacquire/f{i:05d}", text=a.reacquire_text), f"reacquire{i}", "sam")
    best, best_score = None, -1.0
    for o in files(outs, f"f{i:05d}"):
        m = media.mask_array(out_path(o))
        b = media.bbox(m)
        if b is None:
            continue
        dist = abs((b[0] + b[2]) / 2 - (last_box[0] + last_box[2]) / 2) / pw if last_box else 0.0
        score = media.hist_similarity(media.color_hist(frame, m), ref_hist) - 0.5 * dist
        if score > best_score:
            best, best_score = m, score
    if best is None:
        raise RunError(f"could not re-acquire the performer at frame {i}")
    return best, round(best_score, 3)


def track(job, a):
    """Track the performer in windows; when the track is lost, re-identify them by appearance and continue."""
    n = job.state["src"]["frames"]
    seed = job.state["init_mask"]
    frame0 = Image.open(INPUT / job.state["init_mask"]).size
    masks, events, i = [None] * n, [], 0
    hists, areas, last_box = [], [], None
    while i < n:
        e = min(i + a.track_window, n)
        win = _proxy_window(job, i, e - i)
        outs = job.run(graphs.track(win, seed, f"{job.rel}/track/w{i:05d}"), f"track{i}", "sam")
        wm = [str(out_path(o)) for o in files(outs, f"w{i:05d}")]
        lost = None
        for j, path in enumerate(wm):
            m = media.mask_array(path)
            area, b = float((m > 0.5).sum()), media.bbox(m)
            recent = float(np.median(areas[-5:])) if areas else area
            jump = last_box is not None and b is not None and abs((b[0] + b[2]) / 2 - (last_box[0] + last_box[2]) / 2) > 0.15 * frame0[0]
            if b is None or area < 0.35 * recent or jump:
                lost = i + j
                break
            masks[i + j] = path
            areas.append(area)
            last_box = b
            if len(hists) < 30:
                hists.append(media.color_hist(Image.open(job.dir / f"src/frames/{i + j:05d}.png").resize(frame0), m))
        if lost is None:
            if e >= n:
                break
            i = e - 1  # windows overlap by one frame so the seed mask belongs to the window's first frame
            seed = job.stage(f"track/seed_{i:05d}.png", Image.open(masks[i]).convert("L"))
            continue
        if lost == i:
            raise RunError(f"track lost at its own seed frame {i}")
        m, score = _reacquire(job, a, lost, np.mean(hists, axis=0), last_box)
        events.append({"frame": lost, "score": score})
        print(f"[{job.name}] track lost at {lost}, re-acquired (score {score})", flush=True)
        i = lost
        seed = job.stage(f"track/seed_{i:05d}.png", Image.fromarray(((m > 0.5) * 255).astype("uint8")))
        # Judge the next window against the re-acquired mask: after a cut the performer's size and place change.
        areas, last_box = [float((m > 0.5).sum())], media.bbox(m)
    missing = [k for k, m in enumerate(masks) if m is None]
    if missing:
        raise RunError(f"{len(missing)} frames without a mask, first {missing[0]}")
    job.state["track_masks"] = masks
    job.state["track_events"] = events


def crops(job, a):
    src, key = job.state["src"], size_key(a.size)
    scale = src["proxy"][0] / src["width"]
    boxes = []
    for m in job.state["track_masks"]:
        b = media.bbox(media.mask_array(m))
        boxes.append(None if b is None else tuple(v / scale for v in b))
    rects = media.plan_crops(boxes, src["width"], src["height"], a.size[0] / a.size[1], min_h=src["height"] * a.min_crop)
    frames = sorted((job.dir / "src/frames").glob("*.png"))
    crop_dir = job.dir / f"crop_{key}"
    media.render_crops(frames, rects, tuple(a.size), crop_dir / "frames")
    plan = chunks(len(frames))
    staged = []
    for c in plan:
        mp4 = crop_dir / f"c{c.index:02d}.mp4"
        media.encode(crop_dir / "frames/%05d.png", src["fps"], mp4, crf=6, start=c.start, frames=c.frames, pad_to=c.length)
        init = media.crop_mask(job.state["track_masks"][c.start], rects[c.start], scale, tuple(a.size))
        staged.append({"chunk": job.stage(f"{key}/c{c.index:02d}.mp4", mp4), "init": job.stage(f"{key}/init_{c.index:02d}.png", init),
                       "start": c.start, "frames": c.frames, "length": c.length, "keep": c.keep})
    job.state.setdefault("crops", {})[key] = {"rects": rects, "chunks": staged, "dir": str(crop_dir)}


def anchor(job, a):
    key = size_key(a.size)
    crop0 = Image.open(Path(job.state["crops"][key]["dir"]) / "frames/00000.png")
    ew, eh = media.edit_size(*crop0.size)
    crop_name = job.stage(f"{key}/anchor_in.png", crop0.resize((ew, eh), Image.LANCZOS))
    cast = Image.open(a.cast).convert("RGB")
    cw, ch = media.edit_size(*cast.size)
    cast_name = job.stage("cast.png", cast.resize((cw, ch), Image.LANCZOS))
    if a.cast_edit:
        outs = job.run(graphs.edit([cast_name], a.cast_edit, cw, ch, a.seed, f"{job.rel}/cast_edit/c"), "cast_edit", "klein")
        cast_name = job.stage("cast_edit.png", Image.open(out_path(files(outs, "c")[0])).convert("RGB"))
    outs = job.run(graphs.anchor(crop_name, cast_name, ew, eh, a.seed, f"{job.rel}/anchor_{key}/a", batch=a.anchors), "anchor", "klein")
    paths = []
    adir = job.dir / f"anchors_{key}"
    adir.mkdir(exist_ok=True)
    for i, o in enumerate(files(outs, "a")):
        p = adir / f"a{i}.png"
        Image.open(out_path(o)).convert("RGB").resize(tuple(a.size), Image.LANCZOS).save(p)
        paths.append(str(p))
    sheet = Image.new("RGB", (a.size[0] * (len(paths) + 1), a.size[1]))
    sheet.paste(crop0, (0, 0))
    for i, p in enumerate(paths):
        sheet.paste(Image.open(p), (a.size[0] * (i + 1), 0))
    sheet.save(adir / "sheet.jpg", quality=90)
    job.state.setdefault("anchors", {})[key] = paths


def refs(job, a):
    """ref 0 = the chosen anchor (or the raw letterboxed cast with --no-anchor); optional extra view = the cast.
    The anchor's mask is the detected person that best overlaps where the performer stands in frame 0."""
    key = size_key(a.size)
    init = media.mask_array(INPUT / job.state["crops"][key]["chunks"][0]["init"]) > 0.5
    views = []
    if not a.no_anchor:
        views.append(("ref0", Image.open(job.state["anchors"][key][a.anchor]).convert("RGB"), init))
    if a.cast_view or a.no_anchor:
        cast = INPUT / f"{job.rel}/cast_edit.png" if a.cast_edit else a.cast
        views.append(("ref_cast", media.letterbox(Image.open(cast), tuple(a.size)), None))
    out = []
    for name, img, where in views:
        img_name = job.stage(f"{key}/{name}.png", img)
        outs = job.run(graphs.detect(img_name, f"{job.rel}/refs_{key}/{name}", text=a.ref_text), f"detect_{name}", "sam")
        masks = [media.mask_array(out_path(o)) > 0.5 for o in files(outs, name)]
        if not masks:
            raise RunError(f"no '{a.ref_text}' found in {name}")
        best = max(masks, key=lambda m: float((m & where).sum() if where is not None else m.sum()))
        mask_name = job.stage(f"{key}/{name}_mask.png", Image.fromarray(best.astype("uint8") * 255))
        out.append({"image": img_name, "mask": mask_name})
    cv = media.on_black(Image.open(INPUT / out[0]["image"]), INPUT / out[0]["mask"])
    job.state.setdefault("refs", {})[key] = {"views": out, "clip_vision": job.stage(f"{key}/ref_cv.png", cv),
                                              "anchor": None if a.no_anchor else a.anchor}


def check_props(job, a, key, ref):
    """Stop before a take's first segment if the cast holds a prop in the references, or the prompt names one.
    SCAIL-2 sees the source video and draws a prop wherever the performer holds one; a prop that only the references
    have is drawn in where the performer has none and pops in and out between segments. The anchor is the first frame
    repainted, so a prop the performer already holds there is the source's and is allowed in the anchor."""
    names = [n.strip() for n in a.props.split(",") if n.strip()]
    if not names:
        return
    size = tuple(a.size)
    crop = job.state["crops"][key]
    first = job.stage(f"{key}/props_src.png", Image.open(sorted((Path(crop["dir"]) / "frames").glob("*.png"))[0]))
    views = [{"image": first, "mask": crop["chunks"][0]["init"]}] + ref["views"]
    outs = job.run(graphs.props([v["image"] for v in views], names, f"{job.rel}/props_{key}/p"), f"props_{a.take}", "sam")
    found, held = {}, set()
    for i, v in enumerate(views):
        person = media.mask_array(INPUT / v["mask"], size) > 0.5
        q = Image.fromarray(person.astype(np.uint8) * 255).resize((size[0] // 4, size[1] // 4), Image.BILINEAR)
        near = np.asarray(q.filter(ImageFilter.MaxFilter(5)).resize(size, Image.NEAREST)) > 0  # within ~8 px at 704p
        seen = np.zeros(size[::-1], bool)
        for j, n in enumerate(names):  # SAM answers to several of the names for one small object; keep the first
            m = media.mask_array(out_path(files(outs, f"{i}_{j}")[0]), size) > 0.5
            # With nothing to find, SAM now and then returns most of the frame.
            if (m & near).any() and m.sum() < person.sum() / 10 and (m & ~seen).sum() > m.sum() / 2:
                if i == 0:
                    held.add(n)
                elif not (i == 1 and ref.get("anchor") is not None and n in held):
                    found.setdefault(n, []).append(Path(v["image"]).stem)
            seen |= m
    for n in names:
        if re.search(rf"\b{re.escape(n)}s?\b", a.prompt, re.I):
            found.setdefault(n, []).append("--prompt")
    if found:
        what = "; ".join(f"{n} in {', '.join(w)}" for n, w in found.items())
        raise RunError(f"the cast holds a prop ({what}). SCAIL-2 takes props from the source video, so one in the references "
                       f"or prompt pops in wherever the performer has none. Drop it from --prompt and the cast (--cast-edit "
                       f"\"Remove the {next(iter(found))}. Keep everything else the same.\" --redo anchor,refs), or pass --props ''")


def segments(job, a):
    key = size_key(a.size)
    crop, ref = job.state["crops"][key], job.state["refs"][key]
    take = job.state.setdefault("takes", {}).setdefault(a.take, {"size": key, "segments": [], "cfg": {}})
    cfg = dict(graphs.SCAIL, **{k: v for k, v in vars(a).items() if k in graphs.SCAIL and v is not None})
    if not take["segments"]:
        check_props(job, a, key, ref)
    take["cfg"], take["prompt"], take["refs"] = cfg, a.prompt, ref
    src_crops = sorted((Path(crop["dir"]) / "frames").glob("*.png"))
    for k in range(len(take["segments"]), len(crop["chunks"])):
        c = crop["chunks"][k]
        prev = take["segments"][-1] if take["segments"] else None
        best = None
        for attempt in range(a.rerolls + 1):
            seed = a.seed + 1009 * k + 100003 * a.take_index + 7919 * attempt
            prefix = f"{job.rel}/{a.take}/seg{k:02d}_{attempt}"
            g = graphs.segment(c["chunk"], c["length"], a.size[0], a.size[1], [v["image"] for v in ref["views"]],
                               [v["mask"] for v in ref["views"]], ref["clip_vision"], c["init"],
                               prev["tail"] if prev else None, a.prompt, seed, prefix, save_handoff=False,
                               save_ref_mask=(k == 0 and attempt == 0), cfg=cfg)
            t0 = time.time()
            outs = job.run(g, f"{a.take}/seg{k:02d}_{attempt}", "scail")
            secs = round(time.time() - t0)
            frames = [str(out_path(o)) for o in files(outs, "f")][:c["keep"]]
            out_masks = [str(out_path(o)) for o in files(outs, "cast_mask")][:c["keep"]]
            src_masks = [str(out_path(o)) for o in files(outs, "src_mask")][:c["keep"]]
            first = c["start"] + (5 if k else 0)
            src = [str(p) for p in src_crops[first:first + c["keep"]]]
            rec = qc.segment(frames, src, out_masks, src_masks, prev["frames"][-1] if prev else None,
                             str(src_crops[first - 1]) if prev else None,
                             take["segments"][0]["qc"]["drift"] if take["segments"] else None)
            seg = {"k": k, "attempt": attempt, "seed": seed, "secs": secs, "qc": rec, "frames": frames,
                   "cast_masks": out_masks, "src_masks": src_masks,
                   "tail": [annotated(o) for o in files(outs, "f")][c["keep"] - 5:c["keep"]]}
            if k == 0 and attempt == 0:
                ref_masks = files(outs, "ref_mask")
                if ref_masks and media.mask_array(out_path(ref_masks[0])).max() < 0.1:
                    raise RunError("reference_image_mask is all background: the cast would get no identity")
            print(f"[{job.name}/{a.take}] seg {k:02d} try {attempt}: {secs}s qc {rec}", flush=True)
            if best is None or len(rec["flags"]) < len(best["qc"]["flags"]):
                best = seg
            if not rec["flags"]:
                break
        take["segments"].append(best)
        job.save()


def assemble(job, a):
    key = size_key(a.size)
    take, crop, src = job.state["takes"][a.take], job.state["crops"][key], job.state["src"]
    frames = [f for s in take["segments"] for f in s["frames"]]
    cast_masks = [f for s in take["segments"] for f in s["cast_masks"]]
    src_masks = [f for s in take["segments"] for f in s["src_masks"]]
    if len(frames) != src["frames"]:
        raise RunError(f"{len(frames)} frames out for {src['frames']} in")
    audio = job.dir / "src/audio.m4a" if src["audio"] else None
    cdir = job.dir / a.take / "crop"
    pdir = job.dir / a.take / "plate"
    cdir.mkdir(parents=True, exist_ok=True)
    pdir.mkdir(parents=True, exist_ok=True)
    plates = sorted((job.dir / "src/frames").glob("*.png"))
    src_crops = sorted((Path(crop["dir"]) / "frames").glob("*.png"))
    # Match each recast frame's background colours to the source crop's (fixes SCAIL-2 brightness creep).
    out_stats = [media.background_stats(Image.open(f), [src_masks[i], cast_masks[i]]) for i, f in enumerate(frames)]
    src_stats = [media.background_stats(Image.open(src_crops[i]), [src_masks[i], cast_masks[i]]) for i in range(len(frames))]
    fixes = media.color_transforms(out_stats, src_stats)
    # SCAIL-2 squeezes and drifts the frame, so put each one, and its cast mask, back on the source crop.
    size = tuple(a.size)
    adir = job.dir / a.take / "cast_mask"
    adir.mkdir(parents=True, exist_ok=True)
    warps, u = [], None
    for i, f in enumerate(frames):
        person = np.maximum(media.mask_array(src_masks[i], size), media.mask_array(cast_masks[i], size)) > 0.5
        u = media.align(media.apply_color(Image.open(f), fixes[i]), Image.open(src_crops[i]), person, u)
        warps.append(u)
    warps = media.steady_warps(warps, [len(s["frames"]) for s in take["segments"]])
    # SAM loses a part of the cast for a run of frames now and then; the matte looks where it had it within a second.
    q = (size[0] // 4, size[1] // 4)
    sam = np.stack([np.asarray(media.warp(Image.fromarray((media.mask_array(m, q) * 255).astype(np.uint8)), warps[i] / 4, Image.BILINEAR)) > 127
                    for i, m in enumerate(cast_masks)])
    aligned = (media.warp(media.apply_color(Image.open(f), fixes[i]), warps[i]) for i, f in enumerate(frames))
    for i, crop_img in enumerate(aligned):
        cast = Image.fromarray((media.mask_array(cast_masks[i], size) * 255).astype(np.uint8))
        cast = np.asarray(media.warp(cast, warps[i], Image.BILINEAR), np.float32) / 255
        person = np.maximum(media.mask_array(src_masks[i], size), cast) > 0.5
        seen = sam[max(0, i - 24):i + 25].any(0)
        cast = np.maximum(cast, media.cast_matte(crop_img, Image.open(src_crops[i]).convert("RGB"), person, seen))
        Image.fromarray((cast * 255).astype(np.uint8)).save(adir / f"{i:05d}.png")
    cast_masks = [adir / f"{i:05d}.png" for i in range(len(frames))]
    # Colour-match first, then drop hair/glasses/sequin boil. The original plate is pasted around this crop.
    matched = (np.asarray(media.warp(media.apply_color(Image.open(f), fixes[i]), warps[i]), np.float32) for i, f in enumerate(frames))
    masks = media.steady_masks(src_masks, cast_masks, crop["rects"], (src["width"], src["height"]), size)
    for i, (arr, mask) in enumerate(zip(media.deflicker(matched), masks)):
        crop_img = Image.fromarray(arr, "RGB")
        crop_img.save(cdir / f"{i:05d}.png", compress_level=1)  # level 6 took 3/4 of assemble
        media.composite(Image.open(plates[i]), crop_img, mask, crop["rects"][i]).save(pdir / f"{i:05d}.png", compress_level=1)
    out = job.dir / a.take
    media.encode(cdir / "%05d.png", src["fps"], out / f"{job.name}_{a.take}_crop.mp4", crf=14, audio=audio)
    media.encode(pdir / "%05d.png", src["fps"], out / f"{job.name}_{a.take}.mp4", crf=14, audio=audio)
    take["outputs"] = [str(out / f"{job.name}_{a.take}.mp4"), str(out / f"{job.name}_{a.take}_crop.mp4")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("job")
    p.add_argument("--src")
    p.add_argument("--cast")
    p.add_argument("--prompt", default="")
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--size", type=lambda s: [int(v) for v in s.split("x")], default=[512, 896])
    p.add_argument("--min-crop", type=float, default=0.25, help="smallest crop height as a fraction of the source height")
    p.add_argument("--detect-text", default="person:10")
    p.add_argument("--ref-text", default="person:5")
    p.add_argument("--pick", type=int)
    p.add_argument("--track-window", type=int, default=100)
    p.add_argument("--reacquire-text", default="person:40")
    p.add_argument("--anchors", type=int, default=4)
    p.add_argument("--anchor", type=int, default=0)
    p.add_argument("--no-anchor", action="store_true")
    p.add_argument("--cast-view", action="store_true", help="add the letterboxed cast image as an extra reference view")
    p.add_argument("--cast-edit", help="FLUX.2 Klein instruction applied to the cast before the anchor, e.g. to remove a prop")
    p.add_argument("--props", default="cigarette,phone,microphone",
                   help="held props that stop a take if the references show one or the prompt names one; '' to skip")
    p.add_argument("--take", default="t0")
    p.add_argument("--take-index", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--rerolls", type=int, default=1)
    for k in ("dit", "dpo", "lightx2v", "relight", "attention"):
        p.add_argument("--" + k)
    for k in ("dpo_strength", "lightx2v_strength", "relight_strength", "shift", "sparse_tau"):
        p.add_argument("--" + k.replace("_", "-"), dest=k, type=float)
    p.add_argument("--steps", type=int)
    p.add_argument("--stages", default="all")
    p.add_argument("--redo", default="", help="comma list of stages to run again even if done")
    p.add_argument("--transport", default="sdk", choices=["sdk", "raw"])
    a = p.parse_args()
    job = Job(a.job, a.transport)
    wanted = STAGES if a.stages == "all" else a.stages.split(",")
    redo = set(filter(None, a.redo.split(",")))
    done = job.state.setdefault("done", [])
    for name in STAGES:
        if name not in wanted:
            continue
        tag = name if name in ("conform", "detect", "track") else f"{name}:{size_key(a.size)}" + (f":{a.take}" if name in ("segments", "assemble") else "")
        if tag in done and name not in redo and name != "segments":
            continue
        print(f"[{job.name}] {tag} ...", flush=True)
        t0 = time.time()
        try:
            globals()[name](job, a)
        except Exception:
            job.state.setdefault("errors", []).append({"stage": tag, "error": traceback.format_exc()[-3000:], "t": time.time()})
            job.save()
            raise
        if tag not in done:
            done.append(tag)
        job.state.setdefault("timings", {})[tag] = round(time.time() - t0)
        job.save()
        print(f"[{job.name}] {tag} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
