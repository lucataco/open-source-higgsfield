"""App-side media work: ffmpeg conform and encode, the performer-following crop, references and the composite.

SCAIL-2 does badly on small people, and replacement mode re-synthesises the whole frame from a half-resolution pose
latent. So Body Double renders a virtual camera that follows the performer at the model's resolution, then pastes
the recast performer back into the untouched full-resolution plate.
"""
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

FFMPEG = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
HDR_TRANSFERS = ("arib-std-b67", "smpte2084")
TONEMAP = "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p"


def sh(cmd):
    subprocess.run(cmd, check=True)


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-of", "json", str(path)],
                         check=True, capture_output=True, text=True).stdout
    streams = json.loads(out)["streams"]
    v = next(s for s in streams if s["codec_type"] == "video")
    num, den = map(int, v["r_frame_rate"].split("/"))
    return {"width": v["width"], "height": v["height"], "fps": num / den, "frames": int(v["nb_read_frames"]),
            "transfer": v.get("color_transfer", ""), "audio": any(s["codec_type"] == "audio" for s in streams)}


def conform(src, out_dir, seconds, max_fps=30):
    """Full-res PNG frames, the source audio and a 960-wide tracking proxy. Sources above max_fps go to 24 fps."""
    info = probe(src)
    out_dir = Path(out_dir)
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    for f in (out_dir / "frames").glob("*.png"):  # a re-conform to fewer seconds must not keep the old tail
        f.unlink()
    fps = info["fps"] if info["fps"] <= max_fps else 24.0
    vf = [TONEMAP] if info["transfer"] in HDR_TRANSFERS else []
    if fps != info["fps"]:
        vf.append(f"fps={fps}")
    vf.append("setsar=1")
    sh(FFMPEG + ["-t", str(seconds), "-i", str(src), "-vf", ",".join(vf), "-start_number", "0", str(out_dir / "frames/%05d.png")])
    if info["audio"]:
        sh(FFMPEG + ["-t", str(seconds), "-i", str(src), "-vn", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", str(out_dir / "audio.m4a")])
    frames = sorted((out_dir / "frames").glob("*.png"))
    w, h = info["width"], info["height"]
    pw = 960
    ph = round(h * pw / w / 2) * 2
    encode(out_dir / "frames/%05d.png", fps, out_dir / "track_proxy.mp4", vf=f"scale={pw}:{ph}:flags=area", crf=8)
    return {"fps": fps, "frames": len(frames), "width": w, "height": h, "proxy": [pw, ph], "audio": info["audio"]}


def encode(pattern, fps, out, vf=None, crf=10, start=0, frames=None, audio=None, pad_to=None):
    cmd = FFMPEG + ["-framerate", str(fps), "-start_number", str(start), "-i", str(pattern)]
    if audio:
        cmd += ["-i", str(audio)]
    filters = [vf] if vf else []
    if pad_to and frames and pad_to > frames:
        filters.append(f"tpad=stop_mode=clone:stop={pad_to - frames}")
    if filters:
        cmd += ["-vf", ",".join(filters)]
    if frames:
        cmd += ["-frames:v", str(pad_to or frames)]
    cmd += ["-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p"]
    if audio:
        cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
    sh(cmd + [str(out)])


def mask_array(path, size=None):
    im = Image.open(path).convert("L")
    if size and im.size != size:
        im = im.resize(size, Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def bbox(mask, thresh=0.5):
    ys, xs = np.nonzero(mask > thresh)
    if len(xs) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def color_hist(frame, mask, bins=8):
    """Normalised RGB histogram of the pixels under a mask (appearance signature for re-identification)."""
    px = np.asarray(frame.convert("RGB"), dtype=np.uint8)[mask > 0.5]
    if len(px) == 0:
        return np.zeros(bins ** 3)
    q = (px // (256 // bins)).astype(np.int64)
    h = np.bincount(q[:, 0] * bins * bins + q[:, 1] * bins + q[:, 2], minlength=bins ** 3).astype(np.float64)
    return h / h.sum()


def hist_similarity(a, b):
    return float(np.sqrt(a * b).sum())  # Bhattacharyya coefficient, 1 = identical


def _smooth(values, radius):
    """Centred moving average that holds the ends."""
    v = np.asarray(values, dtype=np.float64)
    padded = np.concatenate([np.full(radius, v[0]), v, np.full(radius, v[-1])])
    kernel = np.ones(2 * radius + 1) / (2 * radius + 1)
    return np.convolve(padded, kernel, mode="valid")


def plan_crops(boxes, src_w, src_h, aspect, margin=1.45, min_h=None, radius=12):
    """One crop rect per frame, aspect = width/height, following the performer smoothly.

    boxes are per-frame (x0, y0, x1, y1) in source pixels or None where the track lost them (held from neighbours).
    """
    idx = [i for i, b in enumerate(boxes) if b is not None]
    if not idx:
        raise ValueError("the performer was never tracked")
    filled = []
    for i in range(len(boxes)):
        j = min(idx, key=lambda k: abs(k - i))
        filled.append(boxes[j])
    b = np.array(filled)
    cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
    need_h = np.maximum((b[:, 3] - b[:, 1]) * margin, (b[:, 2] - b[:, 0]) * margin / aspect)
    max_h = min(src_h, src_w / aspect)
    need_h = np.clip(need_h, min_h or 0, max_h)
    # Smooth zoom, but never much tighter than the local need, so limbs are not cut when the performer opens up.
    h = np.maximum(_smooth(need_h, radius), _smooth(need_h, 2))
    cx, cy = _smooth(cx, radius), _smooth(cy, radius)
    rects = []
    for x, y, hh in zip(cx, cy, h):
        hh = min(hh, max_h)
        ww = hh * aspect
        x0 = min(max(x - ww / 2, 0), src_w - ww)
        y0 = min(max(y - hh / 2, 0), src_h - hh)
        rects.append([round(x0, 2), round(y0, 2), round(ww, 2), round(hh, 2)])
    return rects


def _box(rect, w, h, scale=1.0):
    x, y, rw, rh = (v * scale for v in rect)
    return max(0.0, x), max(0.0, y), min(float(w), x + rw), min(float(h), y + rh)


def render_crops(frames, rects, size, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (f, rect) in enumerate(zip(frames, rects)):
        im = Image.open(f).convert("RGB")
        im.resize(size, Image.LANCZOS, box=_box(rect, im.width, im.height)).save(out_dir / f"{i:05d}.png")


def crop_mask(mask_path, rect, scale, size):
    """A mask saved at tracking-proxy resolution, cut to a crop rect (source px) and resized to the model size."""
    im = Image.open(mask_path).convert("L")
    return im.resize(size, Image.BILINEAR, box=_box(rect, im.width, im.height, scale))


def letterbox(img, size, color=(0, 0, 0)):
    img = img.convert("RGB")
    s = min(size[0] / img.width, size[1] / img.height)
    fit = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    canvas = Image.new("RGB", size, color)
    canvas.paste(fit, ((size[0] - fit.width) // 2, (size[1] - fit.height) // 2))
    return canvas


def on_black(img, mask):
    """The person only, on black (what CLIP vision should see, SCAIL-2#30)."""
    m = Image.fromarray((mask_array(mask, img.size) > 0.5).astype(np.uint8) * 255)
    return Image.composite(img.convert("RGB"), Image.new("RGB", img.size), m)


def edit_size(w, h, megapixels=1.0, step=16):
    s = (megapixels * 1e6 / (w * h)) ** 0.5
    return max(step, round(w * s / step) * step), max(step, round(h * s / step) * step)


def background_stats(img, masks, scale=4, grow=0.04):
    """Per-channel mean and std of the pixels outside the dilated masks, at quarter resolution."""
    small = img.convert("RGB").resize((img.width // scale, img.height // scale), Image.BILINEAR)
    a = np.asarray(small, dtype=np.float32)
    keep = np.ones(a.shape[:2], bool)
    for mp in masks:
        im = Image.open(mp).convert("L").resize(small.size, Image.BILINEAR)
        keep &= np.asarray(im.filter(ImageFilter.MaxFilter(max(3, int(small.height * grow) | 1)))) <= 127
    px = a[keep] if keep.any() else a.reshape(-1, 3)
    return np.concatenate([px.mean(0), px.std(0) + 1e-3])


def color_transforms(out_stats, src_stats, radius=12):
    """Per-frame per-channel (gain, offset) that map the recast background onto the source's, smoothed over time."""
    o, s = np.asarray(out_stats), np.asarray(src_stats)
    gain = np.clip(s[:, 3:] / o[:, 3:], 0.7, 1.4)
    offset = s[:, :3] - o[:, :3] * gain
    return [(g, b) for g, b in zip(np.stack([_smooth(gain[:, c], radius) for c in range(3)], 1),
                                    np.stack([_smooth(offset[:, c], radius) for c in range(3)], 1))]


def apply_color(img, transform):
    g, b = transform
    a = np.asarray(img.convert("RGB"), dtype=np.float32) * g + b
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


KNOTS = (9, 5)  # alignment knots spread evenly across and down the crop


def _hats(n, k):
    """n x k weights that interpolate k knots spread evenly over 0..n linearly, at pixel centres."""
    knots = np.round(np.linspace(0, n, k))
    return np.stack([np.interp(np.arange(n) + 0.5, knots, e) for e in np.eye(k)], 1)


def warp(img, u, resample=Image.BICUBIC, pad=64):
    """Sample img at (x + dx(x), y + dy(y)), dx and dy linear between KNOTS, u = (dx at each knot..., dy...) in pixels.

    One mesh of rectangles, so PIL resamples it in a single pass. Edge pixels are held outside the frame.
    """
    kx, ky = KNOTS
    xk, yk = np.round(np.linspace(0, img.width, kx)), np.round(np.linspace(0, img.height, ky))
    sx, sy = xk + u[:kx] + pad, yk + u[kx:] + pad
    mesh = [((int(xk[i]), int(yk[j]), int(xk[i + 1]), int(yk[j + 1])),
             (sx[i], sy[j], sx[i], sy[j + 1], sx[i + 1], sy[j + 1], sx[i + 1], sy[j]))
            for i in range(kx - 1) for j in range(ky - 1)]
    a = np.asarray(img)
    a = np.pad(a, ((pad, pad), (pad, pad)) + ((0, 0),) * (a.ndim - 2), mode="edge")
    return Image.fromarray(a).transform(img.size, Image.MESH, mesh, resample)


def align(out, src, person, u=None, iters=20, stiff=0.1):
    """Knot displacements (see warp) that put a SCAIL frame back on the source crop it was made from.

    SCAIL-2 replacement squeezes the frame sideways by up to 5%, most at the edges, and drifts it, so its background
    lands up to ~40px off the plate around the paste and the crowd there comes out doubled. Gauss-Newton on the luminance
    of the background (away from the person mask), at quarter then half resolution so a cold start still reaches 40px,
    warm-started from the previous frame. A curvature penalty carries the fit across the knots the person hides.
    """
    kx, ky = KNOTS
    u = np.zeros(kx + ky) if u is None else np.asarray(u, np.float64)
    D = np.zeros((kx + ky, kx + ky))
    for k, part in ((kx, slice(0, kx)), (ky, slice(kx, None))):
        d = np.diff(np.eye(k), 2, axis=0)
        D[part, part] = d.T @ d
    for scale in (0.25, 0.5):
        sw, sh = round(out.width * scale), round(out.height * scale)

        def lum(im):
            return im.convert("L").resize((sw, sh), Image.BILINEAR).filter(ImageFilter.GaussianBlur(1))

        o, s = lum(out), np.asarray(lum(src), np.float32)
        keep = Image.fromarray(person.astype(np.uint8) * 255).resize((sw, sh)).filter(ImageFilter.MaxFilter(int(40 * scale) | 1))
        keep = np.asarray(keep) < 128
        b = round(16 * scale)
        keep[:b] = keep[-b:] = False
        keep[:, :b] = keep[:, -b:] = False
        if keep.mean() < 0.1:
            return u
        P, Q = _hats(sw, kx), _hats(sh, ky)
        v = u * scale
        for _ in range(iters):
            ow = np.asarray(warp(o, v, Image.BILINEAR), np.float32)
            gy, gx = np.gradient(ow)
            r = ow - s
            w = (keep & (np.abs(r) < 40)) / np.maximum(1, np.abs(r) / 8)  # Huber, so SCAIL's redrawn faces do not drag the fit
            xy = w * gx * gy
            H = np.block([[P.T @ ((w * gx * gx).sum(0)[:, None] * P), P.T @ xy.T @ Q],
                          [Q.T @ xy @ P, Q.T @ ((w * gy * gy).sum(1)[:, None] * Q)]])
            g = np.concatenate([P.T @ (w * gx * r).sum(0), Q.T @ (w * gy * r).sum(1)])
            lam = stiff * np.median(np.diag(H))
            dv = np.linalg.solve(H + lam * D + 1e-6 * np.diag(H).mean() * np.eye(kx + ky), g + lam * D @ v)
            v -= dv
            if np.abs(dv).max() < 0.01:
                break
        u = v / scale
    return u


def cast_matte(crop, src, person, seen=None, reach=0.07, thresh=40, scale=0.25):
    """The person mask grown over the cast that sticks out of it: pixels where an aligned recast crop differs from the
    source crop, connected to the person within `reach` (a fraction of the crop height), or anywhere inside `seen`.

    SAM on the recast frame keeps to the source silhouette it was seeded with and drops what reaches past it (a face
    turned further than his, a boa wider than his shoulders), and the paste cut those off. Once align() has put the
    crowd back where the source has it, a big difference next to the person is the cast. Growing from the person keeps
    out the crowd faces SCAIL redrew further away. SAM also loses a whole part for a run of frames while it swings
    out past `reach`, so `seen` (any size; where SAM had the cast in nearby frames) lets the growth follow it there.
    """
    w, h = round(crop.width * scale), round(crop.height * scale)
    diff = np.abs(np.asarray(crop.resize((w, h), Image.BOX), np.float32) - np.asarray(src.resize((w, h), Image.BOX), np.float32))
    grown = np.asarray(Image.fromarray(person.astype(np.uint8) * 255).resize((w, h), Image.BOX)) > 127
    allowed = grown | (diff.max(2) > thresh)
    for _ in range(round(reach * h)):
        grown = (np.asarray(Image.fromarray(grown.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(3))) > 0) & allowed
    if seen is not None:
        allowed &= grown | (np.asarray(Image.fromarray(seen.astype(np.uint8) * 255).resize((w, h), Image.BOX)) > 127)
        while True:
            more = (np.asarray(Image.fromarray(grown.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(3))) > 0) & allowed
            if (more == grown).all():
                break
            grown = more
    return np.asarray(Image.fromarray(grown.astype(np.uint8) * 255).resize(crop.size, Image.BILINEAR), np.float32) / 255


def steady_warps(warps, lengths, radius=6):
    """Smooth per-frame align() fits over time. They wobble by a pixel; SCAIL's drift is slow inside a segment and
    jumps at the seams, so each segment (lengths, in frames) is smoothed on its own."""
    warps = np.array(warps)
    start = 0
    for n in lengths:
        warps[start:start + n] = np.stack([_smooth(c, radius) for c in warps[start:start + n].T], 1)
        start += n
    return warps


def _box_blur(a, radius):
    """Separable box mean. a is H,W,C float. radius 0 is a copy."""
    if radius <= 0:
        return a
    p = np.pad(a, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    c = np.pad(p, ((1, 0), (1, 0), (0, 0)), mode="constant")
    c = np.cumsum(np.cumsum(c, axis=0), axis=1)
    k = 2 * radius + 1
    h, w = a.shape[:2]
    return (c[k:k + h, k:k + w] - c[0:h, k:k + w] - c[k:k + h, 0:w] + c[0:h, 0:w]) / (k * k)


def _deflicker_frame(prev, cur, nxt, amount, radius, lo=4, hi=8):
    """Replace fine detail that the 3-frame median does not share.

    Boil is random, so its residual averages out over the box. A coherent coarse residual is something moving faster
    than its own width (an ear, a glasses arm, a cigarette): the median would erase it and ghost it from the neighbours,
    so the filter fades out between lo and hi levels of coarse residual.
    """
    med = np.median(np.stack((prev, cur, nxt), 0), 0)
    residual = cur - med
    coarse = _box_blur(residual, radius)
    still = np.clip((hi - np.abs(coarse).max(2, keepdims=True)) / (hi - lo), 0, 1)
    return np.clip(cur - amount * still * (residual - coarse), 0, 255).astype(np.uint8)


def deflicker(frames, amount=0.9, radius=None):
    """Drop the fine temporal boil of a 6-step SCAIL sample where nothing moves.

    frames: iterable of HxWx3 float arrays, 0..255. Yields uint8 arrays in order.
    A 3-frame median needs both neighbours, so the first and last frame pass through.
    radius defaults to 8px at 704p and scales with height.
    """
    prev = cur = None
    for nxt in frames:
        if radius is None:
            radius = max(2, round(nxt.shape[0] * 8 / 704))
        if cur is not None:
            yield np.clip(cur, 0, 255).astype(np.uint8) if prev is None else _deflicker_frame(prev, cur, nxt, amount, radius)
        prev, cur = cur, nxt
    if cur is not None:
        yield np.clip(cur, 0, 255).astype(np.uint8)


def steady_masks(src_masks, cast_masks, rects, plate_size, size, gap=24, scale=0.25):
    """Per-frame crop-space composite masks: the union of the source and cast masks, with short dropouts closed.

    SAM loses parts of the performer for a run of frames (a boa, the hair, the whole torso, and now and then the source
    mask), and every lost frame pastes the plate back over the cast, which reads as flicker. A plate
    pixel stays covered while the masks covered it within `gap` frames before and within `gap` frames after. This runs
    in plate space, where the crowd holds still while the follow-cam crop pans and zooms, so a pixel uncovered on one
    side (the performer leaving, or someone in front of them) is left alone. The take's ends count as covered, which
    bridges the first frames while the cast track grows from the source silhouette onto the cast's hair and costume.
    """
    W, H = plate_size
    pw, ph = round(W * scale), round(H * scale)
    n = len(rects)
    cover = np.zeros((n, ph, pw), bool)
    for i in range(n):
        m = np.maximum(mask_array(src_masks[i], size), mask_array(cast_masks[i], size))
        x0, y0, x1, y1 = (round(v) for v in _box(rects[i], pw, ph, scale))
        im = Image.fromarray((m > 0.5).astype(np.uint8) * 255).resize((x1 - x0, y1 - y0), Image.BILINEAR)
        cover[i, y0:y1, x0:x1] = np.asarray(im) > 127
    seen = np.zeros((ph, pw), np.int32)
    last = np.empty((n, ph, pw), np.int32)
    for i in range(n):
        seen = np.where(cover[i], i, seen)
        last[i] = seen
    seen = np.full((ph, pw), n - 1, np.int32)
    for i in reversed(range(n)):
        seen = np.where(cover[i], i, seen)
        cover[i] = seen - last[i] <= gap
    for i in range(n):
        held = Image.fromarray(cover[i].astype(np.uint8) * 255).resize(size, Image.BILINEAR, box=_box(rects[i], pw, ph, scale))
        yield np.maximum.reduce([mask_array(src_masks[i], size), mask_array(cast_masks[i], size), np.asarray(held, np.float32) / 255])


def composite(plate, crop, mask, rect, dilate=0.025, feather=0.012, edge=0.02):
    """Paste a recast crop back into the full-res plate inside a crop-space mask (0..1, from steady_masks)."""
    W, H = crop.size
    alpha = Image.fromarray((mask > 0.5).astype(np.uint8) * 255)
    k = max(3, int(H * dilate) | 1)
    alpha = alpha.filter(ImageFilter.MaxFilter(k)).filter(ImageFilter.GaussianBlur(max(1, H * feather)))
    a = np.asarray(alpha, dtype=np.float32) / 255.0
    e = max(1, int(min(W, H) * edge))
    ramp = np.minimum(np.arange(W) + 1, np.arange(W)[::-1] + 1)[None, :].clip(max=e) / e
    ramp = ramp * (np.minimum(np.arange(H) + 1, np.arange(H)[::-1] + 1)[:, None].clip(max=e) / e)
    x, y, w, h = rect
    # Keep the ramp off plate borders: a crop touching the frame edge has nothing to blend into there.
    if x <= 0.5:
        ramp[:, :e] = 1.0
    if y <= 0.5:
        ramp[:e, :] = 1.0
    if x + w >= plate.width - 0.5:
        ramp[:, -e:] = 1.0
    if y + h >= plate.height - 0.5:
        ramp[-e:, :] = 1.0
    a = Image.fromarray(np.clip(a * ramp * 255, 0, 255).astype(np.uint8))
    box = (round(x), round(y), round(x + w), round(y + h))
    size = (box[2] - box[0], box[3] - box[1])
    out = plate.convert("RGB").copy()
    out.paste(crop.resize(size, Image.LANCZOS), box[:2], a.resize(size, Image.BILINEAR))
    return out
