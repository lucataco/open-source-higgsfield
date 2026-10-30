"""Per-segment QC, run before a segment's tail conditions the next one.

seam: mean |dY| across the boundary over the median |dY| of the nearby frames, on quarter-res luma. The same number on
the source at the same index is the baseline, so real motion at a seam is not flagged.
drift: mean luma outside the dilated performer masks, output vs source, relative to segment 0 (SCAIL-2#11 creep).
Drift is reported but does not trigger a re-roll: assembly matches every frame's colours to the source plate.
"""
import numpy as np
from PIL import Image, ImageFilter

SEAM_LIMIT = 1.5  # est.; flag if ratio > max(SEAM_LIMIT, SEAM_LIMIT * source ratio)
DRIFT_LIMIT = 0.03  # est.; fraction of the source background luma


def luma(path, scale=4):
    im = Image.open(path).convert("L")
    return np.asarray(im.resize((im.width // scale, im.height // scale), Image.BILINEAR), dtype=np.float32)


def seam_ratio(frames, b, window=12):
    """frames: luma arrays; b: index of the first frame after the seam."""
    diffs = [float(np.abs(frames[i] - frames[i - 1]).mean()) for i in range(1, len(frames))]
    across = diffs[b - 1]
    nearby = [d for i, d in enumerate(diffs) if abs(i - (b - 1)) <= window and i != b - 1]
    return across / max(float(np.median(nearby)), 1e-3)


def background(path, masks, scale=4, grow=0.04):
    y = luma(path, scale)
    m = None
    for mp in masks:
        im = Image.open(mp).convert("L").resize((y.shape[1], y.shape[0]), Image.BILINEAR)
        k = max(3, int(y.shape[0] * grow) | 1)
        a = np.asarray(im.filter(ImageFilter.MaxFilter(k)), dtype=np.float32) > 127
        m = a if m is None else (m | a)
    bg = y[~m] if m is not None and (~m).any() else y.ravel()
    return float(bg.mean())


def segment(out_frames, src_frames, out_masks, src_masks, prev_out_tail=None, prev_src_tail=None, baseline_drift=None):
    """QC for one segment. *_frames/*_masks are the kept frames of this segment; prev_* the previous segment's last
    kept frame (output and source), for the seam check."""
    rec = {}
    if prev_out_tail:
        n = min(len(out_frames), 13)
        out = [luma(prev_out_tail)] + [luma(f) for f in out_frames[:n]]
        src = [luma(prev_src_tail)] + [luma(f) for f in src_frames[:n]]
        rec["seam"] = round(seam_ratio(out, 1), 3)
        rec["seam_src"] = round(seam_ratio(src, 1), 3)
    picks = sorted({0, len(out_frames) // 2, len(out_frames) - 1})
    o = np.mean([background(out_frames[i], [out_masks[i], src_masks[i]]) for i in picks])
    s = np.mean([background(src_frames[i], [out_masks[i], src_masks[i]]) for i in picks])
    rec["drift"] = round(float(o / max(s, 1e-3) - 1), 4)
    flags = []
    if "seam" in rec and rec["seam"] > max(SEAM_LIMIT, SEAM_LIMIT * rec["seam_src"]):
        flags.append("seam")
    rec["flags"] = flags
    if baseline_drift is not None and abs(rec["drift"] - baseline_drift) > DRIFT_LIMIT:
        rec["warn"] = "drift"
    return rec
