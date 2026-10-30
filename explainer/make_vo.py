"""Generate the explainer narration with google/gemini-3.8-flash-tts on Replicate, one clip per caption.

  REPLICATE_API_TOKEN=... .venv/bin/python explainer/make_vo.py

Clips are cached by voice, style and text, so only edited lines are requested again. Writes vo/vo.json, which
make_explainer.py reads for caption text, cue timing and the audio track.
"""
import hashlib
import json
import os
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OUT = Path(__file__).resolve().parent / "vo"
VOICE = "Kore"
STYLE = "Clear, warm and confident product-demo narrator, at a brisk, natural pace"

# Written form -> spoken form, so the captions keep the real names.
SAY = [("SCAIL-2", "Scale two"), ("SAM 3.1", "Sam three point one"), ("SAM", "Sam"), ("FLUX.2", "Flux two"),
       ("ComfyUI", "Comfy U I"), ("comfy-sdk", "Comfy S D K"), ("int8", "int eight"), ("DGX", "D G X"),
       ("1080p", "ten eighty p"), ("~", "about ")]

# scene -> captions, in order. Each caption is spoken as its own clip.
VO = {
    "s_title": ["This is Body Double, and here's how its workflow is set up."],
    "s_result": ["Same take, new cast. The motion, the camera move and the crowd are all original. Only the performer is regenerated."],
    "s_overview": ["One command runs six stages.",
                   "Model stages are stock ComfyUI graphs, queued locally through comfy-sdk."],
    "s_track": ["SAM 3.1 finds the performer in the first frame and tracks them through the take.",
                "If the track drops them, even across a hard cut, everyone is re-detected and the performer is picked again by appearance."],
    "s_follow": ["SCAIL-2 struggles when the performer is small, so a smoothed virtual camera keeps them large, and only that window is regenerated."],
    "s_anchor": ["Bring a cast image. FLUX.2 Klein paints it into the first frame and offers four options.",
                 "A second edit cleans off the source performer's tattoos.",
                 "That frame anchors identity for the whole take, with full-body and portrait views."],
    "s_graph": ["Here is one segment's graph, exactly as it was queued.",
                "SCAIL-2 runs in int8 with faster attention. An optional relight LoRA lets the cast take on the scene's light.",
                "The follow-cam chunk and SAM's track drive the motion.",
                "The anchor and reference views carry identity.",
                "Six sampling steps later, SAM masks the new cast."],
    "s_segments": ["Ten segments, each its own ComfyUI job, overlapping by five frames.",
                   "QC checks every seam against the source, and a bad segment re-rolls before the next one starts."],
    "s_assemble": ["SCAIL-2's output drifts, so each frame is warped back onto the source and colour-matched.",
                   "SAM masks the new cast, and the matte grows into pixels that changed.",
                   "Only the performer is pasted back. The crowd, set and grain stay pixel-identical."],
    "s_final": ["Thirty seconds at 1080p with the original audio. Everything outside the performer is the source plate."],
    "s_end": ["Body Double. Open weights, stock ComfyUI nodes, one machine."],
}


def spoken(s):
    for a, b in SAY:
        s = s.replace(a, b)
    return s


def predict(model, inp):
    """Run a Replicate model and return its (first) output URL."""
    auth = {"Authorization": f"Bearer {os.environ['REPLICATE_API_TOKEN']}"}
    req = urllib.request.Request(f"https://api.replicate.com/v1/models/{model}/predictions", json.dumps({"input": inp}).encode(),
                                 {**auth, "Content-Type": "application/json", "Prefer": "wait"})
    p = json.load(urllib.request.urlopen(req, timeout=120))
    while p["status"] not in ("succeeded", "failed", "canceled"):
        time.sleep(2)
        p = json.load(urllib.request.urlopen(urllib.request.Request(p["urls"]["get"], headers=auth)))
    if p["status"] != "succeeded":
        raise RuntimeError(f"{model}: {p['status']} {p.get('error')}")
    return p["output"][0] if isinstance(p["output"], list) else p["output"]


def request(text, path):
    url = predict("google/gemini-3.8-flash-tts", {"text": text, "voice": VOICE, "style": STYLE, "output_format": "wav"})
    raw = path.with_suffix(".raw.wav")
    urllib.request.urlretrieve(url, raw)
    # trim the silence the model leaves at either end; cue timing adds its own gaps
    trim = "silenceremove=start_periods=1:start_threshold=-50dB"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-af", f"{trim},areverse,{trim},areverse",
                    "-ar", "48000", "-ac", "1", str(path)], check=True)
    raw.unlink()
    return path


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return float(out)


def main():
    OUT.mkdir(exist_ok=True)
    jobs = []
    for scene, lines in VO.items():
        for s in lines:
            say = spoken(s)
            path = OUT / (hashlib.sha1(f"{VOICE}|{STYLE}|{say}".encode()).hexdigest()[:12] + ".wav")
            jobs.append((scene, s, say, path))
    todo = [(say, path) for _, _, say, path in jobs if not path.exists()]
    with ThreadPoolExecutor(6) as ex:
        for path in ex.map(lambda j: request(*j), todo):
            print("generated", path.name, flush=True)
    for f in OUT.glob("*.wav"):
        if f not in {path for *_, path in jobs}:
            f.unlink()
    vo = {}
    for scene, s, say, path in jobs:
        vo.setdefault(scene, []).append({"cap": s, "say": say, "file": path.name, "dur": round(duration(path), 3)})
    (OUT / "vo.json").write_text(json.dumps({"voice": VOICE, "style": STYLE, "scenes": vo}, indent=1))
    total = sum(c["dur"] for cs in vo.values() for c in cs)
    for scene, cs in vo.items():
        print(f"{scene:12s} " + "  ".join(f"{c['dur']:.1f}" for c in cs))
    print(f"speech total {total:.1f} s")


if __name__ == "__main__":
    main()
