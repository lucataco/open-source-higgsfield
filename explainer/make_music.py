"""Generate the explainer's background music with google/lyria-3-pro on Replicate.

  REPLICATE_API_TOKEN=... .venv/bin/python explainer/make_music.py [seed]

Cached by prompt and seed. make_explainer.py trims the track to the video, fades it out and ducks it under the
narration. The section times follow the explainer's scenes: result reveal, pipeline steps, final result, end card.
"""
import hashlib
import subprocess
import sys
import urllib.request
from pathlib import Path

from make_vo import predict

OUT = Path(__file__).resolve().parent / "music"
PROMPT = """Instrumental background score for a tech product explainer with a voice-over, exactly 2 minutes long. No vocals, no lyrics, no spoken words.
Modern minimal electronic: warm analog synth pads, a soft plucked synth arpeggio, a deep round sub bass and light, crisp drums.
108 BPM, D minor, confident, curious and focused. Keep the midrange sparse and the melody minimal so it sits under a narrator. No drops, no sudden loud hits.
[0:00 - 0:13] Intro: pads swell in and the arpeggio enters.
[0:13 - 0:56] Light groove: a soft kick, closed hi-hats and a pulsing bass join.
[0:56 - 1:45] The same groove with one subtle extra synth layer, steady energy.
[1:45 - 1:53] A gentle lift with fuller pads.
[1:53 - 2:00] Outro: drums drop out, the arpeggio plays over a held chord, and the track ends at 2:00."""


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    OUT.mkdir(exist_ok=True)
    path = OUT / (hashlib.sha1(f"{seed}|{PROMPT}".encode()).hexdigest()[:12] + ".wav")
    if not path.exists():
        raw = path.with_suffix(".raw")
        urllib.request.urlretrieve(predict("google/lyria-3-pro", {"prompt": PROMPT, "seed": seed}), raw)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-ar", "48000", "-ac", "2", str(path)], check=True)
        raw.unlink()
    (OUT / "music.wav").unlink(missing_ok=True)
    (OUT / "music.wav").symlink_to(path.name)
    print(path)


if __name__ == "__main__":
    main()
