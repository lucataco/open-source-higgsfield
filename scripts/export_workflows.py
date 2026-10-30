"""Write example API-format workflows (placeholder file names) to workflows/api/ for reference and check_env.py."""
import json
from pathlib import Path

from bodydouble import graphs

OUT = Path(__file__).resolve().parent.parent / "workflows/api"
EXAMPLES = {
    "detect_sam31": graphs.detect("frame0.png", "bodydouble/example/detect/m", text="person:10"),
    "track_sam31": graphs.track("track_proxy.mp4", "init_mask.png", "bodydouble/example/track/m"),
    "anchor_flux3_klein_edit": graphs.anchor("anchor_in.png", "cast.png", 1344, 768, 1, "bodydouble/example/anchor/a"),
    "scail2_segment_first": graphs.segment("c00.mp4", 81, 1280, 704, ["ref0.png"], ["ref0_mask.png"], "ref_cv.png", "init_00.png",
                                           None, "the cast, described", 1, "bodydouble/example/t0/seg00", save_handoff=False,
                                           save_ref_mask=True),
    "scail2_segment_next": graphs.segment("c01.mp4", 81, 1280, 704, ["ref0.png"], ["ref0_mask.png"], "ref_cv.png", "init_01.png",
                                          [f"bodydouble/example/t0/seg00/f_{i:05d}_.png [output]" for i in range(77, 82)],
                                          "the cast, described", 1010, "bodydouble/example/t0/seg01", save_handoff=False),
}

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, g in EXAMPLES.items():
        (OUT / f"{name}.api.json").write_text(json.dumps(g, indent=1))
        print(name, len(g), "nodes")
