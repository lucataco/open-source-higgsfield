"""Segment math for chained SCAIL-2 takes.

Chunk k starts at source frame STEP*k and is at most SEG frames long. Its first OVERLAP frames re-render the
previous segment's tail. The last chunk is padded up (clone the last frame) to 4n+1, at least 9 frames, and the
padding is trimmed again after assembly, so F frames in always gives F frames out.
"""
from dataclasses import dataclass

SEG, OVERLAP = 81, 5
STEP = SEG - OVERLAP
MIN_LEN = 9


@dataclass(frozen=True)
class Chunk:
    index: int
    start: int  # first source frame
    frames: int  # real source frames in the chunk
    length: int  # frames sent to the model (4n+1, >= frames)

    @property
    def keep(self):
        """Frames of this segment's output that go into the take (overlap dropped for k >= 1, padding dropped)."""
        return self.frames - (OVERLAP if self.index else 0)


def pad_4n1(n):
    n = max(n, MIN_LEN)
    return n + (-(n - 1)) % 4


def chunks(total):
    if total < 1:
        raise ValueError("empty source")
    out, start = [], 0
    while True:
        frames = min(SEG, total - start)
        out.append(Chunk(len(out), start, frames, pad_4n1(frames)))
        if start + frames >= total:
            return out
        start += STEP


def frames_out(plan):
    return sum(c.keep for c in plan)
