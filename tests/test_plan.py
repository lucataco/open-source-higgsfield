from bodydouble.plan import OVERLAP, SEG, chunks, frames_out


def test_frames_in_equals_frames_out():
    for total in (1, 9, 80, 81, 82, 86, 96, 97, 240, 288, 720, 788, 1440):
        plan = chunks(total)
        assert frames_out(plan) == total, total
        for c in plan:
            assert c.length % 4 == 1 and c.length >= c.frames and c.frames <= SEG
            assert c.keep >= 1
        for a, b in zip(plan, plan[1:]):
            assert b.start == a.start + a.frames - OVERLAP


def test_segment_counts():
    assert len(chunks(81)) == 1
    assert len(chunks(82)) == 2
    assert len(chunks(720)) == 10
    assert len(chunks(788)) == 11
