import numpy as np
from PIL import Image

from bodydouble.media import KNOTS, _box_blur, align, cast_matte, deflicker, steady_masks, steady_warps, warp


def test_box_blur_of_a_constant_is_the_constant():
    a = np.full((5, 7, 3), 40.0)
    assert np.allclose(_box_blur(a, 2), 40.0)


def test_deflicker_leaves_a_static_frame_alone():
    frame = np.zeros((32, 32, 3), np.float32)
    frame[8:24, 8:24] = 200
    out = list(deflicker([frame, frame.copy(), frame.copy()], radius=4))
    assert np.array_equal(out[1], frame.astype(np.uint8))


def test_deflicker_drops_speckle_and_keeps_a_moving_block():
    rng = np.random.default_rng(0)
    frames = []
    for i in range(7):
        f = np.full((40, 96, 3), 20, np.float32)
        x = 4 + i * 8
        f[10:30, x:x + 16] = 230
        f[:, 84:] += rng.integers(-40, 41, size=(40, 12, 3))
        frames.append(f)
    out = list(deflicker(frames, amount=0.9, radius=3))

    def centroid(img):
        ys, xs = np.nonzero(img[:, :, 0] > 150)
        return xs.mean()

    for i in range(1, 6):
        assert abs(centroid(out[i]) - centroid(frames[i])) < 1.5
    src = np.stack(frames)[1:-1, :, 86:, 0]
    dst = np.stack(out)[1:-1, :, 86:, 0].astype(np.float32)
    # Pixelwise median of three noise fields, so the drop is real but not total.
    assert dst.std(axis=0).mean() < src.std(axis=0).mean() * 0.65


def test_deflicker_keeps_a_thin_bar_moving_faster_than_its_width():
    frames = []
    for i in range(3):
        f = np.full((24, 64, 3), 30, np.float32)
        f[:, 10 + 12 * i:13 + 12 * i] = 220
        frames.append(f)
    out = list(deflicker(frames, radius=3))
    assert np.array_equal(out[1], frames[1].astype(np.uint8))


def _strips(tmp_path, name, w, h, spans):
    a = np.zeros((h, w), np.uint8)
    for x0, x1 in spans:
        a[:, x0:x1] = 255
    Image.fromarray(a).save(tmp_path / name)
    return tmp_path / name


def test_steady_masks_bridge_a_dropout_but_not_a_departure(tmp_path):
    w, h, n = 32, 16, 12
    src = [_strips(tmp_path, f"s{i}.png", w, h, [(0, 8)]) for i in range(n)]
    # The boa (16-24) drops out for frames 4-6; the cast leaves 26-32 after frame 2 and never comes back.
    cast = [_strips(tmp_path, f"c{i}.png", w, h, ([] if 4 <= i <= 6 else [(16, 24)]) + ([(26, 32)] if i < 3 else []))
            for i in range(n)]
    masks = list(steady_masks(src, cast, [[0, 0, w, h]] * n, (w, h), (w, h), gap=4, scale=0.5))
    assert all(m[:, :8].min() > 0.5 for m in masks)
    assert all(masks[i][:, 17:23].min() > 0.5 for i in (4, 5, 6))
    assert all(masks[i][:, 27:].max() < 0.5 for i in range(3, n))


def test_steady_masks_hold_the_take_start(tmp_path):
    w, h, n = 32, 16, 8
    src = [_strips(tmp_path, f"s{i}.png", w, h, [(0, 8)]) for i in range(n)]
    cast = [_strips(tmp_path, f"c{i}.png", w, h, [(0, 8)] + ([(16, 24)] if i >= 2 else [])) for i in range(n)]
    masks = list(steady_masks(src, cast, [[0, 0, w, h]] * n, (w, h), (w, h), gap=4, scale=0.5))
    assert masks[0][:, 17:23].min() > 0.5


def test_align_undoes_a_sideways_squeeze():
    rng = np.random.default_rng(0)
    src = Image.fromarray(rng.integers(0, 256, (22, 40, 3), dtype=np.uint8)).resize((320, 176), Image.BICUBIC)
    # SCAIL's frame: the source squeezed sideways by 4% and nudged.
    kx, ky = KNOTS
    xk = np.linspace(0, 320, kx)
    out = warp(src, np.concatenate([0.04 * xk - 6, np.full(ky, 1.5)]))
    u = align(out, src, np.zeros((176, 320), bool))
    assert np.abs(u[:kx] + (0.04 * xk - 6) / 1.04).max() < 0.6 and np.abs(u[kx:] + 1.5).max() < 0.6
    back = np.asarray(warp(out, u), np.float32)[16:-16, 16:-16]
    assert np.abs(back - np.asarray(src, np.float32)[16:-16, 16:-16]).mean() < 3


def test_steady_warps_do_not_smooth_across_a_seam():
    warps = np.zeros((20, 14))
    warps[10:] = 5
    out = steady_warps(warps, [10, 10], radius=3)
    assert np.allclose(out[:10], 0) and np.allclose(out[10:], 5)


def test_cast_matte_takes_the_cast_past_the_mask_but_not_the_crowd():
    src = np.full((64, 128, 3), 20, np.uint8)
    src[16:48, 20:36] = 230  # him
    crop = np.full((64, 128, 3), 20, np.uint8)
    crop[16:48, 16:52] = (220, 120, 160)  # the cast, wider than him
    crop[16:48, 96:112] = 200  # a crowd face SCAIL redrew, well away from him
    person = np.zeros((64, 128), bool)
    person[16:48, 20:36] = True
    m = cast_matte(Image.fromarray(crop), Image.fromarray(src), person, reach=0.5, scale=0.5)
    assert m[20:44, 18:50].min() > 0.5
    assert m[:, 90:].max() < 0.5


def test_cast_matte_follows_a_part_sam_lost_as_far_as_sam_saw_it():
    src = np.full((64, 128, 3), 20, np.uint8)
    src[16:48, 20:36] = 230
    crop = np.full((64, 128, 3), 20, np.uint8)
    crop[16:48, 16:84] = (220, 120, 160)  # a boa swung out well past reach
    crop[16:48, 84:100] = 200  # a crowd face SCAIL redrew, touching it
    person = np.zeros((64, 128), bool)
    person[16:48, 20:36] = True
    seen = np.zeros((64, 128), bool)
    seen[12:52, 10:84] = True
    assert cast_matte(Image.fromarray(crop), Image.fromarray(src), person, reach=0.1, scale=0.5)[:, 60:].max() < 0.5
    m = cast_matte(Image.fromarray(crop), Image.fromarray(src), person, seen, reach=0.1, scale=0.5)
    assert m[20:44, 18:80].min() > 0.5
    assert m[:, 88:].max() < 0.5
