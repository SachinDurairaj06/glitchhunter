"""Video triage on synthetic recordings (MP4 through OpenCV's FFmpeg backend)."""

import cv2 as cv
import numpy as np

from glitchhunter.video import VideoTriage, shots, write_video

H, W = 180, 320


def textured(seed):
    rng = np.random.default_rng(seed)
    small = rng.uniform(40, 200, (H // 6, W // 6, 3)).astype(np.float32)
    return np.clip(cv.resize(small, (W, H), interpolation=cv.INTER_CUBIC), 0, 255).astype(np.uint8)


def crate(img, x, y, size=40):
    cv.rectangle(img, (x, y), (x + size, y + size), (40, 90, 160), -1)
    for k in range(0, size, 8):
        cv.line(img, (x + k, y), (x + k, y + size), (20, 50, 100), 2)
    cv.line(img, (x, y), (x + size, y + size), (200, 220, 240), 2)
    return img


def recording(glitchy: bool) -> list[np.ndarray]:
    frames = []
    bg_a, bg_b = textured(1), textured(2)
    for t in range(40):  # shot 1: a crate that should rest
        y = 60 + ((3 if t % 2 else -3) if glitchy else 0)
        frames.append(crate(bg_a.copy(), 140, y))
    for t in range(40):  # shot 2: a bright panel, with NaN blocks when glitchy
        f = bg_b.copy()
        cv.rectangle(f, (100, 40), (220, 140), (230, 240, 250), -1)
        if glitchy:
            for x, yy in ((130, 60), (144, 74), (158, 60), (172, 74)):
                f[yy:yy + 14, x:x + 14] = 0
        frames.append(f)
    return frames


def test_shots_split_at_the_cut():
    assert shots(recording(False)) == [(0, 40), (40, 80)]


def test_triage_finds_glitches_in_a_recording_and_none_in_a_clean_one(tmp_path):
    bad = write_video(recording(True), tmp_path / "bad.mp4")
    good = write_video(recording(False), tmp_path / "good.mp4")
    found = VideoTriage(bad, tmp_path / "bad-triage").run()
    kept = {f["detector"] for f in found["findings"] if f["status"] == "reported"}
    assert {"physics_jitter", "nan_pixels"} <= kept
    assert (tmp_path / "bad-triage" / "REPORT.md").exists()
    clean = VideoTriage(good, tmp_path / "good-triage").run()
    assert [f for f in clean["findings"] if f["status"] == "reported"] == []
