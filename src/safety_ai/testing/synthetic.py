"""Skeleton dan skenario sintetis. Dipakai oleh unit test dan tools/simulate.py (tanpa model, tanpa kamera).

Koordinat lokal: x ke kanan, y ke atas, satuan = tinggi tubuh, titik (0,0) = kaki.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

from ..types import Track

#            0 nose   1 eyeL   2 eyeR   3 earL   4 earR   5 shL    6 shR    7 elL    8 elR
#            9 wrL   10 wrR   11 hipL  12 hipR  13 knL   14 knR   15 ankL  16 ankR
POSES = {
    "stand": [(0, .94), (-.02, .96), (.02, .96), (-.04, .94), (.04, .94), (-.12, .80), (.12, .80), (-.16, .62),
              (.16, .62), (-.17, .45), (.17, .45), (-.07, .50), (.07, .50), (-.08, .25), (.08, .25), (-.08, 0),
              (.08, 0)],
    # berbaring horizontal, kepala di kiri
    "lie": [(-.46, .10), (-.45, .12), (-.45, .08), (-.43, .12), (-.43, .08), (-.30, .15), (-.30, .05), (-.15, .16),
            (-.15, .04), (0, .15), (0, .05), (.05, .13), (.05, .05), (.25, .12), (.25, .06), (.45, .10), (.45, .04)],
    # duduk tegak di lantai
    "sit": [(0, .60), (-.02, .62), (.02, .62), (-.04, .60), (.04, .60), (-.12, .46), (.12, .46), (-.15, .32),
            (.15, .32), (-.15, .18), (.15, .18), (-.07, .14), (.07, .14), (-.10, .22), (.10, .22), (-.14, .02),
            (.14, .02)],
    # membungkuk mengambil mainan (kepala di bawah pinggul)
    "bend": [(.30, .30), (.29, .32), (.31, .32), (.27, .30), (.33, .30), (.22, .38), (.22, .34), (.26, .25),
             (.26, .21), (.30, .12), (.30, .08), (0, .50), (0, .46), (-.02, .25), (-.02, .21), (-.04, 0), (-.04, 0)],
}
POSE_ARR = {k: np.array(v, dtype=float) for k, v in POSES.items()}


def blend(a: str, b: str, t: float) -> np.ndarray:
    t = min(1.0, max(0.0, t))
    return (1 - t) * POSE_ARR[a] + t * POSE_ARR[b]


def make_track(tid: int, local_kp: np.ndarray, cx: float, foot_y: float, h: float, conf: float = 0.9,
               keypoints: bool = True, rng: Optional[np.random.Generator] = None, jitter: float = 0.0,
               cls_name: str = "person") -> Track:
    xs = cx + local_kp[:, 0] * h
    ys = foot_y - local_kp[:, 1] * h
    if rng is not None and jitter > 0:
        xs = xs + rng.normal(0, jitter * h, size=17)
        ys = ys + rng.normal(0, jitter * h, size=17)
    bbox = (float(xs.min() - 0.04 * h), float(ys.min() - 0.03 * h), float(xs.max() + 0.04 * h), float(ys.max() + 0.01 * h))
    kp = np.stack([xs, ys, np.full(17, conf)], axis=1) if keypoints else None
    return Track(id=tid, bbox=bbox, conf=0.9, keypoints=kp, cls_name=cls_name)


# keyframe: (t, pose, cx, foot_y)  -> interpolasi pose dan posisi antar keyframe
Keyframe = tuple


def person_at(keys: list, t: float, h: float, tid: int = 1, **kw) -> Optional[Track]:
    """Track pada waktu t dari daftar keyframe, None jika di luar rentang."""
    if t < keys[0][0] or t > keys[-1][0]:
        return None
    for (t0, p0, x0, y0), (t1, p1, x1, y1) in zip(keys[:-1], keys[1:]):
        if t0 <= t <= t1:
            a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            return make_track(tid, blend(p0, p1, a), x0 + (x1 - x0) * a, y0 + (y1 - y0) * a, h, **kw)
    t_last = keys[-1]
    return make_track(tid, POSE_ARR[t_last[1]], t_last[2], t_last[3], h, **kw)


def run_timeline(module, make_tracks: Callable, duration: float, fps: float = 6.0, camera_id: str = "cam",
                 size: tuple = (640, 480), t0: float = 0.0, wall0: float = 1_700_000_000.0) -> list:
    """Jalankan module.process() pada timeline sintetis. make_tracks(t) -> list[Track].

    Mengembalikan daftar (t, Event).
    """
    from ..types import Frame, FrameContext

    out = []
    n = int(duration * fps)
    for i in range(n + 1):
        t = t0 + i / fps
        frame = Frame(image=np.zeros((size[1], size[0], 3), np.uint8), ts=t, idx=i, wall=wall0 + t)
        ctx = FrameContext(camera_id=camera_id, frame=frame, tracks=make_tracks(t), width=size[0], height=size[1])
        for ev in module.process(ctx):
            out.append((t, ev))
    return out


def draw_track(img: np.ndarray, tr: Track, color=(0, 200, 0)) -> None:
    import cv2

    x1, y1, x2, y2 = [int(v) for v in tr.bbox]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    if tr.keypoints is not None:
        pairs = [(5, 6), (5, 11), (6, 12), (11, 12), (5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14),
                 (14, 16), (0, 5), (0, 6)]
        for a, b in pairs:
            pa, pb = tr.keypoints[a], tr.keypoints[b]
            cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), color, 2)
    cv2.putText(img, f"id{tr.id}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def box_track(tid: int, cx: float, foot_y: float, h: float, w: Optional[float] = None, cls_name: str = "person",
              conf: float = 0.85) -> Track:
    """Track tanpa keypoint (deteksi person biasa) untuk uji safe zone."""
    w = w if w is not None else 0.4 * h
    return Track(id=tid, bbox=(cx - w / 2, foot_y - h, cx + w / 2, foot_y), conf=conf, keypoints=None, cls_name=cls_name)


def path_at(points: list, t: float) -> Optional[tuple]:
    """points: [(t, x, foot_y)] -> posisi interpolasi linear, None di luar rentang."""
    if t < points[0][0] or t > points[-1][0]:
        return None
    for (t0, x0, y0), (t1, x1, y1) in zip(points[:-1], points[1:]):
        if t0 <= t <= t1:
            a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            return x0 + (x1 - x0) * a, y0 + (y1 - y0) * a
    return points[-1][1], points[-1][2]
