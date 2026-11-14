"""Geometri safe zone: poligon ternormalisasi, jarak bertanda, motion gate."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def poly_to_px(poly_norm, width: int, height: int) -> np.ndarray:
    """Poligon ternormalisasi (0..1) -> piksel float32 (N, 2)."""
    arr = np.asarray(poly_norm, dtype=np.float32).reshape(-1, 2).copy()
    arr[:, 0] *= width
    arr[:, 1] *= height
    return arr


def signed_distance(poly_px: np.ndarray, point) -> float:
    """Jarak bertanda titik ke poligon dalam piksel: > 0 di dalam, < 0 di luar, 0 tepat di garis."""
    return float(cv2.pointPolygonTest(poly_px.reshape(-1, 1, 2), (float(point[0]), float(point[1])), True))


def polygon_bbox(poly_norm, margin_frac: float, width: int, height: int) -> tuple:
    """Kotak piksel (x1, y1, x2, y2) yang melingkupi poligon + margin, dipotong ke batas frame."""
    arr = np.asarray(poly_norm, dtype=float)
    x1, y1 = arr.min(axis=0)
    x2, y2 = arr.max(axis=0)
    x1, y1 = max(0.0, x1 - margin_frac), max(0.0, y1 - margin_frac)
    x2, y2 = min(1.0, x2 + margin_frac), min(1.0, y2 + margin_frac)
    return (int(x1 * width), int(y1 * height), int(round(x2 * width)), int(round(y2 * height)))


class MotionGate:
    """Cek frame difference yang murah. Inferensi hanya jika ada gerakan, ditambah heartbeat berkala
    agar anak yang berdiri diam di dekat batas tetap terpantau."""

    def __init__(self, thresh: int = 25, min_area_frac: float = 0.002, heartbeat_s: float = 1.0,
                 region: Optional[tuple] = None, scale: int = 160):
        self.thresh = thresh
        self.min_area_frac = min_area_frac
        self.heartbeat_s = heartbeat_s
        self.region = region
        self.scale = scale
        self._prev: Optional[np.ndarray] = None
        self._last_infer: Optional[float] = None

    def _prepare(self, image: np.ndarray) -> np.ndarray:
        if self.region is not None:
            x1, y1, x2, y2 = self.region
            image = image[y1:y2, x1:x2]
        h, w = image.shape[:2]
        new_w = self.scale
        new_h = max(1, int(h * new_w / max(w, 1)))
        small = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
        return cv2.GaussianBlur(gray, (5, 5), 0)

    def check(self, image: np.ndarray, ts: float) -> bool:
        cur = self._prepare(image)
        moving = True
        if self._prev is not None and self._prev.shape == cur.shape:
            diff = cv2.absdiff(cur, self._prev)
            moving = float((diff > self.thresh).mean()) >= self.min_area_frac
        self._prev = cur
        if self._last_infer is None or moving or (ts - self._last_infer) >= self.heartbeat_s:
            self._last_infer = ts
            return True
        return False
