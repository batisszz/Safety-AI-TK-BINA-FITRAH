"""Kesehatan kamera: drift (kamera bergeser -> ROI salah) dan watchdog stream (offline)."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..types import Event


def _prep(image: np.ndarray, width: int = 320) -> np.ndarray:
    h, w = image.shape[:2]
    small = cv2.resize(image, (width, max(1, int(h * width / w))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    return cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)


class DriftDetector:
    """Bandingkan frame referensi dengan frame sekarang. Geser > thresh_frac lebar frame (2 cek berturut-turut)
    dianggap bergeser. Respons korelasi rendah (adegan berubah total, mis. siang/malam) diabaikan."""

    def __init__(self, reference: np.ndarray, thresh_frac: float = 0.03, interval_s: float = 600.0,
                 min_response: float = 0.1, confirm: int = 2):
        self.ref = _prep(reference)
        self.window = cv2.createHanningWindow((self.ref.shape[1], self.ref.shape[0]), cv2.CV_32F)
        self.thresh, self.interval = thresh_frac, interval_s
        self.min_response, self.confirm = min_response, confirm
        self._last: Optional[float] = None
        self._bad = 0
        self.drifted = False

    def measure(self, image: np.ndarray) -> tuple:
        cur = _prep(image)
        (dx, dy), response = cv2.phaseCorrelate(self.ref, cur, self.window)
        return float(np.hypot(dx, dy) / self.ref.shape[1]), float(response)

    def check(self, image: np.ndarray, ts: float) -> Optional[dict]:
        """Panggil tiap frame; baru mengukur tiap interval_s. Mengembalikan dict saat status drift BERUBAH."""
        if self._last is not None and ts - self._last < self.interval:
            return None
        self._last = ts
        shift, response = self.measure(image)
        if response < self.min_response:
            return None
        if shift > self.thresh:
            self._bad += 1
            if self._bad >= self.confirm and not self.drifted:
                self.drifted = True
                return {"state": "camera_drift", "shift_frac": round(shift, 4)}
        else:
            self._bad = 0
            if self.drifted:
                self.drifted = False
                return {"state": "camera_aligned", "shift_frac": round(shift, 4)}
        return None


def health_event(camera_id: str, state: str, ts: float, wall: float, extra: Optional[dict] = None) -> Event:
    meta = {"state": state}
    if extra:
        meta.update(extra)
    level = 1 if state in ("camera_offline", "camera_drift") else 0
    return Event(alert_type="system_health", level=level, camera_id=camera_id, track_id=None, confidence=1.0,
                 meta=meta, stream_ts=ts, wall=wall)
