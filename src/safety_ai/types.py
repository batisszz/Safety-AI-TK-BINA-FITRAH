"""Tipe data inti yang dipakai bersama oleh semua modul.

Modul tim lain (aktivitas, presensi) cukup memakai FrameContext dan Track ini,
sehingga satu tahap deteksi + tracking bisa dibagi bersama.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

# Indeks keypoint COCO-17 (format YOLOv8-Pose)
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
L_ANKLE, R_ANKLE = 15, 16


@dataclass
class Frame:
    """Satu frame video.

    ts   : detik sejak stream dimulai (untuk file = idx / fps, bukan jam dinding),
           dipakai oleh state machine agar hasil sama pada kecepatan putar berapa pun.
    wall : epoch detik, hanya untuk penanda waktu pada event.
    """

    image: np.ndarray
    ts: float
    idx: int
    wall: float = 0.0


@dataclass
class Track:
    """Satu objek terlacak (orang) pada satu frame, koordinat piksel frame penuh."""

    id: int
    bbox: tuple  # (x1, y1, x2, y2)
    conf: float = 1.0
    keypoints: Optional[np.ndarray] = None  # (17, 3): x, y, conf
    cls_name: str = "person"  # "person" | "child" | "adult"

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def center(self) -> tuple:
        return ((self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0)

    @property
    def foot_point(self) -> tuple:
        """Titik tengah paling bawah bbox (posisi kaki di lantai)."""
        return ((self.bbox[0] + self.bbox[2]) / 2.0, float(self.bbox[3]))


@dataclass
class FrameContext:
    camera_id: str
    frame: Frame
    tracks: list
    width: int
    height: int
    source_name: str = ""


@dataclass
class Event:
    """Event keluaran modul. Field yang diserialisasi = kontrak integrasi (docs/INTEGRATION.md)."""

    alert_type: str  # fall_suspected | fall_confirmed | zone_breach | zone_warning | system_health
    level: int  # 0 = hanya log, 1 = notifikasi guru, 2 = alarm admin
    camera_id: str
    track_id: Optional[int] = None
    confidence: float = 0.0
    meta: dict = field(default_factory=dict)
    stream_ts: float = 0.0
    wall: float = 0.0
    # diisi oleh AlertManager
    event_id: str = ""
    timestamp: str = ""
    snapshot_path: Optional[str] = None
    # hanya untuk menggambar snapshot, tidak diserialisasi
    bbox: Optional[tuple] = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        meta = dict(self.meta)
        meta.setdefault("stream_ts", round(float(self.stream_ts), 3))
        return {
            "event_id": self.event_id,
            "camera_id": self.camera_id,
            "alert_type": self.alert_type,
            "level": int(self.level),
            "track_id": self.track_id,
            "confidence": round(float(self.confidence), 3),
            "timestamp": self.timestamp,
            "snapshot_path": self.snapshot_path,
            "meta": meta,
        }
