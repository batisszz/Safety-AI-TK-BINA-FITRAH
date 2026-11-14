"""Backend deteksi + tracking. Ultralytics di-import malas (lazy) agar logika modul bisa diuji tanpa torch.

Kontrak: infer(image, ts, offset) -> list[Track] dalam koordinat FRAME PENUH (offset = pojok crop).
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..config import InferConfig
from ..types import Track

# ByteTrack untuk CPU lemah. track_buffer dalam FRAME: pada 3-6 FPS, 15 frame = 2,5-5 detik.
TRACKER_YAML = """tracker_type: bytetrack
track_high_thresh: 0.25
track_low_thresh: 0.1
new_track_thresh: 0.25
track_buffer: 15
match_thresh: 0.8
fuse_score: True
"""


def ensure_tracker_yaml(path: Optional[str] = None) -> str:
    if path and Path(path).exists():
        return path
    out = Path(tempfile.gettempdir()) / "safety_ai_bytetrack_cpu.yaml"
    out.write_text(TRACKER_YAML, encoding="utf-8")
    return str(out)


def resolve_model(name: str, models_dir: str = "models") -> str:
    """Path berkas/folder jika ada, kalau tidak models/<name>.pt, kalau tidak nama mentah (Ultralytics mengunduh)."""
    for cand in (Path(name), Path(models_dir) / name, Path(models_dir) / f"{name}.pt"):
        if cand.exists():
            return str(cand)
    return name if name.endswith((".pt", ".onnx")) else f"{name}.pt"


def _np(x):
    if x is None:
        return None
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def parse_result(res, offset: tuple = (0, 0), names: Optional[dict] = None) -> list:
    """Ubah satu hasil Ultralytics (Results) menjadi list[Track]. Track tanpa ID (tracker belum aktif) dilewati."""
    boxes = getattr(res, "boxes", None)
    if boxes is None or boxes.id is None:
        return []
    xyxy, conf, ids, cls = _np(boxes.xyxy), _np(boxes.conf), _np(boxes.id), _np(boxes.cls)
    kps = getattr(res, "keypoints", None)
    kp_data = _np(kps.data) if kps is not None and getattr(kps, "data", None) is not None else None
    ox, oy = offset
    tracks = []
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
        kp = None
        if kp_data is not None and len(kp_data) > i and kp_data[i].shape[0] >= 17:
            kp = kp_data[i].astype(float).copy()
            kp[:, 0] += ox
            kp[:, 1] += oy
        label = (names or {}).get(int(cls[i]), "person") if cls is not None else "person"
        if label not in ("child", "adult"):
            label = "person"
        tracks.append(Track(id=int(ids[i]), bbox=(x1 + ox, y1 + oy, x2 + ox, y2 + oy), conf=float(conf[i]),
                            keypoints=kp, cls_name=label))
    return tracks


class YoloBackend:
    def __init__(self, cfg: InferConfig, models_dir: str = "models"):
        try:
            from ultralytics import YOLO  # lazy
        except ImportError as e:
            raise RuntimeError("Ultralytics belum terpasang. Jalankan: pip install ultralytics "
                               "(untuk uji tanpa model gunakan tools/simulate.py)") from e

        self.cfg = cfg
        if cfg.threads:
            os.environ.setdefault("OMP_NUM_THREADS", str(cfg.threads))
            try:
                import torch
                torch.set_num_threads(cfg.threads)
            except Exception:
                pass
        self.model_path = resolve_model(cfg.model, models_dir)
        self.model = YOLO(self.model_path, task=cfg.resolved_task())
        self.tracker = ensure_tracker_yaml(cfg.tracker)
        self.names = getattr(self.model, "names", None) or {}

    def infer(self, image: np.ndarray, ts: float, offset: tuple = (0, 0)) -> list:
        c = self.cfg
        res = self.model.track(image, persist=True, tracker=self.tracker, imgsz=c.imgsz, conf=c.conf, iou=c.iou,
                               device=c.device, classes=c.classes, verbose=False)[0]
        return parse_result(res, offset, self.names)


class ScriptedBackend:
    """Backend palsu untuk uji dan simulasi: fn(ts) -> list[Track] (koordinat frame penuh)."""

    def __init__(self, fn: Callable):
        self.fn = fn
        self.calls = 0

    def infer(self, image: np.ndarray, ts: float, offset: tuple = (0, 0)) -> list:
        self.calls += 1
        return self.fn(ts)
