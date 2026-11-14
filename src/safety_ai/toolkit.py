"""Fungsi pendukung untuk tools/ (dipisah agar bisa diuji tanpa GUI/kamera)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from .sources.sources import _rtsp_opener, _webcam_opener


# ---------------------------------------------------------------------- ROI
def parse_polygon(text: str) -> list:
    """'0.1,0.9;0.1,0.3;0.9,0.3' -> [[0.1, 0.9], [0.1, 0.3], [0.9, 0.3]] (divalidasi)."""
    pts = [[float(v) for v in p.split(",")] for p in text.split(";") if p.strip()]
    if len(pts) < 3 or any(len(p) != 2 or not all(0.0 <= c <= 1.0 for c in p) for p in pts):
        raise ValueError("poligon butuh >= 3 titik 'x,y' ternormalisasi 0..1, dipisah ';'")
    return pts


def pixels_to_norm(points_px: list, width: int, height: int) -> list:
    return [[round(x / width, 4), round(y / height, 4)] for x, y in points_px]


def write_roi(out_path: str, camera_id: str, polygon: list) -> str:
    """.json -> tulis {"safe_polygon": ...}; .yaml/.yml -> perbarui roi.safe_polygon kamera tsb (komentar YAML hilang)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".json":
        out.write_text(json.dumps({"camera_id": camera_id, "safe_polygon": polygon}, indent=2), encoding="utf-8")
        return str(out)
    data = yaml.safe_load(out.read_text(encoding="utf-8")) if out.exists() else {"cameras": []}
    data = data or {"cameras": []}
    cams = data.setdefault("cameras", [])
    cam = next((c for c in cams if c.get("id") == camera_id), None)
    if cam is None:
        raise KeyError(f"kamera '{camera_id}' tidak ada di {out_path}")
    roi = cam.setdefault("roi", {})
    roi["safe_polygon"] = polygon
    roi.pop("file", None)
    out.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return str(out)


def render_preview(image: np.ndarray, polygon: list, margin_frac: float = 0.02, warn_frac: float = 0.05) -> np.ndarray:
    """Gambar poligon + zona peringatan (kuning) + margin pelanggaran (merah) di atas frame."""
    h, w = image.shape[:2]
    img = image.copy()
    poly = np.array([[int(x * w), int(y * h)] for x, y in polygon], np.int32)
    overlay = img.copy()
    cv2.fillPoly(overlay, [poly], (0, 255, 0))
    img = cv2.addWeighted(overlay, 0.18, img, 0.82, 0)
    cv2.polylines(img, [poly], True, (0, 255, 0), 2)
    # perkiraan band peringatan/pelanggaran dari poligon yang dierosi/didilasi pada mask
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    for frac, color in ((warn_frac, (0, 255, 255)), (-margin_frac, (0, 0, 255))):
        k = max(1, int(abs(frac) * h))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        m = cv2.erode(mask, kernel) if frac > 0 else cv2.dilate(mask, kernel)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, color, 1)
    return img


# ---------------------------------------------------------------------- sumber ad-hoc
def open_capture(source: str):
    """source: angka = webcam, rtsp://... = RTSP, selain itu = berkas video."""
    if source.isdigit():
        return _webcam_opener(int(source))()
    if source.lower().startswith("rtsp://"):
        return _rtsp_opener(source, "tcp")()
    return cv2.VideoCapture(source)


def grab_first_frame(source: str, skip: int = 5) -> np.ndarray:
    cap = open_capture(source)
    if not cap.isOpened():
        raise RuntimeError(f"sumber tidak bisa dibuka: {source}")
    frame = None
    for _ in range(max(1, skip)):
        ok, img = cap.read()
        if ok:
            frame = img
    cap.release()
    if frame is None:
        raise RuntimeError("tidak ada frame yang terbaca")
    return frame


def record_clip(source: str, seconds: float, out_path: str, fps: Optional[float] = None) -> dict:
    cap = open_capture(source)
    if not cap.isOpened():
        raise RuntimeError(f"sumber tidak bisa dibuka: {source}")
    native = cap.get(cv2.CAP_PROP_FPS)
    fps = fps or (native if native and 1 < native < 120 else 15.0)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer, n = None, 0
    limit = int(seconds * fps)
    while n < limit:
        ok, img = cap.read()
        if not ok:
            break
        if writer is None:
            h, w = img.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*("MJPG" if out_path.lower().endswith(".avi") else "mp4v"))
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        writer.write(img)
        n += 1
    cap.release()
    if writer is not None:
        writer.release()
    return {"frames": n, "fps": fps, "seconds": round(n / fps, 2), "path": out_path}


# ---------------------------------------------------------------------- benchmark
def percentiles(values: list, ps=(50, 95)) -> dict:
    if not values:
        return {f"p{p}": None for p in ps}
    return {f"p{p}": float(np.percentile(values, p)) for p in ps}


def recommend_fps(latency_ms: float, n_cameras: int, budget: float = 0.6) -> dict:
    """Perkiraan kasar FPS per kamera yang aman. Asumsi: inferensi semua kamera berbagi CPU secara bergantian,
    dan CPU hanya boleh sibuk `budget` (default 60%) dari waktu. Ukur ulang di komputer sekolah."""
    if latency_ms <= 0 or n_cameras <= 0:
        raise ValueError("latency_ms dan n_cameras harus > 0")
    total = round(budget / (latency_ms / 1000.0), 2)
    per_cam = round(total / n_cameras, 2)
    if per_cam >= 6:
        profile = "normal"
    elif per_cam >= 3:
        profile = "low_cpu"
    else:
        profile = "terlalu_berat"
    return {"max_total_fps": total, "fps_per_camera": per_cam, "profile": profile}
