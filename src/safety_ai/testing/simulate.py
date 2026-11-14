"""Simulasi end-to-end TANPA model dan TANPA kamera.

Membuat video sintetis (kerangka bergerak), lalu menjalankannya lewat runner yang SAMA dengan produksi
(FileSource -> modul -> AlertManager -> sink). Dipakai sebagai smoke test integrasi dan demo untuk tim lain.
Yang tidak dicakup: kualitas deteksi YOLO sungguhan (itu diukur dengan tools/benchmark.py dan evaluate.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from ..config import parse_config
from ..inference.backend import ScriptedBackend
from ..service.runner import build_pipeline
from .synthetic import box_track, draw_track, path_at, person_at

SIZE = (640, 480)
FPS_VIDEO = 25.0
GATE_POLY = [[0.1, 0.9], [0.1, 0.3], [0.9, 0.3], [0.9, 0.9]]
START = "2026-11-12T09:00:00+07:00"


def class_scenario():
    """Kelas: anak 1 jatuh lalu diam (L1 lalu L2), anak 2 berjalan, anak 3 duduk cepat (tidak boleh alert)."""
    k1 = [(0, "stand", 150, 400), (10, "stand", 160, 400), (10.5, "lie", 160, 400), (40, "lie", 160, 400)]
    k2 = [(0, "stand", 450, 400), (40, "stand", 520, 400)]
    k3 = [(0, "stand", 320, 430), (20, "stand", 320, 430), (20.4, "sit", 320, 430), (40, "sit", 320, 430)]

    def fn(ts):
        out = [person_at(k, ts, 200.0, tid=i + 1) for i, k in enumerate((k1, k2, k3))]
        return [t for t in out if t is not None]
    return fn, 40.0


def gate_scenario():
    """Gerbang: anak 1 keluar batas (warning lalu breach), dewasa 2 lewat, pejalan kaki 3 di jalan."""
    child = [(0, 320, 300), (8, 320, 160), (10, 320, 100), (30, 320, 60)]
    adult = [(0, 520, 300), (6, 520, 100), (30, 520, 60)]
    walker = [(0, 50, 90), (30, 600, 90)]

    def fn(ts):
        out = []
        for tid, pts, h, cls in ((1, child, 110.0, "person"), (2, adult, 190.0, "adult"), (3, walker, 110.0, "person")):
            pos = path_at(pts, ts)
            if pos:
                out.append(box_track(tid, pos[0], pos[1], h, cls_name=cls))
        return out
    return fn, 30.0


def render_video(path: Path, fn, duration: float, poly=None) -> None:
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS_VIDEO, SIZE)
    for i in range(int(duration * FPS_VIDEO)):
        ts = i / FPS_VIDEO
        img = np.full((SIZE[1], SIZE[0], 3), 70, np.uint8)
        if poly:
            pts = np.array([[int(x * SIZE[0]), int(y * SIZE[1])] for x, y in poly], np.int32)
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)
        for tr in fn(ts):
            draw_track(img, tr)
        cv2.putText(img, f"t={ts:5.1f}s", (8, SIZE[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        vw.write(img)
    vw.release()


def run_simulation(out_dir: str, with_video: bool = True) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cls_fn, cls_dur = class_scenario()
    gate_fn, gate_dur = gate_scenario()
    cls_video, gate_video = out / "sim_kelas.avi", out / "sim_gerbang.avi"
    render_video(cls_video, cls_fn, cls_dur)
    render_video(gate_video, gate_fn, gate_dur, poly=GATE_POLY)

    cfg = parse_config({
        "output_dir": str(out / "runs"),
        "sinks": [{"type": "console"}, {"type": "jsonl", "path": str(out / "runs/{run}/events_{camera}.jsonl")}],
        "cameras": [
            {"id": "cam_kelas_sim", "modules": ["fall"],
             "source": {"type": "file", "path": str(cls_video), "start_time": START},
             "infer": {"model": "yolov8n-pose", "profile": "normal"}},
            {"id": "cam_gerbang_sim", "modules": ["safezone"],
             "source": {"type": "file", "path": str(gate_video), "start_time": START},
             "infer": {"model": "yolov8n", "profile": "gate"},
             "roi": {"safe_polygon": GATE_POLY, "motion_gate": False}},
        ],
    })
    results = {}
    for cam, fn in ((cfg.camera("cam_kelas_sim"), cls_fn), (cfg.camera("cam_gerbang_sim"), gate_fn)):
        pipe = build_pipeline(cam, cfg, "sim", backend=ScriptedBackend(fn))
        results[cam.id] = pipe.run()
    events = []
    for p in sorted((out / "runs/sim").glob("events_*.jsonl")):
        events += [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return {"stats": results, "events": events, "dir": str(out / "runs/sim")}
