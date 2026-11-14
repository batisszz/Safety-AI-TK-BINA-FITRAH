"""Benchmark inferensi di mesin INI (jalankan di komputer sekolah untuk angka resmi).

  python tools/benchmark.py --model yolov8n-pose --imgsz 480 --device cpu --threads 2 --frames 60
  python tools/benchmark.py --model models/yolov8n-pose_openvino_model --imgsz 320 --cameras 2

Catatan: laptop pengembangan (LOQ) jauh lebih kencang daripada laptop sekolah. Batasi --threads agar mendekati
kondisi sekolah, dan perlakukan angka LOQ sebagai batas atas.
"""
import _bootstrap  # noqa: F401
import argparse
import json
import os
import platform
import time

import numpy as np

from safety_ai.toolkit import grab_first_frame, percentiles, recommend_fps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolov8n-pose")
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--source", default=None, help="video/webcam/rtsp sebagai masukan (default: gambar sintetis)")
    ap.add_argument("--track", action="store_true", help="ukur model.track (dengan ByteTrack), bukan predict")
    ap.add_argument("--cameras", type=int, default=2, help="jumlah kamera yang akan dilayani")
    ap.add_argument("--budget", type=float, default=0.6, help="batas pemakaian CPU (0-1)")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--out", help="simpan hasil JSON")
    a = ap.parse_args()

    if a.threads:
        os.environ["OMP_NUM_THREADS"] = str(a.threads)
    import psutil
    from safety_ai.config import InferConfig
    from safety_ai.inference.backend import ensure_tracker_yaml, resolve_model
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit('Ultralytics belum terpasang. Jalankan: pip install ultralytics')

    if a.threads and not a.model.endswith(("_openvino_model", ".onnx")):
        try:
            import torch
            torch.set_num_threads(a.threads)
        except Exception:
            pass
    path = resolve_model(a.model, a.models_dir)
    task = InferConfig(model=a.model).resolved_task()
    model = YOLO(path, task=task)
    if a.source:
        img = grab_first_frame(a.source)
    else:
        rng = np.random.default_rng(0)
        img = (rng.random((480, 640, 3)) * 255).astype(np.uint8)
    tracker = ensure_tracker_yaml(None)

    def once():
        if a.track:
            return model.track(img, persist=True, tracker=tracker, imgsz=a.imgsz, device=a.device, verbose=False)
        return model.predict(img, imgsz=a.imgsz, device=a.device, verbose=False)

    for _ in range(a.warmup):
        once()
    proc = psutil.Process()
    proc.cpu_percent(None)
    lat = []
    t0 = time.perf_counter()
    for _ in range(a.frames):
        s = time.perf_counter()
        once()
        lat.append((time.perf_counter() - s) * 1000.0)
    wall = time.perf_counter() - t0
    cpu_total = proc.cpu_percent(None) / max(psutil.cpu_count(logical=True), 1)
    pc = percentiles(lat)
    rec = recommend_fps(float(np.median(lat)), a.cameras, a.budget)
    result = {
        "mesin": {"cpu": platform.processor() or platform.machine(), "logical_cores": psutil.cpu_count(logical=True),
                  "ram_gb": round(psutil.virtual_memory().total / 1e9, 1), "os": platform.platform()},
        "model": path, "imgsz": a.imgsz, "device": a.device, "threads": a.threads, "mode": "track" if a.track else "predict",
        "input": a.source or "sintetis (noise): tidak merepresentasikan kerumitan adegan nyata",
        "frames": a.frames, "fps_inferensi": round(a.frames / wall, 2),
        "latensi_ms": {"rata2": round(float(np.mean(lat)), 1), "p50": round(pc["p50"], 1), "p95": round(pc["p95"], 1)},
        "cpu_proses_persen_dari_seluruh_cpu": round(cpu_total, 1), "rss_mb": round(proc.memory_info().rss / 1e6, 0),
        "rekomendasi": {"kameras": a.cameras, "budget_cpu": a.budget, **rec},
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
