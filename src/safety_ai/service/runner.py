"""Runner: satu pipeline per kamera (satu proses per kamera bila lebih dari satu).

    python -m safety_ai.service.runner --config configs/default.yaml --camera cam_kelas_a --show
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..config import AppConfig, CameraConfig, SourceConfig, load_config
from ..events.manager import AlertManager
from ..events.sinks import build_sink
from ..modules.fall.module import FallModule
from ..modules.safezone.module import SafeZoneModule
from ..types import Frame, FrameContext
from .health import DriftDetector, health_event


def build_modules(cam: CameraConfig, app: AppConfig) -> list:
    mods = []
    for name in cam.modules:
        if name == "fall":
            mods.append(FallModule(cam.id, cam.fall))
        elif name == "safezone":
            mods.append(SafeZoneModule(cam.id, cam.roi, timezone=app.timezone))
    return mods


def union_crop(mods: list, width: int, height: int):
    """Crop hanya jika SEMUA modul pada kamera ini meminta crop; hasilnya gabungan kotak."""
    regions = [m.crop_region(width, height) for m in mods]
    if not regions or any(r is None for r in regions):
        return None
    return (min(r[0] for r in regions), min(r[1] for r in regions), max(r[2] for r in regions),
            max(r[3] for r in regions))


class CameraPipeline:
    def __init__(self, cam: CameraConfig, source, backend, manager: AlertManager, modules: list,
                 drift: Optional[DriftDetector] = None, show: bool = False):
        self.cam, self.source, self.backend, self.manager = cam, source, backend, manager
        self.modules, self.drift, self.show = modules, drift, show
        self.offline = False
        self.stats = {"camera_id": cam.id, "frames": 0, "inferred": 0, "skipped_by_gate": 0, "infer_ms_sum": 0.0,
                      "infer_ms_max": 0.0, "process_ms_sum": 0.0, "stream_seconds": 0.0, "wall_seconds": 0.0}
        self._t_start = time.time()

    # ------------------------------------------------------------------ satu frame
    def handle_frame(self, frame: Frame) -> list:
        st = self.stats
        st["frames"] += 1
        st["stream_seconds"] = frame.ts
        h, w = frame.image.shape[:2]
        sent: list = []

        if self.offline:
            self.offline = False
            sent += self.manager.handle([health_event(self.cam.id, "camera_online", frame.ts, frame.wall)], frame)
        if self.drift is not None:
            d = self.drift.check(frame.image, frame.ts)
            if d:
                state = d.pop("state")
                sent += self.manager.handle([health_event(self.cam.id, state, frame.ts, frame.wall, d)], frame)

        t0 = time.perf_counter()
        wants = [m.should_infer(frame) for m in self.modules]   # panggil semua agar gate selalu ter-update
        if not any(wants):
            st["skipped_by_gate"] += 1
            return sent

        crop = union_crop(self.modules, w, h)
        if crop:
            x1, y1, x2, y2 = crop
            image, offset = frame.image[y1:y2, x1:x2], (x1, y1)
        else:
            image, offset = frame.image, (0, 0)
        t1 = time.perf_counter()
        tracks = self.backend.infer(image, frame.ts, offset)
        infer_ms = (time.perf_counter() - t1) * 1000.0
        st["inferred"] += 1
        st["infer_ms_sum"] += infer_ms
        st["infer_ms_max"] = max(st["infer_ms_max"], infer_ms)

        ctx = FrameContext(camera_id=self.cam.id, frame=frame, tracks=tracks, width=w, height=h,
                           source_name=getattr(self.source, "name", ""))
        events = []
        for m in self.modules:
            events += m.process(ctx)
        sent += self.manager.handle(events, frame)
        st["process_ms_sum"] += (time.perf_counter() - t0) * 1000.0
        if self.show:
            self._show(frame, tracks)
        return sent

    # ------------------------------------------------------------------ loop
    def run(self, stop=None, max_seconds: Optional[float] = None) -> dict:
        stall_s = self.cam.source.stall_s
        try:
            while stop is None or not stop.is_set():
                frame = self.source.read()
                if frame is None:
                    if getattr(self.source, "finished", False):
                        break
                    if (self.source.is_live and not self.offline and self.source.seconds_since_frame() > stall_s):
                        self.offline = True
                        now = time.time()
                        self.manager.handle([health_event(self.cam.id, "camera_offline", 0.0, now)], None)
                    continue
                self.handle_frame(frame)
                if max_seconds is not None and frame.ts >= max_seconds:
                    break
                if self.show and (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                    break
        finally:
            self.stats["wall_seconds"] = round(time.time() - self._t_start, 2)
            self.source.close()
            self.manager.sink.close()
            if self.show:
                cv2.destroyAllWindows()
        return self.summary()

    def summary(self) -> dict:
        s = dict(self.stats)
        n = max(1, s["inferred"])
        s["infer_ms_avg"] = round(s["infer_ms_sum"] / n, 2)
        s["effective_fps"] = round(s["frames"] / max(s["wall_seconds"], 1e-6), 2)
        s["alerts"] = dict(self.manager.stats)
        for k in ("infer_ms_sum", "process_ms_sum"):
            s.pop(k, None)
        return s

    def _show(self, frame: Frame, tracks: list) -> None:
        img = frame.image.copy()
        for m in self.modules:
            if m.name == "safezone":
                pts = np.array([[int(x * img.shape[1]), int(y * img.shape[0])] for x, y in m.p.safe_polygon], np.int32)
                cv2.polylines(img, [pts], True, (0, 255, 0), 2)
        for t in tracks:
            x1, y1, x2, y2 = [int(v) for v in t.bbox]
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 200, 0), 2)
            label = f"id{t.id}"
            for m in self.modules:
                st = m.states.get(t.id)
                if st is not None:
                    label += " " + (st.fsm.state if m.name == "fall" else st.state)
            cv2.putText(img, label, (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
        cv2.imshow(self.cam.id, img)


# ---------------------------------------------------------------------- pabrik
def build_pipeline(cam: CameraConfig, app: AppConfig, run: str, backend=None, source=None, show: bool = False,
                   models_dir: str = "models") -> CameraPipeline:
    from ..sources.sources import open_source

    run_dir = Path(app.output_dir) / run
    run_dir.mkdir(parents=True, exist_ok=True)
    source = source or open_source(cam.source, cam.infer.fps)
    if backend is None:
        from ..inference.backend import YoloBackend
        backend = YoloBackend(cam.infer, models_dir)
    sink = build_sink(app.sinks, cam.id, run)
    # jalur jsonl/outbox relatif terhadap output_dir bila belum diawali output_dir
    manager = AlertManager(app.alerts, sink, app.timezone, base_dir=str(run_dir), source_name=getattr(source, "name", ""))
    drift = None
    if cam.drift.enabled and cam.drift.reference and Path(cam.drift.reference).exists():
        drift = DriftDetector(cv2.imread(cam.drift.reference), cam.drift.thresh_frac, cam.drift.interval_s)
    return CameraPipeline(cam, source, backend, manager, build_modules(cam, app), drift=drift, show=show)


def run_camera(config_path: str, camera_id: str, run: str, opts: dict) -> dict:
    app = load_config(config_path)
    cam = app.camera(camera_id)
    if opts.get("source"):
        cam = cam.model_copy(update={"source": SourceConfig(type="file", path=opts["source"],
                                                           realtime=bool(opts.get("realtime")))})
    elif opts.get("realtime") and cam.source.type == "file":
        cam = cam.model_copy(update={"source": cam.source.model_copy(update={"realtime": True})})
    pipe = build_pipeline(cam, app, run, show=bool(opts.get("show")), models_dir=opts.get("models_dir", "models"))
    summary = pipe.run(max_seconds=opts.get("max_seconds"))
    out = Path(app.output_dir) / run / f"stats_{camera_id}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[INFO] {camera_id} selesai: {json.dumps(summary)}", flush=True)
    return summary


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="Runner modul AI Safety (Fall + Safe Zone)")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--camera", action="append", help="id kamera (boleh berulang). Kosong = semua kamera")
    ap.add_argument("--source", help="ganti sumber menjadi file video ini (hanya untuk satu kamera)")
    ap.add_argument("--run", default=datetime.now().strftime("%Y%m%d_%H%M%S"), help="nama folder keluaran")
    ap.add_argument("--show", action="store_true", help="tampilkan jendela (butuh opencv-python, bukan headless)")
    ap.add_argument("--realtime", action="store_true", help="file diputar secepat waktu asli")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--models-dir", default="models")
    args = ap.parse_args(argv)

    app = load_config(args.config)
    ids = args.camera or [c.id for c in app.cameras]
    for i in ids:
        app.camera(i)  # validasi awal
    if args.source and len(ids) != 1:
        ap.error("--source hanya boleh dengan tepat satu --camera")
    opts = {"source": args.source, "show": args.show, "realtime": args.realtime, "max_seconds": args.max_seconds,
            "models_dir": args.models_dir}
    if len(ids) == 1:
        run_camera(args.config, ids[0], args.run, opts)
        return 0
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=run_camera, args=(args.config, i, args.run, opts), name=f"cam-{i}") for i in ids]
    try:
        for p in procs:
            p.start()
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        for p in procs:
            p.terminate()
    return 0 if all(p.exitcode == 0 for p in procs) else 1


if __name__ == "__main__":
    sys.exit(main())
