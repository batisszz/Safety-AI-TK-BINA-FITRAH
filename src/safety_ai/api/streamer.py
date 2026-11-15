"""Pengelola streaming video kamera dan integrasi pipeline AI Safety
untuk disajikan melalui endpoint API (MJPEG stream dan snapshot frame).
"""
from __future__ import annotations

import json
import logging
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Optional

import cv2
import numpy as np

from ..config import AppConfig, CameraConfig, SourceConfig
from ..events.manager import AlertManager
from ..events.sinks import build_sink
from ..inference.backend import ScriptedBackend, YoloBackend
from ..modules.fall.module import FallModule
from ..modules.safezone.module import SafeZoneModule
from ..service.health import DriftDetector, health_event
from ..service.runner import build_modules, union_crop
from ..sources.sources import FileSource, open_source
from ..testing.simulate import class_scenario, gate_scenario
from ..types import Frame, FrameContext
from .alerts import AlertStore, BroadcasterSink, EventBroadcaster
from .annotator import Annotator

logger = logging.getLogger("safety_ai.streamer")


class CameraStreamer:
    """Mengelola satu kamera dalam background thread: membaca frame, inferensi YOLO,

    evaluasi logika AI (fall/safezone), anotasi visual, dan penyediaan buffer MJPEG.
    """

    def __init__(self, cam: CameraConfig, app: AppConfig, alert_store: AlertStore,
                 broadcaster: EventBroadcaster, run_name: str = "web_stream",
                 models_dir: str = "models", force_file_loop: bool = True):
        self.cam = cam
        self.app = app
        self.alert_store = alert_store
        self.broadcaster = broadcaster
        self.run_name = run_name
        self.models_dir = models_dir
        self.force_file_loop = force_file_loop

        self.modules = build_modules(cam, app)
        self.fall_module: Optional[FallModule] = next((m for m in self.modules if isinstance(m, FallModule)), None)
        self.safezone_module: Optional[SafeZoneModule] = next((m for m in self.modules if isinstance(m, SafeZoneModule)), None)

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._latest_raw: Optional[np.ndarray] = None
        self._latest_tracks: list = []
        self._frame_count = 0
        self._fps_actual = 0.0
        self._infer_latency_ms = 0.0
        self._status_text = "NORMAL"
        self._last_frame_time = 0.0
        self._is_running = False

        self._init_source_and_backend()
        self._init_manager()

    def _init_source_and_backend(self) -> None:
        """Inisialisasi sumber video dan model inferensi dengan toleransi kegagalan."""
        src_cfg = self.cam.source

        if src_cfg.type == "file" and src_cfg.path:
            p = Path(src_cfg.path)
            if not p.exists():
                cands = [Path("data/raw") / p.name, Path("data") / p.name]
                for c in cands:
                    if c.exists():
                        src_cfg = src_cfg.model_copy(update={"path": str(c)})
                        break

        # Cek ketersediaan webcam secara aman dan cepat
        if src_cfg.type == "webcam":
            opened = False
            try:
                backend_flag = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
                test_cap = cv2.VideoCapture(src_cfg.index, backend_flag)
                opened = test_cap.isOpened()
                test_cap.release()
            except Exception:
                opened = False

            if not opened:
                fallback_path = "data/raw/uji01.mp4" if "fall" in self.cam.modules else "data/raw/simulasi_outdoor_anak.mp4"
                if Path(fallback_path).exists():
                    logger.warning(
                        f"Webcam index {src_cfg.index} tidak tersedia untuk {self.cam.id}. "
                        f"Fallback otomatis ke video uji: {fallback_path}"
                    )
                    src_cfg = SourceConfig(type="file", path=fallback_path, loop=True, realtime=True)

        if src_cfg.type == "file" and self.force_file_loop:
            src_cfg = src_cfg.model_copy(update={"loop": True, "realtime": True})

        try:
            self.source = open_source(src_cfg, self.cam.infer.fps)
        except Exception as e:
            logger.error(f"Gagal membuka source {src_cfg}: {e}. Memakai synthetic generator.")
            if "fall" in self.cam.modules:
                fn, dur = class_scenario()
            else:
                fn, dur = gate_scenario()
            self.source = None
            self._synthetic_fn = fn

        # Inisialisasi Backend YOLO
        try:
            self.backend = YoloBackend(self.cam.infer, self.models_dir)
        except Exception as e:
            logger.warning(f"Gagal memuat YoloBackend untuk {self.cam.id}: {e}. Menggunakan ScriptedBackend.")
            if "fall" in self.cam.modules:
                fn, _ = class_scenario()
            else:
                fn, _ = gate_scenario()
            self.backend = ScriptedBackend(fn)

    def _init_manager(self) -> None:
        """Inisialisasi AlertManager dengan sink gabungan Broadcaster + File."""
        run_dir = Path(self.app.output_dir) / self.run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        file_sink = build_sink(self.app.sinks, self.cam.id, self.run_name)
        combined_sink = BroadcasterSink(self.alert_store, self.broadcaster, file_sink)

        source_name = getattr(self.source, "name", "stream")
        self.manager = AlertManager(
            self.app.alerts,
            combined_sink,
            self.app.timezone,
            base_dir=str(run_dir),
            source_name=source_name
        )

    def start(self) -> None:
        """Mulai background processing loop."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker_loop, name=f"stream-{self.cam.id}", daemon=True)
        self._thread.start()
        self._is_running = True

    def stop(self) -> None:
        """Hentikan background processing loop."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._is_running = False
        if hasattr(self, "source") and self.source:
            try:
                self.source.close()
            except Exception:
                pass
        if hasattr(self, "manager") and self.manager:
            try:
                self.manager.sink.close()
            except Exception:
                pass

        # Simpan ringkasan statistik ke outputs/<run>/stats_<camera_id>.json
        try:
            summary_data = {
                "camera_id": self.cam.id,
                "frames": self._frame_count,
                "effective_fps": self._fps_actual,
                "infer_ms_avg": self._infer_latency_ms,
                "alerts": dict(self.manager.stats) if hasattr(self, "manager") else {},
            }
            stats_path = Path(self.app.output_dir) / self.run_name / f"stats_{self.cam.id}.json"
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            stats_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _worker_loop(self) -> None:
        fps_target = max(1.0, float(self.cam.infer.fps))
        frame_interval = 1.0 / fps_target
        is_realtime_src = getattr(getattr(self, "source", None), "realtime", False)

        t_prev = time.perf_counter()
        fps_smooth = fps_target

        while not self._stop_event.is_set():
            t_loop_start = time.perf_counter()

            frame = None
            if self.source:
                try:
                    frame = self.source.read()
                except Exception as e:
                    logger.error(f"Error membaca frame {self.cam.id}: {e}")
                    time.sleep(0.05)
                    continue

                # Jika video file habis dan loop aktif
                if frame is None and getattr(self.source, "finished", False):
                    try:
                        self.source.close()
                        self.source = open_source(self.cam.source, self.cam.infer.fps)
                    except Exception:
                        pass
                    time.sleep(0.05)
                    continue

            # Jika frame masih None (misal synthetic fallback)
            if frame is None:
                img = np.full((480, 640, 3), 50, dtype=np.uint8)
                cur_ts = time.time()
                frame = Frame(image=img, ts=cur_ts, idx=self._frame_count, wall=cur_ts)

            # Proses inferensi dan modul deteksi
            h, w = frame.image.shape[:2]
            t_infer0 = time.perf_counter()

            crop = union_crop(self.modules, w, h)
            if crop:
                x1, y1, x2, y2 = crop
                img_infer, offset = frame.image[y1:y2, x1:x2], (x1, y1)
            else:
                img_infer, offset = frame.image, (0, 0)

            try:
                tracks = self.backend.infer(img_infer, frame.ts, offset)
            except Exception as e:
                logger.error(f"Inferensi error pada {self.cam.id}: {e}")
                tracks = []

            infer_ms = (time.perf_counter() - t_infer0) * 1000.0

            # Proses logika AI
            ctx = FrameContext(
                camera_id=self.cam.id,
                frame=frame,
                tracks=tracks,
                width=w,
                height=h,
                source_name=getattr(self.source, "name", "")
            )

            events = []
            for m in self.modules:
                try:
                    events += m.process(ctx)
                except Exception as e:
                    logger.error(f"Error modul {m.name}: {e}")

            if events:
                self.manager.handle(events, frame)

            # Hitung FPS aktual
            t_now = time.perf_counter()
            dt = max(1e-4, t_now - t_prev)
            t_prev = t_now
            fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / dt)

            # Gambar visual anotasi pada frame
            if self.fall_module:
                annotated = Annotator.draw_fall_detection(
                    frame.image, tracks, self.fall_module,
                    fps=fps_smooth, latency_ms=infer_ms
                )
            elif self.safezone_module:
                annotated = Annotator.draw_safezone_detection(
                    frame.image, tracks, self.safezone_module,
                    fps=fps_smooth, latency_ms=infer_ms
                )
            else:
                annotated = frame.image.copy()

            # Encode ke JPEG
            success, jpeg_buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if success:
                jpeg_bytes = jpeg_buf.tobytes()
                with self._lock:
                    self._latest_jpeg = jpeg_bytes
                    self._latest_raw = frame.image
                    self._latest_tracks = tracks
                    self._fps_actual = round(fps_smooth, 1)
                    self._infer_latency_ms = round(infer_ms, 1)
                    self._frame_count += 1
                    self._last_frame_time = time.time()

            # Jika sumber tidak realtime sleep, lakukan sleep manual
            if not is_realtime_src:
                elapsed = time.perf_counter() - t_loop_start
                sleep_time = max(0.005, frame_interval - elapsed)
                time.sleep(sleep_time)

    def get_latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            tracks_info = []
            for t in self._latest_tracks:
                state_str = "NORMAL"
                if self.fall_module:
                    st = self.fall_module.states.get(t.id)
                    if st and hasattr(st, "fsm"):
                        state_str = st.fsm.state
                elif self.safezone_module:
                    st = self.safezone_module.states.get(t.id)
                    if st:
                        state_str = st.state
                tracks_info.append({"id": t.id, "state": state_str, "cls": t.cls_name})

            return {
                "camera_id": self.cam.id,
                "is_running": self._is_running,
                "fps": self._fps_actual,
                "latency_ms": self._infer_latency_ms,
                "frame_count": self._frame_count,
                "tracks_count": len(self._latest_tracks),
                "tracks": tracks_info,
                "module": self.cam.modules[0] if self.cam.modules else "unknown",
                "source_type": self.cam.source.type,
                "last_active": self._last_frame_time,
            }

    def generate_mjpeg(self) -> Generator[bytes, None, None]:
        """Generator stream multipart MJPEG untuk ditonton lewat tag <img> di web."""
        while not self._stop_event.is_set():
            jpeg = self.get_latest_jpeg()
            if jpeg is not None:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
            time.sleep(0.04)  # Maksimal ~25 fps streaming


class CameraManager:
    """Manajer koleksi semua CameraStreamer dalam aplikasi."""

    def __init__(self, app_config: AppConfig, alert_store: AlertStore, broadcaster: EventBroadcaster,
                 run_name: Optional[str] = None):
        self.app = app_config
        self.store = alert_store
        self.broadcaster = broadcaster
        self.run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.streamers: dict[str, CameraStreamer] = {}

    def start_all(self) -> None:
        for cam in self.app.cameras:
            if cam.id not in self.streamers:
                streamer = CameraStreamer(cam, self.app, self.store, self.broadcaster, run_name=self.run_name)
                self.streamers[cam.id] = streamer
                streamer.start()
                logger.info(f"Kamera {cam.id} aktif ({cam.modules}) - Output: outputs/{self.run_name}")

    def stop_all(self) -> None:
        for streamer in self.streamers.values():
            streamer.stop()
        self.streamers.clear()

    def get_streamer(self, camera_id: str) -> Optional[CameraStreamer]:
        return self.streamers.get(camera_id)

    def get_all_stats(self) -> list[dict[str, Any]]:
        return [s.get_stats() for s in self.streamers.values()]

    def acknowledge_fall(self, track_id: int, now: float) -> None:
        """Meneruskan acknowledge ke semua FallModule."""
        for s in self.streamers.values():
            if s.fall_module:
                s.fall_module.acknowledge(track_id, now)
