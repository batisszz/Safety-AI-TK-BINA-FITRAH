"""Sumber video yang dapat diganti lewat konfigurasi: file, webcam, RTSP.

File   : decode hanya frame yang diperlukan (grab/skip), waktu = idx / fps (bukan jam dinding).
Live   : thread pembaca terus menguras buffer, hanya frame TERBARU yang dipakai (delay tidak menumpuk),
         dengan reconnect otomatis (backoff) dan watchdog.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from datetime import datetime
from typing import Callable, Optional

import cv2

from ..config import SourceConfig
from ..types import Frame


class FileSource:
    is_live = False

    def __init__(self, path: str, sample_fps: float, loop: bool = False, realtime: bool = False,
                 start_time: Optional[str] = None):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"video tidak bisa dibuka: {path}")
        self.path = path
        self.name = os.path.basename(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        if self.fps <= 1 or self.fps > 240:
            self.fps = 25.0
        self.step = max(1, round(self.fps / max(sample_fps, 0.1)))
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.loop, self.realtime = loop, realtime
        self.finished = False
        self._pos = 0
        self._offset = 0
        self._idx = 0
        self._t_mono = time.monotonic()
        self._wall0 = datetime.fromisoformat(start_time).timestamp() if start_time else time.time()

    def read(self) -> Optional[Frame]:
        while not self.finished:
            if not self.cap.grab():
                if self.loop and self._pos > 0:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._offset += self._pos
                    self._pos = 0
                    continue
                self.finished = True
                return None
            pos = self._pos
            self._pos += 1
            if pos % self.step != 0:
                continue
            ok, img = self.cap.retrieve()
            if not ok:
                continue
            ts = (pos + self._offset) / self.fps
            if self.realtime:
                delay = self._t_mono + ts - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            idx, self._idx = self._idx, self._idx + 1
            return Frame(image=img, ts=ts, idx=idx, wall=self._wall0 + ts)
        return None

    def seconds_since_frame(self) -> float:
        return 0.0

    def close(self) -> None:
        self.cap.release()


class LiveSource:
    """Webcam atau RTSP. `opener` dapat diganti pada pengujian."""

    is_live = True

    def __init__(self, opener: Callable, sample_fps: float, name: str = "live", reconnect_max_s: float = 30.0):
        self.opener = opener
        self.name = name
        self.interval = 1.0 / max(sample_fps, 0.1)
        self.reconnect_max_s = reconnect_max_s
        self.finished = False
        self.connected = False
        self._lock = threading.Lock()
        self._latest = None
        self._latest_seq = 0
        self._served_seq = 0
        self._last_served = 0.0
        self._last_frame_mono = time.monotonic()
        self._t0 = time.monotonic()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._reader, name=f"reader-{name}", daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        backoff = 1.0
        cap = None
        while not self._stop.is_set():
            try:
                if cap is None:
                    cap = self.opener()
                    if cap is None or not cap.isOpened():
                        raise RuntimeError("tidak bisa membuka sumber")
                    self.connected = True
                    backoff = 1.0
                ok, img = cap.read()
                if not ok or img is None:
                    raise RuntimeError("gagal membaca frame")
                with self._lock:
                    self._latest, self._latest_seq = img, self._latest_seq + 1
                self._last_frame_mono = time.monotonic()
            except Exception:
                self.connected = False
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = None
                if self._stop.wait(backoff):
                    break
                backoff = min(self.reconnect_max_s, backoff * 2)
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    def read(self, timeout: float = 1.0) -> Optional[Frame]:
        end = time.monotonic() + timeout
        while time.monotonic() < end and not self._stop.is_set():
            with self._lock:
                fresh = self._latest is not None and self._latest_seq != self._served_seq
                due = time.monotonic() - self._last_served >= self.interval
                if fresh and due:
                    img, self._served_seq = self._latest, self._latest_seq
                    self._last_served = time.monotonic()
                    now = time.time()
                    return Frame(image=img, ts=time.monotonic() - self._t0, idx=self._served_seq, wall=now)
            time.sleep(0.005)
        return None

    def seconds_since_frame(self) -> float:
        return time.monotonic() - self._last_frame_mono

    def close(self) -> None:
        self._stop.set()
        self.finished = True
        self._thread.join(timeout=2.0)


def _rtsp_opener(url: str, transport: str) -> Callable:
    def open_():
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{transport}"
        try:
            params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000]
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, params)
        except TypeError:  # OpenCV lama tanpa parameter timeout
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        return cap
    return open_


def _webcam_opener(index: int) -> Callable:
    def open_():
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if cap.isOpened():
                return cap
        return cv2.VideoCapture(index)
    return open_


class BrowserSource:
    """Sumber frame yang didorong dari klien browser (mis. kamera HP via getUserMedia)."""

    is_live = True
    realtime = True

    def __init__(self, sample_fps: float, name: str = "browser", stall_s: float = 10.0):
        self.name = name
        self.interval = 1.0 / max(sample_fps, 0.1)
        self.stall_s = stall_s
        self.finished = False
        self.connected = False
        self._lock = threading.Lock()
        self._latest = None
        self._latest_seq = 0
        self._served_seq = 0
        self._last_served = 0.0
        self._last_frame_mono = 0.0
        self._t0 = time.monotonic()
        self._idx = 0

    def push(self, image) -> None:
        with self._lock:
            self._latest = image
            self._latest_seq += 1
            self._last_frame_mono = time.monotonic()
            self.connected = True

    def read(self, timeout: float = 1.0) -> Optional[Frame]:
        end = time.monotonic() + timeout
        while time.monotonic() < end and not self.finished:
            with self._lock:
                fresh = self._latest is not None and self._latest_seq != self._served_seq
                due = time.monotonic() - self._last_served >= self.interval
                if fresh and due:
                    img = self._latest.copy()
                    self._served_seq = self._latest_seq
                    self._last_served = time.monotonic()
                    idx = self._idx
                    self._idx += 1
                    now = time.time()
                    return Frame(image=img, ts=time.monotonic() - self._t0, idx=idx, wall=now)
            time.sleep(0.005)
        return None

    def seconds_since_frame(self) -> float:
        if self._last_frame_mono <= 0:
            return 0.0
        return time.monotonic() - self._last_frame_mono

    def close(self) -> None:
        self.finished = True
        self.connected = False


def open_source(cfg: SourceConfig, sample_fps: float):
    if cfg.type == "file":
        return FileSource(cfg.path, sample_fps, loop=cfg.loop, realtime=cfg.realtime, start_time=cfg.start_time)
    if cfg.type == "webcam":
        return LiveSource(_webcam_opener(cfg.index), sample_fps, name=f"webcam{cfg.index}")
    if cfg.type == "browser":
        return BrowserSource(sample_fps, name="browser", stall_s=cfg.stall_s)
    return LiveSource(_rtsp_opener(cfg.url, cfg.rtsp_transport), sample_fps, name="rtsp")
