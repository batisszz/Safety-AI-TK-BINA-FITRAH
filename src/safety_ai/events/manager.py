"""AlertManager: dedupe/cooldown, batas laju, snapshot, lalu kirim ke sink."""
from __future__ import annotations

import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2

from ..config import AlertsConfig, get_tz
from ..types import Event, Frame


class AlertManager:
    def __init__(self, cfg: AlertsConfig, sink, timezone: str = "Asia/Jakarta", base_dir: str = ".",
                 source_name: str = ""):
        self.cfg = cfg
        self.sink = sink
        self.tz = get_tz(timezone)
        self.base = Path(base_dir)
        self.source_name = source_name
        self._last: dict = {}
        self._recent: deque = deque()
        self.stats = {"emitted": 0, "cooldown_dropped": 0, "rate_dropped": 0}

    def handle(self, events: list, frame: Optional[Frame] = None) -> list:
        sent = []
        for ev in events:
            key = (ev.camera_id, ev.alert_type, ev.track_id, ev.meta.get("state"))
            cooldown = float(self.cfg.cooldown_s.get(ev.alert_type, 30.0))
            last = self._last.get(key)
            if last is not None and 0 <= ev.stream_ts - last < cooldown:
                self.stats["cooldown_dropped"] += 1
                continue
            while self._recent and ev.stream_ts - self._recent[0] > 60.0:
                self._recent.popleft()
            if ev.alert_type != "system_health" and len(self._recent) >= self.cfg.max_per_minute:
                self.stats["rate_dropped"] += 1
                continue
            self._last[key] = ev.stream_ts
            self._recent.append(ev.stream_ts)
            self._finalize(ev, frame)
            self.sink.emit(ev.to_dict())
            self.stats["emitted"] += 1
            sent.append(ev)
        return sent

    def _finalize(self, ev: Event, frame: Optional[Frame]) -> None:
        ev.event_id = str(uuid.uuid4())
        dt = datetime.fromtimestamp(ev.wall, self.tz) if ev.wall else datetime.now(self.tz)
        ev.timestamp = dt.isoformat(timespec="seconds")
        if self.source_name:
            ev.meta.setdefault("source", self.source_name)
        if self.cfg.snapshots and ev.level > 0 and frame is not None:
            ev.snapshot_path = self._save_snapshot(ev, frame, dt)

    def _save_snapshot(self, ev: Event, frame: Frame, dt: datetime) -> Optional[str]:
        try:
            rel = Path(self.cfg.snapshot_dir) / ev.camera_id / dt.strftime("%Y-%m-%d") / \
                f"{dt.strftime('%H%M%S')}_{ev.track_id}_{ev.alert_type}.jpg"
            out = self.base / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            img = frame.image.copy()
            if ev.bbox is not None:
                x1, y1, x2, y2 = [int(v) for v in ev.bbox]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"{ev.alert_type} L{ev.level} {ev.confidence:.2f}", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imwrite(str(out), img, [int(cv2.IMWRITE_JPEG_QUALITY), self.cfg.jpeg_quality])
            return rel.as_posix()
        except Exception:  # snapshot tidak boleh menjatuhkan pengiriman event
            return None
