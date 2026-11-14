"""Modul Safe Zone: YOLOv8 (person) + ByteTrack + ROI poligon dengan histeresis.

INSIDE -> WARNING -> PENDING -> BREACH -> RETURNED -> INSIDE
Hanya track yang pernah terlihat DI DALAM zona yang bisa memicu breach (pejalan kaki di jalan tidak).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from ...config import ZoneParams, get_tz, in_windows
from ...types import Event, FrameContext, Track
from .geometry import MotionGate, polygon_bbox, poly_to_px, signed_distance

INSIDE, WARNING, PENDING, BREACH, RETURNED, OUTSIDE = "INSIDE", "WARNING", "PENDING", "BREACH", "RETURNED", "OUTSIDE"


@dataclass
class ZoneTrack:
    id: int
    first_ts: float
    last_ts: float
    state: str = OUTSIDE
    was_inside: bool = False
    pending_since: Optional[float] = None
    return_since: Optional[float] = None
    last_sample_ts: Optional[float] = None
    last_foot: tuple = (0.0, 0.0)
    last_h: float = 1.0
    meta: dict = field(default_factory=dict)


class SafeZoneModule:
    name = "safezone"

    def __init__(self, camera_id: str, params: ZoneParams, timezone: str = "Asia/Jakarta"):
        if not params.safe_polygon:
            raise ValueError("SafeZoneModule membutuhkan safe_polygon")
        self.camera_id = camera_id
        self.p = params
        self.tz = get_tz(timezone)
        self.states: dict = {}
        self.gate: Optional[MotionGate] = None
        self._poly_cache: dict = {}
        self._busy_until = -1.0
        self._gate_region = None

    # ------------------------------------------------------------------ hook runner
    def crop_region(self, width: int, height: int):
        if not self.p.crop:
            return None
        return polygon_bbox(self.p.safe_polygon, self.p.crop_margin_frac, width, height)

    def should_infer(self, frame) -> bool:
        if not self.p.motion_gate:
            return True
        if self.gate is None:
            h, w = frame.image.shape[:2]
            self.gate = MotionGate(self.p.motion_thresh, self.p.motion_min_area_frac, self.p.heartbeat_s,
                                   region=self.crop_region(w, h))
        moving_or_beat = self.gate.check(frame.image, frame.ts)
        return moving_or_beat or frame.ts <= self._busy_until

    # ------------------------------------------------------------------ klasifikasi anak / dewasa
    def classify(self, tr: Track, width: int, height: int) -> str:
        if tr.cls_name in ("child", "adult"):
            return tr.cls_name
        cal = self.p.child_height_calibration
        if cal and len(cal) >= 2:
            (y1, h1), (y2, h2) = cal[0], cal[1]
            y = min(max(tr.bbox[3] / height, min(y1, y2)), max(y1, y2))  # jepit: jangan ekstrapolasi
            expected = h1 + (h2 - h1) * (y - y1) / ((y2 - y1) or 1e-6)
            if tr.height / height > self.p.adult_ratio * max(expected, 1e-3):
                return "adult"
            return "child"
        return "unknown"  # tanpa kalibrasi dan tanpa kelas: perlakukan sebagai anak (konservatif)

    def _escorted(self, child: Track, adults: list) -> bool:
        fx, fy = child.foot_point
        limit = self.p.escort_distance_factor * max(child.height, 1.0)
        return any(math.hypot(fx - a.foot_point[0], fy - a.foot_point[1]) <= limit for a in adults)

    # ------------------------------------------------------------------ proses
    def process(self, ctx: FrameContext) -> list:
        p = self.p
        now = ctx.frame.ts
        w, h = ctx.width, ctx.height
        dt = datetime.fromtimestamp(ctx.frame.wall, self.tz) if ctx.frame.wall else datetime.now(self.tz)
        current = {t.id for t in ctx.tracks}

        if not in_windows(dt, p.active_windows, default_if_empty=True):
            self.states.clear()
            return []

        key = (w, h)
        if key not in self._poly_cache:
            self._poly_cache[key] = poly_to_px(p.safe_polygon, w, h)
        poly = self._poly_cache[key]

        labels = {t.id: self.classify(t, w, h) for t in ctx.tracks}
        adults = [t for t in ctx.tracks if labels[t.id] == "adult"]
        escort_active = in_windows(dt, p.escort_windows, default_if_empty=False)
        events: list = []
        busy = False

        for tr in ctx.tracks:
            if labels[tr.id] == "adult":
                self.states.pop(tr.id, None)
                continue
            st = self.states.get(tr.id) or self._reassociate(tr, now, current)
            if st is None:
                st = ZoneTrack(tr.id, now, now)
            st.id = tr.id
            self.states[tr.id] = st
            st.last_ts = now
            st.last_foot, st.last_h = tr.foot_point, max(tr.height, 1.0)

            d = signed_distance(poly, tr.foot_point)
            dn = d / h
            # titik kaki = tengah-bawah bbox: yang merusaknya adalah pemotongan di sisi bawah dan samping (bukan atas)
            uncertain = tr.bbox[0] <= 1 or tr.bbox[2] >= w - 1 or tr.bbox[3] >= h - 1
            if st.last_sample_ts is not None and now - st.last_sample_ts > p.max_gap_s:
                st.pending_since = None
                st.return_since = None
            st.last_sample_ts = now
            matured = (now - st.first_ts) >= p.min_track_age_s

            ev = self._step(st, dn, now, uncertain, matured, tr, labels[tr.id],
                            escorted=escort_active and labels[tr.id] != "adult" and self._escorted(tr, adults),
                            frame_wh=(w, h), wall=ctx.frame.wall)
            if ev is not None:
                events.append(ev)
            if st.state in (WARNING, PENDING, BREACH):
                busy = True

        if busy:
            self._busy_until = now + 3.0
        self._cleanup(now, current)
        return events

    # ------------------------------------------------------------------ state machine
    def _step(self, st: ZoneTrack, dn: float, now: float, uncertain: bool, matured: bool, tr: Track, label: str,
              escorted: bool, frame_wh: tuple, wall: float) -> Optional[Event]:
        p = self.p
        m, mw = p.margin_frac, p.warn_frac
        if dn > m:
            st.was_inside = True

        if st.state == OUTSIDE:  # belum pernah di dalam (atau baru muncul di luar)
            if dn > 0:
                st.state = INSIDE if dn > mw else WARNING
            elif st.was_inside or not p.require_was_inside:
                st.state, st.pending_since = (PENDING, now) if dn < -m else (WARNING, None)
            return None

        if st.state == INSIDE:
            if dn < -m and (st.was_inside or not p.require_was_inside):
                st.state, st.pending_since = PENDING, now
            elif dn <= mw:
                st.state = WARNING
                if matured:
                    return self._event("zone_warning", 0, tr, st, dn, now, wall, label, escorted, frame_wh, uncertain)
            return None

        if st.state == WARNING:
            if dn < -m:
                if st.was_inside or not p.require_was_inside:
                    st.state, st.pending_since = PENDING, now
                else:
                    st.state = OUTSIDE
            elif dn > mw:
                st.state = INSIDE
            return None

        if st.state == PENDING:
            if dn >= -m:
                st.state, st.pending_since = WARNING, None
                return None
            if uncertain or not matured:
                return None  # bbox terpotong tepi / track terlalu muda: tahan, jangan majukan timer
            if st.pending_since is None:
                st.pending_since = now
            if now - st.pending_since >= p.breach_hold_s:
                st.state = BREACH
                st.return_since = None
                if escorted and p.escort_mode == "log_only":
                    return self._event("zone_warning", 0, tr, st, dn, now, wall, label, True, frame_wh, uncertain)
                return self._event("zone_breach", 2, tr, st, dn, now, wall, label, escorted, frame_wh, uncertain)
            return None

        if st.state == BREACH:
            if dn > m:
                st.return_since = st.return_since if st.return_since is not None else now
                if now - st.return_since >= p.return_hold_s:
                    st.state = RETURNED
            else:
                st.return_since = None
            return None

        if st.state == RETURNED:
            st.state = INSIDE if dn > mw else WARNING
        return None

    def _event(self, alert_type: str, level: int, tr: Track, st: ZoneTrack, dn: float, now: float, wall: float,
               label: str, escorted: bool, frame_wh: tuple, uncertain: bool) -> Event:
        depth = max(0.0, -dn)
        conf = 0.5 * float(tr.conf) + 0.3 * min(1.0, depth / (3 * self.p.margin_frac)) + 0.2 * (0.5 if uncertain else 1.0)
        w, h = frame_wh
        fx, fy = tr.foot_point
        meta = {"d_norm": round(dn, 4), "label": label, "escorted": bool(escorted),
                "foot": [round(fx / w, 4), round(fy / h, 4)]}
        return Event(alert_type=alert_type, level=level, camera_id=self.camera_id, track_id=tr.id,
                     confidence=round(min(1.0, max(0.0, conf)), 3), meta=meta, stream_ts=now, wall=wall, bbox=tr.bbox)

    # ------------------------------------------------------------------ ID berganti / track hilang
    def _reassociate(self, tr: Track, now: float, current: set) -> Optional[ZoneTrack]:
        best, best_d = None, None
        fx, fy = tr.foot_point
        for tid, st in self.states.items():
            if tid in current or now - st.last_ts > self.p.lost_grace_s:
                continue
            d = math.hypot(fx - st.last_foot[0], fy - st.last_foot[1])
            if d <= 0.8 * max(st.last_h, tr.height) and (best_d is None or d < best_d):
                best, best_d = tid, d
        return self.states.pop(best) if best is not None else None

    def _cleanup(self, now: float, current: set) -> None:
        for tid in list(self.states):
            if tid in current:
                continue
            st = self.states[tid]
            grace = 10.0 if st.state in (PENDING, BREACH) else self.p.lost_grace_s
            if now - st.last_ts > grace:
                del self.states[tid]
