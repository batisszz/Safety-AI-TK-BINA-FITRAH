"""Modul Fall Detection: YOLOv8-Pose + ByteTrack -> fitur per anak -> state machine dua tingkat."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ...config import FallParams
from ...types import Event, FrameContext, Track
from . import features as F
from .state_machine import CONFIRMED, DROPPING, SUSPECTED, FallFSM, Obs


@dataclass
class Sample:
    ts: float
    h: float
    cx: float
    cy: float
    hip_y: Optional[float]
    kp: Optional[np.ndarray]   # (17, 2) hasil smoothing
    vis: Optional[np.ndarray]  # (17,) bool


class TrackCtx:
    def __init__(self, tid: int, p: FallParams, now: float):
        self.id = tid
        self.p = p
        self.first_ts = now
        self.last_ts = now
        self.samples: deque = deque()
        self.kp_s: Optional[np.ndarray] = None
        self.fsm = FallFSM(p, slow_path=p.slow_path_enabled)
        self.h_ref = 1.0
        self.last_center = (0.0, 0.0)
        self.last_bbox: Optional[tuple] = None

    # ------------------------------------------------------------------ fitur
    def observe(self, tr: Track, now: float) -> Obs:
        p = self.p
        h = max(tr.height, 1.0)
        cx, cy = tr.center
        self.last_ts, self.last_center, self.last_bbox = now, (cx, cy), tr.bbox

        kp = tr.keypoints
        hip_y = None
        head_dy_raw = None
        angle = None
        kconf = 0.0
        pose_ok = False
        cur_kp = None
        cur_vis = None
        if kp is not None and len(kp) >= 17:
            self.kp_s = F.smooth_keypoints(self.kp_s, kp, p.kpt_conf, p.ema_alpha)
            bp = F.body_points(self.kp_s_for_vis(kp), p.kpt_conf)
            if bp is not None:
                sh, hip, head_y = bp
                pose_ok = True
                hip_y = float(hip[1])
                head_dy_raw = hip_y - head_y
                angle = F.torso_angle_deg(sh, hip)
                kconf = F.mean_visible_conf(kp, p.kpt_conf)
            cur_kp = self.kp_s[:, :2].copy()
            cur_vis = F.visible_mask(kp, p.kpt_conf)

        # tinggi referensi = tinggi maksimum bbox pada jendela terakhir (anak sedang tegak)
        horizon = now - p.href_window_s
        self.h_ref = max([h] + [s.h for s in self.samples if s.ts >= horizon])
        h_ref = max(self.h_ref, 1.0)

        cur = Sample(now, h, cx, cy, hip_y, cur_kp, cur_vis)
        win_drop = [s for s in self.samples if s.ts >= now - p.drop_window_s] + [cur]
        win_shrink = [s for s in self.samples if s.ts >= now - p.shrink_window_s] + [cur]

        # kecepatan turun (pinggul bila semua sampel di jendela punya pose, kalau tidak titik tengah bbox)
        use_hip = all(s.hip_y is not None for s in win_drop)
        ys = [(s.ts, s.hip_y if use_hip else s.cy) for s in win_drop]
        ts_min, y_min = min(ys, key=lambda t: t[1])
        y_now = ys[-1][1]
        dt = max(now - ts_min, 0.5 * p.drop_window_s)
        drop_speed = max(0.0, (y_now - y_min) / h_ref / dt)
        shrink = max(0.0, (max(s.h for s in win_shrink) - h) / h_ref)
        is_drop = drop_speed >= p.t_drop or shrink >= p.t_shrink

        h_ratio = h / h_ref
        aspect = tr.width / h
        if pose_ok:
            head_dy = head_dy_raw / h_ref
            low = (angle > p.angle_low_deg or h_ratio < p.hratio_low) and head_dy < p.head_hip_min
            strict = (not low) and angle < 30.0
            mode = "pose"
        else:
            low = h_ratio < p.hratio_low or aspect > p.bbox_aspect_low
            strict = (not low) and h_ratio > 0.85
            mode = "bbox_only"

        motion = self._motion(cur, now, h_ref)
        self.samples.append(cur)
        while self.samples and self.samples[0].ts < now - (p.href_window_s + 1.0):
            self.samples.popleft()

        return Obs(ts=now, is_drop=is_drop, drop_speed=drop_speed, low_pose=low, strict_upright=strict,
                   motion=motion, mode=mode, kpt_conf=kconf, angle=angle, h_ratio=h_ratio)

    def kp_s_for_vis(self, raw: np.ndarray) -> np.ndarray:
        """Posisi hasil smoothing, conf mentah (agar visibilitas mengikuti frame sekarang)."""
        out = self.kp_s.copy()
        out[:, 2] = raw[:, 2]
        return out

    def _motion(self, cur: Sample, now: float, h_ref: float) -> Optional[float]:
        """Perpindahan median joint dalam jendela motion_window_s, dalam tinggi-tubuh per detik."""
        w = self.p.motion_window_s
        ref = None
        for s in reversed(self.samples):
            if s.ts <= now - w:
                ref = s
                break
        if ref is None:
            return None
        dt = max(now - ref.ts, 1e-3)
        if cur.kp is not None and ref.kp is not None:
            both = cur.vis & ref.vis
            if both.sum() >= 4:
                d = np.linalg.norm(cur.kp[both] - ref.kp[both], axis=1)
                return float(np.median(d)) / h_ref / dt
        return math.hypot(cur.cx - ref.cx, cur.cy - ref.cy) / h_ref / dt


class FallModule:
    name = "fall"

    def __init__(self, camera_id: str, params: Optional[FallParams] = None):
        self.camera_id = camera_id
        self.p = params or FallParams()
        self.states: dict = {}

    # ------------------------------------------------------------------ API modul
    def should_infer(self, frame) -> bool:
        return True

    def crop_region(self, width: int, height: int):
        return None

    def process(self, ctx: FrameContext) -> list:
        now = ctx.frame.ts
        current = {t.id for t in ctx.tracks}
        events: list = []
        for tr in ctx.tracks:
            if tr.keypoints is None and tr.height < 8:
                continue
            st = self.states.get(tr.id)
            if st is None:
                st = self._reassociate(tr, now, current)
            if st is None:
                st = TrackCtx(tr.id, self.p, now)
            self.states[tr.id] = st
            obs = st.observe(tr, now)
            if now - st.first_ts < self.p.min_track_age_s:
                continue
            for alert_type, level, conf, meta in st.fsm.step(obs):
                events.append(Event(alert_type=alert_type, level=level, camera_id=self.camera_id, track_id=tr.id,
                                    confidence=conf, meta=meta, stream_ts=now, wall=ctx.frame.wall, bbox=tr.bbox))
        self._cleanup(now, current)
        return events

    def acknowledge(self, track_id: int, now: float) -> None:
        st = self.states.get(track_id)
        if st:
            st.fsm.acknowledge(now)

    # ------------------------------------------------------------------ track hilang / ID berganti
    def _reassociate(self, tr: Track, now: float, current: set) -> Optional[TrackCtx]:
        """Track baru muncul dekat track yang baru hilang (oklusi) -> warisi state-nya."""
        best, best_d = None, None
        cx, cy = tr.center
        for tid, st in self.states.items():
            if tid in current:
                continue
            active = st.fsm.state in (DROPPING, SUSPECTED, CONFIRMED)
            grace = self.p.suspected_lost_grace_s if active else self.p.lost_grace_s
            if now - st.last_ts > grace:
                continue
            d = math.hypot(cx - st.last_center[0], cy - st.last_center[1])
            if d <= self.p.reassoc_radius * st.h_ref and (best_d is None or d < best_d):
                best, best_d = tid, d
        if best is None:
            return None
        return self.states.pop(best)

    def _cleanup(self, now: float, current: set) -> None:
        for tid in list(self.states):
            if tid in current:
                continue
            st = self.states[tid]
            active = st.fsm.state in (DROPPING, SUSPECTED, CONFIRMED)
            grace = self.p.suspected_lost_grace_s if active else self.p.lost_grace_s
            if now - st.last_ts > grace:
                del self.states[tid]
