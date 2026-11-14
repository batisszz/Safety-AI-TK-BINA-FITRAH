"""State machine jatuh per anak.

NORMAL -> DROPPING -> SUSPECTED (L1) -> CONFIRMED (L2) -> RESOLVED -> NORMAL

Syarat jatuh = tanda jatuh cepat (DROPPING) + posisi rendah yang bertahan dengan gerak rendah.
Anak yang tiarap bermain tidak memicu alert karena tidak ada tanda jatuh cepat.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ...config import FallParams

NORMAL, DROPPING, SUSPECTED, CONFIRMED, RESOLVED = "NORMAL", "DROPPING", "SUSPECTED", "CONFIRMED", "RESOLVED"


@dataclass
class Obs:
    """Observasi per anak per frame (hasil fitur)."""

    ts: float
    is_drop: bool
    drop_speed: float
    low_pose: bool
    strict_upright: bool
    motion: Optional[float]  # tinggi-tubuh per detik, None = belum diketahui
    mode: str  # "pose" | "bbox_only"
    kpt_conf: float
    angle: Optional[float]
    h_ratio: float


class FallFSM:
    def __init__(self, params: FallParams, slow_path: bool = False):
        self.p = params
        self.slow_path = slow_path
        self.state = NORMAL
        self.low_since: Optional[float] = None
        self.low_after_collapse = False
        self.upright_since: Optional[float] = None
        self.strict_upright_last: Optional[float] = None
        self.still_since: Optional[float] = None
        self.t_drop: Optional[float] = None
        self.peak_drop = 0.0
        self.suspected_ts: Optional[float] = None
        self.resolved_ts: Optional[float] = None
        self.l1_conf = 0.0

    # ------------------------------------------------------------------ util
    def _factor(self, obs: Obs) -> float:
        return self.p.bbox_only_factor if obs.mode == "bbox_only" else 1.0

    def _track_pose_timers(self, obs: Obs) -> None:
        p = self.p
        if obs.low_pose:
            if self.low_since is None:
                self.low_since = obs.ts
                last = self.strict_upright_last
                self.low_after_collapse = last is not None and (obs.ts - last) <= p.slow_transition_s
            self.upright_since = None
        else:
            self.low_since = None
            self.low_after_collapse = False
            if self.upright_since is None:
                self.upright_since = obs.ts
        if obs.strict_upright:
            self.strict_upright_last = obs.ts
        # imobil dengan histeresis: mulai jika <= motion_still, putus jika > motion_still_break
        if obs.motion is not None and obs.motion <= p.motion_still:
            if self.still_since is None:
                self.still_since = obs.ts
        elif obs.motion is None or obs.motion > p.motion_still_break:
            self.still_since = None

    def _conf(self, obs: Obs, held: float, level: int, still: float = 0.0) -> float:
        p = self.p
        c = (0.4 * obs.kpt_conf
             + 0.3 * min(1.0, self.peak_drop / (2 * p.t_drop))
             + 0.3 * min(1.0, held / (2 * p.low_pose_hold_s)))
        if obs.mode == "bbox_only":
            c *= 0.8
        if level == 2:
            c = min(1.0, max(c, self.l1_conf) + 0.1 + 0.2 * min(1.0, still / (2 * p.confirm_still_s)))
        return round(max(0.0, min(1.0, c)), 3)

    def _meta(self, obs: Obs, extra: Optional[dict] = None) -> dict:
        m = {
            "mode": obs.mode,
            "hip_speed": round(self.peak_drop, 2),
            "torso_angle": None if obs.angle is None else round(obs.angle, 1),
            "h_ratio": round(obs.h_ratio, 2),
        }
        if extra:
            m.update(extra)
        return m

    # ------------------------------------------------------------------ langkah utama
    def step(self, obs: Obs) -> list:
        """Proses satu observasi, kembalikan daftar event: [(alert_type, level, confidence, meta)]."""
        p = self.p
        f = self._factor(obs)
        out: list = []
        self._track_pose_timers(obs)
        now = obs.ts

        if self.state == RESOLVED:
            if self.resolved_ts is None or now - self.resolved_ts >= p.refractory_s:
                self.state = NORMAL
            return out

        if self.state == NORMAL:
            if obs.is_drop:
                already_low = self.low_since is not None and (now - self.low_since) > 1.5 * p.drop_window_s
                if not already_low:
                    self.state = DROPPING
                    self.t_drop = now
                    self.peak_drop = obs.drop_speed
            elif self.slow_path and self.low_since is not None and self.low_after_collapse:
                need = p.slow_still_s * f
                if (self.still_since is not None and now - self.still_since >= need
                        and now - self.low_since >= need):
                    self.state = CONFIRMED
                    self.l1_conf = 0.0
                    out.append(("fall_confirmed", 2, self._conf(obs, now - self.low_since, 2, now - self.still_since),
                                self._meta(obs, {"path": "slow"})))

        elif self.state == DROPPING:
            self.peak_drop = max(self.peak_drop, obs.drop_speed)
            hold = p.low_pose_hold_s * f
            if self.low_since is None:
                if now - self.t_drop > p.low_pose_start_window_s:
                    self.state = NORMAL  # duduk cepat / tersandung ringan
            else:
                held = now - self.low_since
                if held >= hold and obs.motion is not None and obs.motion <= p.motion_low:
                    self.state = SUSPECTED
                    self.suspected_ts = now
                    self.l1_conf = self._conf(obs, held, 1)
                    out.append(("fall_suspected", 1, self.l1_conf, self._meta(obs, {"low_pose_s": round(held, 2)})))
                elif now - self.t_drop > p.low_pose_start_window_s + hold + 2.0:
                    self.state = NORMAL

        elif self.state == SUSPECTED:
            confirm = p.confirm_still_s * f
            if self.upright_since is not None and now - self.upright_since >= p.recover_s:
                self.state, self.resolved_ts = RESOLVED, now
            elif self.low_since is not None and self.still_since is not None and now - self.still_since >= confirm:
                self.state = CONFIRMED
                still = now - self.still_since
                out.append(("fall_confirmed", 2, self._conf(obs, now - self.low_since, 2, still),
                            self._meta(obs, {"path": "fast", "still_s": round(still, 2)})))
            elif now - self.suspected_ts > p.suspected_timeout_s:
                self.state, self.resolved_ts = RESOLVED, now  # kedaluwarsa tanpa konfirmasi

        elif self.state == CONFIRMED:
            if self.upright_since is not None and now - self.upright_since >= p.recover_s:
                self.state, self.resolved_ts = RESOLVED, now

        return out

    def acknowledge(self, now: float) -> None:
        """Guru menandai Acknowledged: hentikan eskalasi."""
        if self.state in (SUSPECTED, CONFIRMED):
            self.state, self.resolved_ts = RESOLVED, now
