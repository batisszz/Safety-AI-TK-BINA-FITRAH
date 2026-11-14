from datetime import datetime, timedelta, timezone

import numpy as np

from safety_ai.config import ZoneParams
from safety_ai.modules.safezone.geometry import MotionGate, polygon_bbox, poly_to_px, signed_distance
from safety_ai.modules.safezone.module import SafeZoneModule
from safety_ai.testing.synthetic import box_track, path_at, run_timeline

WIB = timezone(timedelta(hours=7))
POLY = [[0.1, 0.9], [0.1, 0.3], [0.9, 0.3], [0.9, 0.9]]   # frame 640x480 -> garis atas di y = 144
LINE_Y = 0.3 * 480
H_CHILD = 110.0


def wall_at(h, m):
    return datetime(2026, 11, 12, h, m, tzinfo=WIB).timestamp()


def make(**kw):
    return SafeZoneModule("cam_gerbang", ZoneParams(safe_polygon=POLY, motion_gate=False, **kw))


def go(mod, paths, duration=15, fps=5, wall0=1_700_000_000.0):
    """paths: {tid: (points, extra_kwargs)}"""
    def tracks(t):
        out = []
        for tid, (pts, kw) in paths.items():
            pos = path_at(pts, t)
            if pos:
                out.append(box_track(tid, pos[0], pos[1], kw.get("h", H_CHILD), cls_name=kw.get("cls", "person")))
        return out
    return run_timeline(mod, tracks, duration, fps=fps, camera_id="cam_gerbang", wall0=wall0)


def types(events):
    return [e.alert_type for _, e in events]


# ------------------------------------------------------------------ geometri
def test_signed_distance_inside_outside_edge():
    poly = poly_to_px(POLY, 640, 480)
    assert signed_distance(poly, (320, 300)) > 0
    assert signed_distance(poly, (320, 100)) < 0
    assert abs(signed_distance(poly, (320, LINE_Y))) < 1e-3
    assert abs(signed_distance(poly, (320, 100)) + 44) < 1e-3     # 44 px di luar garis


def test_polygon_bbox_with_margin_clipped():
    x1, y1, x2, y2 = polygon_bbox(POLY, 0.12, 640, 480)
    assert (x1, y1) == (int(0), int(0.18 * 480))
    assert x2 == 640 and y2 == 480


# ------------------------------------------------------------------ skenario
def test_child_crosses_line_and_stays_gives_warning_then_breach():
    path = {1: ([(0, 320, 300), (4, 320, 160), (6, 320, 100), (15, 320, 60)], {})}
    events = go(make(), path)
    assert types(events) == ["zone_warning", "zone_breach"]
    breach = events[1][1]
    assert breach.level == 2 and breach.meta["d_norm"] < 0 and 0 < breach.confidence <= 1
    assert events[0][1].level == 0
    t_cross = 4 + (160 - LINE_Y) / (160 - 100) * 2 * -1   # perkiraan; hanya untuk sanity
    assert events[1][0] - 5.0 <= 4.0                       # alarm tidak terlambat jauh dari saat melewati garis


def test_approach_without_crossing_no_breach():
    path = {1: ([(0, 320, 300), (5, 320, 150), (15, 320, 152)], {})}
    assert "zone_breach" not in types(go(make(), path))


def test_flicker_on_the_line_no_breach():
    pts = [(i * 0.2, 320, LINE_Y + (6 if i % 2 else -6)) for i in range(0, 80)]
    pts = [(0, 320, 300), (2, 320, LINE_Y)] + [(2 + t, x, y) for t, x, y in pts]
    assert "zone_breach" not in types(go(make(), {1: (pts, {})}, duration=18))


def test_brief_step_out_and_back_no_breach():
    path = {1: ([(0, 320, 300), (3, 320, 160), (3.4, 320, 120), (4.0, 320, 120), (4.4, 320, 160), (10, 320, 300)], {})}
    assert "zone_breach" not in types(go(make(), path, duration=10))


def test_passerby_outside_never_inside_no_event():
    path = {1: ([(0, 100, 90), (15, 540, 90)], {})}
    assert go(make(), path) == []


def test_breach_then_return_resets_and_can_alert_again():
    mod = make(breach_hold_s=1.0)
    path = {1: ([(0, 320, 300), (3, 320, 160), (4, 320, 100), (8, 320, 100), (9, 320, 300), (12, 320, 300),
                 (13, 320, 160), (14, 320, 100), (22, 320, 100)], {})}
    ev = go(mod, path, duration=22)
    assert types(ev).count("zone_breach") == 2     # cooldown antar-alert diatur AlertManager, bukan modul


def test_adult_by_class_is_ignored():
    path = {1: ([(0, 320, 300), (4, 320, 160), (6, 320, 100), (15, 320, 60)], {"cls": "adult"})}
    assert go(make(), path) == []


def test_adult_by_height_calibration_is_ignored_child_is_not():
    cal = [[0.3, 0.16], [0.9, 0.30]]     # tinggi anak (ternormalisasi) di y=0.3 dan y=0.9
    mod = make(child_height_calibration=cal)
    adult = {1: ([(0, 320, 300), (4, 320, 160), (6, 320, 100), (15, 320, 60)], {"h": 190.0})}
    assert go(mod, adult) == []
    mod = make(child_height_calibration=cal)
    child = {1: ([(0, 320, 300), (4, 320, 160), (6, 320, 100), (15, 320, 60)], {"h": 100.0})}
    assert "zone_breach" in types(go(mod, child))


def test_escorted_child_in_handover_window_is_log_only_but_alone_alarms():
    child = ([(0, 320, 300), (4, 320, 160), (6, 320, 100), (15, 320, 60)], {})
    adult = ([(0, 360, 300), (4, 360, 160), (6, 360, 100), (15, 360, 60)], {"cls": "adult", "h": 190.0})
    ev = go(make(), {1: child, 2: adult}, wall0=wall_at(7, 15))
    assert "zone_breach" not in types(ev) and "zone_warning" in types(ev)
    ev = go(make(), {1: child}, wall0=wall_at(7, 15))                 # anak sendirian
    assert "zone_breach" in types(ev)
    ev = go(make(), {1: child, 2: adult}, wall0=wall_at(13, 0))       # di luar jam serah-terima
    assert "zone_breach" in types(ev)


def test_active_windows_suppress_outside_schedule():
    mod = make(active_windows=["07:00-11:00"])
    child = {1: ([(0, 320, 300), (4, 320, 160), (6, 320, 100), (15, 320, 60)], {})}
    assert go(mod, child, wall0=wall_at(14, 0)) == []
    mod = make(active_windows=["07:00-11:00"])
    assert "zone_breach" in types(go(mod, child, wall0=wall_at(8, 0)))


def test_clipped_bbox_at_frame_edge_holds_timer():
    # anak keluar lewat tepi bawah frame yang berada di luar poligon: bbox terpotong = tidak pasti
    poly = [[0.1, 0.1], [0.1, 0.97], [0.9, 0.97], [0.9, 0.1]]
    mod = SafeZoneModule("c", ZoneParams(safe_polygon=poly, motion_gate=False))
    path = {1: ([(0, 320, 150), (5, 320, 440), (6, 320, 479), (15, 320, 479)], {})}

    def tracks(t):
        pos = path_at(path[1][0], t)
        return [box_track(1, pos[0], pos[1], 110)] if pos else []
    ev = run_timeline(mod, tracks, 15, fps=5, camera_id="c")
    assert "zone_breach" not in types(ev)


def test_id_switch_at_boundary_keeps_was_inside():
    mod = make()

    def tracks(t):
        pos = path_at([(0, 320, 300), (4, 320, 160), (6, 320, 100), (15, 320, 60)], t)
        if not pos or 5.9 < t < 6.1:
            return []
        return [box_track(1 if t < 6 else 9, pos[0], pos[1], H_CHILD)]
    ev = run_timeline(mod, tracks, 15, fps=5, camera_id="cam_gerbang")
    assert "zone_breach" in types(ev) and ev[-1][1].track_id == 9


def test_stale_tracks_cleaned():
    mod = make()
    go(mod, {1: ([(0, 320, 300), (2, 320, 300)], {})}, duration=2)
    assert len(mod.states) == 1
    run_timeline(mod, lambda t: [], 8, fps=5, t0=3.0)
    assert len(mod.states) == 0


# ------------------------------------------------------------------ motion gate
def test_motion_gate_skips_static_scene_but_heartbeats():
    gate = MotionGate(heartbeat_s=1.0)
    img = np.full((480, 640, 3), 90, np.uint8)
    flags = [gate.check(img, i * 0.2) for i in range(16)]   # 3 detik, adegan statis
    assert flags[0] is True
    assert sum(flags) <= 4                                   # hanya heartbeat ~1x/detik
    moving = img.copy()
    moving[200:400, 200:400] = 255
    assert gate.check(moving, 3.3) is True
