import numpy as np
import pytest

from safety_ai.config import FallParams
from safety_ai.modules.fall import features as F
from safety_ai.modules.fall.module import FallModule
from safety_ai.testing.synthetic import POSE_ARR, make_track, person_at, run_timeline

H = 200.0   # tinggi anak (piksel)
X0, Y0 = 320.0, 400.0


def run(keys, duration, h=H, **kw):
    mod = FallModule("cam_test", FallParams())
    return run_timeline(mod, lambda t: [tr] if (tr := person_at(keys, t, h, **kw)) else [], duration), mod


def types(events):
    return [e.alert_type for _, e in events]


# ------------------------------------------------------------------ fitur
def test_torso_angle_standing_vs_lying():
    stand = make_track(1, POSE_ARR["stand"], X0, Y0, H).keypoints
    lie = make_track(1, POSE_ARR["lie"], X0, Y0, H).keypoints
    sh, hip, _ = F.body_points(stand, 0.3)
    assert F.torso_angle_deg(sh, hip) < 10
    sh, hip, _ = F.body_points(lie, 0.3)
    assert F.torso_angle_deg(sh, hip) > 70


def test_body_points_none_when_hips_hidden():
    kp = make_track(1, POSE_ARR["stand"], X0, Y0, H).keypoints.copy()
    kp[11, 2] = kp[12, 2] = 0.05
    assert F.body_points(kp, 0.3) is None


# ------------------------------------------------------------------ skenario utama
def test_fall_then_stay_still_gives_l1_then_l2():
    keys = [(0, "stand", X0, Y0), (5, "stand", X0, Y0), (5.5, "lie", X0, Y0), (30, "lie", X0, Y0)]
    events, _ = run(keys, 30)
    assert types(events) == ["fall_suspected", "fall_confirmed"]
    t_l1, l1 = events[0]
    t_l2, l2 = events[1]
    assert 5.5 <= t_l1 <= 9.5, t_l1          # L1 beberapa detik setelah jatuh
    assert t_l2 - t_l1 >= 4.0                # L2 menunggu imobil
    assert l1.level == 1 and l2.level == 2
    assert 0 < l1.confidence <= 1 and l2.confidence >= l1.confidence
    assert l1.meta["mode"] == "pose" and l1.bbox is not None


def test_fall_and_get_up_gives_l1_only():
    keys = [(0, "stand", X0, Y0), (5, "stand", X0, Y0), (5.5, "lie", X0, Y0), (8.3, "lie", X0, Y0),
            (9.3, "stand", X0, Y0), (20, "stand", X0, Y0)]
    events, _ = run(keys, 20)
    assert types(events) == ["fall_suspected"]


def test_quick_sit_down_no_alert():
    keys = [(0, "stand", X0, Y0), (5, "stand", X0, Y0), (5.4, "sit", X0, Y0), (25, "sit", X0, Y0)]
    events, _ = run(keys, 25)
    assert events == []


def test_lying_down_to_play_slowly_no_alert():
    # tiarap pelan (tanpa tanda jatuh cepat) lalu diam lama: jalur cepat tidak boleh memicu
    keys = [(0, "stand", X0, Y0), (5, "stand", X0, Y0), (9, "lie", X0, Y0), (40, "lie", X0, Y0)]
    events, _ = run(keys, 40)
    assert events == []


def test_already_lying_from_start_no_alert():
    keys = [(0, "lie", X0, Y0), (40, "lie", X0, Y0)]
    events, _ = run(keys, 40)
    assert events == []


def test_walking_child_no_alert():
    keys = [(0, "stand", 100, Y0), (20, "stand", 540, Y0)]
    events, _ = run(keys, 20)
    assert events == []


def test_brief_bend_no_alert_but_long_still_bend_is_known_limitation():
    brief = [(0, "stand", X0, Y0), (5, "stand", X0, Y0), (5.4, "bend", X0, Y0), (6.4, "bend", X0, Y0),
             (6.8, "stand", X0, Y0), (20, "stand", X0, Y0)]
    events, _ = run(brief, 20)
    assert events == []
    # catatan: membungkuk diam > 1,5 detik DAPAT memicu L1. Ini hard negative yang wajib diuji di data nyata.
    long_still = [(0, "stand", X0, Y0), (5, "stand", X0, Y0), (5.4, "bend", X0, Y0), (12, "bend", X0, Y0)]
    events, _ = run(long_still, 12)
    assert "fall_suspected" in types(events)


def test_fall_but_crawling_away_is_not_confirmed():
    # jatuh lalu merangkak menjauh (bergerak): boleh L1, tidak boleh L2
    keys = [(0, "stand", 100, Y0), (5, "stand", 100, Y0), (5.5, "lie", 100, Y0), (7, "lie", 100, Y0),
            (11, "lie", 500, Y0)]
    events, _ = run(keys, 11)
    assert "fall_confirmed" not in types(events)


# ------------------------------------------------------------------ mode degradasi dan track
def test_bbox_only_mode_flagged_and_slower():
    keys = [(0, "stand", X0, Y0), (5, "stand", X0, Y0), (5.5, "lie", X0, Y0), (40, "lie", X0, Y0)]
    events, _ = run(keys, 40, keypoints=False)
    assert types(events) == ["fall_suspected", "fall_confirmed"]
    assert events[0][1].meta["mode"] == "bbox_only"
    kp_events, _ = run(keys, 40)
    assert events[0][0] >= kp_events[0][0]    # konfirmasi bbox_only tidak lebih cepat


def test_low_confidence_keypoints_fall_back_to_bbox():
    keys = [(0, "stand", X0, Y0), (5, "stand", X0, Y0), (5.5, "lie", X0, Y0), (30, "lie", X0, Y0)]
    events, _ = run(keys, 30, conf=0.1)
    assert events and events[0][1].meta["mode"] == "bbox_only"


def test_track_id_switch_after_occlusion_keeps_state():
    keys = [(0, "stand", X0, Y0), (5, "stand", X0, Y0), (5.5, "lie", X0, Y0), (30, "lie", X0, Y0)]
    mod = FallModule("cam", FallParams())

    def tracks(t):
        tid = 1 if t < 8.0 else 2     # ID berganti saat DROPPING, lokasi sama
        tr = person_at(keys, t, H, tid=tid)
        return [tr] if tr and not (7.9 < t < 8.1) else []

    events = run_timeline(mod, tracks, 30)
    assert "fall_suspected" in types(events) and "fall_confirmed" in types(events)


def test_stale_state_is_cleaned_up():
    mod = FallModule("cam", FallParams())
    keys = [(0, "stand", X0, Y0), (3, "stand", X0, Y0)]
    run_timeline(mod, lambda t: [person_at(keys, t, H)] if t <= 3 else [], 3)
    assert len(mod.states) == 1
    run_timeline(mod, lambda t: [], 6, t0=4.0)
    assert len(mod.states) == 0


def test_two_children_independent():
    k1 = [(0, "stand", 150, Y0), (5, "stand", 150, Y0), (5.5, "lie", 150, Y0), (30, "lie", 150, Y0)]
    k2 = [(0, "stand", 450, Y0), (30, "stand", 470, Y0)]
    mod = FallModule("cam", FallParams())
    events = run_timeline(mod, lambda t: [x for x in (person_at(k1, t, H, tid=1), person_at(k2, t, H, tid=2)) if x], 30)
    assert {e.track_id for _, e in events} == {1}


def test_slow_path_disabled_by_default_and_enabled_by_flag():
    keys = [(0, "stand", X0, Y0), (4, "stand", X0, Y0), (10, "lie", X0, Y0), (40, "lie", X0, Y0)]
    events, _ = run(keys, 40)
    assert events == []
    mod = FallModule("cam", FallParams(slow_path_enabled=True))
    events = run_timeline(mod, lambda t: [tr] if (tr := person_at(keys, t, H)) else [], 40)
    assert types(events) == ["fall_confirmed"] and events[0][1].meta["path"] == "slow"


def test_acknowledge_stops_escalation():
    keys = [(0, "stand", X0, Y0), (5, "stand", X0, Y0), (5.5, "lie", X0, Y0), (40, "lie", X0, Y0)]
    mod = FallModule("cam", FallParams())
    first = run_timeline(mod, lambda t: [tr] if (tr := person_at(keys, t, H)) else [], 10)
    assert types(first) == ["fall_suspected"]
    mod.acknowledge(1, 10.0)
    rest = run_timeline(mod, lambda t: [tr] if (tr := person_at(keys, t, H)) else [], 30, t0=10.17)
    assert "fall_confirmed" not in types(rest)
