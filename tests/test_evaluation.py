import csv

import pytest

from safety_ai.evaluation import evaluate, format_markdown, load_gt, select_predictions, wilson


def E(t, typ="fall_suspected", cam="cam", track=1, src="v1.mp4"):
    return {"camera_id": cam, "alert_type": typ, "track_id": track, "meta": {"stream_ts": t, "source": src}}


def G(t, typ="fall", src="v1.mp4", scen="jatuh", t_end=None):
    return {"source": src, "event_type": typ, "t_start": t, "t_end": t_end if t_end is not None else t, "scenario": scen}


def test_wilson_matches_prd_example():
    lo, hi = wilson(34, 40)
    assert abs(lo - 0.709) < 0.002 and abs(hi - 0.929) < 0.002      # contoh di PRD bagian 14.6
    lo, hi = wilson(18, 20)
    assert abs(lo - 0.699) < 0.002 and abs(hi - 0.972) < 0.002
    assert wilson(0, 0) is None


def test_prd_illustration_numbers():
    gt = [G(10 * i) for i in range(40)]
    ev = [E(10 * i + 2) for i in range(34)] + [E(1000 + 10 * i) for i in range(4)]
    r = evaluate(ev, gt, hours=20)
    assert (r["TP"], r["FN"], r["FP"]) == (34, 6, 4)
    assert abs(r["recall"] - 0.85) < 1e-9 and abs(r["precision"] - 34 / 38) < 1e-9
    assert abs(r["f1"] - 0.8718) < 1e-3 and abs(r["false_alarms_per_hour"] - 0.2) < 1e-9
    assert abs(r["false_alarms_per_school_day_4h"] - 0.8) < 1e-9
    assert abs(r["delay_median_s"] - 2.0) < 1e-9


def test_tolerance_window_is_asymmetric():
    r = evaluate([E(20 - 3)], [G(20)], tol_before=2.0)       # 3 detik SEBELUM kejadian: terlalu awal
    assert r["TP"] == 0 and r["FP"] == 1
    r = evaluate([E(20 + 5)], [G(20)])
    assert r["TP"] == 1
    r = evaluate([E(20 + 7)], [G(20)])
    assert r["TP"] == 0


def test_one_to_one_matching_and_source_isolation():
    r = evaluate([E(11), E(12)], [G(10)])
    assert r["TP"] == 1 and r["FP"] == 1
    r = evaluate([E(11, src="lain.mp4", cam="camx")], [G(10, src="v1.mp4")])
    assert r["TP"] == 0 and r["FN"] == 1
    r = evaluate([E(11, src="", cam="cam_a")], [G(10, src="cam_a")])     # cocok lewat camera_id
    assert r["TP"] == 1


def test_fall_level_selection_collapses_episodes():
    ev = [E(10, "fall_suspected"), E(16, "fall_confirmed"), E(50, "fall_confirmed", track=2)]
    l1 = select_predictions(ev, "fall", 1)
    assert sorted(e["meta"]["stream_ts"] for e in l1) == [10, 50]   # confirmed tanpa suspected tetap dihitung
    assert [e["meta"]["stream_ts"] for e in select_predictions(ev, "fall", 2)] == [16, 50]
    r = evaluate(ev, [G(9)], level=2)
    assert r["TP"] == 1 and r["tol_after_s"] == 20.0


def test_scenarios_and_negative_segments_attribute_false_alarms():
    gt = [G(10, scen="jatuh depan"), G(40, scen="jatuh samping"),
          G(0, "negative", scen="duduk", t_end=30, src="v1.mp4")]
    ev = [E(11), E(20)]                       # 11 benar, 20 salah di segmen 'duduk', jatuh samping terlewat
    r = evaluate(ev, gt, hours=1)
    s = r["per_scenario"]
    assert s["jatuh depan"]["recall"] == 1.0 and s["jatuh samping"]["recall"] == 0.0
    assert s["duduk"]["FP"] == 1 and r["fp_unattributed"] == 0


def test_zero_false_alarm_rule_of_three_and_empty_cases():
    r = evaluate([E(11)], [G(10)], hours=20)
    assert r["FP"] == 0 and abs(r["false_alarm_upper95_per_hour_if_zero"] - 0.15) < 1e-9
    r = evaluate([], [])
    assert r["recall"] is None and r["precision"] is None and r["f1"] is None
    assert "n/a" in format_markdown(r)


def test_safezone_module_and_csv_loader(tmp_path):
    p = tmp_path / "gt.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "event_type", "t_start", "t_end", "scenario"])
        w.writerow(["g.mp4", "zone_breach", "10", "", "pelanggaran"])
        w.writerow(["g.mp4", "negative", "0", "9", "dewasa lewat"])
    gt = load_gt(str(p))
    ev = [{"camera_id": "g", "alert_type": "zone_breach", "track_id": 1, "meta": {"stream_ts": 12, "source": "g.mp4"}},
          {"camera_id": "g", "alert_type": "zone_warning", "track_id": 1, "meta": {"stream_ts": 5, "source": "g.mp4"}}]
    r = evaluate(ev, gt, module="safezone", hours=2)
    assert r["TP"] == 1 and r["FP"] == 0 and r["predictions"] == 1
    assert "Recall" in format_markdown(r)
