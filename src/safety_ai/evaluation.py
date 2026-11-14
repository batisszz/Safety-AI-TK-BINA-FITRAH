"""Evaluasi tingkat event (bukan akurasi per frame).

Berkas ground truth (CSV): source,event_type,t_start,t_end,scenario
  - event_type = fall | zone_breach  -> kejadian positif pada waktu t_start (detik video)
  - event_type = negative            -> segmen TANPA kejadian [t_start, t_end] (mis. duduk, merangkak, dewasa lewat)
  - source cocok dengan meta.source (nama berkas video) ATAU camera_id pada event.
"""
from __future__ import annotations

import csv
import glob
import json
import math
from typing import Optional

import numpy as np


def wilson(k: int, n: int, z: float = 1.96) -> Optional[tuple]:
    """Interval kepercayaan 95% (Wilson) untuk proporsi k/n."""
    if n <= 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def load_events(patterns: str) -> list:
    out = []
    for pat in patterns.split(","):
        for path in sorted(glob.glob(pat.strip())):
            with open(path, encoding="utf-8") as fh:
                out += [json.loads(line) for line in fh if line.strip()]
    return out


def load_gt(path: str) -> list:
    rows = []
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            r["t_start"] = float(r["t_start"])
            r["t_end"] = float(r["t_end"]) if r.get("t_end") not in (None, "") else r["t_start"]
            r["scenario"] = r.get("scenario") or "-"
            rows.append(r)
    return rows


def _src(e: dict) -> str:
    return e.get("meta", {}).get("source") or e["camera_id"]


def _src_match(gt_source: str, e: dict) -> bool:
    return gt_source in (e.get("meta", {}).get("source"), e["camera_id"])


def select_predictions(events: list, module: str, level: int) -> list:
    """Fall level 1: episode pertama (suspected, atau confirmed tanpa suspected); level 2: hanya confirmed."""
    if module == "safezone":
        return [e for e in events if e["alert_type"] == "zone_breach"]
    if level == 2:
        return [e for e in events if e["alert_type"] == "fall_confirmed"]
    suspected = [e for e in events if e["alert_type"] == "fall_suspected"]
    out = list(suspected)
    for e in events:
        if e["alert_type"] != "fall_confirmed":
            continue
        t = e["meta"]["stream_ts"]
        has_l1 = any(s["camera_id"] == e["camera_id"] and s["track_id"] == e["track_id"]
                     and 0 <= t - s["meta"]["stream_ts"] <= 60 for s in suspected)
        if not has_l1:
            out.append(e)
    return out


def evaluate(events: list, gt: list, module: str = "fall", level: int = 1, tol_before: float = 2.0,
             tol_after: Optional[float] = None, hours: Optional[float] = None) -> dict:
    pos_type = "fall" if module == "fall" else "zone_breach"
    if tol_after is None:
        tol_after = 20.0 if (module == "fall" and level == 2) else 6.0
    preds = sorted(select_predictions(events, module, level), key=lambda e: e["meta"]["stream_ts"])
    positives = sorted([g for g in gt if g["event_type"] == pos_type], key=lambda g: g["t_start"])
    negatives = [g for g in gt if g["event_type"] == "negative"]

    used = set()
    matches = []   # (gt, pred)
    for g in positives:
        for i, e in enumerate(preds):
            if i in used or not _src_match(g["source"], e):
                continue
            dt = e["meta"]["stream_ts"] - g["t_start"]
            if -tol_before <= dt <= tol_after:
                used.add(i)
                matches.append((g, e))
                break

    tp = len(matches)
    fn = len(positives) - tp
    fp_events = [e for i, e in enumerate(preds) if i not in used]
    fp = len(fp_events)
    delays = [e["meta"]["stream_ts"] - g["t_start"] for g, e in matches]

    recall = tp / len(positives) if positives else None
    precision = tp / (tp + fp) if (tp + fp) else None
    if precision is None or recall is None:
        f1 = None
    else:
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    far = (fp / hours) if hours else None

    scenarios: dict = {}
    for g in positives:
        s = scenarios.setdefault(g["scenario"], {"jumlah": 0, "TP": 0, "FN": 0, "FP": 0})
        s["jumlah"] += 1
    for g, _ in matches:
        scenarios[g["scenario"]]["TP"] += 1
    for s in scenarios.values():
        s["FN"] = s["jumlah"] - s["TP"]
        s["recall"] = s["TP"] / s["jumlah"] if s["jumlah"] else None
    unattributed = 0
    for e in fp_events:
        hit = False
        for n in negatives:
            if _src_match(n["source"], e) and n["t_start"] <= e["meta"]["stream_ts"] <= n["t_end"]:
                s = scenarios.setdefault(n["scenario"], {"jumlah": 0, "TP": 0, "FN": 0, "FP": 0, "recall": None})
                s["FP"] += 1
                hit = True
                break
        if not hit:
            unattributed += 1
    for n in negatives:
        scenarios.setdefault(n["scenario"], {"jumlah": 0, "TP": 0, "FN": 0, "FP": 0, "recall": None})

    return {
        "module": module, "level": level, "tol_before_s": tol_before, "tol_after_s": tol_after,
        "positives": len(positives), "predictions": len(preds), "TP": tp, "FN": fn, "FP": fp,
        "recall": recall, "recall_ci95": wilson(tp, len(positives)), "precision": precision, "f1": f1,
        "hours": hours, "false_alarms_per_hour": far,
        "false_alarms_per_school_day_4h": (far * 4 if far is not None else None),
        "false_alarm_upper95_per_hour_if_zero": (3.0 / hours if (hours and fp == 0) else None),
        "delay_median_s": float(np.median(delays)) if delays else None,
        "delay_p95_s": float(np.percentile(delays, 95)) if delays else None,
        "per_scenario": scenarios, "fp_unattributed": unattributed,
    }


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _num(x, d=2) -> str:
    return "n/a" if x is None else f"{x:.{d}f}"


def format_markdown(r: dict) -> str:
    ci = r["recall_ci95"]
    lines = [
        f"### Evaluasi {r['module']} (level {r['level']}), toleransi waktu -{r['tol_before_s']} s sampai +{r['tol_after_s']} s",
        "",
        "| Metrik | Nilai |", "|---|---|",
        f"| Kejadian nyata (GT) | {r['positives']} |",
        f"| Prediksi | {r['predictions']} |",
        f"| TP / FN / FP | {r['TP']} / {r['FN']} / {r['FP']} |",
        f"| Recall | {_pct(r['recall'])} (CI 95%: {_pct(ci[0]) + ' - ' + _pct(ci[1]) if ci else 'n/a'}) |",
        f"| Precision | {_pct(r['precision'])} |",
        f"| F1 | {_pct(r['f1'])} |",
        f"| Alarm palsu per jam | {_num(r['false_alarms_per_hour'])} (jam rekaman: {r['hours']}) |",
        f"| Alarm palsu per hari sekolah (4 jam) | {_num(r['false_alarms_per_school_day_4h'])} |",
        f"| Delay median / P95 (detik) | {_num(r['delay_median_s'])} / {_num(r['delay_p95_s'])} |",
    ]
    if r["false_alarm_upper95_per_hour_if_zero"] is not None:
        lines.append(f"| Batas atas 95% alarm palsu per jam (0 FP, aturan tiga) | {_num(r['false_alarm_upper95_per_hour_if_zero'])} |")
    lines += ["", "| Skenario | Jumlah | TP | FN | FP | Recall |", "|---|---|---|---|---|---|"]
    for name, s in sorted(r["per_scenario"].items()):
        lines.append(f"| {name} | {s['jumlah']} | {s['TP']} | {s['FN']} | {s['FP']} | {_pct(s.get('recall'))} |")
    lines.append(f"| (FP di luar segmen negatif) | - | - | - | {r['fp_unattributed']} | - |")
    return "\n".join(lines)
