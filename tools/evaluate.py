"""Evaluasi event: recall, precision, F1, alarm palsu per jam, delay, interval kepercayaan, per skenario.

Contoh:
  python tools/evaluate.py --module fall --events "outputs/run01/events_*.jsonl" --gt data/labels/fall_gt.csv --hours 20
  python tools/evaluate.py --module safezone --events "outputs/run02/events_*.jsonl" --gt data/labels/zone_gt.csv --hours 10
"""
import _bootstrap  # noqa: F401
import argparse
import json

from safety_ai.evaluation import evaluate, format_markdown, load_events, load_gt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--module", choices=["fall", "safezone"], required=True)
    ap.add_argument("--events", required=True, help="glob berkas JSONL (pisahkan dengan koma)")
    ap.add_argument("--gt", required=True, help="CSV ground truth")
    ap.add_argument("--level", type=int, choices=[1, 2], default=1, help="fall: 1 = dugaan jatuh, 2 = tidak bangkit")
    ap.add_argument("--tol-before", type=float, default=2.0, help="detik: alert boleh lebih awal dari GT sejauh ini")
    ap.add_argument("--tol-after", type=float, default=None, help="detik: alert boleh lebih lambat (default 6, fall L2 20)")
    ap.add_argument("--hours", type=float, default=None, help="total jam rekaman yang dievaluasi (untuk alarm palsu/jam)")
    ap.add_argument("--out", help="simpan hasil JSON")
    a = ap.parse_args()
    events = load_events(a.events)
    if not events:
        print("[WARN] tidak ada event yang terbaca dari", a.events)
    r = evaluate(events, load_gt(a.gt), a.module, a.level, a.tol_before, a.tol_after, a.hours)
    print(format_markdown(r))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(r, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
