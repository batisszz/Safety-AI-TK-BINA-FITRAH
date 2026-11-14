"""Simulasi end-to-end TANPA model dan TANPA kamera (smoke test integrasi dan demo untuk tim lain).

  python tools/simulate.py --out outputs/sim

Membuat video sintetis, menjalankannya lewat runner asli, dan menampilkan event yang dihasilkan.
Ini membuktikan jalur kode (sumber -> modul -> alert -> sink) benar. Ini BUKAN uji kualitas deteksi YOLO.
"""
import _bootstrap  # noqa: F401
import argparse
import json

from safety_ai.testing.simulate import run_simulation


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="outputs/sim")
    a = ap.parse_args()
    r = run_simulation(a.out)
    print("\nRingkasan event:")
    for e in r["events"]:
        print(f"  {e['camera_id']:<16} {e['alert_type']:<15} L{e['level']} track={e['track_id']} t={e['meta']['stream_ts']}s")
    print("\nFolder hasil:", r["dir"])
    print(json.dumps({k: v["alerts"] for k, v in r["stats"].items()}))


if __name__ == "__main__":
    main()
