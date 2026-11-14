"""Rekam klip uji dari webcam / RTSP / berkas.

  python tools/record_clip.py --source 0 --seconds 120 --out data/raw/sesi01.mp4
  python tools/record_clip.py --source rtsp://localhost:8554/kelasA --seconds 300 --out data/raw/kelasA_01.mp4

Data anak hanya boleh direkam dengan izin sekolah dan orang tua. Folder data/ sudah di-.gitignore.
"""
import _bootstrap  # noqa: F401
import argparse

from safety_ai.toolkit import record_clip


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=None)
    a = ap.parse_args()
    print(record_clip(a.source, a.seconds, a.out, a.fps))


if __name__ == "__main__":
    main()
