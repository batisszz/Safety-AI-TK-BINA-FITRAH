"""Unduh bobot pretrained ke models/ dan (opsional) ekspor ke format CPU.

  python tools/download_models.py                         # unduh yolov8n-pose.pt dan yolov8n.pt
  python tools/download_models.py --export openvino       # + ekspor OpenVINO (CPU Intel)
  python tools/download_models.py --export onnx           # + ekspor ONNX (CPU AMD / umum)

Folder hasil ekspor (mis. models/yolov8n-pose_openvino_model) dipakai pada infer.model di YAML.
"""
import _bootstrap  # noqa: F401
import argparse
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--export", choices=["openvino", "onnx"], default=None)
    ap.add_argument("--pose-imgsz", type=int, default=480)
    ap.add_argument("--det-imgsz", type=int, default=640)
    a = ap.parse_args()
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit('Ultralytics belum terpasang. Jalankan: pip install ultralytics')

    d = Path(a.models_dir)
    d.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(d)  # Ultralytics mengunduh ke direktori kerja
    try:
        for name, imgsz in (("yolov8n-pose", a.pose_imgsz), ("yolov8n", a.det_imgsz)):
            m = YOLO(f"{name}.pt")
            print("OK:", d / f"{name}.pt")
            if a.export:
                out = m.export(format=a.export, imgsz=imgsz)
                print("Ekspor:", out)
    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    main()
