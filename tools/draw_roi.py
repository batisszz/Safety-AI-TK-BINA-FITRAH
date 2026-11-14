"""Gambar poligon zona aman pada frame kamera gerbang (klik), atau berikan poligon lewat --polygon.

Interaktif (butuh opencv-python dengan GUI, bukan versi headless):
  python tools/draw_roi.py --source data/gerbang.mp4 --camera cam_gerbang --out configs/roi_cam_gerbang.json
  Klik kiri = tambah titik | u = batalkan titik | c = hapus semua | s = simpan | q = keluar tanpa simpan

Tanpa GUI:
  python tools/draw_roi.py --source data/gerbang.mp4 --camera cam_gerbang --polygon "0.1,0.9;0.1,0.3;0.9,0.3;0.9,0.9" \
      --out configs/roi_cam_gerbang.json --preview outputs/roi_preview.png

--out berakhiran .json = berkas ROI (rujuk lewat roi.file di YAML). Berakhiran .yaml = memperbarui roi.safe_polygon
kamera tsb di YAML (komentar YAML akan hilang). Frame referensi disimpan untuk pemeriksaan kamera bergeser.
"""
import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

import cv2

from safety_ai.toolkit import grab_first_frame, parse_polygon, pixels_to_norm, render_preview, write_roi


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--camera", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--polygon", help="'x,y;x,y;...' ternormalisasi 0..1 (melewati GUI)")
    ap.add_argument("--preview", help="simpan gambar pratinjau (PNG)")
    ap.add_argument("--ref-out", help="simpan frame referensi (default: di samping --out)")
    a = ap.parse_args()

    frame = grab_first_frame(a.source)
    h, w = frame.shape[:2]
    if a.polygon:
        poly = parse_polygon(a.polygon)
    else:
        pts = []

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                pts.append((x, y))

        win = "draw_roi"
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, on_mouse)
        poly = None
        while True:
            vis = frame.copy()
            for p in pts:
                cv2.circle(vis, p, 4, (0, 255, 255), -1)
            if len(pts) >= 2:
                cv2.polylines(vis, [__import__("numpy").array(pts, "int32")], len(pts) >= 3, (0, 255, 0), 2)
            cv2.imshow(win, vis)
            k = cv2.waitKey(30) & 0xFF
            if k == ord("u") and pts:
                pts.pop()
            elif k == ord("c"):
                pts.clear()
            elif k == ord("s") and len(pts) >= 3:
                poly = pixels_to_norm(pts, w, h)
                break
            elif k in (ord("q"), 27):
                break
        cv2.destroyAllWindows()
        if poly is None:
            print("Dibatalkan, tidak ada yang disimpan.")
            return
    saved = write_roi(a.out, a.camera, poly)
    ref = a.ref_out or str(Path(a.out).with_name(f"ref_{a.camera}.jpg"))
    cv2.imwrite(ref, frame)
    print(f"ROI disimpan: {saved}\nFrame referensi: {ref}\nPoligon: {poly}")
    if a.preview:
        Path(a.preview).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(a.preview, render_preview(frame, poly))
        print("Pratinjau:", a.preview)


if __name__ == "__main__":
    main()
