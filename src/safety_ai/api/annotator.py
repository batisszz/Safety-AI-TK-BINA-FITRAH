"""Modul anotasi visual (Annotator) untuk menampilkan bounding box, skeleton,
poligon safe zone, foot-point, dan status HUD pada frame video.
"""
from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

# Sambungan skeleton COCO-17
SKELETON_PAIRS = [
    # Kepala
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Tubuh atas (bahu & lengan)
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    # Torso (bahu ke pinggul)
    (5, 11), (6, 12), (11, 12),
    # Kaki (pinggul ke lutut ke pergelangan kaki)
    (11, 13), (13, 15), (12, 14), (14, 16)
]

# Warna BGR untuk visualisasi
COLOR_GREEN = (46, 204, 113)     # Normal / Inside
COLOR_YELLOW = (0, 215, 255)     # Warning / Dropping
COLOR_ORANGE = (0, 140, 255)     # Suspected / Pending
COLOR_RED = (40, 40, 255)        # Confirmed / Breach
COLOR_BLUE = (255, 160, 50)      # Resolved / Returned
COLOR_GRAY = (160, 160, 160)     # Outside / Inactive
COLOR_CYAN = (255, 230, 0)       # Arms skeleton
COLOR_LIME = (100, 255, 100)     # Torso skeleton


class Annotator:
    """Kelas pembantu untuk menggambar deteksi AI pada frame kamera."""

    @staticmethod
    def draw_skeleton(img: np.ndarray, keypoints: np.ndarray, min_conf: float = 0.25) -> None:
        """Gambar 17 keypoint COCO dan garis penghubung tubuh."""
        if keypoints is None or len(keypoints) < 17:
            return

        h, w = img.shape[:2]

        # Gambar garis sambungan (limbs)
        for i1, i2 in SKELETON_PAIRS:
            x1, y1, c1 = keypoints[i1]
            x2, y2, c2 = keypoints[i2]
            if c1 >= min_conf and c2 >= min_conf:
                pt1 = (int(np.clip(x1, 0, w - 1)), int(np.clip(y1, 0, h - 1)))
                pt2 = (int(np.clip(x2, 0, w - 1)), int(np.clip(y2, 0, h - 1)))
                # Pilih warna berdasarkan bagian tubuh
                if i1 in (5, 6, 7, 8, 9, 10) and i2 in (5, 6, 7, 8, 9, 10):
                    limb_col = COLOR_CYAN
                elif i1 in (11, 12, 13, 14, 15, 16) and i2 in (11, 12, 13, 14, 15, 16):
                    limb_col = COLOR_YELLOW
                else:
                    limb_col = COLOR_LIME
                cv2.line(img, pt1, pt2, limb_col, 2, cv2.LINE_AA)

        # Gambar titik sendi (keypoints)
        for i, (x, y, conf) in enumerate(keypoints):
            if conf >= min_conf:
                pt = (int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1)))
                cv2.circle(img, pt, 4, (20, 20, 20), -1, cv2.LINE_AA)
                cv2.circle(img, pt, 3, COLOR_GREEN if i > 4 else COLOR_YELLOW, -1, cv2.LINE_AA)

    @staticmethod
    def draw_text_badge(img: np.ndarray, text: str, pt: tuple[int, int],
                        bg_color: tuple[int, int, int], text_color: tuple[int, int, int] = (255, 255, 255),
                        scale: float = 0.45, thickness: int = 1, pad: int = 4) -> None:
        """Gambar badge teks dengan latar belakang persegi bersudut rapi."""
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        x, y = pt
        x1 = max(0, x - pad)
        y1 = max(0, y - th - pad * 2)
        x2 = min(img.shape[1], x + tw + pad)
        y2 = min(img.shape[0], y + pad)

        cv2.rectangle(img, (x1, y1), (x2, y2), bg_color, -1)
        cv2.putText(img, text, (x1 + pad, y2 - pad), cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, thickness, cv2.LINE_AA)

    @classmethod
    def draw_fall_detection(cls, frame_img: np.ndarray, tracks: list, fall_module,
                            fps: float = 0.0, latency_ms: float = 0.0) -> np.ndarray:
        """Anotasi lengkap untuk kamera Fall Detection (Kelas)."""
        img = frame_img.copy()
        h, w = img.shape[:2]

        has_fall_confirmed = False
        has_fall_suspected = False

        for tr in tracks:
            st = fall_module.states.get(tr.id) if fall_module else None
            state_str = st.fsm.state if (st and hasattr(st, "fsm")) else "NORMAL"

            # Tentukan warna status
            if state_str == "CONFIRMED":
                box_color = COLOR_RED
                box_thick = 3
                has_fall_confirmed = True
            elif state_str == "SUSPECTED":
                box_color = COLOR_ORANGE
                box_thick = 2
                has_fall_suspected = True
            elif state_str == "DROPPING":
                box_color = COLOR_YELLOW
                box_thick = 2
            elif state_str == "RESOLVED":
                box_color = COLOR_BLUE
                box_thick = 2
            else:
                box_color = COLOR_GREEN
                box_thick = 2

            # Gambar skeleton jika ada keypoints
            if tr.keypoints is not None:
                cls.draw_skeleton(img, tr.keypoints)

            # Gambar bounding box
            x1, y1, x2, y2 = [int(v) for v in tr.bbox]
            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, box_thick)

            # Label status
            label_main = f"ID:{tr.id} {state_str}"
            cls.draw_text_badge(img, label_main, (x1, max(18, y1 - 4)), box_color)

            # Sub-info: sudut torso, rasio tinggi jika tersedia
            if st and hasattr(st, "samples") and st.samples:
                last_s = st.samples[-1]
                angle_deg = getattr(st.fsm, "_last_angle", None)
                sub_text = f"h:{int(tr.height)}px"
                if angle_deg is not None:
                    sub_text += f" | {int(angle_deg)}deg"
                cls.draw_text_badge(img, sub_text, (x1, y2 + 14), (30, 30, 30), scale=0.38)

        # Gambar HUD atas
        hud_h = 36
        overlay = img[:hud_h, :].copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), (20, 20, 20), -1)
        img[:hud_h, :] = cv2.addWeighted(overlay, 0.75, img[:hud_h, :], 0.25, 0)

        # Teks HUD
        cam_title = "CAM 1: KELAS A [FALL DETECTION]"
        cv2.putText(img, cam_title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Status badge HUD
        if has_fall_confirmed:
            status_text = "BAHAYA: JATUH TERKONFIRMASI!"
            status_color = COLOR_RED
        elif has_fall_suspected:
            status_text = "PERINGATAN: TERDUGA JATUH"
            status_color = COLOR_ORANGE
        else:
            status_text = "STATUS: AMAN"
            status_color = COLOR_GREEN

        cls.draw_text_badge(img, status_text, (w - 240, 25), status_color, scale=0.45, pad=5)

        # Info FPS & Tracks
        info_text = f"FPS: {fps:.1f} | Lat: {latency_ms:.1f}ms | Track: {len(tracks)}"
        cv2.putText(img, info_text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

        return img

    @classmethod
    def draw_safezone_detection(cls, frame_img: np.ndarray, tracks: list, safezone_module,
                                fps: float = 0.0, latency_ms: float = 0.0) -> np.ndarray:
        """Anotasi lengkap untuk kamera Safe Zone Detection (Gerbang)."""
        img = frame_img.copy()
        h, w = img.shape[:2]

        has_breach = False
        has_warning = False

        # Gambar polygon safe zone
        poly = getattr(safezone_module, "p", None)
        poly_pts = getattr(poly, "safe_polygon", None) if poly else None
        if poly_pts and len(poly_pts) >= 3:
            pts_px = np.array([[int(x * w), int(y * h)] for x, y in poly_pts], np.int32)

            # Shading area hijau semi-transparan
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts_px], (46, 204, 113))
            img = cv2.addWeighted(overlay, 0.15, img, 0.85, 0)

            # Garis batas poligon
            cv2.polylines(img, [pts_px], True, (46, 204, 113), 2, cv2.LINE_AA)

            # Label zona aman di dekat titik pertama
            px0, py0 = pts_px[0]
            cls.draw_text_badge(img, "ZONA AMAN TK BINA FITRAH", (px0 + 5, py0 - 5), (46, 160, 80), scale=0.42)

        # Gambar orang dan status zona
        for tr in tracks:
            st = safezone_module.states.get(tr.id) if safezone_module else None
            state_str = st.state if st else "OUTSIDE"
            cls_name = getattr(tr, "cls_name", "person")

            if state_str == "BREACH":
                box_color = COLOR_RED
                box_thick = 3
                has_breach = True
            elif state_str in ("WARNING", "PENDING"):
                box_color = COLOR_YELLOW
                box_thick = 2
                has_warning = True
            elif state_str == "INSIDE":
                box_color = COLOR_GREEN
                box_thick = 2
            elif state_str == "RETURNED":
                box_color = COLOR_BLUE
                box_thick = 2
            else:
                box_color = COLOR_GRAY
                box_thick = 1

            x1, y1, x2, y2 = [int(v) for v in tr.bbox]
            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, box_thick)

            # Titik kaki (foot point)
            fx, fy = int(tr.foot_point[0]), int(tr.foot_point[1])
            cv2.circle(img, (fx, fy), 5, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(img, (fx, fy), 4, box_color, -1, cv2.LINE_AA)
            cv2.line(img, (fx - 7, fy), (fx + 7, fy), box_color, 1, cv2.LINE_AA)
            cv2.line(img, (fx, fy - 7), (fx, fy + 7), box_color, 1, cv2.LINE_AA)

            # Label status
            tag = f"ID:{tr.id} {cls_name} - {state_str}"
            cls.draw_text_badge(img, tag, (x1, max(18, y1 - 4)), box_color)

        # Gambar HUD atas
        hud_h = 36
        overlay = img[:hud_h, :].copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), (20, 20, 20), -1)
        img[:hud_h, :] = cv2.addWeighted(overlay, 0.75, img[:hud_h, :], 0.25, 0)

        # Teks HUD
        cam_title = "CAM 2: GERBANG UTAMA [SAFE ZONE]"
        cv2.putText(img, cam_title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Status badge HUD
        if has_breach:
            status_text = "BAHAYA: KELUAR ZONA AMAN!"
            status_color = COLOR_RED
        elif has_warning:
            status_text = "PERINGATAN: MENDEKATI BATAS"
            status_color = COLOR_YELLOW
        else:
            status_text = "STATUS: AMAN"
            status_color = COLOR_GREEN

        cls.draw_text_badge(img, status_text, (w - 240, 25), status_color, scale=0.45, pad=5)

        # Info FPS & Tracks
        info_text = f"FPS: {fps:.1f} | Lat: {latency_ms:.1f}ms | Track: {len(tracks)}"
        cv2.putText(img, info_text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

        return img
