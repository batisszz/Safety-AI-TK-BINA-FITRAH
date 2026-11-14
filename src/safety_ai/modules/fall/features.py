"""Fitur geometri tubuh dari keypoint COCO-17. Fungsi murni, diuji dengan keypoint sintetis."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ...types import (L_EAR, L_EYE, L_HIP, L_SHOULDER, NOSE, R_EAR, R_EYE, R_HIP, R_SHOULDER)

HEAD_IDS = (NOSE, L_EYE, R_EYE, L_EAR, R_EAR)


def visible_mask(kp: np.ndarray, thr: float) -> np.ndarray:
    return kp[:, 2] >= thr


def mid_point(kp: np.ndarray, ids: tuple, thr: float) -> Optional[np.ndarray]:
    """Rata-rata titik yang terlihat (conf >= thr), None jika tidak ada satu pun."""
    pts = [kp[i, :2] for i in ids if kp[i, 2] >= thr]
    if not pts:
        return None
    return np.mean(pts, axis=0)


def torso_angle_deg(shoulder_mid: np.ndarray, hip_mid: np.ndarray) -> float:
    """Sudut vektor bahu->pinggul terhadap sumbu vertikal gambar. 0 = tegak, 90 = horizontal."""
    dx = float(hip_mid[0] - shoulder_mid[0])
    dy = float(hip_mid[1] - shoulder_mid[1])
    return math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))


def body_points(kp: np.ndarray, thr: float):
    """Kembalikan (shoulder_mid, hip_mid, head_y) atau None jika bahu/pinggul tidak terbaca.

    head_y memakai rata-rata hidung, mata, telinga; cadangannya y bahu.
    """
    sh = mid_point(kp, (L_SHOULDER, R_SHOULDER), thr)
    hip = mid_point(kp, (L_HIP, R_HIP), thr)
    if sh is None or hip is None:
        return None
    head = mid_point(kp, HEAD_IDS, thr)
    head_y = float(head[1]) if head is not None else float(sh[1])
    return sh, hip, head_y


def mean_visible_conf(kp: np.ndarray, thr: float) -> float:
    m = visible_mask(kp, thr)
    return float(kp[m, 2].mean()) if m.any() else 0.0


def smooth_keypoints(prev: Optional[np.ndarray], raw: np.ndarray, thr: float, alpha: float) -> np.ndarray:
    """EMA pada koordinat joint yang terlihat. Joint tidak terlihat mempertahankan nilai lama."""
    if prev is None:
        return raw.copy()
    out = prev.copy()
    vis = raw[:, 2] >= thr
    prev_ok = prev[:, 2] >= thr
    for i in range(len(raw)):
        if vis[i]:
            if prev_ok[i]:
                out[i, :2] = alpha * raw[i, :2] + (1 - alpha) * prev[i, :2]
            else:
                out[i, :2] = raw[i, :2]
        out[i, 2] = raw[i, 2]
    return out
