import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from safety_ai.toolkit import (parse_polygon, percentiles, pixels_to_norm, recommend_fps, record_clip,
                               render_preview, write_roi)

ROOT = Path(__file__).resolve().parents[1]


def test_parse_polygon_valid_and_invalid():
    assert parse_polygon("0.1,0.9;0.1,0.3;0.9,0.3") == [[0.1, 0.9], [0.1, 0.3], [0.9, 0.3]]
    for bad in ("0.1,0.9;0.1,0.3", "0.1,0.9;0.1,0.3;1.5,0.3", "a,b;c,d;e,f"):
        with pytest.raises(ValueError):
            parse_polygon(bad)
    assert pixels_to_norm([(64, 144), (576, 432)], 640, 480) == [[0.1, 0.3], [0.9, 0.9]]


def test_write_roi_json_and_yaml(tmp_path):
    poly = [[0.1, 0.9], [0.1, 0.3], [0.9, 0.3]]
    j = write_roi(str(tmp_path / "roi.json"), "cam_g", poly)
    assert json.loads(Path(j).read_text())["safe_polygon"] == poly
    y = tmp_path / "cams.yaml"
    y.write_text(yaml.safe_dump({"cameras": [{"id": "cam_g", "modules": ["safezone"]}]}))
    write_roi(str(y), "cam_g", poly)
    assert yaml.safe_load(y.read_text())["cameras"][0]["roi"]["safe_polygon"] == poly
    with pytest.raises(KeyError):
        write_roi(str(y), "tidak_ada", poly)


def test_render_preview_draws_something():
    img = np.zeros((240, 320, 3), np.uint8)
    out = render_preview(img, [[0.1, 0.9], [0.1, 0.3], [0.9, 0.3], [0.9, 0.9]])
    assert out.shape == img.shape and out.sum() > 0 and img.sum() == 0


def test_record_clip_from_file(tmp_path):
    src = tmp_path / "s.avi"
    vw = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"MJPG"), 10, (80, 60))
    for i in range(50):
        vw.write(np.full((60, 80, 3), i, np.uint8))
    vw.release()
    r = record_clip(str(src), 2.0, str(tmp_path / "o" / "c.avi"))
    assert r["frames"] == 20 and Path(r["path"]).exists()


def test_recommend_fps_and_percentiles():
    r = recommend_fps(100.0, 2, budget=0.6)        # 100 ms/frame -> 6 fps total -> 3 fps per kamera
    assert r["max_total_fps"] == 6.0 and r["fps_per_camera"] == 3.0 and r["profile"] == "low_cpu"
    assert recommend_fps(40.0, 2)["profile"] == "normal"
    assert recommend_fps(500.0, 2)["profile"] == "terlalu_berat"
    with pytest.raises(ValueError):
        recommend_fps(0, 1)
    assert percentiles([1, 2, 3, 4, 100])["p50"] == 3


def test_cli_draw_roi_headless_and_evaluate(tmp_path):
    vid = tmp_path / "g.avi"
    vw = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 120))
    for i in range(10):
        vw.write(np.full((120, 160, 3), 50, np.uint8))
    vw.release()
    out = tmp_path / "roi.json"
    r = subprocess.run([sys.executable, str(ROOT / "tools/draw_roi.py"), "--source", str(vid), "--camera", "cam_g",
                        "--polygon", "0.1,0.9;0.1,0.3;0.9,0.3;0.9,0.9", "--out", str(out),
                        "--preview", str(tmp_path / "p.png")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert out.exists() and (tmp_path / "p.png").exists() and (tmp_path / "ref_cam_g.jpg").exists()

    ev = tmp_path / "events_x.jsonl"
    ev.write_text(json.dumps({"camera_id": "cam_g", "alert_type": "zone_breach", "track_id": 1,
                              "meta": {"stream_ts": 11.0, "source": "g.avi"}}) + "\n")
    gt = tmp_path / "gt.csv"
    gt.write_text("source,event_type,t_start,t_end,scenario\ng.avi,zone_breach,10,,pelanggaran\n")
    r = subprocess.run([sys.executable, str(ROOT / "tools/evaluate.py"), "--module", "safezone", "--events",
                        str(ev), "--gt", str(gt), "--hours", "1", "--out", str(tmp_path / "r.json")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "Recall" in r.stdout and json.loads((tmp_path / "r.json").read_text())["TP"] == 1
