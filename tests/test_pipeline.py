import json
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from safety_ai.config import InferConfig, in_windows, load_config, parse_config
from safety_ai.inference.backend import ScriptedBackend, parse_result
from safety_ai.service.health import DriftDetector
from safety_ai.service.runner import main
from safety_ai.sources.sources import FileSource, LiveSource
from safety_ai.testing.simulate import run_simulation


# ------------------------------------------------------------------ end-to-end (tanpa model)
def test_end_to_end_simulation(tmp_path):
    r = run_simulation(str(tmp_path))
    by_cam = {}
    for e in r["events"]:
        by_cam.setdefault(e["camera_id"], []).append(e)
    kelas = [(e["alert_type"], e["track_id"]) for e in by_cam["cam_kelas_sim"]]
    assert kelas == [("fall_suspected", 1), ("fall_confirmed", 1)]
    gerbang = [(e["alert_type"], e["track_id"]) for e in by_cam["cam_gerbang_sim"]]
    assert gerbang == [("zone_warning", 1), ("zone_breach", 1)]      # dewasa dan pejalan kaki tidak memicu
    for e in r["events"]:
        assert e["timestamp"].startswith("2026-11-12T09:00") and e["timestamp"].endswith("+07:00")
        assert e["meta"]["source"].endswith(".avi")
        if e["level"] > 0:
            assert (tmp_path / "runs/sim" / e["snapshot_path"]).exists()
    assert r["stats"]["cam_kelas_sim"]["frames"] > 100


def test_simulation_is_deterministic(tmp_path):
    a = run_simulation(str(tmp_path / "a"))["events"]
    b = run_simulation(str(tmp_path / "b"))["events"]
    strip = lambda ev: [(e["camera_id"], e["alert_type"], e["track_id"], e["meta"]["stream_ts"]) for e in ev]
    assert strip(a) == strip(b)


# ------------------------------------------------------------------ sumber video
def _make_video(path, n=100, fps=25):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (160, 120))
    for i in range(n):
        vw.write(np.full((120, 160, 3), i, np.uint8))
    vw.release()


def test_file_source_samples_by_stream_time(tmp_path):
    p = tmp_path / "v.avi"
    _make_video(p, n=100, fps=25)
    src = FileSource(str(p), sample_fps=5)          # 25/5 = ambil tiap 5 frame
    ts = []
    while (f := src.read()) is not None:
        ts.append(f.ts)
    assert len(ts) == 20 and abs(ts[1] - 0.2) < 1e-6 and src.finished
    src.close()


def test_file_source_loop_keeps_time_monotonic(tmp_path):
    p = tmp_path / "v.avi"
    _make_video(p, n=50, fps=25)
    src = FileSource(str(p), sample_fps=5, loop=True)
    ts = [src.read().ts for _ in range(30)]          # lebih dari satu putaran (10 frame/putaran)
    assert all(b > a for a, b in zip(ts, ts[1:]))
    src.close()


def test_file_source_missing_file():
    with pytest.raises(FileNotFoundError):
        FileSource("tidak_ada.mp4", 5)


class _FakeCap:
    def __init__(self, fail_after=None):
        self.n, self.fail_after = 0, fail_after

    def isOpened(self):
        return True

    def read(self):
        time.sleep(0.01)
        self.n += 1
        if self.fail_after is not None and self.n > self.fail_after:
            return False, None
        return True, np.zeros((60, 80, 3), np.uint8)

    def release(self):
        pass


def test_live_source_returns_latest_frame_and_respects_interval():
    src = LiveSource(lambda: _FakeCap(), sample_fps=10)
    t0 = time.monotonic()
    frames = [src.read(timeout=1.0) for _ in range(5)]
    dt = time.monotonic() - t0
    assert all(f is not None for f in frames)
    assert dt >= 0.35                                # 5 frame pada 10 fps tidak boleh lebih cepat dari ~0,4 s
    src.close()


def test_live_source_reconnects_after_failure():
    opened = []

    def opener():
        opened.append(1)
        return _FakeCap(fail_after=3 if len(opened) == 1 else None)

    src = LiveSource(opener, sample_fps=20, reconnect_max_s=1.0)
    got = 0
    end = time.monotonic() + 4.0
    while time.monotonic() < end and got < 12:
        if src.read(timeout=0.5) is not None:
            got += 1
    assert got >= 12 and len(opened) >= 2            # putus lalu tersambung lagi
    src.close()


# ------------------------------------------------------------------ konfigurasi
def test_profile_low_cpu_and_override():
    c = InferConfig(profile="low_cpu")
    assert (c.imgsz, c.fps) == (320, 3.0)
    c = InferConfig(profile="low_cpu", imgsz=416)
    assert (c.imgsz, c.fps) == (416, 3.0)
    with pytest.raises(ValueError):
        InferConfig(imgsz=450)


def test_config_validation_errors():
    base = {"cameras": [{"id": "a", "modules": ["safezone"], "source": {"type": "file", "path": "x.mp4"}}]}
    with pytest.raises(ValueError):
        parse_config(base)                                        # safezone tanpa roi
    with pytest.raises(ValueError):
        parse_config({"cameras": [{"id": "a", "modules": ["fall"], "source": {"type": "file", "path": "x.mp4"},
                                   "infer": {"model": "yolov8n"}}]})   # fall butuh model pose
    with pytest.raises(ValueError):
        parse_config({"cameras": [{"id": "a", "modules": ["fall"], "source": {"type": "rtsp"}}]})


def test_env_expansion_and_roi_file(tmp_path, monkeypatch):
    monkeypatch.setenv("BACKEND_URL", "http://x/api")
    (tmp_path / "roi.json").write_text(json.dumps({"safe_polygon": [[0.1, 0.9], [0.1, 0.3], [0.9, 0.3]]}))
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({
        "sinks": [{"type": "http", "url": "${BACKEND_URL}", "api_key": "${API_KEY:-kosong}"}],
        "cameras": [{"id": "g", "modules": ["safezone"], "source": {"type": "file", "path": "x.mp4"},
                     "infer": {"model": "yolov8n"}, "roi": {"file": str(tmp_path / "roi.json")}}]}))
    app = load_config(str(cfg))
    assert app.sinks[0].url == "http://x/api" and app.sinks[0].api_key == "kosong"
    assert len(app.camera("g").roi.safe_polygon) == 3


def test_in_windows():
    from datetime import datetime
    assert in_windows(datetime(2026, 1, 1, 7, 30), ["07:00-08:00"])
    assert not in_windows(datetime(2026, 1, 1, 9, 0), ["07:00-08:00"])
    assert in_windows(datetime(2026, 1, 1, 9, 0), [])
    assert not in_windows(datetime(2026, 1, 1, 9, 0), [], default_if_empty=False)


# ------------------------------------------------------------------ parser hasil Ultralytics
def test_parse_result_pose_with_offset_and_missing_ids():
    boxes = SimpleNamespace(xyxy=np.array([[10, 20, 50, 120.0]]), conf=np.array([0.9]), cls=np.array([0.0]),
                            id=np.array([5.0]))
    kp = np.zeros((1, 17, 3))
    kp[0, :, 0], kp[0, :, 1], kp[0, :, 2] = 30, 60, 0.8
    res = SimpleNamespace(boxes=boxes, keypoints=SimpleNamespace(data=kp))
    tr = parse_result(res, offset=(100, 200), names={0: "person"})[0]
    assert tr.id == 5 and tr.bbox == (110, 220, 150, 320) and tr.keypoints[0, 0] == 130 and tr.keypoints[0, 1] == 260
    assert tr.cls_name == "person"
    boxes_noid = SimpleNamespace(xyxy=boxes.xyxy, conf=boxes.conf, cls=boxes.cls, id=None)
    assert parse_result(SimpleNamespace(boxes=boxes_noid, keypoints=None)) == []


def test_parse_result_child_adult_labels():
    boxes = SimpleNamespace(xyxy=np.array([[0, 0, 10, 30.0], [20, 0, 40, 80.0]]), conf=np.array([0.9, 0.8]),
                            cls=np.array([0.0, 1.0]), id=np.array([1.0, 2.0]))
    out = parse_result(SimpleNamespace(boxes=boxes, keypoints=None), names={0: "child", 1: "adult"})
    assert [t.cls_name for t in out] == ["child", "adult"] and out[0].keypoints is None


# ------------------------------------------------------------------ drift
def test_drift_detector_flags_shifted_camera_and_ignores_static():
    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur((rng.random((240, 320)) * 255).astype(np.uint8), (0, 0), 2)
    img = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    det = DriftDetector(img, thresh_frac=0.03, interval_s=0)
    assert det.check(img, 0) is None and det.check(img, 1) is None
    shifted = np.roll(img, 20, axis=1)                  # geser ~6% lebar
    assert det.check(shifted, 2) is None                # butuh 2 cek berturut-turut
    out = det.check(shifted, 3)
    assert out and out["state"] == "camera_drift"
    out = det.check(img, 4)
    assert out and out["state"] == "camera_aligned"


# ------------------------------------------------------------------ CLI runner
def test_cli_runner_with_stubbed_yolo(tmp_path, monkeypatch):
    video = tmp_path / "k.avi"
    from safety_ai.testing.simulate import class_scenario, render_video
    fn, dur = class_scenario()
    render_video(video, fn, 25.0)

    class StubYolo:
        def __init__(self, cfg, models_dir="models"):
            self.inner = ScriptedBackend(fn)

        def infer(self, image, ts, offset=(0, 0)):
            return self.inner.infer(image, ts, offset)

    monkeypatch.setattr("safety_ai.inference.backend.YoloBackend", StubYolo)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({
        "output_dir": str(tmp_path / "out"),
        "sinks": [{"type": "jsonl", "path": str(tmp_path / "out/{run}/events_{camera}.jsonl")}],
        "cameras": [{"id": "cam_k", "modules": ["fall"], "source": {"type": "file", "path": "tidak_dipakai.avi"},
                     "infer": {"model": "yolov8n-pose", "profile": "normal"}}]}))
    assert main(["--config", str(cfg), "--camera", "cam_k", "--source", str(video), "--run", "t1"]) == 0
    lines = (tmp_path / "out/t1/events_cam_k.jsonl").read_text().splitlines()
    assert [json.loads(x)["alert_type"] for x in lines] == ["fall_suspected", "fall_confirmed"]
    stats = json.loads((tmp_path / "out/t1/stats_cam_k.json").read_text())
    assert stats["frames"] > 50 and stats["alerts"]["emitted"] == 2


def test_shipped_configs_are_valid():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    app = load_config(str(root / "configs/default.yaml"))
    assert [c.id for c in app.cameras] == ["cam_kelas_a", "cam_gerbang"]
    assert app.camera("cam_kelas_a").infer.imgsz == 320 and app.camera("cam_gerbang").infer.imgsz == 480
    # cameras.example.yaml memakai roi.file yang belum ada: cukup pastikan struktur YAML valid saat file ROI tersedia
    import json, yaml
    raw = yaml.safe_load((root / "configs/cameras.example.yaml").read_text())
    assert {c["id"] for c in raw["cameras"]} == {"cam_kelas_a", "cam_gerbang"}
