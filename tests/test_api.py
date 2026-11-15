"""Unit test untuk API safety-ai: AlertStore, EventBroadcaster, Annotator, dan CameraStreamer."""
from __future__ import annotations

import numpy as np
import pytest

from safety_ai.api.alerts import AlertStore, BroadcasterSink, EventBroadcaster
from safety_ai.api.annotator import Annotator
from safety_ai.config import FallParams, ZoneParams
from safety_ai.modules.fall.module import FallModule
from safety_ai.modules.safezone.module import SafeZoneModule
from safety_ai.types import Frame, FrameContext, Track


def test_alert_store_crud():
    store = AlertStore(max_items=10)
    ev = {
        "event_id": "test-uuid-1",
        "camera_id": "cam_kelas_a",
        "alert_type": "fall_confirmed",
        "level": 2,
        "track_id": 5,
        "confidence": 0.88,
        "timestamp": "2026-11-14T10:00:00+07:00",
        "snapshot_path": "snapshots/cam_kelas_a/test.jpg"
    }

    # Add alert
    saved = store.add(ev)
    assert saved["event_id"] == "test-uuid-1"
    assert saved["status"] == "Unresolved"

    # Get all
    items = store.get_all()
    assert len(items) == 1
    assert items[0]["track_id"] == 5

    # Filter
    assert len(store.get_all(camera_id="cam_gerbang")) == 0
    assert len(store.get_all(alert_type="fall_confirmed")) == 1
    assert len(store.get_all(status="Unresolved")) == 1

    # Update status to Acknowledged
    updated = store.update_status("test-uuid-1", "Acknowledged", operator="Bu Guru", note="Sudah ditolong")
    assert updated is not None
    assert updated["status"] == "Acknowledged"
    assert updated["verified_by"] == "Bu Guru"
    assert updated["verification_note"] == "Sudah ditolong"

    # Stats
    stats = store.stats()
    assert stats["total"] == 1
    assert stats["unresolved"] == 0
    assert stats["fall_alerts"] == 1


def test_broadcaster_sink_emits_to_store():
    store = AlertStore()
    broadcaster = EventBroadcaster()

    class DummySink:
        def __init__(self):
            self.emitted = []
        def emit(self, e):
            self.emitted.append(e)

    inner = DummySink()
    sink = BroadcasterSink(store, broadcaster, inner)

    sink.emit({
        "event_id": "ev-broadcaster-1",
        "camera_id": "cam_gerbang",
        "alert_type": "zone_breach",
        "level": 2,
        "track_id": 3
    })

    assert len(store.get_all()) == 1
    assert len(inner.emitted) == 1
    assert inner.emitted[0]["event_id"] == "ev-broadcaster-1"


def test_annotator_draw_fall():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    mod = FallModule("cam_kelas_a", FallParams())

    # Bikin track dengan keypoint dummy
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[:, 2] = 0.8  # conf
    kp[5] = [200, 100, 0.9]  # left shoulder
    kp[6] = [240, 100, 0.9]  # right shoulder
    kp[11] = [200, 200, 0.9] # left hip
    kp[12] = [240, 200, 0.9] # right hip

    tracks = [
        Track(id=1, bbox=(180, 80, 260, 300), conf=0.9, keypoints=kp, cls_name="person")
    ]

    annotated = Annotator.draw_fall_detection(img, tracks, mod, fps=20.0, latency_ms=15.2)
    assert annotated.shape == img.shape
    # Pastikan piksel digambar (tidak hitam polos lagi)
    assert annotated.sum() > 0


def test_annotator_draw_safezone():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    poly = [[0.1, 0.9], [0.1, 0.3], [0.9, 0.3], [0.9, 0.9]]
    mod = SafeZoneModule("cam_gerbang", ZoneParams(safe_polygon=poly, motion_gate=False))

    tracks = [
        Track(id=2, bbox=(200, 150, 280, 350), conf=0.85, cls_name="child")
    ]

    annotated = Annotator.draw_safezone_detection(img, tracks, mod, fps=25.0, latency_ms=12.0)
    assert annotated.shape == img.shape
    assert annotated.sum() > 0
