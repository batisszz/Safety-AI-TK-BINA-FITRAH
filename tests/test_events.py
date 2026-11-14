import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

from safety_ai.config import AlertsConfig, HttpSinkCfg
from safety_ai.events.manager import AlertManager
from safety_ai.events.sinks import HttpSink, JsonlSink, MultiSink
from safety_ai.types import Event, Frame

REQUIRED = {"event_id", "camera_id", "alert_type", "level", "track_id", "confidence", "timestamp",
            "snapshot_path", "meta"}


class ListSink:
    def __init__(self):
        self.items = []

    def emit(self, e):
        self.items.append(e)


def ev(t, typ="fall_suspected", level=1, track=7, cam="cam_a"):
    return Event(alert_type=typ, level=level, camera_id=cam, track_id=track, confidence=0.8, stream_ts=t,
                 wall=1_800_000_000.0 + t, bbox=(10, 10, 80, 160))


def frame():
    return Frame(np.zeros((240, 320, 3), np.uint8), 0.0, 0, 1_800_000_000.0)


def test_event_contract_fields_and_timestamp(tmp_path):
    sink = ListSink()
    mgr = AlertManager(AlertsConfig(), sink, base_dir=str(tmp_path))
    mgr.handle([ev(1.0)], frame())
    d = sink.items[0]
    assert REQUIRED <= set(d)
    assert len(d["event_id"]) == 36
    assert d["timestamp"].endswith("+07:00") and d["meta"]["stream_ts"] == 1.0
    assert d["snapshot_path"] and (tmp_path / d["snapshot_path"]).exists()
    json.dumps(d)  # harus bisa diserialisasi


def test_level0_has_no_snapshot(tmp_path):
    sink = ListSink()
    AlertManager(AlertsConfig(), sink, base_dir=str(tmp_path)).handle([ev(1.0, "zone_warning", 0)], frame())
    assert sink.items[0]["snapshot_path"] is None


def test_cooldown_per_track_and_type(tmp_path):
    sink = ListSink()
    mgr = AlertManager(AlertsConfig(snapshots=False), sink, base_dir=str(tmp_path))
    mgr.handle([ev(1.0)], frame())
    mgr.handle([ev(10.0)], frame())                       # dalam cooldown 45 s
    mgr.handle([ev(10.0, track=8)], frame())              # track lain: lolos
    mgr.handle([ev(10.0, "fall_confirmed", 2)], frame())  # tipe lain: lolos
    mgr.handle([ev(50.0)], frame())                       # lewat cooldown
    assert len(sink.items) == 4 and mgr.stats["cooldown_dropped"] == 1


def test_rate_limit_but_system_health_always_passes(tmp_path):
    sink = ListSink()
    mgr = AlertManager(AlertsConfig(snapshots=False, max_per_minute=3), sink, base_dir=str(tmp_path))
    mgr.handle([ev(1.0, track=i) for i in range(10)], frame())
    assert len(sink.items) == 3 and mgr.stats["rate_dropped"] == 7
    mgr.handle([ev(2.0, "system_health", 0, track=None)], frame())
    assert len(sink.items) == 4


def test_jsonl_sink_appends_lines(tmp_path):
    p = tmp_path / "x" / "events.jsonl"
    s = JsonlSink(str(p))
    s.emit({"a": 1})
    s.emit({"a": 2})
    s.close()
    assert [json.loads(line)["a"] for line in p.read_text().splitlines()] == [1, 2]


# ------------------------------------------------------------------ HTTP sink + outbox
class _Backend:
    """Server uji: gagal N kali pertama (500), lalu terima. Duplikat event_id dibalas 409."""

    def __init__(self, fail_first=0):
        self.fail_first = fail_first
        self.received = []
        self.keys = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.keys.append(self.headers.get("X-API-Key"))
                if outer.fail_first > 0:
                    outer.fail_first -= 1
                    self.send_response(500)
                elif body["event_id"] in {b["event_id"] for b in outer.received}:
                    self.send_response(409)
                else:
                    outer.received.append(body)
                    self.send_response(201)
                self.end_headers()

            def log_message(self, *a):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_port}/api/safety-events"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self):
        self.server.shutdown()


def _cfg(url, tmp_path):
    return HttpSinkCfg(url=url, api_key="rahasia", outbox=str(tmp_path / "o.db"), timeout_s=2.0, retry_base_s=0.01)


def test_http_sink_delivers_with_api_key(tmp_path):
    be = _Backend()
    sink = HttpSink(_cfg(be.url, tmp_path), start_worker=False)
    sink.emit({"event_id": "e1", "x": 1})
    assert sink.flush(3.0)
    assert be.received[0]["x"] == 1 and be.keys[0] == "rahasia"
    sink.close()
    be.stop()


def test_http_sink_retries_after_backend_error_and_keeps_order(tmp_path):
    be = _Backend(fail_first=2)
    sink = HttpSink(_cfg(be.url, tmp_path), start_worker=False)
    for i in range(3):
        sink.emit({"event_id": f"e{i}"})
    assert sink.flush(5.0)
    assert sorted(b["event_id"] for b in be.received) == ["e0", "e1", "e2"]
    sink.close()
    be.stop()


def test_http_sink_survives_backend_down_then_up(tmp_path):
    cfg = _cfg("http://127.0.0.1:9/never", tmp_path)       # port tertutup
    sink = HttpSink(cfg, start_worker=False)
    sink.emit({"event_id": "e1"})
    assert not sink.flush(0.5) and sink.outbox.pending() == 1   # tetap di antrean
    sink.close()
    be = _Backend()
    sink2 = HttpSink(_cfg(be.url, tmp_path), start_worker=False)  # proses restart, antrean masih ada
    assert sink2.outbox.pending() == 1
    sink2.outbox._db.execute("UPDATE outbox SET next_try=0")
    assert sink2.flush(3.0) and be.received[0]["event_id"] == "e1"
    sink2.close()
    be.stop()


def test_duplicate_409_is_treated_as_delivered(tmp_path):
    be = _Backend()
    be.received.append({"event_id": "e1"})
    sink = HttpSink(_cfg(be.url, tmp_path), start_worker=False)
    sink.emit({"event_id": "e1"})
    assert sink.flush(3.0) and sink.outbox.pending() == 0
    sink.close()
    be.stop()


def test_multisink_isolates_failing_sink():
    class Bad:
        def emit(self, e):
            raise RuntimeError("boom")

    good = ListSink()
    MultiSink([Bad(), good]).emit({"a": 1})
    assert good.items == [{"a": 1}]
