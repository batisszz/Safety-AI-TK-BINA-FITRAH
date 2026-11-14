"""Sink event. Console/JSONL untuk pengembangan, HTTP + outbox untuk produksi (tahan backend mati)."""
from __future__ import annotations

import json
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from ..config import ConsoleSinkCfg, HttpSinkCfg, JsonlSinkCfg


class ConsoleSink:
    def emit(self, event: dict) -> None:
        m = event.get("meta", {})
        print(f"[EVENT] {event['timestamp']} {event['camera_id']} {event['alert_type']} L{event['level']} "
              f"track={event['track_id']} conf={event['confidence']} t={m.get('stream_ts')}", file=sys.stdout, flush=True)

    def close(self) -> None:
        pass


class JsonlSink:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")
        self._lock = threading.Lock()

    def emit(self, event: dict) -> None:
        with self._lock:
            self._fh.write(json.dumps(event, ensure_ascii=False) + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            self._fh.close()


class Outbox:
    """Antrean lokal SQLite: event tidak hilang walau backend mati."""

    def __init__(self, path: str, max_attempts: int = 50):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.max_attempts = max_attempts
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS outbox (event_id TEXT PRIMARY KEY, payload TEXT NOT NULL, "
                         "attempts INTEGER NOT NULL DEFAULT 0, next_try REAL NOT NULL, dead INTEGER NOT NULL DEFAULT 0)")
        self._db.commit()

    def add(self, event: dict) -> None:
        with self._lock:
            self._db.execute("INSERT OR IGNORE INTO outbox(event_id, payload, next_try) VALUES (?,?,?)",
                             (event["event_id"], json.dumps(event, ensure_ascii=False), 0.0))
            self._db.commit()

    def due(self, now: float, limit: int = 20) -> list:
        with self._lock:
            return self._db.execute("SELECT event_id, payload, attempts FROM outbox WHERE dead=0 AND next_try<=? "
                                    "ORDER BY rowid LIMIT ?", (now, limit)).fetchall()

    def delete(self, event_id: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM outbox WHERE event_id=?", (event_id,))
            self._db.commit()

    def fail(self, event_id: str, attempts: int, next_try: float) -> None:
        with self._lock:
            dead = 1 if attempts >= self.max_attempts else 0
            self._db.execute("UPDATE outbox SET attempts=?, next_try=?, dead=? WHERE event_id=?",
                             (attempts, next_try, dead, event_id))
            self._db.commit()

    def pending(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM outbox WHERE dead=0").fetchone()[0]

    def dead(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM outbox WHERE dead=1").fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self._db.close()


class HttpSink:
    """POST JSON ke backend Node.js (header X-API-Key). Gagal -> tetap di outbox, dicoba ulang dengan backoff."""

    def __init__(self, cfg: HttpSinkCfg, start_worker: bool = True):
        self.cfg = cfg
        self.outbox = Outbox(cfg.outbox)
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        if start_worker:
            self._thread = threading.Thread(target=self._worker, name="http-sink", daemon=True)
            self._thread.start()

    def emit(self, event: dict) -> None:
        self.outbox.add(event)
        self._wake.set()

    def _post(self, payload: str) -> bool:
        req = urllib.request.Request(self.cfg.url, data=payload.encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json", "X-API-Key": self.cfg.api_key})
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
                return 200 <= resp.status < 300
        except urllib.error.HTTPError as e:
            return e.code == 409  # duplikat event_id = sudah diterima
        except Exception:
            return False

    def deliver_once(self) -> int:
        """Coba kirim semua event yang jatuh tempo. Mengembalikan jumlah yang berhasil."""
        ok = 0
        for event_id, payload, attempts in self.outbox.due(time.time()):
            if self._post(payload):
                self.outbox.delete(event_id)
                ok += 1
            else:
                attempts += 1
                delay = min(self.cfg.retry_max_s, self.cfg.retry_base_s * (2 ** min(attempts, 10)))
                self.outbox.fail(event_id, attempts, time.time() + delay)
        return ok

    def _worker(self) -> None:
        while not self._stop.is_set():
            self.deliver_once()
            self._wake.wait(timeout=1.0)
            self._wake.clear()

    def flush(self, timeout: float = 5.0) -> bool:
        """Kirim sisa antrean (dipakai saat shutdown). True jika antrean kosong."""
        end = time.time() + timeout
        while time.time() < end and self.outbox.pending() > 0:
            self.deliver_once()
            time.sleep(0.1)
        return self.outbox.pending() == 0

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self.outbox.close()


class MultiSink:
    def __init__(self, sinks: list):
        self.sinks = sinks

    def emit(self, event: dict) -> None:
        for s in self.sinks:
            try:
                s.emit(event)
            except Exception as e:  # satu sink gagal tidak boleh mematikan yang lain
                print(f"[WARN] sink {type(s).__name__} gagal: {e}", file=sys.stderr)

    def close(self) -> None:
        for s in self.sinks:
            try:
                if hasattr(s, "flush"):
                    s.flush(timeout=3.0)
                s.close()
            except Exception:
                pass


def build_sink(cfgs: list, camera_id: str, run: str) -> MultiSink:
    sinks = []
    for c in cfgs:
        if isinstance(c, ConsoleSinkCfg):
            sinks.append(ConsoleSink())
        elif isinstance(c, JsonlSinkCfg):
            sinks.append(JsonlSink(c.path.format(run=run, camera=camera_id)))
        elif isinstance(c, HttpSinkCfg):
            if not c.url:
                print("[WARN] sink http dilewati: url kosong (isi BACKEND_URL di .env)", file=sys.stderr)
                continue
            sinks.append(HttpSink(c.model_copy(update={"outbox": c.outbox.format(run=run, camera=camera_id)})))
    return MultiSink(sinks)
