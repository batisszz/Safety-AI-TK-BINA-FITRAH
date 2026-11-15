"""Manajemen alert, outbox, status verifikasi (Acknowledge/False Alarm/Resolved),
dan EventBroadcaster untuk Server-Sent Events (SSE).
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from datetime import datetime
from typing import Any, Optional


class AlertStore:
    """Penyimpanan alert in-memory yang thread-safe dengan batas maksimum riwayat."""

    def __init__(self, max_items: int = 500):
        self.max_items = max_items
        self._items: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def add(self, event: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            # Set default status jika belum ada
            item = dict(event)
            item.setdefault("status", "Unresolved")
            item.setdefault("created_at", datetime.now().isoformat())
            item.setdefault("verified_by", None)
            item.setdefault("verified_at", None)
            item.setdefault("verification_note", None)

            # Masukkan di awal (terbaru paling depan)
            self._items.insert(0, item)
            if len(self._items) > self.max_items:
                self._items.pop()
            return item

    def get_all(self, camera_id: Optional[str] = None, alert_type: Optional[str] = None,
                level: Optional[int] = None, status: Optional[str] = None,
                limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            results = []
            for it in self._items:
                if camera_id and it.get("camera_id") != camera_id:
                    continue
                if alert_type and it.get("alert_type") != alert_type:
                    continue
                if level is not None and it.get("level") != level:
                    continue
                if status and it.get("status") != status:
                    continue
                results.append(dict(it))
                if len(results) >= limit:
                    break
            return results

    def get_by_id(self, event_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            for it in self._items:
                if it.get("event_id") == event_id:
                    return dict(it)
            return None

    def update_status(self, event_id: str, status: str, operator: Optional[str] = None,
                      note: Optional[str] = None) -> Optional[dict[str, Any]]:
        with self._lock:
            for it in self._items:
                if it.get("event_id") == event_id:
                    it["status"] = status
                    it["verified_by"] = operator or "Guru/Operator"
                    it["verified_at"] = datetime.now().isoformat()
                    if note:
                        it["verification_note"] = note
                    return dict(it)
            return None

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = len(self._items)
            unresolved = sum(1 for x in self._items if x.get("status") == "Unresolved")
            fall_count = sum(1 for x in self._items if "fall" in x.get("alert_type", ""))
            zone_count = sum(1 for x in self._items if "zone" in x.get("alert_type", ""))
            return {
                "total": total,
                "unresolved": unresolved,
                "fall_alerts": fall_count,
                "zone_alerts": zone_count,
            }


class EventBroadcaster:
    """Broadcaster event untuk klien Server-Sent Events (SSE)."""

    def __init__(self):
        self._subscribers: set[asyncio.Queue] = set()
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        q = asyncio.Queue(maxsize=100)
        with self._lock:
            self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            self._subscribers.discard(q)

    def publish_sync(self, event_name: str, data: dict) -> None:
        """Kirim event dari thread sink background ke semua SSE client."""
        msg = f"event: {event_name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        with self._lock:
            subs = list(self._subscribers)
        if not subs:
            return

        def _deliver():
            for q in subs:
                try:
                    q.put_nowait(msg)
                except asyncio.QueueFull:
                    try:
                        q.get_nowait()
                        q.put_nowait(msg)
                    except Exception:
                        pass

        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(_deliver)


class BroadcasterSink:
    """Sink adapter yang menghubungkan AlertManager ke AlertStore dan EventBroadcaster."""

    def __init__(self, store: AlertStore, broadcaster: EventBroadcaster, inner_sink=None):
        self.store = store
        self.broadcaster = broadcaster
        self.inner_sink = inner_sink

    def emit(self, event: dict) -> None:
        # Simpan ke AlertStore in-memory
        enriched = self.store.add(event)

        # Broadcast via SSE ke website
        self.broadcaster.publish_sync("alert", enriched)

        # Teruskan ke sink asli jika ada (misal JSONL/Console)
        if self.inner_sink:
            try:
                self.inner_sink.emit(event)
            except Exception:
                pass

    def close(self) -> None:
        if self.inner_sink and hasattr(self.inner_sink, "close"):
            self.inner_sink.close()
