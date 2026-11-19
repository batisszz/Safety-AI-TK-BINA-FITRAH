"""Server FastAPI untuk modul Safety AI:
- Streaming video real-time (MJPEG) dengan anotasi deteksi (bounding box, skeleton, safe zone).
- Server-Sent Events (SSE) untuk mendorong notifikasi alert ke web secara real-time.
- REST API untuk riwayat alert, verifikasi (Acknowledge / False Alarm / Resolved), dan statistik.
- Penyajian aset statis website dashboard.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import psutil
from fastapi import FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import AppConfig, SourceConfig, load_config
from .alerts import AlertStore, EventBroadcaster
from .streamer import CameraManager

# Global state
app_start_time = time.time()
alert_store = AlertStore(max_items=500)
broadcaster = EventBroadcaster()
camera_manager: Optional[CameraManager] = None
global_config: Optional[AppConfig] = None


class AcknowledgeRequest(BaseModel):
    status: str = Field(..., description="Acknowledged | False Alarm | Resolved")
    operator: Optional[str] = Field("Guru Piket", description="Nama guru atau operator yang memverifikasi")
    note: Optional[str] = Field(None, description="Catatan opsional dari penanganan insiden")


class TriggerAlertRequest(BaseModel):
    camera_id: str = "cam_kelas_a"
    alert_type: str = "fall_confirmed"
    level: int = 2
    track_id: int = 1
    confidence: float = 0.88
    message: str = "Simulasi pengujian alert dari web dashboard"


class SetSourceRequest(BaseModel):
    type: str = Field(..., description="file | webcam | rtsp | browser")
    path: Optional[str] = None
    url: Optional[str] = None
    index: int = 0
    rtsp_transport: str = "tcp"


class RoiRequest(BaseModel):
    safe_polygon: list = Field(..., description="Daftar titik [x,y] ternormalisasi 0..1")


def get_app(config_path: str = "configs/default.yaml", run_name: Optional[str] = None) -> FastAPI:
    global global_config, camera_manager

    global_config = load_config(config_path)
    camera_manager = CameraManager(global_config, alert_store, broadcaster, run_name=run_name)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        broadcaster.set_loop(asyncio.get_running_loop())
        logger_name = "safety_ai.api"
        print(f"[{logger_name}] Memulai camera streamer untuk {len(global_config.cameras)} kamera...", flush=True)
        camera_manager.start_all()
        yield
        # Shutdown
        print(f"[{logger_name}] Menghentikan semua streamer...", flush=True)
        camera_manager.stop_all()

    api_app = FastAPI(
        title="Safety AI API",
        description="API terintegrasi Fall Detection dan Safe Zone Detection untuk SIA TK Bina Fitrah",
        version="1.0.0",
        lifespan=lifespan
    )

    # Izinkan CORS penuh agar website terpisah dapat mengakses API tanpa hambatan
    api_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------ Health & Status
    @api_app.get("/api/health", summary="Status kesehatan sistem")
    async def health():
        cam_stats = camera_manager.get_all_stats() if camera_manager else []
        return {
            "status": "online",
            "uptime_seconds": round(time.time() - app_start_time, 1),
            "timestamp": datetime.now().isoformat(),
            "cameras_active": sum(1 for c in cam_stats if c["is_running"]),
            "cameras_total": len(cam_stats),
            "alerts": alert_store.stats(),
        }

    @api_app.get("/api/stats", summary="Statistik sistem dan deteksi")
    async def system_stats():
        mem = psutil.virtual_memory()
        return {
            "uptime_seconds": round(time.time() - app_start_time, 1),
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": mem.percent,
            "cameras": camera_manager.get_all_stats() if camera_manager else [],
            "alerts": alert_store.stats(),
        }

    # ------------------------------------------------------------------ Kamera & Video Stream
    @api_app.get("/api/cameras", summary="Daftar kamera dan telemetri")
    async def list_cameras():
        if not camera_manager:
            return []
        return camera_manager.get_all_stats()

    @api_app.get("/api/cameras/{camera_id}/source", summary="Konfigurasi sumber kamera aktif")
    async def get_camera_source(camera_id: str):
        if not camera_manager:
            raise HTTPException(status_code=503, detail="Camera manager belum siap")
        info = camera_manager.get_source_info(camera_id)
        if info is None:
            raise HTTPException(status_code=404, detail=f"Kamera '{camera_id}' tidak ditemukan")
        return {"camera_id": camera_id, "source": info}

    @api_app.post("/api/cameras/{camera_id}/source", summary="Ganti sumber kamera (file/webcam/rtsp/browser)")
    async def set_camera_source(camera_id: str, req: SetSourceRequest):
        if not camera_manager:
            raise HTTPException(status_code=503, detail="Camera manager belum siap")
        streamer = camera_manager.get_streamer(camera_id)
        if not streamer:
            raise HTTPException(status_code=404, detail=f"Kamera '{camera_id}' tidak ditemukan")

        stype = (req.type or "").strip().lower()
        if stype == "file":
            if not req.path:
                raise HTTPException(status_code=400, detail="path diperlukan untuk source.type=file")
            src = SourceConfig(type="file", path=req.path, loop=True, realtime=True)
        elif stype == "webcam":
            src = SourceConfig(type="webcam", index=int(req.index))
        elif stype == "rtsp":
            if not req.url:
                raise HTTPException(status_code=400, detail="url diperlukan untuk source.type=rtsp")
            transport = req.rtsp_transport if req.rtsp_transport in ("tcp", "udp") else "tcp"
            src = SourceConfig(type="rtsp", url=req.url, rtsp_transport=transport)
        elif stype == "browser":
            src = SourceConfig(type="browser")
        else:
            raise HTTPException(status_code=400, detail=f"Tipe sumber tidak dikenal: {req.type}")

        info = camera_manager.set_source(camera_id, src)
        return {"status": "ok", "camera_id": camera_id, "source": info}

    @api_app.post("/api/cameras/{camera_id}/source/frame", summary="Kirim frame JPEG dari browser (kamera perangkat)")
    async def push_browser_frame(camera_id: str, request: Request):
        if not camera_manager:
            raise HTTPException(status_code=503, detail="Camera manager belum siap")
        data = await request.body()
        if not data:
            raise HTTPException(status_code=400, detail="Body kosong")
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Gagal decode gambar JPEG")
        ok = camera_manager.push_browser_frame(camera_id, img)
        if not ok:
            raise HTTPException(status_code=409, detail="Kamera tidak dalam mode sumber browser")
        return {"status": "ok"}

    @api_app.get("/api/sources/videos", summary="Daftar file video yang tersedia untuk sumber file")
    async def list_source_videos():
        roots = [Path("data/raw"), Path("data")]
        exts = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
        seen = set()
        items = []
        for root in roots:
            if not root.exists():
                continue
            for p in sorted(root.glob("**/*")):
                if not p.is_file() or p.suffix.lower() not in exts:
                    continue
                try:
                    rel = p.as_posix()
                except Exception:
                    rel = str(p)
                if rel in seen:
                    continue
                seen.add(rel)
                items.append({"path": rel, "name": p.name})
        return {"items": items}

    @api_app.get("/api/cameras/{camera_id}/roi", summary="Ambil konfigurasi ROI kamera")
    async def get_camera_roi(camera_id: str):
        if not camera_manager:
            raise HTTPException(status_code=503, detail="Camera manager belum siap")
        roi = camera_manager.get_roi(camera_id)
        if roi is None:
            raise HTTPException(status_code=404, detail=f"Kamera '{camera_id}' tidak memiliki ROI/safezone")
        return roi

    @api_app.post("/api/cameras/{camera_id}/roi", summary="Simpan dan terapkan ROI poligon kamera")
    async def set_camera_roi(camera_id: str, req: RoiRequest):
        if not camera_manager:
            raise HTTPException(status_code=503, detail="Camera manager belum siap")
        poly = req.safe_polygon
        if not isinstance(poly, list) or len(poly) < 3:
            raise HTTPException(status_code=400, detail="safe_polygon minimal 3 titik")
        norm = []
        for p in poly:
            if not isinstance(p, (list, tuple)) or len(p) != 2:
                raise HTTPException(status_code=400, detail="Setiap titik harus [x, y]")
            x, y = float(p[0]), float(p[1])
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise HTTPException(status_code=400, detail="Koordinat harus ternormalisasi 0..1")
            norm.append([round(x, 4), round(y, 4)])
        roi = camera_manager.set_roi(camera_id, norm)
        if roi is None:
            raise HTTPException(status_code=404, detail=f"Kamera '{camera_id}' tidak memiliki ROI/safezone")
        return {"status": "ok", "roi": roi}

    @api_app.get("/api/cameras/{camera_id}/stream", summary="Stream MJPEG video dengan anotasi deteksi")
    async def stream_camera(camera_id: str):
        if not camera_manager:
            raise HTTPException(status_code=503, detail="Camera manager belum siap")
        streamer = camera_manager.get_streamer(camera_id)
        if not streamer:
            raise HTTPException(status_code=404, detail=f"Kamera '{camera_id}' tidak ditemukan")

        return StreamingResponse(
            streamer.generate_mjpeg(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )

    @api_app.get("/api/cameras/{camera_id}/frame", summary="Ambil satu frame JPEG terbaru")
    async def get_frame(camera_id: str):
        if not camera_manager:
            raise HTTPException(status_code=503, detail="Camera manager belum siap")
        streamer = camera_manager.get_streamer(camera_id)
        if not streamer:
            raise HTTPException(status_code=404, detail=f"Kamera '{camera_id}' tidak ditemukan")

        jpeg = streamer.get_latest_jpeg()
        if not jpeg:
            raise HTTPException(status_code=503, detail="Frame belum tersedia dari kamera")

        return Response(content=jpeg, media_type="image/jpeg")

    # ------------------------------------------------------------------ Alerts & SSE
    @api_app.get("/api/alerts", summary="Daftar riwayat alert")
    async def list_alerts(
        camera_id: Optional[str] = None,
        alert_type: Optional[str] = None,
        level: Optional[int] = None,
        status: Optional[str] = None,
        limit: int = Query(default=50, ge=1, le=200)
    ):
        items = alert_store.get_all(
            camera_id=camera_id,
            alert_type=alert_type,
            level=level,
            status=status,
            limit=limit
        )
        return {
            "total": len(items),
            "summary": alert_store.stats(),
            "items": items
        }

    @api_app.get("/api/alerts/events", summary="Server-Sent Events (SSE) stream notifikasi alert real-time")
    async def sse_alerts(request: Request):
        queue = broadcaster.subscribe()

        async def event_generator():
            try:
                # Kirim handshake awal
                yield f"event: connected\ndata: {{\"status\": \"connected\", \"time\": \"{datetime.now().isoformat()}\"}}\n\n"
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        # Tunggu event baru dengan timeout untuk heartbeat
                        msg = await asyncio.wait_for(queue.get(), timeout=15.0)
                        yield msg
                    except asyncio.TimeoutError:
                        # Heartbeat ping
                        yield f"event: ping\ndata: {{\"time\": \"{datetime.now().isoformat()}\"}}\n\n"
            finally:
                broadcaster.unsubscribe(queue)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    @api_app.post("/api/alerts/{event_id}/acknowledge", summary="Verifikasi alert (Acknowledge / False Alarm / Resolved)")
    async def acknowledge_alert(event_id: str, req: AcknowledgeRequest):
        updated = alert_store.update_status(
            event_id=event_id,
            status=req.status,
            operator=req.operator,
            note=req.note
        )
        if not updated:
            raise HTTPException(status_code=404, detail=f"Alert '{event_id}' tidak ditemukan")

        # Jika jenis fall, teruskan ke FallModule
        if "fall" in updated.get("alert_type", "") and camera_manager and updated.get("track_id") is not None:
            camera_manager.acknowledge_fall(int(updated["track_id"]), time.time())

        # Siarkan perubahan status ke semua klien web
        broadcaster.publish_sync("alert_updated", updated)

        return {"status": "success", "alert": updated}

    # ------------------------------------------------------------------ Snapshots
    @api_app.get("/api/snapshots/{snapshot_path:path}", summary="Akses berkas snapshot gambar insiden")
    async def get_snapshot(snapshot_path: str):
        # Cari file di folder outputs
        cand_paths = [
            Path(snapshot_path),
            Path(global_config.output_dir) / snapshot_path,
            Path(global_config.output_dir) / "web_stream" / snapshot_path,
        ]
        # Cari juga di subfolder outputs
        for base in Path(global_config.output_dir).glob("*/snapshots"):
            cand_paths.append(base.parent / snapshot_path)

        for p in cand_paths:
            if p.exists() and p.is_file():
                return FileResponse(str(p), media_type="image/jpeg")

        raise HTTPException(status_code=404, detail="Snapshot tidak ditemukan")

    # ------------------------------------------------------------------ Testing Helper
    @api_app.post("/api/test/trigger-alert", summary="Picu simulasi alert untuk pengujian audio dan notifikasi web")
    async def trigger_test_alert(req: TriggerAlertRequest):
        now_iso = datetime.now().isoformat()
        test_event = {
            "event_id": str(uuid.uuid4()),
            "camera_id": req.camera_id,
            "alert_type": req.alert_type,
            "level": req.level,
            "track_id": req.track_id,
            "confidence": req.confidence,
            "timestamp": now_iso,
            "snapshot_path": None,
            "meta": {
                "message": req.message,
                "is_simulation": True,
                "stream_ts": 12.5
            }
        }
        enriched = alert_store.add(test_event)
        broadcaster.publish_sync("alert", enriched)
        return {"status": "triggered", "event": enriched}

    # ------------------------------------------------------------------ Website Static Files
    web_dir = Path("web")
    if web_dir.exists():
        api_app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")

    return api_app


app = get_app()


def main(argv: Optional[list] = None) -> int:
    import uvicorn

    parser = argparse.ArgumentParser(description="Safety AI Web API Server")
    parser.add_argument("--config", default="configs/default.yaml", help="Path konfigurasi default.yaml")
    parser.add_argument("--run", default=None, help="Nama subfolder output (default: timestamp YYYYMMDD_HHMMSS)")
    parser.add_argument("--host", default="0.0.0.0", help="Host bind address")
    parser.add_argument("--port", type=int, default=8000, help="Port server")
    parser.add_argument("--reload", action="store_true", help="Auto reload")
    args = parser.parse_args(argv)

    print(f"\n==================================================================")
    print(f" Safety AI CCTV & Alert Monitoring Server")
    print(f" TK Bina Fitrah - Fall Detection & Safe Zone Detection")
    print(f"==================================================================")
    print(f" API Server running at: http://localhost:{args.port}")
    print(f" Web Dashboard at     : http://localhost:{args.port}/")
    print(f" API Documentation at : http://localhost:{args.port}/docs")
    print(f"==================================================================\n")

    if args.run and camera_manager:
        camera_manager.run_name = args.run

    uvicorn.run(
        "safety_ai.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
