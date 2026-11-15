"""Konfigurasi tervalidasi (pydantic). Semua parameter per kamera ada di YAML, tidak ada hardcode."""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

# Profil siap pakai untuk CPU lemah. Nilai eksplisit di YAML selalu menang atas profil.
PROFILES = {
    "low_cpu": {"imgsz": 320, "fps": 3.0},   # Celeron / i3 lama (belum diukur, lihat docs/HARDWARE.md)
    "normal": {"imgsz": 480, "fps": 25.0},    # i5/Ryzen
    "gate": {"imgsz": 480, "fps": 25.0},      # kamera gerbang (crop ROI)
}


# --------------------------------------------------------------------------- sumber
class SourceConfig(BaseModel):
    type: Literal["file", "webcam", "rtsp"]
    path: Optional[str] = None
    url: Optional[str] = None
    index: int = 0
    loop: bool = False
    realtime: bool = False  # file: putar mengikuti kecepatan asli (untuk simulasi CCTV)
    rtsp_transport: Literal["tcp", "udp"] = "tcp"
    start_time: Optional[str] = None  # ISO, jam dinding awal untuk replay file
    stall_s: float = 10.0  # tanpa frame selama ini = kamera dianggap offline

    @model_validator(mode="after")
    def _check(self):
        if self.type == "file" and not self.path:
            raise ValueError("source.type=file membutuhkan 'path'")
        if self.type == "rtsp" and not self.url:
            raise ValueError("source.type=rtsp membutuhkan 'url'")
        return self


# --------------------------------------------------------------------------- inferensi
class InferConfig(BaseModel):
    profile: Optional[Literal["low_cpu", "normal", "gate"]] = None
    model: str = "yolov8n-pose"  # nama (models/<nama>.pt) atau path folder ekspor OpenVINO / berkas .onnx
    task: Optional[Literal["pose", "detect"]] = None
    imgsz: int = 480
    fps: float = 6.0
    conf: float = 0.35
    iou: float = 0.5
    device: str = "cpu"
    threads: Optional[int] = None
    tracker: Optional[str] = None  # path YAML tracker; default configs/bytetrack_cpu.yaml
    classes: list = Field(default_factory=lambda: [0])

    @model_validator(mode="before")
    @classmethod
    def _apply_profile(cls, data):
        if isinstance(data, dict) and data.get("profile"):
            preset = PROFILES[data["profile"]]
            data = {**preset, **data}
        return data

    @field_validator("imgsz")
    @classmethod
    def _imgsz(cls, v):
        if v % 32 != 0:
            raise ValueError("imgsz harus kelipatan 32")
        return v

    def resolved_task(self) -> str:
        if self.task:
            return self.task
        return "pose" if "pose" in self.model.lower() else "detect"


# --------------------------------------------------------------------------- parameter modul
class FallParams(BaseModel):
    """Semua angka adalah TITIK AWAL, bukan hasil kalibrasi. Setel dari data (docs/EXPERIMENTS.md)."""

    kpt_conf: float = 0.3
    ema_alpha: float = 0.5
    min_track_age_s: float = 1.0
    href_window_s: float = 10.0
    # tanda jatuh cepat
    drop_window_s: float = 0.6
    t_drop: float = 0.8          # tinggi-tubuh per detik (pinggul turun)
    shrink_window_s: float = 1.0
    t_shrink: float = 0.30       # penyusutan tinggi bbox relatif terhadap h_ref
    # posisi rendah
    angle_low_deg: float = 60.0
    hratio_low: float = 0.55
    head_hip_min: float = 0.15   # (y_pinggul - y_kepala)/h_ref di bawah ini = kepala tidak di atas pinggul
    bbox_aspect_low: float = 1.2  # mode bbox_only: lebar/tinggi
    # gerak
    motion_window_s: float = 1.0
    motion_low: float = 0.5      # <= ini = "gerak rendah" (syarat L1)
    motion_still: float = 0.2    # <= ini = mulai dihitung imobil (syarat L2)
    motion_still_break: float = 0.3  # > ini = imobil terputus (histeresis)
    # state machine
    low_pose_start_window_s: float = 2.0
    low_pose_hold_s: float = 1.5
    recover_s: float = 1.0
    confirm_still_s: float = 5.0
    suspected_timeout_s: float = 20.0
    slow_transition_s: float = 3.0
    slow_still_s: float = 10.0
    bbox_only_factor: float = 1.5  # pengali waktu konfirmasi saat keypoint tidak dapat dipercaya
    refractory_s: float = 5.0      # jeda setelah RESOLVED sebelum boleh memicu lagi
    slow_path_enabled: bool = False  # jalur jatuh pelan (lemas). Aktifkan hanya setelah hard negative diuji
    # track hilang
    lost_grace_s: float = 3.0
    suspected_lost_grace_s: float = 10.0
    reassoc_radius: float = 0.8  # x h_ref


class ZoneParams(BaseModel):
    file: Optional[str] = None  # JSON hasil tools/draw_roi.py: {"safe_polygon": [[x,y],...]}
    safe_polygon: Optional[list] = None  # koordinat ternormalisasi 0..1
    margin_frac: float = 0.02
    warn_frac: float = 0.05
    breach_hold_s: float = 1.2
    return_hold_s: float = 1.0
    max_gap_s: float = 3.0
    min_track_age_s: float = 0.5
    require_was_inside: bool = True  # hanya track yang pernah terlihat DI DALAM zona yang bisa memicu breach
    lost_grace_s: float = 3.0
    crop: bool = True
    crop_margin_frac: float = 0.12
    # motion gate
    motion_gate: bool = True
    motion_thresh: int = 25
    motion_min_area_frac: float = 0.002
    heartbeat_s: float = 1.0
    # filter anak / dewasa
    child_height_calibration: Optional[list] = None  # [[y_norm, h_norm], [y_norm, h_norm]]
    adult_ratio: float = 1.35
    escort_distance_factor: float = 1.5
    escort_mode: Literal["log_only", "alarm"] = "log_only"
    escort_windows: list = Field(default_factory=lambda: ["06:30-08:30", "10:00-12:00"])
    active_windows: list = Field(default_factory=list)  # kosong = selalu aktif

    @field_validator("safe_polygon")
    @classmethod
    def _poly(cls, v):
        if v is None:
            return v
        if len(v) < 3:
            raise ValueError("safe_polygon minimal 3 titik")
        for p in v:
            if len(p) != 2 or not all(0.0 <= float(c) <= 1.0 for c in p):
                raise ValueError("titik poligon harus [x, y] ternormalisasi 0..1")
        return v


class DriftConfig(BaseModel):
    enabled: bool = False
    reference: Optional[str] = None  # gambar referensi, dibuat oleh tools/draw_roi.py
    interval_s: float = 600.0
    thresh_frac: float = 0.03  # geser > 3% lebar frame = bergeser


class CameraConfig(BaseModel):
    id: str
    source: SourceConfig
    modules: list
    infer: InferConfig = Field(default_factory=InferConfig)
    fall: FallParams = Field(default_factory=FallParams)
    roi: Optional[ZoneParams] = None
    drift: DriftConfig = Field(default_factory=DriftConfig)

    @field_validator("modules")
    @classmethod
    def _modules(cls, v):
        bad = [m for m in v if m not in ("fall", "safezone")]
        if bad:
            raise ValueError(f"modul tidak dikenal: {bad}")
        return v

    @model_validator(mode="after")
    def _check(self):
        if "safezone" in self.modules and (self.roi is None or not (self.roi.safe_polygon or self.roi.file)):
            raise ValueError(f"kamera {self.id}: modul safezone membutuhkan roi.safe_polygon atau roi.file")
        if "fall" in self.modules and self.infer.resolved_task() != "pose":
            raise ValueError(f"kamera {self.id}: modul fall membutuhkan model pose (mis. yolov8n-pose)")
        return self


# --------------------------------------------------------------------------- sink & alert
class ConsoleSinkCfg(BaseModel):
    type: Literal["console"] = "console"


class JsonlSinkCfg(BaseModel):
    type: Literal["jsonl"] = "jsonl"
    path: str = "outputs/{run}/events_{camera}.jsonl"


class HttpSinkCfg(BaseModel):
    type: Literal["http"] = "http"
    url: str
    api_key: str = ""
    outbox: str = "outputs/outbox_{camera}.db"
    timeout_s: float = 5.0
    retry_base_s: float = 2.0
    retry_max_s: float = 300.0


SinkCfg = Union[ConsoleSinkCfg, JsonlSinkCfg, HttpSinkCfg]


class AlertsConfig(BaseModel):
    cooldown_s: dict = Field(
        default_factory=lambda: {
            "fall_suspected": 45.0,
            "fall_confirmed": 45.0,
            "zone_breach": 30.0,
            "zone_warning": 30.0,
            "system_health": 300.0,
        }
    )
    max_per_minute: int = 30
    snapshots: bool = True
    snapshot_dir: str = "snapshots"
    jpeg_quality: int = 80


class AppConfig(BaseModel):
    timezone: str = "Asia/Jakarta"
    run_name: str = "run"
    output_dir: str = "outputs"
    cameras: list
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)
    sinks: list = Field(default_factory=lambda: [ConsoleSinkCfg()])

    @field_validator("cameras")
    @classmethod
    def _cams(cls, v):
        ids = [c.id for c in v]
        if len(ids) != len(set(ids)):
            raise ValueError("id kamera harus unik")
        return v

    def camera(self, cam_id: str) -> CameraConfig:
        for c in self.cameras:
            if c.id == cam_id:
                return c
        raise KeyError(f"kamera '{cam_id}' tidak ada di konfigurasi")


# --------------------------------------------------------------------------- loader
_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def load_dotenv(path: str = ".env") -> None:
    """Pemuat .env sederhana (tanpa dependensi). Tidak menimpa variabel yang sudah ada."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _expand_env(text: str) -> str:
    return _ENV.sub(lambda m: os.environ.get(m.group(1), m.group(2) if m.group(2) is not None else ""), text)


def _parse_sink(d: dict):
    t = d.get("type")
    return {"console": ConsoleSinkCfg, "jsonl": JsonlSinkCfg, "http": HttpSinkCfg}[t](**d)


def parse_config(raw: dict, base_dir: Optional[Path] = None) -> AppConfig:
    raw = dict(raw)
    raw["cameras"] = [CameraConfig(**c) for c in raw.get("cameras", [])]
    if "sinks" in raw:
        raw["sinks"] = [_parse_sink(s) for s in raw["sinks"]]
    if "alerts" in raw and isinstance(raw["alerts"], dict):
        raw["alerts"] = AlertsConfig(**raw["alerts"])
    cfg = AppConfig(**raw)
    for cam in cfg.cameras:
        if cam.roi and cam.roi.file and not cam.roi.safe_polygon:
            p = Path(cam.roi.file)
            if not p.is_absolute() and base_dir and not p.exists():
                p = base_dir / p
            data = json.loads(p.read_text(encoding="utf-8"))
            cam.roi = ZoneParams(**{**cam.roi.model_dump(), "safe_polygon": data["safe_polygon"]})
    return cfg


def load_config(path: str) -> AppConfig:
    load_dotenv()
    p = Path(path)
    text = _expand_env(p.read_text(encoding="utf-8"))
    raw = yaml.safe_load(text) or {}
    return parse_config(raw, base_dir=p.parent)


# --------------------------------------------------------------------------- jendela waktu
def parse_windows(windows: list) -> list:
    """['07:00-08:30'] -> [(420, 510)] dalam menit sejak 00:00."""
    out = []
    for w in windows or []:
        a, b = w.split("-")
        ha, ma = a.split(":")
        hb, mb = b.split(":")
        out.append((int(ha) * 60 + int(ma), int(hb) * 60 + int(mb)))
    return out


def in_windows(dt: datetime, windows: list, default_if_empty: bool = True) -> bool:
    parsed = parse_windows(windows)
    if not parsed:
        return default_if_empty
    m = dt.hour * 60 + dt.minute
    return any(a <= m < b for a, b in parsed)


def get_tz(name: str) -> tzinfo:
    """Zona waktu. Di Windows perlu paket tzdata; jika tidak ada, Asia/Jakarta jatuh ke UTC+7."""
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(name)
    except Exception:
        if name in ("Asia/Jakarta", "WIB"):
            return timezone(timedelta(hours=7), "WIB")
        return timezone.utc
