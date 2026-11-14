# safety-ai: Modul AI Safety (Fall Detection dan Safe Zone Detection)

Bagian AI Engineer (Batis Satriani Omar Ramadhan) untuk *Sistem Informasi Akademik Terintegrasi CCTV dan AI di TK Bina Fitrah* (C100/C200/C300 S1T26K40). Modul ini membaca video kamera, mendeteksi **anak terjatuh** di kelas dan **anak keluar batas aman** di gerbang, lalu mengirim event JSON ke backend. Spesifikasi lengkap ada di PRD (Claude Doc yang menyertai repo ini).

Modul adalah alat bantu keputusan. Peringatannya tetap perlu diverifikasi guru.

## Status pengujian (jujur)

| Bagian | Status |
|---|---|
| Logika Fall (fitur, state machine, track hilang, mode bbox_only) | Lulus unit test dengan keypoint sintetis |
| Logika Safe Zone (histeresis, dewasa/escort, jadwal, ID berganti, bbox terpotong) | Lulus unit test dengan koordinat sintetis |
| Event, cooldown, JSONL, HTTP sink + outbox, evaluasi, config, sumber video, drift | Lulus unit test (73 test) |
| Alur end-to-end (video sintetis, runner asli, sink, snapshot) | Lulus lewat `tools/simulate.py` |
| **Inferensi YOLOv8 sungguhan** (`YoloBackend`), ekspor OpenVINO/ONNX, tracking ByteTrack | **Belum dijalankan.** Hanya parser hasilnya yang diuji dengan data tiruan |
| Kamera RTSP/webcam sungguhan, GUI `draw_roi` interaktif, Windows | **Belum diuji** (reconnect diuji dengan kamera palsu) |
| Kecepatan di CPU sekolah, akurasi pada anak TK, ambang batas | **Belum diukur.** Semua ambang = titik awal untuk disetel dari data |

Artinya: jalur kode dan logika keputusan sudah terverifikasi, kualitas deteksi di lapangan belum. Langkah pertama di mesin Anda adalah bagian "Langkah pertama di mesin Anda" di bawah.

## Pasang

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Untuk hanya menjalankan test dan simulasi (tanpa Ultralytics): `pip install numpy opencv-python-headless pyyaml pydantic psutil pytest tzdata && pip install -e . --no-deps`.

## Coba dalam 2 menit (tanpa model, tanpa kamera)

```bash
pytest                              # 73 test
python tools/simulate.py            # video sintetis -> runner asli -> event JSON
```

Keluaran simulasi: `fall_suspected` lalu `fall_confirmed` untuk anak yang jatuh dan diam, `zone_warning` lalu `zone_breach` untuk anak yang keluar batas. Dewasa yang lewat, pejalan kaki di jalan, dan anak yang duduk cepat tidak memicu apa pun.

## Langkah pertama di mesin Anda (LOQ)

```bash
python tools/download_models.py --export onnx                 # AMD: ONNX. Untuk Intel: --export openvino
python tools/benchmark.py --model models/yolov8n-pose.onnx --imgsz 320 --device cpu --threads 2 --track --cameras 2
python tools/record_clip.py --source 0 --seconds 60 --out data/raw/uji01.mp4     # webcam, orang dewasa memeragakan
python -m safety_ai.service.runner --config configs/default.yaml --camera cam_kelas_a --source data/raw/uji01.mp4 --run uji01
```

Jika ada galat pada `YoloBackend` (versi Ultralytics berbeda), itu bagian yang belum teruji: kirim pesan galatnya. Perbaikannya terlokalisasi di `src/safety_ai/inference/backend.py`.

Laptop LOQ lebih kencang daripada laptop sekolah. Batasi `--threads` dan anggap angkanya batas atas. Angka resmi diambil di laptop sekolah (`docs/HARDWARE.md`).

## Menjalankan

```bash
# satu kamera, video file (offline, secepat mungkin)
python -m safety_ai.service.runner --config configs/default.yaml --camera cam_gerbang --source data/gerbang01.mp4 --run g01

# mensimulasikan CCTV (jalur RTSP penuh): MediaMTX berjalan, lalu
ffmpeg -re -stream_loop -1 -i data/clip.mp4 -c copy -f rtsp rtsp://localhost:8554/kelasA
# ubah source di YAML menjadi {type: rtsp, url: "rtsp://localhost:8554/kelasA"}

# semua kamera (satu proses per kamera)
python -m safety_ai.service.runner --config configs/default.yaml
```

Hasil per eksekusi ada di `outputs/<run>/`: `events_<kamera>.jsonl`, `stats_<kamera>.json` (frame, FPS efektif, latensi inferensi), dan `snapshots/` (hanya lokal).

Jendela pratinjau: `--show` (butuh `opencv-python`, bukan versi headless).

## Pipeline

```
Fall (kamera kelas)
 sumber -> sampling 3-6 FPS -> YOLOv8n-pose + ByteTrack -> filter keypoint + smoothing
 -> fitur per anak (torso_angle, h_ratio, hip_speed, head_below_hip, motion)
 -> state machine NORMAL > DROPPING > SUSPECTED (L1) > CONFIRMED (L2) > RESOLVED
 -> AlertManager (cooldown, snapshot) -> sink

Safe Zone (kamera gerbang)
 sumber -> motion gate -> crop ROI -> YOLOv8n person + ByteTrack -> titik kaki (tengah-bawah bbox)
 -> jarak bertanda ke poligon -> filter dewasa/escort -> state machine INSIDE > WARNING > PENDING > BREACH
 -> AlertManager -> sink
```

Catatan desain yang tidak ada di PRD awal:
- **Hanya track yang pernah terlihat di dalam zona** yang bisa memicu breach (`require_was_inside`). Pejalan kaki yang selalu di jalan tidak memicu alarm. Konsekuensinya, anak yang pertama kali terdeteksi sudah di luar tidak diberi alarm.
- Fall: syarat jatuh = tanda jatuh cepat **dan** posisi rendah yang bertahan dengan gerak rendah. Anak yang tiarap pelan lalu diam tidak memicu (jalur jatuh-pelan `slow_path_enabled` ada tetapi **nonaktif** secara bawaan karena berisiko alarm palsu).
- Membungkuk lalu diam lebih dari 1,5 detik **dapat memicu L1** (batasan yang tercatat di unit test). Hard negative ini wajib diuji di data nyata.

## Struktur

```
configs/      default.yaml (dua kamera contoh), cameras.example.yaml (RTSP produksi)
docs/         INTEGRATION.md  HARDWARE.md  TRAINING.md  EXPERIMENTS.md  templates/
src/safety_ai/
  types.py config.py evaluation.py toolkit.py
  sources/    file, webcam, RTSP (reconnect, frame terbaru saja)
  inference/  YoloBackend (Ultralytics, lazy), ScriptedBackend (uji)
  modules/fall/  features.py state_machine.py module.py
  modules/safezone/  geometry.py module.py
  events/     manager.py (cooldown, snapshot)  sinks.py (console, JSONL, HTTP + outbox)
  service/    runner.py  health.py (drift, watchdog)
  testing/    synthetic.py simulate.py
tools/        simulate evaluate benchmark draw_roi record_clip download_models
tests/        73 unit test, tanpa GPU dan tanpa video nyata
```

## Integrasi dengan tim lain

Lihat `docs/INTEGRATION.md`: kontrak event JSON, endpoint penerima contoh (Node.js), pemetaan ke `safety_alerts`, dan cara menulis modul lain di atas pipeline pose+tracking yang sama.

## Privasi

`data/`, `snapshots/`, `outputs/`, video, gambar, model, dan `.env` ada di `.gitignore`. Jangan memaksa menambahkannya. Data anak di bawah umur, dan perekamannya memerlukan persetujuan sekolah dan orang tua.

## Belum diimplementasi

- Endpoint FastAPI (health/config) dan penerimaan *Acknowledged / False Alarm* dari dashboard (`FallModule.acknowledge()` sudah ada, tetapi belum ada jalur masuknya).
- Klip video pra-kejadian (ring buffer). Yang ada baru snapshot.
- Penyimpanan umpan balik False Alarm sebagai calon hard negative (SAI-FR-13).
- Mode INT8 dan kuantisasi.

## Lisensi

Ultralytics YOLO berlisensi AGPL-3.0. Cukup untuk riset akademik; periksa sebelum didistribusikan ke sekolah lain.
