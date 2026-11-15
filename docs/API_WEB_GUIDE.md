# Panduan Integrasi API & Web Dashboard Safety AI

Dokumen ini menjelaskan implementasi API di dalam `safety-ai` serta website monitoring terpisah untuk menampilkan video stream deteksi (Fall Detection dan Safe Zone) beserta penerimaan alert secara real-time.

---

## 1. Arsitektur Sistem

```
+---------------------------------------------------------------------------------+
|                                KOMPUTER LOKAL AI                                |
|                                                                                 |
|  [ Kamera Kelas A ] ----> YOLOv8-Pose + ByteTrack ----> FallModule              |
|  [ Kamera Gerbang ] ----> YOLOv8 + ROI Geometri   ----> SafeZoneModule          |
|                                                              |                  |
|                                                              v                  |
|                                                         AlertManager            |
|                                                              |                  |
|                                                              v                  |
|                                                    BroadcasterSink              |
|                                                              |                  |
|                                      +-----------------------+---------------+  |
|                                      |                                       |  |
|                                      v                                       v  |
|                                 AlertStore (In-Memory)               EventBroadcaster   |
|                                      |                                       |  |
|  +-----------------------------------+---------------------------------------+  |
|  |                           FASTAPI SERVER (safety_ai.api.server)               |
|  |  - GET  /api/cameras/{id}/stream  (Multipart MJPEG Stream)                   |
|  |  - GET  /api/alerts/events        (Server-Sent Events / SSE)                 |
|  |  - GET  /api/alerts               (REST API Riwayat Alert)                   |
|  |  - POST /api/alerts/{id}/ack      (Verifikasi Guru / Acknowledge)            |
|  |  - GET  /api/snapshots/{path}     (Akses Gambar Bukti Insiden)               |
|  |  - CORS Enabled (allow_origins: *)                                           |
+--+---------------------------------------------------------------------------+--+
                                       ^
                                       | HTTP / SSE / MJPEG
+--------------------------------------+------------------------------------------+
|                       WEBSITE MONITORING TERPISAH                               |
|                                                                                 |
|  [ Kamera 1: Kelas A (Fall) ]       [ Kamera 2: Gerbang (Safe Zone) ]           |
|  - Live MJPEG Stream                - Live MJPEG Stream                         |
|  - Bounding Box & Skeleton          - Bounding Box & ROI Polygon                |
|  - Status: Normal/Suspected/Fall    - Status: Inside/Warning/Breach             |
|                                                                                 |
|  [ Panel Notifikasi Alert Real-Time ]                                           |
|  - Audio Chime Synthesizer (Web Audio API)                                      |
|  - Card Alert (Tingkat Bahaya, Kamera, Waktu, Snapshot Thumb)                   |
|  - Tombol Tindakan: Verifikasi Valid / Alarm Palsu / Selesai                     |
+---------------------------------------------------------------------------------+
```

---

## 2. Spesifikasi Endpoint API (`safety-ai`)

API berjalan secara default pada port `8000` dengan CORS terbuka (`*`) agar dapat diakses dari domain/port mana saja (termasuk localhost:3000, 5173, atau VPS).

### A. Video Streaming & Frame
| Endpoint | Method | Keterangan |
|---|---|---|
| `/api/cameras` | `GET` | Mengambil status, FPS aktual, latency, dan daftar track aktif pada tiap kamera |
| `/api/cameras/{camera_id}/stream` | `GET` | Aliran video MJPEG (`multipart/x-mixed-replace`) dengan visual anotasi |
| `/api/cameras/{camera_id}/frame` | `GET` | Mengambil satu frame JPEG terbaru dari kamera |

### B. Alerts & Server-Sent Events (SSE)
| Endpoint | Method | Keterangan |
|---|---|---|
| `/api/alerts/events` | `GET` | SSE stream real-time. Browser mendengarkan event `alert` dan `alert_updated` |
| `/api/alerts` | `GET` | Mengambil riwayat alert (query: `camera_id`, `alert_type`, `level`, `status`, `limit`) |
| `/api/alerts/{event_id}/acknowledge` | `POST` | Memverifikasi alert (`Acknowledged`, `False Alarm`, `Resolved`) dan memanggil `FallModule.acknowledge()` |
| `/api/test/trigger-alert` | `POST` | Memicu simulasi alert untuk menguji UI & audio chime |

### C. Snapshot & Sistem
| Endpoint | Method | Keterangan |
|---|---|---|
| `/api/snapshots/{snapshot_path}` | `GET` | Menyajikan berkas JPEG snapshot kejadian |
| `/api/health` | `GET` | Status online, uptime, dan jumlah kamera aktif |
| `/api/stats` | `GET` | Telemetri CPU, memori, FPS per kamera, dan total alert |

---

## 3. Cara Menjalankan

### Opsi 1: Menjalankan Sekaligus (All-in-One — Direkomendasikan)
FastAPI sudah dikonfigurasi untuk menyajikan website statis langsung dari folder `web/`:

```bash
# Aktifkan virtual environment
.venv\Scripts\activate

# Jalankan server
python -m safety_ai.api.server --port 8000
```
Buka browser di: **`http://localhost:8000/`**

---

### Opsi 2: Menjalankan Website Terpisah (Multi-Process)
Jika Anda ingin website dijalankan di server terpisah (misalnya port 3000):

**Terminal 1 — Modul AI & API (`safety-ai`):**
```bash
.venv\Scripts\activate
python -m safety_ai.api.server --port 8000
```

**Terminal 2 — Website Terpisah:**
```bash
# Menggunakan Node.js bawaan di folder web:
node web/server.js
# ATAU menggunakan Python http.server:
python -m http.server 3000 --directory web
```
Buka browser di: **`http://localhost:3000/`**
Website akan otomatis terhubung ke API di `http://localhost:8000`. Jika URL API berbeda, klik tombol **⚙️ API URL** di pojok kanan atas website untuk mengaturnya.

---

## 4. Tampilan 2 Kamera di Website

1. **Kamera 1 (Kelas A — Fall Detection)**:
   - Menampilkan kerangka skeleton pose 17 titik sendi tubuh anak (COCO-17).
   - Bounding box dinamis sesuai status:
     - 🟢 **Hijau (NORMAL)**: Anak berdiri/beraktivitas wajar.
     - 🟡 **Kuning (DROPPING)**: Deteksi kecepatan turun tubuh tinggi.
     - 🟠 **Oranye (SUSPECTED)**: Posisi rendah berlanjut (Level 1 - Notifikasi Guru).
     - 🔴 **Merah (CONFIRMED)**: Tiarap/imobil terkonfirmasi (Level 2 - Bahaya Admin).
   - Telemetri FPS, latency inferensi model `yolov8n-pose`, dan jumlah anak terdeteksi.

2. **Kamera 2 (Gerbang Utama — Safe Zone Detection)**:
   - Menampilkan poligon area aman (arsir hijau transparan).
   - Titik acuan kaki (*foot point*) di bawah setiap anak.
   - Bounding box dinamis:
     - 🟢 **Hijau (INSIDE)**: Berada di dalam batas zona aman.
     - 🟡 **Kuning (WARNING)**: Mendekati margin batas gerbang.
     - 🔴 **Merah (BREACH)**: Melewati batas aman tanpa pendamping dewasa (Level 2).
   - Filter pendamping (*escort filter*): Membedakan anak dan dewasa pada jam antar-jemput.

---

## 5. Fitur Interaktif Web Dashboard

- **Audio Chime Otomatis**: Menghasilkan suara peringatan lembut untuk Level 1 dan suara alarm bahaya untuk Level 2 menggunakan Web Audio API murni (tanpa file mp3 eksternal).
- **Verifikasi Insiden**: Guru dapat mengklik tombol "Periksa" pada kartu alert untuk melihat snapshot resolusi penuh, memasukkan nama guru piket, catatan penanganan, dan memilih status (*Verifikasi Valid / Alarm Palsu / Selesai*).
- **Tombol Simulasi Cepat**: Terdapat tombol `⚡ Uji Jatuh` dan `⚡ Uji Zona` di navbar untuk mendemonstrasikan visual alert dan suara audio secara instan kepada pengguna atau penguji.
