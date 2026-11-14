# Kontrak integrasi (untuk tim Full Stack dan AI lain)

Modul ini hanya mengirim **event**, bukan frame. Tim backend cukup menerima JSON di bawah.

## 1. Event JSON

```json
{
  "event_id": "5f1c...-uuid-v4",
  "camera_id": "cam_kelas_a",
  "alert_type": "fall_suspected",
  "level": 1,
  "track_id": 7,
  "confidence": 0.63,
  "timestamp": "2026-11-12T09:00:12+07:00",
  "snapshot_path": "snapshots/cam_kelas_a/2026-11-12/090012_7_fall_suspected.jpg",
  "meta": {"mode": "pose", "hip_speed": 1.4, "torso_angle": 72.0, "h_ratio": 0.41,
           "low_pose_s": 1.7, "stream_ts": 12.16, "source": "sesi01.mp4"}
}
```

| Field | Arti |
|---|---|
| event_id | UUID v4, kunci idempoten. Backend membalas `201` pertama kali dan `409` bila sudah ada (keduanya dianggap sukses oleh modul) |
| camera_id | id kamera di YAML, selaras dengan `cameras.camera_id` |
| alert_type | `fall_suspected`, `fall_confirmed`, `zone_breach`, `zone_warning`, `system_health` |
| level | 0 = hanya log, 1 = notifikasi guru, 2 = alarm admin |
| track_id | ID anak dari ByteTrack, hanya valid pada satu kamera dan satu sesi. Bukan identitas anak |
| confidence | Skor **heuristik** 0 sampai 1 (kualitas keypoint, kecepatan jatuh, durasi). Bukan probabilitas terkalibrasi |
| timestamp | ISO-8601 dengan zona waktu (bawaan Asia/Jakarta, +07:00) |
| snapshot_path | Relatif terhadap `outputs/<run>/`, hanya untuk level >= 1, bisa `null` |
| meta | Bebas per jenis event. `meta.stream_ts` dan `meta.source` dipakai `tools/evaluate.py` |

`child_id` tidak ada: fall dan safe zone tidak mengenali identitas. Pemetaan ke anak (jika perlu) lewat kelas kamera, bukan lewat modul ini.

Level per jenis:

| alert_type | level | Saran penanganan di backend |
|---|---|---|
| fall_suspected | 1 | Notifikasi guru. Simpan di `safety_alerts` dengan status `Unresolved` |
| fall_confirmed | 2 | Danger alarm admin. Notifikasi wali murid (usulan, perlu persetujuan pembimbing) |
| zone_breach | 2 | Danger alarm admin dan notifikasi guru |
| zone_warning | 0 | Hanya log, jangan notifikasi |
| system_health | 0 atau 1 | `meta.state`: `camera_offline` (1), `camera_online` (0), `camera_drift` (1), `camera_aligned` (0). Tampilkan di halaman status kamera |

Nilai `status` yang disarankan pada `safety_alerts`: `Unresolved`, `Acknowledged`, `False Alarm`, `Resolved`. Keterangan kolom `status` dan `timestamp` pada Tabel 2.12 C300 tampak tertukar dengan tabel presensi; perbaiki sekalian.

## 2. Mengirim event

Aktifkan sink HTTP di YAML dan isi `.env`:

```yaml
sinks:
  - {type: http, url: "${BACKEND_URL}", api_key: "${API_KEY}", outbox: "outputs/outbox_{camera}.db"}
```

Modul melakukan `POST` JSON dengan header `X-API-Key`. Jika backend mati atau menjawab bukan 2xx/409, event tetap di antrean lokal (SQLite) dan dicoba ulang dengan backoff eksponensial (2 detik, 4, 8, hingga 300 detik). Setelah 50 percobaan gagal event ditandai `dead` di outbox (tidak dihapus).

Contoh penerima (Express):

```js
const seen = new Set(); // ganti dengan pengecekan event_id di Firestore
app.post('/api/safety-events', (req, res) => {
  if (req.get('X-API-Key') !== process.env.AI_API_KEY) return res.sendStatus(401);
  const e = req.body;
  const required = ['event_id', 'camera_id', 'alert_type', 'level', 'timestamp'];
  if (required.some(k => e[k] === undefined)) return res.sendStatus(400);
  if (seen.has(e.event_id)) return res.sendStatus(409);   // idempoten
  seen.add(e.event_id);
  // simpan ke safety_alerts, lalu kirim notifikasi sesuai level
  res.sendStatus(201);
});
```

Uji tanpa backend: `curl -X POST -H "Content-Type: application/json" -H "X-API-Key: x" -d @event.json $BACKEND_URL`.

Penting: kontrak ini tidak berubah tanpa kesepakatan tim. Menambah field di `meta` aman. Mengubah nama atau arti field lain tidak.

## 3. Keputusan terbuka: snapshot dan lokasi komputer

AI berjalan di komputer sekolah (lokal), sedangkan web dan backend di VPS. **VPS tidak bisa membaca `snapshot_path` di disk lokal.** Pilihan:

1. Kirim metadata saja. Guru melihat kejadian lewat playback CCTV lokal (selaras dengan prinsip tidak ada video anak di cloud). Paling aman secara privasi.
2. Sajikan snapshot dari komputer sekolah lewat endpoint terautentikasi (class-based access) melalui tunnel. Perlu infrastruktur tambahan.
3. Unggah snapshot ke cloud. **Tidak disarankan** (foto anak di cloud bertentangan dengan NFR privasi di C200).

Sampai diputuskan, modul menyimpan snapshot lokal saja dan hanya mengirim path.

## 4. Memakai pipeline yang sama untuk modul lain

Satu proses per kamera sudah melakukan decode, sampling, deteksi dan tracking. Modul lain (aktivitas, presensi) cukup mengimplementasikan:

```python
class SafetyModule(Protocol):
    name: str
    def should_infer(self, frame) -> bool: ...                 # True = butuh deteksi pada frame ini
    def crop_region(self, width, height): ...                  # (x1,y1,x2,y2) atau None = frame penuh
    def process(self, ctx: FrameContext) -> list[Event]: ...   # ctx.tracks: id, bbox, keypoints, conf
```

`FrameContext.tracks` berisi `Track(id, bbox, conf, keypoints[17,3], cls_name)` dalam koordinat frame penuh. Daftarkan modul di `build_modules()` (`service/runner.py`). Jika dua modul pada kamera yang sama butuh model berbeda (pose dan person biasa), pakai satu model pose saja: pose sudah memberi bbox.

Waktu stream (`frame.ts`) selalu dalam detik sejak awal stream (file: `idx / fps`), sehingga hasil sama pada kecepatan putar berapa pun. Jangan memakai jam dinding di logika modul.
