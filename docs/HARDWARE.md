# Spesifikasi komputer dan profil CPU

## Cek spesifikasi komputer sekolah (Windows)

1. Tekan Ctrl + Shift + Esc, buka tab **Performance**.
2. **CPU**: catat nama prosesor, *Cores*, *Logical processors*.
3. **Memory**: catat total RAM. **GPU**: catat apakah ada kartu grafis terpisah.
4. Isi tabel di bawah, lalu jalankan benchmark (bagian berikutnya).

| Item | Laptop LOQ (dev) | Komputer sekolah |
|---|---|---|
| Tipe | LOQ 15APH8 (seri APH = prosesor AMD Ryzen, verifikasi di Task Manager) | (isi) |
| Prosesor | (isi) | (isi, dugaan awal: Celeron atau i3 generasi lama) |
| Core / thread | (isi) | (isi) |
| RAM | (isi) | (isi) |
| GPU | NVIDIA (untuk training) | (isi) |
| Format ekspor yang dipakai | ONNX | OpenVINO jika Intel |

Catatan prosesor lama: Celeron/Pentium generasi lama kadang tidak mendukung AVX2 sehingga jauh lebih lambat. Prosesor dengan 2 core dan RAM 4 GB akan sulit menjalankan dua kamera sekaligus. Ini hanya dugaan; ukur, jangan menebak.

## Benchmark

```bash
python tools/benchmark.py --model models/yolov8n-pose_openvino_model --imgsz 320 --device cpu --threads 2 --track --cameras 2 --source data/raw/uji01.mp4
```

Gunakan `--source` video kelas sungguhan bila ada (gambar sintetis noise tidak merepresentasikan adegan nyata). Keluaran penting: `latensi_ms.p50`/`p95` dan blok `rekomendasi` (`fps_per_camera`, `profile`).

Rumus rekomendasi (kasar): `fps_total = budget_cpu / latensi_detik`, lalu dibagi jumlah kamera, dengan `budget_cpu` bawaan 0,6. Anggapannya inferensi kamera berbagi CPU secara bergantian dan sisa 40% untuk MediaMTX, backend, dan sistem operasi.

| fps_per_camera hasil benchmark | Profil |
|---|---|
| >= 6 | `normal` (imgsz 480, 6 FPS) |
| 3 sampai < 6 | `low_cpu` (imgsz 320, 3 FPS) |
| < 3 | Terlalu berat. Lihat opsi di bawah |

Angka FPS profil `low_cpu` dan `normal` adalah titik awal, belum diverifikasi pada hardware sekolah.

## Jika CPU terlalu lemah

Urut dari yang paling murah:

1. Ekspor ke OpenVINO (Intel) atau ONNX Runtime, jangan jalankan `.pt`.
2. Turunkan `imgsz` ke 256/320 dan `fps` ke 2 sampai 3. Fall dari kamera atas masih mungkin berfungsi, tetapi recall kemungkinan turun: **evaluasi ulang** di data Anda.
3. Safe zone: pertahankan motion gate dan crop ROI (sudah bawaan). Ini paling hemat karena sering melewati inferensi.
4. Batasi thread (`infer.threads: 2`) agar MediaMTX dan backend tetap responsif. Tutup aplikasi lain, atur power plan ke High performance.
5. Jalankan hanya satu kamera pada satu waktu untuk tahap demo.
6. Jika semua gagal, pertimbangkan komputer pengolah AI yang lebih baru (mini-PC generasi terbaru). Ini keputusan sekolah, bukan teknis modul.

## Perbedaan LOQ dan komputer sekolah

LOQ memiliki GPU NVIDIA dan CPU yang jauh lebih kencang. Training dilakukan di LOQ (`device=0`). Benchmark inferensi harus memakai `--device cpu` dan `--threads` kecil. Anggap angka LOQ sebagai batas atas, dan jangan menyimpulkan performa sekolah dari LOQ.
