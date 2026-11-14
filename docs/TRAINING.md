# Data, training, dan evaluasi

Ringkasan operasional dari PRD bagian 13 dan 14. Mulailah dari yang murah: **baseline pretrained + logika temporal** dulu. Fine-tune hanya jika baseline gagal memenuhi target.

## 1. Kumpulkan data (dengan izin)

```bash
python tools/record_clip.py --source 0 --seconds 120 --out data/raw/sesi01.mp4
```

- Sebelum ada izin merekam anak: pakai orang dewasa dan dataset publik (periksa lisensi).
- Untuk fall butuh **hard negative** berjam-jam (duduk lantai, jongkok, merangkak, tiarap, berlari) dan kejadian jatuh simulasi (>= 30, bervariasi arah dan kecepatan).
- Untuk safe zone butuh lintasan melewati dan mendekati garis (>= 40 masing-masing), plus dewasa lewat gerbang.
- Rekam dari posisi dan sudut pemasangan sebenarnya.

## 2. Ground truth

Salin `docs/templates/fall_gt.example.csv` atau `zone_gt.example.csv` ke `data/labels/`, isi dari menonton video: `source` = nama berkas video, `t_start` dalam detik video. Segmen `negative` menandai periode tanpa kejadian agar alarm palsu bisa diatribusikan ke skenario.

## 3. Baseline (tanpa training)

```bash
python -m safety_ai.service.runner --config configs/default.yaml --camera cam_kelas_a --source data/raw/sesi01.mp4 --run base01
python tools/evaluate.py --module fall --events "outputs/base01/events_*.jsonl" --gt data/labels/fall_gt.csv --hours 0.5
```

Tabel hasil di PRD bagian 14.5 diisi dari keluaran perintah ini dan `stats_<kamera>.json`.

## 4. Tuning ambang

Ubah hanya YAML (`fall:` dan `roi:`), jalankan ulang pada **set validasi**, bandingkan di `docs/EXPERIMENTS.md`. Test set dibuka sekali per versi model.

## 5. Fine-tune (jika perlu)

Pisahkan data **per sesi/hari** (lihat `docs/templates/splits.example.yaml`), bukan frame acak.

```bash
yolo detect train data=data/safezone.yaml model=yolov8n.pt imgsz=640 epochs=120 patience=40 batch=16 device=0 seed=42 name=sz_v1
yolo pose   train data=data/fall_pose.yaml model=yolov8n-pose.pt imgsz=480 epochs=120 patience=40 batch=16 device=0 seed=42 name=fall_pose_v1
yolo detect val model=runs/detect/sz_v1/weights/best.pt data=data/safezone.yaml split=test
yolo export model=runs/detect/sz_v1/weights/best.pt format=openvino imgsz=640
```

Epoch 100 sampai 150 (bukan 500 sampai 1000 seperti Tabel 2.17 C300). Setelah ekspor, **evaluasi ulang** pada format ekspor. Hasil fine-tune dipakai hanya jika mengalahkan baseline pada test set yang sama.

## 6. Membaca hasil training

Berkas di `runs/<tugas>/<nama>/`: `results.csv`, `results.png` (loss dan metrik per epoch), `PR_curve.png`, `F1_curve.png`, `confusion_matrix.png`, `val_batch*_pred.jpg`, `weights/best.pt`.

| Gejala | Arti | Tindakan |
|---|---|---|
| Loss train turun, loss val naik | Overfitting | Tambah data beragam, kurangi epoch, early stopping |
| Loss dan mAP buruk dan mendatar | Underfitting | Perbaiki anotasi, tambah data |
| mAP val tinggi, test atau lapangan rendah | Kebocoran data atau domain berbeda | Pisahkan per hari, tambah data lapangan |
| Recall rendah pada anak kecil/jauh | Resolusi efektif kecil | Crop ROI, naikkan imgsz, turunkan conf ke 0,25 sampai 0,3 |
| Skor turun setelah ekspor | Praproses berbeda atau INT8 | Periksa praproses, hindari INT8 |

Metrik model bukan penentu lulus. Penentu lulus adalah metrik event (`tools/evaluate.py`) dan soak test lapangan. Kriteria Go/No-Go ada di PRD bagian 14.8.

Interpretasi angka: 34 dari 40 kejadian terdeteksi = recall 85%, tetapi interval kepercayaan 95% = 71% sampai 93%. Klaim "recall >= 85%" membutuhkan lebih banyak kejadian uji. Nol alarm palsu dalam 20 jam hanya membuktikan laju alarm palsu < sekitar 0,15 per jam (aturan tiga).
