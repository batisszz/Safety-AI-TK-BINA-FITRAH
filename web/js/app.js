/**
 * SIA TK Bina Fitrah - Safety AI Web Dashboard Client
 * Menghubungkan website terpisah dengan API safety-ai via REST, MJPEG stream, dan SSE.
 */

// Konfigurasi API Base URL
function getApiBaseUrl() {
  const saved = localStorage.getItem("safety_api_url");
  if (saved) return saved.replace(/\/+$/, "");
  const h = window.location.hostname;
  const isLocal = h === "localhost" || h === "127.0.0.1";
  if (isLocal && window.location.port !== "8000") {
    return `${window.location.protocol}//${h}:8000`;
  }
  return window.location.origin;
}

let API_BASE = getApiBaseUrl();
let eventSource = null;
let soundEnabled = true;
let currentFilter = "all";
let alertsData = [];

// ==========================================================================
// Web Audio Synthesizer (Zero file external)
// ==========================================================================
class SoundFX {
  constructor() {
    this.ctx = null;
  }

  _init() {
    if (!this.ctx) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (AudioCtx) this.ctx = new AudioCtx();
    }
    if (this.ctx && this.ctx.state === "suspended") {
      this.ctx.resume();
    }
  }

  playWarning() {
    if (!soundEnabled) return;
    try {
      this._init();
      if (!this.ctx) return;
      const now = this.ctx.currentTime;
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();

      osc.type = "sine";
      osc.frequency.setValueAtTime(659.25, now); // E5
      osc.frequency.setValueAtTime(880.00, now + 0.12); // A5

      gain.gain.setValueAtTime(0.2, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.35);

      osc.connect(gain);
      gain.connect(this.ctx.destination);

      osc.start(now);
      osc.stop(now + 0.35);
    } catch (e) {
      console.warn("Audio play warning error:", e);
    }
  }

  playDanger() {
    if (!soundEnabled) return;
    try {
      this._init();
      if (!this.ctx) return;
      const now = this.ctx.currentTime;

      // Dua nada peringatan cepat bernada tinggi
      [0, 0.18].forEach((delay) => {
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();

        osc.type = "sawtooth";
        osc.frequency.setValueAtTime(880, now + delay);
        osc.frequency.exponentialRampToValueAtTime(1174.66, now + delay + 0.14);

        gain.gain.setValueAtTime(0.3, now + delay);
        gain.gain.exponentialRampToValueAtTime(0.001, now + delay + 0.16);

        osc.connect(gain);
        gain.connect(this.ctx.destination);

        osc.start(now + delay);
        osc.stop(now + delay + 0.16);
      });
    } catch (e) {
      console.warn("Audio play danger error:", e);
    }
  }
}

const sfx = new SoundFX();

// ==========================================================================
// Inisialisasi Aliran Video Kamera
// ==========================================================================
function refreshCameraStream(cameraId) {
  const id = cameraId === "cam_kelas_a" ? "stream-kelas" : "stream-gerbang";
  const img = document.getElementById(id);
  if (!img) return;
  img.src = `${API_BASE}/api/cameras/${cameraId}/stream?t=${Date.now()}`;
}

function initCameraStreams() {
  const streamKelas = document.getElementById("stream-kelas");
  const streamGerbang = document.getElementById("stream-gerbang");

  if (streamKelas) {
    streamKelas.src = `${API_BASE}/api/cameras/cam_kelas_a/stream`;
    streamKelas.onerror = () => {
      console.warn("Stream kelas terputus, mencoba memulihkan...");
      setTimeout(() => {
        streamKelas.src = `${API_BASE}/api/cameras/cam_kelas_a/stream?t=${Date.now()}`;
      }, 3000);
    };
  }

  if (streamGerbang) {
    streamGerbang.src = `${API_BASE}/api/cameras/cam_gerbang/stream`;
    streamGerbang.onerror = () => {
      console.warn("Stream gerbang terputus, mencoba memulihkan...");
      setTimeout(() => {
        streamGerbang.src = `${API_BASE}/api/cameras/cam_gerbang/stream?t=${Date.now()}`;
      }, 3000);
    };
  }
}

// ==========================================================================
// Server-Sent Events (SSE) untuk Alert Real-Time
// ==========================================================================
function initSSE() {
  if (eventSource) {
    eventSource.close();
  }

  const sseUrl = `${API_BASE}/api/alerts/events`;
  console.log(`Menyambungkan SSE ke: ${sseUrl}`);
  eventSource = new EventSource(sseUrl);

  const statusDot = document.getElementById("conn-dot");
  const statusText = document.getElementById("conn-text");

  eventSource.onopen = () => {
    console.log("SSE Terhubung ke server Safety AI!");
    if (statusDot) statusDot.classList.remove("disconnected");
    if (statusText) statusText.textContent = "TERHUBUNG KE AI";
  };

  eventSource.addEventListener("connected", (e) => {
    console.log("Handshake SSE diterima:", e.data);
  });

  eventSource.addEventListener("alert", (e) => {
    try {
      const alert = JSON.parse(e.data);
      console.log("[LIVE ALERT]", alert);
      onNewAlertReceived(alert);
    } catch (err) {
      console.error("Gagal parse alert SSE:", err);
    }
  });

  eventSource.addEventListener("alert_updated", (e) => {
    try {
      const updated = JSON.parse(e.data);
      updateAlertInDom(updated);
    } catch (err) {
      console.error("Gagal parse alert_updated:", err);
    }
  });

  eventSource.onerror = (err) => {
    console.warn("Koneksi SSE terputus, reconnect otomatis...", err);
    if (statusDot) statusDot.classList.add("disconnected");
    if (statusText) statusText.textContent = "MENGHUBUNGKAN ULANG...";
  };
}

// Tangani Alert baru yang masuk
function onNewAlertReceived(alert) {
  // Bunyikan audio sesuai level
  if (alert.level >= 2) {
    sfx.playDanger();
  } else if (alert.level === 1) {
    sfx.playWarning();
  }

  // Efek visual glow pada card kamera
  triggerCameraAlertEffect(alert.camera_id, alert.level);

  // Masukkan ke array data lokal
  alertsData.unshift(alert);
  if (alertsData.length > 200) alertsData.pop();

  renderAlerts();
  updateSummaryMetrics();
}

// Efek border merah berdenyut pada kamera saat ada deteksi kritis
function triggerCameraAlertEffect(cameraId, level) {
  const cardId = cameraId === "cam_kelas_a" ? "card-cam-kelas" : "card-cam-gerbang";
  const card = document.getElementById(cardId);
  if (card && level >= 1) {
    card.classList.add("alert-active");
    setTimeout(() => {
      card.classList.remove("alert-active");
    }, 6000);
  }
}

// ==========================================================================
// Pengambilan Data Awal & Telemetri
// ==========================================================================
async function fetchInitialAlerts() {
  try {
    const res = await fetch(`${API_BASE}/api/alerts?limit=50`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    alertsData = data.items || [];
    renderAlerts();
    updateSummaryMetrics();
  } catch (e) {
    console.warn("Gagal memuat alert awal:", e);
  }
}

async function updateTelemetry() {
  try {
    const res = await fetch(`${API_BASE}/api/cameras`);
    if (!res.ok) return;
    const cams = await res.json();

    cams.forEach((cam) => {
      if (cam.camera_id === "cam_kelas_a") {
        setText("fps-kelas", `${cam.fps.toFixed(1)} FPS`);
        setText("lat-kelas", `${cam.latency_ms.toFixed(0)} ms`);
        setText("tracks-kelas", `${cam.tracks_count} Orang`);
        updateCameraStatusBadge("badge-status-kelas", cam);
      } else if (cam.camera_id === "cam_gerbang") {
        setText("fps-gerbang", `${cam.fps.toFixed(1)} FPS`);
        setText("lat-gerbang", `${cam.latency_ms.toFixed(0)} ms`);
        setText("tracks-gerbang", `${cam.tracks_count} Orang`);
        updateCameraStatusBadge("badge-status-gerbang", cam);
      }
    });
  } catch (e) {
    // Abaikan kegagalan sementara
  }
}

function updateCameraStatusBadge(elementId, cam) {
  const el = document.getElementById(elementId);
  if (!el) return;

  const tracks = cam.tracks || [];
  let isDanger = false;
  let isWarning = false;

  for (const t of tracks) {
    if (["CONFIRMED", "BREACH"].includes(t.state)) isDanger = true;
    if (["SUSPECTED", "DROPPING", "WARNING", "PENDING"].includes(t.state)) isWarning = true;
  }

  if (isDanger) {
    el.className = "cam-status-pill status-danger-pill";
    el.textContent = cam.camera_id === "cam_kelas_a" ? "JATUH TERKONFIRMASI!" : "PELANGGARAN ZONA!";
  } else if (isWarning) {
    el.className = "cam-status-pill status-warning-pill";
    el.textContent = cam.camera_id === "cam_kelas_a" ? "TERDUGA JATUH" : "PERINGATAN BATAS";
  } else {
    el.className = "cam-status-pill status-safe-pill";
    el.textContent = "AMAN / NORMAL";
  }
}

function updateSummaryMetrics() {
  const total = alertsData.length;
  const unresolved = alertsData.filter((a) => a.status === "Unresolved").length;
  const falls = alertsData.filter((a) => a.alert_type.includes("fall")).length;
  const zones = alertsData.filter((a) => a.alert_type.includes("zone")).length;

  setText("metric-total-alerts", total);
  setText("metric-unresolved", unresolved);
  setText("metric-falls", falls);
  setText("metric-zones", zones);
  setText("badge-unresolved-count", unresolved);
}

// ==========================================================================
// Rendering Alert Feed
// ==========================================================================
function renderAlerts() {
  const container = document.getElementById("alert-feed");
  if (!container) return;

  const filtered = alertsData.filter((item) => {
    if (currentFilter === "all") return true;
    if (currentFilter === "fall") return item.alert_type.includes("fall");
    if (currentFilter === "safezone") return item.alert_type.includes("zone");
    if (currentFilter === "unresolved") return item.status === "Unresolved";
    return true;
  });

  if (filtered.length === 0) {
    container.innerHTML = `
      <div class="alert-empty">
        <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">🛡️</div>
        <p>Belum ada notifikasi insiden pada filter ini.</p>
        <span style="font-size: 0.75rem; color: var(--text-muted);">Sistem memonitor aktivitas kamera secara aktif.</span>
      </div>
    `;
    return;
  }

  container.innerHTML = filtered
    .map((alert) => {
      const isLevel2 = alert.level >= 2;
      const isLevel1 = alert.level === 1;
      const cardClass = isLevel2 ? "level-2" : isLevel1 ? "level-1" : "resolved";
      const iconClass = isLevel2 ? "danger" : isLevel1 ? "warning" : "safe";

      let iconSymbol = "⚠️";
      let titleReadable = alert.alert_type;

      if (alert.alert_type === "fall_confirmed") {
        iconSymbol = "🚨";
        titleReadable = "JATUH TERKONFIRMASI (Level 2)";
      } else if (alert.alert_type === "fall_suspected") {
        iconSymbol = "⚠️";
        titleReadable = "TERDUGA JATUH (Level 1)";
      } else if (alert.alert_type === "zone_breach") {
        iconSymbol = "⛔";
        titleReadable = "KELUAR ZONA AMAN (Level 2)";
      } else if (alert.alert_type === "zone_warning") {
        iconSymbol = "ℹ️";
        titleReadable = "Mendekati Batas Aman (Level 0)";
      }

      const camName = alert.camera_id === "cam_kelas_a" ? "Kamera Kelas A" : "Kamera Gerbang";
      const statusClass = (alert.status || "Unresolved").replace(/\s+/g, "-");
      const timeFmt = formatTime(alert.timestamp);

      // Snapshot thumb
      const snapshotHtml = alert.snapshot_path
        ? `<img class="alert-snapshot-thumb" src="${API_BASE}/api/snapshots/${alert.snapshot_path}" 
             title="Lihat snapshot kejadian" onclick="openSnapshotModal('${alert.event_id}')" 
             onerror="this.style.display='none'">`
        : "";

      return `
        <div class="alert-card ${cardClass}" id="card-alert-${alert.event_id}">
          <div class="alert-icon-box ${iconClass}">${iconSymbol}</div>
          <div class="alert-body">
            <div class="alert-title-row">
              <span class="alert-type-badge">${titleReadable}</span>
              <span class="alert-cam-tag">${camName}</span>
              <span class="alert-status-badge ${statusClass}">${alert.status || "Unresolved"}</span>
            </div>
            <div class="alert-meta">
              <span>🕒 ${timeFmt}</span>
              <span>👤 Track ID: ${alert.track_id ?? "-"}</span>
              <span>🎯 Conf: ${Math.round((alert.confidence || 0) * 100)}%</span>
              ${formatMetaPills(alert.meta)}
            </div>
          </div>
          <div class="alert-actions">
            ${snapshotHtml}
            <button class="btn-sm btn-warning" onclick="openVerifyModal('${alert.event_id}')" title="Verifikasi Insiden">
              Periksa
            </button>
          </div>
        </div>
      `;
    })
    .join("");
}

function updateAlertInDom(updated) {
  const idx = alertsData.findIndex((a) => a.event_id === updated.event_id);
  if (idx !== -1) {
    alertsData[idx] = updated;
    renderAlerts();
    updateSummaryMetrics();
  }
}

function formatMetaPills(meta) {
  if (!meta) return "";
  const pills = [];
  if (meta.torso_angle) pills.push(`Sudut: ${Math.round(meta.torso_angle)}°`);
  if (meta.hip_speed) pills.push(`Speed: ${meta.hip_speed.toFixed(1)}x`);
  if (meta.mode) pills.push(`Mode: ${meta.mode}`);
  if (meta.message) pills.push(`Info: ${meta.message}`);
  return pills.map((p) => `<span>• ${p}</span>`).join(" ");
}

function formatTime(isoStr) {
  if (!isoStr) return "-";
  try {
    const d = new Date(isoStr);
    return d.toLocaleTimeString("id-ID", { hour: "2-digit", minute: "2-digit", second: "2-digit" }) + " WIB";
  } catch (e) {
    return isoStr;
  }
}

// ==========================================================================
// Verifikasi & Acknowledge Modal
// ==========================================================================
let activeModalAlertId = null;

function openVerifyModal(eventId) {
  const alert = alertsData.find((a) => a.event_id === eventId);
  if (!alert) return;

  activeModalAlertId = eventId;
  setText("modal-alert-title", `Verifikasi: ${alert.alert_type}`);
  setText(
    "modal-alert-desc",
    `Kamera: ${alert.camera_id} | Track: ${alert.track_id} | Waktu: ${formatTime(alert.timestamp)}`
  );

  const imgEl = document.getElementById("modal-snapshot-img");
  if (alert.snapshot_path) {
    imgEl.src = `${API_BASE}/api/snapshots/${alert.snapshot_path}`;
    imgEl.style.display = "block";
  } else {
    imgEl.style.display = "none";
  }

  document.getElementById("verify-operator").value = alert.verified_by || "Guru Piket";
  document.getElementById("verify-note").value = alert.verification_note || "";

  document.getElementById("modal-verify").classList.add("open");
}

function openSnapshotModal(eventId) {
  openVerifyModal(eventId);
}

function closeVerifyModal() {
  document.getElementById("modal-verify").classList.remove("open");
  activeModalAlertId = null;
}

async function submitAlertVerification(newStatus) {
  if (!activeModalAlertId) return;

  const operator = document.getElementById("verify-operator").value.trim();
  const note = document.getElementById("verify-note").value.trim();

  try {
    const res = await fetch(`${API_BASE}/api/alerts/${activeModalAlertId}/acknowledge`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        status: newStatus,
        operator: operator || "Guru Piket",
        note: note || undefined,
      }),
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    updateAlertInDom(data.alert);
    closeVerifyModal();
  } catch (err) {
    alert(`Gagal memverifikasi alert: ${err.message}`);
  }
}

// ==========================================================================
// Pengaturan Koneksi API
// ==========================================================================
function openSettingsModal() {
  document.getElementById("setting-api-url").value = API_BASE;
  document.getElementById("modal-settings").classList.add("open");
}

function closeSettingsModal() {
  document.getElementById("modal-settings").classList.remove("open");
}

function saveApiSettings() {
  const input = document.getElementById("setting-api-url").value.trim();
  if (input) {
    API_BASE = input.replace(/\/+$/, "");
    localStorage.setItem("safety_api_url", API_BASE);
    closeSettingsModal();
    initCameraStreams();
    initSSE();
    fetchInitialAlerts();
    alert(`API Base URL diperbarui ke: ${API_BASE}`);
  }
}

// ==========================================================================
// Kontrol Sumber Kamera (file / browser webcam HP / RTSP)
// ==========================================================================
let activeSourceCameraId = null;
let browserCapture = { cameraId: null, stream: null, interval: null, sending: false };

function stopBrowserCapture(targetCameraId = null) {
  if (targetCameraId && browserCapture.cameraId && browserCapture.cameraId !== targetCameraId) return;
  if (browserCapture.interval) {
    clearInterval(browserCapture.interval);
    browserCapture.interval = null;
  }
  if (browserCapture.stream) {
    try { browserCapture.stream.getTracks().forEach((t) => t.stop()); } catch (e) {}
    browserCapture.stream = null;
  }
  browserCapture.cameraId = null;
  browserCapture.sending = false;
  const preview = document.getElementById("source-browser-preview");
  if (preview) {
    try { preview.srcObject = null; } catch (e) { preview.src = ""; }
  }
}

function onSourceTypeChange() {
  const type = (document.getElementById("source-type") || {}).value || "file";
  document.getElementById("source-panel-file").style.display = type === "file" ? "" : "none";
  document.getElementById("source-panel-browser").style.display = type === "browser" ? "" : "none";
  document.getElementById("source-panel-rtsp").style.display = type === "rtsp" ? "" : "none";
  if (type !== "browser") {
    // hentikan preview lokal bila ganti tipe, capture aktif background dikelola terpisah
    const preview = document.getElementById("source-browser-preview");
    if (preview && preview.srcObject && browserCapture.cameraId !== activeSourceCameraId) {
      // preview sementara dari modal lain tidak dipakai; biarkan capture global apa adanya
    }
  }
}

async function loadSourceVideos(selectedPath) {
  const sel = document.getElementById("source-file-select");
  if (!sel) return;
  sel.innerHTML = '<option value="">Memuat...</option>';
  try {
    const res = await fetch(`${API_BASE}/api/sources/videos`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const items = data.items || [];
    if (items.length === 0) {
      sel.innerHTML = '<option value="">Tidak ada file video ditemukan (data/raw)</option>';
      return;
    }
    sel.innerHTML = items.map((it) => `<option value="${it.path}">${it.name} — ${it.path}</option>`).join("");
    if (selectedPath) sel.value = selectedPath;
  } catch (e) {
    sel.innerHTML = '<option value="">Gagal memuat daftar video</option>';
  }
}

async function openSourceModal(cameraId) {
  activeSourceCameraId = cameraId;
  const camName = cameraId === "cam_kelas_a" ? "Kamera Kelas A" : "Kamera Gerbang";
  setText("source-modal-title", `Ganti Sumber - ${camName}`);
  setText("source-modal-desc", `Pilih sumber frame untuk ${camName}. Perubahan diterapkan ke server AI.`);
  setText("source-status-msg", "");

  // Reset fields
  const host = document.getElementById("rtsp-host"); if (host) host.value = "";
  const port = document.getElementById("rtsp-port"); if (port) port.value = "554";
  const path = document.getElementById("rtsp-path"); if (path) path.value = "stream1";
  const user = document.getElementById("rtsp-user"); if (user) user.value = "";
  const pass = document.getElementById("rtsp-pass"); if (pass) pass.value = "";
  const trans = document.getElementById("rtsp-transport"); if (trans) trans.value = "tcp";

  let curPath = null;
  let curType = "file";
  try {
    const res = await fetch(`${API_BASE}/api/cameras/${cameraId}/source`);
    if (res.ok) {
      const data = await res.json();
      const src = data.source || {};
      curType = src.type || "file";
      if (src.type === "file") curPath = src.path || null;
      if (src.type === "rtsp" && src.url) {
        try {
          const u = new URL(src.url);
          if (host) host.value = u.hostname || "";
          if (port) port.value = u.port || "554";
          if (path) path.value = (u.pathname || "/stream1").replace(/^\//, "");
          if (user) user.value = decodeURIComponent(u.username || "");
          if (pass) pass.value = decodeURIComponent(u.password || "");
        } catch (e) {}
        if (trans) trans.value = src.rtsp_transport || "tcp";
      }
    }
  } catch (e) {}

  const typeSel = document.getElementById("source-type");
  if (typeSel) typeSel.value = (curType === "webcam" ? "file" : curType); // webcam server lama -> tampilkan sebagai pilihan file/browser manual
  await loadSourceVideos(curPath);
  onSourceTypeChange();
  document.getElementById("modal-source").classList.add("open");
}

function closeSourceModal() {
  // jangan hentikan capture browser global di sini; hanya tutup preview modal
  const preview = document.getElementById("source-browser-preview");
  if (preview && browserCapture.cameraId !== activeSourceCameraId) {
    try { preview.srcObject = null; } catch (e) {}
  }
  document.getElementById("modal-source").classList.remove("open");
  activeSourceCameraId = null;
}

function buildRtspUrl() {
  const host = (document.getElementById("rtsp-host").value || "").trim();
  const port = (document.getElementById("rtsp-port").value || "").trim();
  let rpath = (document.getElementById("rtsp-path").value || "").trim();
  const user = (document.getElementById("rtsp-user").value || "").trim();
  const pass = document.getElementById("rtsp-pass").value || "";
  if (!host) throw new Error("IP/Host RTSP harus diisi");
  if (rpath.startsWith("/")) rpath = rpath.slice(1);
  const auth = user ? `${encodeURIComponent(user)}:${encodeURIComponent(pass)}@` : "";
  const portPart = port ? `:${port}` : "";
  const pathPart = rpath ? `/${rpath}` : "";
  return `rtsp://${auth}${host}${portPart}${pathPart}`;
}

async function startBrowserCapture(cameraId, facingMode) {
  stopBrowserCapture(); // satu capture aktif
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("Browser tidak mendukung akses kamera (getUserMedia)");
  }
  const constraints = { video: { facingMode: { ideal: facingMode } }, audio: false };
  const stream = await navigator.mediaDevices.getUserMedia(constraints);
  browserCapture.cameraId = cameraId;
  browserCapture.stream = stream;
  browserCapture.sending = false;

  const preview = document.getElementById("source-browser-preview");
  if (preview) {
    preview.srcObject = stream;
    try { await preview.play(); } catch (e) {}
  }

  const video = document.createElement("video");
  video.autoplay = true; video.playsInline = true; video.muted = true; video.srcObject = stream;
  try { await video.play(); } catch (e) {}
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");

  browserCapture.interval = setInterval(async () => {
    if (browserCapture.sending) return;
    const w = video.videoWidth, h = video.videoHeight;
    if (!w || !h) return;
    canvas.width = w; canvas.height = h;
    ctx.drawImage(video, 0, 0, w, h);
    browserCapture.sending = true;
    try {
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.6));
      if (blob) {
        await fetch(`${API_BASE}/api/cameras/${cameraId}/source/frame`, {
          method: "POST",
          headers: { "Content-Type": "image/jpeg" },
          body: blob,
        });
      }
    } catch (e) {
      // abaikan error kirim sesaat
    } finally {
      browserCapture.sending = false;
    }
  }, 300);
}

async function applyCameraSource() {
  const cameraId = activeSourceCameraId;
  if (!cameraId) return;
  const type = document.getElementById("source-type").value;
  const statusEl = document.getElementById("source-status-msg");
  const setStatus = (msg) => { if (statusEl) statusEl.textContent = msg; };
  try {
    if (type === "file") {
      const path = document.getElementById("source-file-select").value;
      if (!path) throw new Error("Pilih file video terlebih dahulu");
      stopBrowserCapture(cameraId);
      setStatus("Menerapkan sumber file...");
      const res = await fetch(`${API_BASE}/api/cameras/${cameraId}/source`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ type: "file", path }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      refreshCameraStream(cameraId);
      closeSourceModal();
      showToast("Sumber video file diterapkan");
      return;
    }

    if (type === "browser") {
      const facing = document.getElementById("source-facing").value || "environment";
      setStatus("Mengaktifkan mode kamera browser di server...");
      const res = await fetch(`${API_BASE}/api/cameras/${cameraId}/source`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ type: "browser" }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setStatus("Meminta akses kamera perangkat...");
      await startBrowserCapture(cameraId, facing);
      refreshCameraStream(cameraId);
      closeSourceModal();
      showToast("Kamera perangkat aktif. Frame dikirim ke server untuk deteksi AI.");
      return;
    }

    if (type === "rtsp") {
      const url = buildRtspUrl();
      const rtsp_transport = document.getElementById("rtsp-transport").value || "tcp";
      stopBrowserCapture(cameraId);
      setStatus("Menerapkan sumber RTSP...");
      const res = await fetch(`${API_BASE}/api/cameras/${cameraId}/source`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ type: "rtsp", url, rtsp_transport }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      refreshCameraStream(cameraId);
      closeSourceModal();
      showToast("Sumber RTSP diterapkan");
      return;
    }
  } catch (err) {
    console.error(err);
    setStatus("");
    alert(`Gagal menerapkan sumber: ${err.message}`);
  }
}

// ==========================================================================
// ROI Zona Aman (Kamera Gerbang) - gambar poligon di website
// ==========================================================================
let activeRoiCameraId = null;
let roiPoints = []; // normalized [x,y]

function updateRoiPointCount() {
  const el = document.getElementById("roi-point-count");
  if (el) el.textContent = `${roiPoints.length} titik`;
}

function getRoiCanvasContext() {
  const canvas = document.getElementById("roi-canvas");
  const img = document.getElementById("roi-base-image");
  return { canvas, img, ctx: canvas ? canvas.getContext("2d") : null };
}

function syncRoiCanvasSize() {
  const { canvas, img } = getRoiCanvasContext();
  if (!canvas || !img) return;
  const w = img.clientWidth;
  const h = img.clientHeight;
  if (w && h) {
    canvas.width = w;
    canvas.height = h;
  }
}

function drawRoiPolygon() {
  const { canvas, ctx } = getRoiCanvasContext();
  if (!canvas || !ctx) return;
  syncRoiCanvasSize();
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (roiPoints.length === 0) { updateRoiPointCount(); return; }

  const pts = roiPoints.map((p) => ({ x: p[0] * w, y: p[1] * h }));
  if (pts.length >= 2) {
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
    if (pts.length >= 3) ctx.closePath();
    ctx.fillStyle = "rgba(46, 204, 113, 0.18)";
    ctx.strokeStyle = "rgba(46, 204, 113, 0.95)";
    ctx.lineWidth = 2;
    if (pts.length >= 3) ctx.fill();
    ctx.stroke();
  }
  pts.forEach((pt, i) => {
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, 5, 0, Math.PI * 2);
    ctx.fillStyle = "#00d084";
    ctx.fill();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.fillStyle = "#ffffff";
    ctx.font = "12px sans-serif";
    ctx.fillText(String(i + 1), pt.x + 7, pt.y - 7);
  });
  updateRoiPointCount();
}

function onRoiCanvasClick(e) {
  const canvas = document.getElementById("roi-canvas");
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) / rect.width;
  const y = (e.clientY - rect.top) / rect.height;
  if (x >= 0 && x <= 1 && y >= 0 && y <= 1) {
    roiPoints.push([Math.round(x * 10000) / 10000, Math.round(y * 10000) / 10000]);
    drawRoiPolygon();
  }
}

async function reloadRoiFromServer() {
  if (!activeRoiCameraId) return;
  try {
    const res = await fetch(`${API_BASE}/api/cameras/${activeRoiCameraId}/roi`);
    if (res.ok) {
      const data = await res.json();
      roiPoints = Array.isArray(data.safe_polygon) ? data.safe_polygon.map((p) => [Number(p[0]), Number(p[1])]) : [];
    }
  } catch (e) {}
  drawRoiPolygon();
}

async function openRoiModal(cameraId) {
  activeRoiCameraId = cameraId;
  roiPoints = [];
  const img = document.getElementById("roi-base-image");
  const canvas = document.getElementById("roi-canvas");
  if (canvas) {
    canvas.onclick = onRoiCanvasClick;
  }
  document.getElementById("modal-roi").classList.add("open");
  if (img) {
    img.onload = () => {
      syncRoiCanvasSize();
      drawRoiPolygon();
    };
    img.src = `${API_BASE}/api/cameras/${cameraId}/frame?t=${Date.now()}`;
  }
  await reloadRoiFromServer();
  // redraw after modal layout
  setTimeout(() => { syncRoiCanvasSize(); drawRoiPolygon(); }, 50);
}

function closeRoiModal() {
  document.getElementById("modal-roi").classList.remove("open");
  activeRoiCameraId = null;
  roiPoints = [];
}

function undoRoiPoint() {
  if (roiPoints.length > 0) {
    roiPoints.pop();
    drawRoiPolygon();
  }
}

function clearRoiPoints() {
  roiPoints = [];
  drawRoiPolygon();
}

async function saveRoiPolygon() {
  if (!activeRoiCameraId) return;
  if (roiPoints.length < 3) {
    alert("Poligon ROI minimal membutuhkan 3 titik.");
    return;
  }
  try {
    const res = await fetch(`${API_BASE}/api/cameras/${activeRoiCameraId}/roi`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ safe_polygon: roiPoints }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    await res.json();
    showToast("ROI zona aman disimpan dan diterapkan");
    closeRoiModal();
    refreshCameraStream(activeRoiCameraId);
  } catch (err) {
    alert(`Gagal menyimpan ROI: ${err.message}`);
  }
}

// Redraw ROI canvas on resize while modal open
window.addEventListener("resize", () => {
  const modal = document.getElementById("modal-roi");
  if (modal && modal.classList.contains("open")) {
    syncRoiCanvasSize();
    drawRoiPolygon();
  }
});

// ==========================================================================
// Helper Controls (Fullscreen, Snapshot, Mute, Trigger Test)
// ==========================================================================
function toggleFullscreen(elementId) {
  const el = document.getElementById(elementId);
  if (!el) return;
  if (!document.fullscreenElement) {
    el.requestFullscreen().catch((err) => alert(err.message));
  } else {
    document.exitFullscreen();
  }
}

function downloadSnapshot(cameraId) {
  window.open(`${API_BASE}/api/cameras/${cameraId}/frame`, "_blank");
}

function toggleAudio() {
  soundEnabled = !soundEnabled;
  const btn = document.getElementById("btn-toggle-audio");
  if (btn) {
    btn.innerHTML = soundEnabled ? "🔔 Audio: Aktif" : "🔕 Audio: Senyap";
    btn.style.color = soundEnabled ? "#fff" : "var(--text-muted)";
  }
}

async function triggerSimulationAlert(type = "fall_confirmed") {
  try {
    const isFall = type.startsWith("fall");
    const res = await fetch(`${API_BASE}/api/test/trigger-alert`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        camera_id: isFall ? "cam_kelas_a" : "cam_gerbang",
        alert_type: type,
        level: type.includes("confirmed") || type.includes("breach") ? 2 : 1,
        track_id: Math.floor(Math.random() * 20) + 1,
        confidence: 0.88,
        message: "Uji simulasi alert visual & audio dari dashboard",
      }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
  } catch (err) {
    alert(`Gagal memicu simulasi: ${err.message}`);
  }
}

function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

// Jam digital Navbar
function updateClock() {
  const el = document.getElementById("digital-clock");
  if (!el) return;
  const now = new Date();
  el.textContent = now.toLocaleTimeString("id-ID", { hour12: false }) + " WIB";
}

// ==========================================================================
// DOM Ready
// ==========================================================================
document.addEventListener("DOMContentLoaded", () => {
  // Update Jam
  setInterval(updateClock, 1000);
  updateClock();

  // Inisialisasi Aliran Video & SSE
  initCameraStreams();
  initSSE();
  fetchInitialAlerts();

  // Polling berkala telemetri FPS & Track count
  setInterval(updateTelemetry, 2000);
  updateTelemetry();

  // Filter Buttons
  document.querySelectorAll(".filter-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".filter-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      currentFilter = btn.getAttribute("data-filter");
      renderAlerts();
    });
  });

  // Tombol Audio Mute Toggle
  const btnAudio = document.getElementById("btn-toggle-audio");
  if (btnAudio) btnAudio.addEventListener("click", toggleAudio);
});
