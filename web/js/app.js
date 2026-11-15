/**
 * SIA TK Bina Fitrah - Safety AI Web Dashboard Client
 * Menghubungkan website terpisah dengan API safety-ai via REST, MJPEG stream, dan SSE.
 */

// Konfigurasi API Base URL
function getApiBaseUrl() {
  const saved = localStorage.getItem("safety_api_url");
  if (saved) return saved.replace(/\/+$/, "");
  // Jika dibuka lewat browser di localhost port lain (misal 3000), default ke port 8000
  if (window.location.port !== "8000" && window.location.protocol.startsWith("http")) {
    return `${window.location.protocol}//${window.location.hostname}:8000`;
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
