/**
 * ui.js - Logika TAMPILAN saja (tidak menyentuh backend).
 * Dimuat setelah app.js. Semua yang terhubung ke API tetap ada di app.js.
 */
(function () {
    // ---------- Toast ----------
    const toastEl = document.getElementById("toast");
    let toastTimer = null;
    function showToast(msg) {
        if (!toastEl) return;
        toastEl.textContent = msg;
        toastEl.classList.add("show");
        clearTimeout(toastTimer);
        toastTimer = setTimeout(() => toastEl.classList.remove("show"), 2600);
    }

    // ---------- Tanggal hari ini (jam digital diisi app.js) ----------
    const dateEl = document.getElementById("today-date");
    if (dateEl) {
        dateEl.textContent = new Date().toLocaleDateString("id-ID", {
            weekday: "long", day: "numeric", month: "long", year: "numeric",
        });
    }

    // ---------- Tab kamera ----------
    const tabs = document.querySelectorAll(".cam-tab");
    const panes = document.querySelectorAll(".cam-pane");
    tabs.forEach((tab) => {
        tab.addEventListener("click", () => {
            tabs.forEach((t) => t.classList.toggle("active", t === tab));
            panes.forEach((p) => p.classList.toggle("active", p.dataset.cam === tab.dataset.cam));
        });
    });

    // ---------- Status offline/online per kamera ----------
    // app.js sudah memakai img.onerror; di sini cukup menambah listener terpisah.
    document.querySelectorAll(".cam-pane").forEach((pane) => {
        const img = pane.querySelector(".video-stream");
        if (!img) return;
        img.addEventListener("load", () => pane.classList.remove("offline"));
        img.addEventListener("error", () => pane.classList.add("offline"));
    });

    // ---------- Efek alert-active: class dipasang app.js pada .cam-pane (sudah di-style) ----------

    // ---------- "Anak Terdeteksi" mengikuti data telemetri kamera kelas ----------
    const tracksEl = document.getElementById("tracks-kelas");
    const detectedEl = document.getElementById("stat-detected");
    if (tracksEl && detectedEl) {
        new MutationObserver(() => {
            const n = parseInt(tracksEl.textContent, 10);
            detectedEl.textContent = (Number.isNaN(n) ? 0 : n) + " Anak";
        }).observe(tracksEl, { childList: true, characterData: true, subtree: true });
    }

    // ---------- Dropdown akun ----------
    const userPill = document.getElementById("user-pill");
    const dropdown = document.getElementById("user-dropdown");
    if (userPill && dropdown) {
        userPill.addEventListener("click", (e) => {
            e.stopPropagation();
            const open = dropdown.classList.toggle("open");
            userPill.setAttribute("aria-expanded", String(open));
        });
        document.addEventListener("click", (e) => {
            if (!dropdown.contains(e.target)) {
                dropdown.classList.remove("open");
                userPill.setAttribute("aria-expanded", "false");
            }
        });
    }

    // ---------- Tombol statis ----------
    document.querySelectorAll('[data-action="playback"]').forEach((b) =>
        b.addEventListener("click", () => showToast("Fitur playback belum tersedia."))
    );
    const histBtn = document.querySelector('[data-action="history"]');
    if (histBtn) histBtn.addEventListener("click", () => showToast("Riwayat percakapan belum tersedia."));

    const landscape = document.getElementById("link-landscape");
    if (landscape) {
        landscape.addEventListener("click", (e) => {
            e.preventDefault();
            const active = document.querySelector(".cam-pane.active .video-wrapper");
            if (active && typeof toggleFullscreen === "function") toggleFullscreen(active.id);
        });
    }

    // ---------- BinaBot (statis, belum terhubung ke server) ----------
    const chatBody = document.getElementById("chat-body");
    const chatText = document.getElementById("chat-text");
    const chatSend = document.getElementById("chat-send");

    function addMsg(text, who) {
        const wrap = document.createElement("div");
        wrap.className = "msg " + who;
        const bubble = document.createElement("div");
        bubble.className = "bubble";
        bubble.textContent = text;
        const time = document.createElement("small");
        time.textContent = "Sekarang";
        wrap.append(bubble, time);
        chatBody.appendChild(wrap);
        chatBody.scrollTop = chatBody.scrollHeight;
    }

    function sendChat(text) {
        text = (text || "").trim();
        if (!text) return;
        addMsg(text, "user");
        setTimeout(() => {
            addMsg("Maaf, BinaBot masih berupa tampilan contoh dan belum terhubung ke server. Fitur jawaban otomatis akan aktif setelah modul chatbot tersambung.", "bot");
        }, 600);
    }

    if (chatSend) chatSend.addEventListener("click", () => { sendChat(chatText.value); chatText.value = ""; });
    if (chatText) chatText.addEventListener("keydown", (e) => {
        if (e.key === "Enter") { sendChat(chatText.value); chatText.value = ""; }
    });
    document.querySelectorAll(".suggest").forEach((b) =>
        b.addEventListener("click", () => sendChat(b.textContent))
    );
})();