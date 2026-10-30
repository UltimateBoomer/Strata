// serve/web/console.js - the fork's page at /: Monitor and About. No framework, no network beyond this server.
// Monitor reads GET /metrics every second (the engine, the turn scheduler's queue, the model variants, the parked
// conversations on disk).
"use strict";

const $ = (id) => document.getElementById(id);
const SPRITE = "web/sprite.svg";
const icon = (name, cls = "st-icon") => `<svg class="${cls}" aria-hidden="true"><use href="${SPRITE}#i-${name}"/></svg>`;
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const fmt = (n, d = 0) => (n == null || Number.isNaN(n) ? "–" : Number(n).toLocaleString(undefined, {maximumFractionDigits: d, minimumFractionDigits: d}));
const kfmt = (n) => (n == null ? "–" : n >= 1000 ? `${fmt(n / 1000, n >= 10000 ? 0 : 1)}k` : fmt(n));
const ctxfmt = (n) => (n && n % 1024 === 0 ? `${fmt(n / 1024)}K` : kfmt(n));
const gb = (b, d = 1) => (b == null ? "–" : fmt(b / 1073741824, d));
const size = (b) => (b == null ? "–" : b >= 1073741824 ? `${fmt(b / 1073741824, 2)} GB` : `${fmt(b / 1048576)} MB`);
const secs = (s) => (s == null ? "–" : s < 60 ? `${fmt(s, s < 10 ? 1 : 0)} s` : s < 3600 ? `${fmt(s / 60, 1)} min` : `${fmt(s / 3600, 1)} h`);
const ago = (t) => (t ? `${secs(Math.max(0, Date.now() / 1000 - t))} ago` : "–");

const store = {
  get(k, d) { try { const v = localStorage.getItem("strata." + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem("strata." + k, JSON.stringify(v)); return true; } catch (e) { return false; } },
};

function toast(kind, title, text = "", ms = 3500) {
  const names = {info: "info", success: "check", warn: "warning", error: "error"};
  const el = document.createElement("div");
  el.className = `st-toast st-toast--${kind}`;
  el.innerHTML = `${icon(names[kind] || "info")}<div><div class="st-toast__title"></div><div class="t-text"></div></div>`;
  el.querySelector(".st-toast__title").textContent = title;
  el.querySelector(".t-text").textContent = text;
  $("toasts").appendChild(el);
  setTimeout(() => el.remove(), ms);
}
async function copyText(text, btn) {
  try { await navigator.clipboard.writeText(text); } catch (e) {          // http on another host: no async clipboard
    const ta = document.createElement("textarea");
    ta.value = text; document.body.appendChild(ta); ta.select(); document.execCommand("copy"); ta.remove();
  }
  const use = btn && btn.querySelector("use");
  if (use) { use.setAttribute("href", `${SPRITE}#i-check`); setTimeout(() => use.setAttribute("href", `${SPRITE}#i-copy`), 1500); }
}
document.addEventListener("click", (e) => { const b = e.target.closest("[data-copy]"); if (b) copyText(b.dataset.copy, b); });

// ------------------------------------------------------------------ theme and tabs
function setTheme(t, save) {
  document.documentElement.dataset.theme = t;
  if (save) try { localStorage.setItem("strata.theme", t); } catch (e) { /* ignore */ }
  $("theme-icon").setAttribute("href", `${SPRITE}#i-${t === "dark" ? "sun" : "moon"}`);
  $("dark-toggle").setAttribute("aria-checked", String(t === "dark"));
}
const flipTheme = () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark", true);
$("theme-btn").onclick = flipTheme;
$("dark-toggle").onclick = flipTheme;
setTheme(document.documentElement.dataset.theme || "light", false);

const TABS = ["monitor", "about"];
let tab = "monitor";
function showTab(name) {
  tab = TABS.includes(name) ? name : "monitor";
  for (const b of document.querySelectorAll(".st-tab")) b.setAttribute("aria-selected", String(b.dataset.tab === tab));
  for (const v of TABS) $(`view-${v}`).hidden = v !== tab;
  if (location.hash.slice(1) !== tab) history.replaceState(null, "", tab === "monitor" ? location.pathname : `#${tab}`);
  if (lastMetrics) render(lastMetrics);
}
for (const b of document.querySelectorAll(".st-tab")) b.onclick = () => showTab(b.dataset.tab);
window.addEventListener("hashchange", () => showTab(location.hash.slice(1)));

// ------------------------------------------------------------------ server access
let apiKey = store.get("apikey", "");
function headers(json = false) {
  const h = {};
  if (apiKey) h.Authorization = "Bearer " + apiKey;
  if (json) h["Content-Type"] = "application/json";
  return h;
}
$("api-key").value = apiKey;
$("api-key").oninput = () => { apiKey = $("api-key").value.trim(); store.set("apikey", apiKey); };
$("api-key").onchange = () => toast("success", "API key saved", "Kept in this browser only.");
// a JSON call: the answer's body, or an Error carrying its status and message
async function api(path, body) {
  const r = await fetch(path, body === undefined ? {headers: headers(), cache: "no-store"}
    : {method: "POST", headers: headers(true), body: JSON.stringify(body)});
  let j = {};
  try { j = await r.json(); } catch (e) { /* not JSON */ }
  if (!r.ok) {
    const err = new Error((j.error && j.error.message) || (r.status === 401 ? "API key needed: add it under About." : `HTTP ${r.status}`));
    err.status = r.status; err.body = j;
    throw err;
  }
  return j;
}

// ------------------------------------------------------------------ Monitor: the metric cards
const METRICS = [
  {key: "speed", label: "Speed", icon: "gauge"},
  {key: "gpu", label: "GPU load", icon: "gpu", tone: ""},
  {key: "vram", label: "VRAM", icon: "layers"},
  {key: "temp", label: "GPU temp", icon: "thermometer", tone: "warn"},
  {key: "power", label: "Power", icon: "bolt"},
  {key: "pcie", label: "PCIe", icon: "link", tone: "info"},
  {key: "cpu", label: "CPU", icon: "cpu"},
  {key: "disk", label: "Disk read", icon: "disk", tone: "info"},
];
$("metrics").innerHTML = METRICS.map((m) => `
  <div class="st-card metric-card"><div class="st-metric">
    <span class="st-metric__label">${icon(m.icon, "st-icon st-icon--sm")}${esc(m.label)}</span>
    ${m.key === "speed" ? `<div class="speed-values">
      <div><span class="st-metric__value" id="mv-speed">–</span><span class="st-metric__sub" id="ms-speed">Decode</span></div>
      <div class="speed-prefill"><span class="st-metric__value" id="mv-prefill">–</span><span class="st-metric__sub" id="ms-prefill">Prefill</span></div>
    </div>` : `<span class="st-metric__value" id="mv-${m.key}">–</span><span class="st-metric__sub" id="ms-${m.key}"></span>`}
    <svg class="st-metric__spark" id="sp-${m.key}" viewBox="0 0 100 32" preserveAspectRatio="none"${m.tone ? ` data-tone="${m.tone}"` : ""}>
      <path class="area" fill="currentColor" opacity=".12"/><path class="line" fill="none" stroke="currentColor"
      stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/>
      ${m.key === "speed" ? `<g id="sp-prefill" class="speed-prefill"><path class="area" fill="currentColor" opacity=".12"/>
        <path class="line" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"
        stroke-linecap="round" vector-effect="non-scaling-stroke"/></g>` : ""}</svg>
  </div></div>`).join("");
function spark(id, values, max) {
  const svg = $(id);
  const v = (values || []).map((x) => (x == null ? 0 : x));
  if (v.length < 2) { svg.querySelector(".line").setAttribute("d", ""); svg.querySelector(".area").setAttribute("d", ""); return; }
  const top = Math.max(max || 0, ...v, 1);
  const line = v.map((x, i) => `${i ? "L" : "M"}${((i / (v.length - 1)) * 100).toFixed(2)},${(30 - (x / top) * 26).toFixed(2)}`).join("");
  svg.querySelector(".line").setAttribute("d", line);
  svg.querySelector(".area").setAttribute("d", `${line}L100,32L0,32Z`);
}
function setMetric(key, value, unit, sub) {
  $(`mv-${key}`).innerHTML = value == null ? "–" : `${esc(value)}${unit ? `<small>${esc(unit)}</small>` : ""}`;
  $(`ms-${key}`).textContent = sub || "";
}

// ------------------------------------------------------------------ polling, the header
let lastMetrics = null, failures = 0, keyWarned = false, reqShowAll = false, modelOp = null;
async function poll() {
  try {
    lastMetrics = await api(reqShowAll ? "metrics?requests=all" : "metrics");
    failures = 0;
    render(lastMetrics);
  } catch (e) {
    if (e.status === 401) {
      setPill("queued", "API key needed");
      if (!keyWarned) { keyWarned = true; toast("warn", "API key needed", "This server needs a key: add it under About.", 6000); }
    } else if (++failures === 3) setPill("queued", "Server not reachable");
  }
  setTimeout(poll, 1000);
}
let lastPillText = null;
function setPill(state, text) {
  $("pill").dataset.state = state;
  $("pill-text").textContent = text;
  if (text !== lastPillText) { lastPillText = text; fitHeader(); }
}
function fitHeader() {
  const header = document.querySelector(".st-header");
  header.classList.remove("st-header--clip");
  if (header.scrollWidth > header.clientWidth) header.classList.add("st-header--clip");
}
window.addEventListener("resize", fitHeader);

function modelButton(live) {
  const btn = $("unload-btn"), state = (live && live.state) || "";
  const set = (text, title, disabled) => { btn.textContent = text; btn.title = title; btn.disabled = disabled; };
  if (modelOp) return set(modelOp === "load" ? "Loading…" : "Unloading…", "", true);
  if (!state) return set("…", "Waiting for the server", true);
  if (state === "unloaded") return set("Load", "Load the model now. A request also loads it.", false);
  if (state === "reading" || state === "generating" || live.queued > 0) return set("Unload", "A request is running", true);
  return set("Unload", "Unload the model and free the GPU", false);
}
$("unload-btn").onclick = async () => {
  const live = (lastMetrics && lastMetrics.live) || {};
  const load = live.state === "unloaded";
  if (!load && !confirm("Unload the model and free the GPU? Load, or the next request, brings it back.")) return;
  modelOp = load ? "load" : "unload";
  modelButton(live);
  try {
    const r = await api(load ? "load" : "unload", {});
    if (!load && r.status === "busy") toast("warn", "Model is busy", "A request is running.");
    else toast("success", load ? "Model loaded" : "Model unloaded");
  } catch (e) {
    toast("error", load ? "Load failed" : "Unload failed", e.message, 6000);
  } finally {
    modelOp = null;
    if (lastMetrics) render(lastMetrics);
  }
};

function render(m) {
  const live = m.live || {};
  if (modelOp === "load") setPill("idle", "Loading…");
  else if (modelOp === "unload") setPill("idle", "Unloading…");
  else if (live.queued > 0 && live.state !== "reading" && live.state !== "generating") setPill("queued", `${live.queued} queued`);
  else if (live.state === "reading") {
    const pct = live.prompt_total ? Math.round((100 * live.prompt_read) / live.prompt_total) : null;
    setPill("reading", pct != null ? `Reading prompt · ${pct}%` : "Reading prompt");
  } else if (live.state === "generating") setPill("generating", `Generating · ${fmt(live.tok_s, 1)} tok/s`);
  else if (live.state === "unloaded") setPill("idle", "Unloaded");
  else setPill("idle", "Idle");
  modelButton(live);
  const chip = $("variant-chip"), mv = m.models;
  chip.hidden = !mv;
  if (mv) {
    chip.textContent = mv.committed && mv.committed !== mv.active ? `${mv.active} → ${mv.committed}` : mv.active;
    chip.title = `Loaded variant: ${mv.active} (${(mv.variants || {})[mv.active] || ""})`;
  }
  if (tab === "monitor") {
    renderMonitor(m);
    renderScheduler(m.turns);
    renderModels(m.models, m.turns, live);
    renderCache(m.conversation_cache);
  }
  if (tab === "about") renderAbout(m.engine || {}, m.hardware || {}, m.hardware_static || {});
}

// ------------------------------------------------------------------ Monitor: state, metrics, context, requests
function renderMonitor(m) {
  const live = m.live || {}, hw = m.hardware || {}, st = m.hardware_static || {}, eng = m.engine || {}, h = m.history || {};
  const requests = m.requests || [], last = requests[0];
  const on = live.queued > 0 ? "queued" : live.state;
  for (const b of document.querySelectorAll("#state-badges .st-badge")) b.classList.toggle("on", b.dataset.s === on || b.dataset.s === live.state);
  const prog = $("state-progress");
  let label = "Waiting for a request", detail = "", pct = 0;
  delete prog.dataset.tone;
  if (live.state === "reading") {
    label = "Reading prompt";
    prog.dataset.tone = "info";
    if (live.prompt_total) { pct = (100 * live.prompt_read) / live.prompt_total; detail = `${fmt(live.prompt_read)} / ${fmt(live.prompt_total)} tokens · ${fmt(pct)}%`; }
    else detail = `${fmt(live.prompt_tokens)} tokens`;
  } else if (live.state === "generating") {
    label = live.phase ? live.phase[0].toUpperCase() + live.phase.slice(1) : "Generating";
    pct = live.max_tokens ? Math.min(100, (100 * live.generated) / live.max_tokens) : 0;
    detail = `${fmt(live.generated)} tokens · ${fmt(live.tok_s, 1)} tok/s`;
  } else if (live.state === "unloaded") {
    label = "Unloaded"; detail = "Load brings it back. A request does too.";
  } else if (last) {
    detail = `last: ${fmt(last.output_tokens)} tokens${last.decode_tok_s ? ` at ${fmt(last.decode_tok_s, 1)} tok/s` : ""}`;
  }
  $("state-label").textContent = label;
  $("state-detail").textContent = detail;
  $("state-bar").style.width = `${pct}%`;

  const speed = live.state === "generating" ? live.tok_s : last ? last.decode_tok_s : null;
  setMetric("speed", speed == null ? null : fmt(speed, 1), "t/s", live.state === "generating" ? "Decode now" : last ? "Decode last request" : "Decode");
  const prefill = live.state !== "idle" ? live.prefill_tok_s_mean
    : last && last.prompt_ms > 0 ? Math.max(0, last.prompt_tokens - (last.reused || 0)) / (last.prompt_ms / 1000) : null;
  setMetric("prefill", prefill == null ? null : fmt(prefill), "t/s", live.state === "reading" ? "Prefill now" : last ? "Prefill last request" : "Prefill");
  spark("sp-speed", h.tok_s);
  spark("sp-prefill", h.prefill_tok_s_mean);
  setMetric("gpu", hw.gpu_util == null ? null : fmt(hw.gpu_util), "%", st.gpu_name || "");
  spark("sp-gpu", h.gpu_util, 100);
  setMetric("vram", hw.gpu_mem_used == null ? null : gb(hw.gpu_mem_used), hw.gpu_mem_total ? `/ ${gb(hw.gpu_mem_total, 0)} GB` : "GB",
            eng.expert_slots ? `${fmt(eng.expert_slots)} experts cached` : "");
  spark("sp-vram", h.gpu_mem_used, hw.gpu_mem_total);
  setMetric("temp", hw.gpu_temp == null ? null : fmt(hw.gpu_temp), "°C", "");
  spark("sp-temp", h.gpu_temp, 90);
  setMetric("power", hw.gpu_power == null ? null : fmt(hw.gpu_power), "W", hw.gpu_power_limit ? `of ${fmt(hw.gpu_power_limit)} W limit` : "");
  spark("sp-power", h.gpu_power, hw.gpu_power_limit);
  const gen = hw.gpu_pcie_gen_max || hw.gpu_pcie_gen;
  setMetric("pcie", gen ? `Gen${gen}` : null, hw.gpu_pcie_width ? `x${hw.gpu_pcie_width}` : "",
            hw.gpu_pcie_rx_mb == null ? "" : `to GPU ${fmt(hw.gpu_pcie_rx_mb, hw.gpu_pcie_rx_mb < 10 ? 1 : 0)} MB/s`);
  spark("sp-pcie", h.gpu_pcie_rx_mb);
  setMetric("cpu", hw.cpu == null ? null : fmt(hw.cpu), "%", st.threads ? `${st.cores ? `${st.cores} cores · ` : ""}${st.threads} threads` : "");
  spark("sp-cpu", h.cpu, 100);
  if (hw.disk_read_mb == null) setMetric("disk", null, "", "");
  else {
    const big = hw.disk_read_mb >= 1000;
    setMetric("disk", big ? fmt(hw.disk_read_mb / 1024, 2) : fmt(hw.disk_read_mb, hw.disk_read_mb < 10 ? 1 : 0), big ? "GB/s" : "MB/s",
              hw.disk_write_mb == null ? "" : `write ${fmt(hw.disk_write_mb, 1)} MB/s`);
  }
  spark("sp-disk", h.disk_read_mb);

  const ctx = eng.max_context || 0;
  let used = 0;
  if (live.state !== "idle" && live.state !== "unloaded") used = (live.prompt_tokens || 0) + (live.generated || 0);
  else if (last) used = (last.prompt_tokens || 0) + (last.output_tokens || 0);
  const frac = ctx ? Math.min(1, used / ctx) : 0;
  $("ctx-fill").setAttribute("stroke-dasharray", `${(235.6 * frac).toFixed(1)} 314.2`);
  $("ctx-fill").style.opacity = 235.6 * frac >= 3 ? "1" : "0";
  $("ctx-pct").textContent = `${Math.round(frac * 100)}%`;
  $("ctx-sub").textContent = ctx ? `${kfmt(used)} / ${ctxfmt(ctx)}` : "–";
  const cacheBytes = (eng.expert_cache_mib || 0) * 1048576;
  $("slots-text").textContent = eng.expert_slots ? `${fmt(eng.expert_slots)} · ${gb(cacheBytes)} GB` : "–";
  $("slots-bar").style.width = hw.gpu_mem_total ? `${Math.min(100, (100 * cacheBytes) / hw.gpu_mem_total)}%` : "0%";
  $("ram-text").textContent = hw.ram_total ? `${gb(hw.ram_used)} / ${gb(hw.ram_total, 0)} GB` : "–";
  const ramPct = hw.ram_total ? (100 * hw.ram_used) / hw.ram_total : 0;
  $("ram-bar").style.width = `${ramPct}%`;
  if (ramPct > 92) $("ram-progress").dataset.tone = "danger"; else delete $("ram-progress").dataset.tone;

  const badge = {stop: ["", "Done"], length: ["", "Max tokens"], cancel: ["st-badge--queued", "Stopped"],
                 disconnect: ["st-badge--queued", "Closed"], error: ["st-badge--error", "Error"], timeout: ["st-badge--queued", "Timeout"]};
  $("req-body").innerHTML = !requests.length ? `<tr><td colspan="9" class="muted">No requests yet</td></tr>`
    : requests.slice(0, reqShowAll ? requests.length : 12).map((r) => {
      const [cls, text] = badge[r.finish] || ["", r.finish || "–"];
      const t = new Date(r.time * 1000).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit", second: "2-digit"});
      const hit = r.hit_rate == null ? "–" : `${(r.hit_rate * 100).toFixed(1)}%`;
      const fresh = r.prompt_tokens == null ? null : Math.max(0, r.prompt_tokens - (r.reused || 0));
      const rate = fresh > 0 && r.prompt_ms > 0 ? fresh / (r.prompt_ms / 1000) : null;
      return `<tr><td>${esc(t)}</td><td><span class="st-badge ${cls}">${esc(text)}</span></td><td class="num">${fmt(r.prompt_tokens)}</td>
        <td class="num">${fmt(r.reused)}</td><td class="num">${fmt(r.output_tokens)}</td><td class="num">${fmt(rate)}</td>
        <td class="num">${fmt(r.decode_tok_s, 1)}</td><td class="num">${hit}</td><td class="num">${fmt(r.duration_s, 1)} s</td></tr>`;
    }).join("");
  const kept = m.requests_kept == null ? requests.length : m.requests_kept;
  $("req-all").hidden = kept <= 12;
  $("req-all").textContent = reqShowAll ? "Show fewer" : `Show all (${kept})`;
  $("req-wrap").classList.toggle("all", reqShowAll);
  const t = m.totals;
  if (t && t.requests) {
    const read = t.prompt_tokens - t.reused;
    const ps = t.prompt_ms > 0 && read > 0 ? ` at ${fmt(read / (t.prompt_ms / 1000))} tok/s` : "";
    const os = t.decode_ms > 0 && t.output_tokens > 0 ? ` at ${fmt(t.output_tokens / (t.decode_ms / 1000), 1)} tok/s` : "";
    $("req-totals").textContent = `Since ${new Date(t.since * 1000).toLocaleString([], {weekday: "short", hour: "2-digit", minute: "2-digit"})}: ` +
      `${fmt(t.requests)} requests · ${fmt(read)} prompt tokens read${ps} (${fmt(t.reused)} reused) · ${fmt(t.output_tokens)} written${os}`;
  } else $("req-totals").textContent = "";
}
$("req-all").onclick = () => { reqShowAll = !reqShowAll; if (lastMetrics) render(lastMetrics); };

// ------------------------------------------------------------------ Monitor: the turn scheduler (serve/turns.py)
const who = (s) => (s == null ? `<span class="muted">unnamed</span>` : `<span class="mono">${esc(s)}</span>`);
const stat = (label, value) => `<div class="stat"><b>${esc(value)}</b><span>${esc(label)}</span></div>`;
function renderScheduler(t) {
  $("sched-card").hidden = !t;
  if (!t) return;
  const p = t.priority;
  let now;
  if (t.running) {
    const r = t.running;
    now = `${icon("bolt", "st-icon st-icon--sm")}<span>Running</span> ${who(r.session)}` +
      (r.model ? ` <span class="tag">${esc(r.model)}</span>` : "") +
      (r.priority != null ? ` <span class="tag">priority ${fmt(r.priority)}</span>` : "") +
      ` <span class="muted">for ${secs(r.held_s)}</span>` +
      (t.slice_s ? ` <span class="muted">· slice ${secs(t.used_s)} of ${secs(t.slice_s)}</span>` : "") +
      (r.preempting ? ` <span class="tag tag--warn">preempted</span>` : "");
  } else if (t.hold_left_s != null) {
    now = `${icon("clock", "st-icon st-icon--sm")}<span>Held for the follow-up of</span> ${who(t.lane)} <span class="muted">(${secs(t.hold_left_s)} left)</span>`;
  } else if (t.reserved) {
    now = `${icon("clock", "st-icon st-icon--sm")}<span>Reserved for</span> ${who(t.reserved_session)}` +
      (p && p.reserved_for != null ? ` <span class="tag">priority ${fmt(p.reserved_for)}</span>` : "");
  } else {
    now = `<span class="muted">The turn is free</span>` + (t.lane ? ` <span class="muted">· last:</span> ${who(t.lane)}` : "");
  }
  $("sched-now").innerHTML = now;
  const q = t.queue || [];
  $("sched-sum").textContent = q.length ? `${q.length} waiting` : "nobody waiting";
  $("sched-queue").innerHTML = q.length ? q.map((x, i) => `<tr><td class="num">${i + 1}</td><td class="session">${who(x.session)}</td>
      <td>${x.model ? esc(x.model) : `<span class="muted">any</span>`}</td><td class="num">${x.priority == null ? "–" : fmt(x.priority)}</td>
      <td class="num">${secs(x.waited_s)}</td><td>${x.continues ? `<span class="tag tag--accent">continues</span>` : ""}${x.one_shot ? `<span class="tag">one-shot</span>` : ""}${x.deferred ? `<span class="tag tag--warn">deferred</span>` : ""}</td></tr>`).join("")
    : `<tr class="empty"><td colspan="6">No request waiting</td></tr>`;
  const idle = (t.hold_idle_s || 0) + ((p && p.reserve_idle_s) || 0);
  $("sched-stats").innerHTML = [
    stat("switches / kept", `${fmt(t.switches)} / ${fmt(t.kept)}`),
    t.hold_s ? stat("holds hit / missed", `${fmt(t.hold_hits)} / ${fmt(t.hold_misses)}`) : "",
    idle ? stat("idle in holds, reservations", secs(idle)) : "",
    p ? stat("preemptions", fmt(p.preemptions)) : "",
    p ? stat("sessions seen", fmt(p.sessions)) : "",
  ].join("");
  const parts = [t.slice_s ? `slice ${secs(t.slice_s)}` : "no slice", t.max_wait_s ? `max wait ${secs(t.max_wait_s)}` : "no wait limit",
                 t.hold_s ? `hold ${secs(t.hold_s)}` : "no hold"];
  if (p) parts.push(`preempt ${p.preempt} (a band = ${p.delta}${p.preempt === "full" ? `, after ${secs(p.min_run_s)}` : p.preempt === "timeout" ? `, after ${secs(p.preempt_timeout_s)}` : ""})`,
                    p.reserve_s ? `reserve ${secs(p.reserve_s)}` : "no reservations", p.aging_s ? `aging a step per ${secs(p.aging_s)}` : "no aging");
  $("sched-policy").textContent = parts.join(" · ");
}

// ------------------------------------------------------------------ Monitor: the model variants (serve/models.py)
function renderModels(mv, t, live) {
  $("models-card").hidden = !mv;
  if (!mv) return;
  const waiting = (t && t.models && t.models.waiting) || {};
  const names = Object.keys(mv.variants || {});
  $("models-sum").textContent = `${names.length} variants · ${fmt(mv.switches)} switches${mv.switches ? `, ${secs(mv.switch_s / mv.switches)} each` : ""}`;
  const busy = live.state === "reading" || live.state === "generating" || live.queued > 0;
  $("variant-list").innerHTML = names.map((n) => {
    const active = n === mv.active;
    const aliases = (mv.aliases || {})[n] || [];
    const s = (mv.sampling || {})[n] || {};
    const samp = ["temperature", "top_p", "top_k", "min_p", "presence_penalty"].filter((k) => s[k] != null)
      .map((k) => `${k.replace("presence_penalty", "presence")} ${s[k]}`).join(", ");
    const tags = [active ? `<span class="tag tag--accent">${live.state === "unloaded" ? "selected (unloaded)" : "loaded"}</span>` : "",
                  mv.committed === n ? `<span class="tag tag--warn">switching next</span>` : "",
                  waiting[n] ? `<span class="tag tag--info">${waiting[n]} waiting</span>` : "",
                  mv.default === n ? `<span class="tag">default</span>` : "",
                  (mv.saved || []).includes(n) ? `<span class="tag">conversation saved</span>` : ""].join("");
    return `<div class="variant${active ? " on" : ""}"><div class="variant__name"><strong>${esc(n)}</strong>${tags}</div>
      ${active ? `<span class="muted small">for ${secs(mv.resident_s)}</span>`
               : `<button class="st-btn st-btn--secondary" data-load-variant="${esc(n)}"${busy ? " disabled title=\"A request is running\"" : ""}>Load</button>`}
      <div class="variant__meta"><span class="mono">${esc(mv.variants[n])}</span>${aliases.length ? ` · also ${esc(aliases.join(", "))}` : ""}${samp ? ` · ${esc(samp)}` : ""}</div></div>`;
  }).join("");
  const pol = [`a request for another variant switches after ${secs(mv.max_wait_s)}`, `a variant stays at least ${secs(mv.min_resident_s)}`,
               mv.preempt ? "a switch may stop a running request" : "a switch waits for the running request",
               mv.default ? `back to ${mv.default} after ${secs(mv.default_idle_s)} idle` : "no default variant",
               mv.last_switch_s != null ? `last switch ${secs(mv.last_switch_s)}` : ""];
  $("models-policy").textContent = pol.filter(Boolean).join(" · ");
}
$("variant-list").addEventListener("click", async (e) => {
  const b = e.target.closest("[data-load-variant]");
  if (!b) return;
  const name = b.dataset.loadVariant;
  if (!confirm(`Switch the engine to ${name}? The loaded variant's conversation is parked; the switch takes a few seconds.`)) return;
  b.disabled = true; b.textContent = "Loading…";
  try { await api("v1/load", {model: name}); toast("success", `${name} loaded`); }
  catch (err) { toast("error", `${name} not loaded`, err.message, 6000); }
});

// ------------------------------------------------------------------ Monitor: the conversation cache
let ccHover = null;
function colorOf(f) {
  const key = f.session || `${f.variant}/${f.file}`;
  let h = 0;
  for (const c of key) h = (h * 31 + c.charCodeAt(0)) >>> 0;
  return `var(--cc-${h % 8})`;
}
function renderCache(c) {
  $("cc-card").hidden = !c;
  if (!c) return;
  const d = c.disk || {}, files = c.files;
  const budget = (c.budget_mib || 0) * 1048576;
  $("cc-disk").hidden = !files;
  if (files) {
    const list = files.files || [];
    $("cc-sum").textContent = `${fmt(files.total_files)} parked · ${size(files.total_bytes)}${budget ? ` of ${size(budget)}` : ""}`;
    $("cc-mem-text").textContent = budget ? `${fmt((100 * files.total_bytes) / budget)}% of the budget` : "";
    $("cc-bar-label").textContent = files.group ? "Parked files, every variant" : "Parked files";
    $("cc-bar-label").title = files.group || files.dir;
    const total = Math.max(budget, files.total_bytes, 1);
    // the segments: grouped by variant (the loaded one first), oldest first within each, so the bar reads like a disk
    const ordered = [...list].sort((a, b) => (b.own - a.own) || String(a.variant).localeCompare(String(b.variant)) || a.at - b.at);
    $("cc-seg").innerHTML = ordered.map((f) => `<span style="--c:${colorOf(f)};width:${((100 * f.bytes) / total).toFixed(3)}%"
      class="${f.obsolete ? "obsolete" : ""}${ccHover === f.file + "/" + f.variant ? " hl" : ""}" data-f="${f.file}/${esc(f.variant)}"
      title="${esc(`${f.name || f.session || "unnamed"} · ${f.variant || ""} · ${size(f.bytes)} · parked ${ago(f.at)}${f.obsolete ? " · obsolete" : ""}`)}"></span>`).join("");
    $("cc-files").innerHTML = list.length ? list.map((f) => `<tr data-f="${f.file}/${esc(f.variant)}" class="${f.obsolete ? "obsolete" : ""}${ccHover === f.file + "/" + f.variant ? " hl" : ""}">
      <td><span class="swatch" style="--c:${colorOf(f)}"></span></td>
      <td class="session">${f.name ? `<span class="mono">${esc(f.name)}</span>` : f.session ? `<span class="mono muted" title="a session from before the names were kept">#${esc(f.session)}</span>` : `<span class="muted">no session</span>`}</td>
      <td>${f.variant ? (f.own ? `<strong title="the loaded variant">${esc(f.variant)}</strong>` : esc(f.variant)) : "–"}</td>
      <td class="num">${size(f.bytes)}</td><td class="num">${ago(f.at)}</td>
      <td><span class="mono muted">park-${f.file}</span>${f.obsolete ? ` <span class="tag tag--warn">obsolete</span>` : ""}</td></tr>`).join("")
      : `<tr><td colspan="6" class="muted">Nothing parked yet</td></tr>`;
  } else {
    $("cc-sum").textContent = c.enabled ? `${fmt(c.parked)} parked · ${gb(c.bytes)} / ${gb(budget)} GB` : "";
  }
  const share = c.prompt_tokens ? ` (${fmt((100 * c.reused_tokens) / c.prompt_tokens)}%)` : "";
  $("cc-stats").innerHTML = [
    c.last_prompt != null ? stat("reused, last request", `${kfmt(c.last_reused || 0)} / ${kfmt(c.last_prompt)}`) : "",
    c.requests ? stat("tokens reused since start", `${kfmt(c.reused_tokens)}${share}`) : "",
    c.enabled ? stat("parked / restored", `${fmt(c.parks)} / ${fmt(c.restores)}`) : "",
    c.evictions ? stat("evicted", fmt(c.evictions)) : "",
    d.parks ? stat("written (in place)", `${gb(d.written_bytes)} GB (${fmt(d.in_place)}/${fmt(d.parks)})`) : "",
    d.reads ? stat("read back", `${fmt(d.reads)} · ${fmt(d.read_ms / d.reads)} ms`) : "",
    d.borrows ? stat("borrowed", fmt(d.borrows)) : "",
    c.skipped_tail ? stat("tail skips", fmt(c.skipped_tail)) : "",
    d.failures ? stat("disk failures", fmt(d.failures)) : "",
  ].join("");
}
for (const id of ["cc-seg", "cc-files"]) {
  $(id).addEventListener("mouseover", (e) => {
    const el = e.target.closest("[data-f]");
    const f = el ? el.dataset.f : null;
    if (f === ccHover) return;
    ccHover = f;
    for (const x of document.querySelectorAll("#cc-card [data-f]")) x.classList.toggle("hl", x.dataset.f === f);
  });
  $(id).addEventListener("mouseleave", () => { ccHover = null; for (const x of document.querySelectorAll("#cc-card .hl")) x.classList.remove("hl"); });
}

// ------------------------------------------------------------------ About
function facts(el, rows) {
  el.innerHTML = rows.filter((r) => r[1] != null && r[1] !== "").map(([k, v, copy]) =>
    `<dt>${esc(k)}</dt><dd>${copy ? `<code>${esc(v)}</code><button class="st-btn st-btn--icon" data-copy="${esc(v)}" aria-label="Copy">${icon("copy")}</button>` : esc(v)}</dd>`).join("");
}
function renderAbout(eng, hw, st) {
  const kv = {int8: "8-bit", q4_0: "4-bit (Hadamard-rotated)", fp16: "16-bit"}[eng.kv] || eng.kv;
  const mv = lastMetrics && lastMetrics.models;
  facts($("facts-engine"), [
    ["Model", eng.model],
    ["Variant", mv ? `${mv.active} (of ${Object.keys(mv.variants || {}).join(", ")})` : null],
    ["Engine", eng.version ? `v${eng.version}` : "built from source"],
    ["Context", eng.max_context ? `${fmt(eng.max_context)} tokens` : null],
    ["KV cache", kv ? `${kv}${eng.kv_resident ? `, streamed: ${fmt(eng.kv_resident)} positions per layer in VRAM` : ", all in VRAM"}` : null],
    ["Experts in VRAM", eng.expert_slots ? `${fmt(eng.expert_slots)} (${gb((eng.expert_cache_mib || 0) * 1048576)} GB)` : null],
    ["Speculation", eng.spec ? `MTP drafts up to ${Math.max(0, (eng.mtp_max || eng.spec) - 1)} tokens` : null],
    ["Images", eng.images ? "on" : "off"],
  ]);
  facts($("facts-hw"), [
    ["GPU", st.gpu_name ? `${st.gpu_name}${hw.gpu_mem_total ? `, ${gb(hw.gpu_mem_total, 0)} GB` : ""}` : (st.gpu_note || "not readable")],
    ["CPU", st.cpu_name ? `${st.cpu_name}${st.threads ? `, ${st.threads} threads` : ""}` : null],
    ["RAM", hw.ram_total ? `${gb(hw.ram_total, 0)} GB` : null],
  ]);
  facts($("facts-api"), [
    ["OpenAI base URL", `${location.origin}/v1`, true],
    ["Anthropic base URL", location.origin, true],
    ["Model name", eng.model, true],
  ]);
}

// ------------------------------------------------------------------ start
showTab(location.hash.slice(1) || "monitor");
poll();
requestAnimationFrame(fitHeader);
