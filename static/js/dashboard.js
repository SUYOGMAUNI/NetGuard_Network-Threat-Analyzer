/* =============================================================
   dashboard.js — NetGuard v2.0 frontend logic
   - 7 threat classes matching CIC-IDS-2017
   - Flow-level packet feed (fwd/bwd pkts + bytes columns)
   - Dynamic ML stats loaded from /api/model-info
   - Dynamic feature importances from /api/feature-importances
   - Blocked IP panel synced from /api/blocked-ips
   - WebSocket capture_status event handling
   - All state guarded; no hardcoded model metrics
   ============================================================= */

const socket = io();

// ── State ──────────────────────────────────────────────────
let flowCount    = 0;
let threats      = 0;
let alertCount   = 0;
let ppsCounter   = 0;
let captureActive = false;

// All 7 threat classes from CIC-IDS-2017
const THREAT_CLASSES = ['PORT_SCAN', 'DDOS', 'BRUTE_FORCE', 'SQL_INJECT', 'WEB_ATTACK', 'BOTNET'];

// Protocol buckets (now flow-level, no ICMP packet-level)
const protoCounts = { TCP:0, UDP:0, HTTP:0, HTTPS:0, DNS:0, OTHER:0 };

// Threat counts — one entry per attack class
const threatCounts = {};
THREAT_CLASSES.forEach(c => threatCounts[c] = 0);

const MAX_FEED_ROWS    = 60;
const MAX_TRAFFIC_PTS  = 60;
const AUTO_BLOCK_CONF  = 85;   // confidence threshold for auto-blocking (%)

// ── Traffic chart ─────────────────────────────────────────
const trafficLabels = Array.from({length: MAX_TRAFFIC_PTS}, (_, i) => `-${MAX_TRAFFIC_PTS - i}s`);
const trafficFlows  = new Array(MAX_TRAFFIC_PTS).fill(0);
const trafficThreats= new Array(MAX_TRAFFIC_PTS).fill(0);

const tCtx = document.getElementById('trafficChart').getContext('2d');
const trafficChart = new Chart(tCtx, {
  type: 'line',
  data: {
    labels: trafficLabels,
    datasets: [
      {
        label: 'Flows/s',
        data: trafficFlows,
        borderColor: '#d4af37',
        backgroundColor: 'rgba(212,175,55,0.07)',
        borderWidth: 1.5,
        pointRadius: 0,
        fill: true,
        tension: 0.4,
      },
      {
        label: 'Threats/s',
        data: trafficThreats,
        borderColor: '#ff4444',
        backgroundColor: 'rgba(255,68,68,0.06)',
        borderWidth: 1,
        pointRadius: 0,
        fill: true,
        tension: 0.4,
      },
    ],
  },
  options: {
    responsive: true,
    animation: { duration: 300 },
    plugins: {
      legend: { display: true, labels: { color:'#888', font:{ family:'Share Tech Mono', size:10 }, boxWidth:12 } },
    },
    scales: {
      x: { display: false },
      y: {
        min: 0,
        grid: { color: 'rgba(255,255,255,0.04)' },
        ticks: { color: '#666', font: { family: 'Share Tech Mono', size: 10 } },
      },
    },
  },
});

// ── Protocol doughnut ─────────────────────────────────────
const PROTO_COLORS = ['#d4af37', '#00d4ff', '#00ff88', '#ff8844', '#ff4444', '#aa88ff'];
const pCtx = document.getElementById('protoChart').getContext('2d');
const protoChart = new Chart(pCtx, {
  type: 'doughnut',
  data: {
    labels: Object.keys(protoCounts),
    datasets: [{
      data: Object.values(protoCounts),
      backgroundColor: PROTO_COLORS,
      borderColor: '#111',
      borderWidth: 2,
      hoverOffset: 6,
    }],
  },
  options: {
    responsive: true,
    cutout: '68%',
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: { label: ctx => ` ${ctx.label}: ${ctx.parsed} flows` },
      },
    },
  },
});

function updateProtoChart() {
  const vals  = Object.values(protoCounts);
  const total = vals.reduce((a, b) => a + b, 0) || 1;

  protoChart.data.datasets[0].data = vals;
  protoChart.update('none');

  document.getElementById('protoLegend').innerHTML =
    Object.entries(protoCounts).map(([k, v], i) => `
      <div class="proto-row">
        <span class="proto-dot" style="background:${PROTO_COLORS[i]}"></span>
        <span class="proto-name">${k}</span>
        <span class="proto-pct">${Math.round(v / total * 100)}%</span>
      </div>
    `).join('');
}

// ── Gauge ──────────────────────────────────────────────────
function updateGauge(score) {
  const angle  = -90 + (score / 100) * 180;
  const offset = 251 - (score / 100) * 251;

  document.getElementById('gaugeNeedle').setAttribute('transform', `rotate(${angle} 100 105)`);
  document.getElementById('gaugeFill').style.strokeDashoffset = offset;
  document.getElementById('gaugeScore').textContent = score;

  const lbl = document.getElementById('gaugeLbl');
  if (score < 30)       { lbl.textContent = 'LOW RISK';    lbl.style.color = '#00ff88'; }
  else if (score < 65)  { lbl.textContent = 'MEDIUM RISK'; lbl.style.color = '#ffcc00'; }
  else                  { lbl.textContent = 'HIGH RISK';   lbl.style.color = '#ff4444'; }
}

// ── Threat breakdown bars (all 6 attack classes) ──────────
const THREAT_BAR_META = [
  { label: 'Port Scan',   key: 'PORT_SCAN',   color: '#ffcc00' },
  { label: 'DDoS',        key: 'DDOS',        color: '#ff4444' },
  { label: 'Brute Force', key: 'BRUTE_FORCE', color: '#ff8844' },
  { label: 'SQL Inject',  key: 'SQL_INJECT',  color: '#ff4444' },
  { label: 'Web Attack',  key: 'WEB_ATTACK',  color: '#00d4ff' },
  { label: 'Botnet',      key: 'BOTNET',      color: '#aa88ff' },
];

function updateThreatBars() {
  const total = Object.values(threatCounts).reduce((a, b) => a + b, 0) || 1;
  document.getElementById('threatBars').innerHTML = THREAT_BAR_META.map(b => {
    const pct = Math.round(threatCounts[b.key] / total * 100);
    return `<div class="t-bar-row">
      <span class="t-bar-lbl">${b.label}</span>
      <div class="t-bar-track">
        <div class="t-bar-fill" style="width:${pct}%;background:${b.color}"></div>
      </div>
      <span class="t-bar-val" style="color:${b.color}">${threatCounts[b.key]}</span>
    </div>`;
  }).join('');
}

// ── Badge map — all 7 labels ──────────────────────────────
const BADGE_MAP = {
  NORMAL:      '<span class="badge b-safe">&#10003; CLEAN</span>',
  PORT_SCAN:   '<span class="badge b-scan">&#9888; PORT SCAN</span>',
  DDOS:        '<span class="badge b-ddos">&#9888; DDoS</span>',
  BRUTE_FORCE: '<span class="badge b-brute">&#9888; BRUTE FORCE</span>',
  SQL_INJECT:  '<span class="badge b-sqli">&#9888; SQL INJECT</span>',
  WEB_ATTACK:  '<span class="badge b-web">&#9888; WEB ATTACK</span>',
  BOTNET:      '<span class="badge b-bot">&#9888; BOTNET</span>',
};

// ── Flow feed ─────────────────────────────────────────────
function addFlowRow(pkt) {
  const tbody    = document.getElementById('packetFeed');
  const isThreat = pkt.threat !== 'NORMAL';
  const row      = document.createElement('tr');
  if (isThreat) row.classList.add('threat-row');

  // Format byte values
  const fmtBytes = n => n >= 1024 ? `${(n/1024).toFixed(1)}K` : `${n}B`;

  row.innerHTML = `
    <td>${pkt.timestamp}</td>
    <td>${pkt.src_ip}</td>
    <td>${pkt.dst_ip}</td>
    <td>${pkt.dst_port}</td>
    <td>${pkt.protocol}</td>
    <td>${pkt.fwd_pkts ?? '—'}</td>
    <td>${pkt.bwd_pkts ?? '—'}</td>
    <td>${fmtBytes(pkt.fwd_bytes ?? 0)}</td>
    <td>${BADGE_MAP[pkt.threat] || BADGE_MAP.NORMAL}</td>
    <td style="color:${isThreat ? '#ff4444' : '#00ff88'}">${pkt.confidence}%</td>
  `;

  tbody.insertBefore(row, tbody.firstChild);
  if (tbody.rows.length > MAX_FEED_ROWS) tbody.deleteRow(tbody.rows.length - 1);

  flowCount++;
  document.getElementById('pktCount').textContent = `${flowCount} flows`;
}

function clearFeed() {
  document.getElementById('packetFeed').innerHTML = '';
  flowCount = 0;
  document.getElementById('pktCount').textContent = '0 flows';
}

// ── Alerts ────────────────────────────────────────────────
function addAlert(alert) {
  const list  = document.getElementById('alertList');
  const empty = list.querySelector('.alert-empty');
  if (empty) empty.remove();

  alertCount++;
  document.getElementById('alertCount').textContent = alertCount;

  const icon = alert.type === 'critical' ? '🔴' : '🟡';
  const el   = document.createElement('div');
  el.className = `alert-item ${alert.type}`;
  el.innerHTML = `
    <span class="alert-icon">${icon}</span>
    <div class="alert-body">
      <div class="alert-title">${alert.title}</div>
      <div class="alert-desc">
        ${alert.src_ip} &#8594; ${alert.dst_ip}:${alert.port}
        &nbsp;|&nbsp; Confidence: <strong>${alert.confidence}%</strong>
      </div>
    </div>
    <div class="alert-actions">
      <button class="btn-block" onclick="blockIP('${alert.src_ip}')">Block IP</button>
      <span class="alert-time">${alert.time}</span>
    </div>
  `;
  list.insertBefore(el, list.firstChild);
  if (list.children.length > 30) list.removeChild(list.lastChild);
}

function clearAlerts() {
  document.getElementById('alertList').innerHTML =
    '<div class="alert-empty">No alerts yet. Start capture to begin monitoring.</div>';
  alertCount = 0;
  document.getElementById('alertCount').textContent = '0';
}

// ── Blocked IP management ─────────────────────────────────
async function blockIP(ip) {
  try {
    await fetch('/api/block-ip', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ip }),
    });
    await refreshBlockedIPs();
  } catch (e) {
    console.error('Block failed:', e);
  }
}

async function unblockIP(ip) {
  try {
    await fetch('/api/unblock-ip', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ip }),
    });
    await refreshBlockedIPs();
  } catch (e) {
    console.error('Unblock failed:', e);
  }
}

async function refreshBlockedIPs() {
  try {
    const res  = await fetch('/api/blocked-ips');
    const data = await res.json();
    const panel = document.getElementById('blockedPanel');
    const list  = document.getElementById('blockedList');
    const count = document.getElementById('blockedCount');

    document.getElementById('statBlocked').textContent = data.count;
    count.textContent = data.count;

    if (data.count === 0) {
      panel.style.display = 'none';
      return;
    }
    panel.style.display = '';
    list.innerHTML = data.blocked_ips.map(ip => `
      <div class="blocked-row">
        <span class="blocked-ip">${ip}</span>
        <button class="btn-unblock" onclick="unblockIP('${ip}')">Unblock</button>
      </div>
    `).join('');
  } catch (e) {
    console.error('Could not fetch blocked IPs:', e);
  }
}

// ── ML model info — loaded dynamically from API ───────────
async function loadModelInfo() {
  try {
    const res  = await fetch('/api/model-info');
    const info = await res.json();

    // Update stat card
    if (info.accuracy !== null && info.accuracy !== undefined) {
      document.getElementById('statAccuracy').textContent = `${info.accuracy}%`;
      document.getElementById('statAccuracySub').textContent = info.algorithm || 'Ensemble';
    }

    // Build ML grid dynamically
    const fields = [
      { label: 'Algorithm',   val: info.algorithm,                           cls: 'gold', small: true },
      { label: 'Features',    val: info.features,                            cls: ''               },
      { label: 'Classes',     val: info.n_classes,                           cls: ''               },
      { label: 'Accuracy',    val: info.accuracy    != null ? `${info.accuracy}%`    : '—', cls: 'good' },
      { label: 'Precision',   val: info.precision_macro != null ? `${info.precision_macro}%` : '—', cls: 'good' },
      { label: 'Recall',      val: info.recall_macro != null ? `${info.recall_macro}%` : '—', cls: 'good' },
      { label: 'F1 (macro)',  val: info.f1_macro    != null ? info.f1_macro  : '—', cls: 'gold' },
      { label: 'Dataset',     val: info.dataset,                             cls: '',  small: true },
      { label: 'Latency',     val: info.inference_latency_ms != null ? `${info.inference_latency_ms}ms` : '—', cls: 'good' },
      { label: 'Train Samples', val: info.training_samples != null ? info.training_samples.toLocaleString() : '—', cls: '' },
    ];

    document.getElementById('mlGrid').innerHTML = fields.map(f => `
      <div class="ml-card">
        <div class="ml-card-lbl">${f.label}</div>
        <div class="ml-card-val ${f.cls}" ${f.small ? 'style="font-size:0.72rem"' : ''}>${f.val ?? '—'}</div>
      </div>
    `).join('');

  } catch (e) {
    console.error('Could not load model info:', e);
  }
}

// ── Feature importances ───────────────────────────────────
async function loadFeatureImportances() {
  try {
    const res  = await fetch('/api/feature-importances');
    if (!res.ok) {
      document.getElementById('featureImportances').innerHTML =
        '<div class="alert-empty">Feature importances available after training with real dataset.</div>';
      return;
    }
    const data  = await res.json();
    const pairs = Object.entries(data);   // already sorted desc by app.py
    const top   = pairs.slice(0, 12);    // show top 12
    const max   = top[0]?.[1] || 1;

    document.getElementById('featureImportances').innerHTML = top.map(([name, imp]) => {
      const pct = Math.round((imp / max) * 100);
      return `<div class="fi-row">
        <span class="fi-name">${name}</span>
        <div class="fi-track">
          <div class="fi-fill" style="width:${pct}%"></div>
        </div>
        <span class="fi-val">${(imp * 100).toFixed(2)}%</span>
      </div>`;
    }).join('');
  } catch (e) {
    document.getElementById('featureImportances').innerHTML =
      '<div class="alert-empty">Feature importances not available.</div>';
  }
}

// ── 1-second ticker ───────────────────────────────────────
let threatThisSec = 0;

setInterval(() => {
  const fps = ppsCounter;
  ppsCounter = 0;

  document.getElementById('statPPS').textContent = fps;
  if (fps > 0) {
    document.getElementById('statPPSSub').textContent =
      captureActive ? 'Live capture running' : 'Simulation running';
  }

  // Push to charts
  trafficFlows.shift();   trafficFlows.push(fps);
  trafficThreats.shift(); trafficThreats.push(threatThisSec);
  threatThisSec = 0;

  trafficChart.data.datasets[0].data = [...trafficFlows];
  trafficChart.data.datasets[1].data = [...trafficThreats];
  trafficChart.update('none');

  // Gauge: threat ratio as % × 10, capped at 100
  const ratio = (threats / Math.max(flowCount, 1)) * 1000;
  updateGauge(Math.min(100, Math.round(ratio)));

  updateThreatBars();
  updateProtoChart();
}, 1000);

// Refresh blocked IPs every 5s
setInterval(refreshBlockedIPs, 5000);

// ── Socket events ─────────────────────────────────────────
socket.on('connect', () => {
  document.getElementById('statusDot').classList.add('active');
  document.getElementById('statusDot').classList.remove('error');
  document.getElementById('statusLabel').textContent = 'CONNECTED';
  document.getElementById('statusLabel').classList.add('active');
});

socket.on('disconnect', () => {
  document.getElementById('statusDot').classList.remove('active');
  document.getElementById('statusDot').classList.add('error');
  document.getElementById('statusLabel').textContent = 'DISCONNECTED';
  document.getElementById('statusLabel').classList.remove('active');
});

socket.on('capture_status', data => {
  captureActive = data.active;
  if (data.active) {
    document.getElementById('btnStart').classList.add('active');
    document.getElementById('btnStop').classList.remove('active');
  } else {
    document.getElementById('btnStop').classList.add('active');
    document.getElementById('btnStart').classList.remove('active');
  }
});

socket.on('new_packet', pkt => {
  ppsCounter++;

  // Protocol tracking
  const proto = protoCounts.hasOwnProperty(pkt.protocol) ? pkt.protocol : 'OTHER';
  protoCounts[proto]++;

  // Threat tracking
  if (pkt.threat !== 'NORMAL') {
    threats++;
    threatThisSec++;
    if (threatCounts.hasOwnProperty(pkt.threat)) threatCounts[pkt.threat]++;
    document.getElementById('statThreats').textContent = threats;

    // Auto-block high-confidence threats via API
    if (pkt.confidence >= AUTO_BLOCK_CONF) {
      blockIP(pkt.src_ip);   // async, fire-and-forget
    }
  }

  addFlowRow(pkt);
});

socket.on('new_alert', alert => addAlert(alert));

socket.on('error', data => {
  console.error('[NetGuard error]', data.message);
});

socket.on('status', data => {
  console.log('[NetGuard]', data.message, '| version:', data.version);
});

// ── Capture controls ──────────────────────────────────────
function startCapture() {
  fetch('/api/start', { method: 'POST' })
    .then(r => r.json())
    .then(data => {
      if (data.status === 'started' || data.status === 'already_running') {
        document.getElementById('btnStart').classList.add('active');
        document.getElementById('btnStop').classList.remove('active');
        captureActive = true;
      }
    })
    .catch(e => console.error('Start failed:', e));
}

function stopCapture() {
  fetch('/api/stop', { method: 'POST' })
    .then(r => r.json())
    .then(() => {
      document.getElementById('btnStop').classList.add('active');
      document.getElementById('btnStart').classList.remove('active');
      captureActive = false;
    })
    .catch(e => console.error('Stop failed:', e));
}

// ── Init ──────────────────────────────────────────────────
(async function init() {
  await loadModelInfo();
  await loadFeatureImportances();
  await refreshBlockedIPs();
  startCapture();
})();