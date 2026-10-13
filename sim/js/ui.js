import { PAL, PHASE_LABEL, SCORE_LEGEND, ALT_LEGEND, thresholdAt } from './colors.js';

const $ = (sel, root = document) => root.querySelector(sel);
const FT = 0.3048;
const KT = 1852 / 3600;
const NM = 1.852;

export function clock(t) {
  const s = Math.max(0, Math.floor(t));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
}
const hhmm = (t) => clock(t).slice(0, 5);
const fmt = (x, d = 1) => (Number.isFinite(x) ? x.toFixed(d) : '—');
const signed = (x, d = 0) => (Number.isFinite(x) ? `${x >= 0 ? '+' : '−'}${Math.abs(x).toFixed(d)}` : '—');

const COMPONENT_LABEL = {
  cross_track: 'cross-track', track_error: 'track angle', altitude: 'altitude',
  vertical_rate: 'vertical rate', ground_speed: 'ground speed', eta: 'ETA',
};
const KIND_UNIT = { lateral: '°', vertical: ' ft/min', longitudinal: ' kt' };

function setupCanvas(canvas) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const g = canvas.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  return { g, w, h };
}

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/**
 * A time series panel: line, a per-row threshold as a step line, shaded alert spans, presented
 * alert ticks, optional event lines and a playhead. Click or drag to set the time.
 */
class SeriesChart {
  constructor(canvas, onSeek) {
    this.canvas = canvas;
    this.spec = null;
    this.onSeek = onSeek;
    let dragging = false;
    const seek = (e) => {
      if (!this.spec) return;
      const r = canvas.getBoundingClientRect();
      const f = (e.clientX - r.left - this.pad.l) / (r.width - this.pad.l - this.pad.r);
      this.onSeek(this.spec.t0 + Math.min(1, Math.max(0, f)) * (this.spec.t1 - this.spec.t0));
    };
    canvas.addEventListener('pointerdown', (e) => { dragging = true; canvas.setPointerCapture(e.pointerId); seek(e); });
    canvas.addEventListener('pointermove', (e) => { if (dragging) seek(e); });
    canvas.addEventListener('pointerup', () => { dragging = false; });
    this.pad = { l: 34, r: 8, t: 8, b: 18 };
  }

  set(spec) {
    this.spec = spec;
  }

  draw(T) {
    const { g, w, h } = setupCanvas(this.canvas);
    const s = this.spec;
    if (!s) return;
    const { l, r, t, b } = this.pad;
    const pw = w - l - r;
    const ph = h - t - b;
    const X = (x) => l + ((x - s.t0) / (s.t1 - s.t0)) * pw;
    const Y = (y) => t + ph - (Math.min(y, s.yMax) / s.yMax) * ph;
    g.font = '10px ui-sans-serif, system-ui, sans-serif';
    // Grid and axis labels.
    g.strokeStyle = css('--grid');
    g.fillStyle = css('--text-3');
    g.lineWidth = 1;
    for (const v of s.ticks) {
      g.beginPath(); g.moveTo(l, Y(v) + 0.5); g.lineTo(l + pw, Y(v) + 0.5); g.stroke();
      g.textAlign = 'right'; g.textBaseline = 'middle'; g.fillText(s.tickFmt ? s.tickFmt(v) : v, l - 5, Y(v));
    }
    g.textAlign = 'left'; g.textBaseline = 'top';
    g.fillText(hhmm(s.t0), l, t + ph + 4);
    g.textAlign = 'right';
    g.fillText(hhmm(s.t1), l + pw, t + ph + 4);
    // Alert spans.
    g.fillStyle = s.alertFill;
    let open = -1;
    for (let i = 0; i <= s.t.length; i++) {
      const on = i < s.t.length && s.alert[i] > 0;
      if (on && open < 0) open = i;
      if (!on && open >= 0) {
        g.fillRect(X(s.t[open]), t, Math.max(1, X(s.t[i - 1]) - X(s.t[open])), ph);
        open = -1;
      }
    }
    // Event lines.
    for (const ev of s.events || []) {
      g.strokeStyle = ev.color; g.setLineDash([3, 3]);
      g.beginPath(); g.moveTo(X(ev.t) + 0.5, t); g.lineTo(X(ev.t) + 0.5, t + ph); g.stroke();
      g.setLineDash([]);
    }
    // Threshold (step per row's phase).
    g.strokeStyle = s.thrColor; g.lineWidth = 1.2; g.setLineDash([4, 3]);
    g.beginPath();
    for (let i = 0; i < s.t.length; i++) {
      const x = X(s.t[i]);
      const y = Y(s.thr[i]);
      if (i === 0) g.moveTo(x, y); else { g.lineTo(x, Y(s.thr[i - 1])); g.lineTo(x, y); }
    }
    g.stroke(); g.setLineDash([]);
    // Series.
    g.strokeStyle = s.color; g.lineWidth = 1.5;
    g.beginPath();
    let pen = false;
    for (let i = 0; i < s.t.length; i++) {
      const v = s.v[i];
      if (!Number.isFinite(v) || (i > 0 && s.t[i] - s.t[i - 1] > 120)) { pen = false; if (!Number.isFinite(v)) continue; }
      const x = X(s.t[i]);
      const y = Y(v);
      if (!pen) { g.moveTo(x, y); pen = true; } else g.lineTo(x, y);
    }
    g.stroke();
    // Presented alerts.
    g.fillStyle = s.alertColor;
    for (let i = 0; i < s.t.length; i++) {
      if (!s.presented[i]) continue;
      const x = X(s.t[i]);
      g.beginPath(); g.moveTo(x, t); g.lineTo(x - 4, t - 0.5 + 7); g.lineTo(x + 4, t - 0.5 + 7); g.closePath(); g.fill();
    }
    // Playhead.
    if (T >= s.t0 && T <= s.t1) {
      g.strokeStyle = css('--text-1'); g.lineWidth = 1;
      g.beginPath(); g.moveTo(X(T) + 0.5, t); g.lineTo(X(T) + 0.5, t + ph); g.stroke();
    }
    g.fillStyle = css('--text-3'); g.textAlign = 'left'; g.textBaseline = 'top';
    g.fillText(s.title, l + 4, t + 2);
  }
}

export class UI {
  constructor(data, world, state, actions) {
    this.d = data;
    this.world = world;
    this.state = state;
    this.actions = actions;
    this._lastPanel = 0;
    this._feedKey = '';
    this._buildTopbar();
    this._buildLeft();
    this._buildRight();
    this._buildTimeline();
    this._buildLegend();
    this._bindViewport();
    this._bindKeys();
  }

  // ---------- top bar & timeline ----------

  _buildTopbar() {
    const st = this.state;
    $('#day').textContent = `${this.d.meta.day} · test day · UTC`;
    $('#play').addEventListener('click', () => this.actions.togglePlay());
    $('#speed').addEventListener('change', (e) => { st.speed = Number(e.target.value); });
    $('#speed').value = String(st.speed);
    $('#back').addEventListener('click', () => this.actions.seek(st.T - 60));
    $('#fwd').addEventListener('click', () => this.actions.seek(st.T + 60));
    $('#colormode').addEventListener('change', (e) => { st.colorMode = e.target.value; this.actions.restyle(); this._buildLegend(); });
    $('#home').addEventListener('click', () => this.world.home());
  }

  _buildTimeline() {
    const slider = $('#time');
    slider.min = String(this.d.tMin);
    slider.max = String(this.d.tMax);
    slider.step = '1';
    slider.addEventListener('input', () => this.actions.seek(Number(slider.value)));
    this._drawHistogram();
    window.addEventListener('resize', () => this._drawHistogram());
  }

  _drawHistogram() {
    const { g, w, h } = setupCanvas($('#hist'));
    const { airborne, imm, fixed } = this.d.bins;
    const n = airborne.length;
    const maxA = Math.max(...airborne);
    const bw = w / n;
    g.fillStyle = css('--hist');
    for (let i = 0; i < n; i++) {
      const y = (airborne[i] / maxA) * (h - 2);
      g.fillRect(i * bw, h - y, Math.ceil(bw), y);
    }
    // Alerts presented per 10 minutes, as ticks above the traffic.
    const agg = (arr) => {
      const out = [];
      for (let i = 0; i < n; i += 10) out.push(arr.slice(i, i + 10).reduce((a, b) => a + b, 0));
      return out;
    };
    const ai = agg(imm);
    const af = agg(fixed);
    const maxAl = Math.max(1, ...ai, ...(this.state.showFixed ? af : []));
    const sw = w / ai.length;
    for (let i = 0; i < ai.length; i++) {
      g.fillStyle = PAL.imm;
      const yi = (ai[i] / maxAl) * (h - 4);
      g.globalAlpha = 0.85;
      g.fillRect(i * sw + sw * 0.18, h - yi, sw * 0.3, yi);
      if (this.state.showFixed) {
        g.fillStyle = PAL.fixed;
        const yf = (af[i] / maxAl) * (h - 4);
        g.fillRect(i * sw + sw * 0.52, h - yf, sw * 0.3, yf);
      }
    }
    g.globalAlpha = 1;
    const labels = $('#hist-labels');
    labels.innerHTML = '';
    for (let t = Math.ceil(this.d.tMin / 3600) * 3600; t <= this.d.tMax; t += 3600) {
      const span = document.createElement('span');
      span.textContent = hhmm(t);
      span.style.left = `${((t - this.d.tMin) / (this.d.tMax - this.d.tMin)) * 100}%`;
      labels.appendChild(span);
    }
  }

  // ---------- left panel ----------

  _buildLeft() {
    const st = this.state;
    const bind = (id, key, after) => {
      const el = $(id);
      el.checked = !!key.get();
      el.addEventListener('change', () => { key.set(el.checked); after?.(el.checked); });
    };
    const layer = (name) => ({ get: () => st.layers[name], set: (v) => { st.layers[name] = v; this.world.setLayer(name, v); } });
    bind('#l-borders', layer('borders'));
    bind('#l-states', layer('states'));
    bind('#l-rivers', layer('rivers'));
    bind('#l-cities', layer('cities'));
    bind('#l-airports', layer('airports'));
    bind('#l-refs', layer('references'));
    bind('#l-corridor', { get: () => st.layers.corridor, set: (v) => { st.layers.corridor = v; } }, () => this.world.showSelection(st.selected));
    bind('#l-trails', { get: () => st.trails, set: (v) => { st.trails = v; } });
    bind('#l-drops', { get: () => st.drops, set: (v) => { st.drops = v; } });
    bind('#l-halos', { get: () => st.halos, set: (v) => { st.halos = v; } });
    bind('#l-fixed', { get: () => st.showFixed, set: (v) => { st.showFixed = v; } }, () => { this._drawHistogram(); this.actions.restyle(); this._feedKey = ''; });
    bind('#l-onlyref', { get: () => st.onlyRef, set: (v) => { st.onlyRef = v; } });
    for (const name of ['borders', 'states', 'rivers', 'cities', 'airports', 'references']) this.world.setLayer(name, st.layers[name]);

    const range = (id, key, out, f = (v) => v) => {
      const el = $(id);
      el.value = String(st[key]);
      const show = () => { $(out).textContent = f(st[key]); };
      el.addEventListener('input', () => { st[key] = Number(el.value); show(); });
      show();
    };
    range('#vex', 'vex', '#vex-v', (v) => `${v}×`);
    range('#size', 'size', '#size-v', (v) => `${v}×`);
    range('#trailmin', 'trailMin', '#trailmin-v', (v) => `${v} min`);
  }

  _buildLegend() {
    const el = $('#legend');
    const mode = this.state.colorMode;
    const thr = this.d.meta.thresholds.imm;
    if (mode === 'conformance') {
      const stops = SCORE_LEGEND.map(([x, c]) => `${c} ${(x / 2.2) * 100}%`).join(',');
      el.innerHTML = `<div class="legend-title">IMM conformance score ÷ phase threshold</div>
        <div class="ramp" style="background:linear-gradient(90deg,${stops})"></div>
        <div class="ramp-labels"><span>0</span><span style="left:${(1 / 2.2) * 100}%">1 = threshold</span><span style="left:100%">2.2+</span></div>
        <div class="legend-note">Budget-point thresholds, chosen on the tune day: climb ${thr[0].toFixed(2)}, cruise ${thr[1].toFixed(2)}, descent ${thr[2].toFixed(2)}. Score = −log10 P, rarest component.</div>
        <div class="legend-row"><i style="background:${PAL.noRef}"></i>no reference path (not scored)</div>
        <div class="legend-row"><i class="glow" style="background:${PAL.imm}"></i>IMM alert open</div>
        <div class="legend-row"><i class="glow" style="background:${PAL.fixed}"></i>fixed-distance alert open</div>`;
    } else if (mode === 'phase') {
      el.innerHTML = `<div class="legend-title">Flight phase (trailing 60 s vertical rate)</div>
        <div class="legend-row"><i style="background:${PAL.climb}"></i>climb</div>
        <div class="legend-row"><i style="background:${PAL.cruise}"></i>level ("cruise")</div>
        <div class="legend-row"><i style="background:${PAL.descent}"></i>descent</div>
        <div class="legend-row"><i style="background:${PAL.unknown}"></i>unknown</div>`;
    } else {
      const stops = ALT_LEGEND.map(([x, c]) => `${c} ${(x / 13) * 100}%`).join(',');
      el.innerHTML = `<div class="legend-title">Barometric altitude</div>
        <div class="ramp" style="background:linear-gradient(90deg,${stops})"></div>
        <div class="ramp-labels"><span>0</span><span style="left:${(5 / 13) * 100}%">FL160</span><span style="left:${(11 / 13) * 100}%">FL360</span></div>`;
    }
  }

  // ---------- right panel ----------

  _buildRight() {
    for (const tab of document.querySelectorAll('.tab')) {
      tab.addEventListener('click', () => this.showTab(tab.dataset.tab));
    }
    this.scoreChart = new SeriesChart($('#chart-score'), (t) => this.actions.seek(t));
    this.fixedChart = new SeriesChart($('#chart-fixed'), (t) => this.actions.seek(t));
    $('#follow').addEventListener('click', () => this.actions.setFollow(!this.state.follow));
    $('#focus').addEventListener('click', () => this.actions.focusSelected());
    $('#clear').addEventListener('click', () => this.actions.select(-1));
    $('#find').addEventListener('change', (e) => {
      const id = e.target.value.trim().toUpperCase();
      const f = this.d.flights.find((x) => x.id === id || x.id === `T${id.padStart(4, '0')}`);
      if (f) this.actions.select(f.index, { seek: true, focus: true });
      e.target.value = '';
    });
    $('#random-alert').addEventListener('click', () => {
      const pool = this.world.active.filter((a) => a.alert && !a.dim);
      if (pool.length) this.actions.select(pool[Math.floor(Math.random() * pool.length)].flight, { focus: true });
      else this.toast('No IMM alert open at this time');
    });
    $('#random-ref').addEventListener('click', () => {
      const pool = this.world.active.filter((a) => this.d.flights[a.flight].ref && !a.dim);
      if (pool.length) this.actions.select(pool[Math.floor(Math.random() * pool.length)].flight, { focus: true });
    });

    // Events (injected scenarios).
    const kinds = ['all', 'lateral', 'vertical', 'longitudinal'];
    $('#f-kind').innerHTML = kinds.map((k) => `<option value="${k}">${k}</option>`).join('');
    const updateMag = () => {
      const k = $('#f-kind').value;
      const mags = [...new Set(this.d.scenarios.filter((s) => k === 'all' || s.kind === k).map((s) => s.magnitude))].sort((a, b) => a - b);
      $('#f-mag').innerHTML = `<option value="all">all</option>${k === 'all' ? '' : mags.map((m) => `<option value="${m}">${m}${KIND_UNIT[k]}</option>`).join('')}`;
    };
    updateMag();
    $('#f-kind').addEventListener('change', () => { updateMag(); this.renderScenarios(); });
    for (const id of ['#f-mag', '#f-phase', '#f-outcome', '#f-sort']) $(id).addEventListener('change', () => this.renderScenarios());
    $('#exit-scenario').addEventListener('click', () => this.actions.scenario(-1));
    $('#scenario-traffic').addEventListener('change', (e) => { this.state.scenarioTraffic = e.target.checked; });
    this.renderScenarios();

    // Alert feed.
    $('#feed-imm').addEventListener('change', () => { this._feedKey = ''; });
    $('#feed-fixed').addEventListener('change', () => { this._feedKey = ''; });
  }

  showTab(name) {
    for (const tab of document.querySelectorAll('.tab')) tab.classList.toggle('on', tab.dataset.tab === name);
    for (const p of document.querySelectorAll('.tabpanel')) p.hidden = p.id !== `tab-${name}`;
    this.state.tab = name;
    if (name === 'flight') this.refreshCharts();
  }

  _scenarioFilter() {
    const kind = $('#f-kind').value;
    const mag = $('#f-mag').value;
    const phase = $('#f-phase').value;
    const outcome = $('#f-outcome').value;
    return this.d.scenarios.filter((s) => (kind === 'all' || s.kind === kind)
      && (mag === 'all' || s.magnitude === Number(mag))
      && (phase === 'all' || s.phase === Number(phase))
      && (outcome === 'all'
        || (outcome === 'imm' && s.detImm && !s.detFixed)
        || (outcome === 'fixed' && s.detFixed && !s.detImm)
        || (outcome === 'both' && s.detImm && s.detFixed)
        || (outcome === 'none' && !s.detImm && !s.detFixed)
        || (outcome === 'inalert' && s.imm_onset_in_alert)));
  }

  renderScenarios() {
    const list = this._scenarioFilter();
    const sort = $('#f-sort').value;
    if (sort === 'magnitude') list.sort((a, b) => b.magnitude - a.magnitude || a.onset - b.onset);
    else if (sort === 'deviation') list.sort((a, b) => b.max_deviation - a.max_deviation);
    else list.sort((a, b) => a.onset - b.onset);
    const n = list.length;
    const di = list.filter((s) => s.detImm).length;
    const df = list.filter((s) => s.detFixed).length;
    const pct = (k) => (n ? `${((100 * k) / n).toFixed(1)}%` : '—');
    $('#scen-summary').innerHTML = `<b>${n}</b> injected events · detected by
      <span class="imm">IMM ${pct(di)}</span> · <span class="fixed">fixed ${pct(df)}</span>
      <div class="muted">Each test-day reference flight carries one event per kind, drawn by the suite (seed ${'20220620'}). Detected = an alert attributable to the event within 600 s.</div>`;
    const rows = list.slice(0, 400).map((s) => {
      const f = this.d.flights[s.flight];
      const tag = (ok, cls, first) => `<span class="pill ${ok ? cls : 'miss'}">${ok ? `+${Math.round(first)}s` : 'miss'}</span>`;
      return `<li data-i="${s.index}" class="${this.state.scenario === s.index ? 'on' : ''}">
        <span class="t">${hhmm(s.onset)}</span>
        <span class="k">${s.kind} ${s.sign > 0 ? '+' : '−'}${s.magnitude}${KIND_UNIT[s.kind]}</span>
        <span class="p">${PHASE_LABEL[s.phase]}</span>
        <span class="f">${f.id}</span>
        ${tag(s.detImm, 'imm', s.imm_first)}${tag(s.detFixed, 'fixed', s.fixed_first)}
      </li>`;
    });
    $('#scen-list').innerHTML = rows.join('') + (n > 400 ? `<li class="more">… ${n - 400} more, narrow the filter</li>` : '');
    for (const li of document.querySelectorAll('#scen-list li[data-i]')) {
      li.addEventListener('click', () => this.actions.scenario(Number(li.dataset.i)));
    }
  }

  /** Charts for the selected flight or the scenario's injected copy. */
  refreshCharts() {
    const st = this.state;
    const d = this.d;
    const scen = st.scenario >= 0 ? d.scenarios[st.scenario] : null;
    const track = scen ?? (st.selected >= 0 ? d.flights[st.selected] : null);
    if (!track || !(scen || track.ref)) {
      this.scoreChart.set(null);
      this.fixedChart.set(null);
      $('#charts').hidden = !track;
      $('#charts-empty').hidden = !track || !!(scen || track.ref);
      return;
    }
    $('#charts').hidden = false;
    $('#charts-empty').hidden = true;
    const s = d.series(track, ['t', 'imm_score', 'imm_alert', 'fixed_km', 'fixed_alert', 'presented', 'phase']);
    const thrI = s.phase.map((p) => thresholdAt(p, d.meta.thresholds.imm));
    const thrF = s.phase.map((p) => thresholdAt(p, d.meta.thresholds.fixed_km));
    const maxI = Math.max(4, ...Array.from(s.imm_score).filter(Number.isFinite));
    const yI = Math.min(12, Math.ceil(maxI));
    const events = [];
    if (scen) {
      events.push({ t: scen.onset, color: PAL.onset });
      if (scen.t_violation !== null) events.push({ t: scen.onset + scen.t_violation, color: PAL.violation });
    }
    this.scoreChart.set({
      title: 'IMM score (−log10 P) · dashed: threshold', t0: s.t[0], t1: s.t[s.t.length - 1], t: s.t, v: s.imm_score, thr: thrI,
      alert: s.imm_alert, presented: s.presented.map((p) => p & 1), yMax: yI, ticks: [0, 2, 4, 6, 8, 10, 12].filter((v) => v <= yI),
      color: '#e8eefc', thrColor: PAL.imm, alertFill: 'rgba(255,77,94,0.16)', alertColor: PAL.imm, events,
    });
    const maxF = Math.max(25, ...Array.from(s.fixed_km).filter(Number.isFinite));
    const yF = Math.min(80, Math.ceil(maxF / 10) * 10);
    this.fixedChart.set({
      title: 'distance from reference path (km) · fixed-distance baseline', t0: s.t[0], t1: s.t[s.t.length - 1], t: s.t, v: s.fixed_km, thr: thrF,
      alert: s.fixed_alert, presented: s.presented.map((p) => p & 2), yMax: yF, ticks: [0, 10, 20, 40, 60, 80].filter((v) => v <= yF),
      color: '#c9d6ee', thrColor: PAL.fixed, alertFill: 'rgba(180,140,255,0.14)', alertColor: PAL.fixed, events,
    });
    $('#chart-fixed').parentElement.hidden = !st.showFixed;
  }

  // ---------- per-frame panel updates (throttled) ----------

  update(T, active, force = false) {
    const st = this.state;
    $('#clock').textContent = clock(T);
    const slider = $('#time');
    if (document.activeElement !== slider) slider.value = String(T);
    $('#play').textContent = st.playing ? '❚❚' : '▶';
    $('#play').setAttribute('aria-label', st.playing ? 'Pause' : 'Play');
    const now = performance.now();
    if (st.tab === 'flight') {
      this.scoreChart.draw(T);
      if (st.showFixed) this.fixedChart.draw(T);
    }
    if (!force && now - this._lastPanel < 120) return;
    this._lastPanel = now;
    this._stats(T, active);
    this._flight(T, active);
    if (st.tab === 'alerts') this._feed(T);
  }

  _stats(T, active) {
    const vis = active.filter((a) => !a.dim);
    const withRef = vis.filter((a) => this.d.flights[a.flight].ref).length;
    const inImm = vis.filter((a) => a.alert).length;
    const inFixed = vis.filter((a) => a.fixedAlert).length;
    const r = this.d.recentRate(T, 3600);
    const budget = this.d.meta.budget_per_flight_hour;
    $('#stats').innerHTML = `
      <div class="stat"><b>${vis.length}</b><span>airborne</span></div>
      <div class="stat"><b>${withRef}</b><span>with reference</span></div>
      <div class="stat imm"><b>${inImm}</b><span>in IMM alert</span></div>
      ${this.state.showFixed ? `<div class="stat fixed"><b>${inFixed}</b><span>in fixed alert</span></div>` : ''}
      <div class="rate">
        <div class="rate-head">alerts presented, last 60 min, per flight-hour</div>
        ${this._rateBar('IMM', r.imm, budget, PAL.imm, r.nImm)}
        ${this.state.showFixed ? this._rateBar('fixed', r.fixed, budget, PAL.fixed, r.nFixed) : ''}
        <div class="muted">budget ${budget}/h · ${r.hours.toFixed(1)} reference flight-hours · all phases, approximate</div>
      </div>`;
  }

  _rateBar(name, v, budget, color, n) {
    const max = Math.max(budget * 2, v);
    return `<div class="ratebar"><span class="rn">${name}</span><div class="rb"><i style="width:${(v / max) * 100}%;background:${color}"></i>
      <em style="left:${(budget / max) * 100}%"></em></div><span class="rv">${v.toFixed(2)} <small>(${n})</small></span></div>`;
  }

  _flight(T, active) {
    const st = this.state;
    const d = this.d;
    const box = $('#flight-info');
    if (st.selected < 0) {
      box.innerHTML = '<p class="muted">Click an aircraft, an alert in the feed, or an injected event. <kbd>Esc</kbd> clears.</p>';
      $('#comp-bars').innerHTML = '';
      $('#flight-actions').hidden = true;
      return;
    }
    $('#flight-actions').hidden = false;
    $('#follow').classList.toggle('on', st.follow);
    const f = d.flights[st.selected];
    const a = active.find((x) => x.flight === st.selected);
    const scen = st.scenario >= 0 ? d.scenarios[st.scenario] : null;
    const thr = a ? thresholdAt(a.phase, d.meta.thresholds.imm) : NaN;
    const thrF = a ? thresholdAt(a.phase, d.meta.thresholds.fixed_km) : NaN;
    const status = !a ? `<span class="pill miss">not airborne at ${hhmm(T)} (${hhmm(f.t0)}–${hhmm(f.t1)})</span>`
      : a.alert ? '<span class="pill imm">IMM ALERT</span>' : f.ref ? '<span class="pill ok">conforming</span>' : '<span class="pill miss">no reference</span>';
    let html = `<div class="fl-head"><span class="fl-id">${f.id}</span>${status}${a?.fixedAlert && st.showFixed ? '<span class="pill fixed">FIXED ALERT</span>' : ''}</div>
      <div class="fl-sub">${f.ref ? `reference <code>${f.ref}</code> · ${d.refByKey.get(f.ref)?.flights ?? '?'} tune-day flights` : 'no reference path: pair not flown ≥5 times on the tune day'}</div>`;
    if (scen) {
      const dev = a ? a.deviation : NaN;
      const sig = d.meta.significance[scen.kind];
      const devTxt = scen.kind === 'vertical' ? `${fmt(dev, 0)} ft` : `${fmt(dev / 1852, 2)} NM`;
      const sigTxt = scen.kind === 'vertical' ? `${sig} ft` : `${(sig / 1852).toFixed(0)} NM`;
      html += `<div class="scen-box">
        <div class="scen-title">Injected: ${scen.kind} ${scen.sign > 0 ? '+' : '−'}${scen.magnitude}${KIND_UNIT[scen.kind]} at ${clock(scen.onset)} (${PHASE_LABEL[scen.phase]})</div>
        <div class="kv"><span>deviation now</span><b>${devTxt}</b><span>violation at</span><b>${sigTxt}</b></div>
        <div class="kv"><span>violated</span><b>${scen.t_violation === null ? 'not within 600 s' : `+${Math.round(scen.t_violation)} s`}</b>
          <span>event kind</span><b>${scen.t_violation === null ? '—' : scen.t_violation >= 120 ? 'gradual' : 'abrupt'}</b></div>
        <div class="kv"><span class="imm">IMM</span><b>${scen.detImm ? `alert +${Math.round(scen.imm_first)} s${scen.t_violation !== null ? ` (lead ${signed(scen.t_violation - scen.imm_first)} s)` : ''}` : 'missed'}</b>
          <span class="fixed">fixed</span><b>${scen.detFixed ? `alert +${Math.round(scen.fixed_first)} s` : 'missed'}</b></div>
        ${scen.imm_onset_in_alert ? '<div class="warn">Onset fell inside an IMM alert the real flight already had open: nothing can be credited to the event there.</div>' : ''}
      </div>`;
    }
    if (a && a.age > 30) {
      html += `<div class="warn gap">Coverage gap: last fix ${a.age >= 120 ? `${Math.round(a.age / 60)} min` : `${Math.round(a.age)} s`} ago. Position interpolated; score and alert are the monitor's state at that fix.</div>`;
    }
    if (a) {
      html += `<div class="grid4">
        <div><span>phase</span><b>${PHASE_LABEL[a.phase]}</b></div>
        <div><span>altitude</span><b>FL${String(Math.round((a.alt * 1000) / FT / 100)).padStart(3, '0')}</b></div>
        <div><span>ground speed</span><b>${fmt(a.speed / KT, 0)} kt</b></div>
        <div><span>vertical rate</span><b>${fmt((a.vrate / FT) * 60, 0)} ft/min</b></div>
        <div><span>IMM score</span><b class="${a.alert ? 'imm' : ''}">${fmt(a.score, 2)} <small>/ ${fmt(thr, 2)}</small></b></div>
        <div><span>rarest component</span><b>${a.comp >= 0 ? COMPONENT_LABEL[d.meta.components[a.comp]] : '—'}</b></div>
        <div><span>off path</span><b>${fmt(a.fixedKm / NM, 2)} NM <small>/ ${fmt(thrF / NM, 1)}</small></b></div>
        <div><span>alerts presented${scen ? ', injected copy' : ''}</span><b>${scen ? scen.imm_alerts_presented : f.imm_alerts} IMM${st.showFixed ? ` · ${scen ? scen.fixed_alerts_presented : f.fixed_alerts} fixed` : ''}</b></div>
      </div>`;
    }
    box.innerHTML = html;
    // Per-component rarity at the current fix.
    const comps = a ? d.components(a.track, a.k) : null;
    if (comps) {
      const max = Math.max(thr * 2, 4);
      $('#comp-bars').innerHTML = `<div class="comp-head">rarity per component (−log10 P) at the last fix</div>${
        d.meta.components.map((c, i) => {
          const v = comps[i];
          const w = Number.isFinite(v) ? Math.min(1, v / max) * 100 : 0;
          const hot = v > thr;
          return `<div class="comp"><span>${COMPONENT_LABEL[c]}</span><div class="cb"><i class="${hot ? 'hot' : ''}" style="width:${w}%"></i><em style="left:${(thr / max) * 100}%"></em></div><b>${fmt(v, 2)}</b></div>`;
        }).join('')}`;
    } else {
      $('#comp-bars').innerHTML = '';
    }
  }

  _feed(T) {
    const d = this.d;
    const end = d.alertsBefore(T);
    const showImm = $('#feed-imm').checked;
    const showFixed = $('#feed-fixed').checked && this.state.showFixed;
    const key = `${end}|${showImm}|${showFixed}|${this.state.selected}`;
    if (key === this._feedKey) return;
    this._feedKey = key;
    const out = [];
    for (let i = end - 1; i >= 0 && out.length < 60; i--) {
      const a = d.alerts[i];
      if ((a.det === 'imm' && !showImm) || (a.det === 'fixed' && !showFixed)) continue;
      out.push(a);
    }
    $('#feed-count').textContent = `${end} alerts presented since ${hhmm(d.tMin)}`;
    $('#feed').innerHTML = out.map((a) => `<li data-f="${a.flight}" data-t="${a.t}" class="${a.flight === this.state.selected ? 'on' : ''}">
      <span class="t">${clock(a.t)}</span><span class="pill ${a.det}">${a.det === 'imm' ? 'IMM' : 'fixed'}</span>
      <span class="f">${d.flights[a.flight].id}</span><span class="p">${PHASE_LABEL[a.phase]}</span>
      <span class="c">${a.det === 'imm' ? `${COMPONENT_LABEL[d.meta.components[a.comp]] ?? ''} ${fmt(a.score, 1)}` : `${fmt(a.fixedKm / NM, 1)} NM`}</span></li>`).join('');
    for (const li of document.querySelectorAll('#feed li')) {
      li.addEventListener('click', () => {
        this.actions.select(Number(li.dataset.f), { focus: true });
        this.actions.seek(Number(li.dataset.t) - 20);
      });
    }
  }

  // ---------- viewport, keys ----------

  _bindViewport() {
    const canvas = this.world.renderer.domElement;
    const tip = $('#tooltip');
    let down = null;
    canvas.addEventListener('pointerdown', (e) => { down = { x: e.clientX, y: e.clientY }; });
    canvas.addEventListener('pointerup', (e) => {
      if (!down || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 4) return;
      const hit = this.world.pick(e.clientX, e.clientY);
      this.actions.select(hit ? hit.flight : -1);
      down = null;
    });
    canvas.addEventListener('dblclick', (e) => {
      const hit = this.world.pick(e.clientX, e.clientY);
      if (hit) { this.actions.select(hit.flight); this.actions.focusSelected(); }
    });
    canvas.addEventListener('pointermove', (e) => {
      const hit = this.world.pick(e.clientX, e.clientY);
      if (!hit) { tip.hidden = true; canvas.style.cursor = ''; return; }
      canvas.style.cursor = 'pointer';
      const f = this.d.flights[hit.flight];
      const thr = thresholdAt(hit.phase, this.d.meta.thresholds.imm);
      tip.hidden = false;
      tip.style.left = `${e.clientX + 14}px`;
      tip.style.top = `${e.clientY + 12}px`;
      tip.innerHTML = `<b>${f.id}</b> · ${PHASE_LABEL[hit.phase]} · FL${String(Math.round((hit.alt * 1000) / FT / 100)).padStart(3, '0')}<br>${
        f.ref ? `score <b class="${hit.alert ? 'imm' : ''}">${fmt(hit.score, 2)}</b> / ${thr.toFixed(2)}${hit.alert ? ' · <span class="imm">alert</span>' : ''}` : '<span class="muted">no reference</span>'}`;
    });
    canvas.addEventListener('pointerleave', () => { tip.hidden = true; });
  }

  _bindKeys() {
    window.addEventListener('keydown', (e) => {
      if (e.target.matches('input[type=text], input[type=search], select')) return;
      const st = this.state;
      if (e.code === 'Space') { e.preventDefault(); this.actions.togglePlay(); }
      else if (e.key === 'ArrowRight') this.actions.seek(st.T + (e.shiftKey ? 600 : 60));
      else if (e.key === 'ArrowLeft') this.actions.seek(st.T - (e.shiftKey ? 600 : 60));
      else if (e.key === 'f' || e.key === 'F') this.actions.setFollow(!st.follow);
      else if (e.key === 'Escape') { if (st.scenario >= 0) this.actions.scenario(-1); else this.actions.select(-1); }
      else if (e.key === 'h' || e.key === 'H') this.world.home();
      else if (e.key === '1' || e.key === '2' || e.key === '3') {
        st.colorMode = ['conformance', 'phase', 'altitude'][Number(e.key) - 1];
        $('#colormode').value = st.colorMode;
        this.actions.restyle();
        this._buildLegend();
      } else if (e.key === '+' || e.key === '=') this._stepSpeed(1);
      else if (e.key === '-') this._stepSpeed(-1);
    });
  }

  _stepSpeed(dir) {
    const sel = $('#speed');
    const opts = [...sel.options].map((o) => Number(o.value));
    const i = Math.max(0, Math.min(opts.length - 1, opts.indexOf(this.state.speed) + dir));
    this.state.speed = opts[i];
    sel.value = String(opts[i]);
  }

  scenarioBanner(show) {
    $('#scenario-banner').hidden = !show;
    const st = this.state;
    if (show) {
      const s = this.d.scenarios[st.scenario];
      $('#scenario-text').textContent = `${s.kind} ${s.sign > 0 ? '+' : '−'}${s.magnitude}${KIND_UNIT[s.kind]} on ${this.d.flights[s.flight].id} · onset ${clock(s.onset)}`;
    }
    for (const li of document.querySelectorAll('#scen-list li[data-i]')) li.classList.toggle('on', Number(li.dataset.i) === st.scenario);
  }

  toast(text) {
    const el = $('#toast');
    el.textContent = text;
    el.hidden = false;
    clearTimeout(this._toastT);
    this._toastT = setTimeout(() => { el.hidden = true; }, 2200);
  }
}
