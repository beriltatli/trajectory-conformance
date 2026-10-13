// Loads data/sim/scene.json + scene.bin (scripts/sim.py) and answers "where was this track at T".

async function fetchWithProgress(url, onBytes) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = Number(res.headers.get('content-length')) || 0;
  if (!res.body || !total) return res.arrayBuffer();
  const reader = res.body.getReader();
  const out = new Uint8Array(total);
  let at = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    out.set(value, at);
    at += value.length;
    onBytes(at, total);
  }
  return out.buffer;
}

export async function loadScene(dataBase, onProgress) {
  const [meta, basemap] = await Promise.all([
    fetch(`${dataBase}scene.json`).then((r) => {
      if (!r.ok) throw new Error(`${dataBase}scene.json: HTTP ${r.status} — run "make sim-data" (OpenSky, local) or "make demo" (synthetic) first`);
      return r.json();
    }),
    fetch('assets/basemap.json').then((r) => r.json()),
  ]);
  const buf = await fetchWithProgress(`${dataBase}scene.bin`, onProgress);
  return new SceneData(meta, buf, basemap);
}

const lerp = (a, b, f) => a + (b - a) * f;
function lerpAngle(a, b, f) {
  let d = ((b - a + 540) % 360) - 180;
  return a + d * f;
}

export class SceneData {
  constructor(meta, buf, basemap) {
    this.meta = meta;
    this.basemap = basemap;
    this.S = meta.columns.length;
    this.C = Object.fromEntries(meta.columns.map((c, i) => [c, i]));
    this.rows = new Float32Array(buf, 0, meta.rows * this.S);
    this.nc = meta.components.length;
    this.comp = new Float32Array(buf, meta.comp_byte_offset, meta.comp_rows * this.nc);
    this.flights = meta.flights;
    this.flights.forEach((f, i) => { f.index = i; f.kind = 'flight'; });
    this.scenarios = meta.scenarios;
    this.scenarios.forEach((s, i) => {
      s.index = i;
      s.kind_ = 'scenario';
      s.t0 = this.get(s, 0, this.C.t);
      s.t1 = this.get(s, s.rows - 1, this.C.t);
      s.imm_alerts_presented = 0;
      s.fixed_alerts_presented = 0;
      for (let i = 0; i < s.rows; i++) {
        const p = this.get(s, i, this.C.presented);
        if (p & 1) s.imm_alerts_presented++;
        if (p & 2) s.fixed_alerts_presented++;
      }
      s.detImm = s.imm_first !== null;
      s.detFixed = s.fixed_first !== null;
    });
    this.refByKey = new Map(meta.references.map((r) => [r.key, r]));
    this.tMin = Math.min(...this.flights.map((f) => f.t0));
    this.tMax = Math.max(...this.flights.map((f) => f.t1));
    this._buildIndexes();
  }

  get(track, i, col) {
    return this.rows[(track.offset + i) * this.S + col];
  }

  /** Last row with t <= T, or -1 before the track starts. */
  find(track, T) {
    const S = this.S;
    const R = this.rows;
    const base = track.offset;
    let lo = 0;
    let hi = track.rows - 1;
    if (T < R[base * S]) return -1;
    if (T >= R[(base + hi) * S]) return hi;
    while (hi - lo > 1) {
      const m = (lo + hi) >> 1;
      if (R[(base + m) * S] <= T) lo = m;
      else hi = m;
    }
    return lo;
  }

  /** Position interpolated between fixes; everything the monitor knew is from the last fix (causal). */
  sample(track, T, out) {
    const k = this.find(track, T);
    if (k < 0) return false;
    const S = this.S;
    const R = this.rows;
    const C = this.C;
    const a = (track.offset + k) * S;
    let b = a;
    let f = 0;
    let gap = 0;
    if (k < track.rows - 1) {
      b = a + S;
      gap = R[b] - R[a];
      f = gap > 0 ? Math.min(1, (T - R[a]) / gap) : 0;
    }
    out.k = k;
    out.e = lerp(R[a + C.east_km], R[b + C.east_km], f);
    out.n = lerp(R[a + C.north_km], R[b + C.north_km], f);
    out.alt = lerp(R[a + C.alt_km], R[b + C.alt_km], f);
    out.heading = lerpAngle(R[a + C.heading_deg], R[b + C.heading_deg], f);
    out.speed = R[a + C.speed_mps];
    out.vrate = R[a + C.vrate_mps];
    out.phase = R[a + C.phase];
    out.score = R[a + C.imm_score];
    out.alert = R[a + C.imm_alert] > 0;
    out.comp = R[a + C.imm_component];
    out.fixedKm = R[a + C.fixed_km];
    out.fixedAlert = R[a + C.fixed_alert] > 0;
    out.deviation = R[a + C.deviation];
    out.coast = gap > 60;
    out.age = T - R[a];
    return true;
  }

  components(track, k) {
    if (track.comp < 0 || k < 0) return null;
    const o = (track.comp + k) * this.nc;
    return this.comp.subarray(o, o + this.nc);
  }

  /** Columns of a whole track as plain arrays, for charts and paths. */
  series(track, cols) {
    const out = {};
    for (const c of cols) {
      const col = this.C[c];
      const a = new Float32Array(track.rows);
      for (let i = 0; i < track.rows; i++) a[i] = this.rows[(track.offset + i) * this.S + col];
      out[c] = a;
    }
    return out;
  }

  _buildIndexes() {
    // Every presented alert of either detector, in time order, for the feed and the timeline.
    const C = this.C;
    const alerts = [];
    const bin = 60;
    const nb = Math.ceil((this.tMax - this.tMin) / bin) + 1;
    const airborne = new Float32Array(nb);
    const refMinutes = new Float32Array(nb);
    const imm = new Float32Array(nb);
    const fixed = new Float32Array(nb);
    for (const f of this.flights) {
      for (let i = 0; i < f.rows; i++) {
        const t = this.get(f, i, C.t);
        const p = this.get(f, i, C.presented);
        if (p > 0) {
          const row = { t, flight: f.index, phase: this.get(f, i, C.phase), comp: this.get(f, i, C.imm_component),
            score: this.get(f, i, C.imm_score), fixedKm: this.get(f, i, C.fixed_km) };
          if (p & 1) alerts.push({ ...row, det: 'imm' });
          if (p & 2) alerts.push({ ...row, det: 'fixed' });
          const b = Math.floor((t - this.tMin) / bin);
          if (p & 1) imm[b] += 1;
          if (p & 2) fixed[b] += 1;
        }
      }
      const b0 = Math.max(0, Math.floor((f.t0 - this.tMin) / bin));
      const b1 = Math.min(nb - 1, Math.floor((f.t1 - this.tMin) / bin));
      for (let b = b0; b <= b1; b++) {
        airborne[b] += 1;
        if (f.ref) refMinutes[b] += 1;
      }
    }
    alerts.sort((x, y) => x.t - y.t);
    this.alerts = alerts;
    this.alertTimes = Float64Array.from(alerts, (a) => a.t);
    this.bins = { size: bin, airborne, refMinutes, imm, fixed };
  }

  /** Index of the first alert with t > T. */
  alertsBefore(T) {
    const a = this.alertTimes;
    let lo = 0;
    let hi = a.length;
    while (lo < hi) {
      const m = (lo + hi) >> 1;
      if (a[m] <= T) lo = m + 1;
      else hi = m;
    }
    return lo;
  }

  /** Presented alerts per flight-hour of reference flights over the last `windowS` before T. */
  recentRate(T, windowS) {
    const { size, refMinutes, imm, fixed } = this.bins;
    const b1 = Math.floor((T - this.tMin) / size);
    const b0 = Math.max(0, b1 - Math.round(windowS / size) + 1);
    let mins = 0;
    let ni = 0;
    let nf = 0;
    for (let b = b0; b <= b1 && b < refMinutes.length; b++) {
      mins += refMinutes[b];
      ni += imm[b];
      nf += fixed[b];
    }
    const hours = (mins * size) / 3600;
    return { hours, imm: hours > 0 ? ni / hours : 0, fixed: hours > 0 ? nf / hours : 0, nImm: ni, nFixed: nf };
  }
}
