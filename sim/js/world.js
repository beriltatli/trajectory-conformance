import * as THREE from 'three';
import { MapControls } from 'three/addons/controls/MapControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { Line2 } from 'three/addons/lines/Line2.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { LineGeometry } from 'three/addons/lines/LineGeometry.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { PAL, scoreColor, phaseColor, altColor, fade, thresholdAt } from './colors.js';

// World units are kilometres: x east, z south (north is -z), y up. Everything carrying an
// altitude sits in `vgroup`, whose y scale is the vertical exaggeration; aircraft models sit
// outside it so they are not stretched, and are placed at alt * exaggeration by hand.
const TRAIL_CAP = 200000;
const DROP_CAP = 4000;
const AIRCRAFT_CAP = 4000;
const GAP_S = 120;

function aircraftGeometry() {
  // Forward is +x, wings along z, fin up; length 1.
  const parts = [];
  const fuselage = new THREE.CylinderGeometry(0.05, 0.035, 0.86, 10);
  fuselage.rotateZ(-Math.PI / 2);
  parts.push(fuselage);
  const nose = new THREE.ConeGeometry(0.05, 0.16, 10);
  nose.rotateZ(-Math.PI / 2);
  nose.translate(0.51, 0, 0);
  parts.push(nose);
  const wing = new THREE.BufferGeometry();
  // Swept wing as a thin prism.
  const w = [
    [0.12, 0, 0], [-0.1, 0, 0.5], [-0.2, 0, 0.5], [-0.12, 0, 0],
    [0.12, 0, 0], [-0.1, 0, -0.5], [-0.2, 0, -0.5], [-0.12, 0, 0],
  ];
  const quads = [[0, 1, 2, 3], [4, 7, 6, 5]];
  const pos = [];
  for (const q of quads) {
    const [a, b, c, d] = q.map((i) => w[i]);
    for (const tri of [[a, b, c], [a, c, d]]) for (const v of tri) pos.push(...v);
  }
  const top = new Float32Array(pos);
  const bottom = top.slice();
  for (let i = 1; i < bottom.length; i += 3) bottom[i] = -0.012;
  const all = new Float32Array(top.length * 2);
  all.set(top, 0);
  // Bottom faces reversed so both sides are lit.
  for (let t = 0; t < top.length / 9; t++) {
    const o = t * 9;
    all.set([bottom[o], bottom[o + 1], bottom[o + 2], bottom[o + 6], bottom[o + 7], bottom[o + 8], bottom[o + 3], bottom[o + 4], bottom[o + 5]], top.length + o);
  }
  wing.setAttribute('position', new THREE.BufferAttribute(all, 3));
  wing.computeVertexNormals();
  const tailplane = new THREE.BoxGeometry(0.1, 0.01, 0.34);
  tailplane.translate(-0.38, 0.01, 0);
  const fin = new THREE.BoxGeometry(0.14, 0.16, 0.01);
  fin.translate(-0.37, 0.09, 0);
  const engines = [0.2, -0.2].map((z) => {
    const e = new THREE.CylinderGeometry(0.025, 0.025, 0.12, 8);
    e.rotateZ(-Math.PI / 2);
    e.translate(0.04, -0.03, z);
    return e;
  });
  const indexed = [fuselage, nose, tailplane, fin, ...engines].map((g) => g.toNonIndexed());
  for (const g of indexed) g.deleteAttribute('uv');
  return mergeGeometries([...indexed, wing]);
}

function glowTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const g = c.getContext('2d');
  const grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, 'rgba(255,255,255,1)');
  grad.addColorStop(0.25, 'rgba(255,255,255,0.55)');
  grad.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grad;
  g.fillRect(0, 0, 64, 64);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function polylineSegments(lines, y) {
  const out = [];
  for (const flat of lines) {
    for (let i = 0; i + 3 < flat.length; i += 2) out.push(flat[i], y, -flat[i + 1], flat[i + 2], y, -flat[i + 3]);
  }
  return new Float32Array(out);
}

function label(text, cls) {
  const el = document.createElement('div');
  el.className = cls;
  el.textContent = text;
  return new CSS2DObject(el);
}

export class World {
  constructor(container, data, state) {
    this.data = data;
    this.state = state;
    this.container = container;

    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    container.appendChild(renderer.domElement);
    this.renderer = renderer;

    const labels = new CSS2DRenderer();
    labels.domElement.className = 'labels';
    container.appendChild(labels.domElement);
    this.labelRenderer = labels;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(PAL.bg);
    scene.fog = new THREE.Fog(PAL.bg, 700, 2200);
    this.scene = scene;

    const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 6000);
    camera.position.set(-110, 205, 470);
    this.camera = camera;
    const controls = new MapControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.09;
    controls.maxPolarAngle = Math.PI * 0.47;
    controls.minDistance = 1.5;
    controls.maxDistance = 1800;
    controls.zoomToCursor = true;
    controls.target.set(0, 0, 0);
    this.controls = controls;

    scene.add(new THREE.HemisphereLight(0xb8c8ff, 0x0b0f18, 1.1));
    const sun = new THREE.DirectionalLight(0xffffff, 1.6);
    sun.position.set(-200, 500, 150);
    scene.add(sun);

    this.vgroup = new THREE.Group();
    scene.add(this.vgroup);

    this._buildGround();
    this._buildReferences();
    this._buildAircraft();
    this._buildTrails();

    this.selection = new THREE.Group();
    this.vgroup.add(this.selection);
    this.scenarioGroup = new THREE.Group();
    this.vgroup.add(this.scenarioGroup);

    this.selLabel = label('', 'ac-label');
    this.selLabel.visible = false;
    scene.add(this.selLabel);

    this.lineMaterials = [];
    this.active = [];
    this._pool = [];
    this._tmp = { m: new THREE.Matrix4(), q: new THREE.Quaternion(), qp: new THREE.Quaternion(), p: new THREE.Vector3(), s: new THREE.Vector3(), c: new THREE.Color(), c2: new THREE.Color(), v: new THREE.Vector3() };
    this._fly = null;
    this._lastFollow = null;

    window.addEventListener('resize', () => this.resize());
    this.resize();
  }

  // ---------- static layers ----------

  _buildGround() {
    const bm = this.data.basemap;
    const ground = new THREE.Mesh(new THREE.CircleGeometry(2400, 64), new THREE.MeshStandardMaterial({ color: PAL.ground, roughness: 1, metalness: 0 }));
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -0.05;
    this.scene.add(ground);

    // Region box: filled, outlined.
    const shape = new THREE.Shape();
    const r = bm.region;
    shape.moveTo(r[0], r[1]);
    for (let i = 2; i < r.length; i += 2) shape.lineTo(r[i], r[i + 1]);
    const fill = new THREE.Mesh(new THREE.ShapeGeometry(shape), new THREE.MeshStandardMaterial({ color: PAL.region, roughness: 1 }));
    fill.rotation.x = -Math.PI / 2;
    fill.position.y = -0.03;
    this.scene.add(fill);
    const outline = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(Array.from({ length: r.length / 2 + 1 }, (_, i) => new THREE.Vector3(r[(2 * i) % r.length], 0.02, -r[(2 * i + 1) % r.length]))),
      new THREE.LineDashedMaterial({ color: '#3d5680', dashSize: 6, gapSize: 4 }),
    );
    outline.computeLineDistances();
    this.scene.add(outline);

    // 50 km grid over the region.
    const g = [];
    for (let x = -300; x <= 300; x += 50) g.push(x, 0, -300, x, 0, 300, -300, 0, x, 300, 0, x);
    const grid = new THREE.LineSegments(new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(g, 3)),
      new THREE.LineBasicMaterial({ color: PAL.grid, transparent: true, opacity: 0.7 }));
    grid.position.y = -0.01;
    this.scene.add(grid);
    this.grid = grid;

    const mk = (lines, color, opacity, y) => {
      const geo = new THREE.BufferGeometry().setAttribute('position', new THREE.BufferAttribute(polylineSegments(lines, y), 3));
      const obj = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color, transparent: true, opacity }));
      this.scene.add(obj);
      return obj;
    };
    this.layers = {
      borders: mk(bm.borders, PAL.border, 0.9, 0.03),
      states: mk(bm.states, PAL.state, 0.9, 0.02),
      rivers: new THREE.Group(),
    };
    this.layers.rivers.add(mk(bm.rivers, PAL.river, 0.85, 0.025));
    this.layers.rivers.add(mk(bm.lakes, PAL.river, 0.85, 0.025));
    this.scene.add(this.layers.rivers);

    // Cities.
    this.cityLabels = [];
    const cityGroup = new THREE.Group();
    for (const c of bm.cities) {
      const l = label(c.name, `city-label rank${c.rank}`);
      l.position.set(c.east_km, 0.1, -c.north_km);
      l.userData.rank = c.rank;
      cityGroup.add(l);
      this.cityLabels.push(l);
    }
    this.scene.add(cityGroup);
    this.layers.cities = cityGroup;

    // Airports: rings sized by how many flights started or ended there; named ones labelled.
    const apGroup = new THREE.Group();
    const ringGeo = new THREE.RingGeometry(0.8, 1.15, 28);
    ringGeo.rotateX(-Math.PI / 2);
    const ringMat = new THREE.MeshBasicMaterial({ color: '#c9d6ee', transparent: true, opacity: 0.55, side: THREE.DoubleSide, depthWrite: false });
    for (const ap of this.data.meta.airports) {
      if (!ap.name && (ap.endpoints ?? 0) < 15) continue;
      const ring = new THREE.Mesh(ringGeo, ringMat);
      const s = ap.name ? 2.2 + Math.min(4, Math.sqrt(ap.endpoints ?? 1) / 4) : 1.2;
      ring.scale.setScalar(s);
      ring.position.set(ap.east_km, 0.06, -ap.north_km);
      apGroup.add(ring);
      if (ap.name) {
        const l = label(ap.name, 'ap-label');
        l.position.set(ap.east_km, 0.2, -ap.north_km);
        apGroup.add(l);
      }
    }
    this.scene.add(apGroup);
    this.layers.airports = apGroup;
  }

  _buildReferences() {
    // Every reference centreline at its median altitude, faint; the selected one gets a corridor.
    const pos = [];
    for (const ref of this.data.meta.references) {
      for (let i = 0; i + 1 < ref.east_km.length; i++) {
        pos.push(ref.east_km[i], ref.alt_km[i], -ref.north_km[i], ref.east_km[i + 1], ref.alt_km[i + 1], -ref.north_km[i + 1]);
      }
    }
    const geo = new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    this.layers.references = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: '#2f6fb0', transparent: true, opacity: 0.28, depthWrite: false }));
    this.vgroup.add(this.layers.references);
  }

  _buildAircraft() {
    const geo = aircraftGeometry();
    const mat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.45, metalness: 0.15, emissive: 0x111827 });
    this.aircraft = new THREE.InstancedMesh(geo, mat, AIRCRAFT_CAP);
    this.aircraft.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.aircraft.setColorAt(0, new THREE.Color());
    this.aircraft.instanceColor.setUsage(THREE.DynamicDrawUsage);
    this.aircraft.frustumCulled = false;
    this.aircraft.count = 0;
    this.scene.add(this.aircraft);

    this.ghost = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: 0xffffff, transparent: true, opacity: 0.35, depthWrite: false }));
    this.ghost.visible = false;
    this.scene.add(this.ghost);

    const tex = glowTexture();
    const mkHalo = (color, size) => {
      const g = new THREE.BufferGeometry().setAttribute('position', new THREE.BufferAttribute(new Float32Array(AIRCRAFT_CAP * 3), 3));
      g.setDrawRange(0, 0);
      const p = new THREE.Points(g, new THREE.PointsMaterial({ color, size, map: tex, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, sizeAttenuation: false }));
      p.frustumCulled = false;
      this.scene.add(p);
      return p;
    };
    this.haloImm = mkHalo(PAL.imm, 34);
    this.haloFixed = mkHalo(PAL.fixed, 22);
    this.haloSel = mkHalo('#ffffff', 40);
  }

  _buildTrails() {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(TRAIL_CAP * 6), 3).setUsage(THREE.DynamicDrawUsage));
    geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(TRAIL_CAP * 6), 3).setUsage(THREE.DynamicDrawUsage));
    geo.setDrawRange(0, 0);
    this.trails = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.95, depthWrite: false }));
    this.trails.frustumCulled = false;
    this.vgroup.add(this.trails);

    const dgeo = new THREE.BufferGeometry();
    dgeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(DROP_CAP * 6), 3).setUsage(THREE.DynamicDrawUsage));
    dgeo.setDrawRange(0, 0);
    this.drops = new THREE.LineSegments(dgeo, new THREE.LineBasicMaterial({ color: '#2a3c5c', transparent: true, opacity: 0.55, depthWrite: false }));
    this.drops.frustumCulled = false;
    this.vgroup.add(this.drops);
  }

  // ---------- per frame ----------

  _rowColor(track, i, out) {
    const d = this.data;
    const C = d.C;
    const mode = this.state.colorMode;
    if (mode === 'phase') return phaseColor(d.get(track, i, C.phase), out);
    if (mode === 'altitude') return altColor(d.get(track, i, C.alt_km), out);
    const thr = thresholdAt(d.get(track, i, C.phase), d.meta.thresholds.imm);
    return scoreColor(d.get(track, i, C.imm_score), thr, out);
  }

  _sampleColor(smp, out) {
    const mode = this.state.colorMode;
    if (mode === 'phase') return phaseColor(smp.phase, out);
    if (mode === 'altitude') return altColor(smp.alt, out);
    return scoreColor(smp.score, thresholdAt(smp.phase, this.data.meta.thresholds.imm), out);
  }

  /** The tracks on screen at T: each flight, with the selected scenario's injected copy in place of its flight. */
  _tracksAt(T) {
    const st = this.state;
    const d = this.data;
    const scen = st.scenario >= 0 ? d.scenarios[st.scenario] : null;
    const out = this.active;
    out.length = 0;
    let n = 0;
    for (const f of d.flights) {
      if (T < f.t0 || T > f.t1) continue;
      if (st.onlyRef && !f.ref) continue;
      let track = f;
      if (scen && scen.flight === f.index && T >= scen.t0) {
        if (T > scen.t1) continue;
        track = scen;
      }
      const smp = this._pool[n] || (this._pool[n] = {});
      if (!d.sample(track, T, smp)) continue;
      smp.flight = f.index;
      smp.track = track;
      smp.dim = !!scen && f.index !== scen.flight && !st.scenarioTraffic;
      out.push(smp);
      n++;
    }
    return out;
  }

  aircraftScale() {
    const dist = this.camera.position.distanceTo(this.controls.target);
    return this.state.size * THREE.MathUtils.clamp(dist * 0.0075, 0.06, 7);
  }

  update(T, dtReal) {
    const st = this.state;
    const d = this.data;
    const vex = st.vex;
    this.vgroup.scale.y = vex;
    const tmp = this._tmp;
    const active = this._tracksAt(T);

    // Aircraft.
    const scale = this.aircraftScale();
    const mesh = this.aircraft;
    const hiPos = this.haloImm.geometry.attributes.position.array;
    const hfPos = this.haloFixed.geometry.attributes.position.array;
    let hi = 0;
    let hf = 0;
    const pulse = 0.75 + 0.25 * Math.sin(performance.now() / 180);
    for (let i = 0; i < active.length && i < AIRCRAFT_CAP; i++) {
      const a = active[i];
      const y = a.alt * vex;
      tmp.p.set(a.e, y + 0.02 * scale, -a.n);
      const yaw = Math.PI / 2 - THREE.MathUtils.degToRad(a.heading);
      tmp.q.setFromAxisAngle(THREE.Object3D.DEFAULT_UP, yaw);
      const pitch = THREE.MathUtils.clamp(Math.atan2(Number.isFinite(a.vrate) ? a.vrate : 0, Math.max(a.speed || 0, 60)) * 2.5, -0.5, 0.5);
      tmp.qp.setFromAxisAngle(tmp.v.set(0, 0, 1), pitch);
      tmp.q.multiply(tmp.qp);
      tmp.s.setScalar(scale);
      tmp.m.compose(tmp.p, tmp.q, tmp.s);
      mesh.setMatrixAt(i, tmp.m);
      this._sampleColor(a, tmp.c);
      if (a.dim) fade(tmp.c, 0.78);
      else if (a.coast) fade(tmp.c, 0.45);
      mesh.setColorAt(i, tmp.c);
      a.x = a.e;
      a.y = y;
      a.z = -a.n;
      if (!a.dim) {
        if (a.alert && st.halos) { hiPos[hi * 3] = a.e; hiPos[hi * 3 + 1] = y; hiPos[hi * 3 + 2] = -a.n; hi++; }
        if (a.fixedAlert && st.halos && st.showFixed) { hfPos[hf * 3] = a.e; hfPos[hf * 3 + 1] = y; hfPos[hf * 3 + 2] = -a.n; hf++; }
      }
    }
    mesh.count = Math.min(active.length, AIRCRAFT_CAP);
    mesh.instanceMatrix.needsUpdate = true;
    mesh.instanceColor.needsUpdate = true;
    this.haloImm.geometry.setDrawRange(0, hi);
    this.haloImm.geometry.attributes.position.needsUpdate = true;
    this.haloImm.material.size = 30 * pulse;
    this.haloFixed.geometry.setDrawRange(0, hf);
    this.haloFixed.geometry.attributes.position.needsUpdate = true;

    // Selected aircraft: halo + label.
    const sel = st.selected >= 0 ? active.find((a) => a.flight === st.selected) : null;
    const hsPos = this.haloSel.geometry.attributes.position.array;
    if (sel) {
      hsPos[0] = sel.x; hsPos[1] = sel.y; hsPos[2] = sel.z;
      this.haloSel.geometry.setDrawRange(0, 1);
      this.haloSel.geometry.attributes.position.needsUpdate = true;
      this.selLabel.visible = true;
      this.selLabel.position.set(sel.x, sel.y + scale * 0.6, sel.z);
      this.selLabel.element.textContent = d.flights[sel.flight].id;
      this.selLabel.element.classList.toggle('alert', sel.alert);
    } else {
      this.haloSel.geometry.setDrawRange(0, 0);
      this.selLabel.visible = false;
    }
    this.selected = sel;

    // Ghost of the uninjected flight in scenario mode.
    this._updateGhost(T, vex, scale);

    this._updateTrails(T, active);
    this._updateDrops(active);

    // Follow: carry the camera with the aircraft, keeping the orbit around it.
    if (st.follow && sel) {
      tmp.v.set(sel.x, sel.y, sel.z);
      if (this._lastFollow) {
        const delta = tmp.v.clone().sub(this._lastFollow);
        if (delta.lengthSq() < 400) {
          this.camera.position.add(delta);
          this.controls.target.add(delta);
        } else {
          this.flyTo(tmp.v, Math.min(60, this.camera.position.distanceTo(this.controls.target)));
        }
      }
      this._lastFollow = tmp.v.clone();
    } else {
      this._lastFollow = null;
    }

    this._updateFly(dtReal);
    this._updateLabels();
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this.labelRenderer.render(this.scene, this.camera);
    return active;
  }

  _updateGhost(T, vex, scale) {
    const st = this.state;
    const d = this.data;
    if (st.scenario < 0) { this.ghost.visible = false; return; }
    const s = d.scenarios[st.scenario];
    const f = d.flights[s.flight];
    if (T < s.onset || T > s.t1) { this.ghost.visible = false; return; }
    const smp = this._ghostSample || (this._ghostSample = {});
    if (!d.sample(f, T, smp)) { this.ghost.visible = false; return; }
    this.ghost.visible = true;
    this.ghost.position.set(smp.e, smp.alt * vex, -smp.n);
    this.ghost.quaternion.setFromAxisAngle(THREE.Object3D.DEFAULT_UP, Math.PI / 2 - THREE.MathUtils.degToRad(smp.heading));
    this.ghost.scale.setScalar(scale);
  }

  _updateTrails(T, active) {
    const st = this.state;
    const d = this.data;
    const C = d.C;
    const geo = this.trails.geometry;
    if (!st.trails) { geo.setDrawRange(0, 0); return; }
    const pos = geo.attributes.position.array;
    const col = geo.attributes.color.array;
    const span = st.trailMin * 60;
    const c = this._tmp.c;
    const c2 = this._tmp.c2;
    let n = 0;
    const put = (x0, y0, z0, x1, y1, z1, ca, cb) => {
      const o = n * 6;
      pos[o] = x0; pos[o + 1] = y0; pos[o + 2] = z0; pos[o + 3] = x1; pos[o + 4] = y1; pos[o + 5] = z1;
      col[o] = ca.r; col[o + 1] = ca.g; col[o + 2] = ca.b; col[o + 3] = cb.r; col[o + 4] = cb.g; col[o + 5] = cb.b;
      n++;
    };
    for (const a of active) {
      if (n >= TRAIL_CAP - 1) break;
      const tr = a.track;
      const dimBase = a.dim ? 0.8 : 0;
      // Head: from the last fix to the interpolated position.
      let k = a.k;
      const R = d.rows;
      const S = d.S;
      let o = (tr.offset + k) * S;
      this._sampleColor(a, c);
      fade(c, dimBase);
      c2.copy(c);
      put(R[o + C.east_km], R[o + C.alt_km], -R[o + C.north_km], a.e, a.alt, -a.n, c2, c);
      let prevAgeColor = c2.clone();
      while (k > 0 && n < TRAIL_CAP) {
        const ob = (tr.offset + k - 1) * S;
        const tb = R[ob];
        if (T - tb > span) break;
        if (R[o] - tb > GAP_S) break;
        this._rowColor(tr, k - 1, c);
        const age = (T - tb) / span;
        fade(c, Math.min(0.95, dimBase + (1 - dimBase) * age * 0.9));
        put(R[ob + C.east_km], R[ob + C.alt_km], -R[ob + C.north_km], R[o + C.east_km], R[o + C.alt_km], -R[o + C.north_km], c, prevAgeColor);
        prevAgeColor.copy(c);
        k--;
        o = ob;
      }
    }
    geo.setDrawRange(0, n * 2);
    geo.attributes.position.needsUpdate = true;
    geo.attributes.color.needsUpdate = true;
  }

  _updateDrops(active) {
    const geo = this.drops.geometry;
    if (!this.state.drops) { geo.setDrawRange(0, 0); return; }
    const pos = geo.attributes.position.array;
    let n = 0;
    for (const a of active) {
      if (n >= DROP_CAP || a.dim) continue;
      const o = n * 6;
      pos[o] = a.e; pos[o + 1] = a.alt; pos[o + 2] = -a.n; pos[o + 3] = a.e; pos[o + 4] = 0; pos[o + 5] = -a.n;
      n++;
    }
    geo.setDrawRange(0, n * 2);
    geo.attributes.position.needsUpdate = true;
  }

  _updateLabels() {
    const dist = this.camera.position.distanceTo(this.controls.target);
    for (const l of this.cityLabels) l.visible = this.state.layers.cities && (l.userData.rank <= 6 || dist < 320);
  }

  // ---------- selection, scenario ----------

  _line(positions, colors, width, opts = {}) {
    const geo = new LineGeometry();
    geo.setPositions(positions);
    if (colors) geo.setColors(colors);
    const mat = new LineMaterial({ color: opts.color ?? 0xffffff, linewidth: width, vertexColors: !!colors, transparent: true,
      opacity: opts.opacity ?? 1, dashed: !!opts.dashed, dashSize: opts.dash ?? 2, gapSize: opts.gap ?? 1.5, depthWrite: !opts.transparent });
    mat.resolution.set(this.renderer.domElement.width, this.renderer.domElement.height);
    this.lineMaterials.push(mat);
    const line = new Line2(geo, mat);
    if (opts.dashed) line.computeLineDistances();
    return line;
  }

  _clear(group) {
    for (const child of [...group.children]) {
      group.remove(child);
      child.traverse((o) => {
        o.geometry?.dispose();
        if (o.material) {
          this.lineMaterials = this.lineMaterials.filter((m) => m !== o.material);
          o.material.dispose();
        }
        if (o.isCSS2DObject) o.element.remove();
      });
    }
  }

  _trackPath(track, from = 0, to = track.rows) {
    const d = this.data;
    const C = d.C;
    const pos = [];
    const col = [];
    const c = this._tmp.c;
    for (let i = from; i < to; i++) {
      pos.push(d.get(track, i, C.east_km), d.get(track, i, C.alt_km), -d.get(track, i, C.north_km));
      this._rowColor(track, i, c);
      col.push(c.r, c.g, c.b);
    }
    return { pos, col };
  }

  _corridor(ref, group) {
    const n = ref.east_km.length;
    const left = [];
    const right = [];
    const centre = [];
    for (let i = 0; i < n; i++) {
      const i0 = Math.max(0, i - 1);
      const i1 = Math.min(n - 1, i + 1);
      let dx = ref.east_km[i1] - ref.east_km[i0];
      let dy = ref.north_km[i1] - ref.north_km[i0];
      const len = Math.hypot(dx, dy) || 1;
      dx /= len; dy /= len;
      // Two robust sd either side: the range most conforming traffic flew at this station.
      const w = Math.max(0.15, 2 * ref.cross_sd_km[i]);
      const x = ref.east_km[i];
      const y = ref.north_km[i];
      const z = ref.alt_km[i];
      left.push([x - dy * w, z, -(y + dx * w)]);
      right.push([x + dy * w, z, -(y - dx * w)]);
      centre.push(x, z, -y);
    }
    const pos = [];
    const idx = [];
    for (let i = 0; i < n; i++) pos.push(...left[i], ...right[i]);
    for (let i = 0; i + 1 < n; i++) {
      const a = 2 * i;
      idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
    }
    const ribbonGeo = new THREE.BufferGeometry();
    ribbonGeo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    ribbonGeo.setIndex(idx);
    const ribbon = new THREE.Mesh(ribbonGeo, new THREE.MeshBasicMaterial({ color: PAL.ref, transparent: true, opacity: 0.16, side: THREE.DoubleSide, depthWrite: false }));
    group.add(ribbon);
    // Curtain from the centreline to the ground, so the path reads in 3D.
    const cpos = [];
    const cidx = [];
    for (let i = 0; i < n; i++) cpos.push(centre[3 * i], centre[3 * i + 1], centre[3 * i + 2], centre[3 * i], 0, centre[3 * i + 2]);
    for (let i = 0; i + 1 < n; i++) {
      const a = 2 * i;
      cidx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
    }
    const curtainGeo = new THREE.BufferGeometry();
    curtainGeo.setAttribute('position', new THREE.Float32BufferAttribute(cpos, 3));
    curtainGeo.setIndex(cidx);
    group.add(new THREE.Mesh(curtainGeo, new THREE.MeshBasicMaterial({ color: PAL.ref, transparent: true, opacity: 0.05, side: THREE.DoubleSide, depthWrite: false })));
    group.add(this._line(centre, null, 1.6, { color: PAL.ref, opacity: 0.9, dashed: true, dash: 3, gap: 2 }));
    group.add(this._line(left.flat(), null, 1, { color: PAL.ref, opacity: 0.45 }));
    group.add(this._line(right.flat(), null, 1, { color: PAL.ref, opacity: 0.45 }));
  }

  showSelection(flightIndex) {
    this._clear(this.selection);
    if (flightIndex < 0) return;
    const d = this.data;
    const f = d.flights[flightIndex];
    if (f.ref && this.state.layers.corridor) this._corridor(d.refByKey.get(f.ref), this.selection);
    if (this.state.scenario >= 0 && d.scenarios[this.state.scenario].flight === flightIndex) return;
    const { pos, col } = this._trackPath(f);
    if (pos.length >= 6) this.selection.add(this._line(pos, col, 2.6));
    // Shadow on the ground.
    const shadow = [];
    for (let i = 0; i < pos.length; i += 3) shadow.push(pos[i], 0.03, pos[i + 2]);
    if (shadow.length >= 6) this.selection.add(this._line(shadow, null, 1, { color: '#8fa3c7', opacity: 0.35 }));
  }

  showScenario(index) {
    this._clear(this.scenarioGroup);
    if (index < 0) return;
    const d = this.data;
    const s = d.scenarios[index];
    const f = d.flights[s.flight];
    const C = d.C;
    // Uninjected flight over the same window, ghosted; the injected copy coloured by the mode.
    const i0 = Math.max(0, d.find(f, s.t0));
    const i1 = d.find(f, s.t1) + 1;
    const orig = this._trackPath(f, i0, i1);
    if (orig.pos.length >= 6) this.scenarioGroup.add(this._line(orig.pos, null, 1.5, { color: '#e8eefc', opacity: 0.55, dashed: true, dash: 1.2, gap: 1 }));
    const inj = this._trackPath(s);
    if (inj.pos.length >= 6) this.scenarioGroup.add(this._line(inj.pos, inj.col, 3));
    // Before onset: the flight's own track from its start, so the approach to the event is visible.
    const before = this._trackPath(f, 0, i0 + 1);
    if (before.pos.length >= 6) this.scenarioGroup.add(this._line(before.pos, before.col, 1.6, { opacity: 0.7 }));

    const marker = (t, text, cls) => {
      const k = d.find(s, t);
      if (k < 0) return;
      const l = label(text, `marker ${cls}`);
      l.position.set(d.get(s, k, C.east_km), d.get(s, k, C.alt_km), -d.get(s, k, C.north_km));
      this.scenarioGroup.add(l);
    };
    marker(s.onset, 'onset', 'm-onset');
    if (s.t_violation !== null) marker(s.onset + s.t_violation, `violation +${Math.round(s.t_violation)} s`, 'm-violation');
    if (s.imm_first !== null) marker(s.onset + s.imm_first, `IMM alert +${Math.round(s.imm_first)} s`, 'm-imm');
    if (s.fixed_first !== null && this.state.showFixed) marker(s.onset + s.fixed_first, `fixed alert +${Math.round(s.fixed_first)} s`, 'm-fixed');
  }

  // ---------- camera ----------

  flyTo(target, distance, durationS = 0.9, direction = null) {
    const dir = direction ? direction.clone() : this.camera.position.clone().sub(this.controls.target);
    if (dir.lengthSq() < 1e-6) dir.set(0, 1, 1);
    dir.normalize().multiplyScalar(distance ?? this.camera.position.distanceTo(this.controls.target));
    this._fly = {
      t: 0, dur: durationS,
      fromT: this.controls.target.clone(), toT: target.clone(),
      fromP: this.camera.position.clone(), toP: target.clone().add(dir),
    };
  }

  _updateFly(dt) {
    const f = this._fly;
    if (!f) return;
    f.t = Math.min(1, f.t + dt / f.dur);
    const e = f.t < 0.5 ? 4 * f.t ** 3 : 1 - (-2 * f.t + 2) ** 3 / 2;
    this.controls.target.lerpVectors(f.fromT, f.toT, e);
    this.camera.position.lerpVectors(f.fromP, f.toP, e);
    if (f.t >= 1) this._fly = null;
  }

  /** Behind and above the aircraft, looking along its track. */
  chase(a, distance, elevationDeg = 28) {
    const psi = THREE.MathUtils.degToRad(a.heading);
    const el = THREE.MathUtils.degToRad(elevationDeg);
    const dir = new THREE.Vector3(-Math.sin(psi) * Math.cos(el), Math.sin(el), Math.cos(psi) * Math.cos(el));
    this.flyTo(new THREE.Vector3(a.x, a.y, a.z), distance, 1.0, dir);
  }

  home() {
    this._fly = { t: 0, dur: 1.1, fromT: this.controls.target.clone(), toT: new THREE.Vector3(0, 0, 0),
      fromP: this.camera.position.clone(), toP: new THREE.Vector3(-110, 205, 470) };
  }

  /** The active aircraft nearest the pointer within `px` pixels. */
  pick(clientX, clientY, px = 16) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const v = this._tmp.v;
    let best = null;
    let bestD = px * px;
    for (const a of this.active) {
      if (a.dim) continue;
      v.set(a.x, a.y, a.z).project(this.camera);
      if (v.z > 1) continue;
      const sx = rect.left + ((v.x + 1) / 2) * rect.width;
      const sy = rect.top + ((1 - v.y) / 2) * rect.height;
      const dd = (sx - clientX) ** 2 + (sy - clientY) ** 2;
      if (dd < bestD) { bestD = dd; best = a; }
    }
    return best;
  }

  setLayer(name, on) {
    const obj = this.layers[name];
    if (obj) obj.visible = on;
  }

  resize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
    this.labelRenderer.setSize(w, h);
    for (const m of this.lineMaterials) m.resolution.set(w, h);
  }
}
