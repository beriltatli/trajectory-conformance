import * as THREE from 'three';
import { loadScene } from './data.js';
import { World } from './world.js';
import { UI } from './ui.js';

// Served from the repository root (make sim), so the export in data/sim/ is one level up.
const DATA = '../data/sim/';

const state = {
  T: 0,
  playing: true,
  speed: 30,
  colorMode: 'conformance',
  vex: 6,
  size: 1,
  trailMin: 6,
  trails: true,
  drops: false,
  halos: true,
  showFixed: true,
  onlyRef: false,
  follow: false,
  selected: -1,
  scenario: -1,
  scenarioTraffic: false,
  tab: 'flight',
  layers: { borders: true, states: true, rivers: true, cities: true, airports: true, references: false, corridor: true },
};

const progress = document.querySelector('#loading-bar');
const loadingText = document.querySelector('#loading-text');

async function start() {
  let data;
  try {
    data = await loadScene(DATA, (got, total) => {
      progress.style.width = `${(100 * got) / total}%`;
      loadingText.textContent = `Loading traffic · ${(got / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB`;
    });
  } catch (err) {
    loadingText.textContent = String(err.message || err);
    document.querySelector('#loading').classList.add('error');
    return;
  }
  loadingText.textContent = 'Building scene';
  await new Promise((r) => requestAnimationFrame(r));

  // Start mid-afternoon, when the sample has the most traffic in the air.
  state.T = data.tMin + 2.5 * 3600;
  const world = new World(document.querySelector('#viewport'), data, state);

  const actions = {
    togglePlay() { state.playing = !state.playing; },
    seek(t) { state.T = Math.min(data.tMax, Math.max(data.tMin, t)); ui.update(state.T, world.active, true); },
    restyle() { world.showSelection(state.selected); world.showScenario(state.scenario); ui.refreshCharts(); },
    setFollow(on) {
      state.follow = on && state.selected >= 0;
      if (state.follow) actions.focusSelected(40);
    },
    focusSelected(distance = 60) {
      const a = world.active.find((x) => x.flight === state.selected);
      if (a) world.chase(a, distance);
    },
    select(index, opts = {}) {
      if (state.scenario >= 0 && index !== data.scenarios[state.scenario].flight) actions.scenario(-1, false);
      state.selected = index;
      if (index < 0) state.follow = false;
      world.showSelection(index);
      ui.refreshCharts();
      if (index >= 0) {
        ui.showTab(state.tab === 'events' ? 'events' : state.tab === 'alerts' ? 'alerts' : 'flight');
        const f = data.flights[index];
        if (opts.seek && (state.T < f.t0 || state.T > f.t1)) state.T = f.t0 + Math.min(600, (f.t1 - f.t0) / 2);
        if (opts.focus) requestAnimationFrame(() => actions.focusSelected());
      }
      ui.update(state.T, world.active, true);
    },
    scenario(index, reselect = true) {
      state.scenario = index;
      world.showScenario(index);
      ui.scenarioBanner(index >= 0);
      if (index >= 0) {
        const s = data.scenarios[index];
        state.T = s.onset - 90;
        state.speed = Math.min(state.speed, 10);
        document.querySelector('#speed').value = String(state.speed);
        state.playing = true;
        state.selected = s.flight;
        world.showSelection(s.flight);
        ui.refreshCharts();
        ui.showTab('flight');
        // Let the frame place the aircraft, then fly to it and follow.
        requestAnimationFrame(() => requestAnimationFrame(() => { state.follow = true; actions.focusSelected(45); }));
      } else if (reselect) {
        world.showSelection(state.selected);
        ui.refreshCharts();
      }
      ui.update(state.T, world.active, true);
    },
  };

  const ui = new UI(data, world, state, actions);
  // Console handle for inspection: sim.state, sim.world, sim.data.
  window.sim = { state, world, data, actions };
  ui.showTab('flight');
  document.querySelector('#loading').remove();

  let last = performance.now();
  function frame(now) {
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    if (state.playing) {
      state.T += dt * state.speed;
      if (state.T >= data.tMax) { state.T = data.tMax; state.playing = false; }
    }
    const active = world.update(state.T, dt);
    ui.update(state.T, active);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

start();
