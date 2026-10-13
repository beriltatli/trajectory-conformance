import * as THREE from 'three';

// One palette for the 3D view and the panels (style.css repeats the hex values as tokens).
export const PAL = {
  bg: '#070b14',
  ground: '#0b1220',
  region: '#0f1a2c',
  grid: '#152238',
  border: '#5d6f8f',
  state: '#22324b',
  river: '#1d5b86',
  noRef: '#7f8a9c',
  climb: '#4cc38a',
  cruise: '#58a6ff',
  descent: '#f0a35e',
  unknown: '#8a8f98',
  imm: '#ff4d5e',
  fixed: '#b48cff',
  ref: '#3fa7ff',
  onset: '#ffd23f',
  violation: '#ff8a3d',
};

export const PHASES = ['climb', 'cruise', 'descent'];
export const PHASE_LABEL = { '-1': 'unknown', 0: 'climb', 1: 'cruise', 2: 'descent' };

const _bg = new THREE.Color(PAL.bg);

function ramp(stops) {
  const cols = stops.map(([x, c]) => [x, new THREE.Color(c)]);
  return (x, out) => {
    if (x <= cols[0][0]) return out.copy(cols[0][1]);
    for (let i = 1; i < cols.length; i++) {
      if (x <= cols[i][0]) {
        const [x0, c0] = cols[i - 1];
        const [x1, c1] = cols[i];
        return out.copy(c0).lerp(c1, (x - x0) / (x1 - x0));
      }
    }
    return out.copy(cols[cols.length - 1][1]);
  };
}

// Score as a share of the phase's budget threshold: quiet teal, amber at the threshold, red beyond.
const scoreRamp = ramp([[0, '#1f9e8f'], [0.55, '#7cc96a'], [0.85, '#e6d84a'], [1.0, '#ffb63d'], [1.4, '#ff7a3d'], [2.2, '#ff3b4f']]);
const altRamp = ramp([[0, '#3d4fb8'], [2, '#3a8fd1'], [5, '#36c2b4'], [8, '#9bd65a'], [11, '#f6d743'], [13, '#ff9d5c']]);

export function thresholdAt(phase, perPhase) {
  return phase >= 0 && phase <= 2 ? perPhase[phase] : Math.max(perPhase[0], perPhase[1], perPhase[2]);
}

export function scoreColor(score, thr, out) {
  if (!Number.isFinite(score)) return out.set(PAL.noRef);
  return scoreRamp(score / thr, out);
}

export function phaseColor(phase, out) {
  return out.set(phase === 0 ? PAL.climb : phase === 1 ? PAL.cruise : phase === 2 ? PAL.descent : PAL.unknown);
}

export function altColor(altKm, out) {
  return altRamp(altKm, out);
}

export function fade(color, amount) {
  return color.lerp(_bg, amount);
}

export function css(color) {
  return `#${color.getHexString()}`;
}

export const SCORE_LEGEND = [[0, '#1f9e8f'], [0.55, '#7cc96a'], [0.85, '#e6d84a'], [1.0, '#ffb63d'], [1.4, '#ff7a3d'], [2.2, '#ff3b4f']];
export const ALT_LEGEND = [[0, '#3d4fb8'], [2, '#3a8fd1'], [5, '#36c2b4'], [8, '#9bd65a'], [11, '#f6d743'], [13, '#ff9d5c']];
