/* eslint-disable no-console */

const NODE_STYLE = {
  control: { fill: '#2563eb', stroke: '#1d4ed8' },
  goals: { fill: '#0ea5e9', stroke: '#0284c7' },
  selector: { fill: '#7c3aed', stroke: '#5b21b6' },
  phase: { fill: '#f97316', stroke: '#ea580c' },
  router: { fill: '#facc15', stroke: '#eab308' },
  plan_group: { fill: '#22c55e', stroke: '#15803d' },
  features: { fill: '#14b8a6', stroke: '#0f766e' },
  feature_group: { fill: '#2dd4bf', stroke: '#0d9488' },
  synth: { fill: '#ef4444', stroke: '#b91c1c' },
  eval: { fill: '#94a3b8', stroke: '#64748b' },
  trainer: { fill: '#f472b6', stroke: '#db2777' },
  subgraph: { fill: '#9ca3af', stroke: '#4b5563' }
};

const EDGE_COLORS = {
  sub: '#1d4ed8',
  request: '#14b8a6',
  confirm: '#16a34a',
  feature: '#0f766e',
  eval: '#dc2626',
  gate: '#f97316',
  tune: '#7c3aed',
  goal: '#0ea5e9'
};

const DEFAULT_SPEC_PATH = 'sample_data/macrograph_spec_v0.json';
const DEFAULT_TIMELINE_PATH = 'sample_data/macrograph_demo.json';
const FRAME_INTERVAL_MS = 1600;

let canvasEl;
let infoPanelEl;
let sliderEl;
let frameLabelEl;
let playButtonEl;

let currentSpec = null;
let currentPositions = {};
let currentFrames = [];
let currentFrameIndex = 0;
let playHandle = null;

function clamp01(value) {
  return Math.min(1, Math.max(0, Number.isFinite(value) ? value : 0));
}

function hexToRgb(hex) {
  if (!hex || typeof hex !== 'string') return null;
  const normalized = hex.replace('#', '');
  const value = parseInt(normalized, 16);
  if (Number.isNaN(value)) return null;
  if (normalized.length === 3) {
    const r = (value >> 8) & 0xf;
    const g = (value >> 4) & 0xf;
    const b = value & 0xf;
    return {
      r: (r << 4) | r,
      g: (g << 4) | g,
      b: (b << 4) | b
    };
  }
  if (normalized.length === 6) {
    return {
      r: (value >> 16) & 0xff,
      g: (value >> 8) & 0xff,
      b: value & 0xff
    };
  }
  return null;
}

function lightenColor(hex, factor) {
  const rgb = hexToRgb(hex);
  if (!rgb) return hex;
  const f = clamp01(factor);
  const r = Math.round(rgb.r + (255 - rgb.r) * f);
  const g = Math.round(rgb.g + (255 - rgb.g) * f);
  const b = Math.round(rgb.b + (255 - rgb.b) * f);
  return `rgb(${r}, ${g}, ${b})`;
}

function resizeCanvas(canvas) {
  const parent = canvas.parentElement;
  const controls = document.getElementById('timeline-controls');
  const controlsHeight = controls ? controls.offsetHeight : 0;
  const width = parent.clientWidth;
  const height = Math.max(720, window.innerHeight - parent.getBoundingClientRect().top - controlsHeight - 60);
  canvas.width = width;
  canvas.height = height;
}

function layoutPositions(nodes) {
  const order = [
    ['GameControl'],
    ['GoalVector', 'OutcomeMode', 'LearningSupervisor'],
    ['PhaseLayer'],
    ['PlanHub', 'FeatureHub'],
    ['PlanOpening', 'PlanMiddlegame', 'PlanEndgame'],
    ['FeatureTactics', 'FeatureStructure', 'FeatureEndgame'],
    ['KRKSubgraph', 'KPKSubgraph', 'RookTechniquesSubgraph'],
    ['MoveSynth', 'LightEval']
  ];
  const map = {};
  const width = 1200;
  const baseX = 80;
  const levelHeight = 130;

  order.forEach((level, idx) => {
    if (!level) return;
    const gap = width / (level.length + 1);
    level.forEach((nid, posIdx) => {
      map[nid] = {
        x: baseX + gap * (posIdx + 1),
        y: 90 + levelHeight * idx
      };
    });
  });

  nodes.forEach((node) => {
    if (!map[node.id]) {
      const count = Object.keys(map).length;
      const col = count % 5;
      const row = Math.floor(count / 5);
      map[node.id] = {
        x: baseX + 200 * (col + 1),
        y: 90 + levelHeight * (order.length + row)
      };
    }
  });

  return map;
}

function drawNode(ctx, node, position, intensity = 0) {
  const style = NODE_STYLE[node.type] || { fill: '#e2e8f0', stroke: '#475569' };
  const fill = intensity > 0 ? lightenColor(style.fill, 0.45 * intensity + 0.1) : style.fill;
  const stroke = intensity > 0 ? lightenColor(style.stroke, 0.3 * intensity) : style.stroke;
  const radius = 34 + intensity * 12;

  ctx.fillStyle = fill;
  ctx.strokeStyle = stroke;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(position.x, position.y, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = '#0f172a';
  ctx.font = intensity > 0.7 ? 'bold 14px Segoe UI, Arial' : '14px Segoe UI, Arial';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(node.id, position.x, position.y);
}

function drawEdge(ctx, edge, positions, highlights, maxWeight = 1) {
  const src = positions[edge.from];
  const dst = positions[edge.to];
  if (!src || !dst) return;
  const baseColor = EDGE_COLORS[edge.kind] || '#64748b';
  const intensity = Math.max(highlights[edge.from] || 0, highlights[edge.to] || 0);
  const color = intensity > 0 ? lightenColor(baseColor, 0.5 * intensity) : baseColor;
  const norm = clamp01(maxWeight > 0 && Number.isFinite(edge.weight) ? Math.abs(edge.weight) / maxWeight : 1);
  const widthScale = 0.7 + 0.9 * norm;

  ctx.strokeStyle = color;
  ctx.lineWidth = (2 + intensity * 3) * widthScale;
  ctx.globalAlpha = 0.65 + intensity * 0.35;
  ctx.beginPath();
  ctx.moveTo(src.x, src.y);
  ctx.lineTo(dst.x, dst.y);
  ctx.stroke();
  ctx.globalAlpha = 1;

  ctx.fillStyle = color;
  ctx.font = '12px Segoe UI, Arial';
  ctx.textAlign = 'center';
  ctx.fillText(edge.kind, (src.x + dst.x) / 2, (src.y + dst.y) / 2 - 6);
}

function computeHighlights(frame) {
  if (!frame || !frame.macro_frame) return {};
  const highlight = {};
  const macro = frame.macro_frame;

  const planGroups = Array.isArray(macro.plan_groups) ? macro.plan_groups : [];
  let maxPlan = 0;
  planGroups.forEach((group) => {
    const activation = clamp01(group.activation ?? 0);
    highlight[group.id] = Math.max(highlight[group.id] || 0, activation);
    maxPlan = Math.max(maxPlan, activation);
    if (group.id === 'PlanEndgame') {
      const plans = Array.isArray(group.plans) ? group.plans : [];
      plans.forEach((plan) => {
        if (plan === 'KRK') highlight.KRKSubgraph = Math.max(highlight.KRKSubgraph || 0, 0.9);
        if (plan === 'KPK') highlight.KPKSubgraph = Math.max(highlight.KPKSubgraph || 0, 0.9);
        if (plan === 'RookTechniques') highlight.RookTechniquesSubgraph = Math.max(highlight.RookTechniquesSubgraph || 0, 0.85);
      });
    }
  });
  if (maxPlan > 0) {
    highlight.PlanHub = Math.max(highlight.PlanHub || 0, maxPlan);
  }

  const featureGroups = Array.isArray(macro.feature_groups) ? macro.feature_groups : [];
  let maxFeature = 0;
  featureGroups.forEach((group) => {
    const confidence = clamp01(group.confidence ?? 0);
    highlight[group.id] = Math.max(highlight[group.id] || 0, confidence);
    maxFeature = Math.max(maxFeature, confidence);
  });
  if (maxFeature > 0) {
    highlight.FeatureHub = Math.max(highlight.FeatureHub || 0, maxFeature);
  }

  const phaseMix = macro.phase_mix || {};
  const phaseWeights = Object.values(phaseMix).map((value) => clamp01(value));
  if (phaseWeights.length) {
    highlight.PhaseLayer = Math.max(highlight.PhaseLayer || 0, Math.max(...phaseWeights));
  }

  const goalVector = macro.goal_vector || {};
  const goalValues = Object.values(goalVector).map((value) => clamp01(value));
  if (goalValues.length) {
    const maxGoal = Math.max(...goalValues);
    highlight.GoalVector = Math.max(highlight.GoalVector || 0, maxGoal);
    highlight.GameControl = Math.max(highlight.GameControl || 0, maxGoal * 0.6);
  }

  const moveSynth = macro.move_synth || {};
  if (moveSynth.chosen) {
    highlight.MoveSynth = Math.max(highlight.MoveSynth || 0, 1);
  }

  const bindings = macro.bindings || {};
  if (bindings['macro/endgame/krk']) {
    highlight.KRKSubgraph = Math.max(highlight.KRKSubgraph || 0, 1);
  }
  if (bindings['macro/endgame/kpk']) {
    highlight.KPKSubgraph = Math.max(highlight.KPKSubgraph || 0, 1);
  }

  return highlight;
}

function updateInfoPanel(frame) {
  if (!infoPanelEl) return;
  const lines = [];
  if (currentSpec) {
    const nodes = Array.isArray(currentSpec.nodes) ? currentSpec.nodes.length : 0;
    const edges = Array.isArray(currentSpec.edges) ? currentSpec.edges.length : 0;
    lines.push(`Spec version: ${currentSpec.version || 'unknown'}`);
    lines.push(`Nodes: ${nodes} · Edges: ${edges}`);
  }

  if (!frame) {
    lines.push('', 'Load a timeline to see macro-frame activity.');
    infoPanelEl.textContent = lines.join('\n');
    return;
  }

  const macro = frame.macro_frame || {};
  lines.push('', `Frame ${frame.tick ?? currentFrameIndex}: ${frame.label || '\u2014'}`);
  if (frame.board_fen) {
    lines.push(`Board: ${frame.board_fen}`);
  }

  if (macro.phase_mix) {
    lines.push('');
    lines.push('Phase mix:');
    Object.entries(macro.phase_mix).forEach(([phase, weight]) => {
      lines.push(` • ${phase}: ${(clamp01(weight) * 100).toFixed(0)}%`);
    });
  }

  const planGroups = Array.isArray(macro.plan_groups) ? macro.plan_groups : [];
  if (planGroups.length) {
    lines.push('');
    lines.push('Plan groups:');
    planGroups.forEach((group) => {
      lines.push(` • ${group.id}: ${(clamp01(group.activation ?? 0) * 100).toFixed(0)}%`);
    });
  }

  const featureGroups = Array.isArray(macro.feature_groups) ? macro.feature_groups : [];
  if (featureGroups.length) {
    lines.push('');
    lines.push('Feature groups:');
    featureGroups.forEach((group) => {
      lines.push(` • ${group.id}: ${(clamp01(group.confidence ?? 0) * 100).toFixed(0)}%`);
    });
  }

  if (macro.move_synth && macro.move_synth.chosen) {
    const chosen = macro.move_synth.chosen;
    const components = macro.move_synth.components || macro.move_synth.proposals?.[0]?.components || {};
    lines.push('');
    lines.push('Move synthesis:');
    lines.push(` • Chosen move: ${chosen}`);
    if (components && Object.keys(components).length) {
      const parts = Object.entries(components)
        .map(([key, value]) => `${key} ${(clamp01(value) * 100).toFixed(0)}%`)
        .join(', ');
      lines.push(` • Components: ${parts}`);
    }
  }

  infoPanelEl.textContent = lines.join('\n');
}

function updateFrameLabel(frame) {
  if (!frameLabelEl) return;
  if (!currentFrames.length) {
    frameLabelEl.textContent = 'No frames loaded';
    return;
  }
  const note = frame && frame.label ? frame.label : '—';
  frameLabelEl.textContent = `Frame ${currentFrameIndex + 1}/${currentFrames.length} – ${note}`;
}

function drawMacrographFrame() {
  if (!canvasEl || !currentSpec) return;
  resizeCanvas(canvasEl);
  const ctx = canvasEl.getContext('2d');
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

  const nodes = currentSpec.nodes || [];
  const edges = currentSpec.edges || [];
  const frame = currentFrames[currentFrameIndex] || null;
  const highlights = computeHighlights(frame);
  const maxWeight = edges.reduce((acc, edge) => {
    const val = Number.isFinite(edge.weight) ? Math.abs(edge.weight) : NaN;
    return Number.isFinite(val) && val > acc ? val : acc;
  }, 0) || 1;

  edges.forEach((edge) => drawEdge(ctx, edge, currentPositions, highlights, maxWeight));
  nodes.forEach((node) => {
    const position = currentPositions[node.id];
    if (!position) return;
    const intensity = highlights[node.id] || 0;
    drawNode(ctx, node, position, intensity);
  });

  updateInfoPanel(frame);
  updateFrameLabel(frame);
}

async function loadMacrographSpec(path) {
  try {
    const response = await fetch(path, { cache: 'no-store' });
    if (!response.ok) throw new Error(`Failed to load macrograph spec: ${response.status}`);
    const spec = await response.json();
    currentSpec = spec;
    const nodes = Array.isArray(spec.nodes) ? spec.nodes : [];
    currentPositions = layoutPositions(nodes);
    drawMacrographFrame();
  } catch (error) {
    console.error(error);
    alert(error.message);
  }
}

async function loadTimeline(path) {
  stopPlayback();
  try {
    const stamped = path.includes('?') ? `${path}&t=${Date.now()}` : `${path}?t=${Date.now()}`;
    const response = await fetch(stamped, { cache: 'no-store' });
    if (!response.ok) throw new Error(`Failed to load timeline: ${response.status}`);
    const data = await response.json();
    if (!Array.isArray(data)) throw new Error('Timeline JSON must be an array of frames.');
    currentFrames = data;
    currentFrameIndex = 0;
  } catch (error) {
    console.error(error);
    currentFrames = [];
    currentFrameIndex = 0;
    alert(error.message || 'Failed to load timeline data.');
  }
  updateSliderState();
  updatePlayControlState();
  drawMacrographFrame();
}

function handleSpecFile(event) {
  const [file] = event.target.files || [];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    try {
      const spec = JSON.parse(e.target.result);
      currentSpec = spec;
      const nodes = Array.isArray(spec.nodes) ? spec.nodes : [];
      currentPositions = layoutPositions(nodes);
      drawMacrographFrame();
    } catch (err) {
      console.error(err);
      alert('Invalid macrograph JSON file.');
    }
  };
  reader.readAsText(file);
}

function handleTimelineFile(event) {
  const [file] = event.target.files || [];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    try {
      const data = JSON.parse(e.target.result);
      if (!Array.isArray(data)) throw new Error('Timeline JSON must be an array.');
      stopPlayback();
      currentFrames = data;
      currentFrameIndex = 0;
      updateSliderState();
      updatePlayControlState();
      drawMacrographFrame();
    } catch (err) {
      console.error(err);
      alert('Invalid timeline JSON file.');
    }
  };
  reader.readAsText(file);
}

function setFrame(index) {
  if (!currentFrames.length) {
    currentFrameIndex = 0;
    drawMacrographFrame();
    return;
  }
  const clamped = Math.max(0, Math.min(index, currentFrames.length - 1));
  currentFrameIndex = clamped;
  if (sliderEl) {
    sliderEl.value = String(clamped);
  }
  drawMacrographFrame();
}

function updateSliderState() {
  if (!sliderEl) return;
  if (!currentFrames.length) {
    sliderEl.disabled = true;
    sliderEl.min = 0;
    sliderEl.max = 0;
    sliderEl.value = 0;
  } else {
    sliderEl.disabled = false;
    sliderEl.min = 0;
    sliderEl.max = Math.max(currentFrames.length - 1, 0);
    sliderEl.value = String(currentFrameIndex);
  }
}

function updatePlayControlState() {
  if (!playButtonEl) return;
  const playable = currentFrames.length > 1;
  playButtonEl.disabled = !playable;
  playButtonEl.textContent = playHandle ? 'Pause' : 'Play';
}

function stopPlayback() {
  if (playHandle) {
    clearInterval(playHandle);
    playHandle = null;
    updatePlayControlState();
  }
}

function startPlayback() {
  if (playHandle || currentFrames.length <= 1) return;
  playHandle = setInterval(() => {
    const next = (currentFrameIndex + 1) % currentFrames.length;
    setFrame(next);
  }, FRAME_INTERVAL_MS);
  updatePlayControlState();
}

function togglePlayback() {
  if (playHandle) {
    stopPlayback();
  } else {
    startPlayback();
  }
}

function onSliderInput(event) {
  const value = Number.parseInt(event.target.value, 10);
  if (Number.isNaN(value)) return;
  stopPlayback();
  setFrame(value);
}

function onResize() {
  if (!canvasEl) return;
  drawMacrographFrame();
}

async function initialiseViewer() {
  canvasEl = document.getElementById('macrograph-canvas');
  infoPanelEl = document.getElementById('macrograph-info');
  sliderEl = document.getElementById('frame-slider');
  frameLabelEl = document.getElementById('frame-label');
  playButtonEl = document.getElementById('play-btn');

  const specBtn = document.getElementById('load-spec-btn');
  const specFile = document.getElementById('spec-file');
  const timelineBtn = document.getElementById('load-timeline-btn');
  const timelineFile = document.getElementById('timeline-file');

  specBtn?.addEventListener('click', () => loadMacrographSpec(DEFAULT_SPEC_PATH));
  specFile?.addEventListener('change', handleSpecFile);
  timelineBtn?.addEventListener('click', () => loadTimeline(DEFAULT_TIMELINE_PATH));
  timelineFile?.addEventListener('change', handleTimelineFile);
  playButtonEl?.addEventListener('click', togglePlayback);
  sliderEl?.addEventListener('input', onSliderInput);
  window.addEventListener('resize', onResize);
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      stopPlayback();
    }
  });

  await loadMacrographSpec(DEFAULT_SPEC_PATH);
  await loadTimeline(DEFAULT_TIMELINE_PATH);
}

document.addEventListener('DOMContentLoaded', initialiseViewer);
