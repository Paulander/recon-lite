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

function resizeCanvas(canvas) {
  const parent = canvas.parentElement;
  canvas.width = parent.clientWidth - 40;
  canvas.height = Math.max(720, window.innerHeight - 160);
}

function layoutPositions(nodes) {
  const order = [
    ['GameControl'],
    ['GoalVector', 'OutcomeMode', 'LearningSupervisor'],
    ['PhaseLayer'],
    ['PlanHub', 'FeatureHub'],
    ['PlanOpening', 'PlanMiddlegame', 'PlanEndgame'],
    ['FeatureTactics', 'FeatureStructure', 'FeatureEndgame'],
    ['KRKSubgraph'],
    ['MoveSynth', 'LightEval']
  ];
  const map = {};
  const width = 1200;
  const baseX = 80;
  const levelHeight = 120;

  order.forEach((level, idx) => {
    if (!level) return;
    const gap = width / (level.length + 1);
    level.forEach((nid, posIdx) => {
      map[nid] = {
        x: baseX + gap * (posIdx + 1),
        y: 80 + levelHeight * idx
      };
    });
  });

  nodes.forEach((node) => {
    if (!map[node.id]) {
      // place unseen nodes in a grid at bottom
      const count = Object.keys(map).length;
      const col = count % 4;
      const row = Math.floor(count / 4);
      map[node.id] = {
        x: baseX + 200 * (col + 1),
        y: 80 + levelHeight * (order.length + row)
      };
    }
  });

  return map;
}

function drawNode(ctx, node, position) {
  const style = NODE_STYLE[node.type] || { fill: '#e2e8f0', stroke: '#475569' };
  const radius = 38;
  ctx.fillStyle = style.fill;
  ctx.strokeStyle = style.stroke;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(position.x, position.y, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = '#0f172a';
  ctx.font = '14px Segoe UI, Arial';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(node.id, position.x, position.y);
}

function drawEdge(ctx, edge, positions) {
  const src = positions[edge.from];
  const dst = positions[edge.to];
  if (!src || !dst) return;
  const color = EDGE_COLORS[edge.kind] || '#64748b';
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(src.x, src.y);
  ctx.lineTo(dst.x, dst.y);
  ctx.stroke();

  ctx.fillStyle = color;
  ctx.font = '12px Segoe UI, Arial';
  ctx.textAlign = 'center';
  ctx.fillText(edge.kind, (src.x + dst.x) / 2, (src.y + dst.y) / 2 - 6);
}

function summariseNode(node) {
  const metaEntries = Object.entries(node).filter(
    ([key]) => !['id', 'type'].includes(key)
  );
  if (!metaEntries.length) {
    return `${node.id} [${node.type}]`;
  }
  const tail = metaEntries
    .map(([key, value]) => `${key}: ${JSON.stringify(value)}`)
    .join(', ');
  return `${node.id} [${node.type}] – ${tail}`;
}

async function loadMacrograph(path) {
  const res = await fetch(path);
  if (!res.ok) {
    throw new Error(`Failed to load macrograph spec: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

async function renderMacrograph(path) {
  const canvas = document.getElementById('macrograph-canvas');
  resizeCanvas(canvas);
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const spec = await loadMacrograph(path);
  const nodes = spec.nodes || [];
  const edges = spec.edges || [];
  const positions = layoutPositions(nodes);

  edges.forEach((edge) => drawEdge(ctx, edge, positions));
  nodes.forEach((node) => drawNode(ctx, node, positions[node.id]));

  const info = document.getElementById('macrograph-info');
  const version = spec.version || 'unknown';
  const summaryLines = [`Version: ${version}`, 'Nodes:'];
  nodes.forEach((node) => summaryLines.push(` • ${summariseNode(node)}`));
  summaryLines.push('\nEdges:');
  edges.forEach((edge) => summaryLines.push(` • ${edge.from} --${edge.kind}/${edge.weight}--> ${edge.to}`));
  info.textContent = summaryLines.join('\n');
}

function wireControls() {
  const defaultBtn = document.getElementById('load-default-btn');
  const fileInput = document.getElementById('macrograph-file');

  defaultBtn.addEventListener('click', () => {
    renderMacrograph('../../specs/macrograph_v0.json').catch((err) => {
      console.error(err);
      alert(err.message);
    });
  });

  fileInput.addEventListener('change', (event) => {
    const [file] = event.target.files;
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const spec = JSON.parse(e.target.result);
        const blob = new Blob([JSON.stringify(spec)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        renderMacrograph(url).finally(() => URL.revokeObjectURL(url));
      } catch (error) {
        alert('Invalid JSON file.');
      }
    };
    reader.readAsText(file);
  });

  // Auto-load default on first render.
  renderMacrograph('../../specs/macrograph_v0.json').catch((err) => {
    console.error(err);
  });
}

document.addEventListener('DOMContentLoaded', wireControls);
