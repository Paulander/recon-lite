// Lightweight layout helper used when node positions are missing.
// Generates layered rings that roughly match the topology generator.

const CATEGORY_LAYERS = {
  root: 0,
  phase: 1,
  expert: 2,
  arbiter: 2,
  watcher: 3,
  safety: 3,
  endgame: 4,
  tactic: 5,
  other: 6,
};

const BASE_RADIUS = 16;
const RADIUS_STEP = 18;
const HEIGHT_STEP = 10;

function guessCategory(node) {
  const id = String(node.id || "").toLowerCase();
  if (id.includes("root")) return "root";
  if (id.includes("phase")) return "phase";
  if (id.includes("expert")) return "expert";
  if (id.includes("arbiter")) return "arbiter";
  if (id.includes("watcher")) return "watcher";
  if (id.includes("safety")) return "safety";
  if (id.includes("endgame")) return "endgame";
  if (id.includes("tactic")) return "tactic";
  if ((node.type || "").toUpperCase() === "SCRIPT") return "expert";
  if ((node.type || "").toUpperCase() === "TERMINAL") return "other";
  return "other";
}

function ringPositions(nodes, layerIndex) {
  const radius =
    BASE_RADIUS +
    layerIndex * RADIUS_STEP +
    Math.log(nodes.length + 1) * 4;
  const y = layerIndex * HEIGHT_STEP;
  const positions = [];
  const rotation = Math.random() * Math.PI * 2;
  nodes.forEach((node, idx) => {
    if (nodes.length === 1) {
      positions.push([0, y, 0]);
      return;
    }
    const angle = (idx / nodes.length) * Math.PI * 2 + rotation;
    const x = radius * Math.cos(angle);
    const z = radius * Math.sin(angle);
    positions.push([
      Number(x.toFixed(3)),
      Number(y.toFixed(3)),
      Number(z.toFixed(3)),
    ]);
  });
  return positions;
}

/**
 * Ensure every node has a position. Returns a Map keyed by node id.
 */
export function computeLayout(topologyNodes = []) {
  const positions = new Map();
  const missingByLayer = new Map();

  topologyNodes.forEach((node) => {
    const pos = Array.isArray(node.pos) ? node.pos : null;
    if (pos && pos.length === 3 && pos.every((value) => Number.isFinite(value))) {
      positions.set(node.id, pos);
      return;
    }
    const category = guessCategory(node);
    const layerIndex =
      typeof node._layer === "number"
        ? node._layer
        : CATEGORY_LAYERS[category] ?? CATEGORY_LAYERS.other;
    const bucket = missingByLayer.get(layerIndex) || [];
    bucket.push(node);
    missingByLayer.set(layerIndex, bucket);
  });

  Array.from(missingByLayer.entries())
    .sort((a, b) => a[0] - b[0])
    .forEach(([layerIndex, nodes]) => {
      const coords = ringPositions(nodes, layerIndex);
      nodes.forEach((node, idx) => {
        positions.set(node.id, coords[idx]);
      });
    });

  return positions;
}

/**
 * Create a minimal topology object from the first frame when only frames are provided.
 */
export function buildTopologyFromFrames(frames = []) {
  if (!frames.length) {
    return { nodes: [], edges: [] };
  }
  const sampleStates = frames[0].states || {};
  const nodes = Object.keys(sampleStates).map((nodeId) => ({
    id: nodeId,
    label: nodeId,
    type: nodeId.includes("terminal") ? "TERMINAL" : "SCRIPT",
  }));
  return { nodes, edges: [] };
}
