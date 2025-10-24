import * as THREE from "https://unpkg.com/three@0.160.0/build/three.module.js";
import { OrbitControls } from "https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js";
import { computeLayout, buildTopologyFromFrames } from "./layout.js";

const STATE_COLORS = {
  INACTIVE: 0x3a3f4b,
  REQUESTED: 0xf7b733,
  WAITING: 0x2c98f0,
  TRUE: 0x6ede8a,
  CONFIRMED: 0x1f9a5d,
  FAILED: 0xff6b6b,
};

const DEFAULT_STATE_COLOR = 0x6b7280;
const PLAYBACK_INTERVAL_MS = 1400;
const PULSE_DURATION_MS = 900;

const tempVector = new THREE.Vector3();
const tempQuaternion = new THREE.Quaternion();
const tempMatrix = new THREE.Matrix4();
const tempScale = new THREE.Vector3();
const tempPosition = new THREE.Vector3();
tempQuaternion.identity();

const state = {
  nodes: [],
  frames: [],
  topology: null,
  nodeIndex: new Map(),
  baseScales: [],
  pulses: new Map(),
  playing: false,
  lastAdvance: 0,
  tickIndex: 0,
  advanced: false,
  newRequests: new Set(),
  datasetLabel: "",
  labelSpritesBuilt: false,
};

let viewerEl;
let renderer;
let scene;
let camera;
let controls;
let nodeMesh;
let labelGroup;

let playButton;
let timelineSlider;
let tickLabel;
let noteField;
let thoughtsField;
let styleField;
let riskField;
let phaseList;
let statusMessage;
let countsList;
let toggleLabelsCheckbox;
let advancedToggleButton;

function init() {
  cacheDom();
  setupScene();
  bindUI();
  animate(0);
  loadDefaultDataset();
}

function cacheDom() {
  viewerEl = document.getElementById("viewer");
  playButton = document.getElementById("playPause");
  timelineSlider = document.getElementById("timelineSlider");
  tickLabel = document.getElementById("tickLabel");
  noteField = document.getElementById("frameNote");
  thoughtsField = document.getElementById("frameThoughts");
  styleField = document.getElementById("envStyle");
  riskField = document.getElementById("envRisk");
  phaseList = document.getElementById("phaseWeights");
  statusMessage = document.getElementById("statusMessage");
  countsList = document.getElementById("stateCounts");
  toggleLabelsCheckbox = document.getElementById("toggleLabels");
  advancedToggleButton = document.getElementById("toggleAdvanced");
}

function setupScene() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x080b14);

  const width = viewerEl.clientWidth;
  const height = viewerEl.clientHeight;

  camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 2000);
  camera.position.set(0, 140, 320);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(width, height);
  viewerEl.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.maxDistance = 900;
  controls.minDistance = 10;
  controls.enablePan = true;
  controls.target.set(0, 40, 0);
  controls.update();

  const ambientLight = new THREE.AmbientLight(0x92a0b3, 0.8);
  scene.add(ambientLight);

  const keyLight = new THREE.DirectionalLight(0xffffff, 0.9);
  keyLight.position.set(120, 160, 60);
  scene.add(keyLight);

  const rimLight = new THREE.PointLight(0x324867, 0.6, 700);
  rimLight.position.set(-180, -40, -120);
  scene.add(rimLight);

  labelGroup = new THREE.Group();
  labelGroup.visible = false;
  scene.add(labelGroup);

  window.addEventListener("resize", onWindowResize);
}

function disposeNodeMesh() {
  if (nodeMesh) {
    scene.remove(nodeMesh);
    nodeMesh.geometry.dispose();
    nodeMesh.material.dispose();
    nodeMesh = null;
  }
  clearLabels();
  state.labelSpritesBuilt = false;
}

function bindUI() {
  playButton.addEventListener("click", () => {
    state.playing = !state.playing;
    playButton.textContent = state.playing ? "Pause" : "Play";
    state.lastAdvance = performance.now();
  });

  document.getElementById("resetView").addEventListener("click", () => {
    camera.position.set(0, 140, 320);
    controls.target.set(0, 40, 0);
    controls.update();
  });

  timelineSlider.addEventListener("input", (event) => {
    const value = Number(event.target.value);
    setTick(value);
  });

  document.getElementById("fileInput").addEventListener("change", async (event) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    try {
      const text = await file.text();
      const parsed = JSON.parse(text);
      ingestDataset(parsed, file.name);
    } catch (error) {
      console.error("Failed to load dataset:", error);
      statusMessage.textContent = "Could not read dataset file.";
    }
  });

  toggleLabelsCheckbox.addEventListener("change", () => {
    if (toggleLabelsCheckbox.checked) {
      ensureLabelSprites();
      labelGroup.visible = true;
    } else {
      labelGroup.visible = false;
    }
  });

  advancedToggleButton.addEventListener("click", () => {
    state.advanced = !state.advanced;
    document
      .getElementById("frameMeta")
      .classList.toggle("advanced", state.advanced);
    advancedToggleButton.textContent = state.advanced ? "Simple View" : "Advanced View";
  });
}

function setTick(index) {
  if (!state.frames.length) {
    return;
  }
  const clamped = Math.max(0, Math.min(index, state.frames.length - 1));
  state.tickIndex = clamped;
  timelineSlider.value = clamped;

  const frame = state.frames[clamped];
  tickLabel.textContent = `Tick ${frame.tick ?? clamped} / ${state.frames.length - 1}`;
  applyFrame(frame);
}

function applyFrame(frame) {
  if (!frame) {
    return;
  }
  const frameStates = frame.states || {};
  state.newRequests = new Set(frame.new_requests || []);
  const now = performance.now();
  state.newRequests.forEach((nodeId) => {
    if (state.nodeIndex.has(nodeId)) {
      state.pulses.set(nodeId, now);
    }
  });

  const color = new THREE.Color();
  Object.entries(frameStates).forEach(([nodeId, nodeState]) => {
    const index = state.nodeIndex.get(nodeId);
    if (index === undefined || !nodeMesh) {
      return;
    }
    const colorHex = STATE_COLORS[nodeState] ?? DEFAULT_STATE_COLOR;
    color.setHex(colorHex);
    nodeMesh.setColorAt(index, color);
  });

  nodeMesh.instanceColor.needsUpdate = true;
  updateFrameDetails(frame);
}

function updateFrameDetails(frame) {
  const note = frame.note || "—";
  const thoughts = frame.thoughts || "—";
  const env = frame.env || {};
  const style = env.style || "—";
  const risk = typeof env.risk === "number" ? env.risk.toFixed(3) : "—";

  noteField.textContent = note;
  thoughtsField.textContent = thoughts;
  styleField.textContent = style;
  riskField.textContent = risk;

  while (phaseList.firstChild) {
    phaseList.removeChild(phaseList.firstChild);
  }
  const weights = env.phase_weights || env.phaseWeights || {};
  const entries = Object.entries(weights);
  if (!entries.length) {
    const item = document.createElement("li");
    item.textContent = "—";
    phaseList.appendChild(item);
  } else {
    entries
      .sort((a, b) => b[1] - a[1])
      .forEach(([key, value]) => {
        const item = document.createElement("li");
        item.textContent = `${key}: ${value.toFixed(3)}`;
        phaseList.appendChild(item);
      });
  }

  while (countsList.firstChild) {
    countsList.removeChild(countsList.firstChild);
  }
  const counts = {};
  Object.values(frame.states || {}).forEach((nodeState) => {
    counts[nodeState] = (counts[nodeState] || 0) + 1;
  });
  Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .forEach(([key, value]) => {
      const item = document.createElement("li");
      item.textContent = `${key}: ${value}`;
      countsList.appendChild(item);
    });
  if (!countsList.childElementCount) {
    const item = document.createElement("li");
    item.textContent = "No state data";
    countsList.appendChild(item);
  }
}

function animate(time) {
  requestAnimationFrame(animate);

  if (state.playing && state.frames.length) {
    const delta = time - state.lastAdvance;
    if (delta >= PLAYBACK_INTERVAL_MS) {
      const nextTick = (state.tickIndex + 1) % state.frames.length;
      setTick(nextTick);
      state.lastAdvance = time;
    }
  }

  updatePulseScales(time);
  controls.update();
  renderer.render(scene, camera);
}

function updatePulseScales(time) {
  if (!nodeMesh) {
    return;
  }
  let anyChanged = false;
  state.pulses.forEach((startTime, nodeId) => {
    const elapsed = time - startTime;
    const index = state.nodeIndex.get(nodeId);
    if (index === undefined) {
      state.pulses.delete(nodeId);
      return;
    }
    const baseScale = state.baseScales[index] || 1;
    if (elapsed >= PULSE_DURATION_MS) {
      const position = state.nodes[index].__position || [0, 0, 0];
      writeInstanceMatrix(index, position, baseScale);
      state.pulses.delete(nodeId);
      anyChanged = true;
      return;
    }
    const t = elapsed / PULSE_DURATION_MS;
    const factor = 1 + 0.45 * Math.sin(Math.PI * (1 - t));
    const position = state.nodes[index].__position || [0, 0, 0];
    writeInstanceMatrix(index, position, baseScale * factor);
    anyChanged = true;
  });

  if (anyChanged) {
    nodeMesh.instanceMatrix.needsUpdate = true;
  }
}

function onWindowResize() {
  const width = viewerEl.clientWidth;
  const height = viewerEl.clientHeight || 1;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
}

function loadDefaultDataset() {
  fetch("../data/demo_dataset_small.json")
    .then((response) => {
      if (!response.ok) {
        throw new Error(response.statusText);
      }
      return response.json();
    })
    .then((data) => {
      ingestDataset(data, "demo_dataset_small.json");
      statusMessage.textContent = "Loaded demo_dataset_small.json";
    })
    .catch(() => {
      statusMessage.textContent = "Select a dataset (topology+frames or frames JSON).";
    });
}

function ingestDataset(raw, label = "dataset.json") {
  const normalized = normalizeDataset(raw);
  if (!normalized.frames.length) {
    statusMessage.textContent = "Dataset missing frames.";
    return;
  }
  disposeNodeMesh();

  state.frames = normalized.frames;
  state.topology = normalized.topology;
  state.nodes = [...(state.topology.nodes || [])];
  const layout = computeLayout(state.nodes);
  state.nodeIndex = new Map();
  state.baseScales = [];
  state.pulses.clear();
  state.datasetLabel = label;

  if (!state.nodes.length) {
    statusMessage.textContent = "Topology has no nodes.";
    return;
  }

  const geometry = new THREE.SphereGeometry(1, 18, 18);
  const material = new THREE.MeshStandardMaterial({
    vertexColors: true,
    metalness: 0.15,
    roughness: 0.45,
  });
  nodeMesh = new THREE.InstancedMesh(geometry, material, state.nodes.length);
  nodeMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  nodeMesh.instanceColor = new THREE.InstancedBufferAttribute(
    new Float32Array(state.nodes.length * 3),
    3,
    false
  );
  scene.add(nodeMesh);

  const color = new THREE.Color(STATE_COLORS.INACTIVE);
  state.nodes.forEach((node, index) => {
    const position = layout.get(node.id) || [0, 0, 0];
    node.__position = position;
    const baseScale = baseScaleForNode(node);
    state.baseScales[index] = baseScale;
    state.nodeIndex.set(node.id, index);
    writeInstanceMatrix(index, position, baseScale);
    color.setHex(STATE_COLORS.INACTIVE);
    nodeMesh.setColorAt(index, color);
  });
  nodeMesh.instanceMatrix.needsUpdate = true;
  nodeMesh.instanceColor.needsUpdate = true;

  const center = new THREE.Vector3();
  state.nodes.forEach((node) => {
    const position = node.__position || [0, 0, 0];
    center.add(tempPosition.set(position[0], position[1], position[2]));
  });
  center.multiplyScalar(1 / state.nodes.length);

  let maxDistance = 0;
  state.nodes.forEach((node) => {
    const position = node.__position || [0, 0, 0];
    const distance = tempPosition
      .set(position[0], position[1], position[2])
      .distanceTo(center);
    if (distance > maxDistance) {
      maxDistance = distance;
    }
  });

  const orbitTarget = center.clone();
  controls.target.copy(orbitTarget);
  controls.update();

  const radius = Math.max(maxDistance * 1.8, 80);
  camera.position.set(
    orbitTarget.x + radius * 0.5,
    orbitTarget.y + radius * 0.75 + 40,
    orbitTarget.z + radius
  );
  camera.updateProjectionMatrix();

  timelineSlider.max = Math.max(0, state.frames.length - 1);
  timelineSlider.value = 0;
  setTick(0);

  statusMessage.textContent = `Loaded ${label}`;
  if (toggleLabelsCheckbox.checked) {
    ensureLabelSprites();
    labelGroup.visible = true;
  }
}

function normalizeDataset(raw) {
  if (!raw) {
    return { frames: [], topology: { nodes: [], edges: [] } };
  }
  const frames = Array.isArray(raw)
    ? raw
    : Array.isArray(raw.frames)
    ? raw.frames
    : Array.isArray(raw.data?.frames)
    ? raw.data.frames
    : [];

  const topology =
    raw.topology && Array.isArray(raw.topology.nodes)
      ? raw.topology
      : raw.data && Array.isArray(raw.data?.topology?.nodes)
      ? raw.data.topology
      : buildTopologyFromFrames(frames);

  return { frames, topology };
}

function baseScaleForNode(node) {
  const id = String(node.id || "").toLowerCase();
  if (id.includes("root")) return 2.1;
  if (id.includes("phase")) return 1.9;
  if (id.includes("expert")) return 1.7;
  if (id.includes("arbiter")) return 1.8;
  if (id.includes("watcher")) return 1.5;
  if (id.includes("safety")) return 1.45;
  if (id.includes("endgame")) return 1.35;
  if (id.includes("tactic")) return 1.25;
  return (node.type || "").toUpperCase() === "TERMINAL" ? 1.3 : 1.6;
}

function writeInstanceMatrix(index, position, scale) {
  tempVector.set(position[0], position[1], position[2]);
  tempScale.set(scale, scale, scale);
  tempMatrix.compose(tempVector, tempQuaternion, tempScale);
  nodeMesh.setMatrixAt(index, tempMatrix);
}

function ensureLabelSprites() {
  if (state.labelSpritesBuilt) {
    return;
  }
  clearLabels();
  state.nodes.forEach((node, index) => {
    const label = node.label || node.id;
    const sprite = makeLabelSprite(label);
    const position = node.__position || [0, 0, 0];
    const offsetY = state.baseScales[index] * 1.6 + 2;
    sprite.position.set(position[0], position[1] + offsetY, position[2]);
    labelGroup.add(sprite);
  });
  state.labelSpritesBuilt = true;
}

function makeLabelSprite(text) {
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");
  const padding = 8;
  context.font = "28px 'Segoe UI', system-ui, sans-serif";
  const metrics = context.measureText(text);
  const width = metrics.width + padding * 2;
  const height = 40;
  canvas.width = width;
  canvas.height = height;

  context.clearRect(0, 0, width, height);
  context.fillStyle = "rgba(12,17,26,0.84)";
  context.fillRect(0, 0, width, height);
  context.fillStyle = "#f4f6fb";
  context.textBaseline = "middle";
  context.font = "24px 'Segoe UI', system-ui, sans-serif";
  context.fillText(text, padding, height / 2);

  const texture = new THREE.CanvasTexture(canvas);
  texture.minFilter = THREE.LinearFilter;
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;

  const material = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    depthWrite: false,
  });
  const sprite = new THREE.Sprite(material);
  const scale = 4;
  sprite.scale.set(width / scale, height / scale, 1);
  return sprite;
}

function clearLabels() {
  while (labelGroup.children.length) {
    const child = labelGroup.children.pop();
    labelGroup.remove(child);
    if (child.material?.map) {
      child.material.map.dispose();
    }
    child.material?.dispose?.();
    child.geometry?.dispose?.();
  }
  labelGroup.visible = false;
}

init();
