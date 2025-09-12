import { frames, nodeIds, nodePositions } from './network-data.js';
import { createSceneElements, emitParticlesFromNode, updateVisualization } from './visualization.js';
import { createLabelTexture } from './utils.js';

// Three.js setup
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);

let camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.body.appendChild(renderer.domElement);

// Lighting
const ambientLight = new THREE.AmbientLight(0x404040, 0.3);
scene.add(ambientLight);
const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
directionalLight.position.set(5, 5, 5);
directionalLight.castShadow = true;
scene.add(directionalLight);
const pointLight = new THREE.PointLight(0x00ff00, 0.5, 100);
pointLight.position.set(0, 0, 5);
scene.add(pointLight);

// Scene elements
const nodeMeshes = {};
const edgeMeshes = [];
const nodeLabels = {};
const edgeLabelMeshes = [];
const particles = new THREE.Group();
scene.add(particles);
let activeParticles = {};

createSceneElements(scene, nodeMeshes, edgeMeshes, nodeLabels, edgeLabelMeshes);

// Label toggles
document.getElementById('nodeLabels').addEventListener('change', (e) => {
    const visible = e.target.checked;
    Object.values(nodeLabels).forEach(label => label.visible = visible);
});
document.getElementById('edgeLabels').addEventListener('change', (e) => {
    const visible = e.target.checked;
    edgeLabelMeshes.forEach(label => label.visible = visible);
});

// Playback state
let currentTick = 0;
let isPlaying = false;
let playInterval;
let prevFrame = null;
let viewingMode = 'advanced';

// Camera setup
camera.position.set(0, 2, 8);
camera.lookAt(0, 2, 0);

function animate() {
    requestAnimationFrame(animate);
    if (viewingMode === 'advanced') {
        scene.rotation.y += 0.002;
    }
    renderer.render(scene, camera);
}
animate();

// Controls
const slider = document.getElementById('slider');
slider.max = frames.length - 1;
slider.oninput = (e) => {
    currentTick = parseInt(e.target.value);
    prevFrame = updateVisualization(scene, frames, currentTick, viewingMode, nodeMeshes, edgeMeshes, prevFrame, activeParticles);
};

document.getElementById('play').onclick = () => {
    const btn = document.getElementById('play');
    isPlaying = !isPlaying;
    btn.textContent = isPlaying ? 'Pause' : 'Play';
    btn.style.background = isPlaying ? '#FF8C00' : '#4169E1';

    if (isPlaying) {
        playInterval = setInterval(() => {
            if (currentTick < frames.length - 1) {
                currentTick++;
                slider.value = currentTick;
                prevFrame = updateVisualization(scene, frames, currentTick, viewingMode, nodeMeshes, edgeMeshes, prevFrame, activeParticles);
            } else {
                clearInterval(playInterval);
                isPlaying = false;
                btn.textContent = 'Play';
                btn.style.background = '#4169E1';
            }
        }, 600);
    } else {
        clearInterval(playInterval);
    }
};

// View toggle
document.getElementById('viewToggle').onclick = () => {
    viewingMode = viewingMode === 'advanced' ? 'simple' : 'advanced';
    const btn = document.getElementById('viewToggle');
    btn.textContent = viewingMode === 'advanced' ? 'Switch to Simple View' : 'Switch to Advanced View';
    btn.style.background = viewingMode === 'advanced' ? '#FF8C00' : '#4169E1';

    if (viewingMode === 'simple') {
        Object.values(activeParticles).forEach(groupChildren => {
            groupChildren.forEach(child => scene.remove(child));
        });
        activeParticles = {};
        renderer.shadowMap.enabled = false;
        Object.values(nodeMeshes).forEach(mesh => {
            mesh.castShadow = false;
            mesh.receiveShadow = false;
        });
        const aspect = window.innerWidth / window.innerHeight;
        camera = new THREE.OrthographicCamera(-5 * aspect, 5 * aspect, 5, -5, 0.1, 1000);
        camera.position.set(0, 2, 8);
        camera.lookAt(0, 2, 0);
    } else {
        renderer.shadowMap.enabled = true;
        Object.values(nodeMeshes).forEach(mesh => {
            mesh.castShadow = true;
            mesh.receiveShadow = true;
        });
        camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.set(0, 2, 8);
        camera.lookAt(0, 2, 0);
    }

    prevFrame = updateVisualization(scene, frames, currentTick, viewingMode, nodeMeshes, edgeMeshes, prevFrame, activeParticles);
};

// File import
document.getElementById('fileInput').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (event) => {
            frames.length = 0; // Clear existing frames
            frames.push(...JSON.parse(event.target.result));
            frames.sort((a, b) => a.tick - b.tick);
            slider.max = frames.length - 1;
            currentTick = 0;
            slider.value = 0;
            prevFrame = null;
            prevFrame = updateVisualization(scene, frames, currentTick, viewingMode, nodeMeshes, edgeMeshes, prevFrame, activeParticles);
        };
        reader.readAsText(file);
    }
});

// Legend toggle
document.getElementById('toggleLegend').onclick = () => {
    const legend = document.getElementById('legend');
    legend.style.display = legend.style.display === 'none' ? 'block' : 'none';
};

// Initial render
prevFrame = updateVisualization(scene, frames, currentTick, viewingMode, nodeMeshes, edgeMeshes, prevFrame, activeParticles);

// Responsive resize
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

// Initial label visibility
Object.values(nodeLabels).forEach(label => label.visible = true);
edgeLabelMeshes.forEach(label => label.visible = false);