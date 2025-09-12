import { stateColors, nodeIds, nodePositions, edges } from './network-data.js';
import { createLabelTexture } from './utils.js';

export function createSceneElements(scene, nodeMeshes, edgeMeshes, nodeLabels, edgeLabelMeshes) {
    const sphereGeometry = new THREE.SphereGeometry(0.15, 32, 32);
    
    // Create nodes
    nodeIds.forEach(id => {
        const material = new THREE.MeshPhongMaterial({
            color: stateColors['INACTIVE'],
            shininess: 100,
            emissive: 0x111111,
            transparent: true,
            opacity: 0.8
        });
        const sphere = new THREE.Mesh(sphereGeometry, material);
        sphere.position.set(...nodePositions[id]);
        sphere.castShadow = true;
        sphere.receiveShadow = true;
        scene.add(sphere);
        nodeMeshes[id] = sphere;
    });

    // Create edges
    edges.forEach(edge => {
        const points = [
            new THREE.Vector3(...nodePositions[edge.from]),
            new THREE.Vector3(...nodePositions[edge.to])
        ];
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const material = new THREE.LineBasicMaterial({
            color: edge.type === 'sub' ? 0x888888 : 0xDAA520,
            linewidth: 5,
            transparent: true,
            opacity: 0.8
        });
        const line = new THREE.Line(geometry, material);
        scene.add(line);
        edgeMeshes.push({ mesh: line, type: edge.type, label: edge.label });
    });

    // Create node labels
    nodeIds.forEach(id => {
        const spriteMaterial = new THREE.SpriteMaterial({
            map: createLabelTexture(id),
            transparent: true,
            opacity: 0.9
        });
        const sprite = new THREE.Sprite(spriteMaterial);
        sprite.position.set(...nodePositions[id]);
        sprite.position.y += 0.3;
        sprite.scale.set(1.5, 0.4, 1);
        scene.add(sprite);
        nodeLabels[id] = sprite;
    });

    // Create edge labels
    edges.forEach((edge, index) => {
        const midPoint = new THREE.Vector3().lerpVectors(
            new THREE.Vector3(...nodePositions[edge.from]),
            new THREE.Vector3(...nodePositions[edge.to]),
            0.5
        );
        const spriteMaterial = new THREE.SpriteMaterial({
            map: createLabelTexture(edge.label, edge.type === 'sub' ? 0x888888 : 0xDAA520),
            transparent: true,
            opacity: 0.8
        });
        const sprite = new THREE.Sprite(spriteMaterial);
        sprite.position.copy(midPoint);
        sprite.position.y += 0.1;
        sprite.scale.set(1, 0.3, 1);
        scene.add(sprite);
        edgeLabelMeshes.push(sprite);
    });
}

export function emitParticlesFromNode(scene, nodeId, color, activeParticles) {
    if (activeParticles[nodeId]) {
        activeParticles[nodeId].forEach(p => scene.remove(p));
        delete activeParticles[nodeId];
    }

    const particleCount = 200;
    const particleGroup = new THREE.Group();
    const positions = new Float32Array(particleCount * 3);
    const velocities = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);
    const sizes = new Float32Array(particleCount);

    for (let i = 0; i < particleCount; i++) {
        const nodePos = new THREE.Vector3(...nodePositions[nodeId]);
        positions[i * 3] = nodePos.x + (Math.random() - 0.5) * 0.1;
        positions[i * 3 + 1] = nodePos.y + (Math.random() - 0.5) * 0.1;
        positions[i * 3 + 2] = nodePos.z + (Math.random() - 0.5) * 0.1;

        const vel = new THREE.Vector3(
            (Math.random() - 0.5) * 0.05,
            (Math.random() - 0.5) * 0.05,
            (Math.random() - 0.5) * 0.05
        ).normalize().multiplyScalar(0.01 + Math.random() * 0.02);

        velocities[i * 3] = vel.x;
        velocities[i * 3 + 1] = vel.y;
        velocities[i * 3 + 2] = vel.z;

        colors[i * 3] = (color >> 16 & 255) / 255;
        colors[i * 3 + 1] = (color >> 8 & 255) / 255;
        colors[i * 3 + 2] = (color & 255) / 255;

        sizes[i] = Math.random() * 1 + 0.5;
    }

    const particleGeometry = new THREE.BufferGeometry();
    particleGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    particleGeometry.setAttribute('velocity', new THREE.BufferAttribute(velocities, 3));
    particleGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    particleGeometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

    const particleMaterial = new THREE.ShaderMaterial({
        uniforms: { time: { value: 0 } },
        vertexShader: `
            attribute float size;
            attribute vec3 color;
            varying vec3 vColor;
            uniform float time;
            void main() {
                vColor = color;
                vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                gl_PointSize = size * (300.0 / -mvPosition.z) * (1.0 - time * 0.5);
                gl_Position = projectionMatrix * mvPosition;
            }
        `,
        fragmentShader: `
            varying vec3 vColor;
            void main() {
                if (length(gl_PointCoord - vec2(0.5)) > 0.5) discard;
                gl_FragColor = vec4(vColor, 0.3);
            }
        `,
        blending: THREE.AdditiveBlending,
        depthTest: false,
        transparent: true
    });

    const particleSystem = new THREE.Points(particleGeometry, particleMaterial);
    particleGroup.add(particleSystem);

    scene.add(particleGroup);
    activeParticles[nodeId] = particleGroup.children;

    const startTime = Date.now();
    const animateBurst = () => {
        const elapsed = (Date.now() - startTime) / 1000;
        if (elapsed > 2) {
            scene.remove(particleGroup);
            delete activeParticles[nodeId];
            return;
        }
        particleSystem.material.uniforms.time.value = elapsed;
        requestAnimationFrame(animateBurst);
    };
    animateBurst();
}

export function updateVisualization(scene, frames, currentTick, viewingMode, nodeMeshes, edgeMeshes, prevFrame, activeParticles) {
    const frame = frames[currentTick];
    const nodes = frame.nodes;
    const newRequests = frame.new_requests || [];

    // Particles only in advanced mode
    if (viewingMode === 'advanced' && prevFrame) {
        const prevNodes = prevFrame.nodes;
        Object.keys(nodes).forEach(id => {
            const prevState = prevNodes[id] || 'INACTIVE';
            const currState = nodes[id] || 'INACTIVE';
            if (newRequests.includes(id) || currState === 'TRUE' || currState === 'CONFIRMED') {
                const color = stateColors[currState];
                emitParticlesFromNode(scene, id, color, activeParticles);
            }
        });
    }

    // Update nodes
    nodeIds.forEach(id => {
        const state = nodes[id] || 'INACTIVE';
        const color = stateColors[state];
        const mesh = nodeMeshes[id];
        mesh.material.color.setHex(color);
        mesh.material.emissive.setHex(viewingMode === 'advanced' ? color * 0.1 : 0x000000);
        mesh.material.opacity = viewingMode === 'simple' ? 1.0 : 0.8;
        mesh.scale.set(1, 1, 1);
    });

    // Pulse only in advanced
    if (viewingMode === 'advanced') {
        newRequests.forEach(id => {
            if (nodeMeshes[id]) {
                nodeMeshes[id].scale.set(1.8, 1.8, 1.8);
            }
        });
    }

    // Update edges opacity
    edgeMeshes.forEach(({ mesh }) => {
        mesh.material.opacity = viewingMode === 'simple' ? 1.0 : 0.8;
    });

    // Update panel
    document.getElementById('note').textContent = frame.note;
    document.getElementById('thoughts').textContent = frame.thoughts;
    document.querySelector('h3').textContent = `Tick ${frame.tick}`;

    return frame;
}