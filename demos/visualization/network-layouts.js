/**
 * Network Layout Algorithms for ReCoN Visualization
 * 
 * Three layout modes:
 * 1. Hierarchical - Top-down tree structure
 * 2. Force-Directed - Physics-based organic layout
 * 3. Clustered - Grouped panels by node type
 */

class NetworkLayoutEngine {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.nodes = [];
        this.edges = [];
        this.nodePositions = {};
        this.showLabels = true;
        this.showWeights = false;
        this.hideInactive = false;
        this.currentLayout = 'hierarchical';
        
        // Force simulation state
        this.forceSimulation = null;
        this.isDragging = false;
        this.dragNode = null;
        
        // Colors
        this.colors = {
            INACTIVE: '#6e7681',
            ACTIVE: '#58a6ff',
            REQUESTED: '#d29922',
            TRUE: '#3fb950',
            CONFIRMED: '#3fb950',
            FAILED: '#f85149',
            SUPPRESSED: '#a371f7',
            edge: '#30363d',
            edgeActive: '#58a6ff',
            text: '#c9d1d9',
            background: '#0d1117',
        };
        
        // Layer colors for clustered view
        this.layerColors = {
            root: '#f85149',
            ultimate: '#d29922',
            strategic: '#58a6ff',
            tactical: '#a371f7',
            sensors: '#3fb950',
            unknown: '#6e7681',
        };
        
        this.setupCanvas();
        this.setupInteraction();
    }
    
    setupCanvas() {
        // Handle resize
        const resize = () => {
            const rect = this.canvas.parentElement.getBoundingClientRect();
            this.canvas.width = rect.width;
            this.canvas.height = rect.height;
            this.render();
        };
        
        window.addEventListener('resize', resize);
        setTimeout(resize, 100);
    }
    
    setupInteraction() {
        // Mouse interactions for force layout
        this.canvas.addEventListener('mousedown', (e) => {
            if (this.currentLayout !== 'force') return;
            
            const rect = this.canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            // Find clicked node
            for (const [nodeId, pos] of Object.entries(this.nodePositions)) {
                const dx = x - pos.x;
                const dy = y - pos.y;
                if (dx * dx + dy * dy < 400) { // radius 20
                    this.isDragging = true;
                    this.dragNode = nodeId;
                    break;
                }
            }
        });
        
        this.canvas.addEventListener('mousemove', (e) => {
            if (!this.isDragging || !this.dragNode) return;
            
            const rect = this.canvas.getBoundingClientRect();
            this.nodePositions[this.dragNode] = {
                ...this.nodePositions[this.dragNode],
                x: e.clientX - rect.left,
                y: e.clientY - rect.top,
            };
            this.render();
        });
        
        this.canvas.addEventListener('mouseup', () => {
            this.isDragging = false;
            this.dragNode = null;
        });
    }
    
    setData(frame) {
        // Extract nodes and edges from frame
        this.nodes = [];
        this.edges = [];
        
        if (frame.nodes) {
            for (const [nodeId, nodeData] of Object.entries(frame.nodes)) {
                this.nodes.push({
                    id: nodeId,
                    state: nodeData.state || 'INACTIVE',
                    layer: nodeData.layer || 'unknown',
                });
            }
        }
        
        if (frame.edges) {
            this.edges = frame.edges.map(e => ({
                src: e.src,
                dst: e.dst,
                type: e.type,
                weight: e.weight || 1.0,
            }));
        }
    }
    
    computeLayout(layoutType) {
        this.currentLayout = layoutType;
        
        switch (layoutType) {
            case 'hierarchical':
                this.computeHierarchicalLayout();
                break;
            case 'force':
                this.computeForceLayout();
                break;
            case 'clustered':
                this.computeClusteredLayout();
                break;
            default:
                this.computeHierarchicalLayout();
        }
    }
    
    computeHierarchicalLayout() {
        const width = this.canvas.width;
        const height = this.canvas.height;
        const padding = 60;
        
        // Group nodes by layer
        const layers = {
            root: [],
            ultimate: [],
            strategic: [],
            tactical: [],
            sensors: [],
            unknown: [],
        };
        
        for (const node of this.nodes) {
            const layer = node.layer || 'unknown';
            if (layers[layer]) {
                layers[layer].push(node);
            } else {
                layers.unknown.push(node);
            }
        }
        
        // Compute Y positions for each layer
        const layerOrder = ['root', 'ultimate', 'strategic', 'tactical', 'sensors', 'unknown'];
        const activeLayers = layerOrder.filter(l => layers[l].length > 0);
        const layerHeight = (height - 2 * padding) / Math.max(1, activeLayers.length - 1 || 1);
        
        // Position nodes
        activeLayers.forEach((layerName, layerIdx) => {
            const layerNodes = layers[layerName];
            const y = padding + layerIdx * layerHeight;
            
            layerNodes.forEach((node, nodeIdx) => {
                const xSpacing = (width - 2 * padding) / Math.max(1, layerNodes.length);
                const x = padding + xSpacing * (nodeIdx + 0.5);
                
                this.nodePositions[node.id] = {
                    x,
                    y,
                    layer: layerName,
                    state: node.state,
                };
            });
        });
    }
    
    computeForceLayout() {
        const width = this.canvas.width;
        const height = this.canvas.height;
        
        // Initialize positions randomly if not set
        for (const node of this.nodes) {
            if (!this.nodePositions[node.id]) {
                this.nodePositions[node.id] = {
                    x: Math.random() * (width - 100) + 50,
                    y: Math.random() * (height - 100) + 50,
                    vx: 0,
                    vy: 0,
                    layer: node.layer,
                    state: node.state,
                };
            }
        }
        
        // Run force simulation
        const iterations = 50;
        const repulsion = 5000;
        const attraction = 0.05;
        const centerForce = 0.01;
        
        for (let i = 0; i < iterations; i++) {
            // Repulsion between all nodes
            for (let j = 0; j < this.nodes.length; j++) {
                for (let k = j + 1; k < this.nodes.length; k++) {
                    const n1 = this.nodePositions[this.nodes[j].id];
                    const n2 = this.nodePositions[this.nodes[k].id];
                    
                    const dx = n2.x - n1.x;
                    const dy = n2.y - n1.y;
                    const dist = Math.sqrt(dx * dx + dy * dy) + 1;
                    
                    const force = repulsion / (dist * dist);
                    const fx = (dx / dist) * force;
                    const fy = (dy / dist) * force;
                    
                    n1.vx = (n1.vx || 0) - fx;
                    n1.vy = (n1.vy || 0) - fy;
                    n2.vx = (n2.vx || 0) + fx;
                    n2.vy = (n2.vy || 0) + fy;
                }
            }
            
            // Attraction along edges
            for (const edge of this.edges) {
                const src = this.nodePositions[edge.src];
                const dst = this.nodePositions[edge.dst];
                if (!src || !dst) continue;
                
                const dx = dst.x - src.x;
                const dy = dst.y - src.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                
                const force = dist * attraction;
                const fx = (dx / dist) * force;
                const fy = (dy / dist) * force;
                
                src.vx = (src.vx || 0) + fx;
                src.vy = (src.vy || 0) + fy;
                dst.vx = (dst.vx || 0) - fx;
                dst.vy = (dst.vy || 0) - fy;
            }
            
            // Center force
            for (const node of this.nodes) {
                const pos = this.nodePositions[node.id];
                pos.vx = (pos.vx || 0) + (width / 2 - pos.x) * centerForce;
                pos.vy = (pos.vy || 0) + (height / 2 - pos.y) * centerForce;
            }
            
            // Apply velocities with damping
            const damping = 0.8;
            for (const node of this.nodes) {
                const pos = this.nodePositions[node.id];
                pos.x += (pos.vx || 0) * damping;
                pos.y += (pos.vy || 0) * damping;
                pos.vx = (pos.vx || 0) * damping;
                pos.vy = (pos.vy || 0) * damping;
                
                // Constrain to canvas
                pos.x = Math.max(30, Math.min(width - 30, pos.x));
                pos.y = Math.max(30, Math.min(height - 30, pos.y));
            }
        }
    }
    
    computeClusteredLayout() {
        const width = this.canvas.width;
        const height = this.canvas.height;
        const padding = 40;
        
        // Define cluster regions
        const clusters = {
            sensors: { x: padding, y: padding, w: (width - 3 * padding) / 2, h: (height - 3 * padding) / 2 },
            strategic: { x: width / 2 + padding / 2, y: padding, w: (width - 3 * padding) / 2, h: (height - 3 * padding) / 2 },
            root: { x: padding, y: height / 2 + padding / 2, w: (width - 3 * padding) / 2, h: (height - 3 * padding) / 2 },
            other: { x: width / 2 + padding / 2, y: height / 2 + padding / 2, w: (width - 3 * padding) / 2, h: (height - 3 * padding) / 2 },
        };
        
        // Group nodes by cluster
        const nodesByCluster = {
            sensors: [],
            strategic: [],
            root: [],
            other: [],
        };
        
        for (const node of this.nodes) {
            if (node.layer === 'sensors' || node.id.includes('Sensor')) {
                nodesByCluster.sensors.push(node);
            } else if (node.layer === 'strategic' || node.layer === 'ultimate') {
                nodesByCluster.strategic.push(node);
            } else if (node.layer === 'root') {
                nodesByCluster.root.push(node);
            } else {
                nodesByCluster.other.push(node);
            }
        }
        
        // Position nodes within clusters
        for (const [clusterName, clusterNodes] of Object.entries(nodesByCluster)) {
            const region = clusters[clusterName];
            const cols = Math.ceil(Math.sqrt(clusterNodes.length));
            const rows = Math.ceil(clusterNodes.length / cols);
            
            const cellW = region.w / (cols + 1);
            const cellH = region.h / (rows + 1);
            
            clusterNodes.forEach((node, idx) => {
                const col = idx % cols;
                const row = Math.floor(idx / cols);
                
                this.nodePositions[node.id] = {
                    x: region.x + cellW * (col + 1),
                    y: region.y + cellH * (row + 1),
                    layer: node.layer,
                    state: node.state,
                    cluster: clusterName,
                };
            });
        }
    }
    
    render() {
        const ctx = this.ctx;
        const width = this.canvas.width;
        const height = this.canvas.height;
        
        // Clear
        ctx.fillStyle = this.colors.background;
        ctx.fillRect(0, 0, width, height);
        
        if (this.nodes.length === 0) {
            ctx.fillStyle = this.colors.text;
            ctx.font = '14px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Load a JSON file to visualize the network', width / 2, height / 2);
            return;
        }
        
        // Draw cluster backgrounds for clustered layout
        if (this.currentLayout === 'clustered') {
            this.drawClusterBackgrounds();
        }
        
        // Draw edges
        for (const edge of this.edges) {
            const src = this.nodePositions[edge.src];
            const dst = this.nodePositions[edge.dst];
            if (!src || !dst) continue;
            
            // Skip if either node is inactive and hideInactive is true
            if (this.hideInactive && (src.state === 'INACTIVE' || dst.state === 'INACTIVE')) {
                continue;
            }
            
            ctx.beginPath();
            ctx.moveTo(src.x, src.y);
            ctx.lineTo(dst.x, dst.y);
            ctx.strokeStyle = src.state !== 'INACTIVE' && dst.state !== 'INACTIVE' 
                ? this.colors.edgeActive 
                : this.colors.edge;
            ctx.lineWidth = this.showWeights ? Math.max(1, edge.weight * 2) : 1;
            ctx.globalAlpha = 0.5;
            ctx.stroke();
            ctx.globalAlpha = 1;
            
            // Draw weight label
            if (this.showWeights && edge.weight !== 1.0) {
                const midX = (src.x + dst.x) / 2;
                const midY = (src.y + dst.y) / 2;
                ctx.fillStyle = this.colors.text;
                ctx.font = '10px sans-serif';
                ctx.fillText(edge.weight.toFixed(2), midX, midY);
            }
        }
        
        // Draw nodes
        for (const node of this.nodes) {
            const pos = this.nodePositions[node.id];
            if (!pos) continue;
            
            // Skip inactive if hideInactive
            if (this.hideInactive && pos.state === 'INACTIVE') {
                continue;
            }
            
            const color = this.colors[pos.state] || this.colors.INACTIVE;
            const radius = pos.state === 'INACTIVE' ? 12 : 16;
            
            // Node circle
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();
            
            // Border
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2;
            ctx.stroke();
            
            // Label
            if (this.showLabels) {
                ctx.fillStyle = this.colors.text;
                ctx.font = '11px sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText(this.truncateLabel(node.id), pos.x, pos.y + radius + 14);
            }
        }
    }
    
    drawClusterBackgrounds() {
        const ctx = this.ctx;
        const width = this.canvas.width;
        const height = this.canvas.height;
        const padding = 40;
        
        const clusters = [
            { name: 'Sensors', x: padding, y: padding, w: (width - 3 * padding) / 2, h: (height - 3 * padding) / 2, color: this.layerColors.sensors },
            { name: 'Strategic', x: width / 2 + padding / 2, y: padding, w: (width - 3 * padding) / 2, h: (height - 3 * padding) / 2, color: this.layerColors.strategic },
            { name: 'Root', x: padding, y: height / 2 + padding / 2, w: (width - 3 * padding) / 2, h: (height - 3 * padding) / 2, color: this.layerColors.root },
            { name: 'Other', x: width / 2 + padding / 2, y: height / 2 + padding / 2, w: (width - 3 * padding) / 2, h: (height - 3 * padding) / 2, color: this.layerColors.unknown },
        ];
        
        for (const cluster of clusters) {
            // Background
            ctx.fillStyle = cluster.color + '20';
            ctx.fillRect(cluster.x, cluster.y, cluster.w, cluster.h);
            
            // Border
            ctx.strokeStyle = cluster.color + '60';
            ctx.lineWidth = 2;
            ctx.strokeRect(cluster.x, cluster.y, cluster.w, cluster.h);
            
            // Label
            ctx.fillStyle = cluster.color;
            ctx.font = 'bold 12px sans-serif';
            ctx.textAlign = 'left';
            ctx.fillText(cluster.name, cluster.x + 8, cluster.y + 18);
        }
    }
    
    truncateLabel(label, maxLen = 15) {
        if (label.length <= maxLen) return label;
        return label.substring(0, maxLen - 2) + '...';
    }
    
    setShowLabels(show) {
        this.showLabels = show;
        this.render();
    }
    
    setShowWeights(show) {
        this.showWeights = show;
        this.render();
    }
    
    setHideInactive(hide) {
        this.hideInactive = hide;
        this.render();
    }
}

// Global instance
let networkLayout = null;

// Initialize and expose render function
function initNetworkLayout() {
    const canvas = document.getElementById('network-canvas');
    if (!canvas) return;
    
    networkLayout = new NetworkLayoutEngine(canvas);
    
    // Connect toggles
    document.getElementById('toggle-labels')?.addEventListener('change', (e) => {
        networkLayout.setShowLabels(e.target.checked);
    });
    
    document.getElementById('toggle-weights')?.addEventListener('change', (e) => {
        networkLayout.setShowWeights(e.target.checked);
    });
    
    document.getElementById('toggle-inactive')?.addEventListener('change', (e) => {
        networkLayout.setHideInactive(e.target.checked);
    });
}

// Render function called from main visualization
function renderNetwork(frame, layoutType) {
    if (!networkLayout) {
        initNetworkLayout();
    }
    
    if (networkLayout) {
        networkLayout.setData(frame);
        networkLayout.computeLayout(layoutType);
        networkLayout.render();
    }
}

// Initialize on load
document.addEventListener('DOMContentLoaded', initNetworkLayout);

