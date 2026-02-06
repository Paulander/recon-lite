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
        
        // State colors - matching old demo (network-visualization.js)
        this.colors = {
            INACTIVE: '#cfd8dc',
            REQUESTED: '#64b5f6',
            ACTIVE: '#90caf9',
            SUPPRESSED: '#b0bec5',
            WAITING: '#ffd54f',
            TRUE: '#81c784',
            CONFIRMED: '#34d399',
            FAILED: '#e57373',
            edge: '#94a3b8',
            edgeActive: '#1e88e5',
            text: '#2f3a44',
            background: '#0d1117',
        };
        
        // Border colors for emphasis - matching old demo
        this.borderColors = {
            INACTIVE: '#90a4ae',
            REQUESTED: '#1e88e5',
            ACTIVE: '#1976d2',
            SUPPRESSED: '#607d8b',
            WAITING: '#f9a825',
            TRUE: '#2e7d32',
            CONFIRMED: '#047857',
            FAILED: '#b71c1c',
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
        
        // Subgraph colors
        this.subgraphColors = {
            main: '#ffffff',
            krk: '#f85149',
            kpk: '#d29922',
            tactics: '#a371f7',
            tactics_fork: '#a371f7',
            tactics_pin: '#8b5cf6',
            tactics_skewer: '#7c3aed',
            tactics_hangingPiece: '#6d28d9',
            tactics_backRankMate: '#ff6b6b',
            tactics_discoveredAttack: '#f59e0b',
            tactics_doubleCheck: '#ef4444',
            tactics_smotheredMate: '#ec4899',
        };
        
        this.collapsedSubgraphs = {};
        this.nodeRadius = 12;
        
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
        // Extract nodes and edges from frame (with subgraph support)
        this.nodes = [];
        this.edges = [];
        this.subgraphs = frame.subgraphs || {};
        this.collapsedSubgraphs = this.collapsedSubgraphs || {};
        
        if (frame.nodes) {
            for (const [nodeId, nodeData] of Object.entries(frame.nodes)) {
                const subgraph = nodeData.subgraph || 'main';
                
                // Skip internal nodes of collapsed subgraphs
                if (this.collapsedSubgraphs[subgraph] && !nodeId.endsWith('_root')) {
                    continue;
                }
                
                this.nodes.push({
                    id: nodeId,
                    state: nodeData.state || 'INACTIVE',
                    layer: nodeData.layer || 'unknown',
                    subgraph: subgraph,
                    activation: nodeData.activation || 0,
                    p_value: nodeData.p_value || 0,
                    detected: nodeData.detected || false,
                });
            }
        }
        
        if (frame.edges) {
            this.edges = frame.edges.map(e => ({
                src: e.src,
                dst: e.dst,
                type: e.type,
                weight: e.weight || 1.0,
                trace: e.trace || 0,
            })).filter(e => {
                // Filter edges for collapsed subgraphs
                const srcNode = this.nodes.find(n => n.id === e.src);
                const dstNode = this.nodes.find(n => n.id === e.dst);
                return srcNode && dstNode;
            });
        }
    }
    
    toggleSubgraphCollapse(subgraph) {
        this.collapsedSubgraphs[subgraph] = !this.collapsedSubgraphs[subgraph];
    }
    
    setAllSubgraphsCollapsed(collapsed) {
        for (const sg of Object.keys(this.subgraphs)) {
            this.collapsedSubgraphs[sg] = collapsed;
        }
    }
    
    getSubgraphColor(subgraph) {
        // Direct match
        if (this.subgraphColors[subgraph]) {
            return this.subgraphColors[subgraph];
        }
        
        // Check if it's a tactic subgraph
        if (subgraph.startsWith('tactics_')) {
            return '#a371f7'; // Default purple for tactics
        }
        
        return '#ffffff'; // Default white
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
        
        // Build adjacency for parent → child traversal (based on actual edges)
        const children = {};  // parent -> [children]
        for (const edge of this.edges) {
            if (!children[edge.src]) children[edge.src] = [];
            children[edge.src].push(edge.dst);
        }
        
        // Find root nodes (nodes with no incoming edges)
        const hasParent = new Set(this.edges.map(e => e.dst));
        const roots = this.nodes.filter(n => !hasParent.has(n.id)).map(n => n.id);
        
        // BFS to assign depth levels - ensures parent is always above child
        const depth = {};
        const queue = [...roots];
        roots.forEach(r => depth[r] = 0);
        
        while (queue.length > 0) {
            const nodeId = queue.shift();
            const nodeChildren = children[nodeId] || [];
            for (const child of nodeChildren) {
                // Only set depth if not already set (first path wins)
                // OR if this path gives a deeper depth (ensures parent->child)
                const newDepth = depth[nodeId] + 1;
                if (depth[child] === undefined || depth[child] < newDepth) {
                    depth[child] = newDepth;
                    queue.push(child);
                }
            }
        }
        
        // Handle orphan nodes (no edges) - put at depth 0
        for (const node of this.nodes) {
            if (depth[node.id] === undefined) {
                depth[node.id] = 0;
            }
        }
        
        // Group nodes by depth
        const levels = {};
        for (const node of this.nodes) {
            const d = depth[node.id];
            if (!levels[d]) levels[d] = [];
            levels[d].push(node);
        }
        
        const maxDepth = Math.max(0, ...Object.keys(levels).map(Number));
        const levelHeight = (height - 2 * padding) / Math.max(1, maxDepth);
        
        // Dynamic node radius based on total nodes
        const totalNodes = this.nodes.length;
        const nodeRadius = totalNodes < 30 ? 20 : (totalNodes < 100 ? 12 : (totalNodes < 200 ? 8 : 5));
        this.nodeRadius = nodeRadius;
        
        // Position nodes by depth level
        for (const [levelStr, levelNodes] of Object.entries(levels)) {
            const level = parseInt(levelStr);
            const y = padding + level * levelHeight;
            
            // Group by subgraph within level for better organization
            const bySubgraph = {};
            levelNodes.forEach(node => {
                const sg = node.subgraph || 'main';
                if (!bySubgraph[sg]) bySubgraph[sg] = [];
                bySubgraph[sg].push(node);
            });
            
            // Sort subgraphs: 'main' first, then alphabetically
            const subgraphNames = Object.keys(bySubgraph).sort((a, b) => {
                if (a === 'main') return -1;
                if (b === 'main') return 1;
                return a.localeCompare(b);
            });
            
            let xOffset = padding;
            const segmentWidth = (width - 2 * padding) / Math.max(1, subgraphNames.length);
            
            subgraphNames.forEach((sgName) => {
                const sgNodes = bySubgraph[sgName];
                const xSpacing = segmentWidth / Math.max(1, sgNodes.length);
                
                sgNodes.forEach((node, nodeIdx) => {
                    const x = xOffset + xSpacing * (nodeIdx + 0.5);
                    
                    this.nodePositions[node.id] = {
                        x,
                        y,
                        depth: level,
                        subgraph: node.subgraph,
                        state: node.state,
                        activation: node.activation,
                    };
                });
                
                xOffset += segmentWidth;
            });
        }
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
        
        // Draw nodes with dynamic sizing and activation glow
        const baseRadius = this.nodeRadius || 12;
        
        for (const node of this.nodes) {
            const pos = this.nodePositions[node.id];
            if (!pos) continue;
            
            // Skip inactive if hideInactive
            if (this.hideInactive && pos.state === 'INACTIVE') {
                continue;
            }
            
            const activation = node.activation || pos.activation || 0;
            const detected = node.detected || false;
            
            // Get color based on state
            let color = this.colors[pos.state] || this.colors.INACTIVE;
            
            // Override for partial activation (yellow glow)
            if (pos.state === 'INACTIVE' && activation > 0) {
                const alpha = Math.min(1, activation);
                color = `rgba(255, 235, 59, ${alpha})`;
            }
            
            // Size based on state and activation
            let radius = pos.state === 'INACTIVE' ? baseRadius * 0.8 : baseRadius;
            if (activation > 0.5) radius *= 1.2;
            
            // Activation glow
            if (activation > 0.1) {
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, radius + 5, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(255, 235, 59, ${activation * 0.3})`;
                ctx.fill();
            }
            
            // Detected indicator for tactics
            if (detected) {
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, radius + 3, 0, Math.PI * 2);
                ctx.strokeStyle = '#3fb950';
                ctx.lineWidth = 2;
                ctx.stroke();
            }
            
            // Node circle
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();
            
            // Border with state-based color (like old demo)
            const state = pos.state || 'INACTIVE';
            const borderColor = this.borderColors[state] || this.borderColors.INACTIVE;
            const isActive = ['REQUESTED', 'WAITING', 'ACTIVE', 'TRUE', 'CONFIRMED'].includes(state);
            ctx.strokeStyle = borderColor;
            ctx.lineWidth = isActive ? 3 : 2;
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


